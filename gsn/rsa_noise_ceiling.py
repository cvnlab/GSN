import numpy as np
from gsn.utilities import squish, posrect
from gsn.calc_shrunken_covariance import calc_shrunken_covariance
from gsn.construct_nearest_psd_covariance import construct_nearest_psd_covariance
import scipy.stats as stats
from scipy.spatial.distance import pdist

def rsa_noise_ceiling(data,
                     rdmfun = lambda x: pdist(x.T,'correlation'),
                     comparefun = lambda x,y: stats.pearsonr(x,y)[0],
                     numsim = 20,
                     nctrials = None,
                     shrinklevels = np.linspace(0,1,51),
                     mode = 0):
    """
    nc, ncdist, results = rsa_noise_ceiling(data,rdmfun,comparefun,numsim,nctrials)

    <data> is voxels x conditions x trials
    <rdmfun> (optional) is a function that constructs an RDM. Specifically,
      the function should accept as input a data matrix (e.g. voxels x conditions)
      and output a RDM with some dimensionality (can be a column vector, 2D matrix, etc.).
      Default: lambda x: pdist(x.T,'correlation'). This default simply 
      calculates dissimilarity as 1-r, and then extracts the lower triangle
      (excluding the diagonal) as a column vector.
    <comparefun> (optional) is a function that quantifies the similarity of two RDMs.
      Specifically, the function should accept as input two RDMs (in the format
      that is returned by <rdmfun>) and output a scalar. 
      Default: lambda x,y: stats.pearsonr(x,y)[0], returning the Pearson
      correlation between the two RDMs.
    <numsim> (optional) is the number of Monte Carlo simulations to run.
      The final answer is computed as the median across simulations. Default: 20.
    <nctrials> (optional) is the number of trials over which to average for
      the purposes of the noise ceiling estimate. For example, setting
      <nctrials> to 10 will result in the calculation of a noise ceiling 
      estimate for the case in which responses are averaged across 10 trials
      measured for each condition. Default: data.shape[2].

    Use the GSN (generative modeling of signal and noise) method to estimate
    an RSA noise ceiling.

    Note: if <comparefun> ever returns NaN, we automatically replace these
    cases with 0. This is a convenient workaround for degenerate cases, 
    e.g., cases where the signal is generated as all zero.

    Return:
      <nc> as a scalar with the noise ceiling estimate.
      <ncdist> as 1 x <numsim> with the result of each simulation.
        Note that <nc> is simply the median of <ncdist>.
      <results> as a struct with additional details:
        mnN - the estimated mean of the noise (1 x voxels)
        cN  - the estimated covariance of the noise (voxels x voxels)
        mnS - the estimated mean of the signal (1 x voxels)
        cS  - the estimated covariance of the signal (voxels x voxels)
        cSb - the regularized estimated covariance of the signal (voxels x voxels).
              this estimate reflects both a nearest-approximation and 
              a post-hoc scaling, and is used in the Monte Carlo simulations.
        rapprox - the correlation between the nearest-approximation of the 
                  signal covariance and the original signal covariance

    Example:
    data = np.random.randn(100,40,4) + 2*np.random.randn(100,40,4);
    [nc,ncdist,results] = rsa_noise_ceiling(data)

    internal inputs:

    <shrinklevels> (optional) is like the input to calc_shrunken_covariance.py.
      Default: np.linspace(0,1,51).
    <mode> (optional) is
      0 means do the normal thing
      1 means to omit the gain adjustment
    """
    
    # calc
    nvox   = data.shape[0]
    ncond  = data.shape[1]
    ntrial = data.shape[2]
    
    # how many simulated trial averages to perform
    # by default use number contained in the data
    if nctrials is None:
        nctrials = data.shape[2]

    # estimate noise covariance
    mnN, cN, shrinklevelN, nllN = calc_shrunken_covariance(data=np.transpose(data,(2,0,1)),
                                                         shrinklevels=shrinklevels,
                                                         wantfull=1)

    # estimate data covariance
    mnD, cD, shrinklevelD, nllD = calc_shrunken_covariance(data=np.mean(data,2).T,
                                                         shrinklevels=shrinklevels,
                                                         wantfull=1)

    # estimate signal covariance
    mnS = mnD - mnN
    cS  =  cD - cN/ntrial

    # calculate nearest approximation for the noise.
    # this is expected to be PSD already. however, small numerical issues
    # may occassionally arise. so, our strategy is to go ahead and
    # run it through the approximation, and to do a quick assertion to 
    # check sanity. note that we just overwrite cN.
    cN,rapprox0 = construct_nearest_psd_covariance(cN)
    assert(rapprox0 > 0.99)

    # calculate nearest approximation for the signal.
    # this is the more critical case!
    cSb,rapprox = construct_nearest_psd_covariance(cS)

    # scale the nearest approximation to match the average variance 
    # that is observed in the original estimate of the signal covariance.
    if mode == 0:
        sc = posrect(np.mean(np.diag(cS))) / np.mean(np.diag(cSb))  # notice the posrect to ensure non-negative scaling
        cSb, _ = construct_nearest_psd_covariance(cSb * sc)   # impose scaling and run it through construct_nearest_psd_covariance.py for good measure
    elif mode == 1:
        pass

    # perform Monte Carlo simulations
    ncdist = np.zeros((numsim,))
    for rr in range(numsim):
        
        signal = np.random.multivariate_normal(np.squeeze(mnS),cSb,size=ncond) # cond x voxels
        noise = np.random.multivariate_normal(np.squeeze(mnN),cN,size=ncond*nctrials) # ncond*nctrials x voxels
        measurement = signal + np.mean(np.reshape(noise,(ncond,nctrials,nvox)),1)  # cond x voxels
        
        ncdist[rr] = comparefun(rdmfun(signal.T),rdmfun(measurement.T))

    # if comparefun ever outputs NaN, set these cases to 0.
    # for example, you might be correlating an all-zero signal
    # with some data, which may result in NaN.
    ncdist[np.isnan(ncdist)] = 0

    # compute median across simulations
    nc = np.median(ncdist)

    # prepare additional outputs
    results = {'mnN':mnN,
               'cN':cN,
               'mnS':mnS,
               'cS':cS,
               'cSb':cSb,
               'rapprox':rapprox}
                                                        
    return nc,ncdist,results
