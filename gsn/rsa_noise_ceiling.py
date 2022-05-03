import numpy as np
from gsn.utilities import squish, posrect, nanreplace, calc_cod
from gsn.calc_shrunken_covariance import calc_shrunken_covariance
from gsn.construct_nearest_psd_covariance import construct_nearest_psd_covariance
import scipy.stats as stats
from scipy.special import comb
from scipy.spatial.distance import pdist
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

def rsa_noise_ceiling(data,
                      wantverbose = True,
                      rdmfun = lambda x: pdist(x.T, 'correlation'),
                      comparefun = lambda x, y: stats.pearsonr(x, y)[0],
                      wantfig = 1,
                      ncsims = 50,
                      ncconds = 50,
                      nctrials = None,
                      splitmode = 0,
                      scs = np.arange(0, 2.1, 0.1),
                      simchunk = 50,
                      simthresh = 10,
                      maxsimnum = 1000,
                      shrinklevels = np.linspace(0, 1, 51),
                      mode = 0):
    """
    [nc, ncdist, results] = rsa_noise_ceiling(data,rdmfun,comparefun,ncsims,nctrials)

    <data> is voxels x conditions x trials. This indicates the measured
      responses to different conditions on distinct trials. The number of 
      trials must be at least 2.
    <wantverbose> (optional) is whether to print status statements. Default: 1.
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
    <wantfig> (optional) is
      0 means do not make a figure
      1 means plot a figure in a new figure window
      A means write a figure to filename prefix A (e.g., '/path/to/figure'
        will result in /path/to/figure.png being written)
      Default: 1.
    <ncsims> (optional) is the number of Monte Carlo simulations to run for
      the purposes of estimating the RSA noise ceiling. The final answer 
      is computed as the median across simulation results. Default: 50.
    <ncconds> (optional) is the number of conditions to simulate in the Monte
      Carlo simulations. In theory, the RSA noise ceiling estimate should be 
      invariant to number of conditions simulated, but the higher the number,
      the more stable/accurate the results. Default: 50.
    <nctrials> (optional) is the number of trials over which to average for
      the purposes of the noise ceiling estimate. For example, setting
      <nctrials> to 10 will result in the calculation of a noise ceiling 
      estimate for the case in which responses are averaged across 10 trials
      measured for each condition. Default: data.shape[2].
    <splitmode> (optional) controls the way in which trials are divided
      for the data reliability calculation. Specifically, <splitmode> is:
      0 means use only the maximum split-half number of trials. In the case
        of an odd number of trials T, we use floor(T/2).
      1 means use numbers of trials increasing by a factor of 2 starting
        at 1 and including the maximum split-half number of trials.
        For example, if there are 10 trials total, we use [1 2 4 5].
      2 means use the maximum split-half number of trials as well as half
        of that number (rounding down if necessary). For example, if 
        there are 11 trials total, we use [2 5].
      The primary value of using multiple numbers of trials (e.g. options 
      1 and 2) is to provide greater insight for the figure inspection that is
      created. However, in terms of accuracy of RSA noise ceiling estimates,
      option 0 should be fine (and will result in faster execution).
      Default: 0.
    <scs> (optional) controls the way in which the posthoc scaling factor
      is determined. Specifically, <scs> is:
      A where A is a vector of non-negative values. This specifies the
        specific scale factors to evaluate. There is a trade-off between
        speed of execution and the discretization/precision of the results.
      Default: np.arange(0, 2.1, 0.1).
    <simchunk> (optional) is the chunk size for the data-splitting and
      model-based simulations. Default: 50, which indicates to perform 50 
      simulations for each case, and then increment in steps of 50 if 
      necessary to achieve <simthresh>. Must be 2 or greater.
    <simthresh> (optional) is the value for the robustness metric that must
      be exceeded in order to halt the data-splitting simulations. 
      The lower this number, the faster the execution time, but the less 
      accurate the results. Default: 10.
    <maxsimnum> (optional) is the maximum number of simulations to perform
      for the data-splitting simulations. Default: 1000.
    <shrinklevels> (optional) is a non-empty vector (1 x F) of shrinkage fractions
      (between 0 and 1 inclusive) to evaluate. For example, 0.8 means to
      shrink values to 80  of their original size. Default: np.linspace(0,1,51).
    <mode> (optional) is
      0 means use the data-reliability method
      1 means use no scaling.
      2 means scale to match the un-regularized average variance
      Default: 0.

    Use the GSN (generative modeling of signal and noise) method to estimate
    an RSA noise ceiling.

    Note: if <comparefun> ever returns NaN, we automatically replace these
    cases with 0. This is a convenient workaround for degenerate cases, 
    e.g., cases where the signal is generated as all zero.

    Return:
      <nc> as a scalar with the noise ceiling estimate.
      <ncdist> as 1 x <ncsims> with the result of each Monte Carlo simulation.
        Note that <nc> is simply the median of <ncdist>.
      <results> as a struct with additional details:
        mnN - the estimated mean of the noise (1 x voxels)
        cN  - the estimated covariance of the noise (voxels x voxels)
        mnS - the estimated mean of the signal (1 x voxels)
        cS  - the estimated covariance of the signal (voxels x voxels)
        cSb - the regularized estimated covariance of the signal (voxels x voxels).
              This estimate reflects both a nearest-approximation and 
              a post-hoc scaling that is designed to match the data reliability
              estimate. It is this regularized signal covariance that is
              used in the Monte Carlo simulations.
        rapprox - the correlation between the nearest-approximation of the 
                  signal covariance and the original signal covariance
        sc - the post-hoc scaling factor that was selected
        splitr - the data split-half reliability value that was obtained for the
                 largest trial number that was evaluated. (We return the median
                 result across the simulations that were conducted.)
        ncsnr - the 'noise ceiling SNR' estimate for each voxel (1 x voxels).
                This is, for each voxel, the std dev of the estimated signal
                distribution divided by the std dev of the estimated noise
                distribution. Note that this is computed on the originally
                estimated signal covariance and not the regularized signal
                covariance. Also, note that we apply positive rectification to 
                the signal std dev (to prevent non-sensical negative ncsnr values).

    Example:
    data = np.random.randn(100,40,4) + 2*np.random.randn(100,40,4);
    [nc,ncdist,results] = rsa_noise_ceiling(data, splitmode = 1)

    internal inputs:

    <shrinklevels> (optional) is like the input to calc_shrunken_covariance.py.
      Default: np.linspace(0,1,51).
    <mode> (optional) is
      0 means do the normal thing
      1 means to omit the gain adjustment
    """
    
    # get input dimensions
    nvox   = data.shape[0]
    ncond  = data.shape[1]
    ntrial = data.shape[2]
    
    # deal with massaging inputs and sanity checks
    assert ntrial >= 2, f"ntrial must be greater than or equal to 2. got: {ntrial}"
    scs = np.unique(scs)
    assert np.all(scs >= 0), f"all scs must be greater than or equal to 0. got: {scs}"
    assert simchunk >= 2, f"simchunk must be greater than or equal to 2. got: {simchunk}"
    
    ###### ESTIMATION #####
    
    # how many simulated trial averages to perform
    # by default use number contained in the data
    if nctrials is None:
        nctrials = data.shape[2]

    if wantverbose is True:
        print('Estimating noise covariance...')
        
    # estimate noise covariance
    mnN, cN, shrinklevelN, nllN = calc_shrunken_covariance(data = np.transpose(data, (2, 0, 1)),
                                                           shrinklevels = shrinklevels,
                                                           wantfull = 1)
    
    if wantverbose is True:
        print('done.\n')
        print('Estimating data covariance...')        

    # estimate data covariance
    mnD, cD, shrinklevelD, nllD = calc_shrunken_covariance(data = np.mean(data, 2).T,
                                                           shrinklevels = shrinklevels,
                                                           wantfull = 1)
    
    if wantverbose is True:
        print('done.\n')
        print('Estimating signal covariance...')

    # estimate signal covariance
    mnS = mnD - mnN
    cS  =  cD - cN / ntrial
    
    if wantverbose is True:
        print('done.\n')
        
    # prepare some outputs
    sd_noise = np.sqrt(np.diag(cN)) # std of the noise (1 x voxels)
    sd_signal = np.sqrt(posrect(np.diag(cS))) # std of the signal (1 x voxels)
    ncsnr = sd_signal / sd_noise # noise ceiling SNR (1 x voxels)
    
    ##### REGULARIZATION OF COVARIANCES #####
    
    if wantverbose is True:
        print('Regularizing...\n')

    # calculate nearest approximation for the noise.
    # this is expected to be PSD already. however, small numerical issues
    # may occassionally arise. so, our strategy is to go ahead and
    # run it through the approximation, and to do a quick assertion to 
    # check sanity. note that we just overwrite cN.
    cN, rapprox0 = construct_nearest_psd_covariance(cN)
    assert rapprox0 > 0.99, f"Construct nearest psd covariance failed."

    # calculate nearest approximation for the signal.
    # this is the more critical case!
    cSb, rapprox = construct_nearest_psd_covariance(cS)
    
    # deal with scaling of the signal covariance matrix
    if mode == 1:
        
        # do nothing
        sc = 1
        splitr = []
        
    elif mode == 2:
        
        # scale the nearest approximation to match the average variance 
        # that is observed in the original estimate of the signal covariance.
        
        # notice the posrect to ensure non-negative scaling
        sc = posrect(np.mean(np.diag(cS))) / np.mean(np.diag(cSb))
        
        # impose scaling and run it through 
        # construct_nearest_psd_covariance.py for good measure
        cSb, _ = construct_nearest_psd_covariance(cSb * sc)   
        
    elif mode == 0:
        
        if splitmode == 0:
            
            # use only the maximum split-half number of trials
            # in the case of an odd number of trials T, we use floor(T/2).
            splitnums = [np.floor(ntrial / 2)]
            
        elif splitmode == 1:
            
            # use numbers of trials increasing by a factor of 2 starting
            # at 1 and including the maximum split-half number of trials
            splitnums = list(np.unique(list(2 ** np.arange(0, np.floor(np.log2(ntrial / 2)) + 1)) + [np.floor(ntrial / 2)]))
            
        elif splitmode == 2:
            
            # use the maximum split-half number of trials as well as half
            # of that number (rounding down if necessary). For example, if 
            # there are 11 trials total, we use [2 5]
            splitnums = np.array([np.floor(np.floor(ntrial / 2) / 2), np.floor(ntrial / 2)])
            splitnums = list(splitnums[splitnums > 0])
            
        if wantverbose is True:
            print('\tCalculating data split-half reliability...\n')
            
        # first, we need to figure out if we can do exhaustive combinations
        doexhaustive = 1
        combolist = dict()
        validmatrix = dict()
        
        for nn in range(len(splitnums) - 1, -1, -1):
            
            # if the dimensionality seems too large, just get out
            if comb(ntrial, splitnums[nn]) > 2 * simchunk:
                doexhaustive = 0
                break
                
            # calculate the full set of possibilities
            combolist[nn] = np.array(list(itertools.combinations(np.arange(ntrial), int(splitnums[nn]))))
            ncomb = combolist[nn].shape[0]
            
            validmatrix[nn] = np.zeros((ncomb, ncomb))
            
            for r in range(ncomb):
                for c in range(ncomb):
                    if c <= r: # this admits only the upper triangle as potentially valid
                        continue 
                    if len(np.intersect1d(combolist[nn][r], combolist[nn][c])) == 0:
                        validmatrix[nn][r, c] = 1
                        
            # if the number of combinations to process is more than simchunk, just give up
            if np.sum(validmatrix[nn]) > simchunk:
                doexhaustive = 0
                break
        
        # if it looks like we can do it exhaustively, do it!
        if doexhaustive == 1:
            
            if wantverbose is True:
                print('\tDoing exhaustive set of combinations...')
            
            # we will save only one single value with the ultimate result
            datasplitr = np.zeros((len(splitnums),)) 
            
            for nn in range(len(splitnums)):
                
                ncomb = combolist[nn].shape[0]
                temp = []
                
                for r in range(ncomb):
                    for c in range(ncomb):
                        if validmatrix[nn][r,c] == 1:
                            temp.append(nanreplace(comparefun(rdmfun(np.mean(data[:, :, combolist[nn][r]], 2)),
                                                              rdmfun(np.mean(data[:, :, combolist[nn][c]], 2)))))
                            
                datasplitr[nn] = np.median(temp)
                
            splitr = datasplitr[-1] # 1 x 1 with the result for the "most trials" data split case
        
        else: # otherwise, do the random-sampling approach
            
            if wantverbose is True:
                print('\tDoing random-sampling approach...\n')
            
            iicur = 0        # current sim number
            iimax = simchunk # current targeted max

            # empty matrix for simulation results
            datasplitr = np.zeros((len(splitnums), iimax))

            while 1:

                for nn in range(len(splitnums)):
                    for si in range(iicur, iimax):
                        temp = np.random.permutation(ntrial)

                        dataA = data[:, :, temp[:int(splitnums[nn])]]

                        dataB = data[:, :, temp[int(splitnums[nn])+np.arange(splitnums[nn]).astype(int)]]

                        datasplitr[nn,si] = nanreplace(comparefun(rdmfun(np.mean(dataA, axis = 2)),
                                                                  rdmfun(np.mean(dataB, axis = 2))))

                temp = datasplitr
                robustness = np.mean(np.abs(np.median(temp, axis = 1)) / ((stats.iqr(temp, axis = 1) / 2) / np.sqrt(temp.shape[1])))
                
                if robustness > simthresh:
                    if wantverbose is True:
                        print('\t...procedure ended due to robustness threshold exceeded.')
                    break

                iicur = iimax
                iimax += simchunk

                if iimax > maxsimnum:
                    if wantverbose is True:
                        print('\t...procedure ended due to reaching maxsimnum.')
                    break
                    
                # append more zeros so we can keep going
                datasplitr = np.hstack((datasplitr, np.zeros((len(splitnums), simchunk))))

            # scalar with the median result for the "most trials" data split case
            splitr = np.median(datasplitr[-1, :]) 
            
        # done with data split reliability
        if wantverbose is True:
            print('\tdone with data split-half reliability.\n')
            
        # calculate model-based split reliability
        if wantverbose is True:
            print('\tCalculating model split-half reliability...')
            
        iicur = 0        # current sim number
        iimax = simchunk # current targeted max
        
        # empty matrix for results
        modelsplitr = np.zeros((len(scs), len(splitnums), iimax))
        
        # precompute
        tempcS = np.zeros((cSb.shape[0], cSb.shape[1], len(scs)))
            
        for sci in range(len(scs)):
            temp, _ = construct_nearest_psd_covariance(cSb * scs[sci])
            tempcS[:, :, sci] = temp
            
        while 1:
            robustness = np.zeros((len(scs),))
            for sci in tqdm(range(len(scs))):
                for nn in range(len(splitnums)):
                    for si in range(iicur, iimax):
                        signal = np.random.multivariate_normal(np.squeeze(mnS), tempcS[:, :, sci], size = ncconds) # cond x voxels
                        noise = np.random.multivariate_normal(np.squeeze(mnN), cN / splitnums[nn], size = ncconds * 2) # 2*cond x voxels
                        measurementA = signal + noise[:ncconds] # cond x voxels
                        measurementB = signal + noise[ncconds:] # cond x voxels
                        modelsplitr[sci, nn, si] = nanreplace(comparefun(rdmfun(measurementA.T), 
                                                                         rdmfun(measurementB.T)))
                                                
                temp = modelsplitr[sci]
                robustness[sci] = np.mean(np.abs(np.median(temp, axis = 1)) / ((stats.iqr(temp, axis = 1) / 2) / np.sqrt(temp.shape[1])))
            
            robustness = np.mean(robustness)
            
            if robustness > simthresh:
                if wantverbose is True:
                    print('\t...procedure ended due to robustness threshold exceeded.')
                break
                
            iicur = iimax
            iimax += simchunk
            
            if iimax > maxsimnum:
                if wantverbose is True:
                    print('\t...procedure ended due to reaching maxsimnum.')
                break
                        
            # append more zeros so we can keep going
            modelsplitr = np.concatenate((modelsplitr, np.zeros((len(scs), len(splitnums), simchunk))), axis = 2)
                        
        if wantverbose is True:
            print('\tdone with model split-half reliability.\n')
            
        # calculate R^2 between model-based results and the data results and find the max
        if wantverbose is True:
            print('\tFinding best model...')
                
        model_median = np.median(modelsplitr, axis = 2)
        data_median = np.tile(np.median(datasplitr.reshape(-1 ,1), axis = 1), 
                              (len(scs), 1))
        
        # scales x 1
        R2s = calc_cod(model_median, data_median, dim = 1, wantgain = 0, wantmeansub = 0)
        bestii = np.argmax(R2s)
        sc = scs[bestii]
        
        # impose scaling and run it through construct_nearest_psd_covariance
        # for good measure
        cSb, _ = construct_nearest_psd_covariance(cSb * sc)
            
        # if the data split r is higher than all of the model split r, 
        # we should warn the user.
        temp = np.median(modelsplitr[:, -1], axis = 1)
        
        if splitr > np.max(temp):
            print(f'\twarning: the empirical data split r seems to be out of range of the model. something may be wrong; results may be inaccurate. consider increasing the <scs> input.')
            
        # do a sanity check on the smoothness of the R2 results.
        # if they appear to be non-smooth, we should warn the user
        if len(R2s) >= 4:
            temp = calc_cod(np.convolve(R2s, np.array([1, 1, 1]) / 3, 'valid'),
                            R2s[1:-1])
            
        if temp < 90:
            print(f'\twarning: the R2 values appear to be non-smooth (smooth function explains only {temp}% variance). something may be wrong; results may be inaccurate. consider increasing simchunk, simthresh, and/or maxsimnum')

    if wantverbose is True:
        print('\tdone.\n')
        
    ##### SIMULATIONS #####
    # if wantverbose is True:
    #     print('Performing Monte Carlo simulations...')
        
    # perform Monte Carlo simulations
    ncdist = np.zeros((ncsims,))
    for rr in range(ncsims):
        
        signal = np.random.multivariate_normal(np.squeeze(mnS), cSb, size = ncond) # cond x voxels
        noise = np.random.multivariate_normal(np.squeeze(mnN), cN, size = ncond * nctrials) # ncond*nctrials x voxels
        measurement = signal + np.mean(np.reshape(noise, (ncond, nctrials, nvox)), 1)  # cond x voxels
        
        ncdist[rr] = comparefun(rdmfun(signal.T), rdmfun(measurement.T))
        
    if wantverbose is True:
        print('...done with regularization of covariances.\n')
        
        
    ##### MONTE CARLO SIMULATIONS FOR RSA NOISE CEILING #####
    
    if wantverbose is True:
        print('Performing Monte Carlo simulations...')
    
    # output variable for simulations
    ncdist = np.zeros((ncsims,))
    
    # perform monte carlo sims
    for rr in range(ncsims):
        signal = np.random.multivariate_normal(np.squeeze(mnS), cSb, size = ncconds)          # ncconds x voxels
        noise = np.random.multivariate_normal(np.squeeze(mnN), cN / nctrials, size = ncconds) # ncconds x voxels
        measurement = signal + noise                                                          # ncconds x voxels
        ncdist[rr] = nanreplace(comparefun(rdmfun(signal.T), 
                                           rdmfun(measurement.T)))
        
    if wantverbose is True:
        print('done.\n')

    ##### FINISH UP #####
    
    # compute median across simulations
    nc = np.median(ncdist)

    # prepare additional outputs
    results = {'mnN': mnN,
               'cN': cN,
               'mnS': mnS,
               'cS': cS,
               'cSb': cSb,
               'rapprox': rapprox,
               'sc': sc,
               'splitr': splitr,
               'ncsnr': ncsnr}
    
    ##### MAKE A FIGURE #####
    
    if mode is 0 and wantfig is not 0:
    
        if wantverbose is True:
            print('Creating figure...')

        ax = plt.figure(figsize=(48,40)).subplot_mosaic(
                    """
                    AAAAABBBBBCCCCC
                    DDDDDEEEEEFFFFF
                    GGGGGGG.HHHHHHH
                    GGGGGGG.HHHHHHH
                    """
                    ) 

        titleft = 36
        labelft = 24

        ax['A'].hist(np.squeeze(mnS), facecolor = '#442cb4', align = 'left')
        ax['A'].set_ylabel('Frequency', fontsize = labelft)
        ax['A'].set_title('Mean of Signal', fontsize = titleft)
        ax['A'].tick_params(axis = 'both', which = 'major', labelsize = labelft - 2)

        mx = np.max(np.abs(cS.reshape(-1)))
        if mx == 0:
            mx = 1

        im = ax['B'].imshow(cS, clim = (-mx, mx), cmap = 'viridis')
        cb = plt.colorbar(im, ax = ax['B'])
        cb.ax.tick_params(labelsize = labelft - 2)
        ax['B'].set_title('Covariance of Signal', fontsize = titleft)
        ax['B'].tick_params(axis = 'both', which = 'major', labelsize = labelft - 2)

        mx = np.max(np.abs(cSb.reshape(-1)))
        if mx == 0:
            mx = 1

        im = ax['C'].imshow(cSb, clim = (-mx, mx), cmap = 'viridis')
        cb = plt.colorbar(im, ax = ax['C'])
        cb.ax.tick_params(labelsize = labelft - 2)
        ax['C'].set_title('Regularized and scaled', fontsize = titleft)
        ax['C'].tick_params(axis = 'both', which = 'major', labelsize = labelft - 2)

        ax['D'].hist(np.squeeze(mnN), facecolor = '#442cb4', align = 'left')
        ax['D'].set_ylabel('Frequency', fontsize = labelft)
        ax['D'].set_title('Mean of Noise', fontsize = titleft)
        ax['D'].tick_params(axis='both', which = 'major', labelsize = labelft - 2)

        mx = np.max(np.abs(cN.reshape(-1)))
        if mx == 0:
            mx = 1

        im = ax['E'].imshow(cN, clim = (-mx, mx), cmap = 'viridis')
        cb = plt.colorbar(im, ax = ax['E'])
        cb.ax.tick_params(labelsize = labelft - 2)
        ax['E'].set_title('Covariance of Noise', fontsize = titleft)
        ax['E'].tick_params(axis = 'both', which = 'major', labelsize = labelft - 2)

        ax['F'].hist(np.squeeze(ncsnr), facecolor='#442cb4', align = 'left')
        ax['F'].set_ylabel('Frequency', fontsize = labelft)
        ax['F'].set_title('Noise ceiling SNR', fontsize = titleft)
        ax['F'].tick_params(axis = 'both', which = 'major', labelsize = labelft - 2)

        hs = []
        for sci in range(len(scs)):
            md0 = np.median(datasplitr.reshape(-1, 1), axis = 1)
            sd0 = (stats.iqr(datasplitr.reshape(-1, 1), axis = 1)) / 2
            
        nlines = len(scs)
        cm = plt.cm.viridis(np.linspace(0, 1, nlines))

        for sci in range(len(scs)):
            md0 = np.median(modelsplitr[sci], axis = 1) # 1 x n
            se0 = (stats.iqr(modelsplitr[sci], axis = 1) / 2) / np.sqrt(modelsplitr.shape[2]) # 1 x n
        
            if scs[sci] == sc:
                lw0 = 8
                mark0 = 'o'
            else:
                lw0 = 4
                mark0 = 'x'

            ax['G'].plot(np.squeeze(splitnums), md0, f'{mark0}-', linewidth = lw0, color = cm[sci], markersize = 16)

        md0 = np.median(datasplitr.reshape(-1, 1), axis = 1)
        sd0 = (stats.iqr(datasplitr.reshape(-1, 1), axis = 1)) / 2
        se0 = sd0 / np.sqrt(datasplitr.reshape(-1, 1).shape[1])
        ax['G'].plot(np.squeeze(splitnums), md0, 'k-', linewidth = 8, zorder = 100, marker = 'o', markersize = 20, markeredgecolor = 'k', markerfacecolor = 'none')
        ax['G'].set_xlim([np.min(splitnums) - 1, np.max(splitnums) + 1])
        ax['G'].set_xlabel('Number of trials in each split', fontsize = labelft)
        ax['G'].set_ylabel('Similarity (comparefun output)', fontsize = labelft)
        
        if doexhaustive is 1:
            ax['G'].set_title(f"Data (ALL sims); Model ({modelsplitr.shape[2]} sims); splitr={round(splitr, 3)}",fontsize = titleft)
        else:
            ax['G'].set_title(f"Data ({datasplitr.shape[1]} sims); Model ({modelsplitr.shape[2]} sims); splitr={round(splitr, 3)}",fontsize = titleft)
        ax['G'].tick_params(axis = 'both', which = 'major', labelsize = labelft - 2)

        ax['H'].plot(scs, R2s, 'r-', marker = 'o', markeredgecolor = 'r', markerfacecolor = 'none', markersize = 20)
        ax['H'].plot([sc, sc], [np.nanmin(R2s), np.nanmax(R2s)], 'k', linewidth = 8)
        ax['H'].set_xlabel('Scaling factor', fontsize = labelft)
        ax['H'].set_ylabel('R^2 between model and data (%)', fontsize = labelft)
        ax['H'].set_title(f"rapprox = {round(rapprox, 2)}, sc = {round(sc, 2)}, nc = {round(nc, 3)}, +/- {round(stats.iqr(ncdist) / 2 / np.sqrt(len(ncdist)), 3)}",fontsize = titleft)
        ax['H'].tick_params(axis = 'both', which = 'major', labelsize = labelft - 2)

        if wantfig is not 1: 
            plt.savefig(wantfig, facecolor = 'white')
            
        if wantverbose is True:
            print('done.\n')
        
    return nc, ncdist, results
