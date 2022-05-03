import numpy as np
import math
import warnings
import scipy.stats as stats
from gsn.utilities import squish
from gsn.calc_mv_gaussian_pdf import calc_mv_gaussian_pdf

def calc_shrunken_covariance(data, 
                             leaveout = 5, 
                             shrinklevels = np.linspace(0,1,51), 
                             wantfull = 0):
    """
    mn, c, shrinklevel, nll = calc_shrunken_covariance(data,leaveout,shrinklevels,wantfull)

    <data> is a matrix with dimensions observations x variables.
    There can be more variables than observations. In addition, the 
    dimensions of <data> can be observations x variables x cases (where
    the number of cases at least 2); in this special scenario,  
    we perform handling of "mean-subtraction" for each case
    (see details below).
    <leaveout> (optional) is N >= 2 which means leave out 1/N of the data for
    cross-validation purposes. The selection of data points is random.
    Default: 5.
    <shrinklevels> (optional) is a non-empty vector (1 x F) of shrinkage fractions
    (between 0 and 1 inclusive) to evaluate. For example, 0.8 means to
    shrink values to 80  of their original size. Default: np.linspace(0,1,51).
    <wantfull> (optional) is whether to use the identified optimal shrinkage
    fraction to re-estimate the mean and covariance using the full dataset
    (i.e. including the initially left-out data). Default: 0.

    Using (N-1)/N of the data (randomly selected), calculate a covariance 
    matrix and shrink the off-diagonal elements to 0 according to
    <shrinklevels>. (The diagonal elements are left untouched.) The shrinkage
    level that maximizes the likelihood (i.e., minimizes the negative log 
    likelihood) of the left-out 1/N of the data is chosen. If <wantfull>,
    we re-estimate the mean and covariance using all of the data and the
    identified shrinkage level. If not <wantfull>, we just return the 
    shrunken estimate from (N-1)/N of the data.

    Note that we try to detect pathological cases and if detected, we will
    issue warning messages.

    The case where <data> has multiple cases along the third dimension
    is useful for when you have multiple sets of measurements, each of
    which has an unknown mean. In this scenario, we calculate the covariance
    of each case separately (thereby ignoring the mean of each sample)
    and then average (pool) covariance estimates across cases.
    Note that in this special scenario, we perform cross-validation on 
    cases (not observations), the returned <mn> is necessarily all zero,
    and the left-out data are also mean-subtracted before evaluating the
    cross-validated likelihood of covariance estimates.

    Return:
    <mn> as 1 x variables with the estimated mean
    <c> as variables x variables with the estimated covariance matrix
    <shrinklevel> as the shrinkage fraction that was chosen
    <nll> as 1 x F with the mean negative log likelihood on the left-out data.
     one or more values can be NaN (e.g. singular covariance matrices).

    Example:
    import numpy as np
    import matplotlib.pyplot as plt
    from gsn.calc_shrunken_covariance import calc_shrunken_covariance
    
    numvar = 100      # number of variables
    for n in [90, 1000]: #  number of observations
        sigma = 0.5*np.ones((numvar,numvar));
        np.fill_diagonal(sigma, 1)
        x = np.random.multivariate_normal(np.ones((numvar,)),sigma,numvar)
        xcov = np.cov(x);
        mn,c,shrinklevel,nll = calc_shrunken_covariance(x)

        plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(xcov)
        plt.colorbar() 
        plt.title('original')
        plt.subplot(1,3,2)
        plt.imshow(c)
        plt.colorbar() 
        plt.title(f'shrinkage = {round(shrinklevel,2)}')
        plt.subplot(1,3,3)
        plt.plot(np.arange(0,1.02,0.02),nll,'ro-')
        plt.title('mean negative log likelihood')
        plt.show()
    """

    # handle special scenario
    if np.ndim(data) == 3:
        
        # divide into training and validation
        ii = np.random.choice(np.arange(data.shape[2]), 
                             (int(np.round((1/leaveout)*data.shape[2]),)),
                              replace = False)
        
        iinot = np.setdiff1d(np.arange(data.shape[2]), ii)
        
        # calculate covariance from the training data (variables x variables).
        # notice that we separate compute covariance from each case (thereby
        # ignoring the mean of each variable within each case), and then
        # average across the results of each case.
        
        for pp in range(len(iinot)):
            c_ = np.cov(data[:,:,iinot[pp]].T, bias=False) / len(iinot)
            if pp == 0:
                c = c_
            else:
                c += c_
                
        # we know the mean from the training data is just zero (1 x variables)
        mn = np.zeros((1, data.shape[1]), dtype = type(data[0,0,0]))
        
    # otherwise, do the regular thing
    else:
        
        # divide into training and validation
        ii = np.random.choice(np.arange(data.shape[0]), 
                             (int(np.round((1/leaveout)*data.shape[0]),)),
                              replace = False)
        
        iinot = np.setdiff1d(np.arange(data.shape[0]), ii)
        
        # calculate covariance from the training data (variables x variables)
        c = np.cov(data[iinot].T, bias = False)
                
        # calculate the mean from the training data (1 x variables)
        mn = np.mean(data[iinot],0)
            
    # try different shrinkage levels
    nll = np.zeros((len(shrinklevels),), type(data.reshape(-1)[0])) # mean negative log likelihood
    covs = np.zeros((c.shape[0], c.shape[1], len(shrinklevels)), type(data.reshape(-1)[0])) # various covariance matrices
    
    for p in range(len(shrinklevels)):
        
        # shrink the covariance (off-diagonal elements mix with 0)
        c2 = c * shrinklevels[p]
        np.fill_diagonal(c2, np.diag(c)) # preserve the diagonal elements
        
        # record
        covs[:,:,p] = c2

        # evaluate the PDF on the validation data, obtaining log likelihoods
        if np.ndim(data) == 3:

            data_zeromean = data[:,:,ii] - np.mean(data[:,:,ii],0)
            data_zeromean = squish(np.transpose(data_zeromean, (0,2,1)), 2)   
            
            [pr,err] = calc_mv_gaussian_pdf(pts = data_zeromean, 
                                             mn = mn, 
                                             c = c2,  # shrunken covariance 
                                             wantomitexp = 1) 
        else:
            
            [pr,err] = calc_mv_gaussian_pdf(pts = data[ii], 
                                             mn = mn, 
                                             c = c2,  # shrunken covariance 
                                             wantomitexp = 1) 

        if err:
            # set to infinity instead of nan, since np.argmin doesn't
            # handle nans well. 
            nll[p] = np.inf 
            continue
        
        # calculate mean negative log likelihood (lower values mean higher probabilities)
        nll[p] = np.mean(-pr)
    
    # which achieves the minimum?
    min0ix = np.argmin(nll)
    
    shrinklevel = shrinklevels[min0ix]
    
    # do some error checking
    nll = nll.astype(float)
    
    if np.all(np.isnan(nll)):
        warnings.warn('all covariance matrices were singular')
    elif len(np.unique(nll[np.isfinite(nll)])) == 1:
        warnings.warn('there was only one unique finite log-likelihood; something might be wrong?')
    elif np.logical_not(np.isfinite(nll[min0ix])):
        warnings.warn('selected likelihood is not finite; something might be wrong?')
    
    ############ PREPARE FINAL OUTPUT
    
    # in this case, we have to re-estimate using the full dataset
    if wantfull:
        
        # handle special scenario
        if np.ndim(data) == 3:
            
            # let's be efficient and re-use previous calculations
            c = c * (len(iinot) / data.shape[2])
            
            for pp in range(len(ii)):
                c += np.cov(data[:,:,ii[pp]].T, bias = False) / data.shape[2]
                
            # and the mean stays zero, no problem
        
        # handle regular case
        else:
            
            # calculate mean and covariance from all the data
            c = np.cov(data.T, bias = False)
            mn = np.mean(data, axis=0)
            
        # apply shrinkage
        c2 = c * shrinklevel
        np.fill_diagonal(c2, np.diag(c)) # preserve the diagonal elements
        c = c2
              
    # otherwise we just extract the desired shrunken estimate
    else:
        c = covs[:,:,min0ix]
        
    return mn, c, shrinklevel, nll