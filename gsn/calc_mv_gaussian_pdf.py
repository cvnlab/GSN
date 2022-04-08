import numpy as np
import math

def calc_mv_gaussian_pdf(pts, mn, c, wantomitexp = 0):
    """
    f, err = calc_mv_gaussian_pdf(pts,mn,c,wantomitexp)

    <pts> is N x D where D corresponds to different dimensions
    and N corresponds to different data points
    <mn> is 1 x D with the mean of the multivariate Gaussian
    <c> is D x D with the covariance of the multivariate Gaussian
    <wantomitexp> (optional) is whether to omit the final 
    exponentiation. This is useful when probabilities are
    very small. Default: 0.

    Evaluate the probability density function corresponding to
    a multivariate Gaussian governed by <mn> and <c> at the
    data points specified in <pts>.

    Return:
    <f> as N x 1 with the likelihood corresponding to each data point.
    if <wantomitexp>, we get the log likelihood instead.
    <err> is 0 when <c> is positive definite.
    when <c> is not positive definite, <f> is returned as [],
    and <err> is returned as 1.
    """

    # number of variables
    d = pts.shape[1]
    
    # remove distribution mean from the data points
    pts = pts - mn
        
    # decompose covaraince matrix
    try: 
        T = np.linalg.cholesky(c).T # note the necessity of the transpose operation
        
    # if decomposition fails, return error
    except:
        err = 1
        f = []
        return f, err
    
    pts = np.matmul(pts, np.linalg.inv(T))
        
    # finish up
    f = -0.5 * np.sum(pts**2,axis = 1) - np.sum(np.log(np.diag(T))) - d*np.log(2*math.pi)/2
        
    if not wantomitexp:
        f = np.exp(f)
        
    err = 0
        
    return f, err