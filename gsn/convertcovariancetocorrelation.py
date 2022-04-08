import numpy as np
from gsn.utilities import posrect

def convertcovariancetocorrelation(m):
    """
    m, bad = convertcovariancetocorrelation(m)
    
    <m> is a covariance matrix (N x N)
    
    We attempt to normalize the covariance matrix <m> such
    that the diagonals are equal to 1, thereby enabling
    the interpretation of off-diagonal elements as
    correlation values. This is done by dividing each
    element by its associated row-wise and column-wise
    diagonal elements.
    
    This will fail in cases where a diagonal element
    is 0 or negative. In such cases, we set all 
    associated matrix elements to NaN.
    
    Return:
    <m> as the final correlation matrix
    <bad> as a column logical vector indicating any invalid
      variances (i.e., variances that are non-positive)
    
    example:
    c = np.cov(np.random.randn(10,10))
    c[2,2] = -1  # deliberately make a variance invalid
    c2 = convertcovariancetocorrelation(c)
    plt.figure()
    plt.subplot(121)
    plt.imshow(c,clim=(-1,1))
    plt.subplot(122)
    plt.imshow(c2,clim=(-1,1))
    """
    
    # divide elements row-wise and column-wise by their
    # associated diagonal elements
    mdiag = np.diag(m);
    t0 = np.sqrt(posrect(mdiag));  # column vector. note: negative are set to 0!
    m = m / np.matmul(t0,t0.T)

    # mark cases with invalid diagonal variances (<= 0) as NaN!
    bad = mdiag <= 0
    m[bad,:] = np.nan
    m[:,bad] = np.nan

    return m, bad
