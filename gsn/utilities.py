import numpy as np
import warnings
import pdb

def squish(m, num):
    """
    f = squish(m,num)
     <m> is a matrix
     <num> is the positive number of initial dimensions to squish together
     return <m> squished.
     example:
     a = np.asarray([[1,2],[3,4]])
     b = np.asarray([1,2,3,4])
     np.testing.assert_array_equal(squish(a,2), b.T)
    """
    
    msize = m.shape
    # calculate the new dimensions
    newdim = np.r_[np.prod(msize[:num]), msize[num:]].tolist()
    # do the reshape
    f = np.reshape(m, newdim)
    return f

def posrect(m,val=0):
    """
    function m = posrect(m,val)

    <m> is a matrix
    <val> (optional) is the cut-off value. Default: 0.

    positively-rectify <m>.
    basically do: m[m<val] = val.

    example:
    np.testing.assert_array_equal(posrect([2 3 -4]),[2 3 0])
    np.testing.assert_array_equal([2 3 -4],4),[4 4 4])
    """
    
    # make sure even scalars are treated as arrays
    m = np.array(m)
    
    # do it
    m[m<val] = val;
    
    return m

def nanreplace(m, val=0, mode=0):
    """
    Replace NaN or non-finite values in a matrix with a specified value.

    Parameters:
    m (numpy.ndarray): A matrix.
    val (optional, scalar): The value to replace NaNs with. Default is 0.
    mode (optional, int):
        0 means replace all NaNs in m with val.
        1 means if the first element of m is not finite (NaN, -Inf, Inf), fill entire matrix with val.
        2 means if not all elements of m are finite and real, fill entire matrix with val.
        3 means replace any non-finite value in m with val.
        Default is 0.

    Returns:
    numpy.ndarray: The matrix after replacing NaN or non-finite values.

    Examples:
    assert np.array_equal(nanreplace(np.array([1, np.nan]), 0), np.array([1, 0]))
    assert np.array_equal(nanreplace(np.array([np.nan, 2, 3]), 0, 1), np.array([0, 0, 0]))
    """

    if mode == 0:
        m[np.isnan(m)] = val
    elif mode == 1:
        if not np.isfinite(m[0]):
            m[:] = val
    elif mode == 2:
        if not np.all(np.isfinite(m) & np.isreal(m)):
            m[:] = val
    elif mode == 3:
        m[~np.isfinite(m)] = val

    return m

def zerodiv(x, y, val=0, wantcaution=1):
    """zerodiv(data1,data2,val,wantcaution)
    Args:
        <x>,<y> are matrices of the same size or either
                        or both can be scalars.
        <val> (optional) is the value to use when <y> is 0.
                        default: 0.
        <wantcaution> (optional) is whether to perform special
                        handling of weird cases (see below).
                        default: 1.
        calculate x/y but use <val> when y is 0.
        if <wantcaution>, then if the absolute value of one or
                        more elements of y is less than 1e-5
                        (but not exactly 0), we issue a warning
                        and then treat these elements as if they
                        are exactly 0.
        if not <wantcaution>, then we do nothing special.

    note some weird cases:
    if either x or y is [], we return [].

    """

    # Check if either x or y is empty, return empty if so
    if x.size == 0 or y.size == 0:
        return np.array([])

    # handle special case of y being scalar
    if np.isscalar(y):
        if y == 0:
            return np.full(x.shape, val)
        else:
            if wantcaution and abs(y) < 1e-5:   # see allzero.m
                warnings.warn('abs value of divisor is less than 1e-5. we are treating the divisor as 0.')
                return np.full(x.shape, val)
            else:
                return x / y[:, np.newaxis]
    else:
        bad = y == 0
        bad2 = abs(y) < 1e-5  # see allzero.m
        if wantcaution and np.any(np.logical_and(bad2, ~bad)):
            warnings.warn('abs value of one or more divisors is less than 1e-5. we are treating these divisors as 0.')
        if wantcaution:
            tmp = y
            tmp[bad2] = 1
            if x.ndim == 1:
                f = x / tmp
            else:
                f = x / tmp[:, np.newaxis]
            f[bad2] = val
        else:
            tmp = y
            tmp[bad] = 1
            if x.ndim == 1:
                f = x / tmp
            else:
                f = x / tmp[:, np.newaxis]
            f[bad] = val
        return f