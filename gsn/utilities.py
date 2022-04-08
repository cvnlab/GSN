import numpy as np

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

    # function m = posrect(m,val)
    #
    # <m> is a matrix
    # <val> (optional) is the cut-off value. Default: 0.
    #
    # positively-rectify <m>.
    # basically do: m(m<val) = val.
    #
    # example:
    # isequal(posrect([2 3 -4]),[2 3 0])
    # isequal(posrect([2 3 -4],4),[4 4 4])
    
    # make sure even scalars are treated as arrays
    m = np.array(m)
    
    # do it
    m[m<val] = val;
    
    return m