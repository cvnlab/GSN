import numpy as np
import scipy.stats as stats

def construct_nearest_psd_covariance(c1):
    """
    c1,rapprox = construct_nearest_psd_covariance(c1)

    <c> is a square matrix (N x N)

    Use the method of Higham 1988 to obtain the nearest symmetric
    positive semidefinite matrix to <c> (in the Frobenius norm).

    Return:
    <c> as the approximating matrix
    <rapprox> as the correlation between the original matrix
    and the approximating matrix.

    example:
    c1 = np.cov(np.random.randn(100,10))
    c1[1,1] = -1
    c2, rapprox = construct_nearest_psd_covariance(c1)
    plt.figure()
    plt.subplot(121)
    plt.imshow(c1,clim=(-2,2))
    plt.subplot(122)
    plt.imshow(c2,clim=(-2,2))
    T = np.linalg.cholesky(c1) # should error
    T = np.linalg.cholesky(c2) # should succeed
    print(rapprox)
    """
    
    # To match matlab behavior for cases where input is a scalar or 1x1 numpy array
    if np.isscalar(c1) or (hasattr(c1, 'shape') and c1.shape == (1, 1)):
        # Extract scalar value for computation
        scalar_val = c1.item() if hasattr(c1, 'item') else c1
        rapprox = 1 if scalar_val >= 0 else np.nan
        
        # Compute result and preserve input format
        result_val = max(0, scalar_val)
        if np.isscalar(c1):
            return result_val, rapprox
        else:
            # Return as same shaped array
            return np.array([[result_val]]), rapprox
    
    # ensure symmetric
    c1 = (c1 + c1.T)/2
    
    # check if PSD
    try:
        T = np.linalg.cholesky(c1)
        
        # if no error, we don't have to do anything!
        rapprox = 1
        
        return c1, rapprox
    
    except:
                
        # construct nearest PSD matrix (with respect to Frobenius norm)
        try:
            # Singular Value Decomposition
            u, s, v = np.linalg.svd(c1, full_matrices=True)
            c2 = (c1 + v.T @ np.diag(s) @ v) / 2  # Average with symmetric polar factor
        except np.linalg.LinAlgError:  # If SVD fails to converge
            # Eigendecomposition
            d, v = np.linalg.eig(c1)
            d[d < 0] = 0
            c2 = v @ np.diag(d) @ v.T
            
        # ensure symmetric again
        c2 = (c2 + c2.T)/2

        # old
        #u, s, v = np.linalg.svd(c1, full_matrices=True)
        #c2 = (c1 + np.matmul(np.matmul(v.T,np.diag(s)), v)) / 2

        # check that it is indeed PSD
        
        try: 
            T = np.linalg.cholesky(c2)
            
        except:
                        
            # add a small ridge
            c2 = c2 + np.eye(c2.shape[0]) * 1e-10 
                        
            # recurse
            c2, _ = construct_nearest_psd_covariance(c2)
                
        # calculate how good the approximation is
        c1_flat = c1.reshape(-1)
        c2_flat = c2.reshape(-1)
        
        rapprox = stats.pearsonr(c1_flat, c2_flat)[0]
        
        # replace
        c1 = c2
            
        return c1, rapprox
    