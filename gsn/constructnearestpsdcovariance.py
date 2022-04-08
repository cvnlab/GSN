import numpy as np
import scipy.stats as stats

def constructnearestpsdcovariance(c1):
    
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
        u, s, v = np.linalg.svd(c1, full_matrices=True)
        
        c2 = (c1 + np.matmul(np.matmul(v.T,np.diag(s)), v)) / 2
        
        # check that it is indeed PSD
        
        try: 
            T = np.linalg.cholesky(c2)
            
        except:
            
            print('warning: nearest cov is not PSD! attempting to use recursion to fix this.')
            c2, _ = constructnearestpsdcovariance(c2)
                
        # calculate how good the approximation is
        rapprox = stats.pearsonr(c1.reshape(-1), c2.reshape(-1))[0]
        
        # replace
        c1 = c2
        
        return c1, rapprox
    