import numpy as np
from gsn import rsa_noise_ceiling

def perform_gsn(data, opt=None):
    """
    Perform GSN (generative modeling of signal and noise).

    Parameters:
    data (numpy.ndarray): An array of shape (voxels, conditions, trials). This indicates the measured responses to different conditions on distinct trials. The number of trials must be at least 2.

    opt (dict, optional): A dictionary with the following optional fields:
        wantverbose (bool, optional): Whether to print status statements. Default is True.
        wantshrinkage (bool, optional): Whether to use shrinkage in the estimation of covariance. Default is False.

    Returns:
    results: A dictionary with the results containing:
        mnN - the estimated mean of the noise (1 x voxels)
        cN - the raw estimated covariance of the noise (voxels x voxels)
        cNb - the final estimated covariance after biconvex optimization
        shrinklevelN - shrinkage level chosen for cN
        shrinklevelD - shrinkage level chosen for the estimated data covariance
        mnS - the estimated mean of the signal (1 x voxels)
        cS - the raw estimated covariance of the signal (voxels x voxels)
        cSb - the final estimated covariance after biconvex optimization
        ncsnr - the 'noise ceiling SNR' estimate for each voxel (1 x voxels).
                This is, for each voxel, the std dev of the estimated signal
                distribution divided by the std dev of the estimated noise
                distribution. Note that this is computed on the raw
                estimated covariances. Also, note that we apply positive
                rectification (to prevent non-sensical negative ncsnr values).

    History:
    - 2024/01/05 - (1) major change to use the biconvex optimization procedure --
                       we now have cSb and cNb as the final estimates;
                   (2) cSb no longer has the scaling baked in and instead we
                       create a separate temporary variable cSb_rsa;
                   (3) remove the rapprox output

    Example:
    data = np.random.randn(100, 40, 4) * 2 + np.random.randn(100, 40, 4)
    results = perform_gsn(data)
    """

        # Set default options if not provided
    if opt is None:
        opt = {}
    if 'wantverbose' not in opt or opt['wantverbose'] is None:
        opt['wantverbose'] = 1
    if 'wantshrinkage' not in opt or opt['wantshrinkage'] is None:
        opt['wantshrinkage'] = 0

    # Prepare opt for rsa_noise_ceiling.py
    opt['mode'] = 1
    opt['ncsims'] = 0
    opt['wantfig'] = 0
    if opt['wantshrinkage']:
        opt['shrinklevels'] = []  # allow default shrinkage levels
    else:
        opt['shrinklevels'] = [1]  # force only full estimation

    # Call the rsa_noise_ceiling function
    # rsa_noise_ceiling returns a tuple where the third element is the results
    results = rsa_noise_ceiling(data, opt)[2]

    # Remove 'sc' and 'splitr' from results
    #results.pop('sc', None)
    # results.pop('splitr', None)

    return results
