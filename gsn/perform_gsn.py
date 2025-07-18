import numpy as np
from gsn.rsa_noise_ceiling import rsa_noise_ceiling

def perform_gsn(data, opt=None):
    """
    Perform GSN (generative modeling of signal and noise).

    Parameters:
    data (numpy.ndarray): An array of shape (voxels, conditions, trials). This indicates the measured responses to different conditions on distinct trials. The number of trials must be at least 2. 
        Not all conditions need to have the full set of trials (see more information below).

    opt (dict, optional): A dictionary with the following optional fields:
        wantverbose (bool, optional): Whether to print status statements. Default is True.
        wantshrinkage (bool, optional): Whether to use shrinkage in the estimation of covariance. Default is True.

    Regarding uneven number of trials across conditions:
    - It is acceptable that different conditions may have different numbers
      of trials. To indicate the lack of data for certain trials, you can
      include NaNs --- specifically, it is okay if data[:, i, j]
      consists of NaNs for some combination(s) of i and j.
    - However, it must be the case that each condition has at least one 
      trial with valid data (i.e. data[:, i, :] must contain at least one
      valid trial).
    - Also, there must be a sufficient number of conditions with at least
      two trials of valid data (for estimation and cross-validation purposes).
      If this is ever not the case, we will issue an error and crash.
    - If the user provides data with uneven number of trials across
      conditions, our estimation strategy changes:
      - We estimate noise covariance for each condition (ignoring missing data)
        and average the estimated noise covariance across conditions. 
        Note that this approach does not perform any special compensation 
        for the differing numbers of trials available across conditions.
      - We determine the minimum number of trials and randomly select that
        many trials from each condition for the purposes of estimation of 
        data covariance. By doing so, we get a nice fully balanced data subset.
        Note that this approach introduces some stochasticity and
        ignores some portion of the data.
      - The biconvex optimization procedure proceeds as usual. (For the 
        weighting step, we calculate the equivalent "average number of trials" 
        that were actually used for noise covariance estimation.)

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
        numiters - the number of iterations used in the biconvex optimization.
                0 means the first estimate was already positive semi-definite.

    History:
    - 2025/07/15 - add support for uneven number of trials
    - 2024/08/24 - add results['numiters']
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
        opt['wantshrinkage'] = 1

    # Prepare opt for rsa_noise_ceiling.py
    opt['mode'] = 1
    opt['ncsims'] = 0
    opt['wantfig'] = 0
    if opt['wantshrinkage']:
        opt['shrinklevels'] = np.linspace(0,1,51) # allow default shrinkage levels
    else:
        opt['shrinklevels'] = [1]  # force only full estimation

    # Call the rsa_noise_ceiling function
    # rsa_noise_ceiling returns a tuple where the third element is the results
    results = rsa_noise_ceiling(data, opt)[2]

    # Remove 'sc' and 'splitr' from results
    results.pop('sc', None)
    results.pop('splitr', None)

    return results
