import numpy as np
import matplotlib.pyplot as plt

def compute_denoiser(data, V, opt={}):
    """
    Compute an optimal denoising matrix for data based on cross-validation performance.

    Parameters:
    ----------
    data : numpy.ndarray
        Input data with shape (nunits, nconds, ntrials), where:
        - nunits: Number of units (e.g., voxels or features).
        - nconds: Number of conditions.
        - ntrials: Number of trials.

    V : numpy.ndarray
        Basis matrix (nunits x nunits) used for projection and denoising.

    opt : dict, optional
        Dictionary of options for denoising. Possible keys include:
        - 'thresholds' : list or array of thresholds for denoising (default: range(nunits)).
        - 'scoring_fn' : function to compute denoised performance (default: negative_mse_columns).
        - 'threshold_per' : str, 'population' or 'unit', specifying thresholding method (default: 'population').
        - 'cv_mode' : int, cross-validation mode:
            0 - Denoise using single trial against the mean of other trials (default).
            1 - Denoise using the mean of trials against a single trial.
           -1 - Do not perform cross-validation, instead set the threshold to where signal-to-noise ratio (SNR) of the input basis reaches zero.

    Returns:
    -------
    denoiser : numpy.ndarray
        Optimal denoising matrix (nunits x nunits).

    denoised_cv_scores : numpy.ndarray
        Cross-validation performance scores for each threshold (len(thresholds) x ntrials x nunits).

    best_threshold : int or numpy.ndarray
        Optimal threshold. For 'population', this is a single integer. For 'unit', this is an array of thresholds per unit.

    Raises:
    ------
    ValueError
        If 'cv_mode' is -1 and no threshold achieves zero noise ceiling SNR.

    AssertionError
        If 'denoised_cv_scores' contains NaN or infinite values.

    Notes:
    ------
    - The function applies denoising by projecting the data into a lower-dimensional subspace
      defined by the basis matrix `V` and applying a threshold to the eigenvalues of the projection.
    - Cross-validation ensures that denoising does not overfit to the data.

    Example:
    -------
    >>> data = np.random.rand(100, 10, 20)
    >>> V = np.eye(100)
    >>> opt = {'thresholds': np.arange(1, 50), 'cv_mode': 0}
    >>> denoiser, scores, best_thresh = compute_denoiser(data, V, opt)

    """
    nunits, nconds, ntrials = data.shape
    
    # Set default values for dictionary fields if they are missing
    opt.setdefault('thresholds', np.arange(nunits))
    opt.setdefault('scoring_fn', negative_mse_columns)
    opt.setdefault('threshold_per', 'population')
    opt.setdefault('cv_mode', 0)
    
    # Initialize array to hold denoised tuning correlations
    denoised_cv_scores = np.zeros((len(opt['thresholds']), ntrials, nunits))
      
    # Prepare for unit-specific thresholds if needed
    all_denoisers = {} if opt['threshold_per'] == 'unit' else None

    # Iterate through thresholds and trials
    for tt, threshold in enumerate(opt['thresholds']):
        
        if opt['threshold_per'] == 'unit':
            # Store individual trial denoisers for unit-specific thresholding
            all_denoisers[threshold] = np.zeros((ntrials, nunits, nunits))
        
        for tr in range(ntrials):
            
            # Define cross-validation splits based on cv_mode
            if opt['cv_mode'] == 0:
                # Single trial as test, others as training
                dataA = data[:, :, tr].T
                dataB = np.mean(data[:, :, np.setdiff1d(np.arange(ntrials), tr)], axis=2).T
            
            elif opt['cv_mode'] == 1:
                # Average trials as test, single trial as training
                dataA = np.mean(data[:, :, np.setdiff1d(np.arange(ntrials), tr)], axis=2).T
                dataB = data[:, :, tr].T
            
            elif opt['cv_mode'] == -1:
                # Special case: No cross-validation
                data_ctv = data.copy().transpose(1, 2, 0)
                ncsnrs = np.zeros(nunits)
                for i in range(data_ctv.shape[2]):
                    this_eigv = V[:, i]
                    proj_data = np.dot(data_ctv, this_eigv)
                    _, ncsnr, _, _ = compute_noise_ceiling(proj_data[np.newaxis, ...])
                    ncsnrs[i] = ncsnr

                # Find the first index where SNR is zero
                if np.sum(ncsnrs == 0) == 0:
                    raise ValueError('Basis SNR never hits 0. Adjust cross-validation settings.')
                else:
                    best_threshold = np.argwhere(ncsnrs == 0)[0, 0]
                    denoising_fn = np.concatenate([np.ones(best_threshold), np.zeros(nunits - best_threshold)])
                    denoiser = V @ np.diag(denoising_fn) @ V.T
                    return denoiser, denoised_cv_scores, best_threshold
            
            # Define denoising function for current threshold
            denoising_fn = np.concatenate([np.ones(threshold), np.zeros(nunits - threshold)])
            denoiser = V @ np.diag(denoising_fn) @ V.T
            dataA_denoised = dataA @ denoiser

            if opt['threshold_per'] == 'unit':
                all_denoisers[threshold][tr] = denoiser

            # Calculate cross-validation score
            denoised_cv_scores[tt, tr] = opt['scoring_fn'](dataB, dataA_denoised)

    # Check for invalid values in scores
    assert np.all(np.isfinite(denoised_cv_scores)), "denoised_cv_scores contains NaN or inf values."

    # Average scores across trials
    mean_denoised_cv_scores = np.mean(denoised_cv_scores, axis=1)
    
    # Select the best threshold based on population or unit-specific scoring
    if opt['threshold_per'] == 'population':
        best_threshold = opt['thresholds'][np.argmax(np.mean(mean_denoised_cv_scores, axis=1))]
        denoising_fn = np.concatenate([np.ones(best_threshold), np.zeros(nunits - best_threshold)])
        denoiser = V @ np.diag(denoising_fn) @ V.T

    elif opt['threshold_per'] == 'unit':
        best_threshold = opt['thresholds'][np.argmax(mean_denoised_cv_scores, axis=0)]
        denoiser = np.zeros((nunits, nunits))

        # Construct voxel-specific denoiser by averaging trial-wise denoisers
        for u in range(nunits):
            this_denoiser = np.mean(all_denoisers[best_threshold[u]], axis=0)
            denoiser[:, u] = this_denoiser[:, u]

    return denoiser, denoised_cv_scores, best_threshold


def apply_denoiser(data, denoiser, rescue=False):
    """
    Apply a denoising matrix to trial-wise data, optionally rescuing any degraded voxels.

    Parameters:
    ----------
    data : numpy.ndarray
        Input data with shape (nunits, nconds, ntrials), where:
        - nunits: Number of units (e.g., voxels or features).
        - nconds: Number of conditions.
        - ntrials: Number of trials.

    denoiser : numpy.ndarray
        Denoising matrix (nunits x nunits) used to project data into a denoised subspace.

    rescue : bool, optional
        Whether to "rescue" voxels whose noise ceiling signal-to-noise ratio (ncsnr)
        decreases after denoising. Rescued voxels retain their original values and the 
        denoising matrix is adjusted accordingly. Default is False.

    Returns:
    -------
    denoised_data : numpy.ndarray
        Denoised data with the same shape as the input (nunits, nconds, ntrials).

    denoiser_out : numpy.ndarray
        Adjusted denoising matrix. If `rescue=True`, voxels that are rescued act as an 
        identity matrix; otherwise, it is the same as the input denoiser.

    noise : numpy.ndarray
        Residual noise data (nunits, nconds, ntrials), defined as the difference 
        between the original and denoised data.

    ncsnrs : list
        Signal-to-noise ratios before and after denoising, and after rescuing (if applicable).

    ncs : list
        Noise ceilings before and after denoising, and after rescuing (if applicable).

    Notes:
    ------
    - The function computes the noise ceiling and signal-to-noise ratio for the data before 
      and after denoising.
    - If `rescue=True`, any voxels with degraded performance (lower SNR after denoising) 
      are "rescued" by retaining their original values.

    Example:
    -------
    >>> data = np.random.rand(100, 10, 20)
    >>> denoiser = np.eye(100)
    >>> denoised_data, denoiser_out, noise, ncsnrs, ncs = apply_denoiser(data, denoiser)

    """
    nunits, nconds, ntrials = data.shape

    # Initialize array for denoised data
    denoised_data = np.zeros(data.shape)

    # Apply denoising to each trial
    for tr in range(ntrials):
        this_data = data.copy()[:, :, tr].T  # Extract data for this trial
        denoised_data[:, :, tr] = (this_data @ denoiser).T  # Apply denoising matrix

    # Compute initial and final noise ceiling and signal-to-noise ratio
    nc_init, ncsnr_init, _, _ = compute_noise_ceiling(data)
    nc_final, ncsnr_final, _, _ = compute_noise_ceiling(denoised_data)

    # Store signal-to-noise ratios and noise ceilings
    ncsnrs = [ncsnr_init, ncsnr_final]
    ncs = [nc_init, nc_final]

    # Handle voxel "rescue" if performance degrades after denoising
    if rescue:
        denoiser_out = denoiser.copy()
        restore_vox = ncsnr_init > ncsnr_final  # Identify voxels with degraded performance
        denoised_data[restore_vox] = data[restore_vox]  # Restore original data for these voxels

        # Adjust denoising matrix to act as identity for rescued voxels
        identity_matrix = np.eye(nunits)
        for vv in np.where(restore_vox)[0]:
            denoiser_out[:, vv] = identity_matrix[:, vv]

        # Recompute noise ceiling and SNR after rescue
        nc_rescued, ncsnr_rescued, _, _ = compute_noise_ceiling(denoised_data)
        ncsnrs.append(ncsnr_rescued)
        ncs.append(nc_rescued)
    else:
        denoiser_out = denoiser

    # Compute residual noise as the difference between original and denoised data
    noise = np.zeros(data.shape)
    for tr in range(ntrials):
        this_data = data.copy()[:, :, tr].T  # Extract data for this trial
        noise[:, :, tr] = this_data.T - denoised_data[:, :, tr]  # Compute residual

    return denoised_data, denoiser_out, noise, ncsnrs, ncs


def compute_noise_ceiling(data_in):
    """
    Compute the noise ceiling signal-to-noise ratio (SNR) and percentage noise ceiling for each unit.
    
    The function calculates noise ceiling metrics for a given dataset, where the data is typically 
    organized as (units/voxels, conditions, trials). The noise ceiling represents the upper limit of 
    performance that can be explained by the data, taking into account the signal and noise variances.

    Parameters:
    ----------
    data_in : np.ndarray
        A 3D array of shape (units/voxels, conditions, trials), representing the data for which to compute 
        the noise ceiling. Each unit requires more than 1 trial for each condition.

    Returns:
    -------
    noiseceiling : np.ndarray
        The noise ceiling for each unit, expressed as a percentage.
    ncsnr : np.ndarray
        The noise ceiling signal-to-noise ratio (SNR) for each unit.
    
    Notes:
    ------
    - The noise variance is computed as the average variance across trials for each unit.
    - The data variance is computed as the variance of the mean across trials for each unit.
    - The signal variance is estimated by subtracting the noise variance from the data variance.
    - The noise ceiling percentage is based on the SNR of the signal relative to the noise.
    """

    # noisevar: mean variance across trials for each unit
    noisevar = np.mean(np.std(data_in, axis=2, ddof=1) ** 2, axis=1)

    # datavar: variance of the trial means across conditions for each unit
    datavar = np.std(np.mean(data_in, axis=2), axis=1, ddof=1) ** 2

    # signalvar: signal variance, obtained by subtracting noise variance from data variance
    signalvar = np.maximum(datavar - noisevar / data_in.shape[2], 0)  # Ensure non-negative variance

    # ncsnr: signal-to-noise ratio (SNR) for each unit
    ncsnr = np.sqrt(signalvar) / np.sqrt(noisevar)

    # noiseceiling: percentage noise ceiling based on SNR
    noiseceiling = 100 * (ncsnr ** 2 / (ncsnr ** 2 + 1 / data_in.shape[2]))

    return noiseceiling, ncsnr, signalvar, noisevar

def negative_mse_columns(dataA, dataB):
    return -np.mean((dataA - dataB) ** 2, axis = 0)

def pearson_r_columns(mat1, mat2):
    """
    Computes the Pearson correlation coefficient between corresponding columns of two matrices.
    
    Parameters:
    mat1, mat2: np.ndarray
        Two matrices of the same shape (n_rows, n_cols), where each column represents a variable.
        
    Returns:
    np.ndarray
        A 1D array of Pearson correlation coefficients between corresponding columns of mat1 and mat2.
    """
    # Ensure the matrices have the same shape
    assert mat1.shape == mat2.shape, "Input matrices must have the same shape."
    
    # Compute column-wise means
    mean1 = np.mean(mat1, axis=0)
    mean2 = np.mean(mat2, axis=0)
    
    # Subtract the means from each element (center the data)
    mat1_centered = mat1 - mean1
    mat2_centered = mat2 - mean2
    
    # Compute the numerator (covariance between corresponding columns)
    covariance = np.sum(mat1_centered * mat2_centered, axis=0)
    
    # Compute the denominator (standard deviations of corresponding columns)
    std1 = np.sqrt(np.sum(mat1_centered**2, axis=0))
    std2 = np.sqrt(np.sum(mat2_centered**2, axis=0))
    
    # Compute Pearson correlation coefficient
    pearson_r = covariance / (std1 * std2 + 1e-8)
    
    return pearson_r

def match_voxel_profiles(inputA, inputB):
    """
    Scales each condition's voxel profile of responses in inputB to match
    the mean and standard deviation of inputA, for each condition/trial independently.
    Works for both (cond, trial, vox) and (cond, vox) inputs.
    
    Parameters:
    - inputA: numpy array of shape (cond, trial, vox) or (cond, vox), the target distribution.
    - inputB: numpy array of shape (cond, trial, vox) or (cond, vox), the data to be scaled.
    
    Returns:
    - scaled_inputB: numpy array of same shape as inputB, with the voxel profiles in inputB 
                     scaled to match the mean and std of inputA for each condition/trial.
    """
    # Detect if input has trials or not
    if inputA.ndim == 2:  # Shape is (cond, vox)
        inputA = inputA[:, np.newaxis, :]  # Add a singleton trial dimension: (cond, 1, vox)
        inputB = inputB[:, np.newaxis, :]  # Add a singleton trial dimension: (cond, 1, vox)

    # Compute the mean and std for each voxel and trial over conditions (axis 0)
    meanA = np.mean(inputA, axis=0, keepdims=True)  # Shape (1, trial, vox)
    stdA = np.std(inputA, axis=0, keepdims=True)    # Shape (1, trial, vox)

    meanB = np.mean(inputB, axis=0, keepdims=True)  # Shape (1, trial, vox)
    stdB = np.std(inputB, axis=0, keepdims=True)    # Shape (1, trial, vox)

    # Normalize inputB to match inputA's mean and std for each voxel and trial
    scaled_inputB = (inputB - meanB) / (stdB + 1e-8)  # Normalize inputB
    scaled_inputB = scaled_inputB * stdA + meanA      # Rescale to match inputA's mean and std

    # If input was 2D, return the output as 2D as well
    if scaled_inputB.shape[1] == 1:
        scaled_inputB = scaled_inputB[:, 0, :]  # Remove the trial dimension if it was originally 2D

    return scaled_inputB

def random_orthonormal_basis(dim):
    """
    Generate a random orthonormal basis of shape (dim, dim).
    
    Parameters:
    - dim (int): The dimension of the square orthonormal matrix to generate.
    
    Returns:
    - Q (numpy.ndarray): A dim x dim orthonormal matrix.
    """
    # Create a random matrix of shape (dim, dim)
    random_matrix = np.random.randn(dim, dim)
    
    # Perform QR decomposition
    Q, _ = np.linalg.qr(random_matrix)
    
    return Q

def plot_basis_dim_ncsnrs(data, eigvecs, basis_name, threshold=None, subplots=(121, 122)):
    """
    Plot signal and noise standard deviations (SD) and noise ceiling SNR (ncsnr)
    for data projected into a given basis.

    Parameters:
    ----------
    data : numpy.ndarray
        Input data with shape (nunits, nconds, ntrials), where:
        - nunits: Number of units (e.g., voxels or features).
        - nconds: Number of conditions.
        - ntrials: Number of trials.

    eigvecs : numpy.ndarray
        Eigenvector matrix (nunits x nunits) representing the basis to project the data.

    basis_name : str
        Name of the basis (used for plot titles).

    threshold : float or None, optional
        Threshold indicating the optimal principal component (PC) dimension. If provided,
        it is visualized on the plots. Default is None.

    subplots : tuple, optional
        Tuple specifying the subplot indices for signal/noise SD and ncsnr plots. 
        Default is (121, 122).

    Returns:
    -------
    None
        The function generates and displays two plots:
        - Signal and noise SDs for dimensions in the basis.
        - Noise ceiling SNR (ncsnr) for dimensions in the basis.

    Notes:
    ------
    - The signal and noise SDs are computed for data projected into each dimension of the basis.
    - ncsnr (noise ceiling signal-to-noise ratio) quantifies the quality of the projection 
      into each basis dimension.

    Example:
    -------
    >>> data = np.random.rand(100, 10, 20)
    >>> eigvecs = np.eye(100)
    >>> plot_basis_dim_ncsnrs(data, eigvecs, basis_name="cSb", threshold=10)

    """
    # Initialize lists to store results for each basis dimension
    ncsnrs = []     # Noise ceiling signal-to-noise ratios
    sigvars = []    # Signal variances
    noisevars = []  # Noise variances

    # Compute ncsnr, signal variance, and noise variance for each basis dimension
    for i in range(data.shape[2]):
        this_eigv = eigvecs[:, i]  # Select the i-th eigenvector
        proj_data = np.dot(data, this_eigv)  # Project data into this eigenvector's subspace

        # Compute noise ceiling metrics for the projected data
        _, ncsnr, sigvar, noisevar = compute_noise_ceiling(proj_data[np.newaxis, ...])
        ncsnrs.append(ncsnr)
        sigvars.append(sigvar)
        noisevars.append(noisevar)

    # Plot signal and noise standard deviations
    plt.subplot(subplots[0])
    plt.plot(np.sqrt(sigvars), linewidth=3, label='Signal SD')  # Signal SD
    plt.plot(np.sqrt(noisevars), linewidth=3, label='Noise SD')  # Noise SD
    plt.xlabel('Dimension')
    plt.ylabel('Standard Deviation')
    plt.title(f'Signal and Noise SD of Data\nProjected into {basis_name} Basis')
    plt.plot([0, len(sigvars)], [0, 0], 'k--', linewidth=0.4)  # Zero line

    # Add threshold line if specified
    if threshold is not None:
        plt.plot([threshold.mean(), threshold.mean()], [0, 6], 'g--', linewidth=2,
                 label=f'Optimal PC Threshold: {threshold.mean()}')

    # Set y-axis limits and display legend
    plt.ylim([-0.2, 5.1])
    plt.legend()

    # Plot noise ceiling signal-to-noise ratio (ncsnr)
    plt.subplot(subplots[1])
    plt.plot(ncsnrs, linewidth=3, color='m', label='ncsnr')  # ncsnr curve
    plt.title(f'NCSNR of Data\nProjected into {basis_name} Basis')
    plt.xlabel('Dimension')
    plt.ylabel('NCSNR')
    plt.plot([0, len(sigvars)], [0, 0], 'k--', linewidth=0.4)  # Zero line

    # Add threshold line if specified
    if threshold is not None:
        plt.plot([threshold.mean(), threshold.mean()], [0, 6], 'g--', linewidth=2,
                 label=f'Optimal PC Threshold: {threshold.mean()}')

    # Set y-axis limits and display legend
    plt.ylim([-0.05, 1.3])
    plt.legend()

    return
