import numpy as np
import matplotlib.pyplot as plt
from fastprogress import progress_bar

def denoise_data(train_data, test_data, V, thresholds, threshold_per='population', match_profiles=False, rescue = True):
    """
    Denoise data by applying dimensionality reduction and thresholding
    on principal components.

    Parameters:
    train_data (numpy.ndarray): Training data of shape (conditions, trials, voxels).
    test_data (numpy.ndarray): Test data of shape (conditions, trials, voxels).
    V (numpy.ndarray): Eigenvector matrix for dimensionality reduction.
    thresholds (list or numpy.ndarray): List of threshold values to apply for denoising.
    threshold_per (str, optional): Specifies if thresholds should be applied per unit ('unit') 
        or for the entire region of interest ('population'). Default: 'population'.
    match_profiles (bool, optional): If True, match voxel profiles after denoising.
        Default: False.
    rescue (bool, optional): If True, "rescue" voxels by restoring their original values
        if denoising had a negative impact on ncsnr.

    Returns:
    denoised_test_data (numpy.ndarray): Denoised version of test_data with the same shape.
    denoised_tuning_corrs (numpy.ndarray): Correlation matrix between denoised and trial-averaged training data.
    denoiser (numpy.ndarray): Denoising matrix applied to test data.
    best_threshold (float or int): Optimal threshold selected based on correlation maximization.
    ncsnrs (numpy.ndarray): initial and denoised ncsnr values per voxel, stored in a list.
    """
    ncond, ntrial, nvox = train_data.shape
    
    # Validate data consistency
    assert np.all(np.isfinite(train_data)), "train_data contains NaN or inf values."
    assert np.all(np.isfinite(test_data)), "test_data contains NaN or inf values."
    
    # Initialize array to hold denoised tuning correlations
    denoised_tuning_corrs = np.zeros((len(thresholds), ntrial, nvox))
    all_denoisers = {} if threshold_per == 'unit' else None

    # Iterate through each threshold and apply denoising per trial
    for tt, threshold in enumerate(progress_bar(thresholds)):
        
        if threshold_per == 'unit':
            all_denoisers[threshold] = np.zeros((ntrial, nvox, nvox))
        
        for tr in range(ntrial):
            # Split data for cross-validation
            this_train_data = train_data[:, tr]
            trial_avg_data = np.mean(train_data[:, np.setdiff1d(np.arange(ntrial), tr)], axis=1)
            
            # Define denoising function based on threshold
            denoising_fn = np.concatenate([np.ones(threshold), np.zeros(nvox - threshold)])
            denoiser = V @ np.diag(denoising_fn) @ V.T
            denoised_data = this_train_data @ denoiser

            if threshold_per == 'unit':
                all_denoisers[threshold][tr] = denoiser

            # Optionally match profiles to maintain voxel structure
            if match_profiles:
                denoised_data = match_voxel_profiles(this_train_data, denoised_data)

            # Calculate correlation with trial-averaged data
            denoised_tuning_corrs[tt, tr] = pearson_r_columns(denoised_data, trial_avg_data)

    assert np.all(np.isfinite(denoised_tuning_corrs)), "denoised_tuning_corrs contains NaN or inf values."

    # Calculate average tuning correlations across trials
    mean_denoised_tuning_corrs = np.mean(denoised_tuning_corrs, axis=1)
    
    # Initialize denoised test data
    denoised_test_data = np.zeros_like(test_data)

    # Select optimal threshold and apply to test data based on threshold type
    if threshold_per == 'population':
        best_threshold = thresholds[np.argmax(np.mean(mean_denoised_tuning_corrs, axis=1))]
        denoising_fn = np.concatenate([np.ones(best_threshold), np.zeros(nvox - best_threshold)])
        denoiser = V @ np.diag(denoising_fn) @ V.T

        # Apply denoising to each trial in the test data
        for tr in range(ntrial):
            this_test_data = test_data[:, tr]
            denoised_test_data[:, tr] = this_test_data @ denoiser
            if match_profiles:
                denoised_test_data[:, tr] = match_voxel_profiles(this_test_data, denoised_test_data[:, tr])

    elif threshold_per == 'unit':
        best_threshold = thresholds[np.argmax(mean_denoised_tuning_corrs, axis=0)]
        denoiser = np.zeros((nvox, nvox))

        # Construct voxel-specific denoiser by averaging across trials
        for vv in range(nvox):
            this_denoiser = np.mean(all_denoisers[best_threshold[vv]], axis=0)
            denoiser[:, vv] = this_denoiser[:, vv]

        # Apply voxel-specific denoising to each trial in the test data
        for tr in range(ntrial):
            this_test_data = test_data[:, tr]
            denoised_test_data[:, tr] = this_test_data @ denoiser
            if match_profiles:
                denoised_test_data[:, tr] = match_voxel_profiles(this_test_data, denoised_test_data[:, tr])

    # Ensure denoised test data has no NaN or inf values
    assert np.all(np.isfinite(denoised_test_data)), "denoised_test_data contains NaN or inf values."

    
    nc_init, ncsnr_init, _, _ = compute_noise_ceiling(test_data.copy().transpose(2,0,1))
    nc_final, ncsnr_final, _, _ = compute_noise_ceiling(denoised_test_data.copy().transpose(2,0,1))
    
    ncsnrs = [ncsnr_init, ncsnr_final]
    ncs = [nc_init, nc_final]
    
    if rescue:
        restore_vox = ncsnr_init > ncsnr_final
        denoised_test_data[:,:,restore_vox] = test_data[:,:,restore_vox]
        
        # Set rescued voxels in the denoiser to act as an identity matrix
        identity_matrix = np.eye(nvox)
        for vv in np.where(restore_vox)[0]:
            denoiser[:, vv] = identity_matrix[:, vv]
            
        nc_rescued, ncsnr_rescued, _, _ = compute_noise_ceiling(denoised_test_data.copy().transpose(2,0,1))
        ncsnrs.append(ncsnr_rescued)
        ncs.append(nc_rescued)
    
    return denoised_test_data, denoised_tuning_corrs, denoiser, best_threshold, ncsnrs, ncs


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

def plot_basis_dim_ncsnrs(data, eigvecs, basis_name, threshold = None, subplots = (121, 122)):
    
    ncsnrs = []
    sigvars = []
    noisevars = []
    for i in range(data.shape[2]):

        this_eigv = eigvecs[:,i]
        proj_data = np.dot(data, this_eigv)

        _,ncsnr,sigvar,noisevar = compute_noise_ceiling(proj_data[np.newaxis,...])
        ncsnrs.append(ncsnr)
        sigvars.append(sigvar)
        noisevars.append(noisevar)
    
    plt.subplot(subplots[0])
    plt.plot(np.sqrt(sigvars),linewidth=3,label='signal SD')
    plt.plot(np.sqrt(noisevars),linewidth=3,label='noise SD')
    
    plt.xlabel('dimension')
    plt.ylabel('standard dev.')
    plt.title(f'signal and noise SD of data\nprojected into {basis_name} basis')
    plt.plot([0,len(sigvars)],[0,0],'k--',linewidth=0.4)
    if threshold is not None:
        plt.plot([threshold.mean(),threshold.mean()],[0,6],'g--',linewidth=2,label=f'optimal PC threshold: {threshold.mean()}')
    plt.ylim([-0.2,5.1])
    plt.legend()
    
    plt.subplot(subplots[1])
    plt.plot(ncsnrs,linewidth=3,color='m',label='ncsnr')
    plt.title(f'ncsnr of data\nprojected into {basis_name} basis')
    plt.xlabel('dimension')
    plt.ylabel('ncsnr')
   
    plt.plot([0,len(sigvars)],[0,0],'k--',linewidth=0.4)
    if threshold is not None:
        plt.plot([threshold.mean(),threshold.mean()],[0,6],'g--',linewidth=2,label=f'optimal PC threshold: {threshold.mean()}')
    plt.legend()
    plt.ylim([-0.05,1.3])
    
    return