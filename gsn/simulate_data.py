"""
Functions for generating simulated neural data with controlled signal and noise properties.

This module provides tools to generate synthetic neural data with specific covariance 
structures for both signal and noise components. The data generation process allows for:
- Control over signal and noise decay rates
- Alignment between signal and noise principal components
- Separate train and test datasets with matched properties
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def generate_data(nvox, ncond, ntrial, signal_decay=1.0, noise_decay=1.0, 
                 noise_multiplier=1.0, align_alpha=0.0, align_k=0, random_seed=None, 
                 want_fig=False, signal_cov=None, true_signal=None, cluster_units=False, verbose=True):
    """
    Generate synthetic neural data with controlled signal and noise properties.
    
    Args:
        nvox (int):    Number of voxels/units
        ncond (int):   Number of conditions
        ntrial (int):  Number of trials per condition
        signal_decay (float): Rate of eigenvalue decay for signal covariance
        noise_decay (float):  Rate of eigenvalue decay for noise covariance
        noise_multiplier (float): Scaling factor for noise variance
        align_alpha (float): Alignment between signal & noise PCs (1=aligned, 0=orthogonal)
        align_k (int): Number of top PCs to align
        random_seed (int, optional): Random seed for reproducibility
        want_fig (bool): Whether to display a diagnostic figure of the generated data
        signal_cov (ndarray, optional): User-provided signal covariance matrix (nvox, nvox)
                                        If provided, overrides signal_decay parameter
        true_signal (ndarray, optional): User-provided ground truth signal (ncond, nvox)
                                        If provided, overrides signal_cov and signal_decay.
                                        Note: When provided, signal_cov will be calculated
                                        as the sample covariance of this signal.
        cluster_units (bool): Whether to reorder units based on hierarchical clustering
                            of the signal covariance matrix. This is purely cosmetic
                            for visualization and does not affect data properties.
    
    Returns:
        (train_data, test_data, ground_truth)
         - train_data: (nvox, ncond, ntrial)
         - test_data:  (nvox, ncond, ntrial)
         - ground_truth: dict w/ keys:
             'signal'     -> (ncond, nvox)
             'signal_cov' -> (nvox, nvox)
             'noise_cov'  -> (nvox, nvox)
             'U_signal'   -> Original eigenvectors for signal
             'U_noise'    -> Original eigenvectors for noise
             'signal_eigs'
             'noise_eigs'
             'unit_order' -> Original indices of units after clustering (if cluster_units=True)
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState()

    # Check input dimensions if provided
    if signal_cov is not None:
        if signal_cov.shape != (nvox, nvox):
            raise ValueError(f"Provided signal_cov has shape {signal_cov.shape}, expected ({nvox}, {nvox})")
    
    if true_signal is not None:
        if true_signal.shape != (ncond, nvox):
            raise ValueError(f"Provided true_signal has shape {true_signal.shape}, expected ({ncond}, {nvox})")

    # Generate random orthonormal matrices for signal & noise
    U_noise, _, _ = np.linalg.svd(rng.randn(nvox, nvox), full_matrices=True)
    
    # For signal, either use SVD of provided covariance or generate random
    if signal_cov is not None:
        # Use provided signal covariance
        U_signal, signal_eigs, _ = np.linalg.svd(signal_cov)
        signal_cov = np.copy(signal_cov)  # Ensure we have a copy to avoid modifying the input
    else:
        # Generate random orthonormal matrix for signal
        U_signal, _, _ = np.linalg.svd(rng.randn(nvox, nvox), full_matrices=True)
        # Create diagonal eigenvalues
        signal_eigs = 1.0 / (np.arange(1, nvox+1) ** signal_decay)
        # Build signal covariance matrix
        signal_cov = U_signal @ np.diag(signal_eigs) @ U_signal.T

    # Align noise PCs to signal PCs if requested
    if align_k > 0:
        if align_alpha == 1.0:
            U_noise = _shortcut_alignment(U_signal, U_noise, align_k)
        else:
            U_noise = _adjust_alignment_gradient_descent(
                U_signal, U_noise, align_alpha, align_k, verbose=verbose
            )

    # Create diagonal eigenvalues for noise
    noise_eigs = noise_multiplier / (np.arange(1, nvox+1) ** noise_decay)
    # Build noise covariance matrix
    noise_cov = U_noise @ np.diag(noise_eigs) @ U_noise.T

    # Generate the ground truth signal
    if true_signal is not None:
        # Use provided ground truth signal
        true_signal = np.copy(true_signal)  # Ensure we have a copy
        
        # Recalculate signal covariance based on the provided true signal
        # This ensures signal_cov matches the actual covariance of true_signal
        signal_cov = np.cov(true_signal, rowvar=False)
        
        # Recompute signal eigendecomposition for consistency
        U_signal, signal_eigs, _ = np.linalg.svd(signal_cov)
        
        # Re-align noise after recalculating U_signal
        if align_k > 0:
            if align_alpha == 1.0:
                U_noise = _shortcut_alignment(U_signal, U_noise, align_k)
            else:
                U_noise = _adjust_alignment_gradient_descent(
                    U_signal, U_noise, align_alpha, align_k, verbose=verbose
                )
            # Rebuild noise covariance matrix with the realigned eigenvectors
            noise_cov = U_noise @ np.diag(noise_eigs) @ U_noise.T
    else:
        # Generate from covariance
        true_signal = rng.multivariate_normal(
            mean=np.zeros(nvox),
            cov=signal_cov,
            size=ncond
        )  # shape (ncond, nvox)

    # Preallocate train/test data in shape (ntrial, nvox, ncond)
    train_data = np.zeros((ntrial, nvox, ncond))
    test_data = np.zeros((ntrial, nvox, ncond))

    # Generate data
    for t in range(ntrial):
        # Independent noise for each trial
        train_noise = rng.multivariate_normal(
            mean=np.zeros(nvox),
            cov=noise_cov,
            size=ncond
        )  # shape (ncond, nvox)
        test_noise = rng.multivariate_normal(
            mean=np.zeros(nvox),
            cov=noise_cov,
            size=ncond
        )   # shape (ncond, nvox)

        # Add noise to signal
        train_data[t, :, :] = (true_signal + train_noise).T
        test_data[t, :, :]  = (true_signal + test_noise).T

    # Reshape to (nvox, ncond, ntrial)
    train_data = train_data.transpose(1, 2, 0)
    test_data  = test_data.transpose(1, 2, 0)

    # Optionally reorder units based on hierarchical clustering
    unit_order = None
    if cluster_units:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        from scipy.stats import zscore
        
        # Cluster based on ground truth signal patterns
        # true_signal shape is (ncond, nvox), so transpose to get (nvox, ncond)
        # Standardize each unit's activity pattern for better clustering
        signal_for_clustering = zscore(true_signal, axis=0).T  # zscore across conditions for each unit
        
        # Use correlation distance and average linkage for more balanced clusters
        dist = pdist(signal_for_clustering, metric='correlation')
        Z = linkage(dist, method='average')  # Average linkage often gives more balanced clusters
        unit_order = leaves_list(Z)  # Get the reordered indices
        
        # Reorder all relevant matrices and arrays
        train_data = train_data[unit_order]
        test_data = test_data[unit_order]
        true_signal = true_signal[:, unit_order]
        signal_cov = signal_cov[unit_order][:, unit_order]
        noise_cov = noise_cov[unit_order][:, unit_order]
        U_signal = U_signal[unit_order]
        U_noise = U_noise[unit_order]

    ground_truth = {
        'signal':      true_signal,
        'signal_cov':  signal_cov,
        'noise_cov':   noise_cov,
        'U_signal':    U_signal,
        'U_noise':     U_noise,
        'signal_eigs': signal_eigs,
        'noise_eigs':  noise_eigs,
        'user_provided': {
            'signal_cov': signal_cov is not None,
            'true_signal': true_signal is not None
        }
    }
    
    if unit_order is not None:
        ground_truth['unit_order'] = unit_order

    if want_fig:
        plot_data_diagnostic(train_data, ground_truth, {
            'nvox': nvox,
            'ncond': ncond,
            'ntrial': ntrial,
            'signal_decay': signal_decay,
            'noise_decay': noise_decay,
            'noise_multiplier': noise_multiplier,
            'align_alpha': align_alpha,
            'align_k': align_k,
            'random_seed': random_seed,
            'user_provided': ground_truth['user_provided'],
            'clustered': cluster_units
        })

    return train_data, test_data, ground_truth

def _adjust_alignment(U_signal, U_noise, alpha, k, tolerance=1e-9):
    """
    Adjust alignment between the top-k columns of U_signal and U_noise,
    while ensuring that the final U_noise_adjusted is orthonormal.

    Args:
        U_signal : (nvox, nvox) orthonormal (columns are principal dirs)
        U_noise  : (nvox, nvox) orthonormal
        alpha    : float in [0,1], where 1 => perfect alignment, 0 => orthogonal
        k        : int, number of top PCs to align
        tolerance: numeric tolerance for final orthonormal checks

    Returns:
        U_noise_adjusted : (nvox, nvox), orthonormal, with desired alignment in first k PCs
    """
    if not (0 <= alpha <= 1):
        warnings.warn("alpha must be in [0,1]; will be clamped.")
        alpha = max(0, min(alpha, 1))

    nvox = U_signal.shape[0]
    if k > nvox:
        raise ValueError("k cannot exceed the number of columns in U_signal.")
    
    # If k=0, return original noise basis
    if k == 0:
        return U_noise.copy()

    # Start with a copy of U_noise
    U_noise_adjusted = U_noise.copy()

    # For each of the first k components
    for i in range(k):
        v_sig = U_signal[:, i]
        v_noise = U_noise[:, i]

        # Calculate the projection of v_noise onto v_sig
        proj = np.dot(v_noise, v_sig)
        
        # Create a vector that's orthogonal to v_sig
        v_orth = v_noise - proj * v_sig
        v_orth_norm = np.linalg.norm(v_orth)
        
        if v_orth_norm < tolerance:
            # If v_noise is too close to v_sig, find another orthogonal vector
            for j in range(nvox):
                e_j = np.zeros(nvox)
                e_j[j] = 1.0
                v_candidate = e_j - (np.dot(e_j, v_sig) * v_sig)
                v_candidate_norm = np.linalg.norm(v_candidate)
                if v_candidate_norm > tolerance:
                    v_orth = v_candidate / v_candidate_norm
                    break
        else:
            v_orth = v_orth / v_orth_norm

        # Create the aligned vector using exact trigonometric relationships
        # alpha = cos(theta), where theta is the angle between vectors
        # We want to create a vector that makes angle theta with v_sig
        v_aligned = alpha * v_sig + np.sqrt(1 - alpha**2) * v_orth
        v_aligned = v_aligned / np.linalg.norm(v_aligned)

        # Update the i-th column
        U_noise_adjusted[:, i] = v_aligned

        # Orthogonalize all remaining columns with respect to this one
        for j in range(i + 1, nvox):
            v_j = U_noise_adjusted[:, j]
            # Project out the component in the direction of v_aligned
            v_j = v_j - (np.dot(v_j, v_aligned) * v_aligned)
            v_j_norm = np.linalg.norm(v_j)
            if v_j_norm > tolerance:
                U_noise_adjusted[:, j] = v_j / v_j_norm
            else:
                # If the vector becomes degenerate, find a replacement
                for idx in range(nvox):
                    e_idx = np.zeros(nvox)
                    e_idx[idx] = 1.0
                    # Make orthogonal to all previous vectors
                    for m in range(j):
                        e_idx = e_idx - (np.dot(e_idx, U_noise_adjusted[:, m]) * U_noise_adjusted[:, m])
                    e_norm = np.linalg.norm(e_idx)
                    if e_norm > tolerance:
                        U_noise_adjusted[:, j] = e_idx / e_norm
                        break

    # Final orthogonalization pass to ensure numerical stability
    for i in range(nvox):
        v_i = U_noise_adjusted[:, i]
        # Orthogonalize with respect to all previous vectors
        for j in range(i):
            v_i = v_i - (np.dot(v_i, U_noise_adjusted[:, j]) * U_noise_adjusted[:, j])
        v_i_norm = np.linalg.norm(v_i)
        if v_i_norm > tolerance:
            U_noise_adjusted[:, i] = v_i / v_i_norm

    # Verify the alignment
    for i in range(k):
        actual_alpha = np.abs(np.dot(U_signal[:, i], U_noise_adjusted[:, i]))
        if abs(actual_alpha - alpha) > tolerance:
            print(f"Alignment verification failed for dimension {i}: "
                       f"expected {alpha}, got {actual_alpha}")

    return U_noise_adjusted

def _adjust_alignment_gradient_descent(U_signal, U_noise_init, alpha, k,
                                        lr=5e-1, lambda_orth=1.0,
                                        num_steps=10000,
                                        tol_align=1e-6, tol_orth=1e-6,
                                        verbose=True):
    """
    Gradient descent method to align U_noise to U_signal's top-k PCs
    with dot(U_noise[:, i], U_signal[:, i]) ≈ alpha.

    Returns:
        U_noise_aligned (nvox, nvox): new orthonormal basis
    """
    U = U_noise_init.copy()
    nvox = U.shape[0]
    I = np.eye(nvox)

    for step in range(1, num_steps + 1):
        grad = np.zeros_like(U)
        align_vals = []
        for i in range(k):
            dot = np.dot(U[:, i], U_signal[:, i])
            align_vals.append(dot)
            grad[:, i] = (dot - alpha) * U_signal[:, i]
        M = U.T @ U - I
        grad += lambda_orth * ((U @ M))
        U -= lr * grad

        max_align_err = max(abs(a - alpha) for a in align_vals)
        orth_err = np.linalg.norm(U.T @ U - I)

        #if verbose and step % max(1, num_steps // 10) == 0:
            #print(f"Step {step}/{num_steps}: align_err={max_align_err:.2e}, orth_err={orth_err:.2e}")

        if max_align_err < tol_align and orth_err < tol_orth:
            if verbose:   
                print(f"\t\tOptimization complete. Step {step}/{num_steps}: align_err={max_align_err:.2e}, orth_err={orth_err:.2e}")
            return U
               
    print(f"Optimization did not converge. Step {step}/{num_steps}: align_err={max_align_err:.2e}, orth_err={orth_err:.2e}")

    return U

def _shortcut_alignment(U_signal, U_noise, k):
    """
    Directly align the first k noise PCs to the signal PCs without optimization.
    """
    U_noise_adj = U_noise.copy()
    nvox = U_signal.shape[0]

    # Set top k noise PCs equal to signal PCs
    U_noise_adj[:, :k] = U_signal[:, :k]

    # Orthonormalize remaining PCs via Gram-Schmidt
    for i in range(k, nvox):
        v = U_noise_adj[:, i].copy()
        for j in range(k):
            v -= np.dot(v, U_noise_adj[:, j]) * U_noise_adj[:, j]
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            # Choose a random orthogonal vector if degenerate
            v = np.random.randn(nvox)
            for j in range(k):
                v -= np.dot(v, U_noise_adj[:, j]) * U_noise_adj[:, j]
            norm = np.linalg.norm(v)
        U_noise_adj[:, i] = v / norm

    return U_noise_adj

def plot_data_diagnostic(data, ground_truth, params):
    """
    Generate a comprehensive diagnostic figure for simulated data.
    
    Args:
        data (ndarray): The simulated data with shape (nvox, ncond, ntrial)
        ground_truth (dict): Dictionary with ground truth information
        params (dict): Dictionary with simulation parameters
    """
    # Extract important parameters and ground truth values
    nvox = params['nvox']
    ncond = params['ncond']
    ntrial = params['ntrial']
    signal_decay = params['signal_decay']
    noise_decay = params['noise_decay']
    noise_multiplier = params['noise_multiplier']
    align_alpha = params['align_alpha']
    align_k = params['align_k']
    user_provided = params.get('user_provided', {'signal_cov': False, 'true_signal': False})
    
    # Extract ground truth matrices
    signal_cov = ground_truth['signal_cov']
    noise_cov = ground_truth['noise_cov']
    signal_eigs = ground_truth['signal_eigs']
    noise_eigs = ground_truth['noise_eigs']
    U_signal = ground_truth['U_signal']
    U_noise = ground_truth['U_noise']
    true_signal = ground_truth['signal']
    
    # Create example trial data for visualization
    trial_avg = np.mean(data, axis=2)  # Average across trials
    example_trial = data[:, :, 0]      # First trial
    
    # Create figure with GridSpec for flexible subplot layout
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.8, 1, 1])
    
    # Add title with parameters
    title_text = (
        f"Simulated Data: {nvox} units × {ncond} conditions × {ntrial} trials\n"
    )
    
    # Add appropriate source info based on what was user-provided
    if user_provided.get('true_signal', False):
        title_text += "Using user-provided ground truth signal\n"
    elif user_provided.get('signal_cov', False):
        title_text += "Using user-provided signal covariance matrix\n"
    else:
        title_text += f"Signal decay={signal_decay:.2f}, Noise decay={noise_decay:.2f}, "
        title_text += f"Noise multiplier={noise_multiplier:.2f}\n"
    
    title_text += f"Alignment: alpha={align_alpha:.2f} (0=orthogonal, 1=aligned), k={align_k} top PCs"
    
    # Add note about clustering if units were reordered
    if params.get('clustered', False):
        title_text += "\nUnits reordered by hierarchical clustering"
    
    fig.suptitle(title_text, fontsize=14)
    
    # Split the first row into two parts for log and linear eigenvalue plots
    gs_eigen = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0:2])
    
    # Plot 1a: Eigenvalue spectra - Log scale
    ax1a = fig.add_subplot(gs_eigen[0, 0])
    ax1a.semilogy(np.arange(nvox), signal_eigs, 'b-', label='Signal eigenvalues')
    ax1a.semilogy(np.arange(nvox), noise_eigs, 'r-', label='Noise eigenvalues')
    if align_k > 0:
        ax1a.axvline(x=align_k-1, color='gray', linestyle='--', 
                   label=f'Alignment cutoff (k={align_k})')
    ax1a.set_xlabel('Dimension')
    ax1a.set_ylabel('Eigenvalue (log scale)')
    ax1a.set_title('Eigenspectrum - Log Scale')
    ax1a.legend()
    ax1a.grid(True, alpha=0.3)
    
    # Plot 1b: Eigenvalue spectra - Linear scale
    ax1b = fig.add_subplot(gs_eigen[0, 1])
    ax1b.plot(np.arange(nvox), signal_eigs, 'b-', label='Signal eigenvalues')
    ax1b.plot(np.arange(nvox), noise_eigs, 'r-', label='Noise eigenvalues')
    if align_k > 0:
        ax1b.axvline(x=align_k-1, color='gray', linestyle='--',
                   label=f'Alignment cutoff (k={align_k})')
    ax1b.set_xlabel('Dimension')
    ax1b.set_ylabel('Eigenvalue (linear scale)')
    ax1b.set_title('Eigenspectrum - Linear Scale')
    # Show top 25% of dimensions for better visualization in linear scale
    dims_to_show = max(int(nvox * 0.25), align_k + 5)
    ax1b.set_xlim(0, dims_to_show)
    ax1b.legend()
    ax1b.grid(True, alpha=0.3)
    
    # Plot 2: Signal-to-noise ratio per dimension
    ax2 = fig.add_subplot(gs[0, 2:4])
    snr = signal_eigs / noise_eigs
    ax2.plot(np.arange(nvox), snr, 'g-')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('SNR')
    ax2.set_title('Signal-to-Noise Ratio per Dimension')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Signal covariance matrix
    ax3 = fig.add_subplot(gs[1, 0])
    # Calculate clim values similar to gsn_denoise.py
    cov_max = np.percentile(np.abs(signal_cov), 95)
    im3 = ax3.imshow(signal_cov, aspect='equal', cmap='RdBu_r', interpolation='none',
                   vmin=-cov_max, vmax=cov_max)
    plt.colorbar(im3, ax=ax3, label='Covariance')
    
    # Update title to be more informative about signal_cov source
    if user_provided.get('true_signal', False):
        title_suffix = "(Derived from user-provided GT signal)"
    elif user_provided.get('signal_cov', False):
        title_suffix = ""
    else:
        title_prefix = ""
        
    ax3.set_title(f'Signal Covariance Matrix\n{title_suffix}')
    ax3.set_xlabel('Unit')
    ax3.set_ylabel('Unit')
    
    # Plot 4: Noise covariance matrix
    ax4 = fig.add_subplot(gs[1, 1])
    # Use same colormap and scaling approach as signal covariance
    cov_max = np.percentile(np.abs(noise_cov), 95)
    im4 = ax4.imshow(noise_cov, aspect='equal', cmap='RdBu_r', interpolation='none',
                   vmin=-cov_max, vmax=cov_max)
    plt.colorbar(im4, ax=ax4, label='Covariance')
    ax4.set_title('Noise Covariance Matrix')
    ax4.set_xlabel('Unit')
    ax4.set_ylabel('Unit')
    
    # Calculate shared colorbar limits for signal and trial data
    # Combine the ground truth signal and example trial data to find common limits
    all_data = np.concatenate([true_signal.T.flatten(), example_trial.flatten()])
    data_min = np.percentile(all_data, 1)  # Use 1st percentile instead of min to avoid outliers
    data_max = np.percentile(all_data, 99)  # Use 99th percentile instead of max to avoid outliers
    data_abs_max = max(abs(data_min), abs(data_max))
    
    # Use symmetric limits for better visualization
    signal_clim = (-data_abs_max, data_abs_max)
    
    # Plot 5: Example of ground truth signal - show full matrix
    ax5 = fig.add_subplot(gs[1, 2])
    im5 = ax5.imshow(true_signal.T, aspect='auto', cmap='RdBu_r', interpolation='none',
                   vmin=signal_clim[0], vmax=signal_clim[1])
    plt.colorbar(im5, ax=ax5, label='Signal')
    title_prefix = "" if user_provided.get('true_signal', False) else ""
    ax5.set_title(f'{title_prefix}Ground Truth Signal')
    ax5.set_xlabel('Condition')
    ax5.set_ylabel('Unit')
    
    # Plot 6: Example trial data - show full matrix
    ax6 = fig.add_subplot(gs[1, 3])
    im6 = ax6.imshow(example_trial, aspect='auto', cmap='RdBu_r', interpolation='none',
                   vmin=signal_clim[0], vmax=signal_clim[1])
    plt.colorbar(im6, ax=ax6, label='Activity')
    ax6.set_title('Example Single Trial (with noise)')
    ax6.set_xlabel('Condition')
    ax6.set_ylabel('Unit')
    
    # Plot 7: Signal eigenvectors
    ax7 = fig.add_subplot(gs[2, 0])
    im7 = ax7.imshow(U_signal, aspect='auto', cmap='RdBu_r',
                   vmin=-0.3, vmax=0.3)
    plt.colorbar(im7, ax=ax7, label='Weight')
    ax7.set_title(f'Signal Eigenvectors')
    ax7.set_xlabel('Dimension')
    ax7.set_ylabel('Unit')
    
    # Plot 8: Noise eigenvectors (top k)
    ax8 = fig.add_subplot(gs[2, 1])
    im8 = ax8.imshow(U_noise, aspect='auto', cmap='RdBu_r',
                   vmin=-0.3, vmax=0.3)
    plt.colorbar(im8, ax=ax8, label='Weight')
    ax8.set_title(f'Noise Eigenvectors')
    ax8.set_xlabel('Dimension')
    ax8.set_ylabel('Unit')
    
    # Plot 9: Alignment visualization (dot products between signal and noise eigenvectors)
    ax9 = fig.add_subplot(gs[2, 2])
    # Calculate full alignment matrix for all dimensions
    full_alignment_matrix = U_signal.T @ U_noise
    
    # Determine how many dimensions to display in the visualization
    # If nvox is large, subsample the matrix but ensure we include the aligned dimensions
    max_dims_to_show = min(25, nvox)  # Limit to 50x50 at most for readability
    
    if nvox <= max_dims_to_show:
        # If we have fewer dimensions than the limit, show all
        alignment_matrix = full_alignment_matrix
        dims_shown = nvox
    else:
        # Otherwise, show a subset with emphasis on aligned dimensions
        if align_k > 0:
            # Always include the aligned dimensions
            indices = list(range(align_k))
            
            # Add additional dimensions, evenly spaced
            remaining_spots = max_dims_to_show - align_k
            if remaining_spots > 0:
                # Determine spacing for remaining dimensions
                step = (nvox - align_k) / (remaining_spots + 1)
                for i in range(1, remaining_spots + 1):
                    idx = align_k + int(i * step)
                    indices.append(min(idx, nvox - 1))  # Ensure we don't exceed bounds
            
            # Sort indices to maintain proper order
            indices = sorted(indices)
        else:
            # No alignment, just evenly space the indices
            indices = np.linspace(0, nvox-1, max_dims_to_show, dtype=int)
        
        # Extract the submatrix
        alignment_matrix = full_alignment_matrix[indices, :][:, indices]
        dims_shown = len(indices)
    
    # Create the visualization
    im9 = ax9.imshow(alignment_matrix, aspect='equal', cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im9, ax=ax9, label='Dot product')
    ax9.set_title(f'Eigenvector Alignment (dot products)')
    ax9.set_xlabel('Noise dimension')
    ax9.set_ylabel('Signal dimension')
    
    # Add grid lines to better separate the dimensions
    ax9.set_xticks(np.arange(-.5, dims_shown, 1), minor=True)
    ax9.set_yticks(np.arange(-.5, dims_shown, 1), minor=True)
    ax9.grid(which="minor", color="w", linestyle='-', linewidth=0.5, alpha=0.2)
    
    # Add diagonal values text to show exact alignment of corresponding eigenvectors
    if align_k > 0:
        for i in range(min(dims_shown, align_k)):
            ax9.text(i, i, f'{alignment_matrix[i, i]:.2f}', 
                   ha='center', va='center', color='white', fontweight='bold')
    
    # Add some axis labels for clarity, using sparse labeling if there are many dimensions
    if dims_shown <= 10:
        ax9.set_xticks(np.arange(dims_shown))
        ax9.set_yticks(np.arange(dims_shown))
    else:
        # Show fewer ticks for readability
        step = max(1, dims_shown // 10)
        ax9.set_xticks(np.arange(0, dims_shown, step))
        ax9.set_yticks(np.arange(0, dims_shown, step))
    
    # Show the total number of dimensions in title if we're displaying a subset
    if dims_shown < nvox:
        ax9.set_title(f'Eigenvector Alignment (showing {dims_shown}/{nvox} dimensions)')
    
    # Plot 10: Trial-averaged data - also use the same colorbar limits
    ax10 = fig.add_subplot(gs[2, 3])
    im10 = ax10.imshow(trial_avg, aspect='auto', cmap='RdBu_r', interpolation='none',
                     vmin=signal_clim[0], vmax=signal_clim[1])
    plt.colorbar(im10, ax=ax10, label='Activity')
    ax10.set_title(f'Trial-averaged Data ({ntrial} trials)')
    ax10.set_xlabel('Condition')
    ax10.set_ylabel('Unit')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.show()
    
    return fig

    