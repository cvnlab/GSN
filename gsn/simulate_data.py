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

def generate_data(nvox, ncond, ntrial, signal_decay=1.0, noise_decay=1.0, 
                 noise_multiplier=1.0, align_alpha=0.0, align_k=0, random_seed=None):
    """
    Generate synthetic neural data with controlled signal and noise properties.
    
    Args:
        nvox (int):    Number of voxels/units
        ncond (int):   Number of conditions
        ntrial (int):  Number of trials per condition
        signal_decay (float): Rate of eigenvalue decay for signal covariance
        noise_decay (float):  Rate of eigenvalue decay for noise covariance
        noise_multiplier (float): Scaling factor for noise variance
        align_alpha (float): Alignment between signal & noise PCs (0=aligned, 1=orthogonal)
        align_k (int): Number of top PCs to align
        random_seed (int, optional): Random seed for reproducibility
    
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
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState()

    # Generate random orthonormal matrices for signal & noise
    U_signal, _, _ = np.linalg.svd(rng.randn(nvox, nvox), full_matrices=True)
    U_noise,  _, _ = np.linalg.svd(rng.randn(nvox, nvox), full_matrices=True)

    # Possibly adjust noise eigenvectors alignment
    if align_k > 0:
        U_noise = _adjust_alignment(U_signal, U_noise, align_alpha, align_k)

    # Create diagonal eigenvalues
    signal_eigs = 1.0 / (np.arange(1, nvox+1) ** signal_decay)
    noise_eigs  = noise_multiplier / (np.arange(1, nvox+1) ** noise_decay)

    # Build covariance matrices
    signal_cov = U_signal @ np.diag(signal_eigs) @ U_signal.T
    noise_cov  = U_noise  @ np.diag(noise_eigs)  @ U_noise.T

    # Generate the ground truth signal
    true_signal = rng.multivariate_normal(
        mean=np.zeros(nvox),
        cov=signal_cov,
        size=ncond
    )  # shape (ncond, nvox)

    # Preallocate train/test data in shape (ntrial, nvox, ncond)
    train_data = np.zeros((ntrial, nvox, ncond))
    test_data  = np.zeros((ntrial, nvox, ncond))

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

    ground_truth = {
        'signal':      true_signal,
        'signal_cov':  signal_cov,
        'noise_cov':   noise_cov,
        'U_signal':    U_signal,
        'U_noise':     U_noise,
        'signal_eigs': signal_eigs,
        'noise_eigs':  noise_eigs
    }

    return train_data, test_data, ground_truth


def _householder_reflection(v):
    """
    Compute a Householder reflection matrix that maps v to [|v|, 0, ..., 0].
    Used internally for some alignment steps. (Currently unused.)
    """
    v = v.copy()
    n = len(v)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        return np.eye(n)
    v /= v_norm
    e1 = np.zeros(n)
    e1[0] = 1.0
    u = v - e1
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-10:
        return np.eye(n)
    u /= u_norm
    return np.eye(n) - 2 * np.outer(u, u)


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

        # Create a vector that's orthogonal to v_sig
        v_orth = v_noise - (np.dot(v_noise, v_sig) * v_sig)
        v_orth_norm = np.linalg.norm(v_orth)
        
        if v_orth_norm < 1e-10:
            # If v_noise is too close to v_sig, find another orthogonal vector
            for j in range(nvox):
                e_j = np.zeros(nvox)
                e_j[j] = 1.0
                v_candidate = e_j - (np.dot(e_j, v_sig) * v_sig)
                v_candidate_norm = np.linalg.norm(v_candidate)
                if v_candidate_norm > 1e-10:
                    v_orth = v_candidate / v_candidate_norm
                    break
        else:
            v_orth = v_orth / v_orth_norm

        # Create the aligned vector as a weighted combination
        v_aligned = alpha * v_sig + np.sqrt(1 - alpha**2) * v_orth
        v_aligned = v_aligned / np.linalg.norm(v_aligned)

        # Update the i-th column
        U_noise_adjusted[:, i] = v_aligned

        # Orthogonalize all remaining columns with respect to this one
        for j in range(i + 1, nvox):
            v_j = U_noise_adjusted[:, j]
            v_j = v_j - (np.dot(v_j, v_aligned) * v_aligned)
            v_j_norm = np.linalg.norm(v_j)
            if v_j_norm > 1e-10:
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
                    if e_norm > 1e-10:
                        U_noise_adjusted[:, j] = e_idx / e_norm
                        break

    # Final orthogonalization pass to ensure numerical stability
    for i in range(nvox):
        v_i = U_noise_adjusted[:, i]
        # Orthogonalize with respect to all previous vectors
        for j in range(i):
            v_i = v_i - (np.dot(v_i, U_noise_adjusted[:, j]) * U_noise_adjusted[:, j])
        v_i_norm = np.linalg.norm(v_i)
        if v_i_norm > 1e-10:
            U_noise_adjusted[:, i] = v_i / v_i_norm

    return U_noise_adjusted

    