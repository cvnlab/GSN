"""Tests for gsn_denoise.py using simulated data."""

import sys, os
sys.path.insert(0, os.path.abspath("/Users/jacobprince/KonkLab Dropbox/Jacob Prince/Research-Prince/GSNdenoise/GSN"))

import numpy as np
import pytest
from gsn.simulate_data import generate_data
from gsn.gsn_denoise import gsn_denoise, perform_cross_validation, perform_magnitude_thresholding
from gsn.perform_gsn import perform_gsn

def test_basic_functionality():
    # Create synthetic data
    nunits, nconds, ntrials = 10, 20, 5
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test default parameters
    denoiser, scores, threshold = gsn_denoise(data)
    assert denoiser.shape == (nunits, nunits)
    # Denoiser should be symmetric
    assert np.allclose(denoiser, denoiser.T)

def test_cross_validation_unitwise():
    """
    Test cross-validation with 'unit'-wise thresholding.
    """
    nunits, nconds, ntrials = 10, 20, 5
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)  # Identity basis for testing
    
    opt = {
        'cv_mode': 0,
        'cv_threshold_per': 'unit',
        'cv_thresholds': np.arange(1, nunits + 1),
        'cv_scoring_fn': lambda A, B: -np.mean((A - B)**2, axis=0)
    }
    denoiser, cv_scores, threshold = perform_cross_validation(data, basis, opt)
    assert denoiser.shape == (nunits, nunits)
    assert len(threshold) == nunits
    assert all(t <= nunits for t in threshold)

def test_cross_validation_population():
    """
    Test cross-validation with 'population'-wise thresholding.
    """
    nunits, nconds, ntrials = 8, 10, 4
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)
    
    opt = {
        'cv_mode': 0,
        'cv_threshold_per': 'population',
        'cv_thresholds': np.arange(1, nunits + 1),
        'cv_scoring_fn': lambda A, B: -np.mean((A - B)**2)
    }
    denoiser, cv_scores, threshold = perform_cross_validation(data, basis, opt)
    assert denoiser.shape == (nunits, nunits)
    assert isinstance(threshold, (int, np.integer))
    assert threshold <= nunits

def test_cross_validation_mode1():
    """
    Test cross-validation with cv_mode=1 (1/n-1 split).
    """
    nunits, nconds, ntrials = 6, 8, 4
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)
    opt = {
        'cv_mode': 1,
        'cv_threshold_per': 'population',
        'cv_thresholds': np.arange(1, nunits + 1),
        'cv_scoring_fn': lambda A, B: -np.mean((A - B)**2)
    }
    denoiser, cv_scores, threshold = perform_cross_validation(data, basis, opt)
    assert denoiser.shape == (nunits, nunits)
    assert isinstance(threshold, (int, np.integer))

def test_magnitude_thresholding_contiguous():
    nunits, nconds, ntrials = 10, 20, 5
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)
    gsn_results = {'cSb': np.eye(nunits), 'cNb': np.eye(nunits)}

    opt = {
        'mag_type': 0,   # use eigenvalues
        'mag_frac': 0.5,
        'mag_mode': 0,   # contiguous
        'cv_threshold_per': 'population'  # Use population-wise thresholding
    }
    denoiser, _, threshold = perform_magnitude_thresholding(data, basis, gsn_results, opt, V=0)
    # For contiguous mode with population-wise thresholding, threshold should be a scalar
    assert isinstance(threshold, (int, np.integer))
    assert threshold <= nunits

def test_magnitude_thresholding_noncontiguous():
    nunits, nconds, ntrials = 10, 20, 5
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)
    gsn_results = {'cSb': np.eye(nunits), 'cNb': np.eye(nunits)}
    
    opt = {
        'mag_type': 1,   # signal variance
        'mag_frac': 0.2,
        'mag_mode': 1,   # keep all surviving
        'cv_threshold_per': 'unit'  # Required parameter
    }
    denoiser, _, threshold = perform_magnitude_thresholding(data, basis, gsn_results, opt, V=np.eye(nunits))
    assert len(threshold) <= nunits

def test_gsn_with_random_basis():
    """
    Test gsn_denoise with V=4 (random orthonormal basis).
    """
    nunits, nconds, ntrials = 8, 15, 3
    data = np.random.randn(nunits, nconds, ntrials)
    denoiser, scores, threshold = gsn_denoise(data, V=4)
    assert denoiser.shape == (nunits, nunits)
    assert np.allclose(denoiser, denoiser.T)

def test_gsn_with_user_supplied_basis():
    """
    Test gsn_denoise with a user-supplied basis.
    """
    nunits, nconds, ntrials = 8, 15, 3
    data = np.random.randn(nunits, nconds, ntrials)
    Q, _ = np.linalg.qr(np.random.randn(nunits, nunits))
    denoiser, scores, threshold = gsn_denoise(data, V=Q)
    assert denoiser.shape == (nunits, nunits)
    assert np.allclose(denoiser, denoiser.T)

def test_edge_cases():
    # Test single trial
    data = np.random.randn(10, 20, 1)
    with pytest.raises(ValueError):
        gsn_denoise(data)

    # Test invalid V parameter
    data = np.random.randn(10, 20, 5)
    with pytest.raises(ValueError):
        gsn_denoise(data, V=5)

def test_run_gsn_full_pipeline():
    """
    Smoke test on a moderate dataset.
    """
    nunits, nconds, ntrials = 5, 10, 2
    data = np.random.randn(nunits, nconds, ntrials).astype(np.float32)
    denoiser, scores, threshold = gsn_denoise(data, V=0)
    assert denoiser.shape == (nunits, nunits)
    if isinstance(threshold, np.ndarray):
        assert len(threshold) <= nunits
    elif isinstance(threshold, (int, np.int_)):
        assert threshold <= nunits

def test_additional_case_1_random_data_v1():
    """
    Test gsn_denoise with random data and V=1 (GSN canonical transform).
    """
    nunits, nconds, ntrials = 12, 10, 4
    data = np.random.randn(nunits, nconds, ntrials)
    denoiser, scores, threshold = gsn_denoise(data, V=1)
    assert denoiser.shape == (nunits, nunits)

def test_additional_case_2_minimal_data_v2():
    """
    Test gsn_denoise with minimal shape and V=2 (conditional noise basis).
    """
    nunits, nconds, ntrials = 3, 3, 2
    data = np.random.randn(nunits, nconds, ntrials)
    denoiser, scores, threshold = gsn_denoise(data, V=2)
    assert denoiser.shape == (nunits, nunits)
    assert threshold is not None

def test_additional_case_3_large_ntrials():
    """
    Test gsn_denoise when ntrials is large, to ensure cross-validation logic still works.
    """
    nunits, nconds, ntrials = 5, 5, 30
    data = np.random.randn(nunits, nconds, ntrials)
    denoiser, scores, threshold = gsn_denoise(data, V=0)
    assert denoiser.shape == (nunits, nunits)

def test_additional_case_4_many_units_few_conditions():
    """
    Test gsn_denoise with many units but few conditions.
    """
    nunits, nconds, ntrials = 20, 2, 3
    data = np.random.randn(nunits, nconds, ntrials)
    denoiser, scores, threshold = gsn_denoise(data, V=3)
    assert denoiser.shape == (nunits, nunits)

def test_additional_case_5_mag_frac_zero():
    """
    Test perform_magnitude_thresholding with mag_frac=0.0 => everything should survive.
    """
    nunits, nconds, ntrials = 6, 5, 4
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)
    gsn_results = {'cSb': np.eye(nunits), 'cNb': np.eye(nunits)}
    opt = {
        'mag_type': 0,
        'mag_frac': 0.0,
        'mag_mode': 0,
        'cv_threshold_per': 'unit'  # Required parameter
    }
    denoiser, _, threshold = perform_magnitude_thresholding(data, basis, gsn_results, opt, V=0)
    # Because mag_frac=0 => threshold_val=0 => presumably all eigenvalues survive
    assert len(threshold) == nunits

def test_additional_case_6_mag_frac_unity():
    """
    Test perform_magnitude_thresholding with mag_frac=1.0 => only top eigenvalues survive.
    """
    nunits, nconds, ntrials = 8, 8, 3
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)
    gsn_results = {'cSb': np.eye(nunits), 'cNb': np.eye(nunits)}
    opt = {
        'mag_type': 0,
        'mag_frac': 1.0,
        'mag_mode': 1,
        'cv_threshold_per': 'unit'  # Required parameter
    }
    denoiser, _, threshold = perform_magnitude_thresholding(data, basis, gsn_results, opt, V=0)
    assert len(threshold) <= nunits

def test_additional_case_7_cv_thresholds_subset():
    """
    Test cross-validation with a restricted set of thresholds.
    """
    nunits, nconds, ntrials = 7, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)
    opt = {
        'cv_mode': 0,
        'cv_threshold_per': 'population',
        'cv_thresholds': [1, 3, 5],
        'cv_scoring_fn': lambda A, B: -np.mean((A - B)**2)
    }
    denoiser, cv_scores, threshold = perform_cross_validation(data, basis, opt)
    assert threshold in [1, 3, 5]
    assert denoiser.shape == (nunits, nunits)

def test_additional_case_8_non_identity_basis_cv():
    """
    Test cross-validation using a random orthonormal basis for data shape.
    """
    nunits, nconds, ntrials = 8, 6, 4
    data = np.random.randn(nunits, nconds, ntrials)
    Q, _ = np.linalg.qr(np.random.randn(nunits, nunits))
    opt = {
        'cv_mode': 0,
        'cv_threshold_per': 'unit',
        'cv_thresholds': [1, 2, 3, 4, 5, 6, 7, 8],
        'cv_scoring_fn': lambda A, B: -np.mean((A - B)**2, axis=0)
    }
    denoiser, scores, threshold = perform_cross_validation(data, Q, opt)
    assert denoiser.shape == (nunits, nunits)
    assert len(threshold) == nunits

def test_additional_case_9_single_cond_many_trials():
    """
    Test gsn_denoise with only 1 condition but multiple trials.
    Verify that it raises an assertion error since we need at least 2 conditions.
    """
    nunits, nconds, ntrials = 6, 1, 5
    data = np.random.randn(nunits, nconds, ntrials)
    
    with pytest.raises(AssertionError, match="Data must have at least 2 conditions to estimate covariance"):
        denoiser, scores, threshold = gsn_denoise(data, V=0)

def test_additional_case_10_excessive_cv_thresholds():
    """
    Provide more cv_thresholds than basis columns to test safe handling.
    """
    nunits, nconds, ntrials = 5, 5, 2
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)
    opt = {
        'cv_mode': 0,
        'cv_threshold_per': 'population',
        'cv_thresholds': np.arange(1, 15),  # extends beyond nunits dimension
        'cv_scoring_fn': lambda A, B: -np.mean((A - B)**2)
    }
    denoiser, _, thr = perform_cross_validation(data, basis, opt)
    assert denoiser.shape == (nunits, nunits)

def test_additional_case_11_denoisingtype_1():
    """
    Test gsn_denoise with denoisingtype=1 in opt (though not explicitly used in code).
    """
    nunits, nconds, ntrials = 8, 8, 3
    data = np.random.randn(nunits, nconds, ntrials)
    denoiser, scores, threshold = gsn_denoise(data, V=2, opt={'denoisingtype':1})
    assert denoiser.shape == (nunits, nunits)

def test_additional_case_12_check_symmetric_v3():
    """
    Test that the resulting denoiser is symmetric with V=3 (trial-averaged PCA).
    """
    nunits, nconds, ntrials = 10, 2, 4
    data = np.random.randn(nunits, nconds, ntrials)
    denoiser, _, _ = gsn_denoise(data, V=3)
    assert np.allclose(denoiser, denoiser.T)

def test_additional_case_13_custom_scoring_function():
    """
    Test a custom scoring_fn that returns an array with negative absolute differences.
    """
    nunits, nconds, ntrials = 6, 6, 3
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)
    opt = {
        'cv_mode': 0,
        'cv_thresholds': [1, 2, 3, 4, 5, 6],
        'cv_threshold_per': 'unit',
        'cv_scoring_fn': lambda A, B: -np.abs(A - B).mean(axis=0)
    }
    denoiser, scores, threshold = perform_cross_validation(data, basis, opt)
    assert denoiser.shape == (nunits, nunits)

def test_additional_case_14_user_basis_rank_deficient():
    """
    Provide a user-supplied basis that is not full rank. Should still run, but dimension < nunits.
    """
    nunits, nconds, ntrials = 8, 5, 2
    data = np.random.randn(nunits, nconds, ntrials)
    # Make a rank-deficient matrix: shape (nunits, 3)
    basis = np.random.randn(nunits, 3)
    basis[:, 1] = basis[:, 0]  # forcibly reduce rank
    denoiser, _, thr = gsn_denoise(data, V=basis)
    assert denoiser.shape == (nunits, nunits)

def test_additional_case_15_inverted_random_basis_v4():
    """
    Test that a random basis from V=4 doesn't raise shape errors when negative dimension present.
    """
    nunits, nconds, ntrials = 10, 6, 3
    data = np.random.randn(nunits, nconds, ntrials)
    # Just run it, ensure no shape mismatch
    denoiser, _, thr = gsn_denoise(data, V=4)
    assert denoiser.shape == (nunits, nunits)

def test_additional_case_16_magnitude_thresholding_none_survive():
    """
    Provide mag_frac that is guaranteed to exceed all eigenvalues => empty threshold result.
    """
    nunits, nconds, ntrials = 10, 6, 4
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)
    gsn_results = {'cSb': np.eye(nunits) * 1e-6, 'cNb': np.eye(nunits)}
    opt = {
        'mag_type': 0,
        'mag_frac': 1e6,   # extremely large fraction
        'mag_mode': 0,
        'cv_threshold_per': 'unit'  # Required parameter
    }
    denoiser, _, thr = perform_magnitude_thresholding(data, basis, gsn_results, opt, V=0)
    assert denoiser.shape == (nunits, nunits)
    # With extremely high mag_frac, we expect no dimensions to survive
    assert np.allclose(denoiser, np.zeros_like(denoiser))

def test_additional_case_17_skip_cv_if_no_folds():
    """
    Force a scenario where ntrials=2 but we skip 2 folds => effectively no data in cross-validation.
    """
    nunits, nconds, ntrials = 5, 5, 2
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)
    opt = {
        'cv_mode': 0,
        'cv_thresholds': [1, 2, 3, 4],
        'cv_threshold_per': 'unit'  # Required parameter
    }
    denoiser, cv_scores, thr = perform_cross_validation(data, basis, opt)
    assert denoiser.shape == (nunits, nunits)
    assert cv_scores.shape[0] == len(opt['cv_thresholds'])

def test_additional_case_18_custom_gsn_results_v1():
    """
    Manually pass custom gsn_results to magnitude thresholding with V=1 scenario.
    """
    nunits, nconds, ntrials = 6, 4, 3
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)
    gsn_results = {
        'cSb': 2.0 * np.eye(nunits),
        'cNb': 0.5 * np.eye(nunits)
    }
    opt = {
        'mag_type': 0,
        'mag_frac': 0.5,
        'mag_mode': 1,   # keep all that survive
        'cv_threshold_per': 'unit'  # Required parameter
    }
    denoiser, _, thr = perform_magnitude_thresholding(data, basis, gsn_results, opt, V=1)
    assert len(thr) >= 1

def test_additional_case_19_infinite_values_in_data():
    """
    Test that gsn_denoise either raises an error or handles data with np.inf gracefully.
    """
    nunits, nconds, ntrials = 6, 6, 3
    data = np.random.randn(nunits, nconds, ntrials)
    data[0, 0, 0] = np.inf  # introduce infinite value
    with pytest.raises(AssertionError):
        # Depending on your code, you might choose to raise or handle it. If you want no error,
        # you'd need a specific 'np.isinf' check. This test expects an assertion failure.
        gsn_denoise(data, V=0)

def test_additional_case_20_custom_cv_scoring_skip_empty():
    """
    Test a custom cross-validation scoring function that does ratio, ensure no warnings if test_data=0.
    """
    nunits, nconds, ntrials = 6, 6, 3
    data = np.random.randn(nunits, nconds, ntrials)
    basis = np.eye(nunits)
    opt = {
        'cv_mode': 0,
        'cv_thresholds': [1, 2, 3],
        'cv_threshold_per': 'population',
        'cv_scoring_fn': lambda A, B: -np.mean(np.nan_to_num((A - B)/(B + 1e-12))**2)
    }
    denoiser, cv_scores, thr = perform_cross_validation(data, basis, opt)
    assert denoiser.shape == (nunits, nunits)
    assert cv_scores.shape[0] == len(opt['cv_thresholds'])

def test_basic_denoising():
    """Test basic denoising functionality with default parameters."""
    # Base simulation parameters
    sim_opt = {
        'nvox': 50,
        'ntrial': 5,
        'ncond': 200,
        'signal_decay': 2,
        'noise_decay': 1.25,
        'align_k': 50,
        'align_alpha': 0.5,
        'noise_multi': 1
    }
    
    # Generate data
    train_data, test_data, ground_truth = generate_data(
        nvox=sim_opt['nvox'],
        ncond=sim_opt['ncond'],
        ntrial=sim_opt['ntrial'],
        signal_decay=sim_opt['signal_decay'],
        noise_decay=sim_opt['noise_decay'],
        align_alpha=sim_opt['align_alpha'],
        align_k=sim_opt['align_k'],
        noise_multiplier=sim_opt['noise_multi'],
        random_seed=42
    )
    
    # Apply GSN denoising with specific options
    opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': 0,  # Eigen-based thresholding
        'mag_frac': 0.5,  # Keep 50% of dimensions
        'mag_mode': 0,  # Contiguous mode
        'cv_threshold_per': 'unit'  # Required parameter
    }
    denoiser, scores, threshold = gsn_denoise(train_data, V=0, opt=opt)
    
    # Apply denoiser to get denoised data
    # First transpose data to (ncond, nvox, ntrial)
    train_data_reshaped = np.transpose(train_data, (1, 0, 2))
    # Project each condition's data onto denoiser
    denoised_data = np.zeros_like(train_data_reshaped)
    for trial in range(train_data.shape[2]):
        denoised_data[:, :, trial] = train_data_reshaped[:, :, trial] @ denoiser
    # Transpose back to original shape
    denoised_data = np.transpose(denoised_data, (1, 0, 2))  # Back to (nvox, ncond, ntrial)
    
    # Basic checks
    assert denoised_data.shape == train_data.shape
    assert not np.allclose(denoised_data, train_data)  # Should be different from input
    assert np.all(np.isfinite(denoised_data))  # No NaNs or infs
    
    # Check denoiser properties
    assert denoiser.shape == (sim_opt['nvox'], sim_opt['nvox'])
    assert np.all(np.isfinite(denoiser))

def test_noise_levels():
    """Test denoising performance with different noise levels."""
    noise_multipliers = [0.5, 1.0, 2.0, 4.0]
    base_opt = {
        'nvox': 50,
        'ntrial': 5,
        'ncond': 200,
        'signal_decay': 2,
        'noise_decay': 1.25,
        'align_k': 50,
        'align_alpha': 0.5
    }
    
    # Denoising options
    denoise_opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': 0,  # Eigen-based thresholding
        'mag_frac': 0.5,  # Keep 50% of dimensions
        'mag_mode': 0,  # Contiguous mode
        'cv_threshold_per': 'unit'  # Required parameter
    }
    
    improvements = []
    for noise_multi in noise_multipliers:
        # Generate data
        train_data, test_data, ground_truth = generate_data(
            nvox=base_opt['nvox'],
            ncond=base_opt['ncond'],
            ntrial=base_opt['ntrial'],
            signal_decay=base_opt['signal_decay'],
            noise_decay=base_opt['noise_decay'],
            align_alpha=base_opt['align_alpha'],
            align_k=base_opt['align_k'],
            noise_multiplier=noise_multi,
            random_seed=42
        )
        
        # Apply GSN denoising
        denoiser, scores, threshold = gsn_denoise(train_data, V=0, opt=denoise_opt)
        
        # Apply denoiser
        # First transpose data to (ncond, nvox, ntrial)
        train_data_reshaped = np.transpose(train_data, (1, 0, 2))
        # Project each condition's data onto denoiser
        denoised_data = np.zeros_like(train_data_reshaped)
        for trial in range(train_data.shape[2]):
            denoised_data[:, :, trial] = train_data_reshaped[:, :, trial] @ denoiser
        # Transpose back to original shape
        denoised_data = np.transpose(denoised_data, (1, 0, 2))  # Back to (nvox, ncond, ntrial)
        
        # Calculate improvement
        true_signal = ground_truth['signal']
        noisy_corr = np.corrcoef(true_signal.flatten(), 
                                np.mean(train_data, axis=2).T.flatten())[0, 1]
        denoised_corr = np.corrcoef(true_signal.flatten(), 
                                   np.mean(denoised_data, axis=2).T.flatten())[0, 1]
        improvement = denoised_corr - noisy_corr
        improvements.append(improvement)
    
    # Check that denoising helps more when noise is higher
    # For high noise levels (last two), denoising should help
    assert np.mean(improvements[-2:]) > 0, "Denoising should help with high noise"
    # For low noise levels, small degradation is acceptable
    assert np.all(np.array(improvements[:2]) > -0.15), "Denoising shouldn't hurt too much with low noise"

def test_gsn_to_denoise_basic():
    """Test using GSN outputs as inputs to gsn_denoise with basic parameters."""
    # Generate realistic data
    sim_opt = {
        'nvox': 50,
        'ntrial': 5,
        'ncond': 200,
        'signal_decay': 2,
        'noise_decay': 1.25,
        'align_k': 50,
        'align_alpha': 0.5,
        'noise_multi': 1
    }
    
    train_data, test_data, ground_truth = generate_data(
        nvox=sim_opt['nvox'],
        ncond=sim_opt['ncond'],
        ntrial=sim_opt['ntrial'],
        signal_decay=sim_opt['signal_decay'],
        noise_decay=sim_opt['noise_decay'],
        align_alpha=sim_opt['align_alpha'],
        align_k=sim_opt['align_k'],
        noise_multiplier=sim_opt['noise_multi'],
        random_seed=42
    )
    
    # Run GSN to get covariance matrices
    gsn_results = perform_gsn(train_data)
    assert 'cSb' in gsn_results
    assert 'cNb' in gsn_results
    
    # Use GSN results in denoising
    opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': 0,  # Eigen-based thresholding
        'mag_frac': 0.5,  # Keep 50% of dimensions
        'mag_mode': 0,  # Contiguous mode
        'cv_threshold_per': 'unit'  # Required parameter
    }
    denoiser, scores, threshold = gsn_denoise(train_data, V=0, opt=opt)
    
    # Basic checks
    assert denoiser.shape == (sim_opt['nvox'], sim_opt['nvox'])
    assert np.all(np.isfinite(denoiser))
    assert np.allclose(denoiser, denoiser.T)  # Should be symmetric

def test_gsn_to_denoise_high_noise():
    """Test using GSN outputs with high noise data."""
    sim_opt = {
        'nvox': 50,
        'ntrial': 5,
        'ncond': 200,
        'signal_decay': 2,
        'noise_decay': 1.25,
        'align_k': 50,
        'align_alpha': 0.5,
        'noise_multi': 4  # High noise
    }
    
    train_data, test_data, ground_truth = generate_data(
        nvox=sim_opt['nvox'],
        ncond=sim_opt['ncond'],
        ntrial=sim_opt['ntrial'],
        signal_decay=sim_opt['signal_decay'],
        noise_decay=sim_opt['noise_decay'],
        align_alpha=sim_opt['align_alpha'],
        align_k=sim_opt['align_k'],
        noise_multiplier=sim_opt['noise_multi'],
        random_seed=42
    )
    
    # Run GSN
    gsn_results = perform_gsn(train_data)
    
    # Use GSN results with different mag_frac values
    mag_fracs = [0.1, 0.3, 0.5, 0.7, 0.9]
    improvements = []
    
    for mag_frac in mag_fracs:
        opt = {
            'cv_mode': -1,
            'mag_type': 0,
            'mag_frac': mag_frac,
            'mag_mode': 0,
            'cv_threshold_per': 'unit'
        }
        denoiser, _, _ = gsn_denoise(train_data, V=0, opt=opt)
        # Project data into GSN basis and back
        # First transpose data to (ncond, nvox, ntrial)
        train_data_reshaped = np.transpose(train_data, (1, 0, 2))
        # Project each condition's data onto denoiser
        denoised_data = np.zeros_like(train_data_reshaped)
        for trial in range(train_data.shape[2]):
            denoised_data[:, :, trial] = train_data_reshaped[:, :, trial] @ denoiser
        # Transpose back to original shape
        denoised_data = np.transpose(denoised_data, (1, 0, 2))  # Back to (nvox, ncond, ntrial)
        
        # Calculate improvement
        true_signal = ground_truth['signal']
        noisy_corr = np.corrcoef(true_signal.flatten(), 
                                np.mean(train_data, axis=2).T.flatten())[0, 1]
        denoised_corr = np.corrcoef(true_signal.flatten(), 
                                   np.mean(denoised_data, axis=2).T.flatten())[0, 1]
        improvements.append(denoised_corr - noisy_corr)
    
    # With high noise, at least one denoising level should give substantial improvement
    assert max(improvements) > 0.1, "At least one denoising level should give substantial improvement"
    # All denoising levels should help somewhat
    assert np.all(np.array(improvements) > -0.1), "No denoising level should hurt too much"

def test_gsn_to_denoise_low_rank():
    """Test using GSN outputs with low-rank signal structure."""
    sim_opt = {
        'nvox': 100,  # More voxels
        'ntrial': 5,
        'ncond': 200,
        'signal_decay': 5,  # Faster decay -> more low-rank
        'noise_decay': 1.25,
        'align_k': 20,  # Fewer alignment dimensions
        'align_alpha': 0.5,
        'noise_multi': 1
    }
    
    train_data, test_data, ground_truth = generate_data(
        nvox=sim_opt['nvox'],
        ncond=sim_opt['ncond'],
        ntrial=sim_opt['ntrial'],
        signal_decay=sim_opt['signal_decay'],
        noise_decay=sim_opt['noise_decay'],
        align_alpha=sim_opt['align_alpha'],
        align_k=sim_opt['align_k'],
        noise_multiplier=sim_opt['noise_multi'],
        random_seed=42
    )
    
    # Run GSN
    gsn_results = perform_gsn(train_data)
    
    # Test both contiguous and non-contiguous modes
    modes = [0, 1]  # contiguous and non-contiguous
    for mode in modes:
        opt = {
            'cv_mode': -1,
            'mag_type': 0,
            'mag_frac': 0.2,  # Keep only top 20%
            'mag_mode': mode,
            'cv_threshold_per': 'unit'
        }
        denoiser, _, threshold = gsn_denoise(train_data, V=0, opt=opt)
        
        # Check that denoiser has appropriate rank
        eigenvals = np.linalg.eigvalsh(denoiser)
        significant_dims = np.sum(np.abs(eigenvals) > 1e-10)
        assert significant_dims < sim_opt['nvox'], "Denoiser should be low-rank"

def test_gsn_to_denoise_cross_validation():
    """Test using GSN outputs with cross-validation."""
    sim_opt = {
        'nvox': 50,
        'ntrial': 10,  # More trials for better cross-validation
        'ncond': 200,
        'signal_decay': 2,
        'noise_decay': 1.25,
        'align_k': 50,
        'align_alpha': 0.5,
        'noise_multi': 4  # Higher noise to encourage denoising
    }

    train_data, test_data, ground_truth = generate_data(
        nvox=sim_opt['nvox'],
        ncond=sim_opt['ncond'],
        ntrial=sim_opt['ntrial'],
        signal_decay=sim_opt['signal_decay'],
        noise_decay=sim_opt['noise_decay'],
        align_alpha=sim_opt['align_alpha'],
        align_k=sim_opt['align_k'],
        noise_multiplier=sim_opt['noise_multi'],
        random_seed=42
    )

    # Test both unit-wise and population-wise thresholding
    for threshold_per in ['unit', 'population']:
        opt = {
            'cv_mode': 0,  # Use cross-validation
            'cv_threshold_per': threshold_per,
            'cv_thresholds': np.arange(1, sim_opt['nvox'] // 2),  # Only test up to half the dimensions
            'cv_scoring_fn': lambda A, B: -np.mean((A - B)**2, axis=0)
        }
        denoiser, scores, threshold = gsn_denoise(train_data, V=0, opt=opt)

        # Apply to test data
        test_data_mean = np.mean(test_data, axis=2)  # Average across trials first (nvox, ncond)
        denoised_test = test_data_mean.T @ denoiser.T  # Shape: (ncond, nvox)

        # Check that denoising doesn't hurt performance too much
        true_signal = ground_truth['signal']  # Shape: (ncond, nvox)
        test_noisy_corr = np.corrcoef(true_signal.flatten(), test_data_mean.T.flatten())[0, 1]
        test_denoised_corr = np.corrcoef(true_signal.flatten(), denoised_test.flatten())[0, 1]
        assert test_denoised_corr > test_noisy_corr - 0.1, f"Cross-validation with {threshold_per} thresholding shouldn't hurt too much"
        
        # Check that cross-validation scores have meaningful variation
        score_variation = np.std(scores)
        assert score_variation > 1e-6, "Cross-validation scores should show meaningful variation"
        
        # Check that denoiser is not identity (since we're only using half the dimensions)
        identity = np.eye(sim_opt['nvox'])
        denoiser_diff = np.linalg.norm(denoiser - identity, ord='fro')
        assert denoiser_diff > 1e-6, f"{threshold_per} denoiser should not be identity when using half dimensions"

def test_gsn_to_denoise_different_bases():
    """Test using GSN outputs with different basis selection methods."""
    sim_opt = {
        'nvox': 50,
        'ntrial': 5,
        'ncond': 200,
        'signal_decay': 2,
        'noise_decay': 1.25,
        'align_k': 50,
        'align_alpha': 0.5,
        'noise_multi': 2
    }

    train_data, test_data, ground_truth = generate_data(
        nvox=sim_opt['nvox'],
        ncond=sim_opt['ncond'],
        ntrial=sim_opt['ntrial'],
        signal_decay=sim_opt['signal_decay'],
        noise_decay=sim_opt['noise_decay'],
        align_alpha=sim_opt['align_alpha'],
        align_k=sim_opt['align_k'],
        noise_multiplier=sim_opt['noise_multi'],
        random_seed=42
    )

    # Run GSN
    gsn_results = perform_gsn(train_data)

    # Test all basis selection methods
    basis_methods = [0, 1, 2, 3]  # Different V values
    improvements = []
    denoisers = []

    for V in basis_methods:
        opt = {
            'cv_mode': -1,
            'mag_type': 0,
            'mag_frac': 0.5,
            'mag_mode': 0,
            'cv_threshold_per': 'unit'
        }
        denoiser, _, _ = gsn_denoise(train_data, V=V, opt=opt)
        denoisers.append(denoiser)
        
        # Average across trials first
        train_data_mean = np.mean(train_data, axis=2)  # Shape: (nvox, ncond)
        denoised_data = train_data_mean.T @ denoiser.T  # Shape: (ncond, nvox)

        # Calculate improvement
        true_signal = ground_truth['signal']  # Shape: (ncond, nvox)
        noisy_corr = np.corrcoef(true_signal.flatten(), train_data_mean.T.flatten())[0, 1]
        denoised_corr = np.corrcoef(true_signal.flatten(), denoised_data.flatten())[0, 1]
        improvements.append(denoised_corr - noisy_corr)

    # At least one basis method should give some improvement
    assert max(improvements) > 0, "At least one basis method should give improvement"

    # Different basis methods should give different results
    denoiser_diffs = []
    for i in range(len(denoisers)):
        for j in range(i + 1, len(denoisers)):
            diff = np.linalg.norm(denoisers[i] - denoisers[j], ord='fro')
            denoiser_diffs.append(diff)
    assert min(denoiser_diffs) > 1e-6, "Different basis methods should give different results"

def test_gsn_vs_naive_pca_signal_recovery():
    """Test that using GSN's cSb basis achieves better signal recovery than naive PCA basis."""
    # Generate data with strong signal structure and moderate noise
    sim_opt = {
        'nvox': 100,  # More voxels for better differentiation
        'ntrial': 3,  # Fewer trials to make difference more apparent
        'ncond': 200,
        'signal_decay': 2.0,  # Faster decay -> clearer signal structure
        'noise_decay': 1.25,  # Faster decay -> clearer noise structure
        'align_k': 50,        # Fewer alignment dimensions
        'align_alpha': 0.5,   # Moderate alignment
        'noise_multi': 4.0    # Moderate noise to allow better signal recovery
    }

    train_data, test_data, ground_truth = generate_data(
        nvox=sim_opt['nvox'],
        ncond=sim_opt['ncond'],
        ntrial=sim_opt['ntrial'],
        signal_decay=sim_opt['signal_decay'],
        noise_decay=sim_opt['noise_decay'],
        align_alpha=sim_opt['align_alpha'],
        align_k=sim_opt['align_k'],
        noise_multiplier=sim_opt['noise_multi'],
        random_seed=42
    )

    # Get ground truth signal matrix
    true_signal = ground_truth['signal']  # Shape: (ncond, nvox)

    # Method 1: GSN denoising with cSb basis (V=0)
    # Run GSN to get covariance matrices
    gsn_results = perform_gsn(train_data)
    
    # Use GSN results with cSb basis (V=0)
    gsn_opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': 1,  # Signal variance thresholding
        'mag_frac': 0.5,  # Keep 50% of dimensions
        'mag_mode': 0,  # Contiguous mode
        'cv_threshold_per': 'unit'  # Required parameter
    }
    gsn_denoiser, _, _ = gsn_denoise(train_data, V=0, opt=gsn_opt)  # V=0 uses cSb eigenvectors
    
    # Project data into GSN basis and back
    # First transpose data to (ncond, nvox, ntrial)
    train_data_reshaped = np.transpose(train_data, (1, 0, 2))
    # Project each condition's data onto denoiser
    gsn_denoised = np.zeros_like(train_data_reshaped)
    for trial in range(train_data.shape[2]):
        gsn_denoised[:, :, trial] = train_data_reshaped[:, :, trial] @ gsn_denoiser
    # Transpose back to original shape
    gsn_denoised = np.transpose(gsn_denoised, (1, 0, 2))  # Back to (nvox, ncond, ntrial)
    gsn_recovered_signal = np.mean(gsn_denoised, axis=2).T

    # Method 2: GSN denoising with naive PCA basis
    # First compute PCA basis from trial-averaged data
    trial_avg = np.mean(train_data, axis=2)  # Shape: (nvox, ncond)
    trial_avg_reshaped = trial_avg.T  # Shape: (ncond, nvox)
    U, s, Vt = np.linalg.svd(trial_avg_reshaped, full_matrices=False)
    # Use V (right singular vectors) as the basis
    pca_basis = Vt.T  # Shape: (nvox, nvox)
    
    # Use the same GSN denoising framework but with PCA basis
    pca_denoiser, _, _ = gsn_denoise(train_data, V=pca_basis, opt=gsn_opt)
    
    # Project data into PCA basis and back
    # First transpose data to (ncond, nvox, ntrial)
    train_data_reshaped = np.transpose(train_data, (1, 0, 2))
    # Project each condition's data onto denoiser
    pca_denoised = np.zeros_like(train_data_reshaped)
    for trial in range(train_data.shape[2]):
        pca_denoised[:, :, trial] = train_data_reshaped[:, :, trial] @ pca_denoiser
    # Transpose back to original shape
    pca_denoised = np.transpose(pca_denoised, (1, 0, 2))  # Back to (nvox, ncond, ntrial)
    pca_recovered_signal = np.mean(pca_denoised, axis=2).T

    # Calculate correlation with ground truth for both methods
    gsn_corr = np.corrcoef(true_signal.flatten(), gsn_recovered_signal.flatten())[0, 1]
    pca_corr = np.corrcoef(true_signal.flatten(), pca_recovered_signal.flatten())[0, 1]

    print(f"\nGSN correlation: {gsn_corr:.3f}")
    print(f"PCA correlation: {pca_corr:.3f}")
    print(f"Improvement: {gsn_corr - pca_corr:.3f}")

    # GSN's cSb basis should achieve better correlation with ground truth
    assert gsn_corr > pca_corr, f"GSN's cSb basis should achieve better signal recovery than naive PCA basis (GSN: {gsn_corr:.3f}, PCA: {pca_corr:.3f})"
    assert gsn_corr > 0.5, f"GSN's signal recovery should be reasonably good (got {gsn_corr:.3f})"

def test_gsn_vs_naive_pca_robustness():
    """Test that GSN's cSb basis advantage over naive PCA basis is robust across different noise levels."""
    base_opt = {
        'nvox': 100,  # More voxels for better differentiation
        'ntrial': 3,  # Fewer trials to make difference more apparent
        'ncond': 200,
        'signal_decay': 2.0,  # Faster decay -> clearer signal structure
        'noise_decay': 1.25,  # Faster decay -> clearer noise structure
        'align_k': 50,        # Fewer alignment dimensions
        'align_alpha': 0.5,   # Moderate alignment
    }

    noise_levels = [2.0, 4.0, 8.0]  # Test different noise levels
    gsn_improvements = []

    # Explicit denoising parameters
    denoise_opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': 1,  # Signal variance thresholding
        'mag_frac':0.5,  # Keep 50% of dimensions
        'mag_mode':0,  # Contiguous mode
        'cv_threshold_per': 'unit'  # Required parameter
    }

    for noise_multi in noise_levels:
        # Generate data for this noise level
        train_data, test_data, ground_truth = generate_data(
            nvox=base_opt['nvox'],
            ncond=base_opt['ncond'],
            ntrial=base_opt['ntrial'],
            signal_decay=base_opt['signal_decay'],
            noise_decay=base_opt['noise_decay'],
            align_alpha=base_opt['align_alpha'],
            align_k=base_opt['align_k'],
            noise_multiplier=noise_multi,
            random_seed=42
        )

        true_signal = ground_truth['signal']

        # GSN denoising with cSb basis (V=0)
        gsn_results = perform_gsn(train_data)
        gsn_denoiser, _, _ = gsn_denoise(train_data, V=0, opt=denoise_opt)
        
        # Average across trials first
        train_data_mean = np.mean(train_data, axis=2)  # Shape: (nvox, ncond)
        gsn_denoised = train_data_mean.T @ gsn_denoiser.T  # Shape: (ncond, nvox)

        # Naive PCA denoising
        trial_avg = np.mean(train_data, axis=2)  # Shape: (nvox, ncond)
        trial_avg_reshaped = trial_avg.T  # Shape: (ncond, nvox)
        U, s, Vt = np.linalg.svd(trial_avg_reshaped, full_matrices=False)
        # Use V (right singular vectors) as the basis
        pca_basis = Vt.T  # Shape: (nvox, nvox)
        pca_denoiser, _, _ = gsn_denoise(train_data, V=pca_basis, opt=denoise_opt)
        
        # Apply PCA denoiser
        pca_denoised = train_data_mean.T @ pca_denoiser.T  # Shape: (ncond, nvox)

        # Calculate correlations
        gsn_corr = np.corrcoef(true_signal.flatten(), gsn_denoised.flatten())[0, 1]
        pca_corr = np.corrcoef(true_signal.flatten(), pca_denoised.flatten())[0, 1]
        improvement = gsn_corr - pca_corr
        gsn_improvements.append(improvement)

        print(f"\nNoise level: {noise_multi}")
        print(f"GSN correlation: {gsn_corr:.3f}")
        print(f"PCA correlation: {pca_corr:.3f}")
        print(f"Improvement: {improvement:.3f}")

    # GSN's advantage should increase with noise level
    assert all(imp > -0.05 for imp in gsn_improvements), f"GSN should not be much worse than PCA at any noise level (improvements: {gsn_improvements})"
    assert gsn_improvements[-1] > gsn_improvements[0], f"GSN's advantage should be larger at higher noise (improvements: {gsn_improvements})"

def test_gsn_vs_pca_default_params_vary_trials():
    """Test GSN vs PCA with default parameters while varying number of trials."""
    base_opt = {
        'nvox': 100,
        'ncond': 200,
        'signal_decay': 2.5,  # Faster decay for clearer signal structure
        'noise_decay': 1.25,
        'align_k': 50,
        'align_alpha': 0.5,
        'noise_multi': 4.0  # Higher noise to make denoising more important
    }

    trial_counts = [3, 5, 10, 20]  # Test different numbers of trials
    gsn_improvements = []

    # Explicit denoising parameters
    denoise_opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': 1,  # Signal variance thresholding
        'mag_frac': 0.3,  # Keep fewer dimensions to focus on strongest components
        'mag_mode': 0,  # Contiguous mode
        'cv_threshold_per': 'unit'  # Required parameter
    }

    for ntrial in trial_counts:
        # Generate data with this number of trials
        train_data, test_data, ground_truth = generate_data(
            nvox=base_opt['nvox'],
            ncond=base_opt['ncond'],
            ntrial=ntrial,
            signal_decay=base_opt['signal_decay'],
            noise_decay=base_opt['noise_decay'],
            align_alpha=base_opt['align_alpha'],
            align_k=base_opt['align_k'],
            noise_multiplier=base_opt['noise_multi'],
            random_seed=42
        )

        true_signal = ground_truth['signal']

        # GSN denoising with cSb basis (V=0)
        gsn_results = perform_gsn(train_data)
        gsn_denoiser, _, _ = gsn_denoise(train_data, V=0, opt=denoise_opt)
        
        # Project test data into GSN basis and back
        # First transpose data to (ncond, nvox, ntrial)
        test_data_reshaped = np.transpose(test_data, (1, 0, 2))
        # Project each condition's data onto denoiser
        gsn_denoised = np.zeros_like(test_data_reshaped)
        for trial in range(test_data.shape[2]):
            gsn_denoised[:, :, trial] = test_data_reshaped[:, :, trial] @ gsn_denoiser
        # Transpose back to original shape
        gsn_denoised = np.transpose(gsn_denoised, (1, 0, 2))  # Back to (nvox, ncond, ntrial)
        gsn_recovered = np.mean(gsn_denoised, axis=2).T

        # Naive PCA denoising
        trial_avg = np.mean(train_data, axis=2)  # Shape: (nvox, ncond)
        trial_avg_reshaped = trial_avg.T  # Shape: (ncond, nvox)
        U, s, Vt = np.linalg.svd(trial_avg_reshaped, full_matrices=False)
        # Use V (right singular vectors) as the basis
        pca_basis = Vt.T  # Shape: (nvox, nvox)
        pca_denoiser, _, _ = gsn_denoise(train_data, V=pca_basis, opt=denoise_opt)
        
        # Project test data into PCA basis and back
        # First transpose data to (ncond, nvox, ntrial)
        test_data_reshaped = np.transpose(test_data, (1, 0, 2))
        # Project each condition's data onto denoiser
        pca_denoised = np.zeros_like(test_data_reshaped)
        for trial in range(test_data.shape[2]):
            pca_denoised[:, :, trial] = test_data_reshaped[:, :, trial] @ pca_denoiser
        # Transpose back to original shape
        pca_denoised = np.transpose(pca_denoised, (1, 0, 2))  # Back to (nvox, ncond, ntrial)
        pca_recovered = np.mean(pca_denoised, axis=2).T

        # Calculate correlations
        gsn_corr = np.corrcoef(true_signal.flatten(), gsn_recovered.flatten())[0, 1]
        pca_corr = np.corrcoef(true_signal.flatten(), pca_recovered.flatten())[0, 1]
        gsn_improvements.append(gsn_corr - pca_corr)

        # Print detailed results for debugging
        print(f"\nTrial count: {ntrial}")
        print(f"GSN correlation: {gsn_corr:.3f}")
        print(f"PCA correlation: {pca_corr:.3f}")
        print(f"Improvement: {gsn_improvements[-1]:.3f}")

    # GSN should maintain advantage across different trial counts
    assert all(imp > -0.05 for imp in gsn_improvements), f"GSN should not be much worse than PCA (improvements: {gsn_improvements})"
    assert any(imp > 0.05 for imp in gsn_improvements), f"GSN should show substantial improvement in some cases (improvements: {gsn_improvements})"

def test_gsn_vs_pca_default_params_vary_conditions():
    """Test GSN vs PCA with default parameters while varying number of conditions."""
    base_opt = {
        'nvox': 100,
        'ntrial': 3,  # Fewer trials to make differences more apparent
        'signal_decay': 2.5,  # Faster decay for clearer signal structure
        'noise_decay': 1.25,
        'align_k': 50,
        'align_alpha': 0.5,
        'noise_multi': 8.0  # Higher noise to make denoising more important
    }

    condition_counts = [50, 100, 200, 400]  # Test different numbers of conditions
    gsn_improvements = []

    # Explicit denoising parameters
    denoise_opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': 1,  # Signal variance thresholding
        'mag_frac': 0.2,  # Keep fewer dimensions to focus on strongest components
        'mag_mode': 0,  # Contiguous mode
        'cv_threshold_per': 'unit'  # Required parameter
    }

    for ncond in condition_counts:
        # Generate data with this number of conditions
        train_data, test_data, ground_truth = generate_data(
            nvox=base_opt['nvox'],
            ncond=ncond,
            ntrial=base_opt['ntrial'],
            signal_decay=base_opt['signal_decay'],
            noise_decay=base_opt['noise_decay'],
            align_alpha=base_opt['align_alpha'],
            align_k=base_opt['align_k'],
            noise_multiplier=base_opt['noise_multi'],
            random_seed=42
        )

        true_signal = ground_truth['signal']

        # GSN denoising with cSb basis (V=0)
        gsn_results = perform_gsn(train_data)
        gsn_denoiser, _, _ = gsn_denoise(train_data, V=0, opt=denoise_opt)

        # Average across trials first
        test_data_mean = np.mean(test_data, axis=2)  # Shape: (nvox, ncond)
        gsn_denoised = test_data_mean.T @ gsn_denoiser.T  # Shape: (ncond, nvox)

        # Naive PCA denoising
        trial_avg = np.mean(train_data, axis=2)  # Shape: (nvox, ncond)
        trial_avg_reshaped = trial_avg.T  # Shape: (ncond, nvox)
        U, s, Vt = np.linalg.svd(trial_avg_reshaped, full_matrices=False)
        # Use V (right singular vectors) as the basis
        pca_basis = Vt.T  # Shape: (nvox, nvox)
        pca_denoiser, _, _ = gsn_denoise(train_data, V=pca_basis, opt=denoise_opt)

        # Apply PCA denoiser
        pca_denoised = test_data_mean.T @ pca_denoiser.T  # Shape: (ncond, nvox)

        # Calculate correlations
        gsn_corr = np.corrcoef(true_signal.flatten(), gsn_denoised.flatten())[0, 1]
        pca_corr = np.corrcoef(true_signal.flatten(), pca_denoised.flatten())[0, 1]
        gsn_improvements.append(gsn_corr - pca_corr)

        # Print detailed results for debugging
        print(f"\nCondition count: {ncond}")
        print(f"GSN correlation: {gsn_corr:.3f}")
        print(f"PCA correlation: {pca_corr:.3f}")
        print(f"Improvement: {gsn_improvements[-1]:.3f}")

    # GSN should maintain advantage across different condition counts
    assert all(imp > -0.05 for imp in gsn_improvements), f"GSN should not be much worse than PCA (improvements: {gsn_improvements})"
    assert any(imp > 0.03 for imp in gsn_improvements), f"GSN should show substantial improvement in some cases (improvements: {gsn_improvements})"

def test_gsn_vs_pca_default_params_vary_alignment():
    """Test GSN vs PCA with default parameters while varying alignment parameters."""
    base_opt = {
        'nvox': 100,
        'ntrial': 3,  # Fewer trials to make differences more apparent
        'ncond': 200,
        'signal_decay': 2.5,  # Faster decay -> clearer signal structure
        'noise_decay': 1.25,
        'align_k': 50,
        'align_alpha': 0.5,
        'noise_multi': 8.0  # Higher noise to make denoising more important
    }

    # Test different alignment parameters
    alignment_params = [
        {'k': 20, 'alpha': 0.1},   # Very weak alignment, few dimensions
        {'k': 20, 'alpha': 0.9},   # Very strong alignment, few dimensions
        {'k': 80, 'alpha': 0.1},   # Very weak alignment, many dimensions
        {'k': 80, 'alpha': 0.9},   # Very strong alignment, many dimensions
    ]
    gsn_improvements = []

    # Explicit denoising parameters
    denoise_opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': 1,  # Signal variance thresholding
        'mag_frac': 0.2,  # Keep fewer dimensions to focus on strongest components
        'mag_mode': 0,  # Contiguous mode
        'cv_threshold_per': 'unit'  # Required parameter
    }

    for align_params in alignment_params:
        # Generate data with these alignment parameters
        train_data, test_data, ground_truth = generate_data(
            nvox=base_opt['nvox'],
            ncond=base_opt['ncond'],
            ntrial=base_opt['ntrial'],
            signal_decay=base_opt['signal_decay'],
            noise_decay=base_opt['noise_decay'],
            align_alpha=align_params['alpha'],
            align_k=align_params['k'],
            noise_multiplier=base_opt['noise_multi'],
            random_seed=42
        )

        true_signal = ground_truth['signal']

        # GSN denoising with cSb basis (V=0)
        gsn_results = perform_gsn(train_data)
        gsn_denoiser, _, _ = gsn_denoise(train_data, V=0, opt=denoise_opt)

        # Average across trials first
        test_data_mean = np.mean(test_data, axis=2)  # Shape: (nvox, ncond)
        gsn_denoised = test_data_mean.T @ gsn_denoiser.T  # Shape: (ncond, nvox)

        # Naive PCA denoising
        trial_avg = np.mean(train_data, axis=2)  # Shape: (nvox, ncond)
        trial_avg_reshaped = trial_avg.T  # Shape: (ncond, nvox)
        U, s, Vt = np.linalg.svd(trial_avg_reshaped, full_matrices=False)
        # Use V (right singular vectors) as the basis
        pca_basis = Vt.T  # Shape: (nvox, nvox)
        pca_denoiser, _, _ = gsn_denoise(train_data, V=pca_basis, opt=denoise_opt)

        # Apply PCA denoiser
        pca_denoised = test_data_mean.T @ pca_denoiser.T  # Shape: (ncond, nvox)

        # Calculate correlations
        gsn_corr = np.corrcoef(true_signal.flatten(), gsn_denoised.flatten())[0, 1]
        pca_corr = np.corrcoef(true_signal.flatten(), pca_denoised.flatten())[0, 1]
        gsn_improvements.append(gsn_corr - pca_corr)

        # Print detailed results for debugging
        print(f"\nAlignment params: k={align_params['k']}, alpha={align_params['alpha']}")
        print(f"GSN correlation: {gsn_corr:.3f}")
        print(f"PCA correlation: {pca_corr:.3f}")
        print(f"Improvement: {gsn_improvements[-1]:.3f}")

    # GSN should show advantages in at least some cases
    assert any(imp > 0 for imp in gsn_improvements), f"GSN should show improvement in at least some cases (improvements: {gsn_improvements})"

    # GSN's advantage should be larger with weaker alignment (lower alpha)
    weak_align_avg = np.mean([gsn_improvements[0], gsn_improvements[2]])  # alpha = 0.1 cases
    strong_align_avg = np.mean([gsn_improvements[1], gsn_improvements[3]])  # alpha = 0.9 cases
    assert weak_align_avg > strong_align_avg, "GSN's advantage should be larger with weaker alignment"

def test_gsn_vs_pca_default_params_edge_cases():
    """Test GSN vs PCA with default parameters in edge cases."""
    base_opt = {
        'nvox': 100,
        'signal_decay': 2.5,  # Faster decay for clearer signal structure
        'noise_decay': 1.25,
        'noise_multi': 8.0,  # Higher noise to make denoising more important
        'align_k': 50,
        'align_alpha': 0.5
    }

    # Test edge cases
    edge_cases = [
        {'ntrial': 2, 'ncond': 200, 'desc': "minimum trials"},
        {'ntrial': 5, 'ncond': 50, 'desc': "few conditions"},
        {'ntrial': 5, 'ncond': 500, 'desc': "many conditions"},
        {'ntrial': 20, 'ncond': 200, 'desc': "many trials"}
    ]

    # Explicit denoising parameters
    denoise_opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': 1,  # Signal variance thresholding
        'mag_frac': 0.2,  # Keep fewer dimensions to focus on strongest components
        'mag_mode': 0,  # Contiguous mode
        'cv_threshold_per': 'unit'  # Required parameter
    }

    gsn_improvements = []

    for case in edge_cases:
        # Generate data for this edge case
        train_data, test_data, ground_truth = generate_data(
            nvox=base_opt['nvox'],
            ncond=case['ncond'],
            ntrial=case['ntrial'],
            signal_decay=base_opt['signal_decay'],
            noise_decay=base_opt['noise_decay'],
            align_alpha=base_opt['align_alpha'],
            align_k=base_opt['align_k'],
            noise_multiplier=base_opt['noise_multi'],
            random_seed=42
        )

        true_signal = ground_truth['signal']

        # GSN denoising with cSb basis (V=0)
        gsn_results = perform_gsn(train_data)
        gsn_denoiser, _, _ = gsn_denoise(train_data, V=0, opt=denoise_opt)

        # Average across trials first
        test_data_mean = np.mean(test_data, axis=2)  # Shape: (nvox, ncond)
        gsn_denoised = test_data_mean.T @ gsn_denoiser.T  # Shape: (ncond, nvox)

        # Naive PCA denoising
        trial_avg = np.mean(train_data, axis=2)  # Shape: (nvox, ncond)
        trial_avg_reshaped = trial_avg.T  # Shape: (ncond, nvox)
        U, s, Vt = np.linalg.svd(trial_avg_reshaped, full_matrices=False)
        # Use V (right singular vectors) as the basis
        pca_basis = Vt.T  # Shape: (nvox, nvox)
        pca_denoiser, _, _ = gsn_denoise(train_data, V=pca_basis, opt=denoise_opt)

        # Apply PCA denoiser
        pca_denoised = test_data_mean.T @ pca_denoiser.T  # Shape: (ncond, nvox)

        # Calculate correlations
        gsn_corr = np.corrcoef(true_signal.flatten(), gsn_denoised.flatten())[0, 1]
        pca_corr = np.corrcoef(true_signal.flatten(), pca_denoised.flatten())[0, 1]
        gsn_improvements.append(gsn_corr - pca_corr)

        # Print detailed results for debugging
        print(f"\nEdge case: {case['desc']}")
        print(f"GSN correlation: {gsn_corr:.3f}")
        print(f"PCA correlation: {pca_corr:.3f}")
        print(f"Improvement: {gsn_improvements[-1]:.3f}")

    # GSN should not be much worse than PCA in any edge case
    assert all(imp > -0.05 for imp in gsn_improvements), f"GSN should not be much worse than PCA in edge cases (improvements: {gsn_improvements})"
    # GSN should show substantial improvement in at least one edge case
    assert any(imp > 0.03 for imp in gsn_improvements), f"GSN should show substantial improvement in some edge cases (improvements: {gsn_improvements})"

