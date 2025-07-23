"""Test that unit mean handling is consistent across all GSN basis types.

This test verifies that unit means are computed, removed, and restored identically
regardless of which basis (V=0,1,2,3,4 or custom) is used for denoising.

The test covers:
1. Unit mean computation consistency across all basis types
2. Mean restoration for trial-averaged denoising
3. Mean restoration for single-trial denoising  
4. Mean handling with magnitude thresholding
5. Mean handling across different cross-validation modes
6. Dimensional reduction mean consistency
7. Proper handling of zero-mean data

This ensures that the denoising preserves the original unit means regardless of
the chosen basis, which is critical for maintaining the interpretability of
the neural responses.
"""

import sys, os
sys.path.insert(0, os.path.abspath("/Users/jacobprince/KonkLab Dropbox/Jacob Prince/Research-Prince/GSNdenoise-updated/GSN"))

import numpy as np
import pytest
from gsn.gsn_denoise import gsn_denoise

def create_test_data_with_strong_means():
    """Create test data with strong unit means to make mean handling effects visible."""
    np.random.seed(42)  # For reproducibility
    
    nunits = 12
    nconds = 15
    ntrials = 4
    
    # Create data with strong unit-specific means
    unit_means = np.random.randn(nunits) * 5  # Strong unit means
    
    # Base signal data (zero-mean)
    signal = np.random.randn(nunits, nconds) * 2
    noise = np.random.randn(nunits, nconds, ntrials) * 0.5
    
    # Add unit means and construct final data
    data = np.zeros((nunits, nconds, ntrials))
    for t in range(ntrials):
        data[:, :, t] = unit_means[:, np.newaxis] + signal + noise[:, :, t]
    
    return data, unit_means

def test_unit_mean_computation_consistency():
    """Test that unit means are computed identically across all basis types."""
    data, true_unit_means = create_test_data_with_strong_means()
    
    # Test all basis types
    basis_types = [0, 1, 2, 3, 4]
    
    # Store unit means from each basis type
    computed_unit_means = {}
    
    for V in basis_types:
        results = gsn_denoise(data, V=V, wantfig=False)
        computed_unit_means[V] = results['unit_means'].copy()
    
    # Test with custom orthonormal basis
    nunits = data.shape[0]
    custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]
    results_custom = gsn_denoise(data, V=custom_basis, wantfig=False)
    computed_unit_means['custom'] = results_custom['unit_means'].copy()
    
    # Verify all unit means are identical
    reference_means = computed_unit_means[0]
    for V in [1, 2, 3, 4, 'custom']:
        np.testing.assert_array_almost_equal(
            computed_unit_means[V], 
            reference_means,
            decimal=12,
            err_msg=f"Unit means differ between V=0 and V={V}"
        )
    
    # Verify unit means match expected computation
    expected_means = np.mean(np.mean(data, axis=2), axis=1)
    np.testing.assert_array_almost_equal(
        reference_means,
        expected_means,
        decimal=12,
        err_msg="Computed unit means don't match expected calculation"
    )

def test_mean_restoration_consistency():
    """Test that final denoised data has correct means across all basis types."""
    data, true_unit_means = create_test_data_with_strong_means()
    
    basis_types = [0, 1, 2, 3, 4]
    denoised_means = {}
    
    # Test trial-averaged denoising
    for V in basis_types:
        opt = {'denoisingtype': 0}  # Trial-averaged
        results = gsn_denoise(data, V=V, opt=opt, wantfig=False)
        
        # Compute means of denoised data
        denoised_means[V] = np.mean(results['denoiseddata'], axis=1)
    
    # Test with custom basis
    nunits = data.shape[0]
    custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]
    opt = {'denoisingtype': 0}
    results_custom = gsn_denoise(data, V=custom_basis, opt=opt, wantfig=False)
    denoised_means['custom'] = np.mean(results_custom['denoiseddata'], axis=1)
    
    # Check that all denoised data have same unit means as original data
    original_means = np.mean(np.mean(data, axis=2), axis=1)
    
    for V in [0, 1, 2, 3, 4, 'custom']:
        np.testing.assert_array_almost_equal(
            denoised_means[V],
            original_means,
            decimal=10,
            err_msg=f"Denoised data means don't match original for V={V}"
        )

def test_single_trial_mean_consistency():
    """Test mean handling consistency for single-trial denoising."""
    data, true_unit_means = create_test_data_with_strong_means()
    
    basis_types = [0, 1, 2, 3, 4]
    single_trial_means = {}
    
    # Test single-trial denoising
    for V in basis_types:
        opt = {'denoisingtype': 1}  # Single-trial
        results = gsn_denoise(data, V=V, opt=opt, wantfig=False)
        
        # Compute means across conditions and trials for each unit
        single_trial_means[V] = np.mean(results['denoiseddata'], axis=(1, 2))
    
    # Test with custom basis
    nunits = data.shape[0]
    custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]
    opt = {'denoisingtype': 1}
    results_custom = gsn_denoise(data, V=custom_basis, opt=opt, wantfig=False)
    single_trial_means['custom'] = np.mean(results_custom['denoiseddata'], axis=(1, 2))
    
    # Check that all denoised data preserve unit means
    original_means = np.mean(data, axis=(1, 2))
    
    for V in [0, 1, 2, 3, 4, 'custom']:
        np.testing.assert_array_almost_equal(
            single_trial_means[V],
            original_means,
            decimal=10,
            err_msg=f"Single-trial denoised means don't match original for V={V}"
        )

def test_magnitude_thresholding_mean_consistency():
    """Test mean handling with magnitude thresholding instead of cross-validation."""
    data, true_unit_means = create_test_data_with_strong_means()
    
    basis_types = [0, 1, 2, 3]  # V=4 (random) doesn't work well with magnitude thresholding
    mag_denoised_means = {}
    
    for V in basis_types:
        opt = {
            'cv_mode': -1,  # Magnitude thresholding
            'mag_frac': 0.1,
            'mag_selection_mode': 0,
            'denoisingtype': 0
        }
        results = gsn_denoise(data, V=V, opt=opt, wantfig=False)
        mag_denoised_means[V] = np.mean(results['denoiseddata'], axis=1)
    
    # Check consistency across basis types
    original_means = np.mean(np.mean(data, axis=2), axis=1)
    
    for V in basis_types:
        np.testing.assert_array_almost_equal(
            mag_denoised_means[V],
            original_means,
            decimal=10,
            err_msg=f"Magnitude thresholding means don't match original for V={V}"
        )

def test_cross_validation_modes_mean_consistency():
    """Test mean handling across different cross-validation modes."""
    data, true_unit_means = create_test_data_with_strong_means()
    
    # Test both CV modes with both population and unit thresholding
    cv_configs = [
        {'cv_mode': 0, 'cv_threshold_per': 'population'},
        {'cv_mode': 0, 'cv_threshold_per': 'unit'},
        {'cv_mode': 1, 'cv_threshold_per': 'population'},
        {'cv_mode': 1, 'cv_threshold_per': 'unit'}
    ]
    
    original_means = np.mean(np.mean(data, axis=2), axis=1)
    
    for config in cv_configs:
        # Test with V=0 (signal covariance basis)
        results = gsn_denoise(data, V=0, opt=config, wantfig=False)
        
        if config['denoisingtype'] == 1 if 'denoisingtype' in config else False:
            # Single trial case
            denoised_means = np.mean(results['denoiseddata'], axis=(1, 2))
        else:
            # Trial-averaged case
            denoised_means = np.mean(results['denoiseddata'], axis=1)
        
        np.testing.assert_array_almost_equal(
            denoised_means,
            original_means,
            decimal=10,
            err_msg=f"CV config {config} doesn't preserve means"
        )

def test_dimreduce_mean_consistency():
    """Test that dimensionally reduced data also handles means correctly."""
    data, true_unit_means = create_test_data_with_strong_means()
    
    # Use population thresholding to get dimreduce output
    opt = {'cv_threshold_per': 'population', 'denoisingtype': 0}
    results = gsn_denoise(data, V=0, opt=opt, wantfig=False)
    
    # dimreduce should be demeaned data projected into signal subspace
    # Since we demean before projection, the mean of dimreduce should be close to zero
    # (unless there's structure in the demeaned data that aligns with the signal subspace)
    actual_projected_means = np.mean(results['dimreduce'], axis=1)
    
    # The main test is that dimreduce is computed consistently
    # Let's verify by manually computing what dimreduce should be
    signalsubspace = results['signalsubspace']
    trial_avg = np.mean(data, axis=2)
    trial_avg_demeaned = trial_avg - results['unit_means'][:, np.newaxis]
    expected_dimreduce = signalsubspace.T @ trial_avg_demeaned
    
    np.testing.assert_array_almost_equal(
        results['dimreduce'],
        expected_dimreduce,
        decimal=10,
        err_msg="Dimensionally reduced data computation is inconsistent"
    )

def test_zero_mean_data_handling():
    """Test that zero-mean data is handled correctly (no artificial means added)."""
    np.random.seed(123)
    
    nunits, nconds, ntrials = 8, 10, 3
    
    # Create truly zero-mean data
    data = np.random.randn(nunits, nconds, ntrials)
    data = data - np.mean(data, axis=(1, 2), keepdims=True)  # Remove any accidental means
    
    # Verify data is zero-mean
    assert np.allclose(np.mean(data, axis=(1, 2)), 0, atol=1e-12)
    
    # Test with different basis types
    for V in [0, 1, 2, 3, 4]:
        results = gsn_denoise(data, V=V, wantfig=False)
        
        # Unit means should be essentially zero
        np.testing.assert_array_almost_equal(
            results['unit_means'],
            np.zeros(nunits),
            decimal=12,
            err_msg=f"Zero-mean data shows non-zero unit means for V={V}"
        )
        
        # Denoised data should also be essentially zero-mean
        denoised_means = np.mean(results['denoiseddata'], axis=1)
        np.testing.assert_array_almost_equal(
            denoised_means,
            np.zeros(nunits),
            decimal=10,
            err_msg=f"Zero-mean data produces non-zero denoised means for V={V}"
        )

if __name__ == "__main__":
    # Run all tests
    test_unit_mean_computation_consistency()
    test_mean_restoration_consistency()
    test_single_trial_mean_consistency()
    test_magnitude_thresholding_mean_consistency()
    test_cross_validation_modes_mean_consistency()
    test_dimreduce_mean_consistency()
    test_zero_mean_data_handling()
    
    print("All unit mean consistency tests passed!")
