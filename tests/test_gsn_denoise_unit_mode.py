import numpy as np
import pytest
from gsn.gsn_denoise import gsn_denoise

def test_unit_mode_basic():
    """Test that unit-wise mode produces different thresholds for different units."""
    # Create synthetic data where different units have different optimal thresholds
    nunits, nconds, ntrials = 3, 10, 5
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Make first unit have strong signal in first component
    data[0, :, :] = 5.0 * np.random.randn(nconds, ntrials) + np.random.randn(nconds, ntrials) * 0.1
    
    # Make second unit have strong signal in first two components
    data[1, :, :] = 2.5 * np.random.randn(nconds, ntrials) + 2.5 * np.random.randn(nconds, ntrials) + np.random.randn(nconds, ntrials) * 0.1
    
    # Make third unit have weak signal
    data[2, :, :] = np.random.randn(nconds, ntrials) * 0.1

    # Run GSN with unit-wise thresholding
    opt = {'cv_threshold_per': 'unit', 'cv_thresholds': [1, 2, 3]}
    result = gsn_denoise(data, V=0, opt=opt)
    
    # Check that we got different thresholds for different units
    assert len(np.unique(result['best_threshold'])) > 1, "Expected different thresholds for different units"

def test_unit_mode_denoiser_structure():
    """Test that the denoiser matrix has the expected structure in unit-wise mode."""
    nunits, nconds, ntrials = 3, 10, 5
    data = np.random.randn(nunits, nconds, ntrials)
    
    opt = {'cv_threshold_per': 'unit', 'cv_thresholds': [1, 2]}
    result = gsn_denoise(data, V=0, opt=opt)
    
    # Check denoiser dimensions
    assert result['denoiser'].shape == (nunits, nunits), "Denoiser should be nunits x nunits"

def test_unit_mode_vs_population():
    """Test that unit-wise mode gives different results than population mode."""
    nunits, nconds, ntrials = 3, 10, 5
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Run with unit-wise mode
    opt_unit = {'cv_threshold_per': 'unit', 'cv_thresholds': [1, 2]}
    result_unit = gsn_denoise(data, V=0, opt=opt_unit)
    
    # Run with population mode
    opt_pop = {'cv_threshold_per': 'population', 'cv_thresholds': [1, 2]}
    result_pop = gsn_denoise(data, V=0, opt=opt_pop)
    
    # The denoiser matrices should be different
    assert not np.allclose(result_unit['denoiser'], result_pop['denoiser']), \
        "Unit-wise and population denoisers should be different"

def test_unit_mode_denoised_data():
    """Test that denoised data has expected properties in unit-wise mode."""
    nunits, nconds, ntrials = 3, 10, 5
    data = np.random.randn(nunits, nconds, ntrials)
    
    opt = {'cv_threshold_per': 'unit', 'cv_thresholds': [1, 2]}
    result = gsn_denoise(data, V=0, opt=opt)
    
    # Check denoised data dimensions
    assert result['denoiseddata'].shape == (nunits, nconds), \
        "Denoised data should maintain original dimensions"
    
    # Each unit's denoised data should be a linear combination of the original data
    trial_avg = np.mean(data, axis=2)
    for unit_i in range(nunits):
        denoised_unit = result['denoiseddata'][unit_i, :]
        # The denoised data should lie in the span of the original data
        proj = np.linalg.lstsq(trial_avg.T, denoised_unit, rcond=None)[0]
        reconstructed = trial_avg.T @ proj
        assert np.allclose(denoised_unit, reconstructed, rtol=1e-10), \
            f"Denoised data for unit {unit_i} should be in span of original data"

def test_unit_mode_single_trial():
    """Test unit-wise mode with single trial denoising."""
    nunits, nconds, ntrials = 3, 10, 5
    data = np.random.randn(nunits, nconds, ntrials)
    
    opt = {
        'cv_threshold_per': 'unit',
        'cv_thresholds': [1, 2],
        'denoisingtype': 1  # Single trial denoising
    }
    result = gsn_denoise(data, V=0, opt=opt)
    
    # Check denoised data dimensions for single trial
    assert result['denoiseddata'].shape == (nunits, nconds, ntrials), \
        "Single trial denoised data should maintain original dimensions"
    
    # Each trial's denoised data should be a linear combination of the original trial data
    for trial in range(ntrials):
        for unit_i in range(nunits):
            denoised_unit = result['denoiseddata'][unit_i, :, trial]
            # The denoised data should lie in the span of the original trial data
            proj = np.linalg.lstsq(data[:, :, trial].T, denoised_unit, rcond=None)[0]
            reconstructed = data[:, :, trial].T @ proj
            assert np.allclose(denoised_unit, reconstructed, rtol=1e-10), \
                f"Denoised data for unit {unit_i}, trial {trial} should be in span of original data"

def test_unit_mode_extreme_thresholds():
    """Test unit-wise mode with extreme threshold values."""
    nunits, nconds, ntrials = 3, 10, 5
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with threshold larger than number of dimensions
    opt = {'cv_threshold_per': 'unit', 'cv_thresholds': [1, nunits + 5]}
    result = gsn_denoise(data, V=0, opt=opt)
    
    # The denoiser should still be valid
    assert result['denoiser'].shape == (nunits, nunits), \
        "Denoiser should maintain correct dimensions with extreme thresholds"

def test_unit_mode_basis_consistency():
    """Test that unit-wise mode maintains basic linear properties."""
    nunits, nconds, ntrials = 3, 10, 5
    data = np.random.randn(nunits, nconds, ntrials)
    
    opt = {'cv_threshold_per': 'unit', 'cv_thresholds': [1, 2]}
    result = gsn_denoise(data, V=0, opt=opt)
    
    # Test that denoising is a linear operation
    # If we scale the input, the output should scale proportionally
    scale = 2.0
    trial_avg = np.mean(data, axis=2)
    denoised1 = (trial_avg.T @ result['denoiser']).T
    denoised2 = ((scale * trial_avg).T @ result['denoiser']).T
    
    assert np.allclose(denoised2, scale * denoised1, rtol=1e-10), \
        "Denoising should be a linear operation" 