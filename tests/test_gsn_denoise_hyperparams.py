import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from gsn.gsn_denoise import gsn_denoise

# Test data shapes to try
DATA_SHAPES = [
    (5, 5, 3),    # Square case
    (10, 5, 4),   # More units than conditions
    (5, 10, 3),   # More conditions than units
    (20, 3, 5),   # Many units, few conditions
    (3, 20, 5),   # Few units, many conditions
]

# V values to test
V_VALUES = [0, 1, 2, 3, 4]

# Cross-validation modes
CV_MODES = [0, 1, -1]  # -1 for magnitude thresholding

# Threshold per modes
THRESHOLD_PER = ['unit', 'population']

# Magnitude thresholding types
MAG_TYPES = [0, 1]  # 0 for eigenvalue-based, 1 for variance-based

# Magnitude thresholding modes
MAG_MODES = [0, 1]  # 0 for contiguous, 1 for all that survive

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
@pytest.mark.parametrize("V", V_VALUES)
def test_basis_selection_modes(data_shape, V):
    """Test different basis selection modes with various data shapes."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    denoiser, scores, threshold = gsn_denoise(data, V=V)
    assert denoiser.shape == (nunits, nunits)
    assert np.allclose(denoiser, denoiser.T)  # Check symmetry

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
def test_custom_basis(data_shape):
    """Test using a custom basis matrix."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    # Create random orthonormal basis
    basis, _ = np.linalg.qr(np.random.randn(nunits, nunits))
    denoiser, scores, threshold = gsn_denoise(data, V=basis)
    assert denoiser.shape == (nunits, nunits)
    assert np.allclose(denoiser, denoiser.T)

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
@pytest.mark.parametrize("cv_mode", CV_MODES)
@pytest.mark.parametrize("threshold_per", THRESHOLD_PER)
def test_cv_modes_and_thresholding(data_shape, cv_mode, threshold_per):
    """Test different cross-validation modes and threshold types."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    opt = {
        'cv_mode': cv_mode,
        'cv_threshold_per': threshold_per,
        'cv_thresholds': np.arange(1, min(nunits, nconds) + 1)
    }
    denoiser, scores, threshold = gsn_denoise(data, V=0, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    if threshold_per == 'unit':
        assert len(threshold) == nunits
    else:
        assert np.isscalar(threshold)

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
@pytest.mark.parametrize("mag_type", MAG_TYPES)
@pytest.mark.parametrize("mag_mode", MAG_MODES)
def test_magnitude_thresholding_options(data_shape, mag_type, mag_mode):
    """Test different magnitude thresholding options."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': mag_type,
        'mag_mode': mag_mode,
        'mag_frac': 0.5  # Mid-range threshold
    }
    denoiser, scores, threshold = gsn_denoise(data, V=0, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    assert np.allclose(denoiser, denoiser.T)

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
def test_custom_scoring_functions(data_shape):
    """Test different custom scoring functions."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test different scoring functions
    scoring_functions = [
        lambda x, y: -np.mean((x - y)**2, axis=0),  # MSE per unit
        lambda x, y: -np.mean((x - y)**2),          # Single MSE value
        lambda x, y: -np.sum(np.abs(x - y), axis=0),  # L1 loss per unit
        lambda x, y: np.array([np.corrcoef(x[:,i], y[:,i])[0,1] for i in range(y.shape[1])])  # Correlation per unit
    ]
    
    for scoring_fn in scoring_functions:
        opt = {
            'cv_mode': 0,
            'cv_scoring_fn': scoring_fn,
            'cv_threshold_per': 'unit'
        }
        denoiser, scores, threshold = gsn_denoise(data, V=0, opt=opt)
        assert denoiser.shape == (nunits, nunits)
        assert np.allclose(denoiser, denoiser.T)

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
def test_cv_thresholds_variations(data_shape):
    """Test different cross-validation threshold patterns."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test different threshold patterns
    threshold_patterns = [
        np.arange(1, nunits + 1),                    # Full range
        np.array([1, nunits//2, nunits]),            # Sparse range
        np.logspace(0, np.log10(nunits), 5).astype(int),  # Log-spaced
        np.array([1, 2, 3])                          # Fixed small set
    ]
    
    for thresholds in threshold_patterns:
        opt = {
            'cv_mode': 0,
            'cv_threshold_per': 'population',
            'cv_thresholds': thresholds
        }
        denoiser, scores, threshold = gsn_denoise(data, V=0, opt=opt)
        assert denoiser.shape == (nunits, nunits)
        assert threshold in thresholds

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
def test_edge_case_combinations(data_shape):
    """Test edge case combinations of parameters."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test various edge case combinations
    edge_cases = [
        # Magnitude thresholding with zero fraction
        {'cv_mode': -1, 'mag_type': 0, 'mag_mode': 0, 'mag_frac': 0.0},
        # Magnitude thresholding with full fraction
        {'cv_mode': -1, 'mag_type': 0, 'mag_mode': 0, 'mag_frac': 1.0},
        # Cross-validation with single threshold
        {'cv_mode': 0, 'cv_threshold_per': 'population', 'cv_thresholds': [1]},
        # Cross-validation with all possible thresholds
        {'cv_mode': 0, 'cv_threshold_per': 'unit', 'cv_thresholds': np.arange(1, nunits + 1)},
    ]
    
    for opt in edge_cases:
        denoiser, scores, threshold = gsn_denoise(data, V=0, opt=opt)
        assert denoiser.shape == (nunits, nunits)
        assert np.allclose(denoiser, denoiser.T)

def test_parameter_validation():
    """Test parameter validation and error handling."""
    data = np.random.randn(5, 5, 3)
    
    # Test invalid V value
    with pytest.raises(ValueError):
        gsn_denoise(data, V=5)
    
    # Test invalid cv_threshold_per
    with pytest.raises(KeyError):
        gsn_denoise(data, V=0, opt={'cv_threshold_per': 'invalid'})
    
    # Test invalid data shape
    invalid_data = np.random.randn(5, 5)
    with pytest.raises(ValueError):
        gsn_denoise(invalid_data, V=0)
    
    # Test data with too few trials
    invalid_data = np.random.randn(5, 5, 1)
    with pytest.raises(ValueError):
        gsn_denoise(invalid_data, V=0) 