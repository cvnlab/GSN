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

# Magnitude fraction values to test
MAG_FRACS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
@pytest.mark.parametrize("V", V_VALUES)
def test_basis_selection_modes(data_shape, V):
    """Test different basis selection modes."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with population thresholding
    opt = {'cv_threshold_per': 'population'}
    results = gsn_denoise(data, V=V, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert isinstance(results['best_threshold'], (int, np.integer))
    assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
    assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column
    assert results['signalsubspace'].shape[0] == nunits
    assert results['signalsubspace'].shape[1] <= results['fullbasis'].shape[1]  # Can't use more dimensions than available
    assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
    assert results['dimreduce'].shape[1] == nconds

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
def test_custom_basis(data_shape):
    """Test with custom basis."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with different basis dimensions
    basis_dims = [1, max(1, nunits//2), nunits]  # Test different numbers of basis vectors
    
    for dim in basis_dims:
        # Create a random orthonormal basis with dim columns
        V = np.linalg.qr(np.random.randn(nunits, dim))[0]
        
        # Test with population thresholding
        opt = {'cv_threshold_per': 'population'}
        results = gsn_denoise(data, V=V, opt=opt)
        assert results['denoiser'].shape == (nunits, nunits)
        assert results['denoiseddata'].shape == (nunits, nconds)
        assert isinstance(results['best_threshold'], (int, np.integer))
        assert results['fullbasis'].shape == V.shape  # Basis should match input dimensions
        assert results['signalsubspace'].shape[0] == nunits
        assert results['signalsubspace'].shape[1] <= dim  # Can't use more dimensions than provided
        assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
        assert results['dimreduce'].shape[1] == nconds

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
@pytest.mark.parametrize("cv_mode", CV_MODES)
@pytest.mark.parametrize("threshold_per", THRESHOLD_PER)
def test_cv_modes_and_thresholding(threshold_per, cv_mode, data_shape):
    """Test different cross-validation modes and thresholding types."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    
    opt = {
        'cv_mode': cv_mode,
        'cv_threshold_per': threshold_per,
        'cv_thresholds': np.arange(1, nunits + 1),
        'cv_scoring_fn': lambda A, B: -np.mean((A - B)**2)
    }
    
    results = gsn_denoise(data, opt=opt)
    
    # Common assertions for all modes
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['fullbasis'].shape[0] == nunits
    assert results['fullbasis'].shape[1] >= 1
    
    # Check denoiseddata shape based on denoisingtype
    if 'denoisingtype' in opt and opt['denoisingtype'] == 1:
        assert results['denoiseddata'].shape == (nunits, nconds, ntrials)
    else:
        assert results['denoiseddata'].shape[:2] == (nunits, nconds)
    
    if cv_mode < 0:  # Magnitude thresholding mode
        assert results['mags'] is not None
        assert results['dimsretained'] is not None
        assert results['signalsubspace'] is not None
        assert results['dimreduce'] is not None
        assert len(results['mags']) == nunits
        assert isinstance(results['dimsretained'], (int, np.integer))
        assert results['signalsubspace'].shape[0] == nunits
        assert results['dimreduce'].shape[1] == nconds
    else:  # Cross-validation mode
        if threshold_per == 'population':
            assert results['signalsubspace'] is not None
            assert results['dimreduce'] is not None
            assert results['signalsubspace'].shape[0] == nunits
            assert results['dimreduce'].shape[1] == nconds
            assert isinstance(results['best_threshold'], (int, np.integer))
        else:  # 'unit'
            assert len(results['best_threshold']) == nunits
            assert results['cv_scores'].shape[0] == nunits

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
@pytest.mark.parametrize("mag_type", MAG_TYPES)
@pytest.mark.parametrize("mag_mode", MAG_MODES)
def test_magnitude_thresholding_options(mag_type, mag_mode, data_shape):
    """Test different magnitude thresholding options."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': mag_type,
        'mag_mode': mag_mode,
        'mag_frac': 0.5  # Mid-range threshold
    }
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
    assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column
    assert len(results['mags']) == results['fullbasis'].shape[1]  # One magnitude per basis dimension
    assert isinstance(results['dimsretained'], (int, np.integer))
    assert results['signalsubspace'].shape[0] == nunits
    assert results['signalsubspace'].shape[1] <= results['fullbasis'].shape[1]  # Can't use more dimensions than available
    assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
    assert results['dimreduce'].shape[1] == nconds

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
            'cv_threshold_per': 'unit',
            'cv_scoring_fn': scoring_fn
        }
        results = gsn_denoise(data, opt=opt)
        assert results['denoiser'].shape == (nunits, nunits)
        assert results['denoiseddata'].shape == (nunits, nconds)
        assert results['fullbasis'].shape == (nunits, nunits)  # Full basis should be square
        assert len(results['best_threshold']) == nunits  # Unit-wise thresholds
        assert results['cv_scores'].shape[0] == nunits  # One score per unit

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
def test_cv_thresholds_variations(data_shape):
    """Test different cross-validation threshold patterns."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test different threshold patterns
    if nunits <= 3:
        threshold_patterns = [
            np.arange(1, nunits + 1),                    # Full range
            np.array([1, nunits]),                       # Endpoints
            np.array([1, nunits]),                       # Single threshold
            np.array([1])                                # Minimum threshold
        ]
    else:
        threshold_patterns = [
            np.arange(1, nunits + 1),                    # Full range
            np.array([1, nunits//2, nunits]),            # Sparse range
            np.unique(np.logspace(0, np.log10(nunits), 5).astype(int)),  # Log-spaced (unique values)
            np.array([1, 2, 3])                          # Fixed small set
        ]
    
    for thresholds in threshold_patterns:
        opt = {
            'cv_mode': 0,
            'cv_threshold_per': 'population',
            'cv_thresholds': thresholds
        }
        results = gsn_denoise(data, opt=opt)
        assert results['denoiser'].shape == (nunits, nunits)
        assert results['denoiseddata'].shape == (nunits, nconds)
        assert isinstance(results['best_threshold'], (int, np.integer))
        assert results['fullbasis'].shape == (nunits, nunits)  # Full basis should be square
        assert results['signalsubspace'].shape == (nunits, results['best_threshold'])
        assert results['dimreduce'].shape == (results['best_threshold'], nconds)
        # Check symmetry for population thresholding
        assert np.allclose(results['denoiser'], results['denoiser'].T)

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
def test_edge_case_combinations(data_shape):
    """Test edge case combinations of parameters."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test various edge case combinations
    edge_cases = [
        # Magnitude thresholding with zero fraction - use variance-based for small matrices
        {'cv_mode': -1, 'mag_type': 1 if nunits < 5 else 0, 'mag_mode': 0, 'mag_frac': 0.0},
        # Magnitude thresholding with full fraction - use variance-based for small matrices
        {'cv_mode': -1, 'mag_type': 1 if nunits < 5 else 0, 'mag_mode': 0, 'mag_frac': 1.0},
        # Cross-validation with single threshold
        {'cv_mode': 0, 'cv_threshold_per': 'population', 'cv_thresholds': [1]},
        # Cross-validation with all possible thresholds
        {'cv_mode': 0, 'cv_threshold_per': 'unit', 'cv_thresholds': np.arange(1, nunits + 1)},
    ]
    
    for opt in edge_cases:
        results = gsn_denoise(data, opt=opt)
        if opt['cv_mode'] < 0:
            # Magnitude thresholding mode
            assert results['denoiser'].shape == (nunits, nunits)
            assert results['denoiseddata'].shape == (nunits, nconds)
            assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
            assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column
            assert len(results['mags']) == results['fullbasis'].shape[1]  # One magnitude per basis dimension
            assert isinstance(results['dimsretained'], (int, np.integer))
            assert results['signalsubspace'].shape[0] == nunits
            assert results['signalsubspace'].shape[1] <= results['fullbasis'].shape[1]  # Can't use more dimensions than available
            assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
            assert results['dimreduce'].shape[1] == nconds
        else:
            # Cross-validation mode
            if opt.get('cv_threshold_per') == 'population':
                assert results['denoiser'].shape == (nunits, nunits)
                assert results['denoiseddata'].shape == (nunits, nconds)
                assert isinstance(results['best_threshold'], (int, np.integer))
                assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
                assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column
                assert results['signalsubspace'].shape[0] == nunits
                assert results['signalsubspace'].shape[1] <= results['fullbasis'].shape[1]  # Can't use more dimensions than available
                assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
                assert results['dimreduce'].shape[1] == nconds
            else:  # 'unit'
                assert results['denoiser'].shape == (nunits, nunits)
                assert results['denoiseddata'].shape == (nunits, nconds)
                assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
                assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column
                assert len(results['best_threshold']) == nunits  # Unit-wise thresholds
                assert results['cv_scores'].shape[0] == nunits  # One score per unit

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
        
    # Test invalid cv_thresholds
    # Non-positive thresholds
    with pytest.raises(ValueError, match="must be positive integers"):
        gsn_denoise(data, opt={'cv_thresholds': [0, 1, 2]})
    with pytest.raises(ValueError, match="must be positive integers"):
        gsn_denoise(data, opt={'cv_thresholds': [-1, 1, 2]})
        
    # Non-integer thresholds
    with pytest.raises(ValueError, match="must be integers"):
        gsn_denoise(data, opt={'cv_thresholds': [1.5, 2, 3]})
        
    # Unsorted thresholds
    with pytest.raises(ValueError, match="must be in sorted order"):
        gsn_denoise(data, opt={'cv_thresholds': [3, 2, 1]})
        
    # Non-unique thresholds
    with pytest.raises(ValueError, match="must be in sorted order with unique values"):
        gsn_denoise(data, opt={'cv_thresholds': [1, 2, 2, 3]})

def test_basis_functionality():
    """Test different basis options."""
    nunits, nconds, ntrials = 8, 15, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with V=4 (random orthonormal basis)
    opt = {'cv_threshold_per': 'population'}  # Use population thresholding
    results = gsn_denoise(data, V=4, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert isinstance(results['best_threshold'], (int, np.integer))
    assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
    assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column
    assert results['signalsubspace'].shape[0] == nunits
    assert results['signalsubspace'].shape[1] <= results['fullbasis'].shape[1]  # Can't use more dimensions than available
    assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
    assert results['dimreduce'].shape[1] == nconds

def test_identity_basis():
    """Test with identity basis."""
    nunits, nconds, ntrials = 8, 15, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with V=None (identity basis)
    opt = {'cv_threshold_per': 'population'}  # Use population thresholding
    results = gsn_denoise(data, V=None, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert isinstance(results['best_threshold'], (int, np.integer))
    assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
    assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column
    assert results['signalsubspace'].shape[0] == nunits
    assert results['signalsubspace'].shape[1] <= results['fullbasis'].shape[1]  # Can't use more dimensions than available
    assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
    assert results['dimreduce'].shape[1] == nconds

def test_cv_mode_0():
    """Test cross-validation with cv_mode=0 (leave-one-out)."""
    nunits, nconds, ntrials = 6, 8, 4
    data = np.random.randn(nunits, nconds, ntrials)
    opt = {
        'cv_mode': 0,
        'cv_threshold_per': 'population',
        'cv_thresholds': np.arange(1, nunits + 1),
        'cv_scoring_fn': lambda A, B: -np.mean((A - B)**2)
    }
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert isinstance(results['best_threshold'], (int, np.integer))
    assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
    assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column
    assert results['signalsubspace'].shape[0] == nunits
    assert results['signalsubspace'].shape[1] <= results['fullbasis'].shape[1]  # Can't use more dimensions than available
    assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
    assert results['dimreduce'].shape[1] == nconds

def test_cv_mode_1():
    """Test cross-validation with cv_mode=1 (1/n-1 split)."""
    nunits, nconds, ntrials = 6, 8, 4
    data = np.random.randn(nunits, nconds, ntrials)
    opt = {
        'cv_mode': 1,
        'cv_threshold_per': 'population',
        'cv_thresholds': np.arange(1, nunits + 1),
        'cv_scoring_fn': lambda A, B: -np.mean((A - B)**2)
    }
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert isinstance(results['best_threshold'], (int, np.integer))
    assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
    assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column
    assert results['signalsubspace'].shape[0] == nunits
    assert results['signalsubspace'].shape[1] <= results['fullbasis'].shape[1]  # Can't use more dimensions than available
    assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
    assert results['dimreduce'].shape[1] == nconds

def test_cv_mode_minus1():
    """Test cross-validation with cv_mode=-1 (magnitude thresholding)."""
    nunits, nconds, ntrials = 6, 8, 4
    data = np.random.randn(nunits, nconds, ntrials)
    opt = {
        'cv_mode': -1,
        'cv_threshold_per': 'population',
        'cv_thresholds': np.arange(1, nunits + 1),
        'cv_scoring_fn': lambda A, B: -np.mean((A - B)**2)
    }
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
    assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column
    assert len(results['mags']) == results['fullbasis'].shape[1]  # One magnitude per basis dimension
    assert isinstance(results['dimsretained'], (int, np.integer))
    assert results['signalsubspace'].shape[0] == nunits
    assert results['signalsubspace'].shape[1] <= results['fullbasis'].shape[1]  # Can't use more dimensions than available
    assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
    assert results['dimreduce'].shape[1] == nconds 

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
@pytest.mark.parametrize("mag_frac", MAG_FRACS)
def test_magnitude_fractions(data_shape, mag_frac):
    """Test different magnitude fraction values."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with eigenvalue-based magnitude thresholding
    opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': 0,  # Eigenvalue-based
        'mag_mode': 0,  # Contiguous
        'mag_frac': mag_frac
    }
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
    assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column
    assert len(results['mags']) == results['fullbasis'].shape[1]  # One magnitude per basis dimension
    assert isinstance(results['dimsretained'], (int, np.integer))
    assert results['signalsubspace'].shape[0] == nunits
    assert results['signalsubspace'].shape[1] <= results['fullbasis'].shape[1]  # Can't use more dimensions than available
    assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
    assert results['dimreduce'].shape[1] == nconds
    
    # For extreme mag_frac values, check expected behavior
    if mag_frac == 0.0:
        # Should retain all dimensions
        assert results['dimsretained'] == results['fullbasis'].shape[1]
    elif mag_frac == 1.0:
        # Should retain at most one dimension
        assert results['dimsretained'] <= 1
        if results['dimsretained'] == 0:
            assert np.allclose(results['denoiser'], 0)  # Denoiser should be zero matrix
            assert np.allclose(results['denoiseddata'], 0)  # Denoised data should be zero matrix
    else:
        # Should retain some dimensions, but might retain all if the data requires it
        assert 0 < results['dimsretained'] <= results['fullbasis'].shape[1]
    
    # Test with variance-based magnitude thresholding
    opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': 1,  # Variance-based
        'mag_mode': 0,  # Contiguous
        'mag_frac': mag_frac
    }
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
    assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column
    assert len(results['mags']) == results['fullbasis'].shape[1]  # One magnitude per basis dimension
    assert isinstance(results['dimsretained'], (int, np.integer))
    assert results['signalsubspace'].shape[0] == nunits
    assert results['signalsubspace'].shape[1] <= results['fullbasis'].shape[1]  # Can't use more dimensions than available
    assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
    assert results['dimreduce'].shape[1] == nconds
    
    # For extreme mag_frac values, check expected behavior
    if mag_frac == 0.0:
        # Should retain all dimensions
        assert results['dimsretained'] == results['fullbasis'].shape[1]
    elif mag_frac == 1.0:
        # Should retain at most one dimension
        assert results['dimsretained'] <= 1
        if results['dimsretained'] == 0:
            assert np.allclose(results['denoiser'], 0)  # Denoiser should be zero matrix
            assert np.allclose(results['denoiseddata'], 0)  # Denoised data should be zero matrix
    else:
        # Should retain some dimensions, but might retain all if the data requires it
        assert 0 < results['dimsretained'] <= results['fullbasis'].shape[1]
    
    # Test with non-contiguous mode
    opt = {
        'cv_mode': -1,  # Use magnitude thresholding
        'mag_type': 0,  # Eigenvalue-based
        'mag_mode': 1,  # Non-contiguous
        'mag_frac': mag_frac
    }
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
    assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column
    assert len(results['mags']) == results['fullbasis'].shape[1]  # One magnitude per basis dimension
    assert isinstance(results['dimsretained'], (int, np.integer))
    assert results['signalsubspace'].shape[0] == nunits
    assert results['signalsubspace'].shape[1] <= results['fullbasis'].shape[1]  # Can't use more dimensions than available
    assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
    assert results['dimreduce'].shape[1] == nconds
    
    # For extreme mag_frac values, check expected behavior
    if mag_frac == 0.0:
        # Should retain all dimensions
        assert results['dimsretained'] == results['fullbasis'].shape[1]
    elif mag_frac == 1.0:
        # Should retain at most one dimension
        assert results['dimsretained'] <= 1
        if results['dimsretained'] == 0:
            assert np.allclose(results['denoiser'], 0)  # Denoiser should be zero matrix
            assert np.allclose(results['denoiseddata'], 0)  # Denoised data should be zero matrix
    else:
        # Should retain some dimensions, but might retain all if the data requires it
        assert 0 < results['dimsretained'] <= results['fullbasis'].shape[1]

@pytest.mark.parametrize("data_shape", DATA_SHAPES)
def test_magnitude_fraction_monotonicity(data_shape):
    """Test that increasing mag_frac leads to monotonically decreasing dimensions retained."""
    nunits, nconds, ntrials = data_shape
    data = np.random.randn(nunits, nconds, ntrials)

    # Test with eigenvalue-based magnitude thresholding
    prev_dims = float('inf')
    prev_mag_frac = 0.0
    for mag_frac in sorted(MAG_FRACS):
        opt = {
            'cv_mode': -1,  # Use magnitude thresholding
            'mag_type': 0,  # Eigenvalue-based
            'mag_mode': 1,  # Non-contiguous (test strict monotonicity)
            'mag_frac': mag_frac
        }
        results = gsn_denoise(data, opt=opt)
        # For non-contiguous mode, require strict monotonicity
        if mag_frac > prev_mag_frac:
            assert results['dimsretained'] <= prev_dims, f"Dimensions increased from {prev_dims} to {results['dimsretained']} when mag_frac increased from {prev_mag_frac} to {mag_frac}"
        prev_dims = results['dimsretained']
        prev_mag_frac = mag_frac

    # Test with variance-based magnitude thresholding
    prev_dims = float('inf')
    prev_mag_frac = 0.0
    for mag_frac in sorted(MAG_FRACS):
        opt = {
            'cv_mode': -1,  # Use magnitude thresholding
            'mag_type': 1,  # Variance-based
            'mag_mode': 1,  # Non-contiguous (test strict monotonicity)
            'mag_frac': mag_frac
        }
        results = gsn_denoise(data, opt=opt)
        # For non-contiguous mode, require strict monotonicity
        if mag_frac > prev_mag_frac:
            assert results['dimsretained'] <= prev_dims, f"Dimensions increased from {prev_dims} to {results['dimsretained']} when mag_frac increased from {prev_mag_frac} to {mag_frac}"
        prev_dims = results['dimsretained']
        prev_mag_frac = mag_frac 