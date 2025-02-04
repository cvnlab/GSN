"""Tests for gsn_denoise.py using simulated data."""

import sys, os
sys.path.insert(0, os.path.abspath("/Users/jacobprince/KonkLab Dropbox/Jacob Prince/Research-Prince/GSNdenoise/GSN"))

import numpy as np
import pytest
from gsn.gsn_denoise import gsn_denoise

def test_basic_functionality():
    """Test basic functionality of gsn_denoise."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with default options
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert results['fullbasis'].shape[0] == nunits  # Basis should have nunits rows
    assert results['fullbasis'].shape[1] >= 1  # Basis should have at least 1 column

def test_cross_validation_population():
    """Test cross-validation with population thresholding."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with population thresholding
    opt = {'cv_threshold_per': 'population'}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert isinstance(results['best_threshold'], (int, np.integer))
    assert results['fullbasis'].shape == (nunits, nunits)  # Full basis should be square
    assert results['signalsubspace'].shape == (nunits, results['best_threshold'])
    assert results['dimreduce'].shape == (results['best_threshold'], nconds)
    # Check symmetry for population thresholding
    assert np.allclose(results['denoiser'], results['denoiser'].T)

def test_cross_validation_unit():
    """Test cross-validation with unit thresholding."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with unit thresholding
    opt = {'cv_threshold_per': 'unit'}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert results['fullbasis'].shape == (nunits, nunits)  # Full basis should be square
    assert len(results['best_threshold']) == nunits  # One threshold per unit
    assert results['cv_scores'].shape[0] == nunits  # One score per unit

def test_magnitude_thresholding():
    """Test magnitude thresholding mode."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with magnitude thresholding
    opt = {'cv_mode': -1}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert results['fullbasis'].shape == (nunits, nunits)  # Full basis should be square
    assert len(results['mags']) == nunits  # One magnitude per dimension
    assert isinstance(results['dimsretained'], (int, np.integer))  # Number of dimensions retained
    assert results['signalsubspace'].shape == (nunits, results['dimsretained'])
    assert results['dimreduce'].shape == (results['dimsretained'], nconds)
    # Check symmetry for population thresholding
    assert np.allclose(results['denoiser'], results['denoiser'].T)

def test_custom_basis():
    """Test denoising with custom basis."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with different basis dimensions
    basis_dims = [1, nunits//2, nunits]  # Test different numbers of basis vectors
    
    for dim in basis_dims:
        # Create a random orthonormal basis with dim columns
        V = np.linalg.qr(np.random.randn(nunits, dim))[0]
        
        # Test with default options
        results = gsn_denoise(data, V=V)
        assert results['denoiser'].shape == (nunits, nunits)
        assert results['denoiseddata'].shape == (nunits, nconds)
        assert results['fullbasis'].shape == V.shape  # Basis should match input dimensions
        
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
        
        # Test with magnitude thresholding
        opt = {'cv_mode': -1}
        results = gsn_denoise(data, V=V, opt=opt)
        assert results['denoiser'].shape == (nunits, nunits)
        assert results['denoiseddata'].shape == (nunits, nconds)
        assert results['fullbasis'].shape == V.shape  # Basis should match input dimensions
        assert len(results['mags']) == dim  # One magnitude per basis dimension
        assert isinstance(results['dimsretained'], (int, np.integer))
        assert results['signalsubspace'].shape[0] == nunits
        assert results['signalsubspace'].shape[1] <= dim  # Can't use more dimensions than provided
        assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
        assert results['dimreduce'].shape[1] == nconds

def test_custom_scoring():
    """Test denoising with custom scoring function."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Define a custom scoring function
    def custom_score(A, B):
        return -np.mean(np.abs(A - B))
    
    # Test with default options and custom scoring
    opt = {'cv_scoring_fn': custom_score}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert results['fullbasis'].shape == (nunits, nunits)  # Full basis should be square
    
    # Test with population thresholding and custom scoring
    opt = {'cv_threshold_per': 'population', 'cv_scoring_fn': custom_score}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert isinstance(results['best_threshold'], (int, np.integer))
    assert results['fullbasis'].shape == (nunits, nunits)  # Full basis should be square
    assert results['signalsubspace'].shape == (nunits, results['best_threshold'])
    assert results['dimreduce'].shape == (results['best_threshold'], nconds)
    # Check symmetry for population thresholding
    assert np.allclose(results['denoiser'], results['denoiser'].T)
    
    # Test with magnitude thresholding and custom scoring
    opt = {'cv_mode': -1, 'cv_scoring_fn': custom_score}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert results['fullbasis'].shape == (nunits, nunits)  # Full basis should be square
    assert len(results['mags']) == nunits  # One magnitude per dimension
    assert isinstance(results['dimsretained'], (int, np.integer))  # Number of dimensions retained
    assert results['signalsubspace'].shape == (nunits, results['dimsretained'])
    assert results['dimreduce'].shape == (results['dimsretained'], nconds)
    # Check symmetry for population thresholding
    assert np.allclose(results['denoiser'], results['denoiser'].T)

def test_population_thresholding():
    """Test population thresholding mode."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with population thresholding
    opt = {'cv_threshold_per': 'population'}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert isinstance(results['best_threshold'], (int, np.integer))
    assert results['fullbasis'].shape == (nunits, nunits)  # Full basis should be square
    assert results['signalsubspace'].shape == (nunits, results['best_threshold'])
    assert results['dimreduce'].shape == (results['best_threshold'], nconds)
    # Check symmetry for population thresholding
    assert np.allclose(results['denoiser'], results['denoiser'].T)

def test_unit_thresholding():
    """Test unit thresholding mode."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with unit thresholding
    opt = {'cv_threshold_per': 'unit'}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert results['fullbasis'].shape == (nunits, nunits)  # Full basis should be square
    assert len(results['best_threshold']) == nunits  # One threshold per unit
    assert results['cv_scores'].shape[0] == nunits  # One score per unit

def test_cv_mode_0():
    """Test cross-validation with cv_mode=0 (leave-one-out)."""
    nunits = 6
    nconds = 8
    ntrials = 4
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
    assert results['fullbasis'].shape == (nunits, nunits)  # Full basis should be square
    assert results['signalsubspace'].shape == (nunits, results['best_threshold'])
    assert results['dimreduce'].shape == (results['best_threshold'], nconds)
    # Check symmetry for population thresholding
    assert np.allclose(results['denoiser'], results['denoiser'].T)

def test_cv_mode_1():
    """Test cross-validation with cv_mode=1 (1/n-1 split)."""
    nunits = 6
    nconds = 8
    ntrials = 4
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
    assert results['fullbasis'].shape == (nunits, nunits)  # Full basis should be square
    assert results['signalsubspace'].shape == (nunits, results['best_threshold'])
    assert results['dimreduce'].shape == (results['best_threshold'], nconds)
    # Check symmetry for population thresholding
    assert np.allclose(results['denoiser'], results['denoiser'].T)

def test_cv_mode_minus1():
    """Test cross-validation with cv_mode=-1 (magnitude thresholding)."""
    nunits = 6
    nconds = 8
    ntrials = 4
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
    assert results['fullbasis'].shape == (nunits, nunits)  # Full basis should be square
    assert len(results['mags']) == nunits  # One magnitude per dimension
    assert isinstance(results['dimsretained'], (int, np.integer))  # Number of dimensions retained
    assert results['signalsubspace'].shape == (nunits, results['dimsretained'])
    assert results['dimreduce'].shape == (results['dimsretained'], nconds)
    # Check symmetry for population thresholding
    assert np.allclose(results['denoiser'], results['denoiser'].T)

def test_custom_nonsquare_basis():
    """Test denoising with custom non-square basis under various hyperparameters."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)

    # Test different basis dimensions smaller than nunits
    basis_dims = [1, nunits//4, nunits//2]  # Test different numbers of basis vectors

    for dim in basis_dims:
        # Create a random orthonormal basis with dim columns
        V = np.linalg.qr(np.random.randn(nunits, dim))[0]

        # Test with default options
        results = gsn_denoise(data, V=V)
        assert results['denoiser'].shape == (nunits, nunits)
        assert results['denoiseddata'].shape == (nunits, nconds)
        assert results['fullbasis'].shape == V.shape  # Basis should match input dimensions

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

        # Test with magnitude thresholding and different mag_types
        for mag_type in [0, 1]:
            opt = {'cv_mode': -1, 'mag_type': mag_type}
            results = gsn_denoise(data, V=V, opt=opt)
            assert results['denoiser'].shape == (nunits, nunits)
            assert results['denoiseddata'].shape == (nunits, nconds)
            assert results['fullbasis'].shape == V.shape  # Basis should match input dimensions
            assert len(results['mags']) == dim  # One magnitude per basis dimension
            assert isinstance(results['dimsretained'], (int, np.integer))
            if results['signalsubspace'] is not None:  # If any dimensions retained
                assert results['signalsubspace'].shape[0] == nunits
                assert results['signalsubspace'].shape[1] <= dim  # Can't use more dimensions than provided
                assert results['dimreduce'].shape[0] == results['signalsubspace'].shape[1]
                assert results['dimreduce'].shape[1] == nconds

        # Test with single-trial denoising
        opt = {'denoisingtype': 1}
        results = gsn_denoise(data, V=V, opt=opt)
        assert results['denoiser'].shape == (nunits, nunits)
        assert results['denoiseddata'].shape == (nunits, nconds, ntrials)
        assert results['fullbasis'].shape == V.shape

        # Test with different mag_modes
        for mag_mode in [0, 1]:
            opt = {'cv_mode': -1, 'mag_mode': mag_mode}
            results = gsn_denoise(data, V=V, opt=opt)
            assert results['denoiser'].shape == (nunits, nunits)
            assert results['denoiseddata'].shape == (nunits, nconds)
            assert results['fullbasis'].shape == V.shape
            assert len(results['mags']) == dim
            assert isinstance(results['dimsretained'], (int, np.integer))

        # Test with different mag_fracs
        for mag_frac in [0.01, 0.1, 0.5]:
            opt = {'cv_mode': -1, 'mag_frac': mag_frac}
            results = gsn_denoise(data, V=V, opt=opt)
            assert results['denoiser'].shape == (nunits, nunits)
            assert results['denoiseddata'].shape == (nunits, nconds)
            assert results['fullbasis'].shape == V.shape
            assert len(results['mags']) == dim
            assert isinstance(results['dimsretained'], (int, np.integer))

        # Test with custom cv_thresholds
        opt = {'cv_thresholds': np.arange(1, dim+1, 2)}  # Test odd thresholds
        results = gsn_denoise(data, V=V, opt=opt)
        assert results['denoiser'].shape == (nunits, nunits)
        assert results['denoiseddata'].shape == (nunits, nconds)
        assert results['fullbasis'].shape == V.shape

def test_custom_basis_edge_cases():
    """Test edge cases with custom basis."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)

    # Test with minimum basis (1 dimension)
    V = np.linalg.qr(np.random.randn(nunits, 1))[0]
    results = gsn_denoise(data, V=V)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert results['fullbasis'].shape == V.shape

    # Test with magnitude thresholding that retains no dimensions
    V = np.linalg.qr(np.random.randn(nunits, 2))[0]
    opt = {'cv_mode': -1, 'mag_frac': 1.1}  # Set threshold higher than max magnitude
    results = gsn_denoise(data, V=V, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    assert np.allclose(results['denoiser'], 0)  # Should be zero matrix
    assert results['denoiseddata'].shape == (nunits, nconds)
    assert np.allclose(results['denoiseddata'], 0)  # Should be zero matrix
    assert results['fullbasis'].shape == V.shape
    assert len(results['mags']) == 2
    assert results['dimsretained'] == 0
    assert results['signalsubspace'].shape == (nunits, 0)  # Empty but valid shape
    assert results['dimreduce'].shape == (0, nconds)  # Empty but valid shape

    # Test with single condition (edge case)
    data_single_cond = np.random.randn(nunits, 1, ntrials)
    V = np.linalg.qr(np.random.randn(nunits, 3))[0]
    with pytest.raises(AssertionError):  # Should fail with single condition
        gsn_denoise(data_single_cond, V=V)

    # Test with two conditions (minimum required)
    data_two_conds = np.random.randn(nunits, 2, ntrials)
    results = gsn_denoise(data_two_conds, V=V)
    assert results['denoiser'].shape == (nunits, nunits)
    assert results['denoiseddata'].shape == (nunits, 2)
    assert results['fullbasis'].shape == V.shape

def test_parameter_validation():
    """Test parameter validation and error handling."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test invalid V value
    with pytest.raises(ValueError):
        gsn_denoise(data, V=5)
    
    # Test invalid cv_threshold_per
    with pytest.raises(KeyError):
        gsn_denoise(data, V=0, opt={'cv_threshold_per': 'invalid'})
    
    # Test invalid data shape
    invalid_data = np.random.randn(nunits, nconds)
    with pytest.raises(ValueError):
        gsn_denoise(invalid_data, V=0)
    
    # Test data with too few trials
    invalid_data = np.random.randn(nunits, nconds, 1)
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

