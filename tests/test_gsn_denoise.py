"""Tests for gsn_denoise.py using simulated data."""

import sys, os
sys.path.insert(0, os.path.abspath("/Users/jacobprince/KonkLab Dropbox/Jacob Prince/Research-Prince/GSNdenoise/GSN"))

import numpy as np
import pytest
from gsn.simulate_data import generate_data
from gsn.gsn_denoise import gsn_denoise, perform_cross_validation, perform_magnitude_thresholding
from gsn.perform_gsn import perform_gsn

def test_basic_functionality():
    """Test basic functionality of gsn_denoise."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with default options
    denoiser, cv_scores, best_threshold, denoiseddata, fullbasis = gsn_denoise(data)
    assert denoiser.shape == (nunits, nunits)
    assert denoiseddata.shape == (nunits, nconds)
    assert fullbasis.shape[0] == nunits  # Basis should have nunits rows
    assert fullbasis.shape[1] >= 1  # Basis should have at least 1 column

def test_cross_validation_population():
    """Test cross-validation with population thresholding."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with population thresholding
    opt = {'cv_threshold_per': 'population'}
    denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce = gsn_denoise(data, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    assert denoiseddata.shape == (nunits, nconds)
    assert isinstance(best_threshold, (int, np.integer))
    assert fullbasis.shape == (nunits, nunits)  # Full basis should be square
    assert signalsubspace.shape == (nunits, best_threshold)
    assert dimreduce.shape == (best_threshold, nconds)
    # Check symmetry for population thresholding
    assert np.allclose(denoiser, denoiser.T)

def test_cross_validation_unit():
    """Test cross-validation with unit thresholding."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with unit thresholding
    opt = {'cv_threshold_per': 'unit'}
    denoiser, cv_scores, best_threshold, denoiseddata, fullbasis = gsn_denoise(data, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    assert denoiseddata.shape == (nunits, nconds)
    assert fullbasis.shape == (nunits, nunits)  # Full basis should be square
    assert len(best_threshold) == nunits  # One threshold per unit
    assert cv_scores.shape[0] == nunits  # One score per unit

def test_magnitude_thresholding():
    """Test magnitude thresholding mode."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with magnitude thresholding
    opt = {'cv_mode': -1}
    denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, mags, dimsretained, signalsubspace, dimreduce = gsn_denoise(data, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    assert denoiseddata.shape == (nunits, nconds)
    assert fullbasis.shape == (nunits, nunits)  # Full basis should be square
    assert len(mags) == nunits  # One magnitude per dimension
    assert isinstance(dimsretained, (int, np.integer))  # Number of dimensions retained
    assert signalsubspace.shape == (nunits, dimsretained)
    assert dimreduce.shape == (dimsretained, nconds)
    # Check symmetry for population thresholding
    assert np.allclose(denoiser, denoiser.T)

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
        denoiser, cv_scores, best_threshold, denoiseddata, fullbasis = gsn_denoise(data, V=V)
        assert denoiser.shape == (nunits, nunits)
        assert denoiseddata.shape == (nunits, nconds)
        assert fullbasis.shape == V.shape  # Basis should match input dimensions
        
        # Test with population thresholding
        opt = {'cv_threshold_per': 'population'}
        denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce = gsn_denoise(data, V=V, opt=opt)
        assert denoiser.shape == (nunits, nunits)
        assert denoiseddata.shape == (nunits, nconds)
        assert isinstance(best_threshold, (int, np.integer))
        assert fullbasis.shape == V.shape  # Basis should match input dimensions
        assert signalsubspace.shape[0] == nunits
        assert signalsubspace.shape[1] <= dim  # Can't use more dimensions than provided
        assert dimreduce.shape[0] == signalsubspace.shape[1]
        assert dimreduce.shape[1] == nconds
        
        # Test with magnitude thresholding
        opt = {'cv_mode': -1}
        denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, mags, dimsretained, signalsubspace, dimreduce = gsn_denoise(data, V=V, opt=opt)
        assert denoiser.shape == (nunits, nunits)
        assert denoiseddata.shape == (nunits, nconds)
        assert fullbasis.shape == V.shape  # Basis should match input dimensions
        assert len(mags) == dim  # One magnitude per basis dimension
        assert isinstance(dimsretained, (int, np.integer))
        assert signalsubspace.shape[0] == nunits
        assert signalsubspace.shape[1] <= dim  # Can't use more dimensions than provided
        assert dimreduce.shape[0] == signalsubspace.shape[1]
        assert dimreduce.shape[1] == nconds

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
    denoiser, cv_scores, best_threshold, denoiseddata, fullbasis = gsn_denoise(data, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    assert denoiseddata.shape == (nunits, nconds)
    assert fullbasis.shape == (nunits, nunits)  # Full basis should be square
    
    # Test with population thresholding and custom scoring
    opt = {'cv_threshold_per': 'population', 'cv_scoring_fn': custom_score}
    denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce = gsn_denoise(data, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    assert denoiseddata.shape == (nunits, nconds)
    assert isinstance(best_threshold, (int, np.integer))
    assert fullbasis.shape == (nunits, nunits)  # Full basis should be square
    assert signalsubspace.shape == (nunits, best_threshold)
    assert dimreduce.shape == (best_threshold, nconds)
    # Check symmetry for population thresholding
    assert np.allclose(denoiser, denoiser.T)
    
    # Test with magnitude thresholding and custom scoring
    opt = {'cv_mode': -1, 'cv_scoring_fn': custom_score}
    denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, mags, dimsretained, signalsubspace, dimreduce = gsn_denoise(data, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    assert denoiseddata.shape == (nunits, nconds)
    assert fullbasis.shape == (nunits, nunits)  # Full basis should be square
    assert len(mags) == nunits  # One magnitude per dimension
    assert isinstance(dimsretained, (int, np.integer))  # Number of dimensions retained
    assert signalsubspace.shape == (nunits, dimsretained)
    assert dimreduce.shape == (dimsretained, nconds)
    # Check symmetry for population thresholding
    assert np.allclose(denoiser, denoiser.T)

def test_population_thresholding():
    """Test population thresholding mode."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with population thresholding
    opt = {'cv_threshold_per': 'population'}
    denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce = gsn_denoise(data, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    assert denoiseddata.shape == (nunits, nconds)
    assert isinstance(best_threshold, (int, np.integer))
    assert fullbasis.shape == (nunits, nunits)  # Full basis should be square
    assert signalsubspace.shape == (nunits, best_threshold)
    assert dimreduce.shape == (best_threshold, nconds)
    # Check symmetry for population thresholding
    assert np.allclose(denoiser, denoiser.T)

def test_unit_thresholding():
    """Test unit thresholding mode."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with unit thresholding
    opt = {'cv_threshold_per': 'unit'}
    denoiser, cv_scores, best_threshold, denoiseddata, fullbasis = gsn_denoise(data, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    assert denoiseddata.shape == (nunits, nconds)
    assert fullbasis.shape == (nunits, nunits)  # Full basis should be square
    assert len(best_threshold) == nunits  # One threshold per unit
    assert cv_scores.shape[0] == nunits  # One score per unit

def test_magnitude_thresholding():
    """Test magnitude thresholding mode."""
    nunits = 8
    nconds = 10
    ntrials = 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with magnitude thresholding
    opt = {'cv_mode': -1}
    denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, mags, dimsretained, signalsubspace, dimreduce = gsn_denoise(data, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    assert denoiseddata.shape == (nunits, nconds)
    assert fullbasis.shape == (nunits, nunits)  # Full basis should be square
    assert len(mags) == nunits  # One magnitude per dimension
    assert isinstance(dimsretained, (int, np.integer))  # Number of dimensions retained
    assert signalsubspace.shape == (nunits, dimsretained)
    assert dimreduce.shape == (dimsretained, nconds)
    # Check symmetry for population thresholding
    assert np.allclose(denoiser, denoiser.T)

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
    denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce = gsn_denoise(data, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    assert denoiseddata.shape == (nunits, nconds)
    assert isinstance(best_threshold, (int, np.integer))
    assert fullbasis.shape == (nunits, nunits)  # Full basis should be square
    assert signalsubspace.shape == (nunits, best_threshold)
    assert dimreduce.shape == (best_threshold, nconds)
    # Check symmetry for population thresholding
    assert np.allclose(denoiser, denoiser.T)

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
    denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, signalsubspace, dimreduce = gsn_denoise(data, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    assert denoiseddata.shape == (nunits, nconds)
    assert isinstance(best_threshold, (int, np.integer))
    assert fullbasis.shape == (nunits, nunits)  # Full basis should be square
    assert signalsubspace.shape == (nunits, best_threshold)
    assert dimreduce.shape == (best_threshold, nconds)
    # Check symmetry for population thresholding
    assert np.allclose(denoiser, denoiser.T)

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
    denoiser, cv_scores, best_threshold, denoiseddata, fullbasis, mags, dimsretained, signalsubspace, dimreduce = gsn_denoise(data, opt=opt)
    assert denoiser.shape == (nunits, nunits)
    assert denoiseddata.shape == (nunits, nconds)
    assert fullbasis.shape == (nunits, nunits)  # Full basis should be square
    assert len(mags) == nunits  # One magnitude per dimension
    assert isinstance(dimsretained, (int, np.integer))  # Number of dimensions retained
    assert signalsubspace.shape == (nunits, dimsretained)
    assert dimreduce.shape == (dimsretained, nconds)
    # Check symmetry for population thresholding
    assert np.allclose(denoiser, denoiser.T)

