"""Test the plotting functionality of gsn_denoise to identify errors in diagnostic figures.

This test focuses specifically on the plotting/visualization components of the GSN library
to catch any errors that might occur during figure generation.
"""

import sys, os
sys.path.insert(0, os.path.abspath("/Users/jacobprince/KonkLab Dropbox/Jacob Prince/Research-Prince/GSNdenoise-updated/GSN"))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
from gsn.gsn_denoise import gsn_denoise

def create_plotting_test_data():
    """Create test data specifically designed to stress test plotting functions."""
    np.random.seed(42)
    
    nunits = 15
    nconds = 20
    ntrials = 4
    
    # Create data with diverse characteristics to test plotting edge cases
    unit_means = np.random.randn(nunits) * 3
    signal = np.random.randn(nunits, nconds) * 2
    noise = np.random.randn(nunits, nconds, ntrials) * 0.8
    
    data = np.zeros((nunits, nconds, ntrials))
    for t in range(ntrials):
        data[:, :, t] = unit_means[:, np.newaxis] + signal + noise[:, :, t]
    
    return data

def test_basic_plotting_functionality():
    """Test basic plotting functionality with default settings."""
    print("Testing basic plotting functionality...")
    
    data = create_plotting_test_data()
    
    try:
        # Test with wantfig=True to trigger plotting
        results = gsn_denoise(data, V=0, wantfig=True)
        plt.close('all')  # Clean up figures
        print("✓ Basic plotting test passed")
        return True
    except Exception as e:
        print(f"✗ Basic plotting test failed: {e}")
        return False

def test_plotting_all_basis_types():
    """Test plotting functionality across all basis types."""
    print("Testing plotting with all basis types...")
    
    data = create_plotting_test_data()
    basis_types = [0, 1, 2, 3, 4]
    
    for V in basis_types:
        try:
            print(f"  Testing V={V}...")
            results = gsn_denoise(data, V=V, wantfig=True)
            plt.close('all')  # Clean up figures
            print(f"  ✓ V={V} plotting passed")
        except Exception as e:
            print(f"  ✗ V={V} plotting failed: {e}")
            return False
    
    print("✓ All basis types plotting test passed")
    return True

def test_plotting_different_cv_modes():
    """Test plotting with different cross-validation modes."""
    print("Testing plotting with different CV modes...")
    
    data = create_plotting_test_data()
    
    cv_configs = [
        {'cv_mode': 0, 'cv_threshold_per': 'population'},
        {'cv_mode': 0, 'cv_threshold_per': 'unit'},
        {'cv_mode': 1, 'cv_threshold_per': 'population'},
        {'cv_mode': 1, 'cv_threshold_per': 'unit'},
        {'cv_mode': -1, 'mag_frac': 0.1}  # Magnitude thresholding
    ]
    
    for i, config in enumerate(cv_configs):
        try:
            print(f"  Testing config {i+1}: {config}")
            results = gsn_denoise(data, V=0, opt=config, wantfig=True)
            plt.close('all')  # Clean up figures
            print(f"  ✓ Config {i+1} plotting passed")
        except Exception as e:
            print(f"  ✗ Config {i+1} plotting failed: {e}")
            return False
    
    print("✓ All CV modes plotting test passed")
    return True

def test_plotting_different_denoising_types():
    """Test plotting with different denoising types."""
    print("Testing plotting with different denoising types...")
    
    data = create_plotting_test_data()
    
    denoising_configs = [
        {'denoisingtype': 0},  # Trial-averaged
        {'denoisingtype': 1}   # Single-trial
    ]
    
    for i, config in enumerate(denoising_configs):
        try:
            print(f"  Testing denoising type {config['denoisingtype']}...")
            results = gsn_denoise(data, V=0, opt=config, wantfig=True)
            plt.close('all')  # Clean up figures
            print(f"  ✓ Denoising type {config['denoisingtype']} plotting passed")
        except Exception as e:
            print(f"  ✗ Denoising type {config['denoisingtype']} plotting failed: {e}")
            return False
    
    print("✓ All denoising types plotting test passed")
    return True

def test_plotting_with_custom_basis():
    """Test plotting with custom user-supplied basis."""
    print("Testing plotting with custom basis...")
    
    data = create_plotting_test_data()
    nunits = data.shape[0]
    
    try:
        # Create custom orthonormal basis
        custom_basis = np.linalg.qr(np.random.randn(nunits, nunits))[0]
        results = gsn_denoise(data, V=custom_basis, wantfig=True)
        plt.close('all')  # Clean up figures
        print("✓ Custom basis plotting test passed")
        return True
    except Exception as e:
        print(f"✗ Custom basis plotting test failed: {e}")
        return False

def test_plotting_edge_cases():
    """Test plotting with edge case data."""
    print("Testing plotting with edge cases...")
    
    # Test with very small data
    try:
        print("  Testing with minimal data...")
        small_data = np.random.randn(3, 4, 2)
        results = gsn_denoise(small_data, V=0, wantfig=True)
        plt.close('all')
        print("  ✓ Minimal data plotting passed")
    except Exception as e:
        print(f"  ✗ Minimal data plotting failed: {e}")
        return False
    
    # Test with zero-mean data
    try:
        print("  Testing with zero-mean data...")
        zero_mean_data = np.random.randn(8, 10, 3)
        zero_mean_data = zero_mean_data - np.mean(zero_mean_data, axis=(1, 2), keepdims=True)
        results = gsn_denoise(zero_mean_data, V=0, wantfig=True)
        plt.close('all')
        print("  ✓ Zero-mean data plotting passed")
    except Exception as e:
        print(f"  ✗ Zero-mean data plotting failed: {e}")
        return False
    
    # Test with extreme values
    try:
        print("  Testing with extreme values...")
        extreme_data = np.random.randn(6, 8, 3) * 100  # Very large values
        results = gsn_denoise(extreme_data, V=0, wantfig=True)
        plt.close('all')
        print("  ✓ Extreme values plotting passed")
    except Exception as e:
        print(f"  ✗ Extreme values plotting failed: {e}")
        return False
    
    print("✓ All edge cases plotting test passed")
    return True

def test_regenerate_visualization():
    """Test the regenerate visualization functionality."""
    print("Testing regenerate visualization functionality...")
    
    data = create_plotting_test_data()
    
    try:
        # First run without figures
        results = gsn_denoise(data, V=0, wantfig=False)
        
        # Test regenerating visualization
        print("  Testing plot regeneration...")
        results['plot']()  # Should regenerate the visualization
        plt.close('all')
        
        # Test with test data
        print("  Testing plot regeneration with test data...")
        test_data = create_plotting_test_data()
        results['plot'](test_data=test_data)
        plt.close('all')
        
        print("✓ Regenerate visualization test passed")
        return True
    except Exception as e:
        print(f"✗ Regenerate visualization test failed: {e}")
        return False

def test_plotting_error_handling():
    """Test error handling in plotting functions."""
    print("Testing plotting error handling...")
    
    try:
        # Test with problematic data that might cause plotting issues
        data = create_plotting_test_data()
        
        # Test with invalid cv_thresholds that might cause indexing errors
        opt = {
            'cv_thresholds': [1, 2, 100],  # 100 is larger than data dimensions
            'cv_mode': 0
        }
        
        print("  Testing with potentially problematic cv_thresholds...")
        results = gsn_denoise(data, V=0, opt=opt, wantfig=True)
        plt.close('all')
        
        print("✓ Plotting error handling test passed")
        return True
    except Exception as e:
        print(f"✗ Plotting error handling test failed: {e}")
        return False

def test_cross_validation_scores_plotting():
    """Test specific cross-validation scores plotting functionality."""
    print("Testing cross-validation scores plotting...")
    
    data = create_plotting_test_data()
    
    try:
        # Use specific options that should generate CV scores
        opt = {
            'cv_mode': 0,
            'cv_threshold_per': 'unit',
            'cv_thresholds': np.arange(1, 11)  # Test first 10 dimensions
        }
        
        results = gsn_denoise(data, V=0, opt=opt, wantfig=True)
        
        # Check that cv_scores exist and have expected shape
        if 'cv_scores' in results and results['cv_scores'] is not None:
            print(f"  CV scores shape: {results['cv_scores'].shape}")
            print("  ✓ CV scores generated successfully")
        else:
            print("  ! CV scores not found in results")
        
        plt.close('all')
        print("✓ Cross-validation scores plotting test passed")
        return True
    except Exception as e:
        print(f"✗ Cross-validation scores plotting test failed: {e}")
        return False

def run_all_plotting_tests():
    """Run all plotting tests and report results."""
    print("=" * 60)
    print("RUNNING GSN PLOTTING TESTS")
    print("=" * 60)
    
    tests = [
        test_basic_plotting_functionality,
        test_plotting_all_basis_types,
        test_plotting_different_cv_modes,
        test_plotting_different_denoising_types,
        test_plotting_with_custom_basis,
        test_plotting_edge_cases,
        test_regenerate_visualization,
        test_plotting_error_handling,
        test_cross_validation_scores_plotting
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        print()
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")
    
    print()
    print("=" * 60)
    print(f"PLOTTING TESTS SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All plotting tests passed!")
    else:
        print("⚠️  Some plotting tests failed - check output above for details")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_plotting_tests()
    exit(0 if success else 1)
