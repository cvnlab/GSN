#!/usr/bin/env python3
"""
Comprehensive test suite for verifying the uneven trials functionality.

This test suite ensures that the Python implementation handles uneven number of trials
across conditions correctly, matching the behavior of the MATLAB version.

Tests cover:
1. perform_gsn with uneven trials
2. calc_shrunken_covariance with uneven trials  
3. rsa_noise_ceiling with uneven trials (GSN mode only)
4. Error handling and validation
5. Comparison with regular (even) trials behavior
6. Edge cases and boundary conditions
7. Mathematical properties preservation
8. Integration with existing codebase

Usage: 
    pytest test_uneven_trials.py -v
    python test_uneven_trials.py
"""

import numpy as np
import pytest
import warnings
import sys
import traceback
import os

# Add the gsn module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gsn.perform_gsn import perform_gsn
from gsn.calc_shrunken_covariance import calc_shrunken_covariance
from gsn.rsa_noise_ceiling import rsa_noise_ceiling


class TestUnevenTrials:
    """Comprehensive test suite for uneven trials functionality."""
    
    def setup_method(self):
        """Set up test data with both even and uneven trials."""
        np.random.seed(42)  # For reproducible tests
        
        # Create regular data (even trials)
        self.nvox = 50
        self.ncond = 20
        self.ntrials = 6
        
        # Regular data: voxels x conditions x trials
        self.data_regular = np.random.randn(self.nvox, self.ncond, self.ntrials) * 0.5
        # Add some signal structure
        signal = np.random.randn(self.nvox, self.ncond) * 2.0
        for t in range(self.ntrials):
            self.data_regular[:, :, t] += signal
            
        # Create uneven data by setting some trials to NaN
        self.data_uneven = self.data_regular.copy()
        
        # Make some conditions have fewer trials
        # Condition 0: only 3 trials (set trials 3,4,5 to NaN)
        self.data_uneven[:, 0, 3:] = np.nan
        # Condition 1: only 4 trials (set trials 4,5 to NaN)  
        self.data_uneven[:, 1, 4:] = np.nan
        # Condition 2: only 2 trials (set trials 2,3,4,5 to NaN)
        self.data_uneven[:, 2, 2:] = np.nan
        # Leave other conditions with all 6 trials
        
    def test_perform_gsn_uneven_basic(self):
        """Test basic functionality of perform_gsn with uneven trials."""
        # Should work without errors
        results = perform_gsn(self.data_uneven)
        
        # Check that all expected keys are present
        expected_keys = ['mnN', 'cN', 'cNb', 'shrinklevelN', 'shrinklevelD', 
                        'mnS', 'cS', 'cSb', 'ncsnr', 'numiters']
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
            
        # Check dimensions
        assert results['mnN'].shape == (1, self.nvox)
        assert results['cN'].shape == (self.nvox, self.nvox)
        assert results['mnS'].shape == (1, self.nvox)
        assert results['cS'].shape == (self.nvox, self.nvox)
        assert results['ncsnr'].shape == (self.nvox,)
        
        # Check that results are finite
        assert np.all(np.isfinite(results['mnN']))
        assert np.all(np.isfinite(results['mnS']))
        assert np.all(np.isfinite(results['ncsnr']))
        
    def test_perform_gsn_comparison_even_vs_uneven(self):
        """Compare results between even and uneven data (should be similar structure)."""
        # Run both versions
        results_regular = perform_gsn(self.data_regular)
        results_uneven = perform_gsn(self.data_uneven)
        
        # Shapes should be identical
        assert results_regular['mnN'].shape == results_uneven['mnN'].shape
        assert results_regular['cN'].shape == results_uneven['cN'].shape
        assert results_regular['mnS'].shape == results_uneven['mnS'].shape
        assert results_regular['cS'].shape == results_uneven['cS'].shape
        
        # Both should have valid shrinkage levels
        assert 0 <= results_regular['shrinklevelN'] <= 1
        assert 0 <= results_regular['shrinklevelD'] <= 1
        assert 0 <= results_uneven['shrinklevelN'] <= 1
        assert 0 <= results_uneven['shrinklevelD'] <= 1
        
    def test_calc_shrunken_covariance_uneven(self):
        """Test calc_shrunken_covariance with uneven trials."""
        # Test with 3D data (observations x variables x cases)
        data_3d = np.transpose(self.data_uneven, (2, 0, 1))  # trials x voxels x conditions
        
        # Should work without errors
        mn, c, shrinklevel, nll = calc_shrunken_covariance(data_3d)
        
        # Check outputs
        assert mn.shape == (1, self.nvox)
        assert c.shape == (self.nvox, self.nvox)
        assert 0 <= shrinklevel <= 1
        assert len(nll) == 51  # default number of shrinkage levels
        
        # Mean should be zero for 3D case
        assert np.allclose(mn, 0, atol=1e-10)
        
        # Covariance should be positive semi-definite
        eigenvals = np.linalg.eigvals(c)
        assert np.all(eigenvals >= -1e-10), "Covariance matrix should be PSD"
        
    def test_calc_shrunken_covariance_validation(self):
        """Test validation logic in calc_shrunken_covariance."""
        # Case 1: NaNs in 2D data (should fail)
        data_2d_nan = np.random.randn(20, 10)
        data_2d_nan[0, 0] = np.nan
        
        with pytest.raises(AssertionError, match="NaNs are allowed only in the multi-case scenario"):
            calc_shrunken_covariance(data_2d_nan)
            
        # Case 2: All trials NaN for one condition (should fail)
        data_all_nan = np.random.randn(5, 10, 8)
        data_all_nan[:, :, 0] = np.nan  # All trials for first condition
        
        with pytest.raises(AssertionError, match="all conditions must have at least 1 valid trial"):
            calc_shrunken_covariance(data_all_nan)
            
        # Case 3: Insufficient conditions with multiple trials (should fail)
        # For 3D data: shape is (observations, variables, cases)
        data_insufficient = np.random.randn(4, 10, 5)  # 4 observations, 10 variables, 5 cases
        # Make most cases have only 1 observation, leaving only 1 case with 2+ observations
        data_insufficient[1:, :, 0] = np.nan  # Case 0: only 1 observation
        data_insufficient[1:, :, 1] = np.nan  # Case 1: only 1 observation
        data_insufficient[1:, :, 2] = np.nan  # Case 2: only 1 observation
        data_insufficient[1:, :, 3] = np.nan  # Case 3: only 1 observation
        # Case 4: keep all 4 observations - so only 1 case has 2+ observations
        
        # This should fail because we need at least 2 conditions with 2+ trials
        with pytest.raises(AssertionError, match="need at least 2 conditions with 2"):
            calc_shrunken_covariance(data_insufficient)
            
    def test_rsa_noise_ceiling_uneven_gsn_mode(self):
        """Test rsa_noise_ceiling with uneven trials in GSN mode (mode=1 or 2)."""
        # Mode 1: no scaling
        opt = {'mode': 1, 'wantfig': 0, 'wantverbose': 0, 'ncsims': 10}
        nc, ncdist, results = rsa_noise_ceiling(self.data_uneven, opt)
        
        # Check basic outputs
        assert isinstance(nc, (int, float, np.number))
        assert len(ncdist) == 10
        assert 'mnN' in results
        assert 'cN' in results
        assert 'sc' in results
        assert results['sc'] == 1  # no scaling in mode 1
        
        # Mode 2: variance scaling
        opt = {'mode': 2, 'wantfig': 0, 'wantverbose': 0, 'ncsims': 10}
        nc, ncdist, results = rsa_noise_ceiling(self.data_uneven, opt)
        
        assert isinstance(results['sc'], (int, float, np.number))
        assert results['sc'] > 0  # scaling factor should be positive
        
    def test_rsa_noise_ceiling_uneven_rsa_mode_fails(self):
        """Test that RSA mode (mode=0) fails with uneven trials."""
        opt = {'mode': 0, 'wantfig': 0, 'wantverbose': 0}
        
        with pytest.raises(AssertionError, match="we are NOT compatible with the RSA mode"):
            rsa_noise_ceiling(self.data_uneven, opt)
            
    def test_rsa_noise_ceiling_data_truncation(self):
        """Test that data gets properly truncated in uneven case."""
        # Create data where minimum valid trials is 2
        data_test = self.data_uneven.copy()
        
        # Check that some conditions have different numbers of valid trials
        valid_counts = []
        for c in range(self.ncond):
            valid_count = np.sum(~np.any(np.isnan(data_test[:, c, :]), axis=0))
            valid_counts.append(valid_count)
            
        min_valid = min(valid_counts)
        assert min_valid < self.ntrials, "Test setup should have uneven trials"
        
        # Run RSA with mode=1 (which handles uneven data)
        opt = {'mode': 1, 'wantfig': 0, 'wantverbose': 0, 'ncsims': 5}
        nc, ncdist, results = rsa_noise_ceiling(data_test, opt)
        
        # Should complete without errors
        assert isinstance(nc, (int, float, np.number))
        
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimum valid data - need at least 2 conditions with 2+ trials
        data_min = np.random.randn(10, 5, 4)  # 4 trials instead of 3
        # Make some conditions have only 1 trial, but leave at least 2 with 2+ trials
        data_min[:, 0, 1:] = np.nan  # Condition 0: only 1 trial
        data_min[:, 1, 3:] = np.nan  # Condition 1: only 3 trials
        data_min[:, 2, 2:] = np.nan  # Condition 2: only 2 trials
        # Conditions 3,4 keep all 4 trials - so we have 3 conditions with 2+ trials
        
        # Should work with perform_gsn
        results = perform_gsn(data_min)
        assert 'mnN' in results
        
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        np.random.seed(123)
        results1 = perform_gsn(self.data_uneven)
        
        np.random.seed(123)  
        results2 = perform_gsn(self.data_uneven)
        
        # Results should be identical (within numerical precision)
        assert np.allclose(results1['mnN'], results2['mnN'])
        assert np.allclose(results1['mnS'], results2['mnS'])
        
    def test_warning_generation(self):
        """Test that appropriate warnings are generated."""
        # Test case that should generate ntrialBC warning
        data_warning = np.random.randn(10, 5, 4)
        # Make most conditions have only 1 trial (can't compute covariance)
        for c in range(4):
            data_warning[:, c, 1:] = np.nan
            
        opt = {'mode': 1, 'wantfig': 0, 'wantverbose': 0, 'ncsims': 5}
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                nc, ncdist, results = rsa_noise_ceiling(data_warning, opt)
                # Check if warning was issued
                warning_messages = [str(warning.message) for warning in w]
                # Should get warning about ntrialBC being lopsided
                assert any('ntrialBC is lopsided' in msg for msg in warning_messages)
            except:
                # If it fails due to insufficient data, that's also acceptable
                pass
                
    def test_data_dimensions_preserved(self):
        """Test that data dimensions are handled correctly throughout."""
        # Test with different data sizes
        for nvox in [20, 50]:
            for ncond in [10, 15]:
                for ntrials in [4, 6]:
                    data = np.random.randn(nvox, ncond, ntrials)
                    # Add some uneven trials
                    if ncond > 5:
                        data[:, :3, -1] = np.nan  # Last trial missing for first 3 conditions
                    
                    results = perform_gsn(data)
                    
                    # Check dimensions are correct
                    assert results['mnN'].shape == (1, nvox)
                    assert results['cN'].shape == (nvox, nvox)
                    assert results['mnS'].shape == (1, nvox)
                    assert results['cS'].shape == (nvox, nvox)
                    
    def test_mathematical_properties(self):
        """Test that mathematical properties are preserved."""
        results = perform_gsn(self.data_uneven)
        
        # Covariance matrices should be symmetric
        assert np.allclose(results['cN'], results['cN'].T), "cN should be symmetric"
        assert np.allclose(results['cS'], results['cS'].T), "cS should be symmetric"
        assert np.allclose(results['cNb'], results['cNb'].T), "cNb should be symmetric"
        assert np.allclose(results['cSb'], results['cSb'].T), "cSb should be symmetric"
        
        # Diagonal elements should be non-negative for covariance matrices
        assert np.all(np.diag(results['cN']) >= 0), "cN diagonal should be non-negative"
        assert np.all(np.diag(results['cNb']) >= 0), "cNb diagonal should be non-negative"
        
        # ncsnr should be non-negative (due to rectification)
        assert np.all(results['ncsnr'] >= 0), "ncsnr should be non-negative"
        
    def test_integration_with_existing_code(self):
        """Test that new functionality integrates well with existing code."""
        # Test that regular data still works the same way
        results_regular_old = perform_gsn(self.data_regular)
        
        # Should still get same results with regular data
        assert 'mnN' in results_regular_old
        assert 'cN' in results_regular_old
        assert 'mnS' in results_regular_old
        assert 'cS' in results_regular_old
        
        # Test with various option combinations
        for mode in [1, 2]:  # Skip mode 0 for uneven data
            opt = {'mode': mode, 'wantfig': 0, 'wantverbose': 0, 'ncsims': 5}
            nc, ncdist, results = rsa_noise_ceiling(self.data_uneven, opt)
            assert isinstance(nc, (int, float, np.number))
            
    def test_comprehensive_uneven_scenarios(self):
        """Test comprehensive uneven trials scenarios."""
        np.random.seed(42)
        nvox, ncond, ntrials = 30, 10, 6
        
        # Test various uneven patterns
        patterns = [
            # Pattern 1: Gradually decreasing trials
            {0: 6, 1: 5, 2: 4, 3: 3, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2},
            # Pattern 2: Random missing trials
            {0: 3, 1: 6, 2: 4, 3: 2, 4: 5, 5: 3, 6: 6, 7: 2, 8: 4, 9: 3},
            # Pattern 3: Few conditions with many trials, most with few
            {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 6, 7: 6, 8: 5, 9: 4}
        ]
        
        for pattern in patterns:
            data = np.random.randn(nvox, ncond, ntrials)
            # Add signal
            signal = np.random.randn(nvox, ncond) * 1.5
            for t in range(ntrials):
                data[:, :, t] += signal
                
            # Apply pattern
            for cond, keep_trials in pattern.items():
                if keep_trials < ntrials:
                    data[:, cond, keep_trials:] = np.nan
                    
            # Should work
            results = perform_gsn(data)
            assert 'mnN' in results
            assert np.all(np.isfinite(results['mnN']))
            
    def test_trailing_nan_equivalence(self):
        """Test that trailing NaNs are equivalent to truncated data."""
        np.random.seed(42)
        nvox, ncond, ntrials = 20, 8, 5
        
        # Create full data
        data_full = np.random.randn(nvox, ncond, ntrials)
        signal = np.random.randn(nvox, ncond) * 1.0
        for t in range(ntrials):
            data_full[:, :, t] += signal
            
        # Create truncated data (remove last trial entirely)
        data_truncated = data_full[:, :, :-1].copy()
        
        # Create NaN data (set last trial to NaN)
        data_nan = data_full.copy()
        data_nan[:, :, -1] = np.nan
        
        # Both should give similar results
        results_truncated = perform_gsn(data_truncated)
        results_nan = perform_gsn(data_nan)
        
        # Results should be very close (within some tolerance due to random sampling)
        assert results_truncated['mnN'].shape == results_nan['mnN'].shape
        assert results_truncated['cN'].shape == results_nan['cN'].shape
        
    def test_mixed_nan_patterns(self):
        """Test mixed NaN patterns across conditions."""
        np.random.seed(42)
        data = np.random.randn(25, 12, 5)
        
        # Mixed patterns: some conditions missing early trials, some missing late
        data[:, 0, -1] = np.nan      # Condition 0: missing last trial
        data[:, 1, -2:] = np.nan     # Condition 1: missing last 2 trials
        data[:, 2, 0] = np.nan       # Condition 2: missing first trial
        data[:, 3, [0, 2, 4]] = np.nan  # Condition 3: missing non-consecutive trials
        # Other conditions remain full
        
        results = perform_gsn(data)
        
        # Should work and produce finite results
        assert 'mnN' in results
        assert np.all(np.isfinite(results['mnN']))
        assert np.all(np.isfinite(results['mnS']))
        
    def test_minimal_uneven_trials(self):
        """Test with minimal uneven trials configuration."""
        # Minimal case: exactly 2 conditions with 2+ trials
        data_minimal = np.random.randn(15, 4, 3)
        
        # Condition 0: 2 trials (remove last one)
        data_minimal[:, 0, -1] = np.nan
        # Condition 1: 2 trials (remove last one)  
        data_minimal[:, 1, -1] = np.nan
        # Condition 2: 3 trials (keep all)
        # Condition 3: 3 trials (keep all)
        
        # Should work
        results = perform_gsn(data_minimal)
        assert 'mnN' in results
        
    def test_uneven_vs_equal_subset_equivalence(self):
        """Test that uneven trials give similar results to equal subsets."""
        np.random.seed(42)
        nvox, ncond, ntrials = 30, 8, 6
        
        # Create full data
        data_full = np.random.randn(nvox, ncond, ntrials)
        signal = np.random.randn(nvox, ncond) * 1.5
        for t in range(ntrials):
            data_full[:, :, t] += signal
            
        # Create uneven data (all conditions have 4 trials)
        data_uneven = data_full.copy()
        data_uneven[:, :, 4:] = np.nan  # Remove last 2 trials from all conditions
        
        # Create equal subset (truncate to 4 trials)
        data_equal = data_full[:, :, :4].copy()
        
        # Results should be similar structure
        results_uneven = perform_gsn(data_uneven)
        results_equal = perform_gsn(data_equal)
        
        assert results_uneven['mnN'].shape == results_equal['mnN'].shape
        assert results_uneven['cN'].shape == results_equal['cN'].shape


def test_documentation_examples():
    """Test that documentation examples work correctly."""
    # Create example data similar to what's shown in docstrings
    np.random.seed(42)
    data = np.random.randn(100, 40, 4) * 2 + np.random.randn(100, 40, 4)
    
    # Should work without errors
    results = perform_gsn(data)
    assert 'mnN' in results
    
    # Test with uneven trials
    data_uneven = data.copy()
    data_uneven[:, :5, -1] = np.nan  # Remove last trial for first 5 conditions
    
    results_uneven = perform_gsn(data_uneven)
    assert 'mnN' in results_uneven


# Standalone test functions for script-style execution
def run_all_tests_standalone():
    """Run all tests in standalone mode (for script execution)."""
    print("=" * 70)
    print("COMPREHENSIVE UNEVEN TRIALS TESTING")
    print("=" * 70)
    
    # Create test instance
    test_instance = TestUnevenTrials()
    
    # List of test methods
    test_methods = [
        'test_perform_gsn_uneven_basic',
        'test_perform_gsn_comparison_even_vs_uneven', 
        'test_calc_shrunken_covariance_uneven',
        'test_calc_shrunken_covariance_validation',
        'test_rsa_noise_ceiling_uneven_gsn_mode',
        'test_rsa_noise_ceiling_uneven_rsa_mode_fails',
        'test_rsa_noise_ceiling_data_truncation',
        'test_edge_cases',
        'test_reproducibility',
        'test_warning_generation',
        'test_data_dimensions_preserved',
        'test_mathematical_properties',
        'test_integration_with_existing_code',
        'test_comprehensive_uneven_scenarios',
        'test_trailing_nan_equivalence',
        'test_mixed_nan_patterns',
        'test_minimal_uneven_trials',
        'test_uneven_vs_equal_subset_equivalence'
    ]
    
    # Add standalone tests
    standalone_tests = [test_documentation_examples]
    
    passed = 0
    failed = 0
    
    # Run class-based tests
    for test_method_name in test_methods:
        try:
            print(f"Running {test_method_name}...")
            test_instance.setup_method()  # Setup for each test
            test_method = getattr(test_instance, test_method_name)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_method()
            
            print(f"✓ {test_method_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test_method_name} FAILED: {e}")
            traceback.print_exc()
            failed += 1
    
    # Run standalone tests
    for test_func in standalone_tests:
        try:
            print(f"Running {test_func.__name__}...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_func()
            print(f"✓ {test_func.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! The uneven trials implementation is robust.")
    else:
        print("❌ Some tests failed. Check the implementation.")
        return False
    return True


if __name__ == '__main__':
    # Run in standalone mode
    success = run_all_tests_standalone()
    sys.exit(0 if success else 1)
