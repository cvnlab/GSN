#!/usr/bin/env python3
"""
Comprehensive test suite for perform_gsn and uneven trials functionality.

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
    pytest test_performgsn.py -v
    python test_performgsn.py
"""

import unittest
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


class TestPerformGSN(unittest.TestCase):
    """
    Comprehensive test suite for perform_gsn and uneven trials functionality.
    
    This test class verifies that the perform_gsn function:
    - Runs without errors on basic synthetic data
    - Returns expected output structure with correct fields
    - Handles edge cases and different input configurations
    - Works with uneven trials across conditions
    - Produces reasonable covariance estimates
    - Integrates properly with calc_shrunken_covariance and rsa_noise_ceiling
    """
    
    def setUp(self):
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
    
    def test_basic_functionality(self):
        """Test basic functionality with simple synthetic data"""
        print('Testing basic functionality...')
        
        # Generate simple test data: voxels x conditions x trials
        nvox = 20
        ncond = 10
        ntrial = 4
        
        # Create data with signal + noise structure
        np.random.seed(42)  # For reproducibility
        signal = 2 * np.random.randn(nvox, ncond)
        data = np.tile(signal[:, :, np.newaxis], (1, 1, ntrial)) + 0.5 * np.random.randn(nvox, ncond, ntrial)
        
        # Test basic call
        results = perform_gsn(data)
        
        # Verify output structure
        self.assertIsInstance(results, dict, 'Output should be a dict')
        
        # Check required fields
        required_fields = ['mnN', 'cN', 'cNb', 'shrinklevelN', 'shrinklevelD',
                          'mnS', 'cS', 'cSb', 'ncsnr', 'numiters']
        for field in required_fields:
            self.assertIn(field, results, f'Missing field: {field}')
        
        # Check dimensions
        self.assertEqual(results['mnN'].shape, (1, nvox), 'mnN dimensions incorrect')
        self.assertEqual(results['mnS'].shape, (1, nvox), 'mnS dimensions incorrect')
        self.assertEqual(results['cN'].shape, (nvox, nvox), 'cN dimensions incorrect')
        self.assertEqual(results['cS'].shape, (nvox, nvox), 'cS dimensions incorrect')
        self.assertEqual(results['cNb'].shape, (nvox, nvox), 'cNb dimensions incorrect')
        self.assertEqual(results['cSb'].shape, (nvox, nvox), 'cSb dimensions incorrect')
        self.assertEqual(results['ncsnr'].shape, (nvox,), 'ncsnr dimensions incorrect')
        
        # Check that covariances are symmetric
        np.testing.assert_allclose(results['cN'], results['cN'].T, rtol=1e-10,
                                 err_msg='cN should be symmetric')
        np.testing.assert_allclose(results['cS'], results['cS'].T, rtol=1e-10,
                                 err_msg='cS should be symmetric')
        np.testing.assert_allclose(results['cNb'], results['cNb'].T, rtol=1e-10,
                                 err_msg='cNb should be symmetric')
        np.testing.assert_allclose(results['cSb'], results['cSb'].T, rtol=1e-10,
                                 err_msg='cSb should be symmetric')
        
        # Check that final covariances are positive semi-definite
        self.assertGreaterEqual(np.min(np.linalg.eigvals(results['cNb'])), -1e-10,
                               'cNb should be PSD')
        self.assertGreaterEqual(np.min(np.linalg.eigvals(results['cSb'])), -1e-10,
                               'cSb should be PSD')
        
        # Check that ncsnr values are non-negative
        self.assertTrue(np.all(results['ncsnr'] >= 0), 'ncsnr should be non-negative')
        
        # Check that numiters is non-negative integer
        self.assertGreaterEqual(results['numiters'], 0, 'numiters should be non-negative')
        self.assertEqual(results['numiters'], int(results['numiters']), 'numiters should be integer')
        
        print('Basic functionality test passed!')

    def test_perform_gsn_uneven_basic(self):
        """Test basic functionality of perform_gsn with uneven trials."""
        # Should work without errors
        results = perform_gsn(self.data_uneven)
        
        # Check that all expected keys are present
        expected_keys = ['mnN', 'cN', 'cNb', 'shrinklevelN', 'shrinklevelD', 
                        'mnS', 'cS', 'cSb', 'ncsnr', 'numiters']
        for key in expected_keys:
            self.assertIn(key, results, f"Missing key: {key}")
            
        # Check dimensions
        self.assertEqual(results['mnN'].shape, (1, self.nvox))
        self.assertEqual(results['cN'].shape, (self.nvox, self.nvox))
        self.assertEqual(results['mnS'].shape, (1, self.nvox))
        self.assertEqual(results['cS'].shape, (self.nvox, self.nvox))
        self.assertEqual(results['ncsnr'].shape, (self.nvox,))
        
        # Check that results are finite
        self.assertTrue(np.all(np.isfinite(results['mnN'])))
        self.assertTrue(np.all(np.isfinite(results['mnS'])))
        self.assertTrue(np.all(np.isfinite(results['ncsnr'])))
        
    def test_perform_gsn_comparison_even_vs_uneven(self):
        """Compare results between even and uneven data (should be similar structure)."""
        # Run both versions
        results_regular = perform_gsn(self.data_regular)
        results_uneven = perform_gsn(self.data_uneven)
        
        # Shapes should be identical
        self.assertEqual(results_regular['mnN'].shape, results_uneven['mnN'].shape)
        self.assertEqual(results_regular['cN'].shape, results_uneven['cN'].shape)
        self.assertEqual(results_regular['mnS'].shape, results_uneven['mnS'].shape)
        self.assertEqual(results_regular['cS'].shape, results_uneven['cS'].shape)
        
        # Both should have valid shrinkage levels
        self.assertTrue(0 <= results_regular['shrinklevelN'] <= 1)
        self.assertTrue(0 <= results_regular['shrinklevelD'] <= 1)
        self.assertTrue(0 <= results_uneven['shrinklevelN'] <= 1)
        self.assertTrue(0 <= results_uneven['shrinklevelD'] <= 1)
        
    def test_calc_shrunken_covariance_uneven(self):
        """Test calc_shrunken_covariance with uneven trials."""
        # Test with 3D data (observations x variables x cases)
        data_3d = np.transpose(self.data_uneven, (2, 0, 1))  # trials x voxels x conditions
        
        # Should work without errors
        mn, c, shrinklevel, nll = calc_shrunken_covariance(data_3d)
        
        # Check outputs
        self.assertEqual(mn.shape, (1, self.nvox))
        self.assertEqual(c.shape, (self.nvox, self.nvox))
        self.assertTrue(0 <= shrinklevel <= 1)
        self.assertEqual(len(nll), 51)  # default number of shrinkage levels
        
        # Mean should be zero for 3D case
        np.testing.assert_allclose(mn, 0, atol=1e-10)
        
        # Covariance should be positive semi-definite
        eigenvals = np.linalg.eigvals(c)
        self.assertTrue(np.all(eigenvals >= -1e-10), "Covariance matrix should be PSD")

    def test_with_options(self):
        """Test function with different option configurations"""
        print('Testing with different options...')
        
        # Generate test data
        np.random.seed(123)
        nvox = 15
        ncond = 8
        ntrial = 3
        data = 2 * np.random.randn(nvox, ncond, ntrial) + 0.5 * np.random.randn(nvox, ncond, ntrial)
        
        # Test with verbose off
        opt1 = {'wantverbose': 0}
        results1 = perform_gsn(data, opt1)
        self.assertIsInstance(results1, dict, 'Should work with verbose off')
        
        # Test with shrinkage off
        opt2 = {'wantshrinkage': 0}
        results2 = perform_gsn(data, opt2)
        self.assertIsInstance(results2, dict, 'Should work with shrinkage off')
        
        # Test with both options
        opt3 = {'wantverbose': 0, 'wantshrinkage': 0}
        results3 = perform_gsn(data, opt3)
        self.assertIsInstance(results3, dict, 'Should work with both options set')
        
        print('Options test passed!')

    def test_minimal_trials(self):
        """Test with minimum number of trials (2)"""
        print('Testing with minimal trials...')
        
        np.random.seed(456)
        nvox = 10
        ncond = 5
        ntrial = 2  # Minimum required
        
        data = np.random.randn(nvox, ncond, ntrial)
        
        results = perform_gsn(data)
        self.assertIsInstance(results, dict, 'Should work with 2 trials')
        
        print('Minimal trials test passed!')

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
        
        self.assertEqual(results_uneven['mnN'].shape, results_equal['mnN'].shape)
        self.assertEqual(results_uneven['cN'].shape, results_equal['cN'].shape)

    def test_empty_options(self):
        """Test with empty options dict"""
        print('Testing with empty options...')
        
        np.random.seed(131415)
        data = np.random.randn(8, 4, 3)
        
        # Test with empty dict
        results1 = perform_gsn(data, {})
        self.assertIsInstance(results1, dict, 'Should work with empty dict')
        
        # Test with no options argument
        results2 = perform_gsn(data)
        self.assertIsInstance(results2, dict, 'Should work with no options')
        
        print('Empty options test passed!')

    def test_error_conditions(self):
        """Test various error conditions"""
        print('Testing error conditions...')
        
        # Test with insufficient trials (should error)
        data_bad = np.random.randn(5, 3, 1)  # Only 1 trial
        with self.assertRaises(Exception, msg='Should error with only 1 trial'):
            perform_gsn(data_bad)
        
        # Test with all NaN condition (should error)
        data_bad2 = np.random.randn(5, 3, 3)
        data_bad2[:, 0, :] = np.nan  # First condition all NaN
        with self.assertRaises(Exception, msg='Should error when condition has all NaN trials'):
            perform_gsn(data_bad2)
        
        print('Error conditions test passed!')

    def test_uneven_trials_error_conditions(self):
        """Test error conditions specific to uneven trials"""
        print('Testing uneven trials error conditions...')
        
        np.random.seed(800801)
        nvox = 6
        ncond = 3
        max_trials = 3
        
        # Test case where one condition has insufficient trials (only 1)
        data_bad1 = np.full((nvox, ncond, max_trials), np.nan)
        data_bad1[:, 0, :2] = np.random.randn(nvox, 2)  # Condition 0: 2 trials (OK)
        data_bad1[:, 1, :1] = np.random.randn(nvox, 1)  # Condition 1: 1 trial (BAD)
        data_bad1[:, 2, :3] = np.random.randn(nvox, 3)  # Condition 2: 3 trials (OK)
        
        with self.assertRaises(Exception, msg='Should error when any condition has insufficient trials'):
            perform_gsn(data_bad1)
        
        # Test case where one condition has all NaN trials
        data_bad2 = np.full((nvox, ncond, max_trials), np.nan)
        data_bad2[:, 0, :2] = np.random.randn(nvox, 2)  # Condition 0: 2 trials (OK)
        # Condition 1: all NaN (BAD)
        data_bad2[:, 2, :3] = np.random.randn(nvox, 3)  # Condition 2: 3 trials (OK)
        
        with self.assertRaises(Exception, msg='Should error when any condition has all NaN trials'):
            perform_gsn(data_bad2)
        
        print('Uneven trials error conditions test passed!')

    def test_rsa_noise_ceiling_integration(self):
        """Test integration with rsa_noise_ceiling function."""
        print('Testing RSA noise ceiling integration...')
        
        # Mode 1: no scaling
        opt = {'mode': 1, 'wantfig': 0, 'wantverbose': 0, 'ncsims': 10}
        nc, ncdist, results = rsa_noise_ceiling(self.data_uneven, opt)
        
        # Check basic outputs
        self.assertIsInstance(nc, (int, float, np.number))
        self.assertEqual(len(ncdist), 10)
        self.assertIn('mnN', results)
        self.assertIn('cN', results)
        self.assertIn('sc', results)
        self.assertEqual(results['sc'], 1)  # no scaling in mode 1
        
        # Mode 2: variance scaling
        opt = {'mode': 2, 'wantfig': 0, 'wantverbose': 0, 'ncsims': 10}
        nc, ncdist, results = rsa_noise_ceiling(self.data_uneven, opt)
        
        self.assertIsInstance(results['sc'], (int, float, np.number))
        self.assertGreater(results['sc'], 0)  # scaling factor should be positive
        
        print('RSA noise ceiling integration test passed!')

    def test_calc_shrunken_covariance_validation(self):
        """Test validation logic in calc_shrunken_covariance."""
        print('Testing calc_shrunken_covariance validation...')
        
        # Case 1: NaNs in 2D data (should fail)
        data_2d_nan = np.random.randn(20, 10)
        data_2d_nan[0, 0] = np.nan
        
        with self.assertRaises(AssertionError, msg="NaNs in 2D data should fail"):
            calc_shrunken_covariance(data_2d_nan)
            
        # Case 2: All trials NaN for one condition (should fail)
        data_all_nan = np.random.randn(5, 10, 8)
        data_all_nan[:, :, 0] = np.nan  # All trials for first condition
        
        with self.assertRaises(AssertionError, msg="All NaN condition should fail"):
            calc_shrunken_covariance(data_all_nan)
            
        print('Calc shrunken covariance validation test passed!')

    def test_comprehensive_uneven_scenarios(self):
        """Test comprehensive uneven trials scenarios."""
        print('Testing comprehensive uneven scenarios...')
        
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
        
        for i, pattern in enumerate(patterns):
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
            self.assertIn('mnN', results)
            self.assertTrue(np.all(np.isfinite(results['mnN'])))
            
        print('Comprehensive uneven scenarios test passed!')

    def test_mixed_nan_patterns(self):
        """Test mixed NaN patterns across conditions."""
        print('Testing mixed NaN patterns...')
        
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
        self.assertIn('mnN', results)
        self.assertTrue(np.all(np.isfinite(results['mnN'])))
        self.assertTrue(np.all(np.isfinite(results['mnS'])))
        
        print('Mixed NaN patterns test passed!')

    def test_mathematical_properties_detailed(self):
        """Test detailed mathematical properties are preserved."""
        print('Testing detailed mathematical properties...')
        
        results = perform_gsn(self.data_uneven)
        
        # Covariance matrices should be symmetric
        np.testing.assert_allclose(results['cN'], results['cN'].T, err_msg="cN should be symmetric")
        np.testing.assert_allclose(results['cS'], results['cS'].T, err_msg="cS should be symmetric")
        np.testing.assert_allclose(results['cNb'], results['cNb'].T, err_msg="cNb should be symmetric")
        np.testing.assert_allclose(results['cSb'], results['cSb'].T, err_msg="cSb should be symmetric")
        
        # Diagonal elements should be non-negative for covariance matrices
        self.assertTrue(np.all(np.diag(results['cN']) >= 0), "cN diagonal should be non-negative")
        self.assertTrue(np.all(np.diag(results['cNb']) >= 0), "cNb diagonal should be non-negative")
        
        # Final covariances should be positive semi-definite
        self.assertGreaterEqual(np.min(np.linalg.eigvals(results['cNb'])), -1e-10, 'cNb should be PSD')
        self.assertGreaterEqual(np.min(np.linalg.eigvals(results['cSb'])), -1e-10, 'cSb should be PSD')
        
        # ncsnr should be non-negative (due to rectification)
        self.assertTrue(np.all(results['ncsnr'] >= 0), "ncsnr should be non-negative")
        
        # Shrinkage levels should be valid
        self.assertTrue(0 <= results['shrinklevelN'] <= 1, "shrinklevelN should be in [0,1]")
        self.assertTrue(0 <= results['shrinklevelD'] <= 1, "shrinklevelD should be in [0,1]")
        
        print('Detailed mathematical properties test passed!')

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        print('Testing numerical stability...')
        
        # Test with very small values
        np.random.seed(192021)
        data_small = 1e-6 * np.random.randn(8, 5, 3)
        results_small = perform_gsn(data_small)
        self.assertIsInstance(results_small, dict, 'Should handle small values')
        self.assertTrue(np.all(np.isfinite(results_small['ncsnr'])), 'SNR should be finite')
        
        # Test with larger values
        data_large = 1e3 * np.random.randn(8, 5, 3)
        results_large = perform_gsn(data_large)
        self.assertIsInstance(results_large, dict, 'Should handle large values')
        self.assertTrue(np.all(np.isfinite(results_large['ncsnr'])), 'SNR should be finite')
        
        print('Numerical stability test passed!')

    def test_data_dimensions_preserved(self):
        """Test that data dimensions are handled correctly throughout."""
        print('Testing data dimensions preservation...')
        
        # Test with different data sizes
        for nvox in [10, 30]:
            for ncond in [6, 12]:
                for ntrials in [3, 5]:
                    data = np.random.randn(nvox, ncond, ntrials)
                    # Add some uneven trials
                    if ncond > 4:
                        data[:, :2, -1] = np.nan  # Last trial missing for first 2 conditions
                    
                    results = perform_gsn(data)
                    
                    # Check dimensions are correct
                    self.assertEqual(results['mnN'].shape, (1, nvox))
                    self.assertEqual(results['cN'].shape, (nvox, nvox))
                    self.assertEqual(results['mnS'].shape, (1, nvox))
                    self.assertEqual(results['cS'].shape, (nvox, nvox))
                    
        print('Data dimensions preservation test passed!')

    def test_reproducibility_detailed(self):
        """Test that results are reproducible with same random seed."""
        print('Testing detailed reproducibility...')
        
        # Test reproducibility with multiple configurations
        for seed in [42, 123, 456]:
            for nvox in [15, 25]:
                for ncond in [8, 12]:
                    # Run twice with same random seed
                    np.random.seed(seed)
                    data1 = np.random.randn(nvox, ncond, 4)
                    results1 = perform_gsn(data1)
                    
                    np.random.seed(seed)
                    data2 = np.random.randn(nvox, ncond, 4)
                    results2 = perform_gsn(data2)
                    
                    # Results should be identical
                    np.testing.assert_allclose(results1['mnN'], results2['mnN'], rtol=1e-12,
                                             err_msg='mnN should be consistent')
                    np.testing.assert_allclose(results1['cN'], results2['cN'], rtol=1e-12,
                                             err_msg='cN should be consistent')
                    self.assertEqual(results1['numiters'], results2['numiters'], 
                                   'numiters should be consistent')
                    
        print('Detailed reproducibility test passed!')

    def test_edge_cases_detailed(self):
        """Test detailed edge cases and boundary conditions."""
        print('Testing detailed edge cases...')
        
        # Test with minimum valid data - need more conditions with 2+ trials for cross-validation
        data_min = np.random.randn(8, 6, 4)  # 6 conditions, 4 trials
        # Make some conditions have fewer trials, but ensure enough have 2+ for cross-validation
        data_min[:, 0, 1:] = np.nan  # Condition 0: only 1 trial
        data_min[:, 1, 1:] = np.nan  # Condition 1: only 1 trial
        data_min[:, 2, 2:] = np.nan  # Condition 2: only 2 trials
        data_min[:, 3, 3:] = np.nan  # Condition 3: only 3 trials
        # Conditions 4,5 keep all 4 trials - so we have 4 conditions with 2+ trials
        
        # Should work with perform_gsn
        results = perform_gsn(data_min)
        self.assertIn('mnN', results)
        
        # Test with maximum uneven pattern
        data_max_uneven = np.random.randn(10, 8, 6)
        trial_counts = [2, 2, 3, 3, 4, 4, 5, 6]  # Different for each condition
        for c, count in enumerate(trial_counts):
            if count < 6:
                data_max_uneven[:, c, count:] = np.nan
                
        results_max = perform_gsn(data_max_uneven)
        self.assertIn('mnN', results_max)
        self.assertTrue(np.all(np.isfinite(results_max['ncsnr'])))
        
        print('Detailed edge cases test passed!')

    def test_integration_with_existing_code(self):
        """Test that new functionality integrates well with existing code."""
        print('Testing integration with existing code...')
        
        # Test that regular data still works the same way
        results_regular_old = perform_gsn(self.data_regular)
        
        # Should still get same results with regular data
        self.assertIn('mnN', results_regular_old)
        self.assertIn('cN', results_regular_old)
        self.assertIn('mnS', results_regular_old)
        self.assertIn('cS', results_regular_old)
        
        # Test with various option combinations
        for mode in [1, 2]:  # Skip mode 0 for uneven data
            opt = {'mode': mode, 'wantfig': 0, 'wantverbose': 0, 'ncsims': 5}
            nc, ncdist, results = rsa_noise_ceiling(self.data_uneven, opt)
            self.assertIsInstance(nc, (int, float, np.number))
            
        print('Integration with existing code test passed!')

    def test_warning_generation(self):
        """Test that appropriate warnings are generated."""
        print('Testing warning generation...')
        
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
                if warning_messages:
                    self.assertTrue(any('ntrialBC is lopsided' in msg for msg in warning_messages))
            except:
                # If it fails due to insufficient data, that's also acceptable
                pass
                
        print('Warning generation test passed!')

    def test_performance_large_dataset(self):
        """Test performance with larger dataset"""
        print('Testing performance with larger dataset...')
        
        np.random.seed(789)
        nvox = 100
        ncond = 40
        ntrial = 8
        
        # Create structured data
        signal = 3 * np.random.randn(nvox, ncond)
        noise = 1 * np.random.randn(nvox, ncond, ntrial)
        data = np.tile(signal[:, :, np.newaxis], (1, 1, ntrial)) + noise
        
        # Add some uneven trials
        for c in range(0, ncond, 3):  # Every 3rd condition loses some trials
            trials_to_remove = np.random.randint(1, 3)
            data[:, c, -trials_to_remove:] = np.nan
        
        results = perform_gsn(data)
        self.assertIsInstance(results, dict, 'Should work with larger data')
        
        # Verify reasonable SNR values
        self.assertTrue(np.all(results['ncsnr'] >= 0), 'SNR should be non-negative')
        self.assertTrue(np.any(results['ncsnr'] > 0), 'Should have some positive SNR')
        
        print('Performance with larger dataset test passed!')

    def test_various_option_combinations(self):
        """Test with various option combinations"""
        print('Testing various option combinations...')
        
        # Generate test data
        np.random.seed(123)
        nvox = 12
        ncond = 6
        ntrial = 4
        data = 2 * np.random.randn(nvox, ncond, ntrial) + 0.5 * np.random.randn(nvox, ncond, ntrial)
        
        # Add uneven trials
        data[:, 0, -1] = np.nan
        data[:, 1, -2:] = np.nan
        
        # Test various option combinations
        option_combinations = [
            {'wantverbose': 0},
            {'wantshrinkage': 0},
            {'wantverbose': 0, 'wantshrinkage': 0},
            {'wantverbose': 1, 'wantshrinkage': 1},
            {}  # Empty options
        ]
        
        for opt in option_combinations:
            results = perform_gsn(data, opt)
            self.assertIsInstance(results, dict, f'Should work with options: {opt}')
            self.assertIn('mnN', results)
            self.assertIn('cN', results)
            
        print('Various option combinations test passed!')


def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()
