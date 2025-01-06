#!/usr/bin/env python3

"""
Test script to compare Python and MATLAB implementations of GSN denoising.
This script:
1. Generates synthetic data using simulate_data.py
2. Runs the data through the Python GSN implementation
3. Saves both the data and results for comparison with MATLAB
"""

import os
import numpy as np
from scipy.io import savemat
import sys

# Add parent directory to path to import gsn modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsn.simulate_data import generate_data
from gsn.gsn_denoise import gsn_denoise

def generate_test_cases():
    """Generate a variety of test cases with different parameters."""
    
    # Basic test case
    nvox, ncond, ntrial = 10, 20, 5
    train_data, _, ground_truth = generate_data(
        nvox=nvox, 
        ncond=ncond, 
        ntrial=ntrial,
        signal_decay=1.0,
        noise_decay=1.0,
        noise_multiplier=1.0,
        align_alpha=0.0,
        align_k=0,
        random_seed=42
    )
    
    # Run through Python implementation with different settings
    test_cases = []
    
    # Test case 1: Default settings
    results = gsn_denoise(train_data)
    test_cases.append({
        'name': 'default',
        'data': train_data,
        'ground_truth': ground_truth,
        'results': results,
        'params': {'V': None, 'options': {}}
    })
    
    # Test case 2: Population-level thresholding
    results = gsn_denoise(train_data, opt={'cv_threshold_per': 'population'})
    test_cases.append({
        'name': 'population_threshold',
        'data': train_data,
        'ground_truth': ground_truth,
        'results': results,
        'params': {'V': None, 'options': {'cv_threshold_per': 'population'}}
    })
    
    # Test case 3: Magnitude thresholding
    results = gsn_denoise(train_data, opt={'cv_mode': -1})
    test_cases.append({
        'name': 'magnitude_threshold',
        'data': train_data,
        'ground_truth': ground_truth,
        'results': results,
        'params': {'V': None, 'options': {'cv_mode': -1}}
    })
    
    # Test case 4: Custom basis
    nvox = train_data.shape[0]
    rng = np.random.RandomState(42)
    Q, _ = np.linalg.qr(rng.randn(nvox, nvox))
    results = gsn_denoise(train_data, V=Q[:, :5])
    test_cases.append({
        'name': 'custom_basis',
        'data': train_data,
        'ground_truth': ground_truth,
        'results': results,
        'params': {'V': Q[:, :5], 'options': {}}
    })
    
    return test_cases

def save_test_cases(test_cases, output_dir='test_data'):
    """Save test cases to .mat files for MATLAB comparison."""
    os.makedirs(output_dir, exist_ok=True)
    
    for case in test_cases:
        # Convert None values to empty arrays for MATLAB compatibility
        def convert_none_to_empty(x):
            return np.array([]) if x is None else x
        
        # Prepare data for MATLAB
        matlab_data = {
            'data': case['data'],
            'ground_truth_signal': case['ground_truth']['signal'],
            'ground_truth_signal_cov': case['ground_truth']['signal_cov'],
            'ground_truth_noise_cov': case['ground_truth']['noise_cov'],
            'python_denoiser': convert_none_to_empty(case['results'].get('denoiser')),
            'python_denoiseddata': convert_none_to_empty(case['results'].get('denoiseddata')),
            'python_cv_scores': convert_none_to_empty(case['results'].get('cv_scores')),
            'python_best_threshold': convert_none_to_empty(case['results'].get('best_threshold')),
            'python_fullbasis': convert_none_to_empty(case['results'].get('fullbasis')),
            'python_signalsubspace': convert_none_to_empty(case['results'].get('signalsubspace')),
            'python_dimreduce': convert_none_to_empty(case['results'].get('dimreduce')),
            'python_mags': convert_none_to_empty(case['results'].get('mags')),
            'python_dimsretained': convert_none_to_empty(case['results'].get('dimsretained'))
        }
        
        # Add parameters if they exist
        if case['params']['V'] is not None:
            matlab_data['V'] = case['params']['V']
        if case['params']['options']:
            for key, value in case['params']['options'].items():
                matlab_data[f'opt_{key}'] = value
        
        # Save to .mat file
        savemat(os.path.join(output_dir, f'test_case_{case["name"]}.mat'), matlab_data)

def main():
    """Main function to generate and save test cases."""
    test_cases = generate_test_cases()
    save_test_cases(test_cases)
    print("Test cases generated and saved.")

if __name__ == '__main__':
    main() 