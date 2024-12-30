import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gsn.gsn_denoise import gsn_denoise

def test_nearly_singular_data():
    """Test with nearly singular data where some dimensions are almost linearly dependent."""
    nunits, nconds, ntrials = 10, 5, 3
    # Create nearly linearly dependent data
    base = np.random.randn(nunits, nconds, ntrials)
    data = base.copy()
    data[1:4] = base[0:1] + 1e-10 * np.random.randn(3, nconds, ntrials)  # Make rows nearly identical
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    assert not np.any(np.isnan(results['denoiser']))  # Should handle near-singularity gracefully

def test_extreme_magnitude_differences():
    """Test with data having extreme differences in magnitudes across dimensions."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    data[0] *= 1e6  # Make first dimension much larger
    data[-1] *= 1e-6  # Make last dimension much smaller
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    assert not np.any(np.isnan(results['denoiser']))

def test_alternating_signal_noise():
    """Test with alternating strong signal and pure noise dimensions."""
    nunits, nconds, ntrials = 6, 5, 4
    data = np.zeros((nunits, nconds, ntrials))
    # Even indices: strong signal
    data[::2, :, :] = np.repeat(np.random.randn(nunits//2, nconds, 1), ntrials, axis=2)
    # Odd indices: pure noise
    data[1::2, :, :] = np.random.randn(nunits//2, nconds, ntrials)
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_identical_trials():
    """Test with identical trials (zero noise case)."""
    nunits, nconds = 5, 4
    single_trial = np.random.randn(nunits, nconds)
    data = np.repeat(single_trial[:, :, np.newaxis], 3, axis=2)
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    assert np.allclose(results['denoiseddata'], single_trial)

def test_zero_variance_dimensions():
    """Test with dimensions having zero variance."""
    nunits, nconds, ntrials = 5, 4, 3
    data = np.random.randn(nunits, nconds, ntrials)
    data[2] = 0  # Set middle dimension to zero
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    assert not np.any(np.isnan(results['denoiser']))

def test_anti_correlated_dimensions():
    """Test with perfectly anti-correlated dimensions."""
    nunits, nconds, ntrials = 4, 5, 3
    base = np.random.randn(1, nconds, ntrials)
    data = np.zeros((nunits, nconds, ntrials))
    data[0] = base
    data[1] = -base  # Perfect anti-correlation
    data[2:] = np.random.randn(nunits-2, nconds, ntrials)
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_periodic_noise():
    """Test with periodic noise patterns."""
    nunits, nconds, ntrials = 5, 20, 4
    data = np.random.randn(nunits, nconds, ntrials)
    # Add periodic noise
    periodic_noise = np.sin(np.linspace(0, 4*np.pi, nconds))
    data += periodic_noise[np.newaxis, :, np.newaxis]
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_sparse_signal():
    """Test with very sparse signal (mostly zeros with occasional spikes)."""
    nunits, nconds, ntrials = 5, 20, 3
    data = np.zeros((nunits, nconds, ntrials))
    # Add occasional spikes
    spike_positions = np.random.choice(nconds, size=2, replace=False)
    data[:, spike_positions, :] = np.random.randn(nunits, 2, ntrials)
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_structured_noise():
    """Test with structured noise that mimics signal."""
    nunits, nconds, ntrials = 5, 10, 4
    signal = np.random.randn(nunits, nconds, 1)
    structured_noise = np.random.randn(nunits, 1, ntrials)  # Noise correlated across conditions
    data = signal + structured_noise
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_extreme_cv_thresholds():
    """Test with extreme cross-validation threshold patterns."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    opt = {
        'cv_thresholds': np.array([1, nunits//2, nunits-1, nunits]),
        'cv_mode': 0
    }
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)

def test_all_signal_data():
    """Test with data that is pure signal (perfectly repeatable)."""
    nunits, nconds, ntrials = 5, 5, 3
    signal = np.random.randn(nunits, nconds, 1)
    data = np.repeat(signal, ntrials, axis=2)
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    # Should return identity-like denoiser
    assert np.allclose(results['denoiser'], np.eye(nunits), atol=1e-1)

def test_rank_deficient_data():
    """Test with rank-deficient data."""
    nunits, nconds, ntrials = 5, 3, 3  # More units than conditions
    data = np.random.randn(2, nconds, ntrials)  # Generate low-rank data
    data = np.vstack([data, np.zeros((nunits-2, nconds, ntrials))])  # Pad with zeros
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    assert not np.any(np.isnan(results['denoiser']))

def test_minimal_trials():
    """Test with minimum possible number of trials."""
    nunits, nconds = 5, 5
    data = np.random.randn(nunits, nconds, 2)  # Minimum 2 trials
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    assert not np.any(np.isnan(results['denoiser']))

def test_many_trials():
    """Test with unusually large number of trials."""
    nunits, nconds = 5, 5
    data = np.random.randn(nunits, nconds, 1000)  # Many trials
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_oscillating_signal():
    """Test with oscillating signal patterns."""
    nunits, nconds, ntrials = 5, 20, 3
    t = np.linspace(0, 4*np.pi, nconds)
    signal = np.sin(t)[np.newaxis, :, np.newaxis]
    noise = 0.1 * np.random.randn(nunits, nconds, ntrials)
    data = signal + noise
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_step_function_signal():
    """Test with step function signal."""
    nunits, nconds, ntrials = 5, 20, 3
    data = np.random.randn(nunits, nconds, ntrials) * 0.1
    data[:, nconds//2:, :] += 1  # Add step
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_exponential_signal():
    """Test with exponentially decaying signal."""
    nunits, nconds, ntrials = 5, 20, 3
    t = np.linspace(0, 5, nconds)
    signal = np.exp(-t)[np.newaxis, :, np.newaxis]
    noise = 0.1 * np.random.randn(nunits, nconds, ntrials)
    data = signal + noise
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_impulse_signal():
    """Test with impulse signal."""
    nunits, nconds, ntrials = 5, 20, 3
    data = np.random.randn(nunits, nconds, ntrials) * 0.1
    data[:, nconds//2, :] = 10  # Add impulse
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_magnitude_fraction_edge_cases():
    """Test edge cases of magnitude fraction thresholding."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with very small non-zero mag_frac
    opt = {'cv_mode': -1, 'mag_frac': 1e-10}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    
    # Test with mag_frac very close to 1
    opt = {'cv_mode': -1, 'mag_frac': 1 - 1e-10}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)

def test_custom_basis_edge_cases():
    """Test edge cases with custom basis."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with nearly orthogonal basis
    V = np.eye(nunits) + 1e-10 * np.random.randn(nunits, nunits)
    V, _ = np.linalg.qr(V)  # Make orthogonal
    results = gsn_denoise(data, V=V)
    assert results['denoiser'].shape == (nunits, nunits)
    
    # Test with minimal basis (single vector)
    V = np.random.randn(nunits, 1)
    V = V / np.linalg.norm(V)
    results = gsn_denoise(data, V=V)
    assert results['denoiser'].shape == (nunits, nunits)

def test_cross_validation_edge_cases():
    """Test edge cases in cross-validation."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with single threshold
    opt = {'cv_thresholds': [1]}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    
    # Test with all possible thresholds
    opt = {'cv_thresholds': np.arange(1, nunits + 1)}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)

def test_condition_number_variations():
    """Test with matrices having different condition numbers."""
    nunits, nconds, ntrials = 5, 5, 3
    
    # Well-conditioned case
    data = np.random.randn(nunits, nconds, ntrials)
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    
    # Poorly-conditioned case
    U = np.linalg.qr(np.random.randn(nunits, nunits))[0]
    s = np.logspace(0, 10, nunits)  # Exponentially decreasing singular values
    V = np.linalg.qr(np.random.randn(nconds, nconds))[0]
    data_poor = U @ np.diag(s)[:, :nconds] @ V.T
    data_poor = data_poor[:, :, np.newaxis] + 0.1 * np.random.randn(nunits, nconds, ntrials)
    
    results = gsn_denoise(data_poor)
    assert results['denoiser'].shape == (nunits, nunits)
    assert not np.any(np.isnan(results['denoiser']))

def test_basis_selection_edge_cases():
    """Test edge cases in basis selection."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test all basis selection modes with extreme data
    for V in [0, 1, 2, 3, 4]:
        # Make data nearly singular
        data_singular = data.copy()
        data_singular[1:] = data_singular[0:1] + 1e-10 * np.random.randn(nunits-1, nconds, ntrials)
        
        results = gsn_denoise(data_singular, V=V)
        assert results['denoiser'].shape == (nunits, nunits)
        assert not np.any(np.isnan(results['denoiser']))

def test_scoring_function_edge_cases():
    """Test edge cases with different scoring functions."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Test with constant scoring function
    opt = {'cv_scoring_fn': lambda x, y: np.zeros(x.shape[1])}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    
    # Test with scoring function that always returns infinities
    opt = {'cv_scoring_fn': lambda x, y: np.full(x.shape[1], np.inf)}
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)

def test_mixed_scale_data():
    """Test with data having mixed scales across units."""
    nunits, nconds, ntrials = 5, 5, 3
    data = np.random.randn(nunits, nconds, ntrials)
    scales = np.logspace(-3, 3, nunits)
    data = data * scales[:, np.newaxis, np.newaxis]
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    assert not np.any(np.isnan(results['denoiser']))

def test_correlation_structure():
    """Test with data having specific correlation structures."""
    nunits, nconds, ntrials = 5, 5, 3
    
    # Create correlated noise
    cov = np.zeros((nunits, nunits))
    for i in range(nunits):
        for j in range(nunits):
            cov[i,j] = 0.5**abs(i-j)  # Exponentially decaying correlations
    
    noise = np.random.multivariate_normal(np.zeros(nunits), cov, size=(nconds, ntrials))
    data = noise.transpose(2, 0, 1)  # Reshape to (nunits, nconds, ntrials)
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_outlier_trials():
    """Test with data containing outlier trials."""
    nunits, nconds, ntrials = 5, 5, 5
    data = np.random.randn(nunits, nconds, ntrials)
    # Add one outlier trial
    data[:, :, -1] = 10 * np.random.randn(nunits, nconds)
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    assert not np.any(np.isnan(results['denoiser']))

def test_block_diagonal_structure():
    """Test with block diagonal structure in the data."""
    nunits, nconds, ntrials = 6, 10, 5  # More conditions and trials for better estimation
    data = np.zeros((nunits, nconds, ntrials))
    
    # Create two truly independent blocks with distinct patterns
    # First block: sinusoidal patterns with higher frequency
    t = np.linspace(0, 6*np.pi, nconds)
    block1 = 5.0 * np.vstack([  # Increase signal amplitude
        np.sin(t),
        np.sin(2*t),
        np.sin(4*t)
    ])[:, :, np.newaxis]
    
    # Second block: exponential decay patterns with different rates
    t = np.linspace(0, 3, nconds)
    block2 = 5.0 * np.vstack([  # Increase signal amplitude
        np.exp(-t),
        np.exp(-2*t),
        np.exp(-4*t)
    ])[:, :, np.newaxis]
    
    # Repeat patterns across trials and add very small noise
    data[:3] = np.repeat(block1, ntrials, axis=2) + 0.01 * np.random.randn(3, nconds, ntrials)
    data[3:] = np.repeat(block2, ntrials, axis=2) + 0.01 * np.random.randn(3, nconds, ntrials)
    
    # Test only in population mode with explicit thresholds
    opt = {
        'cv_threshold_per': 'population',
        'cv_thresholds': [3, 6]  # Only test full block sizes
    }
    results = gsn_denoise(data, opt=opt)
    assert results['denoiser'].shape == (nunits, nunits)
    
    # Check if block structure is approximately preserved (only for population mode)
    # Use a more lenient tolerance since small numerical deviations are expected
    denoiser_upper = results['denoiser'][:3, 3:]
    denoiser_lower = results['denoiser'][3:, :3]
    assert np.all(np.abs(denoiser_upper) < 0.05), "Upper off-diagonal block should be close to zero"
    assert np.all(np.abs(denoiser_lower) < 0.05), "Lower off-diagonal block should be close to zero"
    
    # For unit mode, just check basic properties
    opt = {'cv_threshold_per': 'unit'}
    results_unit = gsn_denoise(data, opt=opt)
    assert results_unit['denoiser'].shape == (nunits, nunits)
    assert not np.any(np.isnan(results_unit['denoiser']))

def test_repeated_dimensions():
    """Test with repeated dimensions in the data."""
    nunits, nconds, ntrials = 5, 5, 3
    base = np.random.randn(2, nconds, ntrials)
    data = np.vstack([base, base, base[0:1]])  # Create repeated dimensions
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    assert not np.any(np.isnan(results['denoiser']))

def test_increasing_noise_levels():
    """Test with systematically increasing noise levels across trials."""
    nunits, nconds, ntrials = 5, 5, 5
    signal = np.random.randn(nunits, nconds, 1)
    noise_scales = np.linspace(0.1, 2.0, ntrials)
    noise = np.random.randn(nunits, nconds, ntrials) * noise_scales[np.newaxis, np.newaxis, :]
    data = signal + noise
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_frequency_varying_signal():
    """Test with signal having different frequency components."""
    nunits, nconds, ntrials = 5, 50, 3
    t = np.linspace(0, 10*np.pi, nconds)
    # Create signals with different frequencies
    signals = np.vstack([
        np.sin(t),
        np.sin(2*t),
        np.sin(4*t),
        np.sin(8*t),
        np.sin(16*t)
    ])[:, :, np.newaxis]
    noise = 0.1 * np.random.randn(nunits, nconds, ntrials)
    data = signals + noise
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_sparse_noise():
    """Test with sparse noise (occasional large deviations)."""
    nunits, nconds, ntrials = 5, 5, 10
    data = np.random.randn(nunits, nconds, ntrials) * 0.1
    # Add sparse large noise
    for trial in range(ntrials):
        idx = np.random.randint(0, nunits)
        cond = np.random.randint(0, nconds)
        data[idx, cond, trial] += 10 * np.random.randn()
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_locally_correlated_noise():
    """Test with noise that is locally correlated in condition space."""
    nunits, nconds, ntrials = 5, 20, 3
    data = np.zeros((nunits, nconds, ntrials))
    
    # Create locally correlated noise
    for trial in range(ntrials):
        for i in range(nconds):
            # Each condition is correlated with its neighbors
            start = max(0, i-2)
            end = min(nconds, i+3)
            data[:, i, trial] = np.mean(np.random.randn(nunits, end-start), axis=1)
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_hierarchical_signal():
    """Test with hierarchically structured signal."""
    nunits, nconds, ntrials = 8, 5, 3
    # Create hierarchical structure: pairs of units share similar patterns
    base_patterns = np.random.randn(4, nconds, 1)  # 4 base patterns
    data = np.vstack([np.repeat(base_patterns[i:i+1], 2, axis=0) for i in range(4)])
    # Repeat signal across trials before adding noise
    data = np.repeat(data, ntrials, axis=2)
    data += 0.1 * np.random.randn(nunits, nconds, ntrials)
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_missing_data_simulation():
    """Test with simulated missing data (zeros in specific locations)."""
    nunits, nconds, ntrials = 5, 5, 5
    data = np.random.randn(nunits, nconds, ntrials)
    # Simulate missing data with zeros
    mask = np.random.rand(nunits, nconds, ntrials) < 0.2
    data[mask] = 0
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    assert not np.any(np.isnan(results['denoiser']))

def test_permuted_trials():
    """Test with trials that are permuted differently for each unit."""
    nunits, nconds, ntrials = 5, 5, 5
    data = np.random.randn(nunits, nconds, ntrials)
    
    # Permute trials differently for each unit
    for i in range(nunits):
        data[i] = data[i, :, np.random.permutation(ntrials)]
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_condition_specific_noise():
    """Test with condition-specific noise levels."""
    nunits, nconds, ntrials = 5, 5, 5
    signal = np.random.randn(nunits, nconds, 1)
    # Different noise levels for each condition
    noise_scales = np.linspace(0.1, 2.0, nconds)
    noise = np.random.randn(nunits, nconds, ntrials) * noise_scales[np.newaxis, :, np.newaxis]
    data = signal + noise
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_unit_specific_noise():
    """Test with unit-specific noise levels."""
    nunits, nconds, ntrials = 5, 5, 5
    signal = np.random.randn(nunits, nconds, 1)
    # Different noise levels for each unit
    noise_scales = np.linspace(0.1, 2.0, nunits)
    noise = np.random.randn(nunits, nconds, ntrials) * noise_scales[:, np.newaxis, np.newaxis]
    data = signal + noise
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_nonlinear_relationships():
    """Test with nonlinear relationships between units."""
    nunits, nconds, ntrials = 5, 20, 3
    base = np.random.randn(1, nconds, 1)
    data = np.zeros((nunits, nconds, ntrials))
    # Create nonlinear relationships
    data[0] = base
    data[1] = np.square(base)
    data[2] = np.exp(base)
    data[3] = np.sin(base)
    data[4] = np.sign(base)
    data += 0.1 * np.random.randn(nunits, nconds, ntrials)
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)

def test_multimodal_noise():
    """Test with multimodal noise distributions."""
    nunits, nconds, ntrials = 5, 5, 10
    signal = np.random.randn(nunits, nconds, 1)
    # Create bimodal noise
    noise1 = np.random.normal(-2, 0.5, (nunits, nconds, ntrials//2))
    noise2 = np.random.normal(2, 0.5, (nunits, nconds, ntrials - ntrials//2))
    noise = np.concatenate([noise1, noise2], axis=2)
    data = signal + noise
    
    results = gsn_denoise(data)
    assert results['denoiser'].shape == (nunits, nunits)
    assert not np.any(np.isnan(results['denoiser'])) 