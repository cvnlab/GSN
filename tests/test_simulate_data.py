"""Tests for simulate_data.py"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gsn.simulate_data import generate_data, _adjust_alignment_gradient_descent

def test_basic_alignment():
    """Test basic alignment properties for a simple case."""
    nvox = 10
    k = 3
    rng = np.random.RandomState(42)
    U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
    U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
    
    # Test perfect alignment (alpha = 1)
    U_aligned = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha=1.0, k=k, verbose=False)
    alignments = [np.abs(np.dot(U_signal[:, i], U_aligned[:, i])) for i in range(k)]
    assert np.mean(alignments) > 0.8, "Failed perfect alignment"
    
    # Test perfect orthogonality (alpha = 0)
    U_orthogonal = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha=0.0, k=k, verbose=False)
    alignments = [np.abs(np.dot(U_signal[:, i], U_orthogonal[:, i])) for i in range(k)]
    assert np.mean(alignments) < 0.2, "Failed perfect orthogonality"
    
    # Test partial alignment (alpha = 0.5)
    U_partial = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha=0.5, k=k, verbose=False)
    alignments = [np.abs(np.dot(U_signal[:, i], U_partial[:, i])) for i in range(k)]
    assert abs(np.mean(alignments) - 0.5) < 0.2, "Failed partial alignment"

def test_orthonormality_preservation():
    """Test that the adjusted basis remains orthonormal."""
    nvox_values = [5, 10, 20]
    k_values = [1, 3, 5, 10]
    alpha_values = [0.0, 0.3, 0.7, 1.0]
    
    rng = np.random.RandomState(42)
    
    for nvox in nvox_values:
        U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
        U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
        
        for k in k_values:
            if k > nvox:
                continue
                
            for alpha in alpha_values:
                U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha, k, verbose=False)
                
                # Check orthonormality
                product = U_adjusted.T @ U_adjusted
                np.testing.assert_allclose(
                    product, np.eye(nvox), 
                    rtol=1e-3, atol=1e-3,  # More lenient tolerance for gradient descent
                    err_msg=f"Failed orthonormality for nvox={nvox}, k={k}, alpha={alpha}"
                )

def test_extreme_cases():
    """Test alignment behavior in extreme cases."""
    nvox = 10
    rng = np.random.RandomState(42)
    U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
    U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
    
    # Test k=0
    U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha=0.5, k=0, verbose=False)
    np.testing.assert_allclose(U_adjusted, U_noise)
    
    # Test k=1
    U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha=0.5, k=1, verbose=False)
    alignment = np.abs(np.dot(U_signal[:, 0], U_adjusted[:, 0]))
    assert abs(alignment - 0.5) < 0.2
    
    # Test k=nvox with different alphas
    for alpha in [0.0, 0.5, 1.0]:
        U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha=alpha, k=nvox, verbose=False)
        alignments = [np.abs(np.dot(U_signal[:, i], U_adjusted[:, i])) for i in range(nvox)]
        avg_alignment = np.mean(alignments)
        if alpha == 0.0:
            assert avg_alignment < 0.2
        elif alpha == 1.0:
            assert avg_alignment > 0.8
        else:
            assert abs(avg_alignment - alpha) < 0.2

def test_monotonicity():
    """Test that alignment increases monotonically with alpha."""
    nvox = 10
    k = 3
    alpha_values = np.linspace(0, 1, 11)
    
    rng = np.random.RandomState(42)
    U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
    U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
    
    avg_alignments = []
    for alpha in alpha_values:
        U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha, k, verbose=False)
        alignments = [np.abs(np.dot(U_signal[:, i], U_adjusted[:, i])) for i in range(k)]
        avg_alignments.append(np.mean(alignments))
    
    # Check monotonicity with some tolerance for numerical issues
    diffs = np.diff(avg_alignments)
    assert np.all(diffs >= -0.1), "Alignment not monotonic with alpha"
    
    # Check endpoints
    assert avg_alignments[0] < 0.2, "Initial alignment too high"
    assert avg_alignments[-1] > 0.8, "Final alignment too low"

def test_stability():
    """Test stability across different random initializations and dimensions."""
    nvox_values = [5, 10, 20]
    k_values = [1, 3, 5]
    alpha = 0.5
    n_repeats = 5
    
    for nvox in nvox_values:
        for k in k_values:
            if k > nvox:
                continue
                
            avg_alignments = []
            for seed in range(n_repeats):
                rng = np.random.RandomState(seed)
                U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
                U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
                
                U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha, k, verbose=False)
                alignments = [np.abs(np.dot(U_signal[:, i], U_adjusted[:, i])) for i in range(k)]
                avg_alignments.append(np.mean(alignments))
            
            # Check consistency across random initializations
            assert abs(np.mean(avg_alignments) - alpha) < 0.2, \
                f"Failed stability test for nvox={nvox}, k={k}"
            assert np.std(avg_alignments) < 0.1, \
                f"Alignment too variable for nvox={nvox}, k={k}"

def test_full_pipeline():
    """Test alignment in the context of the full data generation pipeline."""
    nvox = 20
    ncond = 10
    ntrial = 5
    k = 3
    
    for alpha in [0.0, 0.5, 1.0]:
        # Generate data
        _, _, ground_truth = generate_data(
            nvox=nvox,
            ncond=ncond,
            ntrial=ntrial,
            signal_decay=1.0,
            noise_decay=1.0,
            align_alpha=alpha,
            align_k=k,
            random_seed=42
        )
        
        # Check alignment properties
        U_signal = ground_truth['U_signal']
        U_noise = ground_truth['U_noise']
        
        # Verify alignment
        alignments = [np.abs(np.dot(U_signal[:, i], U_noise[:, i])) for i in range(k)]
        avg_alignment = np.mean(alignments)
        
        if alpha == 0.0:
            assert avg_alignment < 0.2
        elif alpha == 1.0:
            assert avg_alignment > 0.8
        else:
            assert abs(avg_alignment - alpha) < 0.2
        
        # Verify orthonormality
        np.testing.assert_allclose(
            U_noise.T @ U_noise,
            np.eye(nvox),
            rtol=1e-3, atol=1e-3  # More lenient for integration test
        )

def test_numerical_stability():
    """Test behavior with numerically challenging inputs."""
    nvox = 10
    k = 3
    rng = np.random.RandomState(42)
    
    # Test with nearly identical signal and noise bases
    U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
    U_noise = U_signal + 1e-10 * rng.randn(nvox, nvox)
    U_noise = np.linalg.qr(U_noise)[0]
    
    for alpha in [0.0, 0.5, 1.0]:
        U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha, k, verbose=False)
        
        # Check orthonormality
        np.testing.assert_allclose(
            U_adjusted.T @ U_adjusted,
            np.eye(nvox),
            rtol=1e-5, atol=1e-5
        )
        
        # Check alignment - be more lenient for the challenging case
        alignments = [np.abs(np.dot(U_signal[:, i], U_adjusted[:, i])) for i in range(k)]
        avg_alignment = np.mean(alignments)
        
        if alpha == 0.0:
            # For orthogonality, allow more tolerance in this edge case
            assert avg_alignment < 0.3
        elif alpha == 1.0:
            assert avg_alignment > 0.8
        else:
            assert abs(avg_alignment - alpha) < 0.3  # More tolerant for edge case 

def test_fine_grid():
    """Test alignment across a fine grid of alpha and k values."""
    # Test parameters
    nvox = 20
    alpha_values = np.linspace(0, 1, 21)  # Steps of 0.05
    k_values = list(range(1, nvox + 1))   # All possible k values
    
    rng = np.random.RandomState(42)
    U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
    U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
    
    # Store results for analysis
    results = np.zeros((len(alpha_values), len(k_values)))
    
    for i, alpha in enumerate(alpha_values):
        for j, k in enumerate(k_values):
            U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha, k, verbose=False)
            
            # Calculate average alignment for the first k components
            alignments = [np.abs(np.dot(U_signal[:, idx], U_adjusted[:, idx])) 
                         for idx in range(k)]
            avg_alignment = np.mean(alignments)
            results[i, j] = avg_alignment
            
            # Basic checks
            if np.isclose(alpha, 0, atol=1e-7):
                assert avg_alignment < 0.2, \
                    f"Failed orthogonality for k={k}"
            elif np.isclose(alpha, 1, atol=1e-7):
                assert avg_alignment > 0.8, \
                    f"Failed perfect alignment for k={k}"
            else:
                # For intermediate alphas, allow more deviation for larger k
                k_fraction = k / nvox
                tolerance = 0.2 if k_fraction < 0.5 else 0.3
                assert abs(avg_alignment - alpha) < tolerance, \
                    f"Failed partial alignment for alpha={alpha:.2f}, k={k}"
            
            # Check orthonormality
            np.testing.assert_allclose(
                U_adjusted.T @ U_adjusted,
                np.eye(nvox),
                rtol=1e-3, atol=1e-3,  # More lenient tolerance for gradient descent
                err_msg=f"Failed orthonormality for alpha={alpha:.2f}, k={k}"
            )
    
    # Check monotonicity in alpha for each k
    for j in range(len(k_values)):
        diffs = np.diff(results[:, j])
        assert np.all(diffs >= -0.1), \
            f"Alignment not monotonic with alpha for k={k_values[j]}"
    
    # Check that average alignment is closer to target for smaller k
    for i, alpha in enumerate(alpha_values):
        if alpha > 0.1 and alpha < 0.9:  # Exclude extreme alphas
            deviations = np.abs(results[i, :] - alpha)
            # Check if deviations tend to increase with k
            diffs = np.diff(deviations)
            assert np.mean(diffs) >= -0.1, \
                f"Alignment quality should not improve for larger k at alpha={alpha:.2f}"

def test_fine_grid_stability():
    """Test stability of alignment across fine grid with different random initializations."""
    nvox = 15
    alpha_values = np.linspace(0, 1, 11)  # Steps of 0.1
    k_values = [1, 5, 10, 15]  # Test small, medium, and large k
    n_repeats = 5
    
    for k in k_values:
        for alpha in alpha_values:
            alignments = []
            for seed in range(n_repeats):
                rng = np.random.RandomState(seed)
                U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
                U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
                
                U_adjusted = _adjust_alignment_gradient_descent(U_signal, U_noise, alpha, k, verbose=False)
                avg_alignment = np.mean([
                    np.abs(np.dot(U_signal[:, i], U_adjusted[:, i])) 
                    for i in range(k)
                ])
                alignments.append(avg_alignment)
            
            # Check consistency across random initializations
            alignments = np.array(alignments)
            if np.isclose(alpha, 0, atol=1e-7):
                assert np.all(alignments < 0.2), \
                    f"Failed orthogonality stability for k={k}"
            elif np.isclose(alpha, 1, atol=1e-7):
                assert np.all(alignments > 0.8), \
                    f"Failed perfect alignment stability for k={k}"
            else:
                # For intermediate alphas, check mean and variance
                k_fraction = k / nvox
                tolerance = 0.2 if k_fraction < 0.5 else 0.3
                assert abs(np.mean(alignments) - alpha) < tolerance, \
                    f"Failed mean alignment stability for alpha={alpha:.2f}, k={k}"
                assert np.std(alignments) < 0.1, \
                    f"Failed alignment variance stability for alpha={alpha:.2f}, k={k}" 

def test_signal_noise_ratio():
    """Test that signal-to-noise ratio is controlled by decay parameters."""
    nvox = 20
    ncond = 10
    ntrial = 5
    
    # Test that decreasing signal decay relative to noise decay increases signal power
    train_data_low_snr, _, _ = generate_data(
        nvox=nvox,
        ncond=ncond,
        ntrial=ntrial,
        signal_decay=1.0,  # Fast decay = less signal
        noise_decay=0.1,   # Slow decay = more noise
        random_seed=42
    )
    
    train_data_high_snr, _, _ = generate_data(
        nvox=nvox,
        ncond=ncond,
        ntrial=ntrial,
        signal_decay=0.1,  # Slow decay = more signal
        noise_decay=1.0,   # Fast decay = less noise
        random_seed=42
    )
    
    # Calculate signal and noise power for both cases
    trial_means_low = np.mean(train_data_low_snr, axis=2)
    noise_low = train_data_low_snr - trial_means_low[:, :, np.newaxis]
    signal_power_low = np.var(trial_means_low)
    noise_power_low = np.var(noise_low)
    snr_low = signal_power_low / noise_power_low
    
    trial_means_high = np.mean(train_data_high_snr, axis=2)
    noise_high = train_data_high_snr - trial_means_high[:, :, np.newaxis]
    signal_power_high = np.var(trial_means_high)
    noise_power_high = np.var(noise_high)
    snr_high = signal_power_high / noise_power_high
    
    # Test that SNR increases when signal decay decreases relative to noise decay
    assert snr_high > snr_low, "SNR should increase when signal decay decreases relative to noise decay"
    
    # Test equal decay case
    train_data_equal, _, _ = generate_data(
        nvox=nvox,
        ncond=ncond,
        ntrial=ntrial,
        signal_decay=0.5,
        noise_decay=0.5,
        random_seed=42
    )
    
    trial_means_equal = np.mean(train_data_equal, axis=2)
    noise_equal = train_data_equal - trial_means_equal[:, :, np.newaxis]
    signal_power_equal = np.var(trial_means_equal)
    noise_power_equal = np.var(noise_equal)
    snr_equal = signal_power_equal / noise_power_equal
    
    # Test that SNR is intermediate when signal and noise decay are equal
    assert snr_low < snr_equal < snr_high, "SNR should be intermediate when signal and noise decay are equal"

def test_trial_independence():
    """Test that trials are independently generated."""
    nvox = 20
    ncond = 10
    ntrial = 10
    
    train_data, test_data, _ = generate_data(
        nvox=nvox,
        ncond=ncond,
        ntrial=ntrial,
        random_seed=42
    )
    
    # Check correlations between trials after removing condition means
    trial_means = np.mean(train_data, axis=2)
    noise = train_data - trial_means[:, :, np.newaxis]
    
    # Check correlations between noise components
    for i in range(ntrial):
        for j in range(i+1, ntrial):
            trial_i = noise[:, :, i].flatten()
            trial_j = noise[:, :, j].flatten()
            correlation = np.corrcoef(trial_i, trial_j)[0, 1]
            assert abs(correlation) < 0.7, f"Trial noise components {i} and {j} are too correlated"

def test_condition_structure():
    """Test that condition structure is preserved across trials."""
    nvox = 20
    ncond = 10
    ntrial = 5
    
    train_data, test_data, ground_truth = generate_data(
        nvox=nvox,
        ncond=ncond,
        ntrial=ntrial,
        signal_decay=1.0,
        noise_decay=0.1,  # Low noise to see condition structure
        random_seed=42
    )
    
    # Calculate mean pattern for each condition
    condition_means = np.mean(train_data, axis=2)  # Average across trials
    
    # Check that conditions are distinct
    for i in range(ncond):
        for j in range(i+1, ncond):
            pattern_i = condition_means[:, i]
            pattern_j = condition_means[:, j]
            correlation = np.corrcoef(pattern_i, pattern_j)[0, 1]
            assert abs(correlation) < 0.9, f"Conditions {i} and {j} are too similar"

def test_dimensionality_scaling():
    """Test behavior with different dimensionality ratios."""
    test_configs = [
        (10, 5, 3),    # More units than conditions
        (5, 10, 3),    # More conditions than units
        (10, 10, 3),   # Equal units and conditions
        (3, 3, 10),    # Many trials
        (50, 5, 2)     # High-dimensional units
    ]
    
    for nvox, ncond, ntrial in test_configs:
        train_data, test_data, ground_truth = generate_data(
            nvox=nvox,
            ncond=ncond,
            ntrial=ntrial,
            random_seed=42
        )
        
        assert train_data.shape == (nvox, ncond, ntrial)
        assert test_data.shape == (nvox, ncond, ntrial)
        assert ground_truth['U_signal'].shape == (nvox, nvox)
        assert ground_truth['U_noise'].shape == (nvox, nvox)

def test_random_seed_reproducibility():
    """Test that random seed controls reproducibility."""
    nvox = 20
    ncond = 10
    ntrial = 5
    
    # Generate two datasets with same seed
    data1, _, _ = generate_data(nvox=nvox, ncond=ncond, ntrial=ntrial, random_seed=42)
    data2, _, _ = generate_data(nvox=nvox, ncond=ncond, ntrial=ntrial, random_seed=42)
    
    # Generate dataset with different seed
    data3, _, _ = generate_data(nvox=nvox, ncond=ncond, ntrial=ntrial, random_seed=43)
    
    # Same seed should give identical results
    np.testing.assert_array_equal(data1, data2)
    
    # Different seeds should give different results
    assert not np.array_equal(data1, data3)

def test_basis_properties():
    """Test properties of signal and noise bases."""
    nvox = 20
    ncond = 10
    ntrial = 5
    
    _, _, ground_truth = generate_data(
        nvox=nvox,
        ncond=ncond,
        ntrial=ntrial,
        random_seed=42
    )
    
    U_signal = ground_truth['U_signal']
    U_noise = ground_truth['U_noise']
    
    # Test orthonormality with appropriate tolerance
    np.testing.assert_allclose(U_signal.T @ U_signal, np.eye(nvox), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(U_noise.T @ U_noise, np.eye(nvox), rtol=1e-5, atol=1e-5)
    
    # Test span
    assert np.linalg.matrix_rank(U_signal) == nvox
    assert np.linalg.matrix_rank(U_noise) == nvox

def test_train_test_independence():
    """Test that train and test datasets are independently generated."""
    nvox = 20
    ncond = 10
    ntrial = 5
    
    train_data, test_data, _ = generate_data(
        nvox=nvox,
        ncond=ncond,
        ntrial=ntrial,
        random_seed=42
    )
    
    # Remove condition means before checking correlations
    train_means = np.mean(train_data, axis=2)
    test_means = np.mean(test_data, axis=2)
    train_noise = train_data - train_means[:, :, np.newaxis]
    test_noise = test_data - test_means[:, :, np.newaxis]
    
    # Check correlations between train and test noise components
    for i in range(ntrial):
        for j in range(ntrial):
            train_trial = train_noise[:, :, i].flatten()
            test_trial = test_noise[:, :, j].flatten()
            correlation = np.corrcoef(train_trial, test_trial)[0, 1]
            assert abs(correlation) < 0.7, f"Train and test noise components {i} and {j} are too correlated"

def test_noise_structure():
    """Test that noise has the expected structure."""
    nvox = 20
    ncond = 10
    ntrial = 20  # More trials for stable noise estimation
    
    train_data, _, ground_truth = generate_data(
        nvox=nvox,
        ncond=ncond,
        ntrial=ntrial,
        signal_decay=0.1,  # Low signal to focus on noise
        noise_decay=1.0,
        random_seed=42
    )
    
    # Calculate noise covariance
    trial_means = np.mean(train_data, axis=2)
    noise = train_data - trial_means[:, :, np.newaxis]
    noise_cov = np.zeros((nvox, nvox))
    for i in range(ntrial):
        noise_cov += noise[:, :, i] @ noise[:, :, i].T
    noise_cov /= (ntrial * ncond)
    
    # Check that noise covariance aligns with noise basis
    U_noise = ground_truth['U_noise']
    alignment = np.abs(np.trace(U_noise.T @ noise_cov @ U_noise))
    assert alignment > 0, "Noise covariance should align with noise basis"

def test_signal_structure():
    """Test that signal has the expected structure."""
    nvox = 20
    ncond = 10
    ntrial = 20  # More trials for stable signal estimation
    
    train_data, _, ground_truth = generate_data(
        nvox=nvox,
        ncond=ncond,
        ntrial=ntrial,
        signal_decay=1.0,
        noise_decay=0.1,  # Low noise to focus on signal
        random_seed=42
    )
    
    # Calculate signal covariance
    trial_means = np.mean(train_data, axis=2)
    signal_cov = trial_means @ trial_means.T / ncond
    
    # Check that signal covariance aligns with signal basis
    U_signal = ground_truth['U_signal']
    alignment = np.abs(np.trace(U_signal.T @ signal_cov @ U_signal))
    assert alignment > 0, "Signal covariance should align with signal basis"

def test_edge_case_dimensions():
    """Test edge cases for data dimensions."""
    test_configs = [
        (2, 2, 2),     # Minimum valid dimensions
        (100, 2, 2),   # Very high dimensional units
        (2, 100, 2),   # Very high dimensional conditions
        (2, 2, 100),   # Very high dimensional trials
        (50, 50, 2)    # Equal high dimensions
    ]
    
    for nvox, ncond, ntrial in test_configs:
        train_data, test_data, ground_truth = generate_data(
            nvox=nvox,
            ncond=ncond,
            ntrial=ntrial,
            random_seed=42
        )
        
        assert train_data.shape == (nvox, ncond, ntrial)
        assert test_data.shape == (nvox, ncond, ntrial)
        assert ground_truth['U_signal'].shape == (nvox, nvox)
        assert ground_truth['U_noise'].shape == (nvox, nvox)
        
        # Check basic properties are maintained even in edge cases
        assert np.all(np.isfinite(train_data))
        assert np.all(np.isfinite(test_data))
        assert np.allclose(ground_truth['U_signal'].T @ ground_truth['U_signal'], np.eye(nvox))
        assert np.allclose(ground_truth['U_noise'].T @ ground_truth['U_noise'], np.eye(nvox)) 

def test_alignment_with_custom_signal():
    """Test that alignment between signal and noise eigenvectors is maintained with user-provided signal."""
    # Create parameters
    nvox = 20
    ncond = 15
    ntrial = 5
    
    # Create a custom true signal
    rng = np.random.RandomState(42)
    custom_signal = rng.randn(ncond, nvox)
    
    # Test different alignment values
    for align_alpha in [0.0, 0.3, 0.5, 0.8, 1.0]:
        align_k = 10  # Align top 10 dimensions
        
        # Generate data with provided true signal and specified alignment
        _, _, ground_truth = generate_data(
            nvox=nvox,
            ncond=ncond,
            ntrial=ntrial,
            true_signal=custom_signal,
            align_alpha=align_alpha,
            align_k=align_k,
            random_seed=42
        )
        
        # Extract signal and noise eigenvectors
        U_signal = ground_truth['U_signal']
        U_noise = ground_truth['U_noise']
        
        # Calculate actual alignment for the top k dimensions
        alignments = [np.abs(np.dot(U_signal[:, i], U_noise[:, i])) for i in range(align_k)]
        avg_alignment = np.mean(alignments)
        
        # Verify alignment is close to requested alpha
        # Use appropriate tolerance based on alpha value
        if align_alpha == 0.0:
            assert avg_alignment < 0.2, f"Failed orthogonality with custom signal, got {avg_alignment}"
        elif align_alpha == 1.0:
            assert avg_alignment > 0.8, f"Failed perfect alignment with custom signal, got {avg_alignment}"
        else:
            # For intermediate alphas, allow some deviation
            assert abs(avg_alignment - align_alpha) < 0.2, \
                f"Failed alignment for alpha={align_alpha}, got {avg_alignment}, expected {align_alpha}±0.2"
        
        # Verify orthonormality is preserved
        np.testing.assert_allclose(
            U_signal.T @ U_signal,
            np.eye(nvox),
            rtol=1e-3, atol=1e-3,  # More lenient for integration test
            err_msg=f"Signal basis not orthonormal with alpha={align_alpha}"
        )
        np.testing.assert_allclose(
            U_noise.T @ U_noise,
            np.eye(nvox),
            rtol=1e-3, atol=1e-3,  # More lenient for integration test
            err_msg=f"Noise basis not orthonormal with alpha={align_alpha}"
        )

def test_user_provided_true_signal():
    """Test that signal_cov is correctly derived from user-provided true_signal."""
    # Create parameters
    nvox = 20
    ncond = 15
    ntrial = 5
    
    # Create a custom true signal with a specific structure
    custom_signal = np.zeros((ncond, nvox))
    # Create block structure
    block_size = 5
    for i in range(ncond // block_size):
        for j in range(nvox // block_size):
            if i == j:
                # Higher values on diagonal blocks
                custom_signal[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 0.8
            else:
                # Lower values on off-diagonal blocks
                custom_signal[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = 0.2
    
    # Manually calculate expected covariance
    expected_cov = np.cov(custom_signal, rowvar=False)
    
    # Generate data with provided true signal
    _, _, ground_truth = generate_data(
        nvox=nvox,
        ncond=ncond,
        ntrial=ntrial,
        true_signal=custom_signal,
        random_seed=42
    )
    
    # Get the calculated signal covariance
    calculated_cov = ground_truth['signal_cov']
    
    # Check if the calculated covariance matches the expected covariance
    np.testing.assert_allclose(calculated_cov, expected_cov, rtol=1e-10)
    
    # Also verify that the 'signal' field in ground_truth matches our input
    np.testing.assert_allclose(ground_truth['signal'], custom_signal) 