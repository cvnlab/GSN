"""Tests for simulate_data.py"""

import numpy as np
from gsn.simulate_data import generate_data, _adjust_alignment

def test_basic_alignment():
    """Test basic alignment properties for a simple case."""
    nvox = 10
    k = 3
    rng = np.random.RandomState(42)
    U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
    U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
    
    # Test perfect alignment (alpha = 1)
    U_aligned = _adjust_alignment(U_signal, U_noise, alpha=1.0, k=k)
    alignments = [np.abs(np.dot(U_signal[:, i], U_aligned[:, i])) for i in range(k)]
    assert np.mean(alignments) > 0.8, "Failed perfect alignment"
    
    # Test perfect orthogonality (alpha = 0)
    U_orthogonal = _adjust_alignment(U_signal, U_noise, alpha=0.0, k=k)
    alignments = [np.abs(np.dot(U_signal[:, i], U_orthogonal[:, i])) for i in range(k)]
    assert np.mean(alignments) < 0.2, "Failed perfect orthogonality"
    
    # Test partial alignment (alpha = 0.5)
    U_partial = _adjust_alignment(U_signal, U_noise, alpha=0.5, k=k)
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
                U_adjusted = _adjust_alignment(U_signal, U_noise, alpha, k)
                
                # Check orthonormality
                product = U_adjusted.T @ U_adjusted
                np.testing.assert_allclose(
                    product, np.eye(nvox), 
                    rtol=1e-5, atol=1e-5,
                    err_msg=f"Failed orthonormality for nvox={nvox}, k={k}, alpha={alpha}"
                )

def test_extreme_cases():
    """Test alignment behavior in extreme cases."""
    nvox = 10
    rng = np.random.RandomState(42)
    U_signal = np.linalg.qr(rng.randn(nvox, nvox))[0]
    U_noise = np.linalg.qr(rng.randn(nvox, nvox))[0]
    
    # Test k=0
    U_adjusted = _adjust_alignment(U_signal, U_noise, alpha=0.5, k=0)
    np.testing.assert_allclose(U_adjusted, U_noise)
    
    # Test k=1
    U_adjusted = _adjust_alignment(U_signal, U_noise, alpha=0.5, k=1)
    alignment = np.abs(np.dot(U_signal[:, 0], U_adjusted[:, 0]))
    assert abs(alignment - 0.5) < 0.2
    
    # Test k=nvox with different alphas
    for alpha in [0.0, 0.5, 1.0]:
        U_adjusted = _adjust_alignment(U_signal, U_noise, alpha=alpha, k=nvox)
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
        U_adjusted = _adjust_alignment(U_signal, U_noise, alpha, k)
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
                
                U_adjusted = _adjust_alignment(U_signal, U_noise, alpha, k)
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
            rtol=1e-5, atol=1e-5
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
        U_adjusted = _adjust_alignment(U_signal, U_noise, alpha, k)
        
        # Check orthonormality
        np.testing.assert_allclose(
            U_adjusted.T @ U_adjusted,
            np.eye(nvox),
            rtol=1e-5, atol=1e-5
        )
        
        # Check alignment
        alignments = [np.abs(np.dot(U_signal[:, i], U_adjusted[:, i])) for i in range(k)]
        avg_alignment = np.mean(alignments)
        
        if alpha == 0.0:
            assert avg_alignment < 0.2
        elif alpha == 1.0:
            assert avg_alignment > 0.8
        else:
            assert abs(avg_alignment - alpha) < 0.2 

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
            U_adjusted = _adjust_alignment(U_signal, U_noise, alpha, k)
            
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
                rtol=1e-5, atol=1e-5,
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
                
                U_adjusted = _adjust_alignment(U_signal, U_noise, alpha, k)
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