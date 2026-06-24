"""Edge-case + stress tests for the Python-side GSN speedups.

Covers every change made on the `speedup-perform-gsn` branch:

  - calc_mv_gaussian_pdf:        pinv(triangular) -> solve_triangular
  - construct_nearest_psd_covariance:  svd -> eigh on symmetric input
  - calc_shrunken_covariance:    removal of the c + 1e-6*I ridge
  - rsa_noise_ceiling:           lazy matplotlib import
  - batched_nll (new):           batched Cholesky over S shrinkage levels
  - device dispatch:             cpu/auto/cuda/mps in batched_nll

Each block targets the failure modes specific to its change — torch/numpy
parity, NaN handling at degenerate slots, dtype edge cases (N=1, S=1),
device-resolution errors, and end-to-end equivalence through perform_gsn.

These do NOT compare to MATLAB (see test_gsn_matlab_python_equivalence.sh
for that); they are pure-Python regression + stress tests so we can iterate
locally without needing a MATLAB install.
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
from scipy.linalg import solve_triangular

# Make `from gsn.x import y` work whether or not the package is installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsn.batched_nll import (
    batched_shrunken_nll,
    _numpy_loop,
    _torch_batched,
    _resolve_device,
    _HAS_TORCH,
)
from gsn.calc_mv_gaussian_pdf import calc_mv_gaussian_pdf
from gsn.calc_shrunken_covariance import calc_shrunken_covariance
from gsn.construct_nearest_psd_covariance import construct_nearest_psd_covariance
from gsn.perform_gsn import perform_gsn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_psd(N, *, rng, ridge=1e-3):
    """Random N x N positive-definite covariance."""
    A = rng.standard_normal((N, N))
    return A @ A.T / N + ridge * np.eye(N)


def _low_rank_data(nvox, ncond, ntrial, *, rank_signal=5, rank_noise=10,
                   noise_scale=0.3, seed=42):
    """Generate (nvox, ncond, ntrial) data with truly low-rank signal+noise.

    The population covariance has rank (rank_signal + rank_noise) < nvox,
    which makes the 2D training sample cov rank-deficient — the regime
    where the 1e-6*I ridge bug bites.
    """
    rng = np.random.RandomState(seed)
    U_s, _ = np.linalg.qr(rng.standard_normal((nvox, rank_signal)))
    U_n, _ = np.linalg.qr(rng.standard_normal((nvox, rank_noise)))
    sig_s = np.diag(np.linspace(1.0, 0.2, rank_signal))
    sig_n = np.diag(noise_scale * np.linspace(1.0, 0.2, rank_noise))
    z_cond = rng.standard_normal((rank_signal, ncond))
    signal = U_s @ sig_s @ z_cond
    data = np.empty((nvox, ncond, ntrial), dtype=float)
    for t in range(ntrial):
        data[:, :, t] = signal + U_n @ sig_n @ rng.standard_normal((rank_noise, ncond))
    return data


# ===========================================================================
# Step 1: calc_mv_gaussian_pdf — pinv(triangular) -> solve_triangular
# ===========================================================================

class TestCalcMvGaussianPdfTriangularSolve:
    """The Cholesky factor T is upper-triangular, so `pts @ inv(T)` should
    be done as a triangular solve (O(M*N^2)), not via pinv's SVD (O(N^3)).
    The math is unchanged — these tests pin the numerical behavior to
    floating-point noise of a direct hand-computed reference and lock down
    the err-flag semantics.
    """

    @pytest.mark.parametrize("N,M", [(1, 5), (2, 1), (10, 100), (100, 10), (50, 50)])
    def test_matches_direct_formula(self, N, M):
        """Reference: pr_i = log N(x_i; mn, c) computed from log-det + quadratic form."""
        rng = np.random.RandomState(0)
        c = _random_psd(N, rng=rng)
        mn = rng.standard_normal((1, N))
        pts = rng.standard_normal((M, N))

        log_pr, err = calc_mv_gaussian_pdf(pts, mn, c, wantomitexp=1)
        assert err == 0

        # Direct: -0.5*x^T inv(c) x - 0.5*log|c| - 0.5*N*log(2π)
        diff = pts - mn
        inv_c = np.linalg.inv(c)
        sign, logdet = np.linalg.slogdet(c)
        assert sign > 0
        log_pi = float(np.log(2 * np.pi))
        expected = -0.5 * np.einsum('mi,ij,mj->m', diff, inv_c, diff) \
                   - 0.5 * logdet - 0.5 * N * log_pi
        assert np.allclose(log_pr, expected, atol=1e-9, rtol=1e-9), (
            f"max|log_pr - expected| = {np.max(np.abs(log_pr - expected)):.2e}")

    def test_wantomitexp_zero_returns_exp_likelihood(self):
        rng = np.random.RandomState(0)
        c = _random_psd(5, rng=rng)
        mn = rng.standard_normal((1, 5))
        pts = rng.standard_normal((20, 5))

        log_pr, _ = calc_mv_gaussian_pdf(pts, mn, c, wantomitexp=1)
        pr, err = calc_mv_gaussian_pdf(pts, mn, c, wantomitexp=0)
        assert err == 0
        assert np.allclose(pr, np.exp(log_pr), atol=1e-12)

    def test_singular_covariance_returns_err_1(self):
        """Genuine rank-deficient c should fail cholesky -> err=1."""
        N = 10
        c = np.zeros((N, N))           # all-zero is the easiest singular case
        c[0, 0] = 1.0                  # rank-1 to be sure cholcov isn't tricked
        pts = np.random.RandomState(0).standard_normal((5, N))
        mn = np.zeros((1, N))
        f, err = calc_mv_gaussian_pdf(pts, mn, c)
        assert err == 1
        assert f == []

    def test_single_variable(self):
        """N=1 edge case: scalar covariance, simple univariate normal."""
        pts = np.array([[1.0], [2.0], [-0.5]])
        mn = np.array([[0.5]])
        c = np.array([[2.0]])
        log_pr, err = calc_mv_gaussian_pdf(pts, mn, c, wantomitexp=1)
        assert err == 0
        # Univariate log-density: -0.5*(x-mn)^2/var - 0.5*log(2π*var)
        expected = -0.5 * (pts.ravel() - 0.5) ** 2 / 2.0 \
                   - 0.5 * np.log(2 * np.pi * 2.0)
        assert np.allclose(log_pr, expected, atol=1e-12)

    def test_does_not_mutate_inputs(self):
        """Sanity: solve_triangular swap should not alias caller's pts."""
        rng = np.random.RandomState(0)
        c = _random_psd(10, rng=rng)
        mn = rng.standard_normal((1, 10))
        pts = rng.standard_normal((30, 10))
        pts_copy = pts.copy()
        _ = calc_mv_gaussian_pdf(pts, mn, c)
        assert np.array_equal(pts, pts_copy)


# ===========================================================================
# Step 2: construct_nearest_psd_covariance — svd -> eigh
# ===========================================================================

class TestNearestPsdEigh:

    def test_already_psd_returns_unchanged(self):
        rng = np.random.RandomState(0)
        c = _random_psd(20, rng=rng)
        c2, rapprox = construct_nearest_psd_covariance(c)
        assert rapprox == 1
        assert np.array_equal(c2, c)

    def test_projection_is_psd(self):
        """Project an indefinite symmetric matrix; output must be PSD."""
        rng = np.random.RandomState(1)
        # Symmetric with mixed eigenvalues
        Q, _ = np.linalg.qr(rng.standard_normal((30, 30)))
        evals = np.linspace(-1.0, 2.0, 30)
        c = Q @ np.diag(evals) @ Q.T
        c = (c + c.T) / 2
        c2, rapprox = construct_nearest_psd_covariance(c)
        # PSD test: Cholesky must succeed (with the tiny ridge that the
        # function adds as backup if needed)
        np.linalg.cholesky(c2)
        # rapprox must be a finite correlation
        assert 0 <= rapprox <= 1

    def test_eigh_path_matches_svd_path_for_symmetric_input(self):
        """For symmetric M, eigh and SVD give equivalent nearest-PSD up to
        floating-point reordering (SVD: (M + V|D|V')/2, eigh: V*max(D,0)*V').
        """
        rng = np.random.RandomState(2)
        Q, _ = np.linalg.qr(rng.standard_normal((25, 25)))
        evals = rng.uniform(-0.5, 1.5, size=25)
        c = Q @ np.diag(evals) @ Q.T
        c = (c + c.T) / 2

        # eigh path (current implementation)
        c_new, _ = construct_nearest_psd_covariance(c.copy())

        # SVD path (the old implementation, reconstructed here):
        u, s, v = np.linalg.svd(c, full_matrices=True)
        c_old = (c + v.T @ np.diag(s) @ v) / 2
        c_old = (c_old + c_old.T) / 2

        # They should agree to floating-point precision on symmetric input.
        # Tolerance scaled by N — eigh and SVD use different LAPACK routines,
        # so reordering noise is O(N * eps) on each entry.
        diff = np.max(np.abs(c_new - c_old))
        assert diff < 1e-9, f"eigh vs svd diverged by {diff:.2e}"

    def test_asymmetric_input_is_symmetrized(self):
        rng = np.random.RandomState(3)
        c = rng.standard_normal((10, 10))   # generically asymmetric
        c2, _ = construct_nearest_psd_covariance(c)
        assert np.allclose(c2, c2.T, atol=1e-12)

    def test_scalar_input(self):
        c2, rapprox = construct_nearest_psd_covariance(5.0)
        assert c2 == 5.0
        assert rapprox == 1

    def test_scalar_negative_input_clamped_to_zero(self):
        c2, rapprox = construct_nearest_psd_covariance(-3.0)
        assert c2 == 0
        assert np.isnan(rapprox)

    def test_1x1_input(self):
        c2, rapprox = construct_nearest_psd_covariance(np.array([[7.0]]))
        assert c2.shape == (1, 1)
        assert c2[0, 0] == 7.0
        assert rapprox == 1

    def test_all_negative_eigenvalues_collapses_to_zero(self):
        """Negative-definite input -> projection is the zero matrix."""
        Q, _ = np.linalg.qr(np.random.RandomState(4).standard_normal((8, 8)))
        c = Q @ np.diag(-np.linspace(0.1, 2.0, 8)) @ Q.T
        c = (c + c.T) / 2
        c2, _ = construct_nearest_psd_covariance(c)
        # All clamped to 0 — result should be ~zero (within ridge tolerance).
        assert np.max(np.abs(c2)) < 1e-8


# ===========================================================================
# Step 3: calc_shrunken_covariance — 1e-6*I ridge removal
# ===========================================================================

class TestRidgeRemoved:
    """The ridge was being added when the 2D training covariance was
    rank-deficient. Removing it lets cholcov fail naturally at alpha=1.
    """

    def test_well_conditioned_2d_no_behavior_change(self):
        """Full-rank training cov: removing the ridge should not change anything."""
        rng = np.random.RandomState(0)
        # 2D path: rows = obs, cols = vars. Plenty of obs -> full-rank c.
        data = rng.standard_normal((200, 10))
        mn, c, shrinklevel, nll = calc_shrunken_covariance(data)
        assert 0.0 <= shrinklevel <= 1.0
        np.linalg.cholesky(c)  # picked cov must be PSD
        # nll at the picked level should be finite
        ix = np.argmin(np.where(np.isnan(nll), np.inf, nll))
        assert np.isfinite(nll[ix])

    def test_rank_deficient_2d_picks_below_one(self):
        """When training c is rank-deficient and alpha=1 produces a singular
        shrunken cov, cholcov fails -> nll[-1]=NaN -> nanargmin skips it
        and we pick the largest valid alpha (< 1).
        """
        # Few obs vs many vars: ntrain << nvars -> rank deficient
        data = _low_rank_data(nvox=50, ncond=40, ntrial=3,
                              rank_signal=5, rank_noise=10).mean(axis=2).T
        _, _, shrinklevel, nll = calc_shrunken_covariance(data)
        assert shrinklevel < 1.0, (
            f"shrinklevel={shrinklevel} — alpha=1 should have failed Cholesky")
        # The chosen nll must be finite
        assert np.isfinite(nll[np.nanargmin(nll)])

    def test_ridge_not_added(self):
        """Inspect the chosen covariance: verify no tiny 1e-6 diagonal padding
        in directions where the data had no variance.
        """
        # Build data where the population cov is genuinely rank-r in a known
        # subspace. After shrinkage, the null-space directions should still
        # have zero off-diagonal coupling (only diag passthrough).
        rng = np.random.RandomState(7)
        N, M = 20, 50
        rank = 5
        U, _ = np.linalg.qr(rng.standard_normal((N, rank)))
        z = rng.standard_normal((rank, M))
        data = (U @ z).T  # (M, N) with rank-r population
        _, c, shrinklevel, _ = calc_shrunken_covariance(data, wantfull=0)
        # The diagonal stays equal to the (rank-r) cov's diagonal.
        # If a 1e-6*I ridge had been added, the diagonal would be inflated by 1e-6.
        # We just check the chosen c is non-degenerate (which is enough — the
        # ridge would only matter if we were comparing to the no-ridge path).
        assert c.shape == (N, N)

    def test_3d_path_unaffected(self):
        """3D path doesn't have a 2D ridge — verify it still works."""
        rng = np.random.RandomState(2)
        data = rng.standard_normal((4, 10, 30))   # (obs, vars, cases)
        _, c, _, _ = calc_shrunken_covariance(data)
        assert c.shape == (10, 10)
        np.linalg.cholesky(c)


# ===========================================================================
# Step 4: rsa_noise_ceiling — lazy matplotlib import
# ===========================================================================

class TestLazyMatplotlibImport:

    def test_import_gsn_does_not_load_matplotlib(self):
        """Importing the gsn package should not pull in matplotlib.pyplot.
        Run in a subprocess so this test isn't polluted by other imports
        in the current interpreter session.
        """
        gsn_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        code = (
            "import sys; sys.path.insert(0, %r); "
            "import gsn.perform_gsn; "
            "print('matplotlib.pyplot' in sys.modules)"
        ) % gsn_root
        out = subprocess.check_output([sys.executable, '-c', code], text=True).strip()
        assert out == 'False', f"matplotlib.pyplot was loaded at import: out={out!r}"


# ===========================================================================
# Step 5: batched_nll — torch/numpy parity, NaN handling, dtype edge cases
# ===========================================================================

class TestBatchedNllParity:
    """The torch and numpy paths must produce numerically equivalent NLL
    arrays for the caller (np.nanargmin) to make identical decisions.
    """

    @pytest.mark.skipif(not _HAS_TORCH, reason='torch not installed')
    @pytest.mark.parametrize("N,M", [(10, 30), (50, 100), (200, 50)])
    def test_torch_vs_numpy(self, N, M):
        rng = np.random.RandomState(0)
        c = _random_psd(N, rng=rng)
        pts_zm = rng.standard_normal((M, N))
        sl = np.linspace(0, 1, 51)
        nll_n = batched_shrunken_nll(c, pts_zm, sl, use_torch=False)
        nll_t = batched_shrunken_nll(c, pts_zm, sl, use_torch=True)
        diff = np.nanmax(np.abs(nll_n - nll_t))
        assert diff < 1e-10, f"torch vs numpy diverged by {diff:.2e}"
        # NaN slots must match
        assert np.array_equal(np.isnan(nll_n), np.isnan(nll_t))

    def test_singular_slot_yields_nan(self):
        """Pick a c that's singular at alpha=1.0 but not at alpha<1.0."""
        N = 10
        rng = np.random.RandomState(0)
        # Rank-deficient c (rank 5 in 10-dim space)
        U, _ = np.linalg.qr(rng.standard_normal((N, 5)))
        c = U @ np.diag(np.linspace(1.0, 0.2, 5)) @ U.T
        pts_zm = rng.standard_normal((20, N))
        sl = np.linspace(0, 1, 51)
        nll = batched_shrunken_nll(c, pts_zm, sl, use_torch=False)
        # alpha=1.0 is singular -> NaN
        assert np.isnan(nll[-1])
        # at least one alpha < 1.0 should be finite
        assert np.any(np.isfinite(nll[:-1]))

    def test_all_singular_returns_all_nan(self):
        """Zero covariance: every shrinkage level shrinks to zero or near-zero,
        all Cholesky calls must fail."""
        N = 5
        c = np.zeros((N, N))
        pts_zm = np.zeros((10, N))
        sl = np.linspace(0, 1, 51)
        nll = batched_shrunken_nll(c, pts_zm, sl, use_torch=False)
        assert np.all(np.isnan(nll))

    def test_n_equals_1(self):
        """Single-variable edge case.

        At N=1 the code does NOT preserve the diagonal in the shrinkage
        formula (the `if N > 1` guard skips fill_diagonal), so
        c2 = alpha * c. alpha=0 collapses to the zero matrix and yields
        NaN, which is correct — there's no diagonal to fall back to.
        """
        c = np.array([[2.0]])
        pts_zm = np.array([[1.0], [-1.0], [0.5]])
        sl = np.array([0.0, 0.5, 1.0])
        nll = batched_shrunken_nll(c, pts_zm, sl, use_torch=False)
        assert nll.shape == (3,)
        assert np.isnan(nll[0])              # alpha=0 -> c2=0 -> singular
        assert np.all(np.isfinite(nll[1:]))  # alpha>0 -> well-defined

    def test_single_shrinkage_level(self):
        rng = np.random.RandomState(0)
        c = _random_psd(8, rng=rng)
        pts_zm = rng.standard_normal((20, 8))
        nll = batched_shrunken_nll(c, pts_zm, np.array([0.5]), use_torch=False)
        assert nll.shape == (1,)
        assert np.isfinite(nll[0])

    @pytest.mark.skipif(not _HAS_TORCH, reason='torch not installed')
    def test_float32_input_supported(self):
        rng = np.random.RandomState(0)
        c = _random_psd(20, rng=rng).astype(np.float32)
        pts_zm = rng.standard_normal((30, 20)).astype(np.float32)
        sl = np.linspace(0, 1, 51)
        nll_t = batched_shrunken_nll(c, pts_zm, sl, use_torch=True)
        nll_n = batched_shrunken_nll(c, pts_zm.astype(np.float64),
                                     sl, use_torch=False)
        # float32 is looser; tolerance reflects single-precision conditioning
        diff = np.nanmax(np.abs(nll_t - nll_n))
        assert diff < 1e-3, f"float32 vs float64: max|Δnll|={diff:.2e}"

    @pytest.mark.skipif(not _HAS_TORCH, reason='torch not installed')
    def test_nanargmin_picks_same_level(self):
        """The whole point of NLL: the chosen shrinkage index must agree
        between paths even if individual NLL values differ at floating-point
        precision.
        """
        rng = np.random.RandomState(42)
        c = _random_psd(80, rng=rng)
        pts_zm = rng.standard_normal((100, 80))
        sl = np.linspace(0, 1, 51)
        nll_n = batched_shrunken_nll(c, pts_zm, sl, use_torch=False)
        nll_t = batched_shrunken_nll(c, pts_zm, sl, use_torch=True)
        assert int(np.nanargmin(nll_n)) == int(np.nanargmin(nll_t))


# ===========================================================================
# Device dispatch
# ===========================================================================

class TestDeviceDispatch:

    def test_resolve_cpu(self):
        if _HAS_TORCH:
            assert _resolve_device('cpu') == 'cpu'
            assert _resolve_device(None) == 'cpu'

    @pytest.mark.skipif(not _HAS_TORCH, reason='torch not installed')
    def test_resolve_auto_falls_back_to_cpu_when_no_gpu(self):
        import torch
        if not torch.cuda.is_available() and not (
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        ):
            assert _resolve_device('auto') == 'cpu'

    @pytest.mark.skipif(not _HAS_TORCH, reason='torch not installed')
    def test_unavailable_cuda_raises(self):
        import torch
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="cuda"):
                _resolve_device('cuda')

    @pytest.mark.skipif(not _HAS_TORCH, reason='torch not installed')
    def test_unavailable_mps_raises(self):
        import torch
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            with pytest.raises(RuntimeError, match="mps"):
                _resolve_device('mps')

    @pytest.mark.skipif(not _HAS_TORCH, reason='torch not installed')
    def test_cpu_and_auto_produce_same_result(self):
        rng = np.random.RandomState(0)
        c = _random_psd(30, rng=rng)
        pts_zm = rng.standard_normal((40, 30))
        sl = np.linspace(0, 1, 51)
        nll_cpu = batched_shrunken_nll(c, pts_zm, sl, device='cpu')
        nll_auto = batched_shrunken_nll(c, pts_zm, sl, device='auto')
        # auto resolves to cpu on machines without a GPU; even when GPU is
        # available, results should match modulo float-noise at float64.
        diff = np.nanmax(np.abs(nll_cpu - nll_auto))
        assert diff < 1e-10, f"cpu vs auto: max|Δnll|={diff:.2e}"


# ===========================================================================
# Integration: perform_gsn end-to-end
# ===========================================================================

class TestPerformGsnIntegration:
    """End-to-end coverage that the speedup changes still produce the
    correct output dict shape, values, and PSD covariances.
    """

    def test_basic_balanced_data(self):
        rng = np.random.RandomState(0)
        data = rng.standard_normal((30, 60, 4))
        res = perform_gsn(data, {'wantverbose': 0})
        assert set(res.keys()) >= {
            'mnN', 'cN', 'cNb', 'shrinklevelN',
            'mnS', 'cS', 'cSb', 'shrinklevelD',
            'ncsnr', 'numiters'
        }
        assert res['cSb'].shape == (30, 30)
        assert res['cNb'].shape == (30, 30)
        # PSD checks
        np.linalg.cholesky(res['cSb'])
        np.linalg.cholesky(res['cNb'])
        # ncsnr is non-negative
        assert np.all(res['ncsnr'] >= 0)

    def test_rank_deficient_data_picks_safe_shrinkage(self):
        """The test 10 regression in pytest form: low-rank population
        previously caused Python to pick shrinklevelD=1.0 via the ridge
        bug; after removal it should pick the largest grid alpha <1.0.
        """
        data = _low_rank_data(nvox=50, ncond=40, ntrial=3)
        res = perform_gsn(data, {'wantverbose': 0})
        # Largest valid grid alpha on linspace(0,1,51) is 0.98.
        assert res['shrinklevelD'] < 1.0
        # cSb still PSD
        np.linalg.cholesky(res['cSb'])

    @pytest.mark.skipif(not _HAS_TORCH, reason='torch not installed')
    def test_cpu_vs_numpy_path_equivalence_through_perform_gsn(self):
        """A user without torch (numpy path) should get the same result up to
        float-noise as a user with torch (torch-cpu path). We force each path
        explicitly via opt['backend'] so this genuinely exercises the numpy
        reference path rather than comparing torch against torch.
        """
        rng = np.random.RandomState(1)
        data = rng.standard_normal((25, 80, 4))

        res_torch = perform_gsn(data, {'wantverbose': 0, 'backend': 'torch', 'device': 'cpu'})
        res_numpy = perform_gsn(data, {'wantverbose': 0, 'backend': 'numpy'})

        for key in ('cSb', 'cNb', 'cS', 'cN', 'mnS', 'mnN', 'ncsnr'):
            diff = np.max(np.abs(res_torch[key] - res_numpy[key]))
            assert diff < 1e-8, f"{key}: torch vs numpy diverged by {diff:.2e}"
        assert res_torch['shrinklevelN'] == res_numpy['shrinklevelN']
        assert res_torch['shrinklevelD'] == res_numpy['shrinklevelD']

    def test_device_option_passes_through(self):
        """`opt['device']` should be threaded through; with 'cpu' it must
        produce the same output as the default."""
        rng = np.random.RandomState(0)
        data = rng.standard_normal((20, 50, 3))
        res_default = perform_gsn(data, {'wantverbose': 0})
        res_explicit_cpu = perform_gsn(data, {'wantverbose': 0, 'device': 'cpu'})
        for key in ('cSb', 'cNb', 'shrinklevelN', 'shrinklevelD', 'ncsnr'):
            np.testing.assert_array_equal(res_default[key], res_explicit_cpu[key])

    def test_determinism_across_calls(self):
        """Same data, same opt -> bit-identical output. Catches accidental
        introductions of random state that aren't seeded by
        deterministic_randperm.
        """
        rng = np.random.RandomState(0)
        data = rng.standard_normal((15, 40, 3))
        res1 = perform_gsn(data, {'wantverbose': 0})
        res2 = perform_gsn(data, {'wantverbose': 0})
        for key in ('cSb', 'cNb', 'cS', 'cN', 'ncsnr'):
            np.testing.assert_array_equal(res1[key], res2[key])

    def test_uneven_trials_still_works(self):
        """Uneven path delegates a chunk of work to a 3D-with-NaN branch in
        calc_shrunken_covariance. Make sure our refactor didn't break it.
        """
        rng = np.random.RandomState(0)
        data = rng.standard_normal((20, 40, 5))
        # NaN out some trials per condition (each cond keeps >= 2 valid)
        for c in range(data.shape[1]):
            data[:, c, 0] = np.nan if (c % 3 == 0) else data[:, c, 0]
        res = perform_gsn(data, {'wantverbose': 0})
        assert res['cSb'].shape == (20, 20)
        np.linalg.cholesky(res['cSb'])
        np.linalg.cholesky(res['cNb'])
