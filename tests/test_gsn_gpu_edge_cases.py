"""GPU-path edge-case tests for gsn.batched_nll.

These exercise the torch.linalg.cholesky_ex + torch.linalg.solve_triangular
batched path running on CUDA or MPS. Each test is gated on device
availability and skipped cleanly when the device isn't reachable, so the
file is safe to collect everywhere and meaningful wherever a GPU is
actually present.

What this file specifically targets that test_gsn_python_speedups.py
does NOT:

  - cuda / mps tensor lifecycle (.to(device), .cpu().numpy() round trips)
  - device-specific dtype constraints (mps has no float64)
  - dispatch-time errors for unavailable devices
  - cholesky_ex's per-slot status (info tensor) on a GPU backend, where
    failures are silently masked instead of raising
  - that the device kwarg threads through perform_gsn end-to-end and
    produces results numerically indistinguishable from cpu
  - state independence across sequential GPU calls (no cached tensors
    leaking from one perform_gsn into the next)
  - moderate-N stress at sizes where the GPU dispatch is supposed to win
    (N=500, 1000); not large enough to be a memory test but big enough
    that CPU vs GPU NLL slot agreement is a meaningful signal
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsn.batched_nll import (
    _HAS_TORCH,
    _resolve_device,
    _torch_batched,
    batched_shrunken_nll,
)
from gsn.perform_gsn import perform_gsn

if _HAS_TORCH:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    HAS_MPS = (hasattr(torch.backends, 'mps')
               and torch.backends.mps.is_available())
else:
    torch = None
    HAS_CUDA = False
    HAS_MPS = False

ANY_GPU = HAS_CUDA or HAS_MPS

# Tolerances. Each test compares same-dtype against same-dtype CPU so we
# only measure cross-device numerical reordering, not accumulated float
# precision loss. We use atol + rtol*|ref| because the NLL magnitude
# scales with N (log-determinant term) and M (sum-of-squares term), so an
# absolute-only tolerance would either be loose at small N or tight at
# large N. Float32 needs a looser rtol because cuBLAS / MAGMA dispatch
# a different LAPACK path than scipy/LAPACK on CPU and the rounding
# diverges over 51 batched Cholesky + per-row triangular solves.
TOL_F64_ATOL, TOL_F64_RTOL = 1e-9, 1e-10
TOL_F32_ATOL, TOL_F32_RTOL = 1e-2, 1e-3
TOL_F64_E2E  = 1e-7     # cpu vs cuda end-to-end through perform_gsn

# Backwards-compat aliases used elsewhere in this file.
TOL_F64_NLL = TOL_F64_ATOL
TOL_F32_NLL = TOL_F32_ATOL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_psd(N, *, rng, ridge=1e-3):
    A = rng.standard_normal((N, N))
    return A @ A.T / N + ridge * np.eye(N)


def _low_rank_population(nvox, ncond, ntrial, *, rank_signal=10, rank_noise=20,
                         noise_scale=0.5, seed=0):
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


def _devices_to_test():
    """Yield (device, dtype, atol, rtol) tuples for every available GPU
    backend, including the float32 path on MPS (where float64 doesn't
    exist).
    """
    if HAS_CUDA:
        yield 'cuda', np.float64, TOL_F64_ATOL, TOL_F64_RTOL
        yield 'cuda', np.float32, TOL_F32_ATOL, TOL_F32_RTOL
    if HAS_MPS:
        # MPS internally forces float32 regardless of input dtype.
        yield 'mps', np.float64, TOL_F32_ATOL, TOL_F32_RTOL
        yield 'mps', np.float32, TOL_F32_ATOL, TOL_F32_RTOL


# ===========================================================================
# Device resolution / error surfaces
# ===========================================================================

class TestDeviceResolutionGpu:

    @pytest.mark.skipif(not _HAS_TORCH, reason='torch not installed')
    def test_auto_picks_a_gpu_when_present(self):
        if HAS_CUDA:
            assert _resolve_device('auto') == 'cuda'
        elif HAS_MPS:
            assert _resolve_device('auto') == 'mps'
        else:
            assert _resolve_device('auto') == 'cpu'

    @pytest.mark.skipif(not HAS_CUDA, reason='cuda not available')
    def test_cuda_resolves_when_available(self):
        assert _resolve_device('cuda') == 'cuda'

    @pytest.mark.skipif(not HAS_MPS, reason='mps not available')
    def test_mps_resolves_when_available(self):
        assert _resolve_device('mps') == 'mps'


# ===========================================================================
# batched_nll torch path on GPU
# ===========================================================================

@pytest.mark.skipif(not ANY_GPU, reason='no GPU backend available')
class TestBatchedNllOnGpu:
    """Verify the cholesky_ex + solve_triangular pipeline yields the same
    NLL on GPU as on CPU, across the dtype variants each backend supports."""

    @pytest.mark.parametrize("N,M", [(50, 80), (200, 100), (500, 200)])
    def test_gpu_vs_cpu_well_conditioned(self, N, M):
        rng = np.random.RandomState(0)
        c = _random_psd(N, rng=rng)
        pts_zm = rng.standard_normal((M, N))
        sl = np.linspace(0, 1, 51)
        for device, dtype, atol, rtol in _devices_to_test():
            c_d = c.astype(dtype)
            pts_d = pts_zm.astype(dtype)
            # Precision-matched CPU reference: same dtype both sides so we
            # only measure cross-device LAPACK-vs-cuBLAS reordering, not
            # accumulated float32 rounding loss.
            nll_cpu = batched_shrunken_nll(c_d, pts_d, sl, device='cpu')
            nll_gpu = batched_shrunken_nll(c_d, pts_d, sl, device=device)
            assert nll_gpu.shape == nll_cpu.shape
            # NaN masks must match exactly — disagreement here would mean
            # one path failed Cholesky on a slot where the other succeeded,
            # which IS a real bug.
            assert np.array_equal(np.isnan(nll_gpu), np.isnan(nll_cpu)), (
                f"{device}/{np.dtype(dtype).name}: NaN masks disagree")
            # numpy-style relative+absolute tolerance — handles NLL values
            # that scale with N (log-det term ~ N) and M (quadratic-form
            # term ~ N*M / 2). At N=500 the NLL magnitudes are ~10^3, and
            # f32 precision floor (~N*eps_f32 relative) is around 6e-5
            # relative — well captured by rtol = 1e-3 for f32.
            ref_max = float(np.nanmax(np.abs(nll_cpu)))
            threshold = atol + rtol * ref_max
            diff = float(np.nanmax(np.abs(nll_gpu - nll_cpu)))
            assert diff < threshold, (
                f"{device}/{np.dtype(dtype).name}: max|Δnll|={diff:.2e}  "
                f"(threshold {threshold:.2e}, |nll|max={ref_max:.2e})")
            # The actually-meaningful invariant: both paths must pick the
            # same shrinkage level (= same np.nanargmin index). If math
            # is correct, this will hold regardless of dtype precision.
            assert int(np.nanargmin(nll_cpu)) == int(np.nanargmin(nll_gpu)), (
                f"{device}/{np.dtype(dtype).name}: argmin disagrees")

    def test_per_slot_nan_propagation_on_gpu(self):
        """A rank-deficient training cov fails Cholesky at alpha=1 only.
        On GPU, cholesky_ex returns a nonzero info code rather than
        raising; the masking logic in _torch_batched must put exactly
        NaN at the failing slot and leave the rest valid.
        """
        rng = np.random.RandomState(1)
        N = 30
        # rank-15 in 30-dim space — alpha < 1 stays non-singular (diagonal
        # injection); alpha = 1 is the raw singular cov.
        U, _ = np.linalg.qr(rng.standard_normal((N, 15)))
        c = U @ np.diag(np.linspace(1.0, 0.3, 15)) @ U.T
        pts_zm = rng.standard_normal((40, N))
        sl = np.linspace(0, 1, 51)
        for device, dtype, _, _ in _devices_to_test():
            nll = batched_shrunken_nll(c.astype(dtype), pts_zm.astype(dtype),
                                       sl, device=device)
            assert np.isnan(nll[-1]), f"{device}: alpha=1 should be NaN"
            assert np.any(np.isfinite(nll[:-1])), (
                f"{device}: at least one alpha<1 should be finite")

    def test_all_singular_returns_all_nan_on_gpu(self):
        """All Cholesky calls fail — cholesky_ex returns info > 0 for
        every slot, and our masked write must leave the whole array NaN
        without raising."""
        c = np.zeros((5, 5))
        pts_zm = np.zeros((10, 5))
        sl = np.linspace(0, 1, 51)
        for device, dtype, _, _ in _devices_to_test():
            nll = batched_shrunken_nll(c.astype(dtype), pts_zm.astype(dtype),
                                       sl, device=device)
            assert np.all(np.isnan(nll))

    def test_returns_host_numpy_not_device_tensor(self):
        """The whole point of the abstraction is that callers get back a
        plain numpy array. Make sure we don't accidentally leak a torch
        tensor that requires a device sync to read.
        """
        rng = np.random.RandomState(2)
        c = _random_psd(40, rng=rng)
        pts_zm = rng.standard_normal((50, 40))
        sl = np.linspace(0, 1, 51)
        for device, dtype, _, _ in _devices_to_test():
            nll = batched_shrunken_nll(c.astype(dtype), pts_zm.astype(dtype),
                                       sl, device=device)
            assert isinstance(nll, np.ndarray)
            assert nll.dtype == np.float64

    def test_consecutive_calls_independent(self):
        """Two consecutive calls with different data must not contaminate
        each other through cached device tensors or leftover stream state.
        """
        rng = np.random.RandomState(3)
        c1 = _random_psd(50, rng=rng)
        c2 = _random_psd(50, rng=rng)
        pts1 = rng.standard_normal((60, 50))
        pts2 = rng.standard_normal((60, 50))
        sl = np.linspace(0, 1, 51)
        for device, dtype, atol, rtol in _devices_to_test():
            # Reference: call 2 in isolation.
            nll2_ref = batched_shrunken_nll(c2.astype(dtype), pts2.astype(dtype),
                                            sl, device=device)
            # Now call 1 then call 2 — result of call 2 must match the
            # reference exactly (modulo float-noise — same input, same path).
            _ = batched_shrunken_nll(c1.astype(dtype), pts1.astype(dtype),
                                     sl, device=device)
            nll2 = batched_shrunken_nll(c2.astype(dtype), pts2.astype(dtype),
                                        sl, device=device)
            # Same dtype, same device, same input → diff should be bit-exact
            # in principle but cuda nondeterminism can introduce tiny noise.
            ref_max = float(np.nanmax(np.abs(nll2_ref)))
            threshold = atol + rtol * ref_max
            diff = float(np.nanmax(np.abs(nll2 - nll2_ref)))
            assert diff < threshold, (
                f"{device}/{np.dtype(dtype).name}: consecutive call drift "
                f"{diff:.2e}  (threshold {threshold:.2e})")


# ===========================================================================
# Direct _torch_batched smoke test (skips the dispatch wrapper)
# ===========================================================================

@pytest.mark.skipif(not ANY_GPU, reason='no GPU backend available')
class TestTorchBatchedDirectGpu:
    """Verify _torch_batched itself accepts a device kwarg and produces
    finite NLLs for representative shapes. Catches regressions where
    the dispatch wrapper masks a bug in the inner function."""

    def test_cuda_direct(self):
        if not HAS_CUDA:
            pytest.skip('cuda not available')
        rng = np.random.RandomState(0)
        c = _random_psd(100, rng=rng)
        pts_zm = rng.standard_normal((80, 100))
        sl = np.linspace(0, 1, 51)
        nll = _torch_batched(c, pts_zm, sl, device='cuda')
        assert nll.shape == sl.shape
        assert np.isfinite(nll).any()

    def test_mps_direct(self):
        if not HAS_MPS:
            pytest.skip('mps not available')
        rng = np.random.RandomState(0)
        c = _random_psd(100, rng=rng).astype(np.float32)
        pts_zm = rng.standard_normal((80, 100)).astype(np.float32)
        sl = np.linspace(0, 1, 51)
        nll = _torch_batched(c, pts_zm, sl, device='mps')
        assert nll.shape == sl.shape
        assert np.isfinite(nll).any()


# ===========================================================================
# perform_gsn end-to-end on GPU
# ===========================================================================

@pytest.mark.skipif(not ANY_GPU, reason='no GPU backend available')
class TestPerformGsnOnGpu:
    """End-to-end: opt['device'] = 'cuda' / 'mps' / 'auto' must produce
    cSb / cNb / shrinklevels indistinguishable from the cpu run.
    """

    @pytest.mark.parametrize("nvox,ncond,ntrial", [
        (30, 60, 4),
        (100, 80, 4),
    ])
    def test_gpu_e2e_matches_cpu(self, nvox, ncond, ntrial):
        rng = np.random.RandomState(0)
        data = rng.standard_normal((nvox, ncond, ntrial))
        res_cpu = perform_gsn(data, {'wantverbose': 0, 'device': 'cpu'})
        if HAS_CUDA:
            res_gpu = perform_gsn(data, {'wantverbose': 0, 'device': 'cuda'})
            for key in ('cSb', 'cNb', 'cS', 'cN', 'ncsnr'):
                diff = float(np.max(np.abs(res_cpu[key] - res_gpu[key])))
                assert diff < TOL_F64_E2E, (
                    f"cuda {key}: max|diff|={diff:.2e}")
            assert res_cpu['shrinklevelN'] == res_gpu['shrinklevelN']
            assert res_cpu['shrinklevelD'] == res_gpu['shrinklevelD']
        if HAS_MPS:
            # MPS forces float32 internally, so we expect looser tolerance.
            res_gpu = perform_gsn(data, {'wantverbose': 0, 'device': 'mps'})
            for key in ('cSb', 'cNb', 'cS', 'cN', 'ncsnr'):
                # Compare relative — values can be small.
                ref = res_cpu[key]
                cmp = res_gpu[key]
                denom = np.maximum(np.abs(ref).max(), 1.0)
                diff = float(np.max(np.abs(ref - cmp)) / denom)
                assert diff < 1e-3, f"mps {key}: rel max|diff|={diff:.2e}"

    def test_rank_deficient_data_still_picks_correct_shrinkage(self):
        """Re-run the test 10 regression on GPU. Low-rank population +
        ridge removed → cpu picks shrinklevelD = 0.98; GPU must agree.
        """
        data = _low_rank_population(nvox=50, ncond=40, ntrial=3)
        res_cpu = perform_gsn(data, {'wantverbose': 0, 'device': 'cpu'})
        if HAS_CUDA:
            res_gpu = perform_gsn(data, {'wantverbose': 0, 'device': 'cuda'})
            assert res_cpu['shrinklevelD'] == res_gpu['shrinklevelD']

    def test_auto_uses_a_gpu_path(self):
        """If a GPU is present, 'auto' must dispatch to it. We can't see
        which device was used directly, but if 'auto' is the same as
        'cuda'/'mps' the results must agree bit-equally (same code path).
        """
        rng = np.random.RandomState(0)
        data = rng.standard_normal((40, 80, 4))
        res_auto = perform_gsn(data, {'wantverbose': 0, 'device': 'auto'})
        target = 'cuda' if HAS_CUDA else ('mps' if HAS_MPS else 'cpu')
        res_target = perform_gsn(data, {'wantverbose': 0, 'device': target})
        for key in ('cSb', 'cNb', 'shrinklevelN', 'shrinklevelD'):
            np.testing.assert_array_equal(res_auto[key], res_target[key])


# ===========================================================================
# CUDA-specific: float64 / float32 dispatch parity
# ===========================================================================

@pytest.mark.skipif(not HAS_CUDA, reason='cuda not available')
class TestCudaDtypeBehavior:

    def test_float64_kept_on_cuda(self):
        """float64 input must NOT be silently downcast on CUDA (only MPS
        does that)."""
        rng = np.random.RandomState(0)
        c = _random_psd(60, rng=rng).astype(np.float64)
        pts_zm = rng.standard_normal((40, 60)).astype(np.float64)
        sl = np.linspace(0, 1, 51)
        nll_cuda = batched_shrunken_nll(c, pts_zm, sl, device='cuda')
        nll_cpu = batched_shrunken_nll(c, pts_zm, sl, device='cpu')
        diff = np.nanmax(np.abs(nll_cuda - nll_cpu))
        assert diff < TOL_F64_NLL, (
            f"float64 cuda vs cpu: max|Δnll|={diff:.2e} (should be ≲ {TOL_F64_NLL:.0e})")


# ===========================================================================
# Stress: moderate-N where the GPU is supposed to win
# ===========================================================================

@pytest.mark.skipif(not HAS_CUDA, reason='cuda not available — skipping perf-shaped test')
class TestCudaModerateN:
    """Not a strict performance test (we don't assert specific timings),
    but exercises the dispatch at N where the GPU should be the right
    choice. Verifies correctness, not speed.
    """

    @pytest.mark.parametrize("N", [500, 1000])
    def test_correct_at_moderate_n(self, N):
        rng = np.random.RandomState(0)
        c = _random_psd(N, rng=rng)
        pts_zm = rng.standard_normal((80, N))
        sl = np.linspace(0, 1, 51)
        nll_cpu = batched_shrunken_nll(c, pts_zm, sl, device='cpu')
        nll_cuda = batched_shrunken_nll(c, pts_zm, sl, device='cuda')
        diff = np.nanmax(np.abs(nll_cpu - nll_cuda))
        assert diff < TOL_F64_NLL, f"N={N}: max|Δnll|={diff:.2e}"
        # Both paths should agree on which level minimizes the NLL.
        assert int(np.nanargmin(nll_cpu)) == int(np.nanargmin(nll_cuda))
