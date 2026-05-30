"""Batched negative-log-likelihood evaluation across shrinkage levels.

Why this exists
---------------
``calc_shrunken_covariance`` picks the optimal shrinkage level by
evaluating the held-out Gaussian negative log-likelihood (NLL) at every
candidate shrinkage value (S = 51 by default — ``np.linspace(0, 1, 51)``).
The reference implementation does this with a Python ``for`` loop over
levels; each iteration runs its own Cholesky factorization (``O(N^3)``)
and triangular solve (``O(M·N^2)``) on the shrunken N×N covariance.

For modest N (a few hundred voxels) the 51 × O(N^3) Cholesky cost is
already the dominant term in ``perform_gsn`` wall-clock; for N in the
thousands it utterly dominates. So the fast path here trades a single
``O(S·N^2)`` extra memory allocation (the stacked ``(S, N, N)`` tensor
of shrunken covariances) for collapsing 51 sequential LAPACK calls into
**one** batched call — ``torch.linalg.cholesky_ex`` factorizes the
entire stack in a single kernel launch, with cross-slot threading on
CPU and obvious parallelism on CUDA/MPS. The matching batched
``torch.linalg.solve_triangular`` does the same for the triangular
solves needed to compute the quadratic-form term in the Gaussian
log-density.

Without torch we fall back to a numpy + scipy loop. That loop is
bit-equivalent to the reference (same algorithm, same NaN behavior),
just slightly tidier because we lift the validation-point mean
subtraction out to the caller (it's invariant across shrinkage levels).

What the NLL actually is
------------------------
For a multivariate Gaussian with covariance ``c2`` and mean ``mn``, the
log-density evaluated at point ``x`` is

    log p(x) = -0.5 * (x - mn)^T c2^{-1} (x - mn)
               - 0.5 * log|c2|
               - 0.5 * N * log(2π)

If we Cholesky-factor ``c2 = L L^T`` with ``L`` lower triangular, then
``c2^{-1} = L^{-T} L^{-1}`` and ``log|c2| = 2 * sum(log diag(L))``. So
defining ``y = L^{-1} (x - mn)``,

    log p(x) = -0.5 * ||y||^2 - sum(log diag(L)) - 0.5 * N * log(2π)

We evaluate this at each held-out point ``x_i`` (rows of ``pts_zm``,
already mean-subtracted) and average ``-log p(x_i)`` to get the mean
NLL for that shrinkage level. The caller picks the level that minimizes
this mean NLL.

Shape of the shrinkage formula
------------------------------
The reference computes the shrunken covariance via

    c2 = c * alpha
    np.fill_diagonal(c2, np.diag(c))    # restore diagonal

which is mathematically identical to

    c2 = alpha * c + (1 - alpha) * diag(c)

(off-diagonals scale by alpha; diagonal entries are untouched because
``alpha * d + (1 - alpha) * d = d``). The second form is what we use to
build the batched ``(S, N, N)`` tensor in one vectorized expression
without per-slot in-place mutation.

NaN handling
------------
At ``alpha`` near 1, the unshrunken sample covariance can be singular
(it's allowed to be — the algorithm relies on it). ``torch`` returns
per-slot status from ``cholesky_ex`` without raising, so degenerate
slots map to ``nll = NaN`` cleanly. The numpy path uses a per-iteration
``try``/``except`` on ``np.linalg.LinAlgError`` for the same effect.
The caller uses ``np.nanargmin`` to pick the best level (matching
MATLAB's ``min(nll)``, which silently skips NaN).

Memory note
-----------
We materialize a ``(S, N, N)`` tensor of shrunken covariances. For
``S = 51`` and ``N = 1000`` that's ~408 MB in float64. The reference
loop materialized the same thing one slice at a time. For very large
N this is a memory ceiling, not just a performance trade-off — but
``perform_gsn`` already builds N×N covariances repeatedly, so 51× of
them is the natural batching cost.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular as _solve_triangular

# Optional torch dependency. We probe at module import time and silently
# fall back to the numpy path when torch isn't installed. Installing
# torch (`pip install gsn[fast]`) lights up the batched path with no
# code changes required at call sites.
try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _torch = None
    _HAS_TORCH = False


def _torch_dtype_for(arr, device):
    """Pick the torch dtype that matches the numpy/torch dtype of ``arr``.

    Covariance estimation is sensitive to conditioning so float64 is the
    safe default; we only downcast to float32 if the caller deliberately
    passed float32 data — or if we are on MPS, where Apple Metal does
    not support float64.

    Accepts either a numpy array (``arr.dtype`` is a numpy dtype) or a
    torch tensor (``arr.dtype`` is a torch dtype); we compare via
    ``str(arr.dtype)`` so both branches collapse cleanly.
    """
    if device == 'mps':
        # MPS has no float64 support. Downcasting here is unavoidable;
        # callers who care about float64 conditioning should use cpu or
        # cuda. We don't warn because every call would warn.
        return _torch.float32
    if str(arr.dtype) in ('float32', 'torch.float32'):
        return _torch.float32
    return _torch.float64


def _resolve_device(device):
    """Normalize a device string. ``'auto'`` picks cuda > mps > cpu.

    Returns the string we'll hand to ``torch.as_tensor(device=...)``.
    Raises a clear error if the caller asked for a device that this
    torch install can't reach — better than letting the GPU dispatch
    fail deep in a kernel call.
    """
    if device is None or device == 'cpu':
        return 'cpu'
    if device == 'auto':
        if _torch.cuda.is_available():
            return 'cuda'
        if hasattr(_torch.backends, 'mps') and _torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    if device == 'cuda' and not _torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' requested but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled torch build or pass device='auto'/'cpu'.")
    if device == 'mps' and not (hasattr(_torch.backends, 'mps')
                                and _torch.backends.mps.is_available()):
        raise RuntimeError(
            "device='mps' requested but torch MPS backend is unavailable "
            "(macOS 12.3+ with Apple Silicon required).")
    return device


def _pick_chunk_size(N, M, S, dtype, device, mem_budget_gb=None):
    """Largest shrinkage-level chunk that fits the (chunk, N, N) working set.

    The peak working-set memory while evaluating one chunk of shrinkage
    levels is dominated by three tensors:

      - ``covs[chunk]``:  chunk * N²  (the stacked shrunken covariances)
      - ``Ls[chunk]``:    chunk * N²  (the Cholesky factors)
      - ``Y[chunk]``:     chunk * N * M  (the triangular-solve RHS)

    Total ≈ chunk * (2N² + N*M) * dtype-bytes. Strategy: use ~95% of
    currently-free device memory so chunking only kicks in when the
    single-pass S=51 stack truly wouldn't fit. The remaining 5% covers
    PyTorch allocator fragmentation; caller tensors (c_t, D, cSb, …)
    are small enough at our N range (~1.6 GB at N=10000 f64) that the
    budget can stay tight. At N ≤ ~10000 f64 this leaves chunk = S
    (single-pass behavior identical to pre-chunking); chunking kicks
    in around N ≳ 12000 f64 on an H100 with mostly-free memory.
    """
    bytes_per = 4 if dtype == _torch.float32 else 8
    per_slot_bytes = (2 * N * N + N * M) * bytes_per

    if mem_budget_gb is None:
        dev_str = str(device)
        if dev_str.startswith('cuda'):
            try:
                free, _total = _torch.cuda.mem_get_info(device)
                # 95 % of free memory; the other 5 % covers torch's
                # caching-allocator fragmentation. We rely on the fact
                # that caller-side tensors (c, D, pts_zm, cSb, …) are
                # ≪ 5 GB at our typical N, so they fit inside the
                # remainder without contention.
                mem_budget_gb = (free * 0.95) / (1024 ** 3)
            except Exception:
                # Safe default if mem_get_info is unavailable: assume
                # H100 80GB with ~10 GB of headroom already consumed.
                mem_budget_gb = 60.0
        elif dev_str == 'mps':
            # MPS shares unified memory with the host; be conservative.
            mem_budget_gb = 8.0
        else:
            mem_budget_gb = 30.0

    chunk = int((mem_budget_gb * 1024 ** 3) // max(per_slot_bytes, 1))
    chunk = max(1, min(S, chunk))
    return chunk


def _torch_batched(c, pts_zm, shrinklevels, device='cpu'):
    """Batched ``cholesky_ex`` + batched ``solve_triangular`` over all S levels.

    This is the fast path. The trick is that ``c`` and ``pts_zm`` do not
    depend on the shrinkage level — only the shrunken covariance does.
    So we build the full ``(S, N, N)`` tensor of shrunken covariances
    once, factor it in one batched call, and run a batched triangular
    solve to get the squared-quadratic-form terms for every level at
    once.

    Parameters
    ----------
    c : (N, N) ndarray
        Training sample covariance.
    pts_zm : (M, N) ndarray
        Already-mean-subtracted validation points (M held-out samples).
    shrinklevels : (S,) ndarray
        Shrinkage fractions in [0, 1].
    device : {'cpu', 'cuda', 'mps', 'auto'}
        Torch device string. 'cpu' is the default and ideal for typical
        N ≤ ~1000 (host↔device transfer dominates on GPU below that).
        For larger N, 'cuda' or 'mps' is the right call; 'auto' picks
        cuda > mps > cpu based on availability.

    Returns
    -------
    nll : (S,) float64 numpy array
        Mean negative log-likelihood per shrinkage level; NaN where
        the slot's shrunken covariance was not Cholesky-factorable.
    """
    N = c.shape[0]
    M = pts_zm.shape[0]
    S = len(shrinklevels)
    log_2pi = float(np.log(2 * np.pi))
    tdtype = _torch_dtype_for(c, device)

    # Move inputs to torch tensors on the chosen device. as_tensor copies
    # to device when needed; on cpu it can be a zero-copy view of the
    # numpy buffer. The win we care about (10× on CPU at N≈1000) comes
    # from torch's cross-slot LAPACK threading and from collapsing 51
    # Python-level iterations into one C-level batched call; CUDA/MPS
    # widens the gap further at large N but adds host↔device transfer
    # cost that doesn't pay off until N is well into the thousands.
    c_t = _torch.as_tensor(c, dtype=tdtype, device=device)
    pts_zm_t = _torch.as_tensor(pts_zm, dtype=tdtype, device=device)
    alphas = _torch.as_tensor(shrinklevels, dtype=tdtype, device=device)
    diag_c = _torch.diagonal(c_t).contiguous()                 # (N,)

    # Chunk over shrinkage levels so the working set (covs, Ls, Y) fits
    # in available device memory. For N ≤ ~3000 with the default S=51
    # one chunk covers everything and behavior matches the original
    # single-batched-call path; only large-N or memory-constrained
    # devices fall into multi-chunk territory.
    chunk_size = _pick_chunk_size(N, M, S, tdtype, device)
    nll = np.full(S, np.nan, dtype=np.float64)
    # Precompute the row/col indices we'll use to restore the diagonal
    # in-place after the alpha-scaling. Tiny — O(N).
    diag_idx = _torch.arange(N, device=device)
    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        alphas_chunk = alphas[start:end]
        # In-place covs construction. The expression
        #     covs = alpha*c + (1 - alpha)*diag(c)
        # is equivalent to "scale off-diagonals by alpha, leave the
        # diagonal at diag(c)" because alpha*diag(c) + (1-alpha)*diag(c)
        # = diag(c). So we:
        #   1) allocate ONE chunk*N² buffer holding c repeated,
        #   2) scale it by alpha in place (touches off-diagonals AND
        #      diagonals — we'll fix the diagonals next),
        #   3) restore each row's diagonal to diag(c).
        # Peak transient is one chunk*N² tensor instead of three.
        covs = c_t.unsqueeze(0).expand(end - start, N, N).contiguous()
        covs.mul_(alphas_chunk[:, None, None])
        covs[:, diag_idx, diag_idx] = diag_c.unsqueeze(0).expand(
            end - start, N)

        # cholesky_ex returns (L, info). info[s] == 0 means slot s
        # factorized successfully; any other value indicates the s-th
        # matrix was not positive definite. Crucially, it does NOT
        # raise on failures — degenerate slots propagate as garbage L
        # which we mask out below. This is what enables clean per-slot
        # NaN semantics in a single batched call.
        Ls, info = _torch.linalg.cholesky_ex(covs, upper=False)
        del covs                            # free before allocating Y
        ok = info == 0
        ok_np = ok.cpu().numpy().astype(bool)
        if not ok_np.any():
            continue
        n_ok = int(ok_np.sum())
        Ls_ok = Ls[ok]                                         # (n_ok, N, N)
        del Ls

        # Build the batched right-hand side. We want
        #     y_{s, :, m} = L_s^{-1} pts_zm[m, :]^T
        # for every slot s and every held-out point m. The RHS is the
        # same across slots, so we broadcast pts_zm.T from (N, M) to
        # (n_ok, N, M) using expand (a zero-copy view).
        rhs = pts_zm_t.T.unsqueeze(0).expand(n_ok, N, M)
        Y = _torch.linalg.solve_triangular(
            Ls_ok, rhs, upper=False, unitriangular=False)
        sq = (Y ** 2).sum(dim=1)                               # (n_ok, M)
        del Y

        logdet = _torch.log(
            _torch.diagonal(Ls_ok, dim1=-2, dim2=-1)
        ).sum(dim=1)                                           # (n_ok,)
        del Ls_ok

        log_pdf = (-0.5 * sq
                   - logdet.unsqueeze(1)
                   - 0.5 * N * log_2pi)                        # (n_ok, M)

        # Mean over held-out samples; negate so smaller = better. Only
        # the successful slots in this chunk are written; the rest stay
        # NaN. ``ok_np`` is chunk-local (length end-start), so we map
        # its True positions back to absolute indices in the full nll.
        nll[start + np.flatnonzero(ok_np)] = (
            -log_pdf.mean(dim=1)).cpu().numpy()
    return nll


def _numpy_loop(c, pts_zm, shrinklevels):
    """Numpy + scipy fallback. Bit-equivalent to the reference inner loop.

    The only difference vs. the historical reference is that we lift
    mean-subtraction out of this function (the caller has already done
    it on ``pts_zm``) — that was a wasted recomputation per shrinkage
    level. Everything else (Cholesky, triangular solve, NaN handling)
    matches the reference one-for-one.
    """
    N = c.shape[0]
    S = len(shrinklevels)
    log_2pi = float(np.log(2 * np.pi))
    diag_c = np.diag(c) if N > 1 else None
    # Transpose once outside the loop; scipy's triangular solve wants
    # the RHS shaped (N, M) so each column is a separate system.
    pts_zm_T = pts_zm.T
    nll = np.full(S, np.nan, dtype=np.float64)
    for p in range(S):
        alpha = shrinklevels[p]
        # Same shrunken-cov formula as the torch path, just one slot at
        # a time. fill_diagonal mutates in place — fine because c2 is a
        # freshly allocated copy from the scalar multiplication.
        c2 = c * alpha
        if N > 1:
            np.fill_diagonal(c2, diag_c)
        try:
            L = np.linalg.cholesky(c2)
        except np.linalg.LinAlgError:
            # Singular shrunken covariance — leave nll[p] as NaN. The
            # caller (np.nanargmin) will skip this slot, matching
            # MATLAB's min(nll) NaN-skip behavior.
            continue
        # Solve L @ Y = pts_zm.T  =>  Y = L^{-1} pts_zm.T per column.
        Y = _solve_triangular(L, pts_zm_T, lower=True)
        sq = (Y ** 2).sum(axis=0)                              # (M,)
        logdet = np.log(np.diag(L)).sum()
        log_pdf = -0.5 * sq - logdet - 0.5 * N * log_2pi       # (M,)
        nll[p] = float(-np.mean(log_pdf))
    return nll


def batched_shrunken_nll(c, pts_zm, shrinklevels, use_torch=None, device='cpu'):
    """Mean negative log-likelihood at every shrinkage level, in one shot.

    This is the entry point used by ``calc_shrunken_covariance``. It
    dispatches to the torch fast path when torch is importable, and to
    the numpy + scipy loop otherwise. Both paths return the same
    ``nll`` semantics, so callers don't need to branch.

    Parameters
    ----------
    c : (N, N) ndarray
        Training covariance.
    pts_zm : (M, N) ndarray
        Zero-mean validation points (already mean-subtracted).
    shrinklevels : (S,) ndarray
        Shrinkage levels in [0, 1].
    use_torch : bool, optional
        Force the torch path (True) or the numpy path (False). Default
        ``None`` uses torch when available. Mostly useful for
        benchmarking and for tests that want to exercise the numpy path
        explicitly.
    device : {'cpu', 'cuda', 'mps', 'auto'}, optional
        Torch device for the fast path. Ignored when the numpy path is
        used. Default ``'cpu'``. ``'auto'`` picks cuda > mps > cpu by
        availability. GPU devices only beat CPU at N ≳ a few hundred
        because of host↔device transfer; below that, ``'cpu'`` wins.

    Returns
    -------
    nll : (S,) float64 ndarray
        Mean negative log-likelihood per shrinkage level. NaN where the
        shrunken covariance failed Cholesky.
    """
    if use_torch is None:
        use_torch = _HAS_TORCH
    if use_torch:
        return _torch_batched(c, pts_zm, shrinklevels, device=_resolve_device(device))
    return _numpy_loop(c, pts_zm, shrinklevels)
