"""Device-native end-to-end perform_gsn.

For clean data (no NaN), this runs the entire GSN pipeline — noise+data
covariance estimation, the held-out shrinkage selection, the biconvex
optimization, and ncsnr — on a single backend (numpy or torch on
CPU/CUDA/MPS) without round-tripping through host memory between stages.
Reproduces the same dict that the rsa_noise_ceiling mode=1 path used to
return.

For uneven trials (NaN in data), we fall through to the existing
rsa_noise_ceiling path — that branch has stochastic trial-subsetting
that isn't worth reimplementing here.

What changed vs. the previous calc_shrunken_covariance + rsa_noise_ceiling
flow:

  - **Einsum for the 3D pooled noise covariance.** The reference loops
    over training conditions calling ``np.cov`` per slice. With no NaNs
    that's a single ``einsum('ovc,owc->vw', centered, centered)`` on the
    centered tensor, which keeps everything on-device and saves
    ``ncond_train`` Python-level kernel launches on torch.
  - **Biconvex loop stays on device.** The reference biconvex iteration
    calls ``construct_nearest_psd_covariance`` which used numpy.linalg
    on the host even when torch was available — so cSb/cNb did host↔
    device round trips every iteration. Here ``_nearest_psd`` is written
    against the active backend, so the entire biconvex loop runs on the
    chosen device.
  - **End-to-end on one backend.** Data moves to the device once at
    entry; we only materialize numpy at the very end when building the
    results dict.

The numpy path is bit-equivalent to the previous flow (same algorithm,
same Cholesky / eigh, same NaN handling).
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np

from gsn.utilities import deterministic_randperm
from gsn.batched_nll import (
    _HAS_TORCH,
    _resolve_device,
    _torch_dtype_for,
    _numpy_loop,
    _torch_batched,
)

if _HAS_TORCH:
    import torch
else:
    torch = None


# ---------------------------------------------------------------------------
# Delegation to the old rsa_noise_ceiling path (for NaN / uneven trials)
# ---------------------------------------------------------------------------

# Output selection ---------------------------------------------------------
#
# A caller picks which output items they need via opt['returns']. The
# default matches the legacy perform_gsn output (the four cov matrices
# cN / cS / cNb / cSb). The cheap items below are always returned
# regardless:
#
#     mnN, mnS, ncsnr, shrinklevelN, shrinklevelD, numiters
#
# Opt-in eigenbases:
#     'eigvecs_signal'      eigenvectors of cSb,  (nvox, nvox) f64 →
#                           cast back to the run dtype before returning
#     'eigvals_signal'      eigenvalues of cSb,   (nvox,)
#     'eigvecs_difference'  eigenvectors of (cSb - cNb / ntrial), (nvox, nvox)
#     'eigvals_difference'  eigenvalues of (cSb - cNb / ntrial), (nvox,)
#
# All eigvecs are columns; both pairs are sorted by descending
# eigenvalue. Eigendecomposing inside GSN saves the dominant cost of
# downstream PSN (PSN at nvox=24640 spends 5-10 min on each eigh; auto
# mode does it twice). Saving them once means later PSN calls just
# consume opt['basis'] = <matrix> and skip basis construction.

_VALID_RETURNS = ('cN', 'cS', 'cNb', 'cSb',
                  'eigvecs_signal', 'eigvals_signal',
                  'eigvecs_difference', 'eigvals_difference')
DEFAULT_RETURNS = ('cN', 'cS', 'cNb', 'cSb')


def _normalize_returns(value):
    """Validate the opt['returns'] selector and return it as a set.

    None  → the default (cSb, cNb, cdiff, three eigenbases).
    str   → single-item set.
    iter  → all items must be names from _VALID_RETURNS.
    """
    if value is None:
        return set(DEFAULT_RETURNS)
    if isinstance(value, str):
        value = (value,)
    s = set(value)
    bad = s - set(_VALID_RETURNS)
    if bad:
        raise ValueError(
            f"opt['returns'] contains unknown items {sorted(bad)}; "
            f"valid names are {sorted(_VALID_RETURNS)}")
    return s


def _delegate_uneven(data, opt):
    """Fall back to rsa_noise_ceiling mode=1 for uneven-trials data.

    That code path has multi-step stochastic trial-subsetting and a
    careful NaN handling story that's not worth rewriting here. The
    returned dict honors opt['returns'] by dropping cov matrices the
    caller didn't ask for.
    """
    from gsn.rsa_noise_ceiling import rsa_noise_ceiling
    returns = _normalize_returns(opt.get('returns'))
    opt = dict(opt)
    opt.setdefault('wantverbose', 0)
    opt.setdefault('wantshrinkage', 1)
    opt['mode'] = 1
    opt['ncsims'] = 0
    opt['wantfig'] = 0
    opt['shrinklevels'] = (np.linspace(0, 1, 51) if opt['wantshrinkage']
                           else np.array([1.0]))
    result = rsa_noise_ceiling(data, opt)[2]
    result.pop('sc', None)
    result.pop('splitr', None)

    # rsa_noise_ceiling always returns all four cov matrices; drop the
    # ones the caller didn't ask for. Eigenbases are added post-hoc
    # because rsa_noise_ceiling doesn't compute them.
    want_es = 'eigvecs_signal'     in returns or 'eigvals_signal'     in returns
    want_ed = 'eigvecs_difference' in returns or 'eigvals_difference' in returns
    if want_es:
        d, V = _eigh_descending_numpy(result['cSb'])
        if 'eigvals_signal' in returns: result['eigvals_signal'] = d
        if 'eigvecs_signal' in returns: result['eigvecs_signal'] = V
    if want_ed:
        # ntrial_avg from the data — rsa_noise_ceiling uses the average
        # over conditions with ≥2 valid trials. Approximate it here.
        ntrial_avg = np.nanmean(np.sum(~np.isnan(data[0]), axis=-1))
        d, V = _eigh_descending_numpy(
            result['cSb'] - result['cNb'] / ntrial_avg)
        if 'eigvals_difference' in returns: result['eigvals_difference'] = d
        if 'eigvecs_difference' in returns: result['eigvecs_difference'] = V
    for k in ('cN', 'cS', 'cNb', 'cSb'):
        if k not in returns and k in result:
            del result[k]
    return result


# ===========================================================================
# Numpy path — bit-equivalent to the legacy flow
# ===========================================================================

def _shrunken_cov_3d_numpy(data_3d, leaveout, shrinklevels):
    """3D shrunken cov for (obs, var, case) input via einsum.

    Algorithm matches calc_shrunken_covariance's 3D no-NaN path:
      - split cases (conditions) deterministically into train + validation
      - pooled per-case covariance averaged across training cases (here
        done in one einsum over centered residuals)
      - evaluate held-out Gaussian NLL at every shrinkage level via
        Cholesky + triangular solve
      - pick the level minimizing the NLL (NaN slots skipped)
      - refit pooled cov on all cases at the chosen shrinkage level
        (wantfull=1 semantics)
    """
    obs, var, ncase = data_3d.shape
    perm = deterministic_randperm(ncase)
    val_size = int(np.round(ncase / leaveout))
    ii, iinot = perm[:val_size], perm[val_size:]

    train = data_3d[:, :, iinot]
    centered_train = train - train.mean(axis=0, keepdims=True)
    c = (np.einsum('ovc,owc->vw', centered_train, centered_train)
         / (obs - 1) / train.shape[2])

    val = data_3d[:, :, ii]
    centered_val = val - val.mean(axis=0, keepdims=True)
    # (obs, var, val_size) -> (val_size * obs, var)
    pts_zm = centered_val.transpose(2, 0, 1).reshape(-1, var)

    nll = _numpy_loop(c, pts_zm, shrinklevels)
    if np.all(np.isnan(nll)):
        warnings.warn('All covariance matrices were singular.')
        best = 0
    else:
        best = int(np.nanargmin(nll))
    chosen_level = float(shrinklevels[best])

    centered_full = data_3d - data_3d.mean(axis=0, keepdims=True)
    c_full = (np.einsum('ovc,owc->vw', centered_full, centered_full)
              / (obs - 1) / ncase)
    diag_full = np.diag(c_full).copy()
    c_final = c_full * chosen_level
    if c_final.shape[0] > 1:
        np.fill_diagonal(c_final, diag_full)
    mn = np.zeros((1, var), dtype=data_3d.dtype)
    return mn, c_final, chosen_level


def _shrunken_cov_2d_numpy(data_2d, leaveout, shrinklevels):
    """2D shrunken cov for (obs, var) input. Direct sample cov."""
    nobs, _ = data_2d.shape
    perm = deterministic_randperm(nobs)
    val_size = int(np.round(nobs / leaveout))
    ii, iinot = perm[:val_size], perm[val_size:]
    X_train = data_2d[iinot]
    mn_train = X_train.mean(axis=0)
    centered = X_train - mn_train
    c = (centered.T @ centered) / (X_train.shape[0] - 1)
    pts_zm = data_2d[ii] - mn_train

    nll = _numpy_loop(c, pts_zm, shrinklevels)
    if np.all(np.isnan(nll)):
        warnings.warn('All covariance matrices were singular.')
        best = 0
    else:
        best = int(np.nanargmin(nll))
    chosen_level = float(shrinklevels[best])

    mn_full = data_2d.mean(axis=0)
    centered_full = data_2d - mn_full
    c_full = (centered_full.T @ centered_full) / (nobs - 1)
    diag_full = np.diag(c_full).copy()
    c_final = c_full * chosen_level
    if c_final.shape[0] > 1:
        np.fill_diagonal(c_final, diag_full)
    return mn_full, c_final, chosen_level


def _nearest_psd_numpy(M, eps=1e-10):
    """Project a symmetric matrix to the PSD cone via eigh.

    Equivalent to construct_nearest_psd_covariance's eigh path, but
    inlined here so the fast path doesn't import the public function and
    pay its scalar/1×1 input handling overhead in the inner loop.
    """
    M = (M + M.T) / 2
    try:
        np.linalg.cholesky(M)
        return M
    except np.linalg.LinAlgError:
        pass
    d, v = np.linalg.eigh(M)
    d = np.maximum(d, 0)
    Mp = (v * d) @ v.T
    Mp = (Mp + Mp.T) / 2
    try:
        np.linalg.cholesky(Mp)
    except np.linalg.LinAlgError:
        Mp = Mp + eps * np.eye(Mp.shape[0])
    return Mp


def _biconvex_numpy(cN, cD, cS, ncond, ntrial, max_iters=100):
    """Biconvex cSb / cNb iteration. Convergence: corrcoef on flat N²."""
    cNb = cN
    cSb_old = cS
    cNb_old = cN
    numiters = 0
    for _ in range(max_iters):
        cSb = _nearest_psd_numpy(cD - cNb / ntrial)
        # cNb update formula from rsa_noise_ceiling
        coef_N = ((ncond * (ntrial - 1) * ntrial ** 2)
                  / (ncond * ntrial ** 2 * (ntrial - 1) + ncond - 1))
        coef_D = ((ncond - 1)
                  / (ncond * ntrial ** 2 * (ntrial - 1) + ncond - 1))
        cNb = _nearest_psd_numpy(coef_N * cN + coef_D * ntrial * (cD - cSb))
        if cSb.shape[0] == 1:
            converged = (abs(cSb_old - cSb) < 1e-5
                         and abs(cNb_old - cNb) < 1e-5)
        else:
            r_S = np.corrcoef(cSb_old.flatten(), cSb.flatten())[0, 1]
            r_N = np.corrcoef(cNb_old.flatten(), cNb.flatten())[0, 1]
            converged = r_S > 0.999 and r_N > 0.999
        if converged:
            break
        numiters += 1
        cSb_old = cSb
        cNb_old = cNb
    return cSb, cNb, numiters


def _run_numpy(data_np, opt) -> Dict[str, Any]:
    _, ncond, ntrial = data_np.shape
    shrinklevels = (np.linspace(0, 1, 51) if opt.get('wantshrinkage', True)
                    else np.array([1.0]))
    returns = _normalize_returns(opt.get('returns'))

    # Noise cov: (obs=ntrial, var=nvox, case=ncond)
    data_noise_3d = np.transpose(data_np, (2, 0, 1))
    mnN, cN, shrinklevelN = _shrunken_cov_3d_numpy(data_noise_3d, 5, shrinklevels)

    # Data cov: rows = conditions, cols = voxels
    data_2d = np.mean(data_np, axis=2).T
    mnD, cD, shrinklevelD = _shrunken_cov_2d_numpy(data_2d, 5, shrinklevels)

    mnS = mnD - mnN
    cS = cD - cN / ntrial
    sd_noise = np.sqrt(np.maximum(np.diag(cN), 0))
    sd_signal = np.sqrt(np.maximum(np.diag(cS), 0))
    ncsnr = np.divide(sd_signal, sd_noise,
                      out=np.zeros_like(sd_signal), where=sd_noise != 0)
    cSb, cNb, numiters = _biconvex_numpy(cN, cD, cS, ncond, ntrial)

    return _assemble_result_numpy(
        returns, mnN, mnS, ncsnr, shrinklevelN, shrinklevelD, numiters,
        cN, cS, cNb, cSb, ntrial)


def _assemble_result_numpy(returns, mnN, mnS, ncsnr,
                           shrinklevelN, shrinklevelD, numiters,
                           cN, cS, cNb, cSb, ntrial):
    """Build the public result dict honoring ``returns``.

    Cheap items are always present; cov matrices and eigenbases are
    included only when named in ``returns``.
    """
    result: Dict[str, Any] = {
        'mnN': mnN, 'shrinklevelN': shrinklevelN,
        'mnS': mnS, 'shrinklevelD': shrinklevelD,
        'ncsnr': ncsnr, 'numiters': numiters,
    }
    if 'cN'  in returns: result['cN']  = cN
    if 'cS'  in returns: result['cS']  = cS
    if 'cNb' in returns: result['cNb'] = cNb
    if 'cSb' in returns: result['cSb'] = cSb

    # Eigenbases — opt-in. Sorted descending by eigenvalue so column 0
    # is the top mode. cSb is symmetric (we projected to nearest-PSD
    # in biconvex) so eigh is the right tool; the difference matrix
    # cSb - cNb / ntrial is symmetric but generally indefinite, eigh
    # still applies and the negative eigenvalues are physically
    # meaningful (variance the noise has in those dims minus the
    # signal's; PSN treats them accordingly).
    want_es = 'eigvecs_signal'     in returns or 'eigvals_signal'     in returns
    want_ed = 'eigvecs_difference' in returns or 'eigvals_difference' in returns
    if want_es:
        d, V = _eigh_descending_numpy(cSb)
        if 'eigvals_signal' in returns: result['eigvals_signal'] = d
        if 'eigvecs_signal' in returns: result['eigvecs_signal'] = V
    if want_ed:
        d, V = _eigh_descending_numpy(cSb - cNb / ntrial)
        if 'eigvals_difference' in returns: result['eigvals_difference'] = d
        if 'eigvecs_difference' in returns: result['eigvecs_difference'] = V
    return result


def _eigh_descending_numpy(M):
    """eigh on a symmetric matrix, returned in descending order with
    a deterministic sign convention.

    Matches PSN's eigh_descending_sym exactly so that PSN consuming
    these vectors via opt['basis'] + opt['basis_eigenvalues'] reproduces
    'basis': 'signal' / 'difference' bit-for-bit:
      - LAPACK driver: numpy.linalg.eigh (same as PSN; using scipy
        with driver='evr' picks a different orthonormal basis on
        degenerate eigenspaces).
      - Sort: by eigenvalue descending (value, not magnitude — so
        negative eigenvalues of the indefinite difference matrix
        sort to the tail, which is what PSN expects).
      - Sign: each column's element of largest absolute value made
        positive (zeros mapped to +1).

    Returns (eigenvalues (N,), eigenvectors (N, N) with column i = vec i).
    """
    d, V = np.linalg.eigh(M)
    order = np.argsort(d)[::-1]
    d = d[order]
    V = V[:, order]
    piv = np.argmax(np.abs(V), axis=0)
    sgn = np.sign(V[piv, np.arange(V.shape[1])])
    sgn[sgn == 0] = 1
    V = V * sgn
    return d.astype(M.dtype, copy=False), V.astype(M.dtype, copy=False)


# ===========================================================================
# Torch path — same algorithm, everything stays on device
# ===========================================================================

def _shrunken_cov_3d_torch(data_3d, leaveout, shrinklevels_np, device):
    obs, var, ncase = data_3d.shape
    perm = deterministic_randperm(ncase)
    val_size = int(np.round(ncase / leaveout))
    ii = torch.from_numpy(perm[:val_size].copy()).to(device)
    iinot = torch.from_numpy(perm[val_size:].copy()).to(device)

    # Training pooled cov via einsum. We materialize train/centered_train
    # only briefly; once c is computed we drop them so the (var, var) cov
    # is the only large tensor surviving into the NLL eval.
    train = data_3d.index_select(2, iinot)
    centered_train = train - train.mean(dim=0, keepdim=True)
    del train
    c = (torch.einsum('ovc,owc->vw', centered_train, centered_train)
         / (obs - 1) / centered_train.shape[2])
    del centered_train

    # Same for validation: build pts_zm, then free val / centered_val.
    val = data_3d.index_select(2, ii)
    centered_val = val - val.mean(dim=0, keepdim=True)
    del val
    pts_zm = centered_val.permute(2, 0, 1).reshape(-1, var)
    del centered_val

    nll = _torch_batched(c, pts_zm, shrinklevels_np, device=device)
    del pts_zm
    if np.all(np.isnan(nll)):
        warnings.warn('All covariance matrices were singular.')
        best = 0
    else:
        best = int(np.nanargmin(nll))
    chosen_level = float(shrinklevels_np[best])
    del c

    # Full-data refit (wantfull=1). Same del-on-last-use pattern.
    centered_full = data_3d - data_3d.mean(dim=0, keepdim=True)
    c_full = (torch.einsum('ovc,owc->vw', centered_full, centered_full)
              / (obs - 1) / ncase)
    del centered_full
    # Shrink in place: c_final = alpha*c_full + (1-alpha)*diag(c_full)
    #                = alpha*c_full with diagonal restored.
    diag_full = torch.diagonal(c_full).clone()
    c_full.mul_(chosen_level)
    idx = torch.arange(var, device=device)
    c_full[idx, idx] = diag_full
    mn = torch.zeros((1, var), dtype=data_3d.dtype, device=device)
    return mn, c_full, chosen_level


def _shrunken_cov_2d_torch(data_2d, leaveout, shrinklevels_np, device):
    nobs, _ = data_2d.shape
    perm = deterministic_randperm(nobs)
    val_size = int(np.round(nobs / leaveout))
    ii = torch.from_numpy(perm[:val_size].copy()).to(device)
    iinot = torch.from_numpy(perm[val_size:].copy()).to(device)
    X_train = data_2d.index_select(0, iinot)
    mn_train = X_train.mean(dim=0)
    centered = X_train - mn_train
    c = (centered.T @ centered) / (X_train.shape[0] - 1)
    pts_zm = data_2d.index_select(0, ii) - mn_train

    nll = _torch_batched(c, pts_zm, shrinklevels_np, device=device)
    if np.all(np.isnan(nll)):
        warnings.warn('All covariance matrices were singular.')
        best = 0
    else:
        best = int(np.nanargmin(nll))
    chosen_level = float(shrinklevels_np[best])

    mn_full = data_2d.mean(dim=0)
    centered_full = data_2d - mn_full
    c_full = (centered_full.T @ centered_full) / (nobs - 1)
    diag_full = torch.diag(c_full)
    D_full = torch.diag(diag_full)
    c_final = chosen_level * c_full + (1 - chosen_level) * D_full
    return mn_full, c_final, chosen_level


def _eigh_descending_torch(M, out_dtype=None):
    """eigh on a symmetric device tensor, returned in descending order.

    Mirrors _nearest_psd_torch's robustness pattern:
      - f32 inputs are upcast to f64 for the eigh (cuSOLVER syevd is
        unreliable on near-singular f32)
      - if the device eigh raises (typically the cuSOLVER syevd
        workspace limit at very large N), fall back to
        scipy.linalg.eigh on the host

    Returns (eigvals (N,), eigvecs (N, N) with descending order).
    Output dtype defaults to the input dtype.
    """
    M_sym = (M + M.transpose(-1, -2)).mul_(0.5)
    in_dtype = M_sym.dtype
    orig_device = M_sym.device
    if out_dtype is None:
        out_dtype = in_dtype

    if in_dtype == torch.float32:
        M_eig = M_sym.to(torch.float64)
        del M_sym
    else:
        M_eig = M_sym

    try:
        d, V = torch.linalg.eigh(M_eig)
    except (torch.OutOfMemoryError, RuntimeError) as err:
        warnings.warn(
            f'_eigh_descending_torch: device eigh failed '
            f'({type(err).__name__}); falling back to scipy CPU eigh.')
        import scipy.linalg as _scilin
        M_np = M_eig.cpu().numpy()
        if str(orig_device).startswith('cuda'):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        d_np, V_np = _scilin.eigh(M_np, driver='evr')
        del M_np
        d = torch.from_numpy(d_np).to(orig_device)
        V = torch.from_numpy(V_np).to(orig_device)
        del d_np, V_np
    del M_eig

    # eigh returns ascending order — reverse to descending.
    d = d.flip(0)
    V = V.flip(1)
    # Match PSN's eigh_descending_sym sign convention so callers can
    # consume these vectors directly via opt['basis'] + opt['basis_
    # eigenvalues'] and reproduce PSN's 'signal'/'difference' branches
    # bit-for-bit. Pick the largest-magnitude element in each column
    # and force its sign positive.
    piv = torch.argmax(torch.abs(V), dim=0)
    col_idx = torch.arange(V.shape[1], device=V.device)
    sgn = torch.sign(V[piv, col_idx])
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    V = V * sgn.unsqueeze(0)
    if d.dtype != out_dtype: d = d.to(out_dtype)
    if V.dtype != out_dtype: V = V.to(out_dtype)
    return d, V


def _nearest_psd_torch(M, eps=1e-10):
    """Nearest-PSD projection via eigh with clamping. Preserves the
    same math as MATLAB's construct_nearest_psd_covariance
    (V * max(D, 0) * V').

    f32 → f64 upcast for the eigh (cuSOLVER syevd is unreliable on
    near-singular f32 inputs). Falls back to scipy.linalg.eigh on the
    host if the device eigh raises — common at very large N where
    cuSOLVER syevd hits its workspace limit.
    """
    M = (M + M.transpose(-1, -2)).mul_(0.5)
    try:
        torch.linalg.cholesky(M)
        return M
    except Exception:
        pass

    in_dtype = M.dtype
    orig_device = M.device
    N = M.shape[-1]

    if in_dtype == torch.float32:
        M_eig = M.to(torch.float64)
        del M
    else:
        M_eig = M

    try:
        d, v = torch.linalg.eigh(M_eig)
    except (torch.OutOfMemoryError, RuntimeError) as err:
        warnings.warn(
            f'_nearest_psd_torch: device eigh failed '
            f'({type(err).__name__}); falling back to scipy CPU eigh.')
        import scipy.linalg as _scilin
        M_np = M_eig.cpu().numpy()
        if str(orig_device).startswith('cuda'):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        d_np, v_np = _scilin.eigh(M_np, driver='evr')
        del M_np
        d = torch.from_numpy(d_np).to(orig_device)
        v = torch.from_numpy(v_np).to(orig_device)
        del d_np, v_np
    del M_eig

    d = torch.clamp(d, min=0)
    Mp = (v * d) @ v.transpose(-1, -2)
    del v, d
    if Mp.dtype != in_dtype:
        Mp = Mp.to(in_dtype)
    Mp = (Mp + Mp.transpose(-1, -2)).mul_(0.5)
    try:
        torch.linalg.cholesky(Mp)
    except Exception:
        Mp = Mp + eps * torch.eye(N, device=Mp.device, dtype=Mp.dtype)
    return Mp


def _flat_pearson(a, b):
    """Pearson correlation between flattened a and b. Uses element-wise
    mul + reduce-sum with f64 accumulators rather than torch.dot, since
    cuBLAS/MKL dot kernels cap their length at int32 (~2.1e9 elements);
    element-wise ops use int64 strides and have no such cap."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    n = a.numel()
    use_f64_accum = a.dtype == torch.float32
    accum_kw = {'dtype': torch.float64} if use_f64_accum else {}
    sum_a = a.sum(**accum_kw)
    sum_b = b.sum(**accum_kw)
    sum_ab = (a * b).sum(**accum_kw)
    sum_aa = (a * a).sum(**accum_kw)
    sum_bb = (b * b).sum(**accum_kw)
    ma = sum_a / n
    mb = sum_b / n
    num = sum_ab - n * ma * mb
    var_a = sum_aa - n * ma * ma
    var_b = sum_bb - n * mb * mb
    return float((num / torch.sqrt(var_a * var_b)).item())


def _biconvex_torch(cN, cD, ncond, ntrial, max_iters=100):
    """Biconvex loop on device. Convergence uses pearson correlation of
    the flattened cov matrices, matching the numpy path's criterion
    (modulo floating-point reordering).

    cSb_old at iter 0 is derived inline as cD - cN/ntrial; we never
    materialize a separate cS device tensor, which saves one full-size
    buffer of headroom throughout biconvex.
    """
    cNb = cN
    cSb_old = cD.clone().sub_(cN, alpha=1.0 / ntrial)
    cNb_old = cN
    numiters = 0
    coef_N = ((ncond * (ntrial - 1) * ntrial ** 2)
              / (ncond * ntrial ** 2 * (ntrial - 1) + ncond - 1))
    coef_D = ((ncond - 1)
              / (ncond * ntrial ** 2 * (ntrial - 1) + ncond - 1))
    for _ in range(max_iters):
        cSb = _nearest_psd_torch(
            cD.clone().sub_(cNb, alpha=1.0 / ntrial))
        if cSb.shape[0] == 1:
            r_S_ok = torch.abs(cSb_old - cSb).item() < 1e-5
        else:
            r_S_ok = _flat_pearson(cSb_old, cSb) > 0.999
        del cSb_old

        cNb = _nearest_psd_torch(
            (cN * coef_N)
            .add_(cD, alpha=coef_D * ntrial)
            .sub_(cSb, alpha=coef_D * ntrial))
        if cNb.shape[0] == 1:
            r_N_ok = torch.abs(cNb_old - cNb).item() < 1e-5
        else:
            r_N_ok = _flat_pearson(cNb_old, cNb) > 0.999
        del cNb_old

        if r_S_ok and r_N_ok:
            break
        numiters += 1
        cSb_old = cSb
        cNb_old = cNb
        if str(cN.device).startswith('cuda'):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return cSb, cNb, numiters


def _run_torch(data_np, opt, device) -> Dict[str, Any]:
    dtype = _torch_dtype_for(data_np, device)
    _, ncond, ntrial = data_np.shape
    shrinklevels = (np.linspace(0, 1, 51) if opt.get('wantshrinkage', True)
                    else np.array([1.0]))
    returns = _normalize_returns(opt.get('returns'))


    data = torch.as_tensor(data_np, dtype=dtype, device=device)

    # Noise cov: (obs=ntrial, var=nvox, case=ncond). Free intermediate
    # data tensors as soon as their cov is built so they don't survive
    # across the (much larger) biconvex working set.
    data_noise_3d = data.permute(2, 0, 1).contiguous()
    mnN, cN, shrinklevelN = _shrunken_cov_3d_torch(
        data_noise_3d, 5, shrinklevels, device)
    del data_noise_3d

    data_2d = data.mean(dim=2).T.contiguous()
    del data
    mnD, cD, shrinklevelD = _shrunken_cov_2d_torch(
        data_2d, 5, shrinklevels, device)
    del data_2d

    mnS = mnD - mnN
    # ncsnr only needs the diagonal of cS = cD - cN/ntrial, not the
    # full matrix. Keeping just the diagonal here avoids a full-size
    # cS tensor surviving into biconvex.
    diag_cS = torch.diagonal(cD) - torch.diagonal(cN) / ntrial
    sd_noise = torch.sqrt(torch.clamp(torch.diag(cN), min=0))
    sd_signal = torch.sqrt(torch.clamp(diag_cS, min=0))
    ncsnr = torch.where(
        sd_noise > 0, sd_signal / sd_noise,
        torch.zeros_like(sd_signal))
    del sd_noise, sd_signal, diag_cS

    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

    # Snapshot the requested cN/cS to host BEFORE biconvex. cS is built
    # only for the host snapshot (if requested) and freed before
    # biconvex starts; biconvex derives its iter-0 anchor internally.
    result: Dict[str, Any] = {
        'mnN': mnN.cpu().numpy(),
        'shrinklevelN': shrinklevelN,
        'mnS': mnS.cpu().numpy(),
        'shrinklevelD': shrinklevelD,
        'ncsnr': ncsnr.cpu().numpy(),
    }
    if 'cN' in returns: result['cN'] = cN.cpu().numpy()
    if 'cS' in returns:
        cS_tmp = cD - cN / ntrial
        result['cS'] = cS_tmp.cpu().numpy()
        del cS_tmp
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()

    cSb, cNb, numiters = _biconvex_torch(cN, cD, ncond, ntrial)
    result['numiters'] = numiters
    del cN, cD

    if 'cNb' in returns: result['cNb'] = cNb.cpu().numpy()
    if 'cSb' in returns: result['cSb'] = cSb.cpu().numpy()

    # Eigenbases — opt-in via opt['returns']. Two paths:
    #
    # opt['eigh_device'] = 'host' (default): numpy.linalg.eigh on the
    #   host + deterministic sign convention. Produces vectors bit-
    #   equivalent to PSN's own eigh_descending_sym, so the cached
    #   eigvecs are a drop-in for PSN's internal 'signal' / 'difference'
    #   branches. Cost: one extra (N, N) f64 host eigh (~15-20 min at
    #   N=24640). Still cheaper than re-running the same eigh once per
    #   downstream PSN call.
    #
    # opt['eigh_device'] = 'device' (opt-in): torch eigh on the active
    #   device (CUDA/MPS), with the same f64 upcast + sign convention.
    #   Much faster at large N (~1-2 min at N=24640 on H100). Produces
    #   a mathematically valid orthonormal basis, but picks DIFFERENT
    #   rotations on degenerate (zero-eigenvalue) subspaces of cSb —
    #   always present when nvox > ncond - 1 (the typical EEG case).
    #   PSN's downstream threshold selection uses
    #     noise_proj_diag = V[:, i].T @ cNb @ V[:, i]
    #   which IS sensitive to those rotations even though
    #   signal_proj_diag is not. So device-cached eigvecs feeding PSN
    #   will produce results that diverge by a few percent from
    #   PSN running its own eigh. Choose this when you don't need
    #   exact PSN parity and want the GSN run to finish faster.
    want_es = 'eigvecs_signal'     in returns or 'eigvals_signal'     in returns
    want_ed = 'eigvecs_difference' in returns or 'eigvals_difference' in returns
    if want_es or want_ed:
        eigh_device = opt.get('eigh_device', 'host')
        if eigh_device not in ('host', 'device'):
            raise ValueError(
                f"opt['eigh_device'] must be 'host' or 'device'; "
                f"got {eigh_device!r}")
        if eigh_device == 'host':
            cSb_np = cSb.cpu().numpy()
            if want_es:
                d, V = _eigh_descending_numpy(cSb_np)
                if 'eigvals_signal' in returns: result['eigvals_signal'] = d
                if 'eigvecs_signal' in returns: result['eigvecs_signal'] = V
            if want_ed:
                cNb_np = cNb.cpu().numpy()
                d, V = _eigh_descending_numpy(cSb_np - cNb_np / ntrial)
                if 'eigvals_difference' in returns: result['eigvals_difference'] = d
                if 'eigvecs_difference' in returns: result['eigvecs_difference'] = V
            del cSb_np
        else:
            if want_es:
                d, V = _eigh_descending_torch(cSb)
                if 'eigvals_signal' in returns: result['eigvals_signal'] = d.cpu().numpy()
                if 'eigvecs_signal' in returns: result['eigvecs_signal'] = V.cpu().numpy()
                del d, V
                if str(device).startswith('cuda'):
                    torch.cuda.empty_cache()
            if want_ed:
                diff = cSb.clone().sub_(cNb, alpha=1.0 / ntrial)
                d, V = _eigh_descending_torch(diff)
                del diff
                if 'eigvals_difference' in returns: result['eigvals_difference'] = d.cpu().numpy()
                if 'eigvecs_difference' in returns: result['eigvecs_difference'] = V.cpu().numpy()
                del d, V
                if str(device).startswith('cuda'):
                    torch.cuda.empty_cache()
    del cNb, cSb

    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return result


# ===========================================================================
# Public entry point
# ===========================================================================

def fast_perform_gsn(data: np.ndarray, opt: Optional[Dict] = None) -> Dict[str, Any]:
    """Drop-in replacement for the old perform_gsn implementation.

    Parameters
    ----------
    data : (nvox, ncond, ntrial) ndarray
    opt : dict, optional
        Honors:
          - wantverbose (bool)
          - wantshrinkage (bool)
          - device ({'cpu', 'cuda', 'mps', 'auto'}) — only relevant when
            torch is installed; falls back to numpy on cpu otherwise.
          - returns (iterable of str, optional): which cov matrices to
            include in the result dict. Default ``('cN', 'cS', 'cNb',
            'cSb')`` — the four matrices the legacy perform_gsn always
            returned. Pass an iterable like ``['cSb', 'cNb']`` if you
            don't need cN / cS and want to save host memory at large N.
            Valid names: ``'cN', 'cS', 'cNb', 'cSb'``.

    Returns
    -------
    dict. Always present (cheap):
      mnN, mnS, ncsnr, shrinklevelN, shrinklevelD, numiters.

    Present iff named in ``opt['returns']``:
      cN, cS, cNb, cSb — each (N, N).
    """
    if opt is None:
        opt = {}
    opt.setdefault('wantverbose', 0)
    opt.setdefault('wantshrinkage', True)

    if np.isnan(data).any():
        return _delegate_uneven(data, opt)

    device_str = opt.get('device', 'cpu')
    if device_str != 'cpu' and not _HAS_TORCH:
        # Asked for GPU but no torch — silently demote to cpu+numpy.
        device_str = 'cpu'

    if _HAS_TORCH:
        device = _resolve_device(device_str)
        return _run_torch(data, opt, device)
    return _run_numpy(data, opt)
