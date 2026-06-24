"""Device-native end-to-end perform_gsn.

For clean data (no NaN), this runs the entire GSN pipeline (noise+data
covariance estimation, the held-out shrinkage selection, the biconvex
optimization, and ncsnr) on a single backend (numpy or torch on
CPU/CUDA/MPS) without round-tripping through host memory between stages.
Reproduces the same dict that the rsa_noise_ceiling mode=1 path used to
return.

For uneven trials (NaN in data), opt['uneven'] selects the estimator:
'fast' (default) is a NaN-aware whole-trial path, 'missing' handles per-unit
missingness (see gsn.missing_units), and 'reference' delegates to the original
rsa_noise_ceiling path.

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
# downstream PSN at large N (eigh is O(N^3), and auto mode does it
# twice). Saving them once means later PSN calls just consume
# opt['basis'] = <matrix> and skip basis construction.

_VALID_RETURNS = ('cN', 'cS', 'cNb', 'cSb',
                  'eigvecs_signal', 'eigvals_signal',
                  'eigvecs_difference', 'eigvals_difference')
DEFAULT_RETURNS = ('cN', 'cS', 'cNb', 'cSb')


def _normalize_returns(value):
    """Validate the opt['returns'] selector and return it as a set.

    None  → the default set DEFAULT_RETURNS (``'cN', 'cS', 'cNb', 'cSb'``).
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


def _get_shrinklevels(opt):
    """Shrinkage levels to evaluate. Honors a user-provided opt['shrinklevels']
    (matching the reference rsa_noise_ceiling); otherwise the default 0..1 grid
    of 51, or just [1.0] when wantshrinkage is False."""
    lv = opt.get('shrinklevels')
    if lv is not None:
        return np.asarray(lv, dtype=float)
    if opt.get('wantshrinkage', True):
        return np.linspace(0, 1, 51)
    return np.array([1.0])


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
    opt['shrinklevels'] = _get_shrinklevels(opt)
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


def _biconvex_numpy(cN, cD, cS, ncond, ntrial, max_iters=100, ntrialBC=None):
    """Biconvex cSb / cNb iteration. Convergence: corrcoef on flat N².

    ``ntrial`` is used for the ``cD - cNb/ntrial`` step (the per-trial-average
    noise scaling); ``ntrialBC`` drives the cNb update coefficients. They
    differ only for uneven-trials data, where rsa_noise_ceiling uses
    ntrial = min(validcnt) for the former and ntrialBC = average count for the
    latter. ``ntrialBC=None`` (even data) makes both equal -> bit-unchanged.
    """
    if ntrialBC is None:
        ntrialBC = ntrial
    cNb = cN
    cSb_old = cS
    cNb_old = cN
    numiters = 0
    for _ in range(max_iters):
        cSb = _nearest_psd_numpy(cD - cNb / ntrial)
        # cNb update formula from rsa_noise_ceiling (coefficients use ntrialBC)
        coef_N = ((ncond * (ntrialBC - 1) * ntrialBC ** 2)
                  / (ncond * ntrialBC ** 2 * (ntrialBC - 1) + ncond - 1))
        coef_D = ((ncond - 1)
                  / (ncond * ntrialBC ** 2 * (ntrialBC - 1) + ncond - 1))
        cNb = _nearest_psd_numpy(coef_N * cN + coef_D * ntrialBC * (cD - cSb))
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
    shrinklevels = _get_shrinklevels(opt)
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


# ===========================================================================
# Uneven-trials path (NaN-padded data)
# ---------------------------------------------------------------------------
# Mirrors rsa_noise_ceiling mode=1 so the covariance outputs match the
# reference (hence MATLAB) to numerical precision, but the per-condition
# noise covariance is one vectorized GEMM instead of a Python loop, and the
# biconvex / eigh reuse the fast even-path machinery. Key facts replicated
# exactly:
#   * noise cov  = average over conditions (>=2 valid trials) of each
#                  condition's unbiased sample cov (np.cov bias=False).
#   * data cov   = condition means after truncating every condition to
#                  min(validcnt) randomly-chosen valid trials.
#   * ntrial     = min(validcnt)         -> cS, cSb (= cD - cNb/ntrial).
#   * ntrialBC   = mean valid count over conditions with >1 trial -> biconvex
#                  coefficients.
#   * diff basis = cSb - cNb / mean(validcnt)  (matches _delegate_uneven).
# deterministic_randperm is stateless (depends only on its length arg), so
# replaying the same length arguments reproduces the reference's random
# trial choices bit-for-bit.
# ===========================================================================

def _uneven_validity(data_np):
    """Per-(cond, trial) validity mask and per-condition valid counts.

    A trial is valid for a condition iff no voxel is NaN (NaN-padding marks
    whole missing trials). Returns (valid (ncond, ntrial) bool,
    validcnt (ncond,) int).
    """
    valid = ~np.isnan(data_np).any(axis=0)        # (ncond, ntrial)
    validcnt = valid.sum(axis=1).astype(int)      # (ncond,)
    return valid, validcnt


def _percond_avg_cov_numpy(data_np, valid, validcnt, cond_idx):
    """Average of per-condition unbiased sample covariances over cond_idx.

    For each condition with >=2 valid trials: covariance of its valid trials
    normalized by (T_c - 1) (== np.cov(..., bias=False)); then averaged across
    those conditions. Done as one GEMM over weighted centered residuals.
    """
    nvox = data_np.shape[0]
    sub = data_np[:, cond_idx, :]                       # (nvox, k, ntrial)
    vsub = valid[cond_idx]                               # (k, ntrial)
    cnt = validcnt[cond_idx].astype(np.float64)         # (k,)
    d0 = np.where(vsub[None], sub, 0.0)                 # NaN -> 0
    mean_c = d0.sum(axis=2) / np.maximum(cnt, 1.0)      # (nvox, k)
    r = np.where(vsub[None], d0 - mean_c[:, :, None], 0.0)
    w = np.where(cnt >= 2, 1.0 / np.maximum(cnt - 1.0, 1.0), 0.0)
    nvalid = int((cnt >= 2).sum())
    if nvalid < 1:
        raise AssertionError('no condition with at least two valid observations')
    rw = r * np.sqrt(w)[None, :, None]
    R = rw.reshape(nvox, -1)
    return (R @ R.T) / nvalid


def _shrunken_noise_cov_uneven_numpy(data_np, valid, validcnt, leaveout, shrinklevels):
    """cN for uneven trials: CV shrinkage selection over conditions + full refit.

    Reproduces calc_shrunken_covariance's 3D uneven path (cases = conditions).
    """
    nvox, ncond, _ = data_np.shape
    perm = deterministic_randperm(ncond)
    val_size = int(np.round(ncond / leaveout))
    ii, iinot = perm[:val_size], perm[val_size:]

    c_train = _percond_avg_cov_numpy(data_np, valid, validcnt, iinot)
    pts = []
    for q in ii:
        vix = valid[q]
        if int(vix.sum()) > 1:
            vt = data_np[:, q, vix].T                   # (T_q, nvox)
            pts.append(vt - vt.mean(axis=0))
    if not pts:                                         # match reference assertion
        raise AssertionError(
            'validation data did not have any conditions with at least two '
            'observations')
    pts_zm = np.vstack(pts)

    nll = _numpy_loop(c_train, pts_zm, shrinklevels)
    if np.all(np.isnan(nll)):
        warnings.warn('All covariance matrices were singular.')
        best = 0
    else:
        best = int(np.nanargmin(nll))
    chosen_level = float(shrinklevels[best])

    c_full = _percond_avg_cov_numpy(data_np, valid, validcnt, np.arange(ncond))
    diag_full = np.diag(c_full).copy()
    c_final = c_full * chosen_level
    if c_final.shape[0] > 1:
        np.fill_diagonal(c_final, diag_full)
    mn = np.zeros((1, nvox), dtype=data_np.dtype)
    return mn, c_final, chosen_level


def _truncate_min_trials_numpy(data_np, valid, validcnt):
    """Random subset of each condition's valid trials down to min(validcnt).

    Exactly reproduces rsa_noise_ceiling's uneven data-cov truncation: valid
    trials in ascending index order, permuted by deterministic_randperm
    (depends only on the count), first ntrial_min kept.
    """
    nvox, ncond, _ = data_np.shape
    ntrial_min = int(validcnt.min())
    newdata = np.empty((nvox, ncond, ntrial_min), dtype=data_np.dtype)
    for p in range(ncond):
        vidx = np.flatnonzero(valid[p])                 # ascending valid indices
        temp = data_np[:, p, vidx]                      # (nvox, T_p)
        ix = deterministic_randperm(temp.shape[1])
        newdata[:, p, :] = temp[:, ix[:ntrial_min]]
    return newdata, ntrial_min


def _run_numpy_uneven(data_np, opt) -> Dict[str, Any]:
    nvox, ncond, _ = data_np.shape
    shrinklevels = _get_shrinklevels(opt)
    returns = _normalize_returns(opt.get('returns'))

    valid, validcnt = _uneven_validity(data_np)
    if not np.all(validcnt >= 1):
        raise AssertionError('all conditions must have at least 1 valid trial')

    mnN, cN, shrinklevelN = _shrunken_noise_cov_uneven_numpy(
        data_np, valid, validcnt, 5, shrinklevels)

    newdata, ntrial_min = _truncate_min_trials_numpy(data_np, valid, validcnt)
    data_2d = np.mean(newdata, axis=2).T
    mnD, cD, shrinklevelD = _shrunken_cov_2d_numpy(data_2d, 5, shrinklevels)

    ntrialBC = float(np.sum(validcnt[validcnt > 1]) / ncond)
    if ntrialBC < 1:
        warnings.warn('ntrialBC is lopsided! setting to 1')
        ntrialBC = 1.0

    mnS = mnD - mnN
    cS = cD - cN / ntrial_min
    sd_noise = np.sqrt(np.maximum(np.diag(cN), 0))
    sd_signal = np.sqrt(np.maximum(np.diag(cS), 0))
    ncsnr = np.divide(sd_signal, sd_noise,
                      out=np.zeros_like(sd_signal), where=sd_noise != 0)
    cSb, cNb, numiters = _biconvex_numpy(
        cN, cD, cS, ncond, ntrial_min, ntrialBC=ntrialBC)

    # _assemble uses its ntrial arg ONLY for the difference eigenbasis
    # divisor; the reference (_delegate_uneven) uses the mean valid count.
    ntrial_avg = float(np.nanmean(validcnt))
    return _assemble_result_numpy(
        returns, mnN, mnS, ncsnr, shrinklevelN, shrinklevelD, numiters,
        cN, cS, cNb, cSb, ntrial_avg)


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


def _biconvex_torch(cN, cD, ncond, ntrial, max_iters=100, ntrialBC=None):
    """Biconvex loop on device. Convergence uses pearson correlation of
    the flattened cov matrices, matching the numpy path's criterion
    (modulo floating-point reordering).

    cSb_old at iter 0 is derived inline as cD - cN/ntrial; we never
    materialize a separate cS device tensor, which saves one full-size
    buffer of headroom throughout biconvex.

    ``ntrial`` scales the cD - cNb/ntrial step; ``ntrialBC`` drives the cNb
    update coefficients (they differ only for uneven-trials data).
    ``ntrialBC=None`` makes both equal -> even-path bit-unchanged.
    """
    if ntrialBC is None:
        ntrialBC = ntrial
    cNb = cN
    cSb_old = cD.clone().sub_(cN, alpha=1.0 / ntrial)
    cNb_old = cN
    numiters = 0
    coef_N = ((ncond * (ntrialBC - 1) * ntrialBC ** 2)
              / (ncond * ntrialBC ** 2 * (ntrialBC - 1) + ncond - 1))
    coef_D = ((ncond - 1)
              / (ncond * ntrialBC ** 2 * (ntrialBC - 1) + ncond - 1))
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
            .add_(cD, alpha=coef_D * ntrialBC)
            .sub_(cSb, alpha=coef_D * ntrialBC))
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
    shrinklevels = _get_shrinklevels(opt)
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
    #   branches. Cost: one extra (N, N) f64 host eigh — still cheaper
    #   than re-running the same eigh once per downstream PSN call.
    #
    # opt['eigh_device'] = 'device' (opt-in): torch eigh on the active
    #   device (CUDA/MPS), with the same f64 upcast + sign convention.
    #   Much faster at large N. Produces a mathematically valid
    #   orthonormal basis, but picks DIFFERENT rotations on
    #   degenerate (zero-eigenvalue) subspaces of cSb — always
    #   present when nvox > ncond - 1, which is common for the kinds
    #   of high-dimensional data PSN is intended for. PSN's
    #   downstream threshold selection uses
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
# Torch uneven-trials path — same algorithm as _run_numpy_uneven on device
# ===========================================================================

def _percond_avg_cov_torch(data, valid, validcnt_f, cond_idx, device):
    """Torch version of _percond_avg_cov_numpy. ``data`` must already have
    NaN positions replaced by 0 (so masked sums are clean)."""
    nvox = data.shape[0]
    sub = data.index_select(1, cond_idx)                  # (nvox, k, ntrial)
    vsub = valid.index_select(0, cond_idx)                # (k, ntrial) bool
    cnt = validcnt_f.index_select(0, cond_idx)            # (k,) float
    mean_c = sub.sum(dim=2) / torch.clamp(cnt, min=1.0).unsqueeze(0)  # (nvox, k)
    zero = torch.zeros((), dtype=data.dtype, device=device)
    r = torch.where(vsub.unsqueeze(0), sub - mean_c.unsqueeze(2), zero)
    w = torch.where(cnt >= 2, 1.0 / torch.clamp(cnt - 1.0, min=1.0),
                    torch.zeros_like(cnt))
    nvalid = int((cnt >= 2).sum().item())
    if nvalid < 1:
        raise AssertionError('no condition with at least two valid observations')
    rw = r * torch.sqrt(w).view(1, -1, 1)
    R = rw.reshape(nvox, -1)
    return (R @ R.T) / nvalid


def _shrunken_noise_cov_uneven_torch(data, valid, validcnt_f, valid_np,
                                     leaveout, shrinklevels, device):
    nvox, ncond, _ = data.shape
    perm = deterministic_randperm(ncond)
    val_size = int(np.round(ncond / leaveout))
    iinot = torch.from_numpy(perm[val_size:].copy()).to(device)

    c_train = _percond_avg_cov_torch(data, valid, validcnt_f, iinot, device)
    pts = []
    for q in perm[:val_size]:
        vix = valid_np[q]
        if int(vix.sum()) > 1:
            cols = torch.from_numpy(np.flatnonzero(vix)).to(device)
            vt = data[:, int(q), :].index_select(1, cols).T   # (T_q, nvox)
            pts.append(vt - vt.mean(dim=0))
    if not pts:                                         # match reference assertion
        raise AssertionError(
            'validation data did not have any conditions with at least two '
            'observations')
    pts_zm = torch.cat(pts, dim=0)

    nll = _torch_batched(c_train, pts_zm, shrinklevels, device=device)
    del pts_zm, c_train
    if np.all(np.isnan(nll)):
        warnings.warn('All covariance matrices were singular.')
        best = 0
    else:
        best = int(np.nanargmin(nll))
    chosen_level = float(shrinklevels[best])

    c_full = _percond_avg_cov_torch(
        data, valid, validcnt_f, torch.arange(ncond, device=device), device)
    diag_full = torch.diagonal(c_full).clone()
    c_full.mul_(chosen_level)
    idx = torch.arange(nvox, device=device)
    c_full[idx, idx] = diag_full
    mn = torch.zeros((1, nvox), dtype=data.dtype, device=device)
    return mn, c_full, chosen_level


def _run_torch_uneven(data_np, opt, device) -> Dict[str, Any]:
    dtype = _torch_dtype_for(data_np, device)
    nvox, ncond, _ = data_np.shape
    shrinklevels = _get_shrinklevels(opt)
    returns = _normalize_returns(opt.get('returns'))

    valid_np, validcnt_np = _uneven_validity(data_np)
    if not np.all(validcnt_np >= 1):
        raise AssertionError('all conditions must have at least 1 valid trial')

    valid = torch.as_tensor(valid_np, device=device)
    validcnt_f = torch.as_tensor(validcnt_np, dtype=dtype, device=device)
    data_raw = torch.as_tensor(data_np, dtype=dtype, device=device)
    zero = torch.zeros((), dtype=dtype, device=device)
    data = torch.where(valid.unsqueeze(0), data_raw, zero)   # NaN -> 0
    del data_raw

    mnN, cN, shrinklevelN = _shrunken_noise_cov_uneven_torch(
        data, valid, validcnt_f, valid_np, 5, shrinklevels, device)
    del data, valid, validcnt_f
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

    # Data cov: truncate to min trials in numpy (deterministic), then 2D torch.
    newdata, ntrial_min = _truncate_min_trials_numpy(data_np, valid_np, validcnt_np)
    data_2d = torch.as_tensor(
        np.mean(newdata, axis=2).T, dtype=dtype, device=device).contiguous()
    del newdata
    mnD, cD, shrinklevelD = _shrunken_cov_2d_torch(data_2d, 5, shrinklevels, device)
    del data_2d

    ntrialBC = float(np.sum(validcnt_np[validcnt_np > 1]) / ncond)
    if ntrialBC < 1:
        warnings.warn('ntrialBC is lopsided! setting to 1')
        ntrialBC = 1.0

    mnS = mnD - mnN
    diag_cS = torch.diagonal(cD) - torch.diagonal(cN) / ntrial_min
    sd_noise = torch.sqrt(torch.clamp(torch.diag(cN), min=0))
    sd_signal = torch.sqrt(torch.clamp(diag_cS, min=0))
    ncsnr = torch.where(sd_noise > 0, sd_signal / sd_noise,
                        torch.zeros_like(sd_signal))
    del sd_noise, sd_signal, diag_cS
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

    result: Dict[str, Any] = {
        'mnN': mnN.cpu().numpy(), 'shrinklevelN': shrinklevelN,
        'mnS': mnS.cpu().numpy(), 'shrinklevelD': shrinklevelD,
        'ncsnr': ncsnr.cpu().numpy(),
    }
    if 'cN' in returns:
        result['cN'] = cN.cpu().numpy()
    if 'cS' in returns:
        cS_tmp = cD - cN / ntrial_min
        result['cS'] = cS_tmp.cpu().numpy()
        del cS_tmp
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()

    cSb, cNb, numiters = _biconvex_torch(
        cN, cD, ncond, ntrial_min, ntrialBC=ntrialBC)
    result['numiters'] = numiters
    del cN, cD
    if 'cNb' in returns:
        result['cNb'] = cNb.cpu().numpy()
    if 'cSb' in returns:
        result['cSb'] = cSb.cpu().numpy()

    # Difference basis divisor matches _delegate_uneven (mean valid count).
    ntrial_avg = float(np.nanmean(validcnt_np))
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
                d, V = _eigh_descending_numpy(cSb_np - cNb_np / ntrial_avg)
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
                diff = cSb.clone().sub_(cNb, alpha=1.0 / ntrial_avg)
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
          - backend ({'auto', 'numpy', 'torch'}): which compute path to use.
            'auto' (default) uses torch if installed, else numpy; 'numpy'
            forces the reference numpy path; 'torch' forces the torch path
            (errors if torch is missing).
          - device ({'cpu', 'cuda', 'mps', 'auto'}): the torch device, used
            only by the torch backend. Under backend='auto' with torch
            unavailable, a non-cpu request falls back to cpu+numpy.
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
    # Copy so we never mutate the caller's opt dict.
    opt = {} if opt is None else dict(opt)
    opt.setdefault('wantverbose', 0)
    opt.setdefault('wantshrinkage', True)

    # Input guards (match rsa_noise_ceiling): voxels x conditions x trials,
    # at least 2 trials so the per-condition covariance is defined.
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError('data must be voxels x conditions x trials (3D)')
    if data.shape[2] < 2:
        raise ValueError('Number of trials must be at least 2.')

    uneven = np.isnan(data).any()
    # opt['uneven'] selects how missing data is handled:
    #   'fast'      (default): whole-trial uneven path (a trial is valid only
    #               if every unit is present); fast NaN-aware estimation.
    #   'reference': original rsa_noise_ceiling delegation (parity oracle).
    #   'missing':  missing-units estimation for per-UNIT missingness (a
    #               trial may have some units present and others missing); no
    #               good data discarded. See gsn.missing_units.
    uneven_mode = opt.get('uneven', 'fast')
    if uneven_mode not in ('fast', 'reference', 'missing'):
        raise ValueError(
            f"opt['uneven'] must be 'fast', 'reference', or 'missing'; got {uneven_mode!r}")
    if uneven and uneven_mode == 'reference':
        return _delegate_uneven(data, opt)

    # Backend selection. 'auto' (default) uses the torch path when torch is
    # installed and the numpy path otherwise; 'numpy' / 'torch' force a path.
    # The numpy path is the reference implementation; the torch path (cpu or
    # gpu) is faster and is validated against numpy in tests/.
    backend = opt.get('backend', 'auto')
    if backend not in ('numpy', 'torch', 'auto'):
        raise ValueError(
            f"opt['backend'] must be 'numpy', 'torch', or 'auto'; got {backend!r}")
    if backend == 'torch' and not _HAS_TORCH:
        raise RuntimeError(
            "opt['backend']='torch' but torch is not installed "
            "(pip install gsn[fast]).")
    use_torch = (backend == 'torch') or (backend == 'auto' and _HAS_TORCH)

    device_str = opt.get('device', 'cpu')
    if device_str != 'cpu' and not use_torch:
        if backend == 'numpy':
            raise ValueError(
                "opt['device'] requests a non-cpu device but opt['backend']='numpy' "
                "(the numpy backend is cpu-only).")
        # backend='auto' with torch unavailable: silently demote to cpu+numpy.
        device_str = 'cpu'

    if uneven and uneven_mode == 'missing':
        if use_torch:
            from gsn.missing_units import run_missing_units_torch
            return run_missing_units_torch(data, opt, _resolve_device(device_str))
        from gsn.missing_units import run_missing_units_numpy
        return run_missing_units_numpy(data, opt)

    if uneven:
        if use_torch:
            device = _resolve_device(device_str)
            return _run_torch_uneven(data, opt, device)
        return _run_numpy_uneven(data, opt)

    if use_torch:
        device = _resolve_device(device_str)
        return _run_torch(data, opt, device)
    return _run_numpy(data, opt)
