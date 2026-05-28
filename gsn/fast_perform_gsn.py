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

def _delegate_uneven(data, opt):
    """Fall back to rsa_noise_ceiling mode=1 for uneven-trials data.

    That code path has multi-step stochastic trial-subsetting and a
    careful NaN handling story that's not worth rewriting here. The
    returned dict has the same shape as the fast path's.
    """
    from gsn.rsa_noise_ceiling import rsa_noise_ceiling
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

    return {
        'mnN': mnN, 'cN': cN, 'cNb': cNb, 'shrinklevelN': shrinklevelN,
        'mnS': mnS, 'cS': cS, 'cSb': cSb, 'shrinklevelD': shrinklevelD,
        'ncsnr': ncsnr, 'numiters': numiters,
    }


# ===========================================================================
# Torch path — same algorithm, everything stays on device
# ===========================================================================

def _shrunken_cov_3d_torch(data_3d, leaveout, shrinklevels_np, device):
    obs, var, ncase = data_3d.shape
    perm = deterministic_randperm(ncase)
    val_size = int(np.round(ncase / leaveout))
    ii = torch.from_numpy(perm[:val_size].copy()).to(device)
    iinot = torch.from_numpy(perm[val_size:].copy()).to(device)

    train = data_3d.index_select(2, iinot)
    centered_train = train - train.mean(dim=0, keepdim=True)
    c = (torch.einsum('ovc,owc->vw', centered_train, centered_train)
         / (obs - 1) / train.shape[2])

    val = data_3d.index_select(2, ii)
    centered_val = val - val.mean(dim=0, keepdim=True)
    pts_zm = centered_val.permute(2, 0, 1).reshape(-1, var)

    nll = _torch_batched(c, pts_zm, shrinklevels_np, device=device)
    if np.all(np.isnan(nll)):
        warnings.warn('All covariance matrices were singular.')
        best = 0
    else:
        best = int(np.nanargmin(nll))
    chosen_level = float(shrinklevels_np[best])

    centered_full = data_3d - data_3d.mean(dim=0, keepdim=True)
    c_full = (torch.einsum('ovc,owc->vw', centered_full, centered_full)
              / (obs - 1) / ncase)
    diag_full = torch.diag(c_full)
    D_full = torch.diag(diag_full)
    c_final = chosen_level * c_full + (1 - chosen_level) * D_full
    mn = torch.zeros((1, var), dtype=data_3d.dtype, device=device)
    return mn, c_final, chosen_level


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


def _nearest_psd_torch(M, eps=1e-10):
    M = (M + M.transpose(-1, -2)) / 2
    try:
        torch.linalg.cholesky(M)
        return M
    except Exception:
        pass
    d, v = torch.linalg.eigh(M)
    d = torch.clamp(d, min=0)
    Mp = (v * d) @ v.transpose(-1, -2)
    Mp = (Mp + Mp.transpose(-1, -2)) / 2
    try:
        torch.linalg.cholesky(Mp)
    except Exception:
        Mp = Mp + eps * torch.eye(Mp.shape[0], device=Mp.device, dtype=Mp.dtype)
    return Mp


def _biconvex_torch(cN, cD, cS, ncond, ntrial, max_iters=100):
    """Biconvex loop on device. Convergence uses torch.corrcoef so we
    match the numpy path's stopping criterion bit-for-bit (modulo
    floating-point reordering)."""
    cNb = cN
    cSb_old = cS
    cNb_old = cN
    numiters = 0
    for _ in range(max_iters):
        cSb = _nearest_psd_torch(cD - cNb / ntrial)
        coef_N = ((ncond * (ntrial - 1) * ntrial ** 2)
                  / (ncond * ntrial ** 2 * (ntrial - 1) + ncond - 1))
        coef_D = ((ncond - 1)
                  / (ncond * ntrial ** 2 * (ntrial - 1) + ncond - 1))
        cNb = _nearest_psd_torch(coef_N * cN + coef_D * ntrial * (cD - cSb))
        if cSb.shape[0] == 1:
            converged = (torch.abs(cSb_old - cSb).item() < 1e-5
                         and torch.abs(cNb_old - cNb).item() < 1e-5)
        else:
            # torch.corrcoef expects each row to be a variable; we want
            # the pearson correlation of two flat N²-vectors. Stack them
            # as 2 rows.
            stack_S = torch.stack([cSb_old.flatten(), cSb.flatten()])
            stack_N = torch.stack([cNb_old.flatten(), cNb.flatten()])
            r_S = float(torch.corrcoef(stack_S)[0, 1].item())
            r_N = float(torch.corrcoef(stack_N)[0, 1].item())
            converged = r_S > 0.999 and r_N > 0.999
        if converged:
            break
        numiters += 1
        cSb_old = cSb
        cNb_old = cNb
    return cSb, cNb, numiters


def _run_torch(data_np, opt, device) -> Dict[str, Any]:
    dtype = _torch_dtype_for(data_np, device)
    _, ncond, ntrial = data_np.shape
    shrinklevels = (np.linspace(0, 1, 51) if opt.get('wantshrinkage', True)
                    else np.array([1.0]))

    data = torch.as_tensor(data_np, dtype=dtype, device=device)

    # Noise cov: (obs=ntrial, var=nvox, case=ncond)
    data_noise_3d = data.permute(2, 0, 1).contiguous()
    mnN, cN, shrinklevelN = _shrunken_cov_3d_torch(
        data_noise_3d, 5, shrinklevels, device)

    data_2d = data.mean(dim=2).T.contiguous()
    mnD, cD, shrinklevelD = _shrunken_cov_2d_torch(
        data_2d, 5, shrinklevels, device)

    mnS = mnD - mnN
    cS = cD - cN / ntrial
    sd_noise = torch.sqrt(torch.clamp(torch.diag(cN), min=0))
    sd_signal = torch.sqrt(torch.clamp(torch.diag(cS), min=0))
    ncsnr = torch.where(
        sd_noise > 0, sd_signal / sd_noise,
        torch.zeros_like(sd_signal))

    cSb, cNb, numiters = _biconvex_torch(cN, cD, cS, ncond, ntrial)

    return {
        'mnN': mnN.cpu().numpy(),
        'cN': cN.cpu().numpy(),
        'cNb': cNb.cpu().numpy(),
        'shrinklevelN': shrinklevelN,
        'mnS': mnS.cpu().numpy(),
        'cS': cS.cpu().numpy(),
        'cSb': cSb.cpu().numpy(),
        'shrinklevelD': shrinklevelD,
        'ncsnr': ncsnr.cpu().numpy(),
        'numiters': numiters,
    }


# ===========================================================================
# Public entry point
# ===========================================================================

def fast_perform_gsn(data: np.ndarray, opt: Optional[Dict] = None) -> Dict[str, Any]:
    """Drop-in replacement for the old perform_gsn implementation.

    Parameters
    ----------
    data : (nvox, ncond, ntrial) ndarray
    opt : dict, optional
        Honors the same keys as the old perform_gsn:
          - wantverbose (bool)
          - wantshrinkage (bool)
          - device ({'cpu', 'cuda', 'mps', 'auto'}) — only relevant when
            torch is installed; falls back to numpy on cpu otherwise.

    Returns
    -------
    dict with mnN, cN, cNb, shrinklevelN, mnS, cS, cSb, shrinklevelD,
    ncsnr, numiters.
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
