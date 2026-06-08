"""Missing-units GSN for per-unit (not just whole-trial) missing data.

When artifacts are rejected per electrode/unit rather than per whole trial,
a trial can have some units present and others missing. Standard GSN (and the
whole-trial uneven path) requires a trial to be all-present or all-dropped, so
it would discard the good units' data on any partially-missing trial. This
module estimates each covariance entry from whatever data is present instead:
every entry uses exactly the observations available for that unit (or unit
pair), discarding no good data.

Estimators (each reduces EXACTLY to standard GSN when no data is missing):

  noise cov  cN[i,j] = average over conditions (with >=2 common-clean trials)
             of the UNBIASED pairwise sample covariance of units i,j over the
             trials where BOTH are clean (each pair centered on its shared-clean
             trials, so the estimate is unbiased for any overlap pattern).

  data cov   cD[i,j] = unbiased pairwise covariance, across conditions, of the
             per-unit condition-means (each mean over that unit's clean trials),
             using the conditions where both units are defined.

  bias       cS = cD - cN (.) alpha,  where the per-entry factor
             alpha[i,j] = avg_c [ n_ij,c / (n_i,c * n_j,c) ]
             (common-clean trials over the product of each unit's clean count,
             averaged over conditions where both are defined). This generalizes
             the scalar 1/ntrial term: with no missing data n_i=n_j=n_ij=T for
             all conditions so alpha = 1/T everywhere.

The biconvex iteration uses the exact per-entry alpha for the cSb step; its
regularizer coefficients use a single effective scalar ntrial (mean clean
count), which also collapses to the standard coefficients when complete.

Shrinkage levels are selected on the complete-data subset (trials with no
missing unit for cN; conditions with all units defined for cD) by reusing the
existing tested selectors, then applied to the missing-units covariances; on
complete data this matches the standard selection exactly.

This is a NEW estimator with no MATLAB reference (MATLAB GSN does whole-trial
min-truncation for uneven data); correctness is pinned by (a) exact reduction
to the even path on complete data and (b) a brute-force pairwise reference.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np

from gsn.fast_perform_gsn import (
    torch,
    _HAS_TORCH,
    _nearest_psd_numpy,
    _nearest_psd_torch,
    _eigh_descending_torch,
    _torch_dtype_for,
    _shrunken_cov_2d_numpy,
    _shrunken_noise_cov_uneven_numpy,
    _uneven_validity,
    _eigh_descending_numpy,
    _normalize_returns,
)


# ---------------------------------------------------------------------------
# Missing-units covariance primitives
# ---------------------------------------------------------------------------

def _pairwise_cov(Xm, Mb):
    """Unbiased pairwise-complete covariance for one block.

    Xm (nvox, T) = data with non-present entries zeroed; Mb (nvox, T) = float
    presence mask. Each pair (i,j) is centered on the mean over the trials they
    SHARE (not each unit's own mean), so the estimate is unbiased for every
    overlap pattern:

        cov[i,j] = ( S2[i,j] - Si[i,j]*Si[j,i]/K[i,j] ) / (K[i,j] - 1),  K>=2

    with S2 = Xm Xm^T, Si = Xm Mb^T, K = Mb Mb^T. Returns (cov, K). Entries with
    K < 2 are 0 in cov (and the caller gates on K).
    """
    S2 = Xm @ Xm.T
    Si = Xm @ Mb.T                                     # Si[i,j] = sum_{common} x_i
    K = Mb @ Mb.T                                       # common-present counts
    Kpos = np.where(K > 0, K, 1.0)
    num = S2 - Si * Si.T / Kpos                         # Si[i,j]*Si[j,i] = (sum x_i)(sum x_j)
    cov = np.divide(num, K - 1.0, out=np.zeros_like(num), where=K >= 2)
    return cov, K


def _missing_cn_alpha(data, M, conds):
    """Missing-units noise covariance + per-entry bias factor alpha.

    data (nvox, ncond, ntrial) with NaN at missing (unit, cond, trial);
    M (nvox, ncond, ntrial) bool validity; conds = condition indices to pool.
    cN[i,j] = average over conditions (with >=2 shared-clean trials) of the
    UNBIASED pairwise covariance of i,j over their shared-clean trials.
    Returns (cN (nvox, nvox), alpha (nvox, nvox)).
    """
    nvox = data.shape[0]
    cN_acc = np.zeros((nvox, nvox)); cN_cnt = np.zeros((nvox, nvox))
    a_num = np.zeros((nvox, nvox)); a_cnt = np.zeros((nvox, nvox))
    for c in conds:
        Mb = M[:, c, :].astype(np.float64)            # (nvox, ntrial)
        Xm = np.where(M[:, c, :], data[:, c, :], 0.0)  # NaN -> 0
        cov, Nc = _pairwise_cov(Xm, Mb)
        ge2 = Nc >= 2
        cN_acc += np.where(ge2, cov, 0.0)
        cN_cnt += ge2
        ni = Mb.sum(1)                                 # clean trials per unit
        defined = ni >= 1
        both = np.outer(defined, defined)
        term = np.divide(Nc, np.outer(ni, ni),
                         out=np.zeros_like(Nc), where=both)
        a_num += np.where(both, term, 0.0)
        a_cnt += both
    cN = np.divide(cN_acc, cN_cnt, out=np.zeros_like(cN_acc), where=cN_cnt > 0)
    alpha = np.divide(a_num, a_cnt, out=np.zeros_like(a_num), where=a_cnt > 0)
    return cN, alpha


def _condition_means(data, M):
    """Per-unit condition means over clean trials. Returns (CM (nvox, ncond)
    with NaN where a unit has no clean trial, Dmask (nvox, ncond) bool defined,
    n_ic (nvox, ncond) clean-trial counts)."""
    n_ic = M.sum(2)                                    # (nvox, ncond)
    x0 = np.where(M, data, 0.0)
    CM = np.divide(x0.sum(2), n_ic,
                   out=np.full(n_ic.shape, np.nan, float), where=n_ic >= 1)
    return CM, (n_ic >= 1), n_ic


def _missing_cov2d(CM, Dmask):
    """Missing-units covariance across the columns (conditions) of CM.

    CM (nvox, ncond) with NaN where undefined; Dmask bool same shape.
    cD[i,j] is the UNBIASED pairwise covariance over conditions where both
    units are defined, centered on each pair's shared-defined conditions.
    """
    CMm = np.where(Dmask, CM, 0.0)
    cov, _ = _pairwise_cov(CMm, Dmask.astype(np.float64))
    return cov


def _shrink_to_diag(c, level):
    """Apply shrinkage: scale off-diagonal by level, keep diagonal."""
    out = c * level
    if out.shape[0] > 1:
        np.fill_diagonal(out, np.diag(c))
    return out


# ---------------------------------------------------------------------------
# Shrinkage-level selection on the complete-data subset (reuses tested code)
# ---------------------------------------------------------------------------

def _level_complete_trials(data, M, shrinklevels):
    if len(shrinklevels) == 1:
        return float(shrinklevels[0])
    complete = M.all(axis=0)                            # (ncond, ntrial)
    dc = np.where(complete[None, :, :], data, np.nan)   # non-complete trials -> whole NaN
    try:
        valid, validcnt = _uneven_validity(dc)
        _, _, lvl = _shrunken_noise_cov_uneven_numpy(dc, valid, validcnt, 5, shrinklevels)
        return lvl
    except Exception:
        warnings.warn('missing-units: cN shrink level fell back to 1.0 '
                      '(too few complete trials)')
        return 1.0


def _level_complete_conds(CM, Dmask, shrinklevels):
    if len(shrinklevels) == 1:
        return float(shrinklevels[0])
    comp = Dmask.all(axis=0)                            # conditions with all units defined
    if int(comp.sum()) < 5:
        warnings.warn('missing-units: cD shrink level fell back to 1.0 '
                      '(too few complete conditions)')
        return 1.0
    try:
        _, _, lvl = _shrunken_cov_2d_numpy(CM[:, comp].T, 5, shrinklevels)
        return lvl
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Biconvex with per-entry alpha
# ---------------------------------------------------------------------------

def _biconvex_missing(cN, cD, alpha, ncond, ntrial_eff, max_iters=100):
    """Biconvex cSb/cNb with the per-entry bias factor alpha in the cSb step
    and an effective scalar ntrial in the regularizer coefficients. Reduces to
    _biconvex_numpy when alpha == 1/ntrial and ntrial_eff == ntrial."""
    e = float(ntrial_eff)
    coef_N = ((ncond * (e - 1) * e ** 2)
              / (ncond * e ** 2 * (e - 1) + ncond - 1))
    coef_D = ((ncond - 1)
              / (ncond * e ** 2 * (e - 1) + ncond - 1))
    cNb = cN
    cSb_old = cD - cN * alpha
    cNb_old = cN
    numiters = 0
    for _ in range(max_iters):
        cSb = _nearest_psd_numpy(cD - cNb * alpha)
        cNb = _nearest_psd_numpy(coef_N * cN + coef_D * e * (cD - cSb))
        if cSb.shape[0] == 1:
            converged = (abs(cSb_old - cSb) < 1e-5 and abs(cNb_old - cNb) < 1e-5)
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


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def run_missing_units_numpy(data, opt) -> Dict[str, Any]:
    nvox, ncond, _ = data.shape
    shrinklevels = (np.linspace(0, 1, 51) if opt.get('wantshrinkage', True)
                    else np.array([1.0]))
    returns = _normalize_returns(opt.get('returns'))
    M = ~np.isnan(data)
    allc = np.arange(ncond)

    cN_raw, alpha = _missing_cn_alpha(data, M, allc)
    CM, Dmask, n_ic = _condition_means(data, M)
    cD_raw = _missing_cov2d(CM, Dmask)

    lvlN = _level_complete_trials(data, M, shrinklevels)
    lvlD = _level_complete_conds(CM, Dmask, shrinklevels)
    cN = _shrink_to_diag(cN_raw, lvlN)
    cD = _shrink_to_diag(cD_raw, lvlD)

    cS = cD - cN * alpha
    sd_noise = np.sqrt(np.maximum(np.diag(cN), 0))
    sd_signal = np.sqrt(np.maximum(np.diag(cS), 0))
    ncsnr = np.divide(sd_signal, sd_noise,
                      out=np.zeros_like(sd_signal), where=sd_noise != 0)

    ntrial_eff = float(np.mean(n_ic[Dmask])) if Dmask.any() else 1.0
    cSb, cNb, numiters = _biconvex_missing(cN, cD, alpha, ncond, ntrial_eff)

    mnN = np.zeros((1, nvox))
    mnD = np.nanmean(np.where(Dmask, CM, np.nan), axis=1)[None, :]
    mnS = mnD - mnN

    result: Dict[str, Any] = {
        'mnN': mnN, 'shrinklevelN': lvlN, 'mnS': mnS, 'shrinklevelD': lvlD,
        'ncsnr': ncsnr, 'numiters': numiters,
    }
    if 'cN' in returns: result['cN'] = cN
    if 'cS' in returns: result['cS'] = cS
    if 'cNb' in returns: result['cNb'] = cNb
    if 'cSb' in returns: result['cSb'] = cSb
    want_es = 'eigvecs_signal'     in returns or 'eigvals_signal'     in returns
    want_ed = 'eigvecs_difference' in returns or 'eigvals_difference' in returns
    if want_es:
        d, V = _eigh_descending_numpy(cSb)
        if 'eigvals_signal' in returns: result['eigvals_signal'] = d
        if 'eigvecs_signal' in returns: result['eigvecs_signal'] = V
    if want_ed:
        d, V = _eigh_descending_numpy(cSb - cNb * alpha)
        if 'eigvals_difference' in returns: result['eigvals_difference'] = d
        if 'eigvecs_difference' in returns: result['eigvecs_difference'] = V
    return result


# ===========================================================================
# Torch path — same estimator on device. Heavy covariance loop + biconvex run
# on the active device; shrinkage levels are picked on the host (cheap scalars,
# reuses the numpy selectors) and applied on device.
# ===========================================================================

def _pairwise_cov_torch(Xm, Mb):
    """Torch version of _pairwise_cov. Returns (cov, K)."""
    S2 = Xm @ Xm.T
    Si = Xm @ Mb.T
    K = Mb @ Mb.T
    Kpos = torch.where(K > 0, K, torch.ones_like(K))
    num = S2 - Si * Si.T / Kpos
    Km1 = torch.where(K > 1, K - 1.0, torch.ones_like(K))
    cov = torch.where(K >= 2, num / Km1, torch.zeros_like(num))
    return cov, K


def _missing_cn_alpha_torch(data, M, device, dtype):
    """Torch version of _missing_cn_alpha over all conditions."""
    nvox, ncond, _ = data.shape
    z = lambda: torch.zeros((nvox, nvox), dtype=dtype, device=device)
    cN_acc, cN_cnt, a_num, a_cnt = z(), z(), z(), z()
    zero = torch.zeros((), dtype=dtype, device=device)
    for c in range(ncond):
        Mb = M[:, c, :].to(dtype)
        Xm = torch.where(M[:, c, :], data[:, c, :], zero)
        cov, Nc = _pairwise_cov_torch(Xm, Mb)
        ge2 = Nc >= 2
        cN_acc += torch.where(ge2, cov, torch.zeros_like(cov))
        cN_cnt += ge2.to(dtype)
        ni = Mb.sum(1)
        defined = ni >= 1
        both = defined[:, None] & defined[None, :]
        denom = ni[:, None] * ni[None, :]
        term = torch.where(both, Nc / torch.where(denom > 0, denom, torch.ones_like(denom)),
                           torch.zeros_like(Nc))
        a_num += term
        a_cnt += both.to(dtype)
    cN = torch.where(cN_cnt > 0, cN_acc / torch.where(cN_cnt > 0, cN_cnt, torch.ones_like(cN_cnt)),
                     torch.zeros_like(cN_acc))
    alpha = torch.where(a_cnt > 0, a_num / torch.where(a_cnt > 0, a_cnt, torch.ones_like(a_cnt)),
                        torch.zeros_like(a_num))
    return cN, alpha


def _condition_means_torch(data, M, dtype):
    n_ic = M.sum(2).to(dtype)
    zero = torch.zeros((), dtype=dtype, device=data.device)
    x0 = torch.where(M, data, zero)
    CM = x0.sum(2) / torch.where(n_ic >= 1, n_ic, torch.ones_like(n_ic))
    Dmask = n_ic >= 1
    return CM, Dmask, n_ic


def _missing_cov2d_torch(CM, Dmask, dtype):
    zero = torch.zeros((), dtype=dtype, device=CM.device)
    CMm = torch.where(Dmask, CM, zero)
    cov, _ = _pairwise_cov_torch(CMm, Dmask.to(dtype))
    return cov


def _shrink_to_diag_torch(c, level):
    out = c * level
    if out.shape[0] > 1:
        idx = torch.arange(out.shape[0], device=out.device)
        out[idx, idx] = torch.diagonal(c)
    return out


def _biconvex_missing_torch(cN, cD, alpha, ncond, ntrial_eff, max_iters=100):
    e = float(ntrial_eff)
    coef_N = ((ncond * (e - 1) * e ** 2) / (ncond * e ** 2 * (e - 1) + ncond - 1))
    coef_D = ((ncond - 1) / (ncond * e ** 2 * (e - 1) + ncond - 1))
    cNb = cN
    cSb_old = cD - cN * alpha
    cNb_old = cN
    numiters = 0
    for _ in range(max_iters):
        cSb = _nearest_psd_torch(cD - cNb * alpha)
        cNb = _nearest_psd_torch(coef_N * cN + coef_D * e * (cD - cSb))
        if cSb.shape[0] == 1:
            converged = (torch.abs(cSb_old - cSb).item() < 1e-5
                         and torch.abs(cNb_old - cNb).item() < 1e-5)
        else:
            r_S = _flat_corr(cSb_old, cSb)
            r_N = _flat_corr(cNb_old, cNb)
            converged = r_S > 0.999 and r_N > 0.999
        if converged:
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


def _flat_corr(a, b):
    af, bf = a.flatten(), b.flatten()
    af = af - af.mean(); bf = bf - bf.mean()
    d = torch.sqrt((af * af).sum() * (bf * bf).sum())
    return float((af * bf).sum() / d) if d > 0 else 0.0


def run_missing_units_torch(data_np, opt, device) -> Dict[str, Any]:
    dtype = _torch_dtype_for(data_np, device)
    nvox, ncond, _ = data_np.shape
    shrinklevels = (np.linspace(0, 1, 51) if opt.get('wantshrinkage', True)
                    else np.array([1.0]))
    returns = _normalize_returns(opt.get('returns'))

    M_np = ~np.isnan(data_np)
    # shrinkage levels on the host (cheap scalars; reuse numpy selectors)
    CM_np, Dmask_np, _ = _condition_means(data_np, M_np)
    lvlN = _level_complete_trials(data_np, M_np, shrinklevels)
    lvlD = _level_complete_conds(CM_np, Dmask_np, shrinklevels)
    mnD = np.nanmean(np.where(Dmask_np, CM_np, np.nan), axis=1)[None, :]

    data = torch.as_tensor(data_np, dtype=dtype, device=device)
    M = torch.as_tensor(M_np, device=device)
    cN_raw, alpha = _missing_cn_alpha_torch(data, M, device, dtype)
    CM, Dmask, n_ic = _condition_means_torch(data, M, dtype)
    del data
    cD_raw = _missing_cov2d_torch(CM, Dmask, dtype)
    cN = _shrink_to_diag_torch(cN_raw, lvlN)
    cD = _shrink_to_diag_torch(cD_raw, lvlD)
    cS = cD - cN * alpha
    sd_noise = torch.sqrt(torch.clamp(torch.diagonal(cN), min=0))
    sd_signal = torch.sqrt(torch.clamp(torch.diagonal(cS), min=0))
    ncsnr = torch.where(sd_noise > 0, sd_signal / sd_noise, torch.zeros_like(sd_signal))
    ntrial_eff = float(n_ic[Dmask].mean().item()) if bool(Dmask.any()) else 1.0
    cSb, cNb, numiters = _biconvex_missing_torch(cN, cD, alpha, ncond, ntrial_eff)

    result: Dict[str, Any] = {
        'mnN': np.zeros((1, nvox)), 'shrinklevelN': lvlN,
        'mnS': mnD, 'shrinklevelD': lvlD,
        'ncsnr': ncsnr.cpu().numpy(), 'numiters': numiters,
    }
    if 'cN' in returns: result['cN'] = cN.cpu().numpy()
    if 'cS' in returns: result['cS'] = cS.cpu().numpy()
    if 'cNb' in returns: result['cNb'] = cNb.cpu().numpy()
    if 'cSb' in returns: result['cSb'] = cSb.cpu().numpy()
    want_es = 'eigvecs_signal'     in returns or 'eigvals_signal'     in returns
    want_ed = 'eigvecs_difference' in returns or 'eigvals_difference' in returns
    if want_es:
        d, V = _eigh_descending_torch(cSb)
        if 'eigvals_signal' in returns: result['eigvals_signal'] = d.cpu().numpy()
        if 'eigvecs_signal' in returns: result['eigvecs_signal'] = V.cpu().numpy()
    if want_ed:
        diff = cSb - cNb * alpha
        d, V = _eigh_descending_torch(diff)
        if 'eigvals_difference' in returns: result['eigvals_difference'] = d.cpu().numpy()
        if 'eigvecs_difference' in returns: result['eigvecs_difference'] = V.cpu().numpy()
    return result
