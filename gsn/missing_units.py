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
    _nearest_psd_numpy,
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
