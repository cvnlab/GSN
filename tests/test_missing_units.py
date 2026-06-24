"""Tests for missing-units GSN (per-unit missing data).

Correctness is pinned three ways:
  1. EXACT reduction to the even (complete-data) path when nothing is missing.
  2. Brute-force, triple-nested-loop references for cN, cD, and alpha on small
     synthetic per-unit-missing data.
  3. Structural guarantees (symmetry, PSD of cSb/cNb, alpha edge cases).
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pytest

from gsn.fast_perform_gsn import fast_perform_gsn, _run_numpy, _biconvex_numpy
from gsn.missing_units import (
    run_missing_units_numpy, _missing_cn_alpha, _condition_means,
    _missing_cov2d, _biconvex_missing,
)

RET = ['cN', 'cS', 'cNb', 'cSb', 'eigvecs_signal', 'eigvals_signal',
       'eigvecs_difference', 'eigvals_difference']


# --------------------------------------------------------------------------
# helpers / data
# --------------------------------------------------------------------------

def complete_data(nvox=6, ncond=40, ntrial=6, seed=0, latent=3):
    rng = np.random.RandomState(seed)
    sig = rng.randn(nvox, latent) @ rng.randn(latent, ncond)
    return (sig[:, :, None] + 0.8 * rng.randn(nvox, ncond, ntrial)).astype(np.float64)


def punch_holes(data, frac, seed=1):
    """Randomly NaN individual (unit, cond, trial) entries (per-unit missing),
    guaranteeing every unit keeps >=2 trials in >=2 conditions."""
    rng = np.random.RandomState(seed)
    d = data.copy()
    mask = rng.rand(*d.shape) < frac
    d[mask] = np.nan
    # safety: ensure each (unit) has some data and pairs can overlap
    nvox, ncond, ntrial = d.shape
    for i in range(nvox):
        M = ~np.isnan(d[i])
        if M.sum(1).max() < 2:        # force >=2 trials in cond 0
            d[i, 0, :2] = data[i, 0, :2]
    return d


# --------------------------------------------------------------------------
# brute-force references (deliberately slow + explicit)
# --------------------------------------------------------------------------

def brute_alpha(M):
    nvox, ncond, _ = M.shape
    A = np.zeros((nvox, nvox))
    for i in range(nvox):
        for j in range(nvox):
            s, n = 0.0, 0
            for c in range(ncond):
                ni, nj = M[i, c].sum(), M[j, c].sum()
                if ni >= 1 and nj >= 1:
                    nij = (M[i, c] & M[j, c]).sum()
                    s += nij / (ni * nj); n += 1
            A[i, j] = s / n if n else 0.0
    return A


def brute_cN(data, M):
    """Pairwise-complete: each pair centered on its SHARED-clean trials."""
    nvox, ncond, _ = data.shape
    acc = np.zeros((nvox, nvox)); cnt = np.zeros((nvox, nvox))
    for c in range(ncond):
        for i in range(nvox):
            for j in range(nvox):
                common = M[i, c] & M[j, c]
                nij = common.sum()
                if nij >= 2:
                    xi = data[i, c, common]; xj = data[j, c, common]
                    ri = xi - xi.mean(); rj = xj - xj.mean()
                    acc[i, j] += (ri * rj).sum() / (nij - 1)
                    cnt[i, j] += 1
    return np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)


def brute_cD(data, M):
    """Pairwise-complete over conditions: each pair centered on its
    shared-defined conditions."""
    nvox, ncond, _ = data.shape
    CM = np.full((nvox, ncond), np.nan)
    for i in range(nvox):
        for c in range(ncond):
            if M[i, c].any():
                CM[i, c] = data[i, c, M[i, c]].mean()
    D = ~np.isnan(CM)
    cD = np.zeros((nvox, nvox))
    for i in range(nvox):
        for j in range(nvox):
            common = D[i] & D[j]
            n = common.sum()
            if n >= 2:
                xi = CM[i, common]; xj = CM[j, common]
                cD[i, j] = ((xi - xi.mean()) * (xj - xj.mean())).sum() / (n - 1)
    return cD


# --------------------------------------------------------------------------
# 1. exact reduction to even path on complete data
# --------------------------------------------------------------------------

@pytest.mark.parametrize('shrink', [False, True])
def test_reduces_to_even_path_on_complete_data(shrink):
    for seed in range(3):
        data = complete_data(seed=seed)
        opt = {'returns': RET, 'wantshrinkage': shrink}
        even = _run_numpy(data.copy(), opt)
        avail = run_missing_units_numpy(data.copy(), {**opt, 'uneven': 'missing'})
        for k in ('cN', 'cS', 'cNb', 'cSb', 'ncsnr', 'mnN', 'mnS',
                  'eigvals_signal', 'eigvals_difference'):
            assert np.allclose(np.asarray(even[k], float), np.asarray(avail[k], float),
                               atol=1e-7, rtol=1e-6), f'{k} (shrink={shrink}, seed={seed})'
        assert even['shrinklevelN'] == avail['shrinklevelN']
        assert even['shrinklevelD'] == avail['shrinklevelD']
        # signal subspace agrees (eigvecs unique up to null-space rotation)
        dr, Vr = even['eigvals_signal'], even['eigvecs_signal']
        rank = int((dr > 1e-9 * max(abs(dr).max(), 1)).sum())
        Pe = Vr[:, :rank] @ Vr[:, :rank].T
        Va = avail['eigvecs_signal']
        Pa = Va[:, :rank] @ Va[:, :rank].T
        assert np.allclose(Pe, Pa, atol=1e-6)


# --------------------------------------------------------------------------
# 1b. UNBIASEDNESS under partial overlap (the test that catches the
#     available-means bias: biased estimator would inflate off-diagonals by
#     k/(k-1)*(1-1/n_i-1/n_j+k/(n_i n_j)) = 10/9 here).
# --------------------------------------------------------------------------

def test_cn_unbiased_under_partial_overlap():
    rng = np.random.RandomState(0)
    nvox, ntrial, ncond = 4, 4, 20000
    A = rng.randn(nvox, 3)
    SN = A @ A.T + 0.5 * np.eye(nvox)           # known noise covariance
    L = np.linalg.cholesky(SN)
    # fixed partial-overlap mask: each unit clean on 3 of 4 trials, every
    # off-diagonal pair shares exactly k=2 (n_i = n_j = 3) -> stresses the bias
    Mfix = np.array([[1, 1, 1, 0],
                     [0, 1, 1, 1],
                     [1, 0, 1, 1],
                     [1, 1, 0, 1]], bool)
    Z = rng.randn(nvox, ncond, ntrial)
    data = np.einsum('ij,jct->ict', L, Z)       # noise only, signal = 0
    data = np.where(Mfix[:, None, :], data, np.nan)
    M = ~np.isnan(data)
    cN, _ = _missing_cn_alpha(data, M, np.arange(ncond))
    # unbiased -> converges to SN (off-diagonals NOT inflated by ~11%)
    rel = np.max(np.abs(cN - SN)) / np.max(np.abs(SN))
    assert rel < 0.05, f'cN not unbiased under partial overlap (rel err {rel:.3f})'
    # the biased available-means estimator would put off-diagonals at ~10/9*SN;
    # confirm we are nowhere near that
    od = ~np.eye(nvox, dtype=bool)
    biased = (10.0 / 9.0) * SN
    assert (np.abs(cN - SN)[od].mean() < np.abs(biased - SN)[od].mean() * 0.5)


# --------------------------------------------------------------------------
# 2. brute-force references for cN / cD / alpha
# --------------------------------------------------------------------------

@pytest.mark.parametrize('frac', [0.1, 0.25, 0.4])
def test_cn_alpha_match_bruteforce(frac):
    data = punch_holes(complete_data(nvox=7, ncond=30, ntrial=6, seed=2), frac, seed=frac_to_seed(frac))
    M = ~np.isnan(data)
    cN, alpha = _missing_cn_alpha(data, M, np.arange(data.shape[1]))
    assert np.allclose(cN, brute_cN(data, M), atol=1e-10), 'cN vs brute'
    assert np.allclose(alpha, brute_alpha(M), atol=1e-12), 'alpha vs brute'
    assert np.allclose(cN, cN.T, atol=1e-12), 'cN symmetric'
    assert np.allclose(alpha, alpha.T, atol=1e-12), 'alpha symmetric'


@pytest.mark.parametrize('frac', [0.1, 0.25, 0.4])
def test_cd_match_bruteforce(frac):
    data = punch_holes(complete_data(nvox=7, ncond=30, ntrial=6, seed=3), frac, seed=frac_to_seed(frac) + 9)
    M = ~np.isnan(data)
    CM, Dmask, _ = _condition_means(data, M)
    cD = _missing_cov2d(CM, Dmask)
    assert np.allclose(cD, brute_cD(data, M), atol=1e-10), 'cD vs brute'
    assert np.allclose(cD, cD.T, atol=1e-12), 'cD symmetric'


def frac_to_seed(frac):
    return int(round(frac * 100))


# --------------------------------------------------------------------------
# 3. alpha closed-form checks (the worked examples from discussion)
# --------------------------------------------------------------------------

def test_alpha_one_over_ntrial_when_complete():
    M = np.ones((4, 5, 6), bool)              # all clean, 6 trials
    _, alpha = _missing_cn_alpha(np.zeros((4, 5, 6)), M, np.arange(5))
    assert np.allclose(alpha, 1.0 / 6.0)


def test_alpha_partial_overlap_one_ninth():
    # one condition, 5 trials; unit A clean on {0,1,2}, unit B clean on {2,3,4}
    M = np.zeros((2, 1, 5), bool)
    M[0, 0, [0, 1, 2]] = True
    M[1, 0, [2, 3, 4]] = True
    _, alpha = _missing_cn_alpha(np.zeros((2, 1, 5)), M, np.arange(1))
    # n_i=3, n_j=3, common={2} -> n_ij=1 -> 1/(3*3)=1/9
    assert alpha[0, 1] == pytest.approx(1.0 / 9.0)
    assert alpha[0, 0] == pytest.approx(1.0 / 3.0)   # diagonal = 1/n


def test_alpha_zero_when_no_shared_trial():
    M = np.zeros((2, 1, 6), bool)
    M[0, 0, [0, 1, 2]] = True
    M[1, 0, [3, 4, 5]] = True                  # disjoint
    _, alpha = _missing_cn_alpha(np.zeros((2, 1, 6)), M, np.arange(1))
    assert alpha[0, 1] == 0.0


# --------------------------------------------------------------------------
# 4. biconvex reduction + PSD
# --------------------------------------------------------------------------

def test_biconvex_missing_reduces_to_scalar():
    rng = np.random.RandomState(0)
    A = rng.randn(5, 8); cN = A @ A.T / 8
    B = rng.randn(5, 8); cD = B @ B.T / 8 + cN / 6
    T = 6
    alpha = np.full((5, 5), 1.0 / T)
    cSb1, cNb1, n1 = _biconvex_missing(cN, cD, alpha, 40, T)
    cSb2, cNb2, n2 = _biconvex_numpy(cN, cD, cD - cN / T, 40, T)
    assert np.allclose(cSb1, cSb2, atol=1e-10)
    assert np.allclose(cNb1, cNb2, atol=1e-10)
    assert n1 == n2


def test_outputs_psd_and_symmetric():
    data = punch_holes(complete_data(nvox=8, ncond=40, ntrial=6, seed=5), 0.2, seed=7)
    r = run_missing_units_numpy(data, {'returns': ['cN', 'cS', 'cNb', 'cSb']})
    for k in ('cN', 'cSb', 'cNb'):
        M = r[k]
        assert np.allclose(M, M.T, atol=1e-9), f'{k} symmetric'
        ev = np.linalg.eigvalsh((M + M.T) / 2)
        assert ev.min() > -1e-8, f'{k} PSD (min eig {ev.min():.2e})'


# --------------------------------------------------------------------------
# 4b. torch path matches numpy
# --------------------------------------------------------------------------

def test_torch_matches_numpy():
    from gsn.fast_perform_gsn import _HAS_TORCH, _resolve_device
    if not _HAS_TORCH:
        import pytest as _pt
        _pt.skip('torch not installed')
    from gsn.missing_units import run_missing_units_torch
    for seed in range(2):
        data = punch_holes(complete_data(nvox=8, ncond=30, ntrial=6, seed=seed), 0.25,
                           seed=seed + 3)
        opt = {'returns': RET}
        rn = run_missing_units_numpy(data.copy(), opt)
        rt = run_missing_units_torch(data.copy(), opt, _resolve_device('cpu'))
        for k in ('cN', 'cS', 'cNb', 'cSb', 'ncsnr', 'mnN', 'mnS',
                  'eigvals_signal', 'eigvals_difference'):
            assert np.allclose(np.asarray(rn[k], float), np.asarray(rt[k], float),
                               atol=1e-7, rtol=1e-6), f'{k} (seed={seed})'
        assert rn['shrinklevelN'] == rt['shrinklevelN']
        assert rn['shrinklevelD'] == rt['shrinklevelD']
        dr, Vr = rn['eigvals_signal'], rn['eigvecs_signal']
        rank = int((dr > 1e-9 * max(abs(dr).max(), 1)).sum())
        Pn = Vr[:, :rank] @ Vr[:, :rank].T
        Vt = rt['eigvecs_signal']
        Pt = Vt[:, :rank] @ Vt[:, :rank].T
        assert np.allclose(Pn, Pt, atol=1e-6)


# --------------------------------------------------------------------------
# 5. routing + uses-all-data sanity
# --------------------------------------------------------------------------

def test_routing_via_fast_perform_gsn():
    data = punch_holes(complete_data(nvox=6, ncond=30, ntrial=6, seed=8), 0.2, seed=11)
    r = fast_perform_gsn(data, {'returns': ['cN', 'cS', 'cNb', 'cSb'],
                                'uneven': 'missing', 'device': 'cpu'})
    for k in ('cN', 'cS', 'cNb', 'cSb', 'ncsnr', 'mnN', 'mnS'):
        assert k in r
        assert np.isfinite(np.asarray(r[k], float)).all(), f'{k} finite'


def test_uses_more_data_than_whole_trial():
    # per-unit missing where almost every trial has SOME missing unit:
    # whole-trial path would drop nearly everything; missing-units keeps it.
    data = complete_data(nvox=10, ncond=40, ntrial=6, seed=4)
    rng = np.random.RandomState(0)
    # knock out one random unit on most trials -> few fully-complete trials
    for c in range(40):
        for t in range(6):
            if rng.rand() < 0.7:
                data[rng.randint(10), c, t] = np.nan
    M = ~np.isnan(data)
    frac_complete_trials = np.mean(M.all(axis=0))
    assert frac_complete_trials < 0.5     # whole-trial would be starved
    cN, _ = _missing_cn_alpha(data, M, np.arange(40))
    # every pair still estimated (no NaN / all-zero rows) thanks to missing-units
    assert np.isfinite(cN).all()
    assert (np.abs(cN).sum(1) > 0).all()
