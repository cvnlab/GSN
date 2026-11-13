"""The noise shrinkage level must come from held-out residuals scaled back to the noise variance.

Even trials: the reference is the exact likelihood of each condition's t-1 orthonormal (Helmert) contrasts.
Uneven trials: the reference scales each condition's mean-subtracted trials by sqrt(t/(t-1)), t = its valid trials.
Scoring raw mean-subtracted residuals instead picks too little shrinkage at few trials.
"""
import warnings

import numpy as np
import pytest
from scipy.linalg import helmert

from gsn.batched_nll import _HAS_TORCH, _numpy_loop
from gsn.calc_shrunken_covariance import calc_shrunken_covariance
from gsn.fast_perform_gsn import fast_perform_gsn
from gsn.utilities import deterministic_randperm

LEVELS = np.linspace(0, 1, 51)


def _simulate(nunits, nconds, ntrials, seed, correlated=True):
    """Signal plus noise; the noise is either correlated across units (1/sqrt(i) spectrum) or independent."""
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal((nunits, nconds))
    if correlated:
        U = np.linalg.qr(rng.standard_normal((nunits, nunits)))[0]
        L = U * (3 / np.sqrt(np.arange(1, nunits + 1)))
    else:
        L = np.diag(rng.uniform(0.5, 2, nunits))
    noise = np.einsum('ij,jct->ict', L, rng.standard_normal((nunits, nconds, ntrials)))
    return signal[:, :, None] + noise


def _drop_trials(data, seed):
    """Set random whole trials to NaN, keeping at least one trial per condition."""
    data = data.copy()
    rng = np.random.default_rng(seed)
    for c in range(data.shape[1]):
        mask = rng.random(data.shape[2]) < 0.3
        if mask.all():
            mask[0] = False
        data[:, c, mask] = np.nan
    return data


def _valid(data, c):
    x = data[:, c, :]
    return x[:, ~np.isnan(x).any(axis=0)]


def _reference_level(data):
    """Reference level, using the package's split of conditions and its training covariance."""
    nconds = data.shape[1]
    perm = deterministic_randperm(nconds)
    nval = int(np.round(nconds / 5))
    val, train = perm[:nval], perm[nval:]
    c_train = np.mean([np.cov(_valid(data, c)) for c in train if _valid(data, c).shape[1] > 1], axis=0)
    rows = []
    for c in val:
        x = _valid(data, c)
        t = x.shape[1]
        if t < 2:
            continue
        if np.isnan(data).any():
            rows.append((x - x.mean(axis=1, keepdims=True)).T * np.sqrt(t / (t - 1)))
        else:
            rows.append((x @ helmert(t).T).T)
    return float(LEVELS[np.nanargmin(_numpy_loop(c_train, np.vstack(rows), LEVELS))])


def _level(data, **opt):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(fast_perform_gsn(data, opt)['shrinklevelN'])


SHAPES = [(50, 100, 2), (50, 100, 3), (50, 100, 5), (100, 60, 3)]


@pytest.mark.parametrize('shape', SHAPES)
def test_level_matches_exact_likelihood(shape):
    data = _simulate(*shape, seed=0)
    assert _level(data, backend='numpy') == _reference_level(data)


def test_correlated_noise_at_two_trials_is_shrunk_enough():
    """50 units x 100 conditions x 2 trials: exact likelihood picks 0.64-0.72; raw residuals picked 0.86-0.90."""
    for seed in range(3):
        assert _level(_simulate(50, 100, 2, seed), backend='numpy') <= 0.75


@pytest.mark.parametrize('shape', [(50, 100, 3), (50, 100, 5)])
def test_level_matches_reference_uneven_trials(shape):
    data = _drop_trials(_simulate(*shape, seed=1), seed=2)
    ref = _reference_level(data)
    assert _level(data, backend='numpy') == ref
    assert _level(data, uneven='reference') == ref
    assert _level(data, uneven='missing', backend='numpy') == ref


@pytest.mark.parametrize('uneven', [False, True])
def test_legacy_estimator_matches_reference(uneven):
    data = _simulate(50, 100, 3, seed=3)
    if uneven:
        data = _drop_trials(data, seed=4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, _, level, _ = calc_shrunken_covariance(np.transpose(data, (2, 0, 1)), wantfull=1)
    assert float(level) == _reference_level(data)


@pytest.mark.skipif(not _HAS_TORCH, reason='torch not installed')
@pytest.mark.parametrize('uneven', [False, True])
def test_torch_level_matches_reference(uneven):
    data = _simulate(50, 100, 3, seed=5)
    if uneven:
        data = _drop_trials(data, seed=6)
    assert _level(data, backend='torch') == _reference_level(data)
