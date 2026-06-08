"""Regression tests for fast_perform_gsn / batched_nll bug fixes."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pytest

from gsn.batched_nll import batched_shrunken_nll
from gsn.calc_shrunken_covariance import calc_shrunken_covariance
from gsn.fast_perform_gsn import fast_perform_gsn


def test_one_variable_torch_nll_no_alias_crash():
    """N==1 diagonal assignment used to raise a torch memory-alias error."""
    batched_shrunken_nll(np.array([[2.0]]), np.random.RandomState(0).randn(10, 1),
                         np.array([1.0]), use_torch=True, device='cpu')
    calc_shrunken_covariance(np.random.RandomState(0).randn(10, 1),
                             shrinklevels=np.array([1.0]))
    r = fast_perform_gsn(np.random.RandomState(0).randn(1, 6, 3), {'device': 'cpu'})
    assert np.isfinite(r['ncsnr']).all()


@pytest.mark.parametrize('level', [0.0, 0.3, 1.0])
def test_honors_user_shrinklevels(level):
    """fast path must use opt['shrinklevels'] (it previously ignored it)."""
    data = np.random.default_rng(3).normal(size=(3, 8, 4))
    r = fast_perform_gsn(data, {'wantshrinkage': 1, 'shrinklevels': np.array([level]),
                                'device': 'cpu', 'returns': ['cN', 'cS']})
    assert r['shrinklevelN'] == level
    assert r['shrinklevelD'] == level


def test_degenerate_validation_split_asserts():
    """A held-out condition with <2 valid trials must assert, not silently
    pick shrink level 0 (matches the reference)."""
    data = np.random.default_rng(0).normal(size=(3, 6, 3))
    data[:, 5, 1:] = np.nan          # cond 5 (held out by deterministic_randperm) -> 1 valid trial
    with pytest.raises(AssertionError):
        fast_perform_gsn(data, {'wantshrinkage': 1, 'wantverbose': 0,
                                'device': 'cpu', 'returns': ['cN', 'cS']})


def test_ntrial_guard():
    with pytest.raises(ValueError):
        fast_perform_gsn(np.random.RandomState(0).randn(3, 6, 1), {'device': 'cpu'})
    with pytest.raises(ValueError):
        fast_perform_gsn(np.random.RandomState(0).randn(3, 6), {'device': 'cpu'})
