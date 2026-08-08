"""Verification tests for OptFom_Setup_Sampler_Sweep().

# ============================================================
# MATLAB GROUND TRUTH (lines 3804-3842)
# Resets BEST itick FOM fields to -inf.
# full-sweep: loop_range = 0..N-1 (all indices)
# middle: loop_range = argsort(|full_sample_range|)
# Unsupported mode → ValueError.
# ============================================================
"""
import math
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Setup_Sampler_Sweep.py_impl import OptFom_Setup_Sampler_Sweep


def _best():
    return SimpleNamespace(positive_itick_FOM=0, negative_itick_FOM=0,
                           positive_itick_in_loop=3, negative_itick_in_loop=-3,
                           itick_FOM=0, itick_in_cluster=2, cluster=[1, 2, 3])


def _op(mode='full-sweep', box_size=5):
    return SimpleNamespace(TS_SRCH_MODE=mode, itick_box_size=box_size)


def test_full_sweep_loop_range_length():
    """full-sweep: loop_range covers all indices."""
    r = np.arange(-5, 6)
    lr, _, _, _, _, _ = OptFom_Setup_Sampler_Sweep(r, _best(), _op('full-sweep'))
    assert len(lr) == len(r)


def test_best_fom_reset_to_neginf():
    """BEST itick FOM fields are reset to -inf."""
    r = np.arange(5)
    _, BEST_out, _, _, _, _ = OptFom_Setup_Sampler_Sweep(r, _best(), _op())
    assert math.isinf(BEST_out.positive_itick_FOM) and BEST_out.positive_itick_FOM < 0
    assert math.isinf(BEST_out.negative_itick_FOM) and BEST_out.negative_itick_FOM < 0
    assert math.isinf(BEST_out.itick_FOM) and BEST_out.itick_FOM < 0


def test_middle_search_returns_sorted_by_abs():
    """middle mode: loop_range is sorted by |full_sample_range|."""
    r = np.array([3, -1, 0, 2, -2])
    lr, _, middle_search, _, _, _ = OptFom_Setup_Sampler_Sweep(r, _best(), _op('middle'))
    assert middle_search == 1
    abs_vals = np.abs(r[lr])
    assert np.all(np.diff(abs_vals) >= 0)


def test_unsupported_mode_raises():
    """Unknown mode raises ValueError."""
    with pytest.raises(ValueError):
        OptFom_Setup_Sampler_Sweep(np.arange(5), _best(), _op('unknown'))


def test_full_sweep_sets_box_search_zero():
    """full-sweep: box_search=0, middle_search=0."""
    r = np.arange(5)
    _, _, ms, bs, _, _ = OptFom_Setup_Sampler_Sweep(r, _best(), _op('full-sweep'))
    assert bs == 0 and ms == 0
