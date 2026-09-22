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


# ---------------------------------------------------------------------------
# Against COM Octave. This decides the ORDER the sampling phases are tried in,
# which the middle-search termination depends on; none of the assertions above
# pinned that order. MATLAB indexes from 1, so each reference sequence is
# stated and the 0-based one derived from it.
# ---------------------------------------------------------------------------

_SWEEP_RANGE = np.arange(-16, 16)
_OCT_FULL_1BASED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_OCT_MIDDLE_1BASED = [17, 16, 18, 15, 19, 14, 20, 13, 21, 12]


def _sweep(mode):
    OP = SimpleNamespace(TS_SRCH_MODE=mode, itick_box_size=5)
    return OptFom_Setup_Sampler_Sweep(_SWEEP_RANGE, SimpleNamespace(), OP)


def test_full_sweep_order_matches_com_octave():
    loop_range, _b, middle, box = _sweep('full-sweep')[:4]
    lr = np.asarray(loop_range).ravel()
    assert lr.size == _SWEEP_RANGE.size
    assert list(lr[:10]) == [v - 1 for v in _OCT_FULL_1BASED], (
        'full-sweep order is %r; COM Octave gives %r 1-based'
        % (list(lr[:10]), _OCT_FULL_1BASED))
    assert int(middle) == 0 and int(box) == 0


def test_middle_search_works_outward_from_the_centre():
    loop_range, _b, middle, box = _sweep('middle')[:4]
    lr = np.asarray(loop_range).ravel()
    assert list(lr[:10]) == [v - 1 for v in _OCT_MIDDLE_1BASED], (
        'middle order is %r; COM Octave gives %r 1-based'
        % (list(lr[:10]), _OCT_MIDDLE_1BASED))
    assert int(middle) == 1 and int(box) == 0
    # the point of the mode, stated: each step moves further from itick 0
    picked = _SWEEP_RANGE[lr]
    assert np.all(np.diff(np.abs(picked)) >= 0), (
        'middle search did not move monotonically outward: %r' % list(picked[:8]))


def test_every_phase_is_visited_exactly_once():
    for mode in ('full-sweep', 'middle'):
        lr = np.asarray(_sweep(mode)[0]).ravel()
        assert sorted(lr) == list(range(_SWEEP_RANGE.size)), (
            '%s visits %r' % (mode, sorted(lr)[:6]))


def test_unknown_mode_is_rejected():
    """COM Octave errors with 'unsuported TS_SRCH_MODE'; the port must not
    quietly fall through to a default order."""
    with pytest.raises(ValueError, match='TS_SRCH_MODE'):
        _sweep('box')
