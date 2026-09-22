"""Verification tests for OptFom_Find_Sample_Point().

# ============================================================
# MATLAB GROUND TRUTH (lines 3515-3537)
# ts_anchor=0: cursor from Muller-Mueller (cursor_sample_index)
# ts_anchor=1: cursor = sbr_peak_i (peak sample)
# ts_anchor=2: cursor = argmax(possible_cursor - possible_precursor)
# All indices 0-based in Python.
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Find_Sample_Point.py_impl import OptFom_Find_Sample_Point


def _param(ts_anchor=0, M=8, ndfe=0, bmax=None):
    return SimpleNamespace(ts_anchor=ts_anchor, samples_per_ui=M, ndfe=ndfe,
                           bmax=bmax or [0.0])


def _op():
    return SimpleNamespace(CDR='MM')


def _pulse(N=100, peak=50, M=8):
    x = np.arange(N)
    return np.exp(-0.5 * ((x - peak) / 5) ** 2)


def test_anchor0_returns_valid_cursor():
    """ts_anchor=0: cursor is a valid 0-based index."""
    sbr = _pulse()
    search = np.arange(30, 70)
    cursor_i, nzc, peak_i = OptFom_Find_Sample_Point(sbr, _param(0), _op(), search)
    if nzc == 0:
        assert 0 <= cursor_i < len(sbr)


def test_anchor1_returns_peak():
    """ts_anchor=1: cursor equals sbr_peak_i."""
    sbr = _pulse()
    search = np.arange(30, 70)
    cursor_i, nzc, peak_i = OptFom_Find_Sample_Point(sbr, _param(1), _op(), search)
    assert nzc == 0
    assert cursor_i == peak_i


def test_anchor2_returns_valid_cursor():
    """ts_anchor=2: max-DV cursor is a valid 0-based index."""
    sbr = _pulse(N=100, peak=50, M=8)
    search = np.arange(30, 70)
    cursor_i, nzc, _ = OptFom_Find_Sample_Point(sbr, _param(2), _op(), search)
    assert nzc == 0
    assert 0 <= cursor_i < len(sbr)


def test_invalid_anchor_raises():
    """ts_anchor outside {0,1,2} raises ValueError."""
    sbr = _pulse()
    with pytest.raises(ValueError, match='ts_anchor'):
        OptFom_Find_Sample_Point(sbr, _param(3), _op(), np.arange(30, 70))


def test_returns_three_outputs():
    """Always returns (cursor_i, no_zero_crossing, sbr_peak_i)."""
    sbr = _pulse()
    result = OptFom_Find_Sample_Point(sbr, _param(1), _op(), np.arange(30, 70))
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Against COM Octave, for each ts_anchor mode.
#
# ts_anchor selects where the cursor is placed: 0 keeps the mueller-muller
# crossing, 1 moves to the pulse peak, 2 to the maximum of cursor minus
# precursor. Choosing the wrong one shifts every sampled quantity, and none of
# the assertions above pinned which sample comes back.
# ---------------------------------------------------------------------------

_FSP_M, _FSP_PEAK = 32, 320
_OCT_1BASED = {0: 308, 1: 321, 2: 321}


def _fsp_pulse():
    n = _FSP_M * 20
    t = np.arange(n, dtype=float)
    p = 0.60 * np.exp(-((t - _FSP_PEAK) / 9.0) ** 2)
    p += 0.20 * np.exp(-((t - (_FSP_PEAK + _FSP_M)) / 14.0) ** 2)
    p += 0.07 * np.exp(-((t - (_FSP_PEAK + 2 * _FSP_M)) / 18.0) ** 2)
    p += 0.05 * np.exp(-((t - (_FSP_PEAK - _FSP_M)) / 14.0) ** 2)
    p[t < _FSP_PEAK - 4 * _FSP_M] = 0.0
    return p


def _fsp(anchor, p=None):
    p = _fsp_pulse() if p is None else p
    par = SimpleNamespace(samples_per_ui=_FSP_M, ndfe=2,
                          bmax=np.array([0.85, 0.85]), ts_anchor=anchor)
    c, nz, pk = OptFom_Find_Sample_Point(p, par, SimpleNamespace(CDR='MM'),
                                         np.arange(len(p)))
    return int(np.asarray(c).ravel()[0]), int(np.asarray(nz).ravel()[0]), \
        int(np.asarray(pk).ravel()[0])


def test_each_anchor_matches_com_octave():
    for anchor, want1 in _OCT_1BASED.items():
        cursor, nz, _pk = _fsp(anchor)
        assert nz == 0, 'no_zero_crossing set for ts_anchor=%d' % anchor
        assert cursor == want1 - 1, (
            'ts_anchor=%d gave cursor %d; COM Octave gives %d 1-based, so %d '
            'is expected here' % (anchor, cursor, want1, want1 - 1))


def test_anchor_0_and_1_disagree_on_this_pulse():
    """The modes must be distinguishable, or pinning them proves nothing.
    (Anchor 2 happens to coincide with the peak on this pulse, which is why
    the mueller-muller case is the one used to separate them.)"""
    assert _fsp(0)[0] != _fsp(1)[0]


def test_anchor_1_returns_the_reported_peak():
    cursor, _nz, peak = _fsp(1)
    assert cursor == peak, 'ts_anchor=1 must place the cursor on the peak'
