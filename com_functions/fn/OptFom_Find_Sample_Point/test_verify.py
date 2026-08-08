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
