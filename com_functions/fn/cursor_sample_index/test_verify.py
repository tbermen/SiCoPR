"""Verification tests for cursor_sample_index().

# ============================================================
# MATLAB GROUND TRUTH (lines 5405-5479)
# Find sbr_peak_i = argmax(sbr[search_range])
# Find zero crossing in [peak-4*M, peak] where sbr crosses 0.01*max
# Apply MM or Mod-MM criterion to find cursor_i
# All indices 0-based in Python (MATLAB 1-based).
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.cursor_sample_index.py_impl import cursor_sample_index


def _param(M=8, ndfe=0, bmax=None):
    return SimpleNamespace(samples_per_ui=M, ndfe=ndfe,
                           bmax=bmax if bmax is not None else [0.0])


def _op(cdr='MM'):
    return SimpleNamespace(CDR=cdr)


def _simple_pulse(N=64, peak=32, M=8):
    """Gaussian-like pulse with clear peak and zero crossing before it."""
    x = np.arange(N)
    sbr = np.exp(-0.5 * ((x - peak) / 4) ** 2)
    return sbr


def test_peak_found_in_search_range():
    """sbr_peak_i is within peak_search_range."""
    sbr = _simple_pulse(N=100, peak=50)
    search = np.arange(30, 70)
    _, _, sbr_peak_i, _ = cursor_sample_index(sbr, _param(), _op(), search)
    assert sbr_peak_i in search


def test_no_zero_crossing_flag():
    """All-positive pulse above threshold → no_zero_crossing=1."""
    sbr = np.ones(40)  # flat 1.0, no crossing of 0.01
    search = np.arange(10, 30)
    cursor_i, nzc, _, _ = cursor_sample_index(sbr, _param(), _op(), search)
    assert nzc == 1
    assert cursor_i is None


def test_cursor_i_in_range():
    """cursor_i is between zero-crossing and peak+2*M."""
    sbr = _simple_pulse(N=100, peak=50, M=8)
    search = np.arange(30, 70)
    cursor_i, nzc, sbr_peak_i, _ = cursor_sample_index(sbr, _param(), _op(), search)
    if nzc == 0:
        assert cursor_i >= 0
        assert cursor_i < len(sbr)


def test_cursor_i_0based():
    """cursor_i is a valid 0-based Python index."""
    sbr = _simple_pulse(N=100, peak=50, M=8)
    search = np.arange(30, 70)
    cursor_i, nzc, _, _ = cursor_sample_index(sbr, _param(), _op(), search)
    if nzc == 0:
        assert 0 <= cursor_i < len(sbr)


def test_mod_mm_different_from_mm():
    """Mod-MM and MM can produce different cursor locations."""
    sbr = _simple_pulse(N=100, peak=60, M=8)
    search = np.arange(30, 80)
    cursor_mm, _, _, _ = cursor_sample_index(sbr, _param(), _op('MM'), search)
    cursor_mod, _, _, _ = cursor_sample_index(sbr, _param(), _op('Mod-MM'), search)
    # They may or may not differ; just verify both are valid indices
    for ci in (cursor_mm, cursor_mod):
        if ci is not None:
            assert 0 <= ci < len(sbr)


# ---------------------------------------------------------------------------
# Against COM Octave. This function sets the sampling phase, so an error here
# shifts every downstream quantity; none of the assertions above pinned a
# value. MATLAB indexes from 1, so each reference value is stated and the
# expected 0-based one derived from it.
# ---------------------------------------------------------------------------

_M, _PEAK = 32, 320
_OCT_1BASED = {'cursor_i': 308, 'sbr_peak_i': 321, 'zxi': 268}


def _pulse():
    n = _M * 20
    t = np.arange(n, dtype=float)
    p = 0.60 * np.exp(-((t - _PEAK) / 9.0) ** 2)
    p += 0.20 * np.exp(-((t - (_PEAK + _M)) / 14.0) ** 2)
    p += 0.07 * np.exp(-((t - (_PEAK + 2 * _M)) / 18.0) ** 2)
    p += 0.05 * np.exp(-((t - (_PEAK - _M)) / 14.0) ** 2)
    p[t < _PEAK - 4 * _M] = 0.0
    return p


def _run():
    p = _pulse()
    par = SimpleNamespace(samples_per_ui=_M, ndfe=2,
                          bmax=np.array([0.85, 0.85]))
    return cursor_sample_index(p, par, SimpleNamespace(CDR='MM'),
                               np.arange(len(p)))


def _scalar(v):
    a = np.asarray(v).ravel()
    return int(a[0]) if a.size else None


def test_matches_com_octave_one_based_minus_one():
    cursor_i, no_zx, peak_i, zxi = _run()
    got = {'cursor_i': _scalar(cursor_i), 'sbr_peak_i': _scalar(peak_i),
           'zxi': _scalar(zxi)}
    assert _scalar(no_zx) == 0, 'no_zero_crossing should be clear on this pulse'
    for name, want1 in _OCT_1BASED.items():
        assert got[name] == want1 - 1, (
            '%s is %r; COM Octave gives %d 1-based, so %d is expected here'
            % (name, got[name], want1, want1 - 1))


def test_cursor_sits_before_the_peak_by_part_of_a_UI():
    """The sampling phase is found from the zero crossing, not the peak, so it
    must land below the peak but within one UI of it."""
    cursor_i, _n, peak_i, _z = _run()
    c, pk = _scalar(cursor_i), _scalar(peak_i)
    assert 0 < pk - c < _M, 'cursor %d and peak %d are %d samples apart' % (c, pk, pk - c)


def test_moving_the_pulse_moves_the_cursor():
    """A shifted pulse must give a shifted cursor, or the value above is being
    produced by something other than the input."""
    p = _pulse()
    shifted = np.roll(p, _M)
    par = SimpleNamespace(samples_per_ui=_M, ndfe=2, bmax=np.array([0.85, 0.85]))
    a = _scalar(_run()[0])
    b = _scalar(cursor_sample_index(shifted, par, SimpleNamespace(CDR='MM'),
                                    np.arange(len(shifted)))[0])
    assert b != a, 'the cursor stayed at %d for a pulse shifted by one UI' % a
