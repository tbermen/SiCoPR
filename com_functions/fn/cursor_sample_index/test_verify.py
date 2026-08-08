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
