"""Verification tests for rangelimit().

# ============================================================
# MATLAB GROUND TRUTH (lines 9401-9412)
# iend = find(schFreqAxis >= param.flim, 1, 'first')
# if found:  schout=sch(1:iend,:,:); limited=1
# else:      schout=sch; param.flim=schFreqAxis(end); limited=0
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.rangelimit.py_impl import rangelimit


def _param(flim):
    return SimpleNamespace(flim=flim)


def test_limit_within_range():
    """flim inside the array → limited=1, output truncated."""
    freq = np.linspace(0, 50e9, 51)
    sch = np.ones((51, 2, 2))
    param = _param(flim=20e9)
    schout, fout, limited, p = rangelimit(sch, freq, param)
    assert limited == 1
    assert fout[-1] >= 20e9
    assert len(fout) < len(freq)
    assert schout.shape[0] == len(fout)


def test_limit_above_max():
    """flim above all frequencies → limited=0, param.flim updated."""
    freq = np.array([1e9, 2e9, 3e9])
    sch = np.ones((3, 2, 2))
    param = _param(flim=100e9)
    schout, fout, limited, p = rangelimit(sch, freq, param)
    assert limited == 0
    assert np.array_equal(fout, freq)
    assert p.flim == pytest.approx(3e9)


def test_first_match_included():
    """The frequency point exactly at flim is included in output."""
    freq = np.array([1e9, 5e9, 10e9, 20e9])
    sch = np.zeros((4, 2, 2))
    param = _param(flim=5e9)
    schout, fout, limited, _ = rangelimit(sch, freq, param)
    assert limited == 1
    assert fout[-1] == pytest.approx(5e9)


def test_output_shape_preserved():
    """sch shape along axes 1+ is preserved."""
    freq = np.linspace(0, 40e9, 41)
    sch = np.ones((41, 3, 3))
    param = _param(flim=10e9)
    schout, _, _, _ = rangelimit(sch, freq, param)
    assert schout.shape[1:] == (3, 3)


def test_param_not_mutated():
    """Original param object is not mutated."""
    freq = np.array([1e9, 2e9, 3e9])
    sch = np.ones((3,))
    param = _param(flim=100e9)
    rangelimit(sch, freq, param)
    assert param.flim == 100e9
