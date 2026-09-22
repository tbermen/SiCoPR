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


# ============================================================
# COM Octave oracle values (2026-09-22)
# rangelimit extracted verbatim from the reference and run under Octave on a
# 6-point axis [0 1 2 3 4 5] GHz with a 6x4 sch.
# ============================================================

OCT_FAX = np.array([0.0, 1e9, 2e9, 3e9, 4e9, 5e9])


def _sch2():
    return np.arange(1, 25, dtype=float).reshape(6, 4, order='F')


def test_oracle_caller_param_is_not_touched():
    """MATLAB structs pass by value, so the caller's param keeps its flim even
    though the function assigns to it.

    COM Octave, flim=99e9: pout.flim -> 5e9, param.flim in the caller -> 99e9.
    """
    param = _param(flim=99e9)
    schout, fout, limited, p = rangelimit(_sch2(), OCT_FAX, param)
    assert limited == 0
    assert p.flim == pytest.approx(5e9)
    assert param.flim == pytest.approx(99e9)
    assert p is not param
    np.testing.assert_array_equal(fout, OCT_FAX)


def test_oracle_flim_zero_keeps_one_row():
    """flim=0 matches the first point, so exactly one row survives.

    COM Octave: fo -> 0 (1x1), so -> [1 7 13 19] (the first row), limited -> 1.
    """
    schout, fout, limited, _ = rangelimit(_sch2(), OCT_FAX, _param(flim=0.0))
    assert limited == 1
    np.testing.assert_array_equal(fout, [0.0])
    np.testing.assert_array_equal(schout, [[1.0, 7.0, 13.0, 19.0]])


def test_oracle_unsorted_axis_takes_the_first_match_not_the_smallest():
    """find(...,1,'first') is positional, not ordered.

    COM Octave, fax=[0 4e9 1e9 5e9 2e9 3e9], flim=3e9 -> fo=[0 4e9], limited=1.
    """
    fu = np.array([0.0, 4e9, 1e9, 5e9, 2e9, 3e9])
    _, fout, limited, _ = rangelimit(_sch2(), fu, _param(flim=3e9))
    assert limited == 1
    np.testing.assert_array_equal(fout, [0.0, 4e9])


def test_oracle_nan_in_axis_never_matches():
    """NaN >= flim is false, so a NaN is carried through, not matched.

    COM Octave, fax=[0 1e9 NaN 3e9 4e9 5e9], flim=3e9 -> fo=[0 1e9 NaN 3e9].
    """
    fn = np.array([0.0, 1e9, np.nan, 3e9, 4e9, 5e9])
    _, fout, limited, _ = rangelimit(_sch2(), fn, _param(flim=3e9))
    assert limited == 1
    assert len(fout) == 4
    assert np.isnan(fout[2])


def test_oracle_row_sch_is_an_out_of_bound_read():
    """sch(1:iend,:,:) reads rows, so a 1xN sch cannot be truncated at iend>1.

    COM Octave, sch=1:6 as a 1x6 row, 6-point axis, flim=3e9 (iend=4):
        "error: sch(4,_,_): out of bound 1 (dimensions are 1x6)".
    numpy's sch[:4] handed back the whole row and called it limited.
    """
    row = np.arange(1.0, 7.0).reshape(1, 6)
    with pytest.raises(IndexError):
        rangelimit(row, OCT_FAX, _param(flim=3e9))


def test_oracle_column_sch_truncates():
    """The same sch as a 6x1 column is a legal read.

    COM Octave, sch=(1:6).', flim=3e9 -> so=[1 2 3 4] (4x1), limited=1.
    """
    col = np.arange(1.0, 7.0).reshape(6, 1)
    schout, _, limited, _ = rangelimit(col, OCT_FAX, _param(flim=3e9))
    assert limited == 1
    np.testing.assert_array_equal(schout.ravel(), [1.0, 2.0, 3.0, 4.0])


def test_oracle_3d_sch_slices_the_first_axis_only():
    """COM Octave, sch=reshape(1:24,6,2,2), flim=2e9 -> size(so) = [3 2 2]."""
    s3 = np.arange(1, 25, dtype=float).reshape(6, 2, 2, order='F')
    schout, _, limited, _ = rangelimit(s3, OCT_FAX, _param(flim=2e9))
    assert limited == 1
    assert schout.shape == (3, 2, 2)


def test_oracle_empty_axis_raises():
    """COM Octave, empty axis: "schFreqAxis(0): subscripts must be ... integers"."""
    with pytest.raises(IndexError):
        rangelimit(np.zeros((0, 2)), np.array([]), _param(flim=3e9))
