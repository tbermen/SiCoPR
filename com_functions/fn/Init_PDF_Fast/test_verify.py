"""Verification tests for Init_PDF_Fast().

# ============================================================
# MATLAB GROUND TRUTH
# EmptyPDF with BinSize=0.1; values=[-0.1,0,0.1]; probs=[0.25,0.5,0.25]
#   rvd = round([-0.1,0,0.1]/0.1) = [-1,0,1]
#   pdf.x = arange(-1,2)*0.1 = [-0.1,0,0.1]
#   pdf.Min = -1
#   bin_placement = [-1,0,1]-(-1) = [0,1,2]
#   pdf.y[0] = 0.25 (direct assign)
#   pdf.y[1] += 0.5 → 0.5
#   pdf.y[2] += 0.25 → 0.25
#   pdf.y = [0.25, 0.5, 0.25]
#
# No normalization; caller responsibility.
# Duplicate values → same bin: probabilities accumulate.
#   values=[0,0], probs=[0.3,0.7]: rvd=[0,0], bin_placement=[0,0]
#   pdf.y[0]=0.3 then pdf.y[0]+=0.7 → 1.0
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.Init_PDF_Fast.py_impl import Init_PDF_Fast


def make_empty(binsize=0.1):
    return SimpleNamespace(BinSize=binsize, Min=0,
                           y=np.array([0.0]), x=np.array([0.0]))


def test_three_point_symmetric():
    """values=[-0.1,0,0.1]: y=[0.25,0.5,0.25], x=[-0.1,0,0.1], Min=-1."""
    out = Init_PDF_Fast(make_empty(), np.array([-0.1, 0.0, 0.1]),
                        np.array([0.25, 0.5, 0.25]))
    np.testing.assert_allclose(out.y, [0.25, 0.5, 0.25], atol=1e-12)
    np.testing.assert_allclose(out.x, [-0.1, 0.0, 0.1], atol=1e-12)
    assert out.Min == -1


def test_no_normalization():
    """No normalization: sum(y) = sum(probs) if no duplicates."""
    out = Init_PDF_Fast(make_empty(), np.array([-0.1, 0.1]),
                        np.array([0.3, 0.7]))
    assert np.sum(out.y) == pytest.approx(1.0, rel=1e-12)


def test_duplicate_values_accumulate():
    """Duplicate values → same bin: probabilities accumulate to 1.0."""
    out = Init_PDF_Fast(make_empty(), np.array([0.0, 0.0]),
                        np.array([0.3, 0.7]))
    assert out.y[0] == pytest.approx(1.0, rel=1e-12)


def test_binsize_preserved():
    out = Init_PDF_Fast(make_empty(binsize=0.05),
                        np.array([-0.05, 0.0, 0.05]),
                        np.array([0.2, 0.6, 0.2]))
    assert out.BinSize == pytest.approx(0.05)


def test_x_length_matches_y():
    out = Init_PDF_Fast(make_empty(), np.array([-0.2, -0.1, 0.0, 0.1, 0.2]),
                        np.array([0.1, 0.2, 0.4, 0.2, 0.1]))
    assert len(out.x) == len(out.y)


# ============================================================
# Executed-reference checks — values produced by running the reference
# function under Octave (tools/octave_oracle.py, COM Octave 4p16p0), not by
# reading the MATLAB.  EmptyPDF is {BinSize, Min=0, y=1, x=0} throughout.
#
# pdf.x only spans rvd(1)..rvd(end), so bin_placement falls off the array as
# soon as `values` is not ascending, and MATLAB stops there:
#   Init_PDF_Fast(E,[0 -0.2 0.3],[0.2 0.3 0.5]), BinSize=0.1
#     -> "pdf(-1): subscripts must be either integers 1 to (2^63)-1 or
#         logicals"
#   Init_PDF_Fast(E,[0.3 0.2 0.1],[0.2 0.3 0.5])  -> "pdf(0): ..."
#   Init_PDF_Fast(E,[-0.1 NaN 0.1],[...])         -> "pdf(nan): ..."
# A negative index is legal in numpy, so the first of those quietly added
# 0.3 to the wrong bin and answered y=[0.2 0 0.3 0.5].
#
# round() half away from zero on values/BinSize:
#   Init_PDF_Fast(E,[0.5 1.5 2.5],[0.2 0.5 0.3]),  BinSize=1
#     -> Min=1,  x=[1 2 3],    y=[0.2 0.5 0.3]
#      (np.round would give rvd=[0 2 2]: a different axis and a collision)
#   Init_PDF_Fast(E,[-2.5 -1.5 -0.5],[0.2 0.5 0.3]), BinSize=1
#     -> Min=-3, x=[-3 -2 -1], y=[0.2 0.5 0.3]
#
# Other confirmed agreements:
#   values=[-0.2 0.3], probs=[0.4 0.6], BinSize=0.1
#     -> Min=-2, six bins, y=[0.4 0 0 0 0 0.6]
#   values=[0.01 0.02 0.03] (all round to bin 0) -> Min=0, x=[0], y=[1]
#   values=[0.2 0.2 0.2]  -> Min=2, x=[0.2], y=[1]  (probabilities accumulate)
#   probs longer than values is fine; probs shorter is an out-of-bound error;
#   empty values errors on rounded_values_div_binsize(1).
#
# MEASURED DIVERGENCE, not fixed here: pdf.x is built by MATLAB's colon
# operator (BinSize*rvd(1):BinSize:BinSize*rvd(end)) while this port builds
# arange(rvd(1),rvd(end)+1)*BinSize.  The element COUNT always agrees (checked
# over 88 binsize/range combinations), but the values differ in the last bit
# for most binsizes — e.g. the [-0.2 0.3] case above gives Octave
# x(4)=0.10000000000000003, x(6)=0.29999999999999999 against
# 0.10000000000000001 and 0.30000000000000004 here.  Octave's Range is not a
# plain base+i*step (that reproduction failed on 8 of the same 88 cases) and
# MATLAB's colon is a third implementation, so the reference value cannot be
# settled from Octave alone.  pdf.x is not read by the only caller — conv_fct
# rebuilds p.x from p.Min — so the axis is pinned loosely below.
# ============================================================


def test_unsorted_values_refused():
    """A value that rounds below rvd(1) indexes off the front of pdf.y."""
    with pytest.raises(IndexError):
        Init_PDF_Fast(make_empty(), np.array([0.0, -0.2, 0.3]),
                      np.array([0.2, 0.3, 0.5]))


def test_descending_values_refused():
    with pytest.raises(IndexError):
        Init_PDF_Fast(make_empty(), np.array([0.3, 0.2, 0.1]),
                      np.array([0.2, 0.3, 0.5]))


def test_nan_value_refused():
    with pytest.raises(IndexError):
        Init_PDF_Fast(make_empty(), np.array([-0.1, np.nan, 0.1]),
                      np.array([0.2, 0.3, 0.5]))


def test_round_half_away_from_zero_on_values():
    pos = Init_PDF_Fast(make_empty(binsize=1.0), np.array([0.5, 1.5, 2.5]),
                        np.array([0.2, 0.5, 0.3]))
    assert pos.Min == 1
    np.testing.assert_allclose(pos.x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(pos.y, [0.2, 0.5, 0.3])

    neg = Init_PDF_Fast(make_empty(binsize=1.0), np.array([-2.5, -1.5, -0.5]),
                        np.array([0.2, 0.5, 0.3]))
    assert neg.Min == -3
    np.testing.assert_allclose(neg.x, [-3.0, -2.0, -1.0])
    np.testing.assert_allclose(neg.y, [0.2, 0.5, 0.3])


def test_gap_between_values_leaves_empty_bins():
    out = Init_PDF_Fast(make_empty(), np.array([-0.2, 0.3]),
                        np.array([0.4, 0.6]))
    assert out.Min == -2
    np.testing.assert_allclose(out.x, [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3],
                               rtol=0, atol=1e-16)
    np.testing.assert_allclose(out.y, [0.4, 0.0, 0.0, 0.0, 0.0, 0.6])


def test_values_all_rounding_into_one_bin():
    out = Init_PDF_Fast(make_empty(), np.array([0.01, 0.02, 0.03]),
                        np.array([0.2, 0.3, 0.5]))
    assert out.Min == 0
    np.testing.assert_allclose(out.x, [0.0])
    np.testing.assert_allclose(out.y, [1.0])

    same = Init_PDF_Fast(make_empty(), np.array([0.2, 0.2, 0.2]),
                         np.array([0.2, 0.3, 0.5]))
    assert same.Min == 2
    np.testing.assert_allclose(same.x, [0.2])
    np.testing.assert_allclose(same.y, [1.0])


def test_probs_longer_than_values_is_allowed():
    out = Init_PDF_Fast(make_empty(), np.array([-0.1, 0.0, 0.1]),
                        np.array([0.25, 0.25, 0.25, 0.25]))
    np.testing.assert_allclose(out.y, [0.25, 0.25, 0.25])


def test_probs_shorter_than_values_refused():
    with pytest.raises(IndexError):
        Init_PDF_Fast(make_empty(), np.array([-0.1, 0.0, 0.1]),
                      np.array([0.5, 0.5]))


def test_empty_values_refused():
    with pytest.raises(IndexError):
        Init_PDF_Fast(make_empty(), np.array([]), np.array([]))
