"""Verification tests for cdf_to_ber_contour() — BER threshold crossing finder.

# ============================================================
# MATLAB GROUND TRUTH
# find(cdf.y > specBER, 1, 'first') returns 1-based index.
# Bottom = cdf.x at that index.
# Top = cdf.x at (length(cdf.y) - first_flipped_idx + 1), 1-based.
#
# Case 1 — symmetric bathtub CDF:
#   y=[0.1, 0.3, 0.7, 0.3, 0.1], x=[-2,-1,0,1,2], specBER=0.25
#   Bottom: find(y>0.25,1,'first') = 2 (1-based) → x(2) = -1
#   Flipped y=[0.1,0.3,0.7,0.3,0.1], find(>0.25,1,'first')=2
#   nidx = 5-2+1 = 4 (1-based) → x(4) = 1
#   → noise_bottom=-1, noise_top=1
#
# Case 2 — asymmetric CDF:
#   y=[0.1, 0.4, 0.6, 0.2, 0.05], x=[0,1,2,3,4], specBER=0.25
#   Bottom: find(y>0.25,1,'first') = 2 → x(2) = 1
#   Flipped y=[0.05,0.2,0.6,0.4,0.1], find(>0.25,1,'first') = 3
#   nidx = 5-3+1 = 3 → x(3) = 2
#   → noise_bottom=1, noise_top=2
#
# Case 3 — strict inequality at boundary (y==specBER not counted):
#   y=[0.25, 0.5, 0.25], x=[0,1,2], specBER=0.25
#   find(y>0.25,1,'first') = 2 → x(2) = 1
#   Flipped=[0.25,0.5,0.25], find(>0.25,1,'first')=2
#   nidx = 3-2+1 = 2 → x(2) = 1
#   → noise_bottom=1, noise_top=1
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.cdf_to_ber_contour.py_impl import cdf_to_ber_contour


def _make_cdf(y, x):
    c = SimpleNamespace()
    c.y = np.asarray(y, dtype=float)
    c.x = np.asarray(x, dtype=float)
    return c


def test_symmetric_bathtub():
    c = _make_cdf([0.1, 0.3, 0.7, 0.3, 0.1], [-2.0, -1.0, 0.0, 1.0, 2.0])
    bottom, top = cdf_to_ber_contour(c, 0.25)
    assert bottom == pytest.approx(-1.0)
    assert top == pytest.approx(1.0)


def test_asymmetric_cdf():
    c = _make_cdf([0.1, 0.4, 0.6, 0.2, 0.05], [0.0, 1.0, 2.0, 3.0, 4.0])
    bottom, top = cdf_to_ber_contour(c, 0.25)
    assert bottom == pytest.approx(1.0)
    assert top == pytest.approx(2.0)


def test_strict_inequality_at_boundary():
    """y == specBER is NOT counted — strict > only."""
    c = _make_cdf([0.25, 0.5, 0.25], [0.0, 1.0, 2.0])
    bottom, top = cdf_to_ber_contour(c, 0.25)
    assert bottom == pytest.approx(1.0)
    assert top == pytest.approx(1.0)


def test_bottom_leq_top():
    """noise_bottom must always be <= noise_top for a valid CDF."""
    c = _make_cdf([0.05, 0.2, 0.5, 0.2, 0.05], [-2.0, -1.0, 0.0, 1.0, 2.0])
    bottom, top = cdf_to_ber_contour(c, 0.1)
    assert bottom <= top


def test_returns_scalars():
    c = _make_cdf([0.1, 0.3, 0.7, 0.3, 0.1], [-2.0, -1.0, 0.0, 1.0, 2.0])
    bottom, top = cdf_to_ber_contour(c, 0.25)
    assert np.isscalar(bottom)
    assert np.isscalar(top)


# ============================================================
# Executed-reference checks — values produced by running the reference
# function under Octave (tools/octave_oracle.py, COM Octave 4p16p0), not by
# reading the MATLAB.  cdf.y=[0.1 0.3 0.7 0.3 0.1], cdf.x=[-2 -1 0 1 2]
# unless stated.
#
# Nothing above specBER: BOTH find() calls are empty, so
#   specBER = 0.9  -> noise_bottom = [] and noise_top = []   (1x0, not a
#                     voltage; nidx = length(cdf.y) - [] + 1 is empty too)
#   specBER = NaN  -> [] and []   (nothing compares > NaN)
#   cdf.y=[0.1], x=[1.5], specBER=0.25 -> [] and []
# np.argmax on an all-False mask answers 0, so the old code returned
# (cdf.x(1), cdf.x(end)) = (-2.0, 2.0) — the whole axis reported as the eye.
#
# Cases that do have a crossing, confirmed against Octave:
#   y=[0.5 0.6 0.7],  x=[0 1 2], spec=0.1  -> (0, 2)
#   y=[0.1 NaN 0.7 0.3 0.1],     spec=0.25 -> (0, 1)   NaN never satisfies >
#   y=[0.25 0.5 0.25], x=[0 1 2], spec=0.25 -> (1, 1)  strict >, so the
#                                              equal ends are excluded
#   y=[0 .1 .4 .8 1], x=[0..4],  spec=0.25 -> (2, 4)
#   y=[0.7], x=[1.5],            spec=0.25 -> (1.5, 1.5)
#   column-vector cdf.x/cdf.y with the nominal data -> (-1, 1), same as rows
# ============================================================


def test_no_crossing_refuses():
    """specBER above every cdf.y: MATLAB yields an empty contour, not the axis."""
    c = _make_cdf([0.1, 0.3, 0.7, 0.3, 0.1], [-2.0, -1.0, 0.0, 1.0, 2.0])
    with pytest.raises(IndexError):
        cdf_to_ber_contour(c, 0.9)


def test_nan_specber_refuses():
    c = _make_cdf([0.1, 0.3, 0.7, 0.3, 0.1], [-2.0, -1.0, 0.0, 1.0, 2.0])
    with pytest.raises(IndexError):
        cdf_to_ber_contour(c, float('nan'))


def test_single_point_no_crossing_refuses():
    c = _make_cdf([0.1], [1.5])
    with pytest.raises(IndexError):
        cdf_to_ber_contour(c, 0.25)


def test_all_above_spans_whole_axis():
    c = _make_cdf([0.5, 0.6, 0.7], [0.0, 1.0, 2.0])
    assert cdf_to_ber_contour(c, 0.1) == (0.0, 2.0)


def test_nan_in_cdf_y_is_skipped():
    c = _make_cdf([0.1, np.nan, 0.7, 0.3, 0.1], [-2.0, -1.0, 0.0, 1.0, 2.0])
    assert cdf_to_ber_contour(c, 0.25) == (0.0, 1.0)


def test_single_point_with_crossing():
    c = _make_cdf([0.7], [1.5])
    assert cdf_to_ber_contour(c, 0.25) == (1.5, 1.5)


def test_column_vectors_match_rows():
    c = _make_cdf([0.1, 0.3, 0.7, 0.3, 0.1], [-2.0, -1.0, 0.0, 1.0, 2.0])
    c.y = c.y.reshape(-1, 1)
    c.x = c.x.reshape(-1, 1)
    assert cdf_to_ber_contour(c, 0.25) == (-1.0, 1.0)
