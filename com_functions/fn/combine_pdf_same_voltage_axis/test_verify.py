"""Verification tests for combine_pdf_same_voltage_axis().

# ============================================================
# MATLAB GROUND TRUTH
# PDF struct: x (voltage array), y (probability), BinSize
# min1=pdf1.x(1), min2=pdf2.x(1), shift=round(|min1-min2|/BinSize)
#
# Case: pdf1.x=[-0.2..0.2], pdf2.x=[-0.1..0.1], BinSize=0.1
#   min1=-0.2 < min2=-0.1 → shift pdf2 right by 1 bin
#   x2_new = [pdf1.x(1), pdf2.x] = [-0.2,-0.1,0,0.1] (4 points)
#   y2_new = [0, 1, 2, 1]
#   L1=5, L2=4 → right-pad y2_new with 1 zero → y2_new=[0,1,2,1,0]
#   out.y = [1,2,3,2,1]+[0,1,2,1,0] = [1,3,5,3,1]
#   out.x = [-0.2,-0.1,0,0.1,0.2]
#
# Symmetric case (pdf1↔pdf2 swapped): same result
#
# Equal range: out.y = y1 + y2 (direct)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.combine_pdf_same_voltage_axis.py_impl import combine_pdf_same_voltage_axis


def make_pdf(x, y, binsize=0.1):
    return SimpleNamespace(x=np.array(x, dtype=float),
                           y=np.array(y, dtype=float),
                           BinSize=binsize)


_X5 = [-0.2, -0.1, 0.0, 0.1, 0.2]
_X3 = [-0.1, 0.0, 0.1]


def test_pdf1_extends_left():
    """pdf1 wider on left: pdf2 shifted right, zero-padded, then added."""
    pdf1 = make_pdf(_X5, [1, 2, 3, 2, 1])
    pdf2 = make_pdf(_X3, [1, 2, 1])
    out = combine_pdf_same_voltage_axis(pdf1, pdf2)
    np.testing.assert_allclose(out.y, [1, 3, 5, 3, 1])
    np.testing.assert_allclose(out.x, _X5)


def test_pdf2_extends_left():
    """pdf2 wider on left: symmetric of previous case."""
    pdf1 = make_pdf(_X3, [1, 2, 1])
    pdf2 = make_pdf(_X5, [1, 2, 3, 2, 1])
    out = combine_pdf_same_voltage_axis(pdf1, pdf2)
    np.testing.assert_allclose(out.y, [1, 3, 5, 3, 1])
    np.testing.assert_allclose(out.x, _X5)


def test_equal_range_direct_add():
    """Same x range → simple element-wise addition."""
    pdf1 = make_pdf(_X3, [1, 2, 1])
    pdf2 = make_pdf(_X3, [0, 1, 0])
    out = combine_pdf_same_voltage_axis(pdf1, pdf2)
    np.testing.assert_allclose(out.y, [1, 3, 1])
    np.testing.assert_allclose(out.x, _X3)


def test_binsize_mismatch_raises():
    pdf1 = make_pdf(_X3, [1, 2, 1], binsize=0.1)
    pdf2 = make_pdf(_X3, [1, 2, 1], binsize=0.2)
    with pytest.raises(ValueError):
        combine_pdf_same_voltage_axis(pdf1, pdf2)


def test_output_binsize_preserved():
    pdf1 = make_pdf(_X5, [1, 2, 3, 2, 1])
    pdf2 = make_pdf(_X3, [1, 2, 1])
    out = combine_pdf_same_voltage_axis(pdf1, pdf2)
    assert out.BinSize == pytest.approx(0.1)


# ============================================================
# Executed-reference checks — values produced by running the reference
# function under Octave (tools/octave_oracle.py, COM Octave 4p16p0), not by
# reading the MATLAB.
#
# Row vs column.  Every axis is rebuilt with horizontal concatenation
# ([pdf1.x(1:n) pdf2.x] and [zeros(1,n) pdf2.y]), so the fields have to be
# rows.  Probed with pdf1.x/.y as 3x1 columns and pdf2.x/.y as 1x3 rows,
# equal Min (shift 0):
#   out.y comes back 3x3 = [1 2 1; 2 3 2; 1 2 1] and out.x 1x3 — implicit
#   expansion, not the 1x3 [1 3 1] that flattening produces.
# The mirror case (pdf1 rows, pdf2 columns) gives the same 3x3.
# With BOTH sides columns and a nonzero shift:
#   "horizontal dimensions mismatch (1x1 vs 3x1)".
# np.concatenate flattens all of that away, so Python answered [1 3 1] in
# every one of those cases.
#
# round() half away from zero in shift_amount = round(|min1-min2|/BinSize):
#   pdf1.x=[-1.5 -0.5 0.5], pdf2.x=[-1 0 1], BinSize=1 -> |diff|/bin = 0.5
#     -> shift 1 -> out.x=[-1.5 -1 0 1], out.y=[1 3 3 1]
#     (np.round(0.5)=0 would give out.x=[-1 0 1], out.y=[2 4 2])
#   swapping the two pdfs gives the same [-1.5 -1 0 1] / [1 3 3 1]
#   pdf1.x=[-2.5 -1.5 -0.5] -> |diff|/bin = 1.5 -> shift 2
#     -> out.x=[-2.5 -1.5 -1 0 1], out.y=[1 2 2 2 1]
#
# shift_amount past the end of the donor axis is an error, not a short slice:
#   pdf1.x=[-0.3 -0.2], pdf2.x=[0 0.1 0.2], BinSize=0.1 -> shift 3
#     -> "pdf1(3): out of bound 2 (dimensions are 1x2)"
#
# Agreements confirmed: the nominal both-ways case, equal axes, the
# both-columns/shift-0 case, a non-integer bin offset
# (pdf1.x=[-0.25 -0.15], pdf2.x=[-0.1 0] -> out.x=[-0.25 -0.1 0], y=[1 2 1]),
# len(y) != len(x) (nonconformant + error), an empty pdf2 (out of bound),
# and BinSize NaN on both sides (NaN ~= NaN, so 'bin size must be equal').
# ============================================================


def test_column_vector_field_refused():
    """A column pdf2.y makes [zeros(1,n) pdf2.y] a dimension error in MATLAB."""
    pdf1 = make_pdf(_X5, [1, 2, 3, 2, 1])
    pdf2 = make_pdf(_X3, [1, 2, 1])
    pdf2.x = pdf2.x.reshape(-1, 1)
    pdf2.y = pdf2.y.reshape(-1, 1)
    with pytest.raises(ValueError):
        combine_pdf_same_voltage_axis(pdf1, pdf2)


def test_mixed_orientation_refused():
    """pdf1 columns + pdf2 rows implicit-expands to a 3x3 in MATLAB."""
    pdf1 = make_pdf(_X3, [1, 2, 1])
    pdf1.x = pdf1.x.reshape(-1, 1)
    pdf1.y = pdf1.y.reshape(-1, 1)
    pdf2 = make_pdf(_X3, [0, 1, 0])
    with pytest.raises(ValueError):
        combine_pdf_same_voltage_axis(pdf1, pdf2)


def test_row_shaped_2d_still_accepted():
    """A 1xN two-dimensional array is a MATLAB row and is handled as one."""
    pdf1 = make_pdf(_X5, [1, 2, 3, 2, 1])
    pdf1.x = pdf1.x.reshape(1, -1)
    pdf1.y = pdf1.y.reshape(1, -1)
    pdf2 = make_pdf(_X3, [1, 2, 1])
    out = combine_pdf_same_voltage_axis(pdf1, pdf2)
    np.testing.assert_allclose(out.y, [1, 3, 5, 3, 1])


def test_round_half_away_from_zero_in_shift():
    """|min1-min2|/BinSize == 0.5 exactly: MATLAB shifts by one bin, not zero."""
    pdf1 = make_pdf([-1.5, -0.5, 0.5], [1, 2, 1], binsize=1.0)
    pdf2 = make_pdf([-1.0, 0.0, 1.0], [1, 2, 1], binsize=1.0)
    out = combine_pdf_same_voltage_axis(pdf1, pdf2)
    np.testing.assert_allclose(out.x, [-1.5, -1.0, 0.0, 1.0])
    np.testing.assert_allclose(out.y, [1, 3, 3, 1])

    # min1 > min2 takes the other branch and must round the same way
    out2 = combine_pdf_same_voltage_axis(pdf2, pdf1)
    np.testing.assert_allclose(out2.x, [-1.5, -1.0, 0.0, 1.0])
    np.testing.assert_allclose(out2.y, [1, 3, 3, 1])


def test_shift_of_one_and_a_half_bins():
    pdf1 = make_pdf([-2.5, -1.5, -0.5], [1, 2, 1], binsize=1.0)
    pdf2 = make_pdf([-1.0, 0.0, 1.0], [1, 2, 1], binsize=1.0)
    out = combine_pdf_same_voltage_axis(pdf1, pdf2)
    np.testing.assert_allclose(out.x, [-2.5, -1.5, -1.0, 0.0, 1.0])
    np.testing.assert_allclose(out.y, [1, 2, 2, 2, 1])


def test_shift_past_end_of_donor_axis_refused():
    pdf1 = make_pdf([-0.3, -0.2], [1, 1])
    pdf2 = make_pdf([0.0, 0.1, 0.2], [2, 2, 2])
    with pytest.raises(ValueError):
        combine_pdf_same_voltage_axis(pdf1, pdf2)


def test_non_integer_bin_offset():
    pdf1 = make_pdf([-0.25, -0.15], [1, 1])
    pdf2 = make_pdf([-0.1, 0.0], [1, 1])
    out = combine_pdf_same_voltage_axis(pdf1, pdf2)
    np.testing.assert_allclose(out.x, [-0.25, -0.1, 0.0])
    np.testing.assert_allclose(out.y, [1, 2, 1])


def test_nan_binsize_refused():
    """NaN ~= NaN is true, so the reference reports unequal bin sizes."""
    pdf1 = make_pdf(_X3, [1, 2, 1], binsize=float('nan'))
    pdf2 = make_pdf(_X3, [1, 2, 1], binsize=float('nan'))
    with pytest.raises(ValueError):
        combine_pdf_same_voltage_axis(pdf1, pdf2)
