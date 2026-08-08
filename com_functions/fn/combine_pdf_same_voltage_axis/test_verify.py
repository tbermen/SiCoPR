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
