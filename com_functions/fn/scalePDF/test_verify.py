"""Verification tests for scalePDF() — scale the x-axis of a PDF struct.

# ============================================================
# MATLAB GROUND TRUTH
# pdf_out.Min = floor(pdf.Min * scale_factor)
# pdf_out.x   = (pdf_out.Min:-pdf_out.Min) * BinSize
# pdf_out.y   = interp1(pdf.x*scale_factor, pdf.y, pdf_out.x)  [linear]
# NaN fix: y[0]=y[1], y[-1]=y[-2]
# Normalise: pdf_out.y /= sum
#
# Test PDF: Min=-2, BinSize=1.0, x=[-2,-1,0,1,2], y=[0.1,0.2,0.4,0.2,0.1]
#
# scale_factor=1.0 (identity):
#   Min=floor(-2*1)=-2, x=[-2,-1,0,1,2]
#   interp at same points → y unchanged (after normalise) = [0.1,0.2,0.4,0.2,0.1]
#
# scale_factor=2.0:
#   Min=floor(-2*2)=-4, x=[-4,-3,-2,-1,0,1,2,3,4]*1.0
#   old x scaled = [-2,-1,0,1,2]*2=[-4,-2,0,2,4]
#   interp at [-4,-3,-2,-1,0,1,2,3,4] on data at [-4,-2,0,2,4]:
#     x=-4→0.1, x=-3→0.15, x=-2→0.2, x=-1→0.3, x=0→0.4,
#     x=1→0.3, x=2→0.2, x=3→0.15, x=4→0.1
#   NaN fix: y[0]=y[1]=0.15, y[-1]=y[-2]=0.15 (edges already valid here)
#   sum=2.0, normalised=[0.075,0.075,0.1,0.15,0.2,0.15,0.1,0.075,0.075]
#   Wait — let me redo: original y=[0.1,0.2,0.4,0.2,0.1] at x=[-4,-2,0,2,4]
#     x=-3: interp between -4(0.1) and -2(0.2) → 0.15
#     x=-1: interp between -2(0.2) and  0(0.4) → 0.3
#     x= 1: interp between  0(0.4) and  2(0.2) → 0.3
#     x= 3: interp between  2(0.2) and  4(0.1) → 0.15
#   raw_y=[0.1,0.15,0.2,0.3,0.4,0.3,0.2,0.15,0.1], sum=1.9
#   after NaN fix (edges already ok): same
#   normalised = raw_y/1.9 (each value / 1.9)
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.scalePDF.py_impl import scalePDF


def _make_pdf(min_idx, binsize, y):
    p = SimpleNamespace()
    p.Min = min_idx
    p.BinSize = binsize
    p.x = np.arange(min_idx, -min_idx + 1) * binsize
    p.y = np.asarray(y, dtype=float)
    return p


def test_identity_scale():
    """scale_factor=1 → same Min, same length, normalised y sums to 1."""
    p = _make_pdf(-2, 1.0, [0.1, 0.2, 0.4, 0.2, 0.1])
    out = scalePDF(p, 1.0)
    assert out.Min == -2
    assert len(out.x) == 5
    assert np.sum(out.y) == pytest.approx(1.0, rel=1e-12)


def test_scale_factor_2_min():
    """Min scales and floors correctly."""
    p = _make_pdf(-2, 1.0, [0.1, 0.2, 0.4, 0.2, 0.1])
    out = scalePDF(p, 2.0)
    assert out.Min == -4


def test_scale_factor_2_length():
    """scale_factor=2 doubles grid length (2*4+1=9)."""
    p = _make_pdf(-2, 1.0, [0.1, 0.2, 0.4, 0.2, 0.1])
    out = scalePDF(p, 2.0)
    assert len(out.x) == 9


def test_normalised():
    """Output y always sums to 1."""
    p = _make_pdf(-3, 0.5, [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
    for sf in [0.5, 1.0, 1.5, 2.0]:
        out = scalePDF(p, sf)
        assert np.sum(out.y) == pytest.approx(1.0, rel=1e-10)


def test_x_axis_symmetric():
    """Output x-axis is symmetric around zero."""
    p = _make_pdf(-2, 1.0, [0.1, 0.2, 0.4, 0.2, 0.1])
    out = scalePDF(p, 2.0)
    np.testing.assert_allclose(out.x, -out.x[::-1], atol=1e-12)


def test_binsize_preserved():
    """BinSize is not changed."""
    p = _make_pdf(-2, 0.5, [0.1, 0.2, 0.4, 0.2, 0.1])
    out = scalePDF(p, 2.0)
    assert out.BinSize == 0.5


def test_all_y_nonneg():
    """All y values must be non-negative after interpolation."""
    p = _make_pdf(-2, 1.0, [0.1, 0.2, 0.4, 0.2, 0.1])
    out = scalePDF(p, 2.0)
    assert np.all(out.y >= 0)
