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


# ---------------------------------------------------------------------------
# Against COM Octave, via tools/octave_oracle.py. The assertions above check
# shape and monotonicity; these pin the values.
# ---------------------------------------------------------------------------

_BINSIZE, _MIN = 1e-3, -6


def _a_pdf():
    """A small asymmetric discrete PDF on a 13-bin axis, summing to 1."""
    y = np.array([0.01, 0.02, 0.05, 0.09, 0.14, 0.18, 0.20,
                  0.13, 0.08, 0.05, 0.03, 0.015, 0.005])
    x = (np.arange(_MIN, _MIN + len(y))) * _BINSIZE
    return SimpleNamespace(BinSize=_BINSIZE, Min=_MIN, y=y, x=x)


_OCT_BINSIZE, _OCT_MIN, _OCT_LEN = 0.001, -15, 31
_OCT_Y3 = [0.0056022408963585443, 0.0056022408963585443, 0.0072028811524609843]


def test_matches_com_octave():
    q = scalePDF(_a_pdf(), 2.5)
    assert q.BinSize == _OCT_BINSIZE, 'BinSize %r, COM Octave %r' % (q.BinSize, _OCT_BINSIZE)
    assert q.Min == _OCT_MIN, 'Min %r, COM Octave %r' % (q.Min, _OCT_MIN)
    y = np.asarray(q.y).ravel()
    assert y.size == _OCT_LEN, 'len(y) %d, COM Octave %d' % (y.size, _OCT_LEN)
    assert np.allclose(y[:3], _OCT_Y3, rtol=0, atol=1e-16), (
        'y[:3] is %r, COM Octave gives %r' % (list(y[:3]), _OCT_Y3))


def test_scaling_preserves_total_probability():
    q = scalePDF(_a_pdf(), 2.5)
    assert abs(float(np.sum(np.asarray(q.y))) - 1.0) < 1e-12, (
        'scaled PDF sums to %r' % float(np.sum(np.asarray(q.y))))


# ---------------------------------------------------------------------------
# COM Octave, 2026-09-22.
#
# interp1's default method returns NaN OUTSIDE the data range. np.interp clamps
# to the end values instead, so it never produces a NaN and the reference's two
# "NAN interp work around" lines (which patch only y(1) and y(end)) become
# no-ops. That is harmless only while at most one point falls outside at each
# end. On a left-heavy grid -- max(x) < -min(x), so pdf_out.x reaches further
# right than the scaled input does -- MATLAB returns NaN for everything past
# the data, and the whole normalised result is NaN.
#
# The port's own header called this "harmless"; it is not. Reachable through
# scaleCDF and through adjust_Rx_noise_for_quantization, which import this.
# ---------------------------------------------------------------------------

def test_octave_left_heavy_grid_is_all_nan():
    """COM Octave: Min=-8, x=(-8:0)*0.05, scale 1.0 -> 17 values, all NaN.

    pdf_out.x spans (Min:-Min) = -0.40..0.40 while the input only reaches 0.0,
    so every point right of zero is outside the data and interp1 gives NaN.
    Normalising by a sum that is NaN then takes the rest with it.
    """
    y = np.array([0.02, 0.05, 0.09, 0.14, 0.20, 0.22, 0.15, 0.08, 0.05])
    x = np.arange(-8, 1) * 0.05
    out = scalePDF(SimpleNamespace(BinSize=0.05, Min=-8, x=x, y=y), 1.0)
    got = np.asarray(out.y)
    assert got.size == 17
    assert np.isnan(got).all(), (
        'clamping instead of NaN hides that the scaled pdf does not cover the '
        'output grid; MATLAB returns NaN here')


def test_octave_symmetric_grid_still_returns_finite_values():
    """The guard: an ordinary symmetric pdf must NOT become NaN."""
    y = np.array([0.05, 0.15, 0.30, 0.30, 0.15, 0.05])
    x = np.arange(-3, 3) * 0.05
    out = scalePDF(SimpleNamespace(BinSize=0.05, Min=-3, x=x, y=y), 1.0)
    got = np.asarray(out.y)
    assert np.isfinite(got).all()
    assert abs(float(np.sum(got)) - 1.0) < 1e-12
