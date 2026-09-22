"""Verification tests for pdf_to_cdf() — PDF to combined CDF transform.

# ============================================================
# MATLAB GROUND TRUTH
# Hand-computed from the three-step formula.
#
# Case 1 — symmetric PDF  [0.1, 0.2, 0.4, 0.2, 0.1]:
#   yB = cumsum  = [0.1, 0.3, 0.7, 0.9, 1.0]
#   flip(y)      = [0.1, 0.2, 0.4, 0.2, 0.1]   (same, symmetric)
#   cumsum(flip) = [0.1, 0.3, 0.7, 0.9, 1.0]
#   yT = flip back = [1.0, 0.9, 0.7, 0.3, 0.1]
#   y  = min(yB,yT) = [0.1, 0.3, 0.7, 0.3, 0.1]
#
# Case 2 — asymmetric PDF  [0.1, 0.3, 0.4, 0.15, 0.05]:
#   yB           = [0.10, 0.40, 0.80, 0.95, 1.00]
#   flip(y)      = [0.05, 0.15, 0.40, 0.30, 0.10]
#   cumsum(flip) = [0.05, 0.20, 0.60, 0.90, 1.00]
#   yT = flip back = [1.00, 0.90, 0.60, 0.20, 0.05]
#   y  = min      = [0.10, 0.40, 0.60, 0.20, 0.05]
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.pdf_to_cdf.py_impl import pdf_to_cdf


def _make_pdf(y, x=None):
    p = SimpleNamespace()
    p.y = np.asarray(y, dtype=float)
    p.x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, dtype=float)
    return p


def test_symmetric_pdf_yB():
    c = pdf_to_cdf(_make_pdf([0.1, 0.2, 0.4, 0.2, 0.1]))
    np.testing.assert_allclose(c.yB, [0.1, 0.3, 0.7, 0.9, 1.0], atol=1e-12)


def test_symmetric_pdf_yT():
    c = pdf_to_cdf(_make_pdf([0.1, 0.2, 0.4, 0.2, 0.1]))
    np.testing.assert_allclose(c.yT, [1.0, 0.9, 0.7, 0.3, 0.1], atol=1e-12)


def test_symmetric_pdf_y_min():
    c = pdf_to_cdf(_make_pdf([0.1, 0.2, 0.4, 0.2, 0.1]))
    np.testing.assert_allclose(c.y, [0.1, 0.3, 0.7, 0.3, 0.1], atol=1e-12)


def test_asymmetric_pdf():
    c = pdf_to_cdf(_make_pdf([0.1, 0.3, 0.4, 0.15, 0.05]))
    np.testing.assert_allclose(c.yB, [0.10, 0.40, 0.80, 0.95, 1.00], atol=1e-12)
    np.testing.assert_allclose(c.yT, [1.00, 0.90, 0.60, 0.20, 0.05], atol=1e-12)
    np.testing.assert_allclose(c.y,  [0.10, 0.40, 0.60, 0.20, 0.05], atol=1e-12)


def test_yB_ends_at_one():
    """yB[-1] must equal 1 for any normalised PDF."""
    c = pdf_to_cdf(_make_pdf([0.25, 0.25, 0.25, 0.25]))
    assert c.yB[-1] == pytest.approx(1.0)


def test_yT_starts_at_one():
    """yT[0] must equal 1 for any normalised PDF."""
    c = pdf_to_cdf(_make_pdf([0.25, 0.25, 0.25, 0.25]))
    assert c.yT[0] == pytest.approx(1.0)


def test_x_axis_copied():
    """cdf.x must be identical to pdf.x."""
    x = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
    c = pdf_to_cdf(_make_pdf([0.1, 0.2, 0.4, 0.2, 0.1], x))
    np.testing.assert_array_equal(c.x, x)


def test_y_leq_yB_and_yT():
    """cdf.y must be <= both yB and yT everywhere."""
    c = pdf_to_cdf(_make_pdf([0.05, 0.2, 0.5, 0.2, 0.05]))
    assert np.all(c.y <= c.yB + 1e-15)
    assert np.all(c.y <= c.yT + 1e-15)


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


_OCT_YB = [0.01, 0.029999999999999999, 0.080000000000000002, 0.16999999999999998]
_OCT_YT = [1.0, 0.98999999999999999, 0.96999999999999997, 0.91999999999999993]


def test_matches_com_octave():
    c = pdf_to_cdf(_a_pdf())
    for name, want in (('yB', _OCT_YB), ('yT', _OCT_YT)):
        got = np.asarray(getattr(c, name)).ravel()[:4]
        assert np.allclose(got, want, rtol=0, atol=1e-15), (
            '%s[:4] is %r, COM Octave gives %r' % (name, list(got), want))


def test_the_two_tails_are_complementary():
    """yB accumulates from the bottom and yT from the top, so yB[k] + yT[k]
    must be one PDF bin more than 1 at every k."""
    p = _a_pdf()
    c = pdf_to_cdf(p)
    s = np.asarray(c.yB).ravel() + np.asarray(c.yT).ravel()
    assert np.allclose(s, 1.0 + np.asarray(p.y).ravel(), rtol=0, atol=1e-12), (
        'yB + yT is %r' % list(s[:4]))
