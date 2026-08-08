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
