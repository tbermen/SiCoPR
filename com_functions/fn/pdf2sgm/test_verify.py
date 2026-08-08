"""Verification tests for pdf2sgm() — weighted standard deviation of a PDF.

# ============================================================
# MATLAB GROUND TRUTH
# Formula: avg = sum(x.*y),  sgm = sqrt(sum((x-avg).^2 .* y))
#
# Case 1 — zero-mean symmetric PDF:
#   x=[-1, 0, 1],  y=[0.25, 0.50, 0.25]
#   avg = (-1)*0.25 + 0*0.50 + 1*0.25 = 0
#   sgm = sqrt(1*0.25 + 0*0.50 + 1*0.25) = sqrt(0.5) ≈ 0.70710678
#
# Case 2 — degenerate (all mass at one point):
#   x=[0, 1, 2],  y=[0, 1, 0]
#   avg = 1,  sgm = sqrt(0) = 0
#
# Case 3 — non-zero mean:
#   x=[0, 1, 2],  y=[0.25, 0.50, 0.25]
#   avg = 0*0.25 + 1*0.50 + 2*0.25 = 1.0
#   sgm = sqrt((0-1)^2*0.25 + (1-1)^2*0.50 + (2-1)^2*0.25)
#       = sqrt(0.25 + 0 + 0.25) = sqrt(0.5) ≈ 0.70710678
#
# Case 4 — uniform distribution on [-2,-1,0,1,2] (y=0.2 each):
#   avg = 0 (symmetric)
#   sgm = sqrt((4+1+0+1+4)*0.2) = sqrt(2.0) ≈ 1.41421356
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.pdf2sgm.py_impl import pdf2sgm


def _make_pdf(x, y):
    p = SimpleNamespace()
    p.x = np.asarray(x, dtype=float)
    p.y = np.asarray(y, dtype=float)
    return p


def test_zero_mean_symmetric():
    p = _make_pdf([-1.0, 0.0, 1.0], [0.25, 0.50, 0.25])
    assert pdf2sgm(p) == pytest.approx(np.sqrt(0.5), rel=1e-12)


def test_degenerate_all_mass_at_one_point():
    p = _make_pdf([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
    assert pdf2sgm(p) == pytest.approx(0.0, abs=1e-12)


def test_nonzero_mean():
    p = _make_pdf([0.0, 1.0, 2.0], [0.25, 0.50, 0.25])
    assert pdf2sgm(p) == pytest.approx(np.sqrt(0.5), rel=1e-12)


def test_uniform():
    p = _make_pdf([-2.0, -1.0, 0.0, 1.0, 2.0], [0.2, 0.2, 0.2, 0.2, 0.2])
    assert pdf2sgm(p) == pytest.approx(np.sqrt(2.0), rel=1e-12)


def test_returns_scalar():
    p = _make_pdf([-1.0, 0.0, 1.0], [0.25, 0.50, 0.25])
    result = pdf2sgm(p)
    assert np.isscalar(result) or result.ndim == 0


def test_nonnegative():
    p = _make_pdf([-1.0, 0.0, 1.0], [0.25, 0.50, 0.25])
    assert pdf2sgm(p) >= 0.0
