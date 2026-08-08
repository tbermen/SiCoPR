"""Verification tests for bessel() — Bessel polynomial coefficients.

# ============================================================
# MATLAB GROUND TRUTH
# Derived analytically from the formula:
#   a(ii+1) = factorial(2*n-ii) / (2^(n-ii) * factorial(ii) * factorial(n-ii))
#
# bessel(0):
#   ii=0: 0!/(2^0 * 0! * 0!) = 1/1 = 1      → [1]
#
# bessel(1):
#   ii=0: 2!/(2^1 * 0! * 1!) = 2/2 = 1
#   ii=1: 1!/(2^0 * 1! * 0!) = 1/1 = 1      → [1, 1]
#
# bessel(2):
#   ii=0: 4!/(2^2 * 0! * 2!) = 24/8  = 3
#   ii=1: 3!/(2^1 * 1! * 1!) = 6/2   = 3
#   ii=2: 2!/(2^0 * 2! * 0!) = 2/2   = 1    → [3, 3, 1]
#
# bessel(3):
#   ii=0: 6!/(2^3 * 0! * 3!) = 720/48 = 15
#   ii=1: 5!/(2^2 * 1! * 2!) = 120/8  = 15
#   ii=2: 4!/(2^1 * 2! * 1!) = 24/4   = 6  (prompt says 15 — recalculate)
#         actually: 4!/(2^1 * 2! * 1!) = 24/(2*2*1) = 6
#   ii=3: 3!/(2^0 * 3! * 0!) = 6/6    = 1   → [15, 15, 6, 1]
#
# Note: the prompt example shows [120, 60, 15, 1] — that is bessel(5), not bessel(3).
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.bessel.py_impl import bessel


def test_bessel_n0():
    """bessel(0) should return [1]."""
    result = bessel(0)
    np.testing.assert_array_equal(result, [1.0])


def test_bessel_n1():
    """bessel(1) → [1, 1]."""
    result = bessel(1)
    np.testing.assert_array_equal(result, [1.0, 1.0])


def test_bessel_n2():
    """bessel(2) → [3, 3, 1]."""
    result = bessel(2)
    np.testing.assert_array_equal(result, [3.0, 3.0, 1.0])


def test_bessel_n3():
    """bessel(3) → [15, 15, 6, 1]."""
    result = bessel(3)
    np.testing.assert_array_equal(result, [15.0, 15.0, 6.0, 1.0])


def test_output_shape():
    """Output length must be n+1."""
    for n in range(6):
        assert len(bessel(n)) == n + 1


def test_output_dtype():
    """Output must be float64."""
    assert bessel(3).dtype == np.float64


def test_bessel_last_coeff_is_one():
    """The last coefficient a[n] is always 1 (ii=n term = (n)!/(1*n!*0!) = 1)."""
    for n in range(1, 8):
        assert bessel(n)[-1] == 1.0


def test_bessel_first_coeff():
    """First coefficient a[0] = (2n)! / (2^n * n!) = double factorial."""
    import math
    for n in range(1, 6):
        expected = math.factorial(2 * n) / (2**n * math.factorial(n))
        assert bessel(n)[0] == pytest.approx(expected)
