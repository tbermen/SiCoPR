"""Verification tests for r_parrelell2() — 2-port S-params for a shunt resistor.

# ============================================================
# MATLAB GROUND TRUTH
# s11=s22 = -zref/(rpad*(zref/rpad+2)) = -zref/(zref+2*rpad)
# s21=s12 = 2/(zref/rpad+2) = 2*rpad/(zref+2*rpad)
#
# zref=50, rpad=50:   denom=150, s11=-50/150=-1/3, s21=100/150=2/3
# zref=50, rpad→∞:    s11→0, s21→1  (open circuit: no effect)
# zref=50, rpad=25:   denom=100, s11=-50/100=-0.5, s21=50/100=0.5
# Lossless check: |s11|^2 + |s21|^2 = 1/9+4/9 = 5/9 (lossy network)
# Energy conservation: s11+s21 = (-zref+2rpad)/(zref+2rpad) ≠ 1 (shunt load)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.r_parrelell2.py_impl import r_parrelell2


def test_known_values():
    S = r_parrelell2(50.0, np.array([1e9]), 50.0)
    assert S.Parameters[0, 0, 0] == pytest.approx(-1/3, rel=1e-10)
    assert S.Parameters[1, 0, 0] == pytest.approx(2/3, rel=1e-10)


def test_large_rpad_approaches_thru():
    """Very large rpad → open circuit → s11≈0, s21≈1."""
    S = r_parrelell2(50.0, np.array([1e9]), 1e12)
    assert abs(S.Parameters[0, 0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert abs(S.Parameters[1, 0, 0]) == pytest.approx(1.0, rel=1e-6)


def test_quarter_ohm():
    S = r_parrelell2(50.0, np.array([1e9]), 25.0)
    assert S.Parameters[0, 0, 0] == pytest.approx(-0.5, rel=1e-10)
    assert S.Parameters[1, 0, 0] == pytest.approx(0.5, rel=1e-10)


def test_symmetry():
    S = r_parrelell2(50.0, np.array([1e9, 2e9]), 75.0)
    np.testing.assert_array_equal(S.Parameters[0, 0, :], S.Parameters[1, 1, :])
    np.testing.assert_array_equal(S.Parameters[0, 1, :], S.Parameters[1, 0, :])


def test_s11_negative():
    """Shunt resistor always produces negative s11 (unlike series)."""
    S = r_parrelell2(50.0, np.array([1e9]), 100.0)
    assert S.Parameters[0, 0, 0].real < 0


def test_output_shape():
    S = r_parrelell2(50.0, np.linspace(1e9, 10e9, 5), 50.0)
    assert S.Parameters.shape == (2, 2, 5)
