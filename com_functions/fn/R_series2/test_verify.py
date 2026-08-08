"""Verification tests for R_series2() — 2-port S-params for a series resistor.

# ============================================================
# MATLAB GROUND TRUTH
# s11=s22=R/(R+2*zref),  s21=s12=2*zref/(R+2*zref)
#
# zref=50, R=50:  s11=50/150=1/3,  s21=100/150=2/3
# zref=50, R=0:   s11=0,           s21=1   (short: lossless thru)
# zref=50, R→∞:   s11→1,           s21→0   (open: full reflection)
# zref=50, R=100: s11=100/200=0.5, s21=100/200=0.5
#
# Note: s11+s21 = R/(R+2Z) + 2Z/(R+2Z) = (R+2Z)/(R+2Z) = 1  (always)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.R_series2.py_impl import R_series2


def test_known_values():
    S = R_series2(50.0, np.array([1e9]), 50.0)
    assert S.Parameters[0, 0, 0] == pytest.approx(1/3, rel=1e-10)
    assert S.Parameters[1, 0, 0] == pytest.approx(2/3, rel=1e-10)


def test_zero_resistance_thru():
    S = R_series2(50.0, np.array([1e9]), 0.0)
    assert S.Parameters[0, 0, 0] == pytest.approx(0.0)
    assert S.Parameters[1, 0, 0] == pytest.approx(1.0)


def test_symmetry_s11_eq_s22():
    S = R_series2(50.0, np.array([1e9, 2e9]), 75.0)
    np.testing.assert_array_equal(S.Parameters[0, 0, :], S.Parameters[1, 1, :])


def test_symmetry_s12_eq_s21():
    S = R_series2(50.0, np.array([1e9, 2e9]), 75.0)
    np.testing.assert_array_equal(S.Parameters[0, 1, :], S.Parameters[1, 0, :])


def test_s11_plus_s21_equals_one():
    """s11 + s21 = 1 always (energy conservation for this lossless-style formula)."""
    S = R_series2(50.0, np.array([1e9]), 100.0)
    assert (S.Parameters[0, 0, 0] + S.Parameters[1, 0, 0]).real == pytest.approx(1.0)


def test_output_shape():
    S = R_series2(50.0, np.linspace(1e9, 10e9, 7), 50.0)
    assert S.Parameters.shape == (2, 2, 7)


def test_frequency_independent():
    """Result is constant across frequencies."""
    f = np.array([1e9, 5e9, 10e9])
    S = R_series2(50.0, f, 50.0)
    assert np.all(S.Parameters[0, 0, :] == S.Parameters[0, 0, 0])
