"""Verification tests for s_for_c2() — 2-port S-params for a shunt capacitor.

# ============================================================
# MATLAB GROUND TRUTH
# Y = jωC,  s11=s22=-Y*Z/(2+Y*Z),  s21=s12=2/(2+Y*Z)
# where Y*Z = j*2π*f*cpad*zref
#
# f=0 (DC): Y*Z=0, s11=0, s21=1
# f=∞:      s11→-1, s21→0
#
# Specific: zref=50, cpad=1e-12, f=1e9
#   Y*Z = j*2π*1e9*1e-12*50 = j*2π*0.05 ≈ j*0.3142
#   denom = 2 + j*0.3142
#   s21 = 2/(2+j*0.3142), |s21| ≈ 2/|2+j0.3142| = 2/2.0246 ≈ 0.9878
#   s11 = -j*0.3142/(2+j*0.3142), |s11| ≈ 0.3142/2.0246 ≈ 0.1552
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.s_for_c2.py_impl import s_for_c2


def test_dc_thru():
    S = s_for_c2(50.0, np.array([0.0]), 1e-12)
    assert S.Parameters[1, 0, 0] == pytest.approx(1.0, abs=1e-12)
    assert S.Parameters[0, 0, 0] == pytest.approx(0.0, abs=1e-12)


def test_symmetry():
    S = s_for_c2(50.0, np.array([1e9, 5e9]), 1e-12)
    np.testing.assert_array_equal(S.Parameters[0, 0, :], S.Parameters[1, 1, :])
    np.testing.assert_array_equal(S.Parameters[0, 1, :], S.Parameters[1, 0, :])


def test_high_freq_approaches_short():
    S = s_for_c2(50.0, np.array([1e15]), 1e-12)
    assert abs(S.Parameters[1, 0, 0]) == pytest.approx(0.0, abs=1e-3)
    assert abs(S.Parameters[0, 0, 0]) == pytest.approx(1.0, abs=1e-3)


def test_s11_purely_imaginary_contribution():
    """s11 = -jωCZ/(2+jωCZ): numerator is purely imaginary."""
    S = s_for_c2(50.0, np.array([1e9]), 1e-12)
    ycz = 1j * 2 * np.pi * 1e9 * 1e-12 * 50
    expected_s11 = -ycz / (2 + ycz)
    assert S.Parameters[0, 0, 0] == pytest.approx(expected_s11, rel=1e-10)


def test_output_shape():
    S = s_for_c2(50.0, np.linspace(0, 10e9, 11), 1e-12)
    assert S.Parameters.shape == (2, 2, 11)


def test_passive_network():
    """|s11|^2 + |s21|^2 ≤ 1 (passivity) for all frequencies."""
    S = s_for_c2(50.0, np.linspace(1e8, 100e9, 50), 1e-12)
    power = np.abs(S.Parameters[0, 0, :])**2 + np.abs(S.Parameters[1, 0, :])**2
    assert np.all(power <= 1.0 + 1e-12)
