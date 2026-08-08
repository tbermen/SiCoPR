"""Verification tests for H_interp().

# ============================================================
# MATLAB GROUND TRUTH (lines 2180-2198)
# mag_db_new = pchip(f_old, 20*log10(|S21|), f_new)
# ph_new = pchip(f_old, unwrap(angle(S21)), f_new)
# H_new = 10^(mag_db_new/20) * exp(j*ph_new)
# H_new(isinf(H_new)) = 0
# inq = last index where f_new <= fb/2
# H_new(inq+1:end) = 0
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.H_interp.py_impl import H_interp


def test_output_length():
    """Output length == len(f_new)."""
    f_old = np.linspace(0.1e9, 50e9, 50)
    S21 = np.ones(50)
    f_new = np.linspace(0.1e9, 50e9, 200)
    H = H_interp(S21, f_old, f_new, 50e9)
    assert len(H) == 200


def test_above_fb2_zeroed():
    """Points above fb/2 are zero."""
    fb = 20e9
    f_old = np.linspace(1e9, 30e9, 30)
    S21 = np.ones(30) + 0j
    f_new = np.linspace(1e9, 30e9, 60)
    H = H_interp(S21, f_old, f_new, fb)
    above = f_new > fb / 2
    np.testing.assert_allclose(H[above], 0.0, atol=1e-12)


def test_no_inf_in_output():
    """No infinite values in output."""
    f_old = np.linspace(1e9, 40e9, 40)
    S21 = np.ones(40) * 0.5
    f_new = np.linspace(1e9, 40e9, 80)
    H = H_interp(S21, f_old, f_new, 40e9)
    assert np.all(np.isfinite(np.abs(H)))


def test_unity_magnitude_preserved_below_fb2():
    """Flat unity magnitude → interpolated magnitude ≈ 1 below fb/2."""
    fb = 40e9
    f_old = np.linspace(1e9, 40e9, 40)
    S21 = np.ones(40) + 0j
    f_new = np.linspace(1e9, 40e9, 40)
    H = H_interp(S21, f_old, f_new, fb)
    below = f_new <= fb / 2
    np.testing.assert_allclose(np.abs(H[below]), 1.0, atol=1e-6)


def test_phase_linear_interpolated():
    """Linear phase S21 → output phase is also linear (pchip = linear for linear data)."""
    f_old = np.linspace(1e9, 20e9, 20)
    phase = -np.pi * f_old / 20e9
    S21 = np.exp(1j * phase)
    f_new = np.linspace(1e9, 20e9, 40)
    H = H_interp(S21, f_old, f_new, 40e9)
    expected_phase = -np.pi * f_new / 20e9
    below = f_new <= 20e9
    np.testing.assert_allclose(np.angle(H[below]) % (2 * np.pi),
                                expected_phase[below] % (2 * np.pi), atol=1e-5)
