"""Verification tests for TD_CTLE().

# ============================================================
# MATLAB GROUND TRUTH
# p1_ctle = -2*pi*f_p1  (analog pole frequency, rad/s)
# p2_ctle = -2*pi*f_p2
# z_ctle  = -2*pi*f_z * 10^(kacdc_dB/20)
#
# f_p1=1e9, f_p2=2e9, f_z=0.5e9:
#   p1_ctle = -2*pi*1e9 ≈ -6.28318e9
#   p2_ctle = -2*pi*2e9 ≈ -12.56637e9
#   z_ctle (kacdc_dB=0) = -2*pi*0.5e9 ≈ -3.14159e9
#   z_ctle (kacdc_dB=20) = -2*pi*0.5e9*10 = -2*pi*5e9 ≈ -31.4159e9
#
# Bilinear poles/zeros are inside unit circle (stable) for
#   any f_p,f_z << fb*oversampling.
#
# filter applied to delta [1,0,...,0]:
#   output[0] = B_filt[0] / A_filt[0] = B_filt[0]  (A monic)
#   output length = input length
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.TD_CTLE.py_impl import TD_CTLE


def delta(n=32):
    ir = np.zeros(n)
    ir[0] = 1.0
    return ir


def test_p1_ctle_value():
    """p1_ctle = -2*pi*f_p1."""
    _, p1_ctle, _, _ = TD_CTLE(delta(), 25e9, 0.5e9, 1e9, 2e9, 0.0, 2)
    assert p1_ctle == pytest.approx(-2 * np.pi * 1e9, rel=1e-12)


def test_p2_ctle_value():
    """p2_ctle = -2*pi*f_p2."""
    _, _, p2_ctle, _ = TD_CTLE(delta(), 25e9, 0.5e9, 1e9, 2e9, 0.0, 2)
    assert p2_ctle == pytest.approx(-2 * np.pi * 2e9, rel=1e-12)


def test_z_ctle_zero_dB():
    """kacdc_dB=0: z_ctle = -2*pi*f_z."""
    _, _, _, z_ctle = TD_CTLE(delta(), 25e9, 0.5e9, 1e9, 2e9, 0.0, 2)
    assert z_ctle == pytest.approx(-2 * np.pi * 0.5e9, rel=1e-12)


def test_z_ctle_positive_dB():
    """kacdc_dB=20: z_ctle = -2*pi*f_z*10."""
    _, _, _, z_ctle = TD_CTLE(delta(), 25e9, 0.5e9, 1e9, 2e9, 20.0, 2)
    assert z_ctle == pytest.approx(-2 * np.pi * 0.5e9 * 10, rel=1e-12)


def test_output_length_matches_input():
    """Output length equals input length."""
    ir = np.zeros(50)
    ir[0] = 1.0
    ir_out, _, _, _ = TD_CTLE(ir, 25e9, 5e9, 10e9, 20e9, 0.0, 2)
    assert len(ir_out) == 50


def test_output_is_finite():
    """No NaN or inf in output for reasonable parameters."""
    ir = np.zeros(64)
    ir[0] = 1.0
    ir_out, _, _, _ = TD_CTLE(ir, 25e9, 5e9, 10e9, 20e9, -3.0, 2)
    assert np.all(np.isfinite(ir_out))
