"""Verification tests for FD_CTLE().

# ============================================================
# MATLAB GROUND TRUTH
# hctf = (10^(kacdc_dB/20) + j*freq/f_z) / ((1+j*freq/f_p1)*(1+j*freq/f_p2))
#
# At DC (freq=0):
#   hctf = 10^(kacdc_dB/20) / 1  (purely real)
#   kacdc_dB=0   → H_dc = 1
#   kacdc_dB=20  → H_dc = 10
#   kacdc_dB=-20 → H_dc = 0.1
#
# kacdc_dB=0, f_z=f_p1=f_p2=1e9, freq=1e9:
#   num = 1 + j
#   den = (1+j)*(1+j) = 2j
#   H = (1+j)/(2j) = (1+j)*(-j)/2 = (1-j)/2  →  |H|=1/sqrt(2)
#
# kacdc_dB=0, f_z=1e9, f_p1=2e9, f_p2=3e9, freq=1e9:
#   num = 1+j
#   den = (1+0.5j)*(1+j/3) = 5/6 + 5j/6 = (5/6)*(1+j)
#   H = (1+j)/((5/6)*(1+j)) = 6/5 = 1.2  (real)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.FD_CTLE.py_impl import FD_CTLE


def test_dc_gain_zero_dB():
    """kacdc_dB=0, freq=0 → H=1."""
    H = FD_CTLE(np.array([0.0]), 1e9, 2e9, 4e9, kacdc_dB=0.0)
    assert H[0] == pytest.approx(1.0)


def test_dc_gain_positive_dB():
    """kacdc_dB=20, freq=0 → H=10."""
    H = FD_CTLE(np.array([0.0]), 1e9, 2e9, 4e9, kacdc_dB=20.0)
    assert H[0].real == pytest.approx(10.0, rel=1e-10)


def test_dc_gain_negative_dB():
    """kacdc_dB=-20, freq=0 → H=0.1."""
    H = FD_CTLE(np.array([0.0]), 1e9, 2e9, 4e9, kacdc_dB=-20.0)
    assert H[0].real == pytest.approx(0.1, rel=1e-10)


def test_equal_pole_zero():
    """kacdc_dB=0, f_z=f_p1=f_p2=1e9, freq=1e9 → H=(1-j)/2."""
    H = FD_CTLE(np.array([1e9]), 1e9, 1e9, 1e9, kacdc_dB=0.0)
    assert H[0].real == pytest.approx(0.5, rel=1e-10)
    assert H[0].imag == pytest.approx(-0.5, rel=1e-10)


def test_real_gain_case():
    """kacdc_dB=0, f_z=1e9, f_p1=2e9, f_p2=3e9, freq=1e9 → H=1.2 (real)."""
    H = FD_CTLE(np.array([1e9]), 1e9, 2e9, 3e9, kacdc_dB=0.0)
    assert H[0].real == pytest.approx(1.2, rel=1e-10)
    assert abs(H[0].imag) < 1e-10


def test_output_length_matches_input():
    freq = np.linspace(0, 50e9, 25)
    H = FD_CTLE(freq, 5e9, 15e9, 30e9, kacdc_dB=-3.0)
    assert len(H) == 25
