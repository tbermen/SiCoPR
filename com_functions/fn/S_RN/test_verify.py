"""Verification tests for S_RN().

# ============================================================
# MATLAB GROUND TRUTH
# S_RN_of_f = eta_0/2 * |H_CTF * H_R|^2
# H_CTF = (10^(G_DC/20) + j*f/z1) * (10^(G_DC2/20) + j*f/zlf)
#          / ((1+j*f/p1) * (1+j*f/p2) * (1+j*f/plf))
# H_R = 1 / polyval([1,2.613126,3.414214,2.613126,1], j*f/(f_r*fb))
#
# G_DC=0, G_DC2=0: H_CTF numerator at DC = (1 + 0) * (1 + 0) = 1
#                  H_CTF denominator at DC = 1 * 1 * 1 = 1  → H_CTF(0)=1
# H_R(0) = 1/polyval(...,0) = 1/1 = 1
# S_RN(0) = eta_0/2 * 1 = eta_0/2
#
# f → ∞: H_CTF and H_R fall off → S_RN decreases
# Output length == len(f)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.S_RN.py_impl import S_RN


def make_param(fb=106.25e9, f_r=0.75, eta_0=1e-14,
               fp1=15e9, fz=6e9, fp2=30e9, fHP=0.01):
    return SimpleNamespace(
        fb=fb, f_r=f_r, eta_0=eta_0,
        CTLE_fp1=np.array([fp1]),
        CTLE_fz=np.array([fz]),
        CTLE_fp2=np.array([fp2]),
        f_HP=np.array([fHP]),
    )


def test_dc_value():
    """At f=0, G_DC=0, G_DC2=0: S_RN(0) = eta_0/2."""
    f = np.array([0.0])
    p = make_param(eta_0=2.0, fHP=1.0)
    out = S_RN(f, 0.0, 0.0, p)
    assert float(out[0]) == pytest.approx(1.0, rel=1e-6)


def test_output_length():
    f = np.linspace(0, 50e9, 15)
    out = S_RN(f, 0.0, 0.0, make_param())
    assert len(out) == 15


def test_non_negative():
    """S_RN is real and non-negative everywhere."""
    f = np.linspace(0, 100e9, 20)
    out = S_RN(f, 0.0, 0.0, make_param())
    assert np.all(np.isreal(out))
    assert np.all(out >= 0)


def test_positive_gdc_increases_output():
    """Positive G_DC amplifies numerator → S_RN increases at DC."""
    f = np.array([0.0])
    p = make_param(eta_0=1.0, fHP=1.0)
    out0 = float(S_RN(f, 0.0, 0.0, p)[0])
    out10 = float(S_RN(f, 10.0, 0.0, p)[0])
    assert out10 > out0
