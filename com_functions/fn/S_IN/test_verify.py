"""Verification tests for S_IN().

# ============================================================
# MATLAB GROUND TRUTH
# S_IN_of_f = Ns_of_f / 2 * |Hn21 * H_CTF * H_R|^2
# With Hn21 = 0 → S_IN = 0
# With sigma_ns=0 (→ tiny) and clause_178: Ns ≈ 0 → S_IN ≈ 0
# At DC (f=0): H_CTF(0) = 1 (G_DC=0,G_DC2=0); H_R(0)=1; Ns_178(0)=flat value
#   S_IN(0) = Ns(0)/2 * |Hn21(0)|^2
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.S_IN.py_impl import S_IN


def make_param(fb=100e9, f_r=0.75, sigma_ns=0.01, f_hp=10e9):
    return SimpleNamespace(
        fb=fb, f_r=f_r, sigma_ns=sigma_ns, f_hp=f_hp,
        CTLE_fp1=np.array([15e9]),
        CTLE_fz=np.array([6e9]),
        CTLE_fp2=np.array([30e9]),
        f_HP=np.array([0.01]),
    )


def make_op(rit='clause_178'):
    return SimpleNamespace(RIT_REF_PTR=rit)


def test_zero_noise_path():
    """Hn21 = 0 → S_IN = 0."""
    f = np.linspace(0, 50e9, 10)
    out = S_IN(f, np.zeros(10), 0.0, 0.0, make_param(), make_op())
    np.testing.assert_allclose(out, 0.0)


def test_output_length():
    f = np.linspace(1e9, 50e9, 12)
    out = S_IN(f, np.ones(12), 0.0, 0.0, make_param(), make_op())
    assert len(out) == 12


def test_non_negative():
    """S_IN is real and non-negative."""
    f = np.linspace(1e9, 50e9, 15)
    out = S_IN(f, np.ones(15), 0.0, 0.0, make_param(), make_op())
    assert np.all(out >= 0)


def test_scales_quadratically_with_noise_path():
    """Doubling |Hn21| quadruples S_IN."""
    f = np.array([10e9, 20e9])
    p = make_param()
    op = make_op()
    out1 = S_IN(f, np.ones(2), 0.0, 0.0, p, op)
    out2 = S_IN(f, 2.0 * np.ones(2), 0.0, 0.0, p, op)
    np.testing.assert_allclose(out2, 4 * out1, rtol=1e-10)
