"""Verification tests for N_s().

# ============================================================
# MATLAB GROUND TRUTH
# clause_178: Ns_of_f(1:inq) = 2*sigma_ns^2 / f_b  (flat spectrum)
#   f_b=100e9, sigma_ns=0.01, f=[0,25e9,50e9,75e9,100e9]
#   inq = find(f <= 50e9, 1,'last') = 3 (1-based, elements 0..50e9)
#   Ns[:3] = 2*0.01^2/100e9 = 2e-12
#   Ns[3:] = 0
#
# clause_179:
#   beta = 1 - (2*f_hp/f_b)*atan(f_b/(2*f_hp))
#   Ns[:inq] = (2*sigma_ns^2)/(beta*f_b) * (f[:inq]/f_hp)^2 / (1+(f[:inq]/f_hp)^2)
#   At f=0: Ns=0 (DC is zero)
#
# unsupported RIT → ValueError
# f_hp<=0 with clause_179 → ValueError
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.N_s.py_impl import N_s


def make_op(rit):
    return SimpleNamespace(RIT_REF_PTR=rit)


def make_param(fb=100e9, f_hp=10e9):
    return SimpleNamespace(fb=fb, f_hp=f_hp)


def test_clause_178_flat():
    """clause_178: Ns is flat (constant) up to fb/2."""
    f = np.array([0.0, 25e9, 50e9, 75e9, 100e9])
    p = make_param(fb=100e9)
    op = make_op('clause_178')
    out = N_s(f, p, 0.01, op)
    expected_val = 2 * 0.01**2 / 100e9
    np.testing.assert_allclose(out[:3], expected_val)
    assert out[3] == 0.0
    assert out[4] == 0.0


def test_clause_178_output_length():
    f = np.linspace(0, 100e9, 20)
    out = N_s(f, make_param(), 0.005, make_op('clause_178'))
    assert len(out) == 20


def test_clause_179_dc_is_zero():
    """clause_179: Ns(0) = 0 because f[0]=0."""
    f = np.array([0.0, 10e9, 50e9])
    out = N_s(f, make_param(fb=100e9, f_hp=10e9), 0.01, make_op('clause_179'))
    assert out[0] == pytest.approx(0.0)


def test_clause_179_positive():
    """clause_179: all in-band values ≥ 0."""
    f = np.linspace(0, 50e9, 10)
    out = N_s(f, make_param(fb=100e9, f_hp=5e9), 0.01, make_op('clause_179'))
    assert np.all(out >= 0)


def test_annex_176d_equals_clause_179():
    """annex_176d uses same formula as clause_179."""
    f = np.linspace(1e9, 50e9, 5)
    p = make_param(fb=100e9, f_hp=10e9)
    out1 = N_s(f, p, 0.01, make_op('clause_179'))
    out2 = N_s(f, p, 0.01, make_op('annex_176d'))
    np.testing.assert_allclose(out1, out2)


def test_unsupported_rit_raises():
    with pytest.raises(ValueError):
        N_s(np.array([1e9]), make_param(), 0.01, make_op('bad_clause'))


def test_clause_179_fhp_zero_raises():
    with pytest.raises((ValueError, ZeroDivisionError)):
        N_s(np.array([1e9]), make_param(f_hp=0), 0.01, make_op('clause_179'))
