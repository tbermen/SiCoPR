"""Verification tests for SL().

# ============================================================
# MATLAB GROUND TRUTH (lines 4368-4403)
# S = sparameters struct; S.Parameters shape (2, 2, N)
# R=0: return S unchanged (with warning)
# R=R_0: return S unchanged (else branch)
# R > R_0: cascade series resistor then S
# R < R_0: cascade shunt resistor then S
#
# Series: s_series = R/(R+2*Z0); s21_series = 2*Z0/(R+2*Z0)
# Shunt:  s11_shunt = -Z0/(Z0+2*Rpar); s21_shunt = 2*Rpar/(Z0+2*Rpar)
#          where Rpar = -R*Z0/(R-Z0) (from R < Z0 branch)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.SL.py_impl import SL


def _sparams(N=10, val=0.5):
    """Simple sparameters struct with Parameters of shape (2,2,N)."""
    p = np.zeros((2, 2, N), dtype=complex)
    p[0, 1, :] = p[1, 0, :] = val  # S12=S21=val
    return SimpleNamespace(Parameters=p, Frequencies=np.linspace(1e9, 20e9, N),
                           Impedance=50.0, NumPorts=2)


def test_r_equals_r0_returns_unchanged():
    """R=R_0: identity cascade → Parameters unchanged."""
    S = _sparams()
    SLD = SL(S, np.linspace(1e9, 20e9, 10), 50.0, 50.0)
    np.testing.assert_array_equal(SLD.Parameters, S.Parameters)


def test_r_zero_returns_s_unchanged():
    """R=0: returns S without cascade (with MATLAB warning in original)."""
    S = _sparams()
    SLD = SL(S, np.linspace(1e9, 20e9, 10), 0.0, 50.0)
    np.testing.assert_array_equal(SLD.Parameters, S.Parameters)


def test_r_greater_modifies_s11():
    """R > R_0: series cascade changes S11."""
    S = _sparams()
    f = np.linspace(1e9, 20e9, 10)
    SLD = SL(S, f, 100.0, 50.0)
    # S11 is no longer zero
    assert not np.allclose(SLD.Parameters[0, 0, :], 0.0)


def test_r_less_modifies_s11():
    """R < R_0: shunt cascade changes S11."""
    S = _sparams()
    f = np.linspace(1e9, 20e9, 10)
    SLD = SL(S, f, 25.0, 50.0)
    assert not np.allclose(SLD.Parameters[0, 0, :], 0.0)


def test_input_s_not_mutated():
    """Original S.Parameters is not modified."""
    S = _sparams()
    original = S.Parameters.copy()
    SL(S, np.linspace(1e9, 20e9, 10), 100.0, 50.0)
    np.testing.assert_array_equal(S.Parameters, original)
