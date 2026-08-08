"""Verification tests for make_pkg().

# ============================================================
# MATLAB GROUND TRUTH (lines 8359-8405)
# Returns 4 S-parameter arrays (s11,s12,s21,s22) for package model:
#   pad capacitor → (optional L/C comps) → transmission line → ball capacitor
# f(f<eps)=eps to prevent division by zero
# For matched tline (pkg_z=Z0, len=0): s11≈0, s21≈1
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.make_pkg.py_impl import make_pkg


def _param():
    return SimpleNamespace(
        pkg_tau=0.0,
        pkg_gamma0_a1_a2=np.array([0.0, 0.01, 0.001]),
        Z0=50.0,
    )


def test_output_length():
    """Output arrays have length N == len(f)."""
    N = 20
    f = np.linspace(1e9, 50e9, N)
    s11, s12, s21, s22 = make_pkg(f, 0.0, 1e-12, 1e-12, 50.0, _param())
    for arr in (s11, s12, s21, s22):
        assert len(arr) == N


def test_output_is_complex():
    """S-parameters are complex arrays."""
    f = np.linspace(1e9, 50e9, 10)
    s11, s12, s21, s22 = make_pkg(f, 0.01, 1e-12, 1e-12, 50.0, _param())
    assert np.iscomplexobj(s11)


def test_symmetric_tline_s21_near_one():
    """Matched tline (pkg_z=Z0) with tiny length → |s21| close to 1."""
    f = np.linspace(1e8, 10e9, 50)
    s11, _, s21, _ = make_pkg(f, 0.001, 1e-15, 1e-15, 50.0, _param())
    # For very small caps and very short tline, s21 ≈ 1
    np.testing.assert_allclose(np.abs(s21), 1.0, atol=0.1)


def test_with_lcomp():
    """Optional lcomp argument accepted without error."""
    f = np.linspace(1e9, 20e9, 10)
    s11, s12, s21, s22 = make_pkg(f, 0.01, 1e-12, 1e-12, 50.0, _param(), 0.1e-9)
    assert len(s21) == 10


def test_with_cbump():
    """Optional cbump argument accepted without error."""
    f = np.linspace(1e9, 20e9, 10)
    s11, s12, s21, s22 = make_pkg(f, 0.01, 1e-12, 1e-12, 50.0, _param(), 0.0, 0.1e-12)
    assert len(s21) == 10
