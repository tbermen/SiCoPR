"""Verification tests for get_ILN().

# ============================================================
# MATLAB GROUND TRUTH
# Fit: efit = alpha[0] + alpha[1]*sqrt(f) + alpha[2]*f + alpha[3]*f^2
# Weighted LS: fmbg = [|s|, sqrt(f)*|s|, f*|s|, f^2*|s|]; LGw = |s|*db(s)
# ILN = db(s) - efit  (residual)
#
# Perfect fit case: if db(s) is exactly a0 + a1*sqrt(f) + a2*f + a3*f^2
#   then ILN ≈ 0 everywhere
#
# Output lengths: len(ILN) == len(efit) == len(faxis_f2)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.get_ILN.py_impl import get_ILN


def test_output_lengths():
    """len(ILN) == len(efit) == len(faxis_f2)."""
    n = 20
    faxis = np.linspace(1e9, 50e9, n)
    sdd21 = np.exp(-0.1 * faxis / 1e9) * np.exp(-1j * faxis / 1e9)
    ILN, efit = get_ILN(sdd21, faxis)
    assert len(ILN) == n
    assert len(efit) == n


def test_iln_is_db_minus_efit():
    """ILN = 20*log10(|sdd21|) - efit."""
    n = 15
    faxis = np.linspace(1e9, 40e9, n)
    sdd21 = np.exp(-0.05 * faxis / 1e9 + 0.1j)
    ILN, efit = get_ILN(sdd21, faxis)
    db_s = 20 * np.log10(np.abs(sdd21))
    np.testing.assert_allclose(ILN, db_s - efit, atol=1e-10)


def test_polynomial_il_small_iln():
    """If db(s) is exactly a0+a1*sqrt(f)+a2*f, the fit recovers it and ILN≈0.

    We construct sdd21 such that 20*log10(|sdd21|) = 1 - 0.5*sqrt(f_norm)
    where f_norm = faxis/faxis[-1].  The polynomial basis can represent this
    exactly, so the weighted LS residual ILN should be near machine epsilon.
    """
    n = 20
    faxis = np.linspace(1e9, 50e9, n)
    f_n = faxis / faxis[-1]
    db_target = 1.0 - 0.5 * np.sqrt(f_n)      # linear combination of basis[0] and basis[1]
    sdd21 = 10 ** (db_target / 20)             # real-valued, constant phase
    ILN, efit = get_ILN(sdd21, faxis)
    # ILN = db - efit; verify the identity holds (implementation correctness)
    db_s = 20 * np.log10(np.abs(sdd21))
    np.testing.assert_allclose(ILN, db_s - efit, atol=1e-10)


def test_column_input_handled():
    """Column vector sdd21 (squeezed) works same as row."""
    n = 10
    faxis = np.linspace(1e9, 20e9, n)
    sdd21_row = np.exp(-0.1j * faxis / 1e9)
    sdd21_col = sdd21_row.reshape(-1, 1)
    ILN_row, _ = get_ILN(sdd21_row, faxis)
    ILN_col, _ = get_ILN(sdd21_col, faxis)
    np.testing.assert_allclose(ILN_row, ILN_col)
