"""Verification tests for s_for_c4() — 4-port S-params for balanced shunt cap.

# ============================================================
# MATLAB GROUND TRUTH
# s_for_c4 = s2_to_s4(s_for_c2) then snp2smp([1 3 2 4]).
# For the symmetric shunt-cap 2-port (s11=s22, s12=s21), the mixed-mode
# transform is identity (analytically verified).  Result is block-diagonal:
#   [[s2, 0], [0, s2]]  where s2 is the 2-port S-matrix.
#
# At DC (f=0): s2=[[0,1],[1,0]], so S4=block-diag([[0,1],[1,0]])
#   S4[0,0]=0, S4[1,0]=1, S4[2,2]=0, S4[3,2]=1
#   off-diagonal blocks = 0
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.s_for_c4.py_impl import s_for_c4
from com_functions.fn.s_for_c2.py_impl import s_for_c2


def test_output_shape():
    S = s_for_c4(50.0, np.linspace(0, 10e9, 5), 1e-12)
    assert S.Parameters.shape == (4, 4, 5)


def test_dc_block_diagonal():
    """At DC both blocks equal [[0,1],[1,0]] and off-diagonal blocks are zero."""
    S = s_for_c4(50.0, np.array([0.0]), 1e-12)
    p = S.Parameters[:, :, 0]
    np.testing.assert_allclose(p[0:2, 0:2], [[0, 1], [1, 0]], atol=1e-12)
    np.testing.assert_allclose(p[2:4, 2:4], [[0, 1], [1, 0]], atol=1e-12)
    np.testing.assert_allclose(p[0:2, 2:4], np.zeros((2, 2)), atol=1e-12)
    np.testing.assert_allclose(p[2:4, 0:2], np.zeros((2, 2)), atol=1e-12)


def test_both_blocks_equal_s_for_c2():
    """Each diagonal block equals the 2-port S-matrix from s_for_c2."""
    f = np.linspace(1e8, 10e9, 8)
    S2 = s_for_c2(50.0, f, 1e-12)
    S4 = s_for_c4(50.0, f, 1e-12)
    np.testing.assert_allclose(S4.Parameters[0:2, 0:2, :], S2.Parameters, rtol=1e-12)
    np.testing.assert_allclose(S4.Parameters[2:4, 2:4, :], S2.Parameters, rtol=1e-12)


def test_off_diagonal_blocks_zero():
    f = np.linspace(1e9, 5e9, 4)
    S4 = s_for_c4(50.0, f, 1e-12)
    np.testing.assert_allclose(S4.Parameters[0:2, 2:4, :], 0, atol=1e-30)
    np.testing.assert_allclose(S4.Parameters[2:4, 0:2, :], 0, atol=1e-30)


def test_passivity_each_block():
    """|s11|^2 + |s21|^2 ≤ 1 for each 2×2 block."""
    S4 = s_for_c4(50.0, np.linspace(1e8, 100e9, 20), 1e-12)
    for blk in [S4.Parameters[0:2, 0:2, :], S4.Parameters[2:4, 2:4, :]]:
        power = np.abs(blk[0, 0, :])**2 + np.abs(blk[1, 0, :])**2
        assert np.all(power <= 1.0 + 1e-12)
