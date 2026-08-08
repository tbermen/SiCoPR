"""Verification tests for pam() — Grey-coded PAM4 NRZ-pair mapping.

# ============================================================
# MATLAB GROUND TRUTH
# Grey-code mapping (from source):
#   [-1,-1] → -1       (PAM4 level 0)
#   [-1, 1] → -1/3     (PAM4 level 1)
#   [ 1, 1] →  1/3     (PAM4 level 2)
#   [ 1,-1] →  1       (PAM4 level 3)
#
# pam([-1,-1, -1,1, 1,1, 1,-1]) → [-1, -1/3, 1/3, 1]
# pam([-1,-1]) → [-1]
# pam([-1, 1]) → [-1/3]
# pam([ 1, 1]) → [ 1/3]
# pam([ 1,-1]) → [  1 ]
# Odd-length input: last element ignored (floor(N/2) pairs).
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.pam.py_impl import pam


def test_all_four_symbols():
    bits = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
    result = pam(bits)
    np.testing.assert_allclose(result, [-1.0, -1.0/3, 1.0/3, 1.0], rtol=1e-12)


def test_level_minus1():
    assert pam(np.array([-1.0, -1.0]))[0] == pytest.approx(-1.0)


def test_level_minus_third():
    assert pam(np.array([-1.0, 1.0]))[0] == pytest.approx(-1.0/3)


def test_level_plus_third():
    assert pam(np.array([1.0, 1.0]))[0] == pytest.approx(1.0/3)


def test_level_plus1():
    assert pam(np.array([1.0, -1.0]))[0] == pytest.approx(1.0)


def test_output_length():
    """Output length is floor(input_length / 2)."""
    bits = np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0])
    assert len(pam(bits)) == 3


def test_odd_length_truncates():
    """Odd-length input: last element ignored."""
    bits = np.array([-1.0, -1.0, 1.0])   # 1 complete pair
    assert len(pam(bits)) == 1
    assert pam(bits)[0] == pytest.approx(-1.0)
