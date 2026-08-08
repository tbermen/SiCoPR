"""Verification tests for PRBS13Q().

# ============================================================
# MATLAB GROUND TRUTH (lines 4174-4192)
# LFSR with taps=[13,12,2,1], seed=[0,0,0,0,0,1,0,1,0,1,0,1,1]
# generates 2^13-2 = 8190 NRZ bits.
# PAM4 mapping pairs NRZ bits → {-1, -1/3, 1/3, 1} (Grey coding)
# seq has 4095 PAM4 samples
# syms: integers {0,1,2,3}
# syms_nrz: NRZ bit sequence
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.PRBS13Q.py_impl import PRBS13Q


def test_returns_three_outputs():
    seq, syms, syms_nrz = PRBS13Q()
    assert seq is not None
    assert syms is not None
    assert syms_nrz is not None


def test_seq_values_are_pam4():
    """seq contains only {-1, -1/3, 1/3, 1}."""
    seq, _, _ = PRBS13Q()
    valid = {-1.0, -1.0/3.0, 1.0/3.0, 1.0}
    for v in seq:
        assert any(abs(v - x) < 1e-10 for x in valid), f'unexpected value {v}'


def test_syms_integer_range():
    """syms contains integers in {0, 1, 2, 3}."""
    _, syms, _ = PRBS13Q()
    assert np.all((syms >= 0) & (syms <= 3))


def test_seq_length():
    """seq has 4095 samples (2^13-2 bits → 4095 PAM4 symbols)."""
    seq, _, _ = PRBS13Q()
    assert len(seq) == 4095


def test_syms_nrz_is_binary():
    """syms_nrz values are +1 or -1."""
    _, _, syms_nrz = PRBS13Q()
    unique = np.unique(syms_nrz)
    assert set(unique.tolist()) == {-1.0, 1.0}


def test_all_four_levels_present():
    """All four PAM4 levels appear in seq."""
    seq, _, _ = PRBS13Q()
    levels = {-1.0, -1.0/3.0, 1.0/3.0, 1.0}
    for lvl in levels:
        assert any(abs(v - lvl) < 1e-10 for v in seq), f'level {lvl} not found'
