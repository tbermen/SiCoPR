"""Verification tests for OptFom_FD_or_TD_Fields().

# ============================================================
# MATLAB GROUND TRUTH (lines 3505-3514)
# if TDMODE: return 'uneq_pulse_response', 'ctle_pulse_response'
# else:       return 'uneq_imp_response',  'ctle_imp_response'
# ============================================================
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_FD_or_TD_Fields.py_impl import OptFom_FD_or_TD_Fields


def test_td_mode_true():
    uneq, ctle = OptFom_FD_or_TD_Fields(True)
    assert uneq == 'uneq_pulse_response'
    assert ctle == 'ctle_pulse_response'


def test_fd_mode_false():
    uneq, ctle = OptFom_FD_or_TD_Fields(False)
    assert uneq == 'uneq_imp_response'
    assert ctle == 'ctle_imp_response'


def test_td_mode_int_1():
    """TDMODE=1 (truthy) → TD field names."""
    uneq, ctle = OptFom_FD_or_TD_Fields(1)
    assert uneq == 'uneq_pulse_response'
    assert ctle == 'ctle_pulse_response'


def test_fd_mode_int_0():
    """TDMODE=0 (falsy) → FD field names."""
    uneq, ctle = OptFom_FD_or_TD_Fields(0)
    assert uneq == 'uneq_imp_response'
    assert ctle == 'ctle_imp_response'
