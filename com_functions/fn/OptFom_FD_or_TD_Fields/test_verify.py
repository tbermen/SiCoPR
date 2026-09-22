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


# ---------------------------------------------------------------------------
# COM Octave oracle (2026-09-22), reference OptFom_FD_or_TD_Fields run verbatim
# under Octave from octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical
# to matlab/com_ieee8023_4p16p0.m for this function).
#
# `if TDMODE` in MATLAB is true only for a NON-EMPTY value whose elements are
# ALL non-zero.  The previous body used Python's `if TDMODE:`, which RAISED
# ValueError on any numpy array of other than one element and answered TD for
# the list [1, 0].  Measured under Octave:
#     TDMODE            uneq_field
#     []                uneq_imp_response
#     [1 0]             uneq_imp_response
#     [1 1]             uneq_pulse_response
#     ''                uneq_imp_response
#     'x'               uneq_pulse_response
#     -1                uneq_pulse_response
# ---------------------------------------------------------------------------
import numpy as np
import pytest


@pytest.mark.parametrize('tdmode,expect_td', [
    (np.array([]), False),
    (np.array([1, 0]), False),
    (np.array([1, 1]), True),
    ([1, 0], False),
    ([], False),
    ('', False),
    ('x', True),
    (-1, True),
    (2, True),
])
def test_matlab_if_semantics(tdmode, expect_td):
    uneq, ctle = OptFom_FD_or_TD_Fields(tdmode)
    if expect_td:
        assert (uneq, ctle) == ('uneq_pulse_response', 'ctle_pulse_response')
    else:
        assert (uneq, ctle) == ('uneq_imp_response', 'ctle_imp_response')
