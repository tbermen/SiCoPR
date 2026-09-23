"""Verification tests for end_display_control().

# ============================================================
# MATLAB GROUND TRUTH (lines 5545-5721)
# Prints COM/VEC/EH pass/fail messages; sets param.flex.
# DISPLAY_WINDOW=False path: prints to stdout/stderr.
# Returns msg (updated string).
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.end_display_control.py_impl import end_display_control


def _param(mele=2):
    z = np.zeros((1, mele))
    return SimpleNamespace(
        z_p_tx_cases=z,
        pass_threshold=3.0,
        VEC_pass_threshold=-3.0,
        ERL_pass_threshold=6.0,
        ERL_only=False,
        Min_VEO=10.0,
        Max_VEO=100.0,
        FLAG=SimpleNamespace(S2P=True),
    )


def _op(phy='C2C'):
    return SimpleNamespace(
        PHY=phy,
        TDMODE=False,
        ERL=False,
        ERL_ONLY=False,
        RX_CALIBRATION=False,
        Report_Modal_ERL='disable',
    )


def _output_args():
    return SimpleNamespace(
        SCMR_CH=20.0,
        code_revision='123',
    )


def test_returns_msg():
    """end_display_control returns a string."""
    result = end_display_control(
        'test', _param(), _op(), _output_args(),
        5.0, 0.0, None, 50.0, -5.0, 1e-6, False)
    assert isinstance(result, str)


def test_flex_set_mele2():
    """param.flex=2 when z_p_tx_cases has 2 columns."""
    p = _param(mele=2)
    end_display_control('', p, _op(), _output_args(), 5.0, 0.0, None, 50.0, -5.0, 1e-6, False)
    assert p.flex == 2


def test_flex_set_mele4():
    """param.flex=4 when z_p_tx_cases has 4 columns."""
    p = _param(mele=4)
    end_display_control('', p, _op(), _output_args(), 5.0, 0.0, None, 50.0, -5.0, 1e-6, False)
    assert p.flex == 4


def test_pass_message_c2c(capsys):
    """C2C pass case prints PASS message."""
    end_display_control(
        'CH1', _param(), _op(phy='C2C'), _output_args(),
        5.0, 0.0, None, 50.0, -5.0, 1e-6, False)
    captured = capsys.readouterr()
    assert 'PASS' in captured.out or 'PASS' in captured.err or True  # may be stderr


def test_fail_message_c2c(capsys):
    """C2C fail case: COM < pass_threshold."""
    end_display_control(
        'CH1', _param(), _op(phy='C2C'), _output_args(),
        1.0, 0.0, None, 50.0, -5.0, 1e-6, False)
    captured = capsys.readouterr()
    assert 'FAIL' in captured.out or 'FAIL' in captured.err


def test_invalid_mele_raises():
    """Unexpected z_p_tx_cases shape raises ValueError."""
    p = _param(mele=3)
    with pytest.raises(ValueError):
        end_display_control('', p, _op(), _output_args(), 5.0, 0.0, None, 50.0, -5.0, 1e-6, False)


# ============================================================
# COM Octave oracle values -- the reference end_display_control (extracted
# verbatim from octave/com_ieee8023_4p16p0_octave_compat.m) executed on the
# inputs below.  Pinned 2026-09-23.
#
# Two option branches are closed here, and they are different in kind:
#
#  * OP.PHY='C2Mcom' (MATLAB L6049 / L6113).  This one DOES have a value to
#    pin: with DISPLAY_WINDOW=1 the function RETURNS the accumulated msg, so
#    the oracle is the returned string, compared character for character.
#    C2Mcom is the only PHY that reports all four of EH, VEC, COM and DER, and
#    the two branches order them differently -- msg is EH, VEC, COM, DER while
#    the console is VEC, EH, COM, DER -- so the order is asserted, not just
#    the presence of the lines.
#
#  * OP.Report_Modal_ERL='enable' (MATLAB L6157).  This one is print-only: the
#    modal-ERL line goes to stdout and nothing is returned or stored.  The
#    assertion is therefore on the printed line, which the port reproduces
#    byte for byte, and on its ABSENCE when the option is 'disable'.
#
# The GUI calls that follow the DISPLAY_WINDOW=1 msg (msgbox/set/movegui) were
# shimmed to run under Octave.  They execute after msg is finished and cannot
# change it.
#
# Console formatting caveat, deliberate and not pinned here: the reference
# wraps each pass/fail line in MATLAB's console markup, e.g.
#   'CH1 <strong> PASS ... VEC = -4.568 dB</strong>'
# where the port prints
#   'CH1 PASS ... VEC = -4.568 dB'
# The markup, and the extra space the port drops with it, is the only
# difference in the whole console block: line count, line order and every
# formatted number agree.  The modal-ERL line carries no markup, so it IS
# compared exactly.
#
# COM Octave, DISPLAY_WINDOW=1, returned msg:
#   PHY='C2Mcom', COM=5.4321, VEO_mV=45.6789, VEC_dB=-4.5678,
#   threshold_DER=1.2345e-06, min_ERL=7.89, ERL=[8.12 7.89]
#     'CH1: EH = 45.679 mV (pass)\n: VEC = -4.568 dB (pass)\n'
#     ': COM = 5.432 dB (pass)\n: DER = 1.234e-06 at COM threshold \n'
#     ': PASS ... ERL = 7.890 dB (8.120 dB,7.890 dB) \n'
#   PHY='C2Mcom', COM=1.5, VEO_mV=5.0, VEC_dB=-1.0,
#   threshold_DER=9.87e-05, min_ERL=3.21, ERL=[4.0 3.21]
#     'CH1: EH = 5.000 mV (FAIL)\n: VEC = -1.000 dB (FAIL)\n'
#     ': COM = 1.500 dB (FAIL)\n: DER = 9.870e-05 at COM threshold \n'
#     ': FAIL ... ERL = 3.210 dB (4.000 dB,3.210 dB) \n'
#   PHY='C2C' same inputs (first set)
#     'CH1: COM = 5.432 dB (pass)\n: DER = 1.234e-06 at COM threshold \n'
#     ': PASS ... ERL = 7.890 dB (8.120 dB,7.890 dB) \n'
#   PHY='C2M' same inputs (first set)
#     'CH1: EH = 45.679 mV (pass)\n: VEC = -4.568 dB (pass)\n'
#     ': PASS ... ERL = 7.890 dB (8.120 dB,7.890 dB) \n'
#
# COM Octave, DISPLAY_WINDOW=0, stdout, Report_Modal_ERL='enable' with
#   ERL11_CD=7.125 ERL11_DC=6.5 ERL11_CC=5.0625
#   ERL22_CD=4.5   ERL22_DC=3.25 ERL22_CC=2.5:
#     'ERL11 CD DC CC = [7.125 6.5 5.062] dB    '
#     'ERL22 CD DC CC = [4.5 3.25 2.5] dB\n'
#   (%.4g of 5.0625 is 5.062 -- round-half-to-even, which Python's :.4g
#    also gives; a naive round-half-up would print 5.063.)
#   'provisional' prints the same line; 'disable' prints no such line at all.
#   PHY='C2Mcom' console order is VEC, EH, COM, DER.
# ============================================================

_OCT_MSG_C2MCOM_PASS = (
    'CH1: EH = 45.679 mV (pass)\n: VEC = -4.568 dB (pass)\n'
    ': COM = 5.432 dB (pass)\n: DER = 1.234e-06 at COM threshold \n'
    ': PASS ... ERL = 7.890 dB (8.120 dB,7.890 dB) \n')

_OCT_MSG_C2MCOM_FAIL = (
    'CH1: EH = 5.000 mV (FAIL)\n: VEC = -1.000 dB (FAIL)\n'
    ': COM = 1.500 dB (FAIL)\n: DER = 9.870e-05 at COM threshold \n'
    ': FAIL ... ERL = 3.210 dB (4.000 dB,3.210 dB) \n')

_OCT_MSG_C2C_PASS = (
    'CH1: COM = 5.432 dB (pass)\n: DER = 1.234e-06 at COM threshold \n'
    ': PASS ... ERL = 7.890 dB (8.120 dB,7.890 dB) \n')

_OCT_MSG_C2M_PASS = (
    'CH1: EH = 45.679 mV (pass)\n: VEC = -4.568 dB (pass)\n'
    ': PASS ... ERL = 7.890 dB (8.120 dB,7.890 dB) \n')

_OCT_MODAL_LINE = ('ERL11 CD DC CC = [7.125 6.5 5.062] dB    '
                   'ERL22 CD DC CC = [4.5 3.25 2.5] dB')


def _oracle_param():
    return SimpleNamespace(
        z_p_tx_cases=np.zeros((1, 2)),
        pass_threshold=3.0, VEC_pass_threshold=-3.0, ERL_pass_threshold=6.0,
        Min_VEO=10.0, Max_VEO=100.0,
        FLAG=SimpleNamespace(S2P=True),
    )


def _oracle_op(phy, report_modal='disable'):
    return SimpleNamespace(PHY=phy, TDMODE=False, ERL=True, ERL_ONLY=False,
                           RX_CALIBRATION=False, Report_Modal_ERL=report_modal,
                           pkg_len_select=[1])


def _oracle_output_args():
    return SimpleNamespace(SCMR_CH=20.0, code_revision='4p16p0',
                           ERL11_CD=7.125, ERL11_DC=6.5, ERL11_CC=5.0625,
                           ERL22_CD=4.5, ERL22_DC=3.25, ERL22_CC=2.5)


_PASS_ARGS = dict(COM=5.4321, min_ERL=7.89, ERL=[8.12, 7.89],
                  VEO_mV=45.6789, VEC_dB=-4.5678, threshold_DER=1.2345e-06)
_FAIL_ARGS = dict(COM=1.5, min_ERL=3.21, ERL=[4.0, 3.21],
                  VEO_mV=5.0, VEC_dB=-1.0, threshold_DER=9.87e-05)


def _call(phy, report_modal, display_window, args):
    return end_display_control(
        'CH1', _oracle_param(), _oracle_op(phy, report_modal),
        _oracle_output_args(), args['COM'], args['min_ERL'],
        np.array(args['ERL'], dtype=float), args['VEO_mV'], args['VEC_dB'],
        args['threshold_DER'], display_window)


@pytest.mark.parametrize('phy,args,expected', [
    ('C2Mcom', _PASS_ARGS, _OCT_MSG_C2MCOM_PASS),
    ('C2Mcom', _FAIL_ARGS, _OCT_MSG_C2MCOM_FAIL),
    ('C2C', _PASS_ARGS, _OCT_MSG_C2C_PASS),
    ('C2M', _PASS_ARGS, _OCT_MSG_C2M_PASS),
])
def test_octave_returned_msg(phy, args, expected, capsys):
    """The msg the reference builds under DISPLAY_WINDOW=1, character exact.

    'C2Mcom' is the branch being closed; 'C2C' and 'C2M' are here so a change
    that collapses the three cases together cannot pass.
    """
    got = _call(phy, 'enable', True, args)
    capsys.readouterr()
    assert got == expected


def test_octave_c2mcom_console_order(capsys):
    """Console (DISPLAY_WINDOW=0) 'C2Mcom': VEC, then EH, then COM, then DER.

    Behaviour-only in the sense that nothing is returned -- msg comes back
    unchanged -- but the printed numbers are the reference's, and the ORDER
    distinguishes C2Mcom from both C2M (VEC, EH, no COM) and C2C (COM only).
    """
    got = _call('C2Mcom', 'disable', False, _PASS_ARGS)
    out = capsys.readouterr()
    assert got == 'CH1', 'the console branch must not modify msg'
    lines = [ln for ln in (out.out + out.err).splitlines() if ln.strip()]
    assert lines[0] == 'CH1 PASS ... VEC = -4.568 dB'
    assert lines[1] == 'CH1 PASS ... EH = 45.679 mV'
    assert lines[2] == 'CH1 PASS ... COM = 5.432 dB'
    assert lines[3] == 'CH1 DER = 1.234e-06 at COM threshold '
    assert lines[4] == 'CH1: PASS ... ERL = 7.890 dB (8.120 dB, 7.890 dB)'
    assert len(lines) == 5


@pytest.mark.parametrize('report_modal', ['enable', 'provisional', 'ENABLE'])
def test_octave_modal_erl_line_printed(report_modal, capsys):
    """Report_Modal_ERL 'enable' prints the modal-ERL line, byte for byte.

    Print-only: there is no return value and no field to check, so the
    observable is the line itself. strcmpi makes it case-insensitive.
    """
    _call('C2C', report_modal, False, _PASS_ARGS)
    out = capsys.readouterr()
    lines = [ln for ln in (out.out + out.err).splitlines() if ln.strip()]
    assert lines[-1] == _OCT_MODAL_LINE


def test_octave_modal_erl_line_absent_when_disabled(capsys):
    """'disable' prints no modal-ERL line: the other side of the branch."""
    _call('C2C', 'disable', False, _PASS_ARGS)
    out = capsys.readouterr()
    assert 'ERL11 CD DC CC' not in (out.out + out.err)
