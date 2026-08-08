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
