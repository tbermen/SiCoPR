"""Verification tests for get_PulseR().

# ============================================================
# MATLAB GROUND TRUTH (lines 6655-6682)
# cb_step=False: pulse = filter(ones(1,M), 1, ir)  [running sum of M samples]
# cb_step=True:  pulse = filter(drive_pulse, 1, ir) [shaped edge]
# PDR = (1+pulse)./(1-pulse) * ZT * 2
# result.PDR, result.pulse
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.get_PulseR.py_impl import get_PulseR


def _param(M=4, fb=50e9, TR_TDR=10.0):
    return SimpleNamespace(samples_per_ui=M, fb=fb, TR_TDR=TR_TDR)


def test_no_cbstep_pulse_is_running_sum():
    """cb_step=False: pulse[k] = sum of ir[k-M+1..k] (rectangular filter)."""
    ir = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    M = 4
    result = get_PulseR(ir, _param(M=M), False, 1.0)
    # pulse[3] should be 1.0 (only ir[0] in window), pulse[7] should be 1.0
    assert result.pulse[3] == pytest.approx(1.0)
    assert result.pulse[7] == pytest.approx(1.0)


def test_output_length_matches_input():
    """Output length == len(ir)."""
    ir = np.zeros(20)
    result = get_PulseR(ir, _param(), False, 50.0)
    assert len(result.PDR) == 20
    assert len(result.pulse) == 20


def test_pdr_formula():
    """PDR = (1+pulse)/(1-pulse)*ZT*2."""
    ir = np.zeros(8)
    result = get_PulseR(ir, _param(), False, 50.0)
    expected_pdr = (1 + result.pulse) / (1 - result.pulse) * 50.0 * 2
    np.testing.assert_allclose(result.PDR, expected_pdr)


def test_returns_namespace_with_fields():
    """Result has PDR and pulse fields."""
    ir = np.ones(8) * 0.01
    result = get_PulseR(ir, _param(), False, 25.0)
    assert hasattr(result, 'PDR')
    assert hasattr(result, 'pulse')


def test_cbstep_true_output_length():
    """cb_step=True: output length still matches ir length."""
    ir = np.zeros(40)
    result = get_PulseR(ir, _param(), True, 50.0)
    assert len(result.PDR) == 40
