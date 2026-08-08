"""Verification tests for get_StepR().

# ============================================================
# MATLAB GROUND TRUTH (lines 6876-6902)
# cb_step=False: pulse = cumsum(ir)
# cb_step=True:  pulse = filter(drive_pulse, 1, ir) [shaped edge]
# TDR_response = (1+pulse)./(1-pulse) * ZT * 2
# result.ZSR, result.pulse
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.get_StepR.py_impl import get_StepR


def _param(M=4, fb=50e9, TR_TDR=10.0):
    return SimpleNamespace(samples_per_ui=M, fb=fb, TR_TDR=TR_TDR)


def test_no_cbstep_pulse_is_cumsum():
    """cb_step=False: pulse = cumsum(ir)."""
    ir = np.array([0.0, 0.1, 0.2, 0.0, 0.0])
    result = get_StepR(ir, _param(), False, 50.0)
    np.testing.assert_allclose(result.pulse, np.cumsum(ir))


def test_output_length_matches_input():
    """Output length == len(ir)."""
    ir = np.zeros(20)
    result = get_StepR(ir, _param(), False, 50.0)
    assert len(result.ZSR) == 20
    assert len(result.pulse) == 20


def test_zsr_formula():
    """ZSR = (1+pulse)/(1-pulse)*ZT*2."""
    ir = np.zeros(8)
    result = get_StepR(ir, _param(), False, 50.0)
    expected = (1 + result.pulse) / (1 - result.pulse) * 50.0 * 2
    np.testing.assert_allclose(result.ZSR, expected)


def test_returns_namespace_with_fields():
    """Result has ZSR and pulse fields."""
    ir = np.ones(8) * 0.01
    result = get_StepR(ir, _param(), False, 25.0)
    assert hasattr(result, 'ZSR')
    assert hasattr(result, 'pulse')


def test_cbstep_true_output_length():
    """cb_step=True: output still matches ir length."""
    ir = np.zeros(40)
    result = get_StepR(ir, _param(), True, 50.0)
    assert len(result.ZSR) == 40
