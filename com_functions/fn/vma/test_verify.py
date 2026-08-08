"""Verification tests for vma().

# ============================================================
# MATLAB GROUND TRUTH (lines 11337-11365)
# Uses PRBS13Q sequence to find patterns: 7 consecutive 3s and 6 consecutive 0s.
# PR is upsampled, convolved with PRBS signal, then P_3/P_0 measured.
# VMA = P_3 - P_0
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.vma.py_impl import vma


def test_returns_namespace_with_fields():
    """Returns SimpleNamespace with P_3, P_0, VMA."""
    M = 4
    N = 50000
    PR = np.zeros(N)
    PR[0] = 1.0  # impulse at 0
    result = vma(PR, M)
    assert hasattr(result, 'P_3')
    assert hasattr(result, 'P_0')
    assert hasattr(result, 'VMA')


def test_vma_equals_p3_minus_p0():
    """VMA = P_3 - P_0."""
    M = 4
    N = 50000
    PR = np.zeros(N)
    PR[0] = 1.0
    result = vma(PR, M)
    assert result.VMA == pytest.approx(result.P_3 - result.P_0)


def test_output_is_finite():
    """P_3, P_0, VMA are all finite."""
    M = 4
    N = 50000
    PR = np.zeros(N)
    PR[0] = 1.0
    result = vma(PR, M)
    assert np.isfinite(result.VMA)
    assert np.isfinite(result.P_3)
    assert np.isfinite(result.P_0)


def test_p3_greater_p0_for_unity_pulse():
    """For a unit impulse response, P_3 > P_0 (higher PAM level → higher mean)."""
    M = 4
    N = 50000
    PR = np.zeros(N)
    PR[0] = 1.0
    result = vma(PR, M)
    assert result.P_3 > result.P_0
