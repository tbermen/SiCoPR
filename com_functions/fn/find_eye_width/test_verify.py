"""Verification tests for find_eye_width().

# ============================================================
# MATLAB GROUND TRUTH (lines 5723-5785)
# Left_EW = min(L1, L0), Right_EW = min(R1, R0).
# Open eye: EW = half_UI (left) or samples_per_UI-half_UI (right).
# Closed eye: EW = 0.
# Interpolation uses vref_intersect for normal case.
# ============================================================
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.find_eye_width.py_impl import find_eye_width


def _open_eye(N=32, half_UI=16):
    """Eye contour where top is always above vref and bottom always below."""
    ec = np.zeros((N, 2))
    ec[:, 0] = 1.0   # top eye always above vref=0
    ec[:, 1] = -1.0  # bottom eye always below vref=0
    return ec


def _closed_eye(N=32, half_UI=16):
    """Eye contour where top is always below vref (closed eye)."""
    ec = np.zeros((N, 2))
    ec[:, 0] = -1.0  # top below vref=0 → closed
    ec[:, 1] = 1.0   # bottom above vref=0 → closed
    return ec


def test_open_eye_left():
    """Fully open eye returns Left_EW = half_UI."""
    half_UI = 16
    L, R = find_eye_width(_open_eye(32, half_UI), half_UI, 32, 0.0)
    assert L == half_UI


def test_open_eye_right():
    """Fully open eye returns Right_EW = samples_per_UI - half_UI."""
    half_UI = 16
    samples_per_UI = 32
    L, R = find_eye_width(_open_eye(samples_per_UI, half_UI), half_UI, samples_per_UI, 0.0)
    assert R == samples_per_UI - half_UI


def test_closed_eye_returns_zero():
    """Fully closed eye → Left_EW = 0 and Right_EW = 0."""
    L, R = find_eye_width(_closed_eye(32, 16), 16, 32, 0.0)
    assert L == 0
    assert R == 0


def test_returns_two_values():
    """Function returns a 2-tuple."""
    result = find_eye_width(_open_eye(), 16, 32, 0.0)
    assert len(result) == 2


def test_symmetric_eye():
    """Symmetric eye → Left_EW == Right_EW."""
    N = 32
    half_UI = 16
    ec = np.zeros((N, 2))
    # Linear ramp from +1 down to -1 and back (symmetric crossing)
    ec[:, 0] = np.concatenate([np.linspace(1, -1, half_UI + 1), np.linspace(-1, 1, N - half_UI - 1)])
    ec[:, 1] = -ec[:, 0]
    L, R = find_eye_width(ec, half_UI, N, 0.0)
    assert abs(L - R) <= 2  # symmetric to within 2 samples
