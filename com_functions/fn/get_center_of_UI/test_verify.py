"""Verification tests for get_center_of_UI() — centre-of-UI index finder.

# ============================================================
# MATLAB GROUND TRUTH
# UI_window = 0:1/N:1-1/N  (N points)
# [~, half_UI] = min(abs(UI_window - 0.5))  — 1-based in MATLAB
# Python returns 0-based = MATLAB result - 1.
#
# N=1:  UI_window=[0],           argmin([0.5])=0        (MATLAB: 1)
# N=2:  UI_window=[0, 0.5],      argmin([0.5, 0])=1     (MATLAB: 2)
# N=4:  UI_window=[0,0.25,0.5,0.75], argmin=2           (MATLAB: 3)
# N=5:  UI_window=[0,0.2,0.4,0.6,0.8], distances=[0.5,0.3,0.1,0.1,0.3]
#       first min at index 2 (tie broken to lower index)  (MATLAB: 3)
# N=8:  UI_window[4]=0.5 exactly, argmin=4               (MATLAB: 5)
# N=32: UI_window[16]=0.5 exactly, argmin=16             (MATLAB: 17)
# ============================================================
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.get_center_of_UI.py_impl import get_center_of_UI


def test_n1():
    assert get_center_of_UI(1) == 0


def test_n2():
    assert get_center_of_UI(2) == 1


def test_n4_exact_centre():
    """Even N=4: UI_window[2] = 0.5 exactly."""
    assert get_center_of_UI(4) == 2


def test_n5_tie_broken_to_lower():
    """Odd N=5: tie at indices 2 (0.4) and 3 (0.6); argmin returns first."""
    assert get_center_of_UI(5) == 2


def test_n8_exact_centre():
    assert get_center_of_UI(8) == 4


def test_n32_exact_centre():
    assert get_center_of_UI(32) == 16


def test_returns_int():
    assert isinstance(get_center_of_UI(4), int)


def test_result_in_valid_range():
    """Result must be a valid 0-based index into a samples_per_UI-length array."""
    for n in [1, 2, 3, 4, 5, 7, 8, 16, 32]:
        h = get_center_of_UI(n)
        assert 0 <= h < n
