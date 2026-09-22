"""Verification tests for hrem() — shrink DFE tap window toward zero.

# ============================================================
# MATLAB GROUND TRUTH
# out = [h(1:index-1), h(idx:idx+N_bf-1)-sign(...).*min(bmaxg,|...|), h(end)]
# Segment operation: x - sign(x)*min(bmaxg,|x|)
#
# Case 1: h=[0,1,2,3,4], index=2, N_bf=3, bmaxg=1.5
#   Part1: h(1:1)=[0]
#   Middle h(2:4)=[1,2,3]: sign=[1,1,1], min(1.5,[1,2,3])=[1,1.5,1.5]
#   shrunk=[1-1, 2-1.5, 3-1.5]=[0, 0.5, 1.5]
#   Part3: h(5:5)=[4]
#   → [0, 0, 0.5, 1.5, 4]
#
# Case 2: h=[1,-3,2], index=2, N_bf=1, bmaxg=5
#   Middle h(2:2)=[-3]: sign=-1, min(5,3)=3, shrunk=-3-(-1)*3=0
#   → [1, 0, 2]
#
# Case 3: h=[1,0.5,2], index=1, N_bf=2, bmaxg=10  (bmaxg large → zeros all)
#   Middle h(1:2)=[1,0.5]: shrunk=[0,0]
#   Part3: h(3:3)=[2]
#   → [0, 0, 2]
#
# Case 4: negative values, bmaxg=0.5
#   h=[-1.0, -0.3, 0.8], index=1, N_bf=3, bmaxg=0.5
#   sign=[-1,-1,1], min(0.5,[1,0.3,0.8])=[0.5,0.3,0.5]
#   shrunk=[-1-(-1)*0.5, -0.3-(-1)*0.3, 0.8-1*0.5]=[-0.5, 0, 0.3]
#   → [-0.5, 0, 0.3]
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.hrem.py_impl import hrem


def test_nominal_middle_window():
    h = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    result = hrem(h, 2, 3, 1.5)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.5, 1.5, 4.0], atol=1e-12)


def test_single_tap_negative():
    h = np.array([1.0, -3.0, 2.0])
    result = hrem(h, 2, 1, 5.0)
    np.testing.assert_allclose(result, [1.0, 0.0, 2.0], atol=1e-12)


def test_large_bmaxg_zeros_segment():
    """bmaxg larger than all |values| → middle segment goes to zero."""
    h = np.array([1.0, 0.5, 2.0])
    result = hrem(h, 1, 2, 10.0)
    np.testing.assert_allclose(result, [0.0, 0.0, 2.0], atol=1e-12)


def test_negative_values():
    h = np.array([-1.0, -0.3, 0.8])
    result = hrem(h, 1, 3, 0.5)
    np.testing.assert_allclose(result, [-0.5, 0.0, 0.3], atol=1e-12)


def test_output_length_unchanged():
    h = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert len(hrem(h, 2, 2, 1.0)) == len(h)


def test_untouched_elements_preserved():
    """Elements outside the window are bit-for-bit identical."""
    h = np.array([9.0, 1.0, 2.0, 8.0])
    result = hrem(h, 2, 2, 0.5)
    assert result[0] == 9.0
    assert result[3] == 8.0


def test_zero_element_in_window():
    """sign(0) = 0 in NumPy; a zero element stays zero."""
    h = np.array([1.0, 0.0, 1.0])
    result = hrem(h, 2, 1, 5.0)
    assert result[1] == 0.0


# ---------------------------------------------------------------------------
# Values below came from running hrem() out of
# octave/com_ieee8023_4p16p0_octave_compat.m under COM Octave via
# tools/octave_oracle.py, rather than from reading the MATLAB.
# ---------------------------------------------------------------------------

def test_oracle_row_nominal():
    """COM Octave: hrem([0 1 2 3 4], 2, 3, 1.5) -> [0 0 0.5 1.5 4]"""
    np.testing.assert_allclose(hrem(np.array([0., 1, 2, 3, 4]), 2, 3, 1.5),
                               [0.0, 0.0, 0.5, 1.5, 4.0])


def test_oracle_negative_values_shrink_toward_zero():
    """COM Octave: hrem([-2 -0.5 0.5 2], 1, 4, 1.0) -> [-1 0 0 1]"""
    np.testing.assert_allclose(hrem(np.array([-2., -0.5, 0.5, 2.0]), 1, 4, 1.0),
                               [-1.0, 0.0, 0.0, 1.0])


def test_oracle_large_bmaxg_zeros_whole_window():
    """COM Octave: hrem([0 1 2 3 4], 1, 5, 10) -> [0 0 0 0 0]"""
    np.testing.assert_allclose(hrem(np.array([0., 1, 2, 3, 4]), 1, 5, 10.0),
                               np.zeros(5))


def test_oracle_column_input_rejected():
    """COM Octave: a column h errors 'horizontal dimensions mismatch (1x1 vs 3x1)'.

    MATLAB L7941 concatenates the three pieces horizontally, which only works
    for a row. Accepting a column would return a result MATLAB cannot produce.
    """
    with pytest.raises(ValueError):
        hrem(np.array([[0.], [1.], [2.], [3.], [4.]]), 2, 3, 1.5)


def test_oracle_window_past_end_rejected():
    """COM Octave: hrem([0 1 2 3 4], 3, 5, 1.5) errors 'h(7): out of bound 5'.

    A numpy slice would silently shorten the window and return a 5-element
    answer built from a 3-element segment -- wrong, and quiet about it.
    """
    with pytest.raises(IndexError):
        hrem(np.array([0., 1, 2, 3, 4]), 3, 5, 1.5)
