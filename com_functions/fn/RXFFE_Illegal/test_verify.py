"""Verification tests for RXFFE_Illegal().

# ============================================================
# MATLAB GROUND TRUTH
# Ccur_i = param.RxFFE_cmx + 1  [1-based]; Python cursor = param.RxFFE_cmx [0-based]
# Check cursor:  C[cursor] < ffe_main_cursor_min → 1
# Check post-tap1: |C[cursor+1]| > ffe_post_tap1_max → 1
# Check post-tapN: any(|C[cursor+2:last]| > ffe_tapn_max) → 1
# Check pre-tap1:  |C[cursor-1]| > ffe_pre_tap1_max → 1
# Check pre-tapN:  any(|C[:cursor-1]| > ffe_tapn_max) → 1
# All pass → 0
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.RXFFE_Illegal.py_impl import RXFFE_Illegal


def make_param(cmx=2, main_min=0.5, post1_max=0.4, pre1_max=0.4, tapn_max=0.3,
               post_len=2, pre_len=2):
    return SimpleNamespace(
        RxFFE_cmx=cmx,
        ffe_main_cursor_min=main_min,
        ffe_post_tap1_max=post1_max,
        ffe_pre_tap1_max=pre1_max,
        ffe_tapn_max=tapn_max,
        ffe_post_tap_len=post_len,
        ffe_pre_tap_len=pre_len,
    )


def test_legal_taps():
    """All taps within limits → 0."""
    C = np.array([0.0, 0.1, 0.8, 0.2, 0.0])
    assert RXFFE_Illegal(C, make_param(cmx=2)) == 0


def test_cursor_too_small():
    """Cursor below min → 1."""
    C = np.array([0.0, 0.1, 0.3, 0.2, 0.0])
    assert RXFFE_Illegal(C, make_param(cmx=2, main_min=0.5)) == 1


def test_post_tap1_too_large():
    """|post-cursor tap 1| > ffe_post_tap1_max → 1."""
    C = np.array([0.0, 0.1, 0.8, 0.5, 0.0])
    assert RXFFE_Illegal(C, make_param(cmx=2, post1_max=0.4)) == 1


def test_post_tapN_too_large():
    """Higher-order post-cursor > ffe_tapn_max → 1."""
    C = np.array([0.0, 0.1, 0.8, 0.2, 0.5])
    assert RXFFE_Illegal(C, make_param(cmx=2, tapn_max=0.3, post1_max=0.4)) == 1


def test_pre_tap1_too_large():
    """|pre-cursor tap 1| > ffe_pre_tap1_max → 1."""
    C = np.array([0.0, 0.5, 0.8, 0.2, 0.0])
    assert RXFFE_Illegal(C, make_param(cmx=2, pre1_max=0.4)) == 1


def test_no_pre_or_post_taps_always_legal_if_cursor_ok():
    """With ffe_pre_tap_len=ffe_post_tap_len=0, only cursor is checked."""
    C = np.array([0.9, 0.8, 0.7])
    p = make_param(cmx=1, main_min=0.5, post_len=0, pre_len=0)
    assert RXFFE_Illegal(C, p) == 0
