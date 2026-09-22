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


# ---------------------------------------------------------------------------
# Values below came from running RXFFE_Illegal() out of
# octave/com_ieee8023_4p16p0_octave_compat.m under COM Octave via
# tools/octave_oracle.py, rather than from reading the MATLAB.
# ---------------------------------------------------------------------------

def _oracle_param():
    return SimpleNamespace(RxFFE_cmx=2, ffe_main_cursor_min=0.5,
                           ffe_post_tap_len=2, ffe_pre_tap_len=2,
                           ffe_tapn_max=0.2, ffe_post_tap1_max=0.3,
                           ffe_pre_tap1_max=0.3)


_LEGAL = [0.1, 0.2, 1.0, -0.25, 0.15]


@pytest.mark.parametrize('label,C,last_index,expect', [
    ('legal, last_index given',    _LEGAL,                      5,    0),
    ('legal, last_index omitted',  _LEGAL,                      None, 0),
    ('cursor below min',           [0.1, 0.2, 0.1, -0.25, 0.15], None, 1),
    ('post tap1 over max',         [0.1, 0.2, 1.0, -0.4, 0.15],  None, 1),
    ('post tapn over max',         [0.1, 0.2, 1.0, -0.25, 0.9],  None, 1),
    ('pre tap1 over max',          [0.1, 0.9, 1.0, -0.25, 0.15], None, 1),
    ('pre tapn over max',          [0.9, 0.2, 1.0, -0.25, 0.15], None, 1),
    # strict comparisons: a tap sitting exactly on a limit is LEGAL
    ('tapn exactly at max',        [0.2, 0.2, 1.0, -0.25, 0.2],  None, 0),
    ('cursor exactly at min',      [0.1, 0.2, 0.5, -0.25, 0.15], None, 0),
    # last_index keeps the Backoff region out of the check
    ('last_index hides bad tap',   [0.1, 0.2, 1.0, -0.25, 0.9],  4,    0),
])
def test_oracle_legality_table(label, C, last_index, expect):
    P = _oracle_param()
    arr = np.array(C, dtype=float)
    got = (RXFFE_Illegal(arr, P) if last_index is None
           else RXFFE_Illegal(arr, P, last_index))
    assert int(got) == expect, label


def test_oracle_column_input_matches_row():
    """COM Octave accepts a column C and returns the same verdict as a row."""
    P = _oracle_param()
    arr = np.array(_LEGAL, dtype=float)
    assert int(RXFFE_Illegal(arr.reshape(-1, 1), P)) == int(RXFFE_Illegal(arr, P))
