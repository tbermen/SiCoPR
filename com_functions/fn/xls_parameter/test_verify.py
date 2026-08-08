"""Verification tests for xls_parameter().

# ============================================================
# MATLAB GROUND TRUTH (lines 11408-11493)
# Case-insensitive search in 2D param_sheet.
# Not found + no default → KeyError (MATLAB: missingParameter error)
# Not found + default → return default
# Multiple matches → ValueError
# Single match → return cell[col+1]
# eval_if_string=True + string value → eval(p)
# ============================================================
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.xls_parameter.py_impl import xls_parameter


def test_not_found_no_default_raises():
    """Not found with no default → KeyError."""
    sheet = [['A', 1], ['B', 2]]
    with pytest.raises(KeyError):
        xls_parameter(sheet, 'C')


def test_not_found_with_default_returns_default():
    """Not found with default → returns default."""
    sheet = [['A', 1]]
    p = xls_parameter(sheet, 'X', default_value=99)
    assert p == 99


def test_found_case_insensitive():
    """Case-insensitive match returns value."""
    sheet = [['TxFFE_Taps', 5]]
    p = xls_parameter(sheet, 'txffe_taps')
    assert p == 5


def test_multiple_matches_raises():
    """Multiple matches → ValueError."""
    sheet = [['ffe', 1, 'ffe', 2]]
    with pytest.raises(ValueError, match='occurrences'):
        xls_parameter(sheet, 'ffe')


def test_eval_if_string():
    """eval_if_string=True with string → eval'd value."""
    sheet = [['TapLen', '[1, 2, 3]']]
    p = xls_parameter(sheet, 'TapLen', eval_if_string=True)
    assert p == [1, 2, 3]


def test_no_eval_by_default():
    """eval_if_string defaults to False → string returned as-is."""
    sheet = [['Name', 'hello']]
    p = xls_parameter(sheet, 'Name')
    assert p == 'hello'
