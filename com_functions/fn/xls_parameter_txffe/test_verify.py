"""Verification tests for xls_parameter_txffe().

# ============================================================
# MATLAB GROUND TRUTH (lines 11494-11515)
# Case-insensitive search in 2D param_sheet for param_name.
# Returns (p, found=1) if found; (0, found=0) if not found.
# Raises ValueError if multiple matches.
# p = eval(p) if isinstance(p, str).
# ============================================================
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.xls_parameter_txffe.py_impl import xls_parameter_txffe


def test_not_found():
    """Missing param_name → (0, 0)."""
    sheet = [['A', 1], ['B', 2]]
    p, found = xls_parameter_txffe(sheet, 'C')
    assert found == 0
    assert p == 0


def test_found_case_insensitive():
    """Case-insensitive match returns (value, 1)."""
    sheet = [['TxFFE_Taps', 5], ['Other', 10]]
    p, found = xls_parameter_txffe(sheet, 'txffe_taps')
    assert found == 1
    assert p == 5


def test_multiple_matches_raises():
    """Multiple matches → ValueError."""
    sheet = [['ffe', 1, 'ffe', 2]]
    with pytest.raises(ValueError, match='occurrences'):
        xls_parameter_txffe(sheet, 'ffe')


def test_string_value_evaled():
    """String value is evaluated via eval()."""
    sheet = [['TapLen', '[1,2,3]']]
    p, found = xls_parameter_txffe(sheet, 'TapLen')
    assert found == 1
    assert p == [1, 2, 3]


def test_exact_name_match():
    """Exact case match still works."""
    sheet = [['Alpha', 0.5]]
    p, found = xls_parameter_txffe(sheet, 'Alpha')
    assert found == 1
    assert p == pytest.approx(0.5)
