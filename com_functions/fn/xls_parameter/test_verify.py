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


# ============================================================
# COM Octave oracle — xls_parameter run verbatim under Octave, with
# missingParameter, on the cell array {'A',1; 'B',2; 'Kappa1','1+1'}.
#
#   ('B')                        -> 2
#   ('b')                        -> 2            strcmpi, so case-insensitive
#   ('ZZ')                       -> error: The data for mandatory parameter ZZ
#                                   is missing or incorrect
#   ('ZZ', 0, 7)                 -> 7
#   ('ZZ', 0, '1+1')             -> '1+1'
#   ('ZZ', true, '1+1')          -> 2            the DEFAULT is eval'd too
#   ('ZZ', true, '[92 92; 70 70]') -> the 2x2 matrix
#   ('Kappa1', true)             -> 2
#   ('Kappa1', 0)                -> '1+1'
#   ('A', true)                  -> 1            ischar false, so no eval
#   ('B', 0, 999)                -> 2            a default found is ignored
#   on {'A',1; 'a',2}: error: 2 occurrences of "A" found — even with a default
#   on {'A',1; 2,'B'}: error: param_sheet(_,3): out of bound 2
#   on {' B ',1}: (' B ' vs 'B') no match, strcmpi does not trim
# ============================================================
_SHEET = [['A', 1], ['B', 2], ['Kappa1', '1+1']]


def test_octave_oracle_lookup_and_eval():
    assert xls_parameter(_SHEET, 'B') == 2
    assert xls_parameter(_SHEET, 'b') == 2
    assert xls_parameter(_SHEET, 'ZZ', False, 7) == 7
    assert xls_parameter(_SHEET, 'ZZ', False, '1+1') == '1+1'
    assert xls_parameter(_SHEET, 'Kappa1', True) == 2
    assert xls_parameter(_SHEET, 'Kappa1', False) == '1+1'
    assert xls_parameter(_SHEET, 'A', True) == 1
    assert xls_parameter(_SHEET, 'B', False, 999) == 2


def test_octave_oracle_string_default_is_evaluated():
    """MATLAB falls through to `if ischar(p) && eval_if_string`, so a string
    default is eval'd as well. Octave: ('ZZ', true, '1+1') -> 2."""
    assert xls_parameter(_SHEET, 'ZZ', True, '1+1') == 2
    assert xls_parameter(_SHEET, 'ZZ', True, '[1, 2, 3]') == [1, 2, 3]


def test_octave_oracle_duplicate_beats_default():
    """The duplicate check runs before the default is consulted."""
    with pytest.raises(ValueError, match='occurrences'):
        xls_parameter([['A', 1], ['a', 2]], 'A', False, 5)


def test_octave_oracle_match_in_last_column_refuses():
    """Octave: param_sheet(_,3): out of bound 2 (dimensions are 2x2)."""
    with pytest.raises(IndexError):
        xls_parameter([['A', 1], [2, 'B']], 'B')


def test_octave_oracle_strcmpi_does_not_trim():
    """' B ' does not match 'B', so the default is returned."""
    assert xls_parameter([[' B ', 1]], 'B', False, -1) == -1
