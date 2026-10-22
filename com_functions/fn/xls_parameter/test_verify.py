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
import numpy as np

from com_functions.fn.xls_parameter.py_impl import xls_parameter
from com_functions.fn.xls_parameter.py_impl import xls_eval, MatlabValueError
from com_functions.fn.read_ParamConfigFile.test_verify import COLON_PINS, BRACKET_PINS


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
    assert p.tolist() == [1.0, 2.0, 3.0]        # a 1x3 double, not a Python list


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
    assert xls_parameter(_SHEET, 'ZZ', True, '[1, 2, 3]').tolist() == [1.0, 2.0, 3.0]
    assert xls_parameter(_SHEET, 'ZZ', True, '[92 92; 70 70]').tolist() == [[92.0, 92.0],
                                                                             [70.0, 70.0]]


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


# ============================================================
# COM Octave (Octave 11.3.0, 2026-10-21): MATLAB vector expressions in a value
# cell. xls_parameter run verbatim from octave/com_ieee8023_4p17p0_octave_compat.m
# (identical to matlab/com_ieee8023_4p17p0.m ML 12431-12516) via
# tools/octave_oracle.py, on the sheet {'K', text} with eval_if_string true:
# ML 12452 is p=eval(p), so a cell may hold any MATLAB expression, for a
# .xlsx, a .csv (xlsread, ML 10623) and a .mat (load, ML 10621) alike.
# 802.3ck writes b_max(2..N_b) as [0.3 0.2*ones(1,4) 0.1*ones(1,6)]; the port
# refused any product or function call in a cell (finding F19): a raw
# "could not convert string to float" inside a bracket, and outside one the
# text came back unevaluated from Python's eval, which also ran any Python.
# Each entry is (text, Octave's size, Octave's value), bit for bit.
# ============================================================
VECTOR_PINS = [
    ('0.15*ones(1,3)', (1, 3), [[0.15, 0.15, 0.15]]),
    ('[0.3 0.2*ones(1,4) 0.1*ones(1,6)]', (1, 11),
     [[0.3, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]),
    ('[0.05 -0.03*ones(1,10)]', (1, 11),
     [[0.05, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03]]),
    ('-0.03*ones(1,13)', (1, 13), [[-0.03] * 13]),
    ('[0:0.02:0.14]', (1, 8), [[0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]]),
    ('1e-5*[1 2]', (1, 2), [[1e-05, 2e-05]]),
    ('zeros(1,3)', (1, 3), [[0.0, 0.0, 0.0]]),
    ('0.15 * ones(1,3)', (1, 3), [[0.15, 0.15, 0.15]]),
    ('ones(1,3)*0.15', (1, 3), [[0.15, 0.15, 0.15]]),
    ('0.15.*ones(1,3)', (1, 3), [[0.15, 0.15, 0.15]]),
    ('[0.3 0.2*ones(1,0) 0.1]', (1, 2), [[0.3, 0.1]]),
    ('[0.05 - 0.03*ones(1,2)]', (1, 2), [[0.020000000000000004, 0.020000000000000004]]),
    ('[0.05 -0.03 * ones(1,2)]', (1, 3), [[0.05, -0.03, -0.03]]),
    ('[0.25 0.15*ones(1,3) 0.05*ones(1,7)]', (1, 11),
     [[0.25, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]]),
    ('-ones(1,2)', (1, 2), [[-1.0, -1.0]]),
    ('2*-ones(1,2)', (1, 2), [[-2.0, -2.0]]),
    ('0.1*ones(1,3)+0.05', (1, 3),
     [[0.15000000000000002, 0.15000000000000002, 0.15000000000000002]]),
    ('0.15*ones(1,3)*2', (1, 3), [[0.3, 0.3, 0.3]]),
    ('0.15*ones(3)', (3, 3), [[0.15, 0.15, 0.15], [0.15, 0.15, 0.15], [0.15, 0.15, 0.15]]),
    ('ones(2,3)', (2, 3), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    ('zeros(2)', (2, 2), [[0.0, 0.0], [0.0, 0.0]]),
    ('ones', (1, 1), [[1.0]]),
    ('[0.1*ones(1,2); 0.2*ones(1,2)]', (2, 2), [[0.1, 0.1], [0.2, 0.2]]),
    ('[1 2]*2', (1, 2), [[2.0, 4.0]]),
    ('[1 2]./4', (1, 2), [[0.25, 0.5]]),
    ('[1;2]*[3 4]', (2, 2), [[3.0, 4.0], [6.0, 8.0]]),
    ('1e-5*[1 2;3 4]', (2, 2), [[1e-05, 2e-05], [3.0000000000000004e-05, 4e-05]]),
    ('2*(1+2)', (1, 1), [[6.0]]),
    ('6/4', (1, 1), [[1.5]]),
    ('ones(1,-2)', (1, 0), [[]]),
]

# The same run, texts on which COM Octave stops with an error. The port raises
# MatlabValueError, a ValueError that config readers report with the keyword.
VECTOR_ERRORS = [
    'ones(1,2.5)',                 # conversion of 2.5 to int64_t value failed
    '0.15*ones(1,3',               # syntax error
    'foo(1,3)',                    # 'foo' undefined
    '[1 2]+[1 2 3]',               # operator +: nonconformant arguments
    '[1 2; 3]',                    # vertical dimensions mismatch (1x2 vs 1x1)
    '[1 2]*[3 4]',                 # operator *: nonconformant arguments
    '__import__("os").getcwd()',   # '__import__' undefined: never Python's eval
]


@pytest.mark.parametrize('text,size,want', VECTOR_PINS, ids=[t for t, _, _ in VECTOR_PINS])
def test_octave_vector_expression_in_a_value_cell(text, size, want):
    got = np.asarray(xls_parameter([['K', text]], 'K', True), dtype=float)
    if 0 in size:
        assert got.size == 0
        return
    got = got.reshape(size) if got.ndim < 2 else got
    assert got.shape == size
    assert got.tolist() == want                      # bit for bit


@pytest.mark.parametrize('text', VECTOR_ERRORS)
def test_octave_rejected_value_cell_is_a_named_error(text):
    with pytest.raises(MatlabValueError):
        xls_parameter([['K', text]], 'K', True)


def test_value_cell_shapes_are_the_ports():
    """A row is 1-D, a scalar a float, a bracket literal always an array, as
    the config reader has always returned them."""
    assert xls_eval('2*(1+2)') == 6.0 and isinstance(xls_eval('2*(1+2)'), float)
    assert xls_eval('[46.25]').shape == (1,)
    assert xls_eval('0.15*ones(1,3)').shape == (3,)
    assert xls_eval('[1:0]').shape == (0,)
    assert xls_eval('6/4', as_matrix=True).shape == (1,)


# The colon and the whitespace rules now live here, in xls_eval; their COM
# Octave pins (findings F06, F12, F18) are kept where they were first written
# and run here as well, so a mutant of this file meets them.
@pytest.mark.parametrize('text,want', COLON_PINS + BRACKET_PINS,
                         ids=[t for t, _ in COLON_PINS + BRACKET_PINS])
def test_octave_colon_and_bracket_pins(text, want):
    got = np.asarray(xls_eval(text, as_matrix=True), dtype=float)
    want = np.asarray(want, dtype=float)
    if want.size == 0:
        assert got.size == 0
        return
    assert got.shape == want.shape
    assert got.tolist() == want.tolist()             # bit for bit
