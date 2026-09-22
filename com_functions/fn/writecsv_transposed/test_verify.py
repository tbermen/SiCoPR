"""Verification tests for writecsv_transposed().

# ============================================================
# MATLAB GROUND TRUTH (lines 11384-11407)
# Writes 'Field','Value' header then one row per field.
# struct values → '[struct]'; numeric → str; str → str; empty → ''.
# ============================================================
"""
import os
import csv
import pytest
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.writecsv_transposed.py_impl import writecsv_transposed


def _read_csv(path):
    with open(path, newline='') as f:
        return list(csv.reader(f))


def test_header_row(tmp_path):
    """First row is Field, Value."""
    ns = SimpleNamespace(x=1)
    fn = str(tmp_path / 'out.csv')
    writecsv_transposed(ns, fn)
    rows = _read_csv(fn)
    assert rows[0] == ['Field', 'Value']


def test_numeric_field(tmp_path):
    """Numeric value → string representation."""
    ns = SimpleNamespace(alpha=3.14)
    fn = str(tmp_path / 'out.csv')
    writecsv_transposed(ns, fn)
    rows = _read_csv(fn)
    assert rows[1][0] == 'alpha'
    assert '3.14' in rows[1][1]


def test_string_field(tmp_path):
    """String value passes through unchanged."""
    ns = SimpleNamespace(name='hello')
    fn = str(tmp_path / 'out.csv')
    writecsv_transposed(ns, fn)
    rows = _read_csv(fn)
    assert rows[1][1] == 'hello'


def test_dict_input(tmp_path):
    """dict input works like SimpleNamespace."""
    d = {'a': 1, 'b': 'x'}
    fn = str(tmp_path / 'out.csv')
    writecsv_transposed(d, fn)
    rows = _read_csv(fn)
    assert rows[0] == ['Field', 'Value']
    assert len(rows) == 3  # header + 2 data rows


def test_row_count(tmp_path):
    """Number of data rows == number of fields."""
    ns = SimpleNamespace(a=1, b=2, c=3)
    fn = str(tmp_path / 'out.csv')
    writecsv_transposed(ns, fn)
    rows = _read_csv(fn)
    assert len(rows) == 4  # header + 3 fields


# ============================================================
# COM Octave oracle values (2026-09-22)
# PARTIAL ORACLE. The reference body ends in `writecell`, which Octave does
# not implement -- octave/com_ieee8023_4p16p0_octave_compat.m replaces the
# whole function with fprintf for exactly that reason, so the compat body is
# NOT this function and the file layout below could not be run against the
# reference. What IS the reference's, and what these pin, is the value->string
# conversion: num2str is a shared builtin and was run under Octave via
# tools/octave_oracle.py for every string below.
#
# Divergences these caught:
#   1. numeric values went through str(), so a non-integer printed all 17
#      digits where num2str prints five, and an array printed numpy's
#      '[1 2 3]' rather than num2str's '1  2  3';
#   2. ML 11399 is string(val) for a logical, which is 'true'/'false', not
#      Python's 'True'/'False';
#   3. there was no '[unsupported]' arm, so a cell array fell through to
#      str() as well.
# Array cases below are all non-negative: MATLAB widens the num2str field by
# one when an array holds a negative and Octave does not, and that difference
# is not decidable here.
# ============================================================

import numpy as np   # noqa: E402
from com_functions.fn.writecsv_transposed.py_impl import (   # noqa: E402
    _num2str, _val_to_str)


@pytest.mark.parametrize('value,text', [
    (3.141592653589793, '3.1416'), (1 / 3, '0.33333'), (42.5, '42.5'),
    (1.23456789, '1.2346'), (0.1 + 0.2, '0.3'), (123456.7, '123456.7'),
    (1e-13, '1e-13'), (1e-5, '1e-05'), (123456789, '123456789'),
    (1e15, '1000000000000000'), (1e16, '1e+16'), (1e20, '1e+20'),
    (2.0 ** 53, '9007199254740992'), (0, '0'), (-7, '-7'),
    (2.220446049250313e-16, '2.2204e-16'),
    (complex(1, 2), '1+2i'), (complex(3.141592653589793, 1), '3.1416+1i'),
    (complex(1 / 3, 1 / 7), '0.33333+0.14286i'),
])
def test_octave_num2str_scalars(value, text):
    """COM Octave num2str of each scalar."""
    assert _num2str(value) == text


@pytest.mark.parametrize('value,text', [
    ([1, 2, 3], '1  2  3'),
    ([100, 2], '100    2'),
    ([1.5, 2.5], '1.5         2.5'),
    ([3.141592653589793, 1 / 3], '3.1416     0.33333'),
    ([1e-13, 1], '1e-13           1'),
    ([0.001, 2], '0.001           2'),
    ([0.0, 0.25, 0.5, 0.75, 1.0],
     '0        0.25         0.5        0.75           1'),
    ([], ''),
])
def test_octave_num2str_arrays(value, text):
    """COM Octave num2str of each array: elements right-aligned in a fixed
    field, the result then trimmed. numpy's str() gives none of this."""
    assert _num2str(np.asarray(value, dtype=float)) == text


def test_octave_num2str_matrix_rows():
    """COM Octave: num2str([1 2;3 4]) is a 2x4 char array, '1  2' / '3  4'."""
    assert _num2str(np.array([[1, 2], [3, 4]])) == '1  2\n3  4'


@pytest.mark.parametrize('value,text', [
    (True, 'true'), (False, 'false'),
    (np.bool_(True), 'true'),
])
def test_logical_uses_matlab_spelling(value, text):
    """ML 11399: string(val) for a logical is 'true'/'false'. Octave has no
    string class, so this one comes from the reference source, not a run."""
    assert _val_to_str(value) == text


@pytest.mark.parametrize('value,text', [
    ('hello', 'hello'), ('', ''), (None, ''), (np.array([]), ''),
    (SimpleNamespace(a=1), '[struct]'), ({'a': 1}, '[struct]'),
    (['a', 'b'], '[unsupported]'), ((), ''),
])
def test_value_branches(value, text):
    """The remaining arms of the reference's if/elseif chain, in its order:
    ischar, isnumeric, islogical, isstruct, isempty, else '[unsupported]'."""
    assert _val_to_str(value) == text


def test_mixed_struct_rows(tmp_path):
    """Every converted value in one file. The two-column layout is the port's
    (writecell is not runnable here); the value strings are the reference's."""
    ns = SimpleNamespace(COM=3.1415926535, taps=np.array([0.5, 0.25, 1.0]),
                         name='run_a', flag=True, blank=np.array([]),
                         sub=SimpleNamespace(x=1), n=4)
    fn = str(tmp_path / 'out.csv')
    writecsv_transposed(ns, fn)
    assert _read_csv(fn) == [
        ['Field', 'Value'],
        ['COM', '3.1416'],
        ['taps', '0.5        0.25           1'],
        ['name', 'run_a'],
        ['flag', 'true'],
        ['blank', ''],
        ['sub', '[struct]'],
        ['n', '4'],
    ]
