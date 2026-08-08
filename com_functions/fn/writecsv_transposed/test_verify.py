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
