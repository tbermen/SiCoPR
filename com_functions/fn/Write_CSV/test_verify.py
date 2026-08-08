"""Verification tests for Write_CSV().

# ============================================================
# MATLAB GROUND TRUTH (lines 4742-4767)
# header: comma-joined field names
# data:   comma-joined values (num2str for scalars, mat2str for arrays, 'struct' for structs)
# ============================================================
"""
import os
import pytest
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.Write_CSV.py_impl import Write_CSV


def _read(path):
    with open(path) as f:
        return f.read().strip().splitlines()


def test_header_contains_field_names(tmp_path):
    """First line is comma-joined field names."""
    ns = SimpleNamespace(alpha=1.0, beta=2.0)
    fn = str(tmp_path / 'out.csv')
    Write_CSV(ns, fn)
    lines = _read(fn)
    assert 'alpha' in lines[0]
    assert 'beta' in lines[0]


def test_scalar_value_in_data_row(tmp_path):
    """Scalar numeric value appears in data row."""
    ns = SimpleNamespace(x=42.5)
    fn = str(tmp_path / 'out.csv')
    Write_CSV(ns, fn)
    lines = _read(fn)
    assert '42.5' in lines[1]


def test_struct_value_becomes_struct(tmp_path):
    """Struct-like value → 'struct' in data row."""
    ns = SimpleNamespace(inner=SimpleNamespace(a=1))
    fn = str(tmp_path / 'out.csv')
    Write_CSV(ns, fn)
    lines = _read(fn)
    assert 'struct' in lines[1]


def test_dict_input(tmp_path):
    """dict input produces same output as SimpleNamespace."""
    d = {'a': 1}
    fn = str(tmp_path / 'out.csv')
    Write_CSV(d, fn)
    lines = _read(fn)
    assert lines[0] == 'a'
    assert '1' in lines[1]


def test_string_value_passthrough(tmp_path):
    """String value passes through as-is."""
    ns = SimpleNamespace(label='test_run')
    fn = str(tmp_path / 'out.csv')
    Write_CSV(ns, fn)
    lines = _read(fn)
    assert 'test_run' in lines[1]
