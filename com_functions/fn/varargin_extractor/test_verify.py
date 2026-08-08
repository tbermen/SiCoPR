"""Verification tests for varargin_extractor() — pop-first from varargs.

# ============================================================
# MATLAB GROUND TRUTH
# isempty(varargin) → no args:   out_var=[], varg_out={}
# one arg:                        out_var=arg1, varg_out={}
# many args:                      out_var=arg1, varg_out={arg2,...}
#
# varargin_extractor()       → (None, [])
# varargin_extractor(42)     → (42, [])
# varargin_extractor(1,2,3)  → (1, [2, 3])
# ============================================================
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.varargin_extractor.py_impl import varargin_extractor


def test_empty():
    out, rest = varargin_extractor()
    assert out is None
    assert rest == []


def test_single_arg():
    out, rest = varargin_extractor(42)
    assert out == 42
    assert rest == []


def test_multiple_args():
    out, rest = varargin_extractor(1, 2, 3)
    assert out == 1
    assert rest == [2, 3]


def test_string_args():
    out, rest = varargin_extractor('hello', 'world')
    assert out == 'hello'
    assert rest == ['world']


def test_rest_is_list():
    _, rest = varargin_extractor(1, 2, 3)
    assert isinstance(rest, list)
