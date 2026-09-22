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


# ============================================================
# Probed against the executed reference (COM Octave oracle,
# tools/octave_oracle.py) for wrong types, missing arguments and unexpected
# counts. NO behavioural divergence was found:
#   0 args -> out_var is 0x0 [],      varg_out is a 0x0 cell
#   1 arg  -> out_var is that arg,    varg_out is a 1x0 cell
#   3 args -> out_var is the first,   varg_out is a 1x2 cell of the rest
# The only difference is the empty sentinel's TYPE: MATLAB returns [], this
# port returns None. Both mean "absent" and no caller distinguishes them, so
# it is left as documented rather than changed.
# ============================================================

def test_octave_arity_contract():
    """COM Octave shapes: () -> (0x0, 0x0 cell); (5,) -> (5, 1x0 cell);
    (5,'abc',[1 2 3]) -> (5, 1x2 cell)."""
    out_var, varg_out = varargin_extractor()
    assert out_var is None and varg_out == []
    out_var, varg_out = varargin_extractor(5)
    assert out_var == 5 and varg_out == []
    out_var, varg_out = varargin_extractor(5, 'abc', [1, 2, 3])
    assert out_var == 5 and varg_out == ['abc', [1, 2, 3]]


def test_octave_first_argument_empty_is_still_consumed():
    """COM Octave: varargin_extractor([], 7) -> out_var 0x0, varg_out {7}.
    An empty first argument is popped like any other, not skipped."""
    out_var, varg_out = varargin_extractor([], 7)
    assert out_var == []
    assert varg_out == [7]


def test_octave_extra_outputs_are_not_invented():
    """The reference declares exactly two outputs, so the port returns a
    2-tuple for every arity -- never a bare value."""
    for args in ((), (1,), (1, 2), (1, 2, 3, 4)):
        assert len(varargin_extractor(*args)) == 2
