"""Verification tests for Bread_Crumb_Chdata_Reduction().

# ============================================================
# MATLAB GROUND TRUTH (lines 1038-1082)
# First line of fields_file: '#reduce' (remove listed fields) or
#   '#include' (keep only listed fields, remove everything else).
# Blank lines are ignored.
# Unknown first-line token raises error.
# ============================================================
"""
import os
import pytest
import tempfile
from types import SimpleNamespace
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.Bread_Crumb_Chdata_Reduction.py_impl import Bread_Crumb_Chdata_Reduction


def _chdata(fields):
    return [SimpleNamespace(**fields)]


def _write_file(lines, tmp_dir):
    p = os.path.join(tmp_dir, 'fields.txt')
    with open(p, 'w') as f:
        f.write('\n'.join(lines))
    return p


def test_reduce_removes_listed_fields(tmp_path):
    """#reduce removes exactly the listed fields."""
    cd = _chdata({'a': 1, 'b': 2, 'c': 3})
    fpath = _write_file(['#reduce', 'a', 'c'], str(tmp_path))
    result = Bread_Crumb_Chdata_Reduction(cd, fpath)
    assert not hasattr(result[0], 'a')
    assert not hasattr(result[0], 'c')
    assert hasattr(result[0], 'b')


def test_include_keeps_only_listed_fields(tmp_path):
    """#include keeps only the listed fields."""
    cd = _chdata({'a': 1, 'b': 2, 'c': 3})
    fpath = _write_file(['#include', 'b'], str(tmp_path))
    result = Bread_Crumb_Chdata_Reduction(cd, fpath)
    assert hasattr(result[0], 'b')
    assert not hasattr(result[0], 'a')
    assert not hasattr(result[0], 'c')


def test_bad_first_line_raises(tmp_path):
    """Unknown first-line token raises ValueError."""
    cd = _chdata({'a': 1})
    fpath = _write_file(['#unknown', 'a'], str(tmp_path))
    with pytest.raises(ValueError):
        Bread_Crumb_Chdata_Reduction(cd, fpath)


def test_reduce_missing_field_is_noop(tmp_path):
    """#reduce ignores fields not present in chdata."""
    cd = _chdata({'a': 1, 'b': 2})
    fpath = _write_file(['#reduce', 'nonexistent'], str(tmp_path))
    result = Bread_Crumb_Chdata_Reduction(cd, fpath)
    assert hasattr(result[0], 'a')
    assert hasattr(result[0], 'b')


def test_blank_lines_ignored(tmp_path):
    """Blank lines in the file are skipped."""
    cd = _chdata({'a': 1, 'b': 2})
    fpath = _write_file(['#reduce', '', 'a', ''], str(tmp_path))
    result = Bread_Crumb_Chdata_Reduction(cd, fpath)
    assert not hasattr(result[0], 'a')
    assert hasattr(result[0], 'b')


# ---------------------------------------------------------------------------
# Against COM Octave, both directives.
#
# This decides which chdata fields survive into output_args, so the answer is
# the exact set of field names -- and the assertions above checked structure
# rather than which names came back. It is the last leaf function in the
# translation inventory without a value check.
# ---------------------------------------------------------------------------

_OCT_REDUCE = ['base', 'sdd21', 'type']
_OCT_INCLUDE = ['base', 'sdd21']


def _oct_chdata():
    return [SimpleNamespace(base='chA', sdd21=[1, 2, 3], sdd21_raw=[9, 9],
                            sdd11_raw=[8, 8], type='THRU'),
            SimpleNamespace(base='chB', sdd21=[4, 5, 6], sdd21_raw=[7, 7],
                            sdd11_raw=[6, 6], type='NEXT')]


def _oct_fields_file(tmp_path, first_line, names):
    p = tmp_path / ('%s.txt' % first_line.lstrip('#'))
    p.write_text('\n'.join([first_line] + list(names)) + '\n')
    return str(p)


def test_reduce_removes_exactly_the_listed_fields(tmp_path):
    p = _oct_fields_file(tmp_path, '#reduce', ['sdd21_raw', 'sdd11_raw'])
    out = Bread_Crumb_Chdata_Reduction(_oct_chdata(), p)
    for i, ch in enumerate(out):
        assert sorted(vars(ch)) == _OCT_REDUCE, (
            'channel %d kept %r; COM Octave keeps %r'
            % (i, sorted(vars(ch)), _OCT_REDUCE))


def test_include_keeps_exactly_the_listed_fields(tmp_path):
    p = _oct_fields_file(tmp_path, '#include', ['base', 'sdd21'])
    out = Bread_Crumb_Chdata_Reduction(_oct_chdata(), p)
    for i, ch in enumerate(out):
        assert sorted(vars(ch)) == _OCT_INCLUDE, (
            'channel %d kept %r; COM Octave keeps %r'
            % (i, sorted(vars(ch)), _OCT_INCLUDE))


def test_the_two_directives_disagree(tmp_path):
    """#reduce and #include over the same name list must not give the same
    answer, or the tests above are pinning one behaviour twice."""
    names = ['base', 'sdd21']
    a = Bread_Crumb_Chdata_Reduction(
        _oct_chdata(), _oct_fields_file(tmp_path, '#reduce', names))
    b = Bread_Crumb_Chdata_Reduction(
        _oct_chdata(), _oct_fields_file(tmp_path, '#include', names))
    assert sorted(vars(a[0])) != sorted(vars(b[0]))


def test_values_of_the_surviving_fields_are_untouched(tmp_path):
    p = _oct_fields_file(tmp_path, '#include', ['base', 'sdd21'])
    out = Bread_Crumb_Chdata_Reduction(_oct_chdata(), p)
    assert out[0].base == 'chA' and list(out[0].sdd21) == [1, 2, 3]
    assert out[1].base == 'chB' and list(out[1].sdd21) == [4, 5, 6]
