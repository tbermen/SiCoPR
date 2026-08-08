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
