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


# ============================================================
# COM Octave oracle values (2026-09-22)
# Write_CSV run verbatim from octave/com_ieee8023_4p16p0_octave_compat.m via
# tools/octave_oracle.py, with str2csv as its subfunction, writing a real file
# whose BYTES are pinned below. Four divergences these caught:
#   1. num2str is five significant digits for a non-integer, not str(): the
#      reference writes 3.1416 where the port wrote 3.1415926535;
#   2. mat2str is 15 significant digits, ';' between rows, and never wraps;
#      np.array2string is 8 digits and breaks the line at 75 characters,
#      which would split a CSV field in two;
#   3. MATLAB's fopen 'w' is binary, so the line ending is LF; Python's text
#      mode wrote CRLF on Windows;
#   4. a struct with no fields is an error in str2csv, not two blank lines.
# ============================================================

import numpy as np   # noqa: E402


def _bytes(tmp_path, output_args, name='out.csv'):
    fn = str(tmp_path / name)
    Write_CSV(output_args, fn)
    with open(fn, 'rb') as f:
        return f.read()


def test_octave_mixed_fields_bytes(tmp_path):
    """COM Octave, the whole file for a struct of every field kind."""
    ns = SimpleNamespace(COM=3.1415926535, VEC=0.123456789, label='run_a',
                         taps=np.array([0.5, -0.25, 1 / 3]), n=4,
                         empty=np.array([]), sub=SimpleNamespace(x=1),
                         big=1e-13)
    assert _bytes(tmp_path, ns) == (
        b'COM,VEC,label,taps,n,empty,sub,big\n'
        b'3.1416,0.12346,run_a,"[0.5 -0.25 0.333333333333333]",4,,struct,1e-13\n')


def test_octave_line_ending_is_lf(tmp_path):
    """COM Octave: b'only\n1.5\n' -- LF, not CRLF, on Windows too."""
    assert _bytes(tmp_path, SimpleNamespace(only=1.5)) == b'only\n1.5\n'


def test_octave_logical_is_one_and_zero(tmp_path):
    """COM Octave: num2str(true) is '1', not 'True'."""
    assert _bytes(tmp_path, SimpleNamespace(flag=True, f2=False)) == \
        b'flag,f2\n1,0\n'


def test_octave_row_and_column_vectors(tmp_path):
    """COM Octave: mat2str separates columns with a space and rows with ';'."""
    ns = SimpleNamespace(r=np.array([1, 2, 3]), c=np.array([[1], [2], [3]]))
    assert _bytes(tmp_path, ns) == b'r,c\n"[1 2 3]","[1;2;3]"\n'


def test_octave_matrix_field(tmp_path):
    """COM Octave: mat2str([1 2;3 4]) -> [1 2;3 4]."""
    ns = SimpleNamespace(m=np.array([[1, 2], [3, 4]]))
    assert _bytes(tmp_path, ns) == b'm\n"[1 2;3 4]"\n'


def test_octave_nan_and_inf(tmp_path):
    """COM Octave: NaN, Inf and -Inf keep MATLAB's spelling."""
    ns = SimpleNamespace(a=float('nan'), b=float('inf'), c=float('-inf'))
    assert _bytes(tmp_path, ns) == b'a,b,c\nNaN,Inf,-Inf\n'


def test_octave_long_vector_is_one_line(tmp_path):
    """COM Octave: 30 elements on ONE line at 15 significant digits. This is
    the np.array2string line-wrap and precision defect in one case."""
    ns = SimpleNamespace(v=np.linspace(0, 1, 30))
    out = _bytes(tmp_path, ns)
    assert out == (
        b'v\n"[0 0.0344827586206897 0.0689655172413793 0.103448275862069 '
        b'0.137931034482759 0.172413793103448 0.206896551724138 '
        b'0.241379310344828 0.275862068965517 0.310344827586207 '
        b'0.344827586206897 0.379310344827586 0.413793103448276 '
        b'0.448275862068966 0.482758620689655 0.517241379310345 '
        b'0.551724137931034 0.586206896551724 0.620689655172414 '
        b'0.655172413793103 0.689655172413793 0.724137931034483 '
        b'0.758620689655172 0.793103448275862 0.827586206896552 '
        b'0.862068965517241 0.896551724137931 0.931034482758621 '
        b'0.96551724137931 1]"\n')
    assert out.count(b'\n') == 2


def test_octave_char_with_comma_is_not_quoted(tmp_path):
    """COM Octave: a char field containing a comma is written raw, so the row
    gains a column. Faithful, and worth pinning so nobody 'fixes' it."""
    assert _bytes(tmp_path, SimpleNamespace(s='a,b', k=1)) == b's,k\na,b,1\n'


def test_octave_empty_struct_errors(tmp_path):
    """COM Octave, Write_CSV(struct()):
        error: cell_tmp(_,0): subscripts must be either integers
               1 to (2^63)-1 or logicals
        str2csv at line 6 column 1 / Write_CSV at line 21 column 1"""
    with pytest.raises(IndexError):
        Write_CSV(SimpleNamespace(), str(tmp_path / 'nf.csv'))


@pytest.mark.parametrize('value,text', [
    (3.141592653589793, '3.1416'), (1 / 3, '0.33333'),
    (1.23456789, '1.2346'), (-0.000123456, '-0.00012346'),
    (0.1 + 0.2, '0.3'), (42.5, '42.5'), (123456.7, '123456.7'),
    (1e10 + 0.5, '10000000000.5'), (1e-13, '1e-13'), (1e-5, '1e-05'),
    (2.220446049250313e-16, '2.2204e-16'), (123456789, '123456789'),
    (1e15, '1000000000000000'), (1e16, '1e+16'), (1e20, '1e+20'),
    (2.0 ** 53, '9007199254740992'),
    (1.7976931348623157e308, '1.797693134862316e+308'),
    (-0.0, '-0'), (0, '0'), (100000, '100000'), (-7, '-7'),
    (complex(1, 2), '1+2i'), (complex(3.141592653589793, 1), '3.1416+1i'),
    (complex(1 / 3, 1 / 7), '0.33333+0.14286i'),
])
def test_octave_num2str_scalar_formats(tmp_path, value, text):
    """COM Octave: num2str of each value, straight out of the reference."""
    assert _bytes(tmp_path, SimpleNamespace(x=value)) == \
        ('x\n%s\n' % text).encode()
