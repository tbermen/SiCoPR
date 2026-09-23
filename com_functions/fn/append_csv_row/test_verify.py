# ============================================================
# MATLAB GROUND TRUTH for append_csv_row (L5157-5187)
#   - First call on a non-existent file writes the header row.
#   - Numeric cells formatted with %.6g; strings quoted with ".
#   - Empty row_cells writes only the header (no data line).
# ============================================================
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.append_csv_row.py_impl import append_csv_row


def test_writes_header_then_row(tmp_path):
    p = str(tmp_path / 'log.csv')
    append_csv_row(p, ['iter', 'L1_w', 'reason'], [])
    append_csv_row(p, ['iter', 'L1_w', 'reason'], [3, 1.234567, 'Evaluate Candidate'])
    lines = open(p).read().splitlines()
    assert lines[0] == 'iter,L1_w,reason'
    assert lines[1] == '3,1.23457,"Evaluate Candidate"'


def test_header_only_when_row_empty(tmp_path):
    p = str(tmp_path / 'log.csv')
    append_csv_row(p, ['a', 'b'], [])
    assert open(p).read().splitlines() == ['a,b']


def test_appends_without_rewriting_header(tmp_path):
    p = str(tmp_path / 'log.csv')
    append_csv_row(p, ['a'], [1])
    append_csv_row(p, ['a'], [2])
    lines = open(p).read().splitlines()
    # header written once (file did not exist on first call)
    assert lines == ['a', '1', '2']


def test_non_numeric_non_string_becomes_empty_quotes(tmp_path):
    p = str(tmp_path / 'log.csv')
    append_csv_row(p, ['a', 'b'], [None, 5])
    assert open(p).read().splitlines()[1] == '"",5'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))


# ---------------------------------------------------------------------------
# COM Octave oracle tests.
#
# append_csv_row returns nothing, so the probe ran it verbatim out of
# octave/com_ieee8023_4p16p0_octave_compat.m under Octave and read the file it
# wrote back with fileread().  The extracted body was first checked identical
# to matlab/com_ieee8023_4p16p0.m.  Each expectation below is the exact file
# content Octave produced, with newlines shown as separate list entries.
# ---------------------------------------------------------------------------
import warnings

import numpy as np


def _write(tmp_path, name, row, header=('h',)):
    p = str(tmp_path / name)
    append_csv_row(p, list(header), row)
    return open(p).read().splitlines()


@pytest.mark.parametrize('value,expected', [
    (float('inf'), 'Inf'),          # COM Octave {Inf}  -> Inf,  port gave inf
    (float('-inf'), '-Inf'),        # COM Octave {-Inf} -> -Inf, port gave -inf
    (float('nan'), 'NaN'),          # COM Octave {NaN}  -> NaN,  port gave nan
])
def test_oracle_non_finite_uses_matlab_capitalisation(tmp_path, value,
                                                      expected):
    """sprintf('%.6g', Inf) is 'Inf' in MATLAB and 'inf' in Python."""
    assert _write(tmp_path, 'x.csv', [value]) == ['h', expected]


@pytest.mark.parametrize('value', [True, False, np.bool_(True),
                                   np.array([True, False])])
def test_oracle_logical_is_not_numeric(tmp_path, value):
    """isnumeric(true) is FALSE in MATLAB, so a logical goes to the else arm.

    COM Octave, append_csv_row(f, {'h'}, {true}) and {false} both write "".
    bool is an int subclass in Python, so the port wrote 1 and 0.
    """
    assert _write(tmp_path, 'x.csv', [value]) == ['h', '""']


def test_oracle_empty_numeric_formats_to_an_empty_field(tmp_path):
    """COM Octave, {[]}: sprintf('%.6g', []) is '', so the field is empty --
    not the "" the port wrote for anything it did not recognise."""
    assert _write(tmp_path, 'x.csv', [np.array([])]) == ['h', '']


def test_oracle_numeric_array_reapplies_the_format(tmp_path):
    """sprintf reapplies the format to every element, with no separator.

    COM Octave: {[1 2 3]} -> 123, and {[1 2; 3 4]} -> 1324, which is
    COLUMN-major order.  The port wrote "" for both.
    """
    assert _write(tmp_path, 'a.csv', [np.array([1.0, 2.0, 3.0])]) == ['h', '123']
    assert _write(tmp_path, 'b.csv',
                  [np.array([[1.0, 2.0], [3.0, 4.0]])]) == ['h', '1324']


def test_oracle_complex_drops_the_imaginary_part(tmp_path):
    """COM Octave, {1+2i}: isnumeric is true and sprintf prints the real part
    only, giving 1.  The port wrote ""."""
    assert _write(tmp_path, 'x.csv', [1 + 2j]) == ['h', '1']


def test_oracle_numeric_header_cell_is_an_error(tmp_path):
    """strjoin needs a cell array of strings.

    COM Octave, header {1,'b'}: "error: Invalid call to strjoin."  The port
    coerced with str() and wrote 1,b.
    """
    with pytest.raises(TypeError, match='strjoin'):
        append_csv_row(str(tmp_path / 'x.csv'), [1, 'b'], [5])


def test_oracle_unopenable_path_warns_and_returns(tmp_path):
    """COM Octave, a path under a directory that does not exist: the function
    warns and returns normally.  The port raised FileNotFoundError."""
    p = str(tmp_path / 'no_such_dir' / 'x.csv')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        append_csv_row(p, ['h'], [1])
    assert any('Could not open' in str(x.message) for x in w)
    assert not os.path.exists(p)


def test_oracle_finite_formatting_matches(tmp_path):
    """The %.6g cases that already agreed, pinned so a rewrite cannot move
    them.  COM Octave values, in order below."""
    cases = [(3, '3'), (1.234567, '1.23457'), (1234567, '1.23457e+06'),
             (1e-7, '1e-07'), (-0.5, '-0.5'), (0.1 + 0.2, '0.3'),
             (1.0 / 3.0, '0.333333'), (-0.0, '-0'), (1e100, '1e+100'),
             (0, '0')]
    for i, (v, want) in enumerate(cases):
        assert _write(tmp_path, 'f%d.csv' % i, [v]) == ['h', want], v
    # strings are quoted as-is, commas and all
    assert _write(tmp_path, 's.csv', ['a,b']) == ['h', '"a,b"']
    # an empty header cell array writes a bare newline
    assert _write(tmp_path, 'e.csv', [5], header=()) == ['', '5']
