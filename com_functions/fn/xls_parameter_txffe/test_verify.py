"""Verification tests for xls_parameter_txffe().

# ============================================================
# MATLAB GROUND TRUTH (lines 11494-11515)
# Case-insensitive search in 2D param_sheet for param_name.
# Returns (p, found=1) if found; (0, found=0) if not found.
# Raises ValueError if multiple matches.
# p = eval(p) if isinstance(p, str).
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.xls_parameter_txffe.py_impl import xls_parameter_txffe


def test_not_found():
    """Missing param_name → (0, 0)."""
    sheet = [['A', 1], ['B', 2]]
    p, found = xls_parameter_txffe(sheet, 'C')
    assert found == 0
    assert p == 0


def test_found_case_insensitive():
    """Case-insensitive match returns (value, 1)."""
    sheet = [['TxFFE_Taps', 5], ['Other', 10]]
    p, found = xls_parameter_txffe(sheet, 'txffe_taps')
    assert found == 1
    assert p == 5


def test_multiple_matches_raises():
    """Multiple matches → ValueError."""
    sheet = [['ffe', 1, 'ffe', 2]]
    with pytest.raises(ValueError, match='occurrences'):
        xls_parameter_txffe(sheet, 'ffe')


def test_string_value_evaled():
    """String value is evaluated via eval().

    COM Octave returns a 1x3 double for '[1, 2, 3]', not a Python list.
    """
    sheet = [['TapLen', '[1,2,3]']]
    p, found = xls_parameter_txffe(sheet, 'TapLen')
    assert found == 1
    np.testing.assert_array_equal(p, [1.0, 2.0, 3.0])


def test_exact_name_match():
    """Exact case match still works."""
    sheet = [['Alpha', 0.5]]
    p, found = xls_parameter_txffe(sheet, 'Alpha')
    assert found == 1
    assert p == pytest.approx(0.5)


# ============================================================
# COM Octave oracle values (2026-09-22)
# xls_parameter_txffe extracted verbatim from the reference and run under
# Octave on a one-row cell array.  The MATLAB eval() at the end reads MATLAB
# array syntax, which Python's eval() cannot parse at all -- and every TxFFE
# range in a COM config workbook is written in it.
# ============================================================

# examples/akinwale_CR_22dB_VendorX, COM_Settings sheet, the c(-1) value cell.
OCT_CM1 = [-0.34000000000000002, -0.32000000000000001, -0.30000000000000004,
           -0.28000000000000003, -0.26000000000000001, -0.24000000000000002,
           -0.22000000000000003, -0.20000000000000001, -0.18000000000000002,
           -0.16000000000000003, -0.14000000000000001, -0.12000000000000002,
           -0.10000000000000003, -0.080000000000000016, -0.059999999999999998,
           -0.040000000000000036, -0.020000000000000018, 0]


def test_oracle_real_config_txffe_range():
    """'[ -0.34:.02:0]' is what a real config holds for c(-1).

    COM Octave: 18 points, the last exactly 0.  Python's eval() raised
    SyntaxError on it, so the port could not read any real TxFFE cell.
    """
    p, found = xls_parameter_txffe([['c(-1)', '[ -0.34:.02:0]']], 'c(-1)')
    assert found == 1
    np.testing.assert_array_equal(p, OCT_CM1)


def test_oracle_ascending_range():
    """COM Octave, '[ 0:.02:0.14]' -> 8 points."""
    p, _ = xls_parameter_txffe([['c(-2)', '[ 0:.02:0.14]']], 'c(-2)')
    np.testing.assert_array_equal(
        p, [0, 0.02, 0.040000000000000001, 0.059999999999999998,
            0.080000000000000002, 0.10000000000000001, 0.12,
            0.14000000000000001])


def test_oracle_range_endpoint_is_not_snapped():
    """The count uses a tolerant floor, and the last point is base+n*inc.

    COM Octave, '[-0.2:0.05:0.05]' -> 6 points ending 0.049999999999999989,
    NOT 0.05: (0.05+0.2)/0.05 is 4.999999999999999 in binary, so a plain
    floor gives five points and np.linspace gives an exact 0.05.
    """
    p, _ = xls_parameter_txffe([['k', '[-0.2:0.05:0.05]']], 'k')
    assert len(p) == 6
    assert p[-1] == 0.049999999999999989
    assert p[-1] != 0.05


def test_oracle_descending_range():
    """COM Octave, '[0.14:-.02:0]' -> 8 points down to exactly 0."""
    p, _ = xls_parameter_txffe([['k', '[0.14:-.02:0]']], 'k')
    np.testing.assert_array_equal(
        p, [0.14000000000000001, 0.12000000000000001, 0.10000000000000001,
            0.080000000000000016, 0.060000000000000012, 0.040000000000000008,
            0.020000000000000018, 0])


def test_oracle_backwards_range_is_empty():
    """COM Octave, '[ 0.14:.02:0]' -> 1x0 empty (this is the template cell in
    the column to the right of the real value, so it does get read)."""
    p, found = xls_parameter_txffe([['k', '[ 0.14:.02:0]']], 'k')
    assert found == 1
    assert np.asarray(p).size == 0


def test_oracle_space_separated_matrix():
    """COM Octave: '[1 0 0]' -> [1 0 0]; '[1 2 3; 4 5 6]' -> 2x3.

    Python's eval() raised SyntaxError on both.
    """
    p, _ = xls_parameter_txffe([['k', '[1 0 0]']], 'k')
    np.testing.assert_array_equal(p, [1.0, 0.0, 0.0])
    p, _ = xls_parameter_txffe([['k', '[1 2 3; 4 5 6]']], 'k')
    np.testing.assert_array_equal(p, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_oracle_bare_range_without_brackets():
    """COM Octave, '0:.02:0.06' -> 4 points; the brackets are optional."""
    p, _ = xls_parameter_txffe([['k', '0:.02:0.06']], 'k')
    np.testing.assert_array_equal(
        p, [0, 0.02, 0.040000000000000001, 0.059999999999999998])


def test_oracle_plain_number_and_expression():
    """COM Octave: '0.4' -> 0.4, '2*3' -> 6, '[]' -> empty."""
    assert xls_parameter_txffe([['k', '0.4']], 'k') == (0.4, 1)
    assert xls_parameter_txffe([['k', '2*3']], 'k') == (6, 1)
    assert np.asarray(xls_parameter_txffe([['k', '[]']], 'k')[0]).size == 0


def test_oracle_junk_string_still_raises():
    """COM Octave, 'not code': "syntax error".  MATLAB refuses, so we refuse."""
    with pytest.raises(Exception):
        xls_parameter_txffe([['k', 'not code']], 'k')


def test_oracle_key_is_not_stripped():
    """strcmpi does not trim, so a sheet key with a trailing space is a miss.

    COM Octave, sheet key 'c(-1) ' looked up as 'c(-1)' -> p=0, found=0.
    """
    assert xls_parameter_txffe([['c(-1) ', '0.5']], 'c(-1)') == (0, 0)


# ============================================================
# COM Octave (Octave 11.3.0, 2026-10-06): eval of each config string, as the
# reference's xls_parameter does with eval_if_string. MATLAB's colon is
# start + k*step, counted with a tolerant floor, and the last element is set to
# the limit only when that sum overshoots it. Findings F06 (a range built by
# linspace over a rounded count, so [0:0.3:1] reached 1) and F12 (no clamp, so
# [0:0.1:0.3] ended at 0.30000000000000004).
# Octave is the proxy here: MATLAB forms the second half of a range from its
# end, so a few elements can differ from MATLAB by one ulp. Not pinned.
# ============================================================
COLON_PINS = [
    ('[0:0.3:1]', [0.0, 0.3, 0.6, 0.8999999999999999]),
    ('[-0.2:0.05:0.05]', [-0.2, -0.15000000000000002, -0.1, -0.04999999999999999, 0.0,
                          0.04999999999999999]),
    ('[0:0.1:0.3]', [0.0, 0.1, 0.2, 0.3]),
    ('[-0.3:0.1:0]', [-0.3, -0.19999999999999998, -0.09999999999999998, 0.0]),
    ('[1:0]', []),
    ('[0.14:-.02:0]', [0.14, 0.12000000000000001, 0.1, 0.08000000000000002, 0.06000000000000001,
                       0.04000000000000001, 0.020000000000000018, 0.0]),
    ('[0.1e-4:0.5e-5:0.5e-4]', [1e-05, 1.5000000000000002e-05, 2e-05, 2.5000000000000005e-05,
                                3.0000000000000004e-05, 3.5000000000000004e-05, 4e-05, 4.5e-05,
                                5e-05]),
    ('[0:0.25:1 ; 1:0.25:2]', [[0.0, 0.25, 0.5, 0.75, 1.0], [1.0, 1.25, 1.5, 1.75, 2.0]]),
    ('[ -0.34:.02:0]', [-0.34, -0.32, -0.30000000000000004, -0.28, -0.26, -0.24000000000000002,
                        -0.22000000000000003, -0.2, -0.18000000000000002, -0.16000000000000003,
                        -0.14, -0.12000000000000002, -0.10000000000000003, -0.08000000000000002,
                        -0.06, -0.040000000000000036, -0.020000000000000018, 0.0]),
    ('[-12:4:0]', [-12.0, -8.0, -4.0, 0.0]),
    ('[0:0.1:1]', [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001,
                   0.7000000000000001, 0.8, 0.9, 1.0]),
    ('[1:3]', [1.0, 2.0, 3.0]),
]


@pytest.mark.parametrize('text,want', COLON_PINS, ids=[t for t, _ in COLON_PINS])
def test_octave_colon_ranges(text, want):
    got = np.asarray(xls_parameter_txffe([['k', text]], 'k')[0], dtype=float)
    want = np.asarray(want, dtype=float)
    if want.size == 0:
        assert got.size == 0
        return
    assert got.shape == want.shape
    assert got.tolist() == want.tolist()        # bit for bit
