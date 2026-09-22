"""Verification tests for Full_Grid_Matrix().

# ============================================================
# MATLAB GROUND TRUTH (lines 2119-2179)
# Full_Grid_Matrix({[1 2], [100 200]}) →
#   [[1, 100], [1, 200], [2, 100], [2, 200]]
# ============================================================
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.Full_Grid_Matrix.py_impl import Full_Grid_Matrix


def test_two_column_numeric():
    """2×2 case matches MATLAB example."""
    result = Full_Grid_Matrix([[1, 2], [100, 200]])
    assert result == [[1, 100], [1, 200], [2, 100], [2, 200]]


def test_three_column():
    """3 columns → num_cases = product of lengths."""
    result = Full_Grid_Matrix([[1, 2], [3, 4], [5, 6]])
    assert len(result) == 8
    assert len(result[0]) == 3


def test_single_column():
    """Single column → num_cases == length of that column."""
    result = Full_Grid_Matrix([[10, 20, 30]])
    assert result == [[10], [20], [30]]


def test_mixed_string_numeric():
    """Mixed cell/numeric: first column strings, second numeric."""
    result = Full_Grid_Matrix([['A', 'B'], [1, 2]])
    assert len(result) == 4
    assert result[0] == ['A', 1]
    assert result[1] == ['A', 2]
    assert result[2] == ['B', 1]
    assert result[3] == ['B', 2]


def test_single_element_columns():
    """Columns with one element each → one row."""
    result = Full_Grid_Matrix([[7], [8], [9]])
    assert result == [[7, 8, 9]]


def test_non_list_raises():
    """Non-list input raises ValueError."""
    import pytest
    with pytest.raises((ValueError, TypeError)):
        Full_Grid_Matrix('not a list')


# ---------------------------------------------------------------------------
# Against COM Octave: the cartesian product, and the ORDER it comes out in.
# The order is the part a property test cannot see, and the part a downstream
# index depends on.
# ---------------------------------------------------------------------------

_OCT_ROWS = [[0.0, -1.0], [0.0, 0.0], [0.0, 1.0],
             [0.5, -1.0], [0.5, 0.0], [0.5, 1.0],
             [1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]


def test_matches_com_octave_including_row_order():
    out = np.asarray(Full_Grid_Matrix([np.array([0.0, 0.5, 1.0]),
                                       np.array([-1.0, 0.0, 1.0])]))
    assert out.shape == (9, 2), 'shape %s, COM Octave gives (9, 2)' % (out.shape,)
    assert out.tolist() == _OCT_ROWS, (
        'row order differs from COM Octave.\n got %r\n want %r'
        % (out.tolist(), _OCT_ROWS))


def test_first_variable_varies_slowest():
    """The ordering above in one sentence, so a reordering is caught by intent
    and not only by the literal."""
    out = np.asarray(Full_Grid_Matrix([np.array([0.0, 0.5, 1.0]),
                                       np.array([-1.0, 0.0, 1.0])]))
    assert list(out[:3, 0]) == [0.0, 0.0, 0.0]
    assert list(out[:3, 1]) == [-1.0, 0.0, 1.0]
