"""Verification tests for OptFom_Build_TXFFE().

# ============================================================
# MATLAB GROUND TRUTH (lines 2734-2816)
# Builds txffe_matrix from param.tx_ffe_cm/cp fields.
# txffe_sweep_indices are 1-based MATLAB convention.
# txffe_cursor_vector = 1 - sum(abs(TXFFE_grid), axis=1).
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.OptFom_Build_TXFFE.py_impl import OptFom_Build_TXFFE


def _param_simple():
    """1 precursor, 1 postcursor."""
    return SimpleNamespace(
        tx_ffe_cm1_values=[-0.1, 0.0, 0.1],
        tx_ffe_cp1_values=[-0.05, 0.0, 0.05],
    )


def _param_no_taps():
    """No tx_ffe fields → empty grid."""
    return SimpleNamespace()


def _param_single_tap():
    """Only one postcursor tap with single value."""
    return SimpleNamespace(tx_ffe_cp1_values=[0.0])


def test_returns_five_outputs():
    """OptFom_Build_TXFFE returns 5 values."""
    result = OptFom_Build_TXFFE(_param_simple())
    assert len(result) == 5


def test_cursor_position():
    """cur = num_pre + 1 = 2 for 1 precursor."""
    _, cur, _, _, _ = OptFom_Build_TXFFE(_param_simple())
    assert cur == 2


def test_txffe_matrix_columns():
    """txffe_matrix has num_taps+1 columns (precursors + cursor + postcursors)."""
    txffe_matrix, cur, _, _, _ = OptFom_Build_TXFFE(_param_simple())
    # 1 precursor + cursor + 1 postcursor = 3 columns
    assert txffe_matrix.shape[1] == 3


def test_txffe_matrix_rows():
    """txffe_matrix has 3*3=9 rows (3 pre values * 3 post values)."""
    txffe_matrix, _, _, _, _ = OptFom_Build_TXFFE(_param_simple())
    assert txffe_matrix.shape[0] == 9


def test_cursor_vector_shape():
    """txffe_cursor_vector shape matches number of grid rows."""
    txffe_matrix, _, _, _, txffe_cursor_vector = OptFom_Build_TXFFE(_param_simple())
    assert txffe_cursor_vector.shape[0] == txffe_matrix.shape[0]


def test_sweep_indices_1based():
    """txffe_sweep_indices are 1-based (min value >= 1)."""
    _, _, txffe_sweep_indices, _, _ = OptFom_Build_TXFFE(_param_simple())
    if len(txffe_sweep_indices) > 0:
        assert int(np.min(txffe_sweep_indices)) >= 1


def test_no_taps_returns_scalar_grid():
    """No tx_ffe fields → TXFFE_grid is scalar 0, cur=1."""
    txffe_matrix, cur, sweep_idx, full_tx, cursor_vec = OptFom_Build_TXFFE(_param_no_taps())
    assert cur == 1
    assert len(sweep_idx) == 0


# ---------------------------------------------------------------------------
# Tx FFE vector length / cursor placement.
#
# MATLAB (com_ieee8023_4p15p0.m L2789-2816) drops leading all-zero fixed taps
# and decrements `cur` to match, but the `auto_count_trigger` latch means that
# once ANY tap is non-trivial every later tap is retained INCLUDING its zeros —
# so a real FFE keeps a weight per tap, e.g. [0 1 0] with the cursor at 2.
#
# Pinned because the shipped dj configs declare c(-4)..c(-1)=0 and c(1)=0, which
# collapses to the scalar [1.0]; that is MATLAB's own behaviour (its no-crosstalk
# reference reports TXLE_taps = 1) and must not be "fixed" into a padded vector.
def test_all_zero_fixed_taps_collapse_to_cursor_only():
    """Shipped dj config: 4 pre + 1 post, all single-valued 0 -> [1.0], cur=1."""
    param = SimpleNamespace(
        tx_ffe_cm1_values=0, tx_ffe_cm2_values=0,
        tx_ffe_cm3_values=0, tx_ffe_cm4_values=0,
        tx_ffe_cp1_values=0)
    m, cur, _, _, _ = OptFom_Build_TXFFE(param)
    assert m.shape == (1, 1)
    assert m[0, 0] == 1.0
    assert cur == 1


def test_zero_taps_are_retained_once_a_tap_is_non_trivial():
    """A swept pre-tap latches auto_count_trigger: zeros after it are KEPT."""
    param = SimpleNamespace(
        tx_ffe_cm1_values=[0, -0.05], tx_ffe_cm2_values=0,
        tx_ffe_cp1_values=0)
    m, cur, _, _, _ = OptFom_Build_TXFFE(param)
    assert m.shape == (2, 3)          # [pre1, cursor, post1] - the zero post tap kept
    assert cur == 2                   # cursor sits at index 2 (1-based)
    np.testing.assert_allclose(m[0], [0.0, 1.0, 0.0])
    # cursor weight is 1 - sum|other taps|
    np.testing.assert_allclose(m[1], [-0.05, 0.95, 0.0])


# ============================================================
# COM Octave oracle — OptFom_Build_TXFFE and Full_Grid_Matrix extracted
# verbatim from octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m) and run by tools/octave_oracle.py.
#
# DIVERGENCE 1, and it bites the plainest real case: MATLAB's sort is STABLE,
# so `sort(txffe_lengths(idx),'descend')` leaves equal-length taps in
# ascending tap order.  The port wrote argsort(ascending, stable)[::-1],
# which REVERSES every tie.  One swept precursor and one swept postcursor of
# equal length gave txffe_sweep_indices [2 1] where the reference gives
# [1 2]; three equal taps gave [3 2 1] for [1 2 3]; lengths [2 4 2 4] gave
# [4 2 3 1] for [2 4 1 3].  These indices drive the order OptFom_Local_Search
# walks the taps in, so the search visited them back to front.
#
# DIVERGENCE 2: MATLAB's regexp is NOT anchored.  A field that merely
# contains tx_ffe_cm<n>_values -- xtx_ffe_cm1_values_extra, or
# tx_ffe_cm1_values_old -- is counted as a precursor, and the sprintf'd
# lookup that follows then fails: COM Octave says "structure has no member
# 'tx_ffe_cm1_values'".  The anchored ^...$ silently skipped such a field and
# answered with the wrong tap count.
#
# DIVERGENCE 3: a tap value that is a matrix.  Full_Grid_Matrix takes
# num_cases from cellfun('length',in) (the longest dimension, 3 for a 2x3)
# but lays out all six elements of in{k}(:), so num_repeats is 0.5 and repmat
# refuses it -- COM Octave: "conversion of 0.5 to int64_t value failed".
# ravel() had flattened it to six values and answered.  A column vector and a
# 1x1 are both fine and agree with the reference.
#
# STILL DIVERGENT, and NOT in this directory: an EMPTY tap value.  The
# reference answers with zero-row results (txffe_matrix 0x3, cur 2,
# txffe_sweep_indices [2], FULL_tx_index_vector 0x2, txffe_cursor_vector
# 0x1); Python raises ZeroDivisionError at
# com_functions/fn/Full_Grid_Matrix/py_impl.py line 34, where MATLAB's
# num_cases/length(C) is 0/0 = NaN.  That belongs to Full_Grid_Matrix.
# ============================================================

def _p(**fields):
    return SimpleNamespace(**{k: np.asarray(v, dtype=float)
                              for k, v in fields.items()})


def test_oracle_sweep_index_ties_keep_ascending_tap_order():
    """MATLAB's 'descend' sort is stable; ties must NOT come back reversed."""
    _, _, sw, _, _ = OptFom_Build_TXFFE(
        _p(tx_ffe_cm1_values=[-0.1, 0.0, 0.1],
           tx_ffe_cp1_values=[-0.05, 0.0, 0.05]))
    np.testing.assert_array_equal(sw, [1, 2])

    _, _, sw, _, _ = OptFom_Build_TXFFE(
        _p(tx_ffe_cm1_values=[0, 0.1], tx_ffe_cm2_values=[0, 0.2],
           tx_ffe_cm3_values=[0, 0.3], tx_ffe_cp1_values=[0.0]))
    np.testing.assert_array_equal(sw, [1, 2, 3])

    # tap order is [cm2 cm1 cp1 cp2], so the lengths are [2 4 2 4]
    _, _, sw, _, _ = OptFom_Build_TXFFE(
        _p(tx_ffe_cm2_values=[0, 0.1], tx_ffe_cm1_values=[0, .1, .2, .3],
           tx_ffe_cp1_values=[0, 0.2], tx_ffe_cp2_values=[0, .1, .2, .3]))
    np.testing.assert_array_equal(sw, [2, 4, 1, 3])

    # a strictly descending run is unaffected either way
    _, _, sw, _, _ = OptFom_Build_TXFFE(
        _p(tx_ffe_cm1_values=[0, .1, .2, .3, .4], tx_ffe_cp1_values=[0, .1, .2]))
    np.testing.assert_array_equal(sw, [1, 2])


def test_oracle_one_precursor_one_postcursor_full_grid():
    """The whole 9-row answer, pinned against COM Octave."""
    m, cur, sw, full, cv = OptFom_Build_TXFFE(
        _p(tx_ffe_cm1_values=[-0.1, 0.0, 0.1],
           tx_ffe_cp1_values=[-0.05, 0.0, 0.05]))
    assert int(cur) == 2
    np.testing.assert_array_equal(sw, [1, 2])
    np.testing.assert_allclose(m, [
        [-0.1, 0.85, -0.05], [-0.1, 0.90, 0.00], [-0.1, 0.85, 0.05],
        [0.0, 0.95, -0.05], [0.0, 1.00, 0.00], [0.0, 0.95, 0.05],
        [0.1, 0.85, -0.05], [0.1, 0.90, 0.00], [0.1, 0.85, 0.05]],
        rtol=1e-14, atol=0)
    np.testing.assert_allclose(cv.ravel(),
                               [0.85, 0.90, 0.85, 0.95, 1.0, 0.95,
                                0.85, 0.90, 0.85], rtol=1e-14, atol=0)
    np.testing.assert_array_equal(np.asarray(full), [
        [1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3],
        [3, 1], [3, 2], [3, 3]])


def test_oracle_two_precursors_one_fixed_zero():
    """cm2 = 0 is dropped and the cursor moves back to 2."""
    m, cur, sw, _, _ = OptFom_Build_TXFFE(
        _p(tx_ffe_cm1_values=[-0.1, 0.0, 0.1], tx_ffe_cm2_values=0.0,
           tx_ffe_cp1_values=[0.0, -0.05]))
    assert int(cur) == 2
    np.testing.assert_array_equal(sw, [2, 3])
    np.testing.assert_allclose(m, [
        [-0.1, 0.90, 0.00], [-0.1, 0.85, -0.05],
        [0.0, 1.00, 0.00], [0.0, 0.95, -0.05],
        [0.1, 0.90, 0.00], [0.1, 0.85, -0.05]], rtol=1e-14, atol=0)


def test_unanchored_regexp_counts_a_field_that_merely_contains_the_pattern():
    """The reference errors on the missing sprintf'd field; so must the port."""
    for extra in ('xtx_ffe_cm1_values_extra', 'tx_ffe_cm1_values_old'):
        param = _p(tx_ffe_cp1_values=[0.0, 0.1])
        setattr(param, extra, np.array([0.0, 0.1]))
        with pytest.raises(AttributeError, match='tx_ffe_cm1_values'):
            OptFom_Build_TXFFE(param)


def test_matrix_tap_value_is_refused():
    """Full_Grid_Matrix cannot lay out a non-vector sweep variable."""
    with pytest.raises(ValueError, match='must be a vector'):
        OptFom_Build_TXFFE(_p(tx_ffe_cm1_values=[[0, .1, .2], [.3, .4, .5]],
                              tx_ffe_cp1_values=0.0))


def test_column_and_one_by_one_tap_values_are_accepted():
    """length() is the longest dimension, so a column is just a vector."""
    col = OptFom_Build_TXFFE(_p(tx_ffe_cm1_values=[[-0.1], [0.0], [0.1]],
                                tx_ffe_cp1_values=[0.0]))
    row = OptFom_Build_TXFFE(_p(tx_ffe_cm1_values=[-0.1, 0.0, 0.1],
                                tx_ffe_cp1_values=[0.0]))
    np.testing.assert_array_equal(col[0], row[0])
    assert int(col[1]) == int(row[1])
    one = OptFom_Build_TXFFE(_p(tx_ffe_cm1_values=[[0.0]],
                                tx_ffe_cp1_values=[0.0, 0.1]))
    assert one[0].shape == (2, 2)


def test_missing_or_non_contiguous_tap_numbering_is_refused():
    """cm10 alone, or cm1+cm3, makes the sprintf'd cm1/cm2 lookup fail."""
    with pytest.raises(AttributeError):
        OptFom_Build_TXFFE(_p(tx_ffe_cm10_values=[0.0, 0.1],
                              tx_ffe_cp1_values=0.0))
    with pytest.raises(AttributeError):
        OptFom_Build_TXFFE(_p(tx_ffe_cm1_values=[0.0, 0.1],
                              tx_ffe_cm3_values=[0.0, 0.2],
                              tx_ffe_cp1_values=0.0))
