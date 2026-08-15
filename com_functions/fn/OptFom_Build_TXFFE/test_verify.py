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
