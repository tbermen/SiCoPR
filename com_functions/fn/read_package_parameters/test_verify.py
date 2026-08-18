"""Verification tests for read_package_parameters().

# ============================================================
# MATLAB GROUND TRUTH (lines 10525-10593)
# Reads package parameters from a parameter block.
# mele=1/2/4: sets param.flex accordingly.
# mele=2: expands z_p_* to 4 columns with zeros.
# mele other: raises ValueError.
#
# z_p ORIENTATION — the fixtures below are in SPREADSHEET orientation:
#   rows = package segments, columns = package cases.
# MATLAB transposes all four z_p keywords on read (L10678/10689/10695/10701 each
# end in .'), so the stored array is (ncases, mele) and the engine indexes
# [case, :]. A fixture written the other way round silently passes the shape
# check whenever the matrix is square, which previously hid a real defect: the
# RX/NEXT/FEXT package was built from a row of the matrix rather than a case
# column, over-stating package length (e.g. 111 mm instead of 13.8 mm) and
# adding roughly 15 dB of spurious loss.
# ============================================================
"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.read_package_parameters.py_impl import read_package_parameters
from types import SimpleNamespace


def _param_block_2col():
    """2-column flex package parameter block."""
    return {
        'C_p': np.array([0.1, 0.1]),
        'R_d': np.array([50.0, 50.0]),
        'A_v': np.array([0.5, 0.5]),
        'A_fe': np.array([0.3, 0.3]),
        'A_ne': np.array([0.2, 0.2]),
        # 2 segments (rows) x 1 case (col) -> stored (1 case, 2 elements)
        'z_p (TX)': np.array([[5.0], [10.0]]),
        'z_p (NEXT)': np.array([[3.0], [6.0]]),
        'z_p (FEXT)': np.array([[4.0], [8.0]]),
        'z_p (RX)': np.array([[2.0], [4.0]]),
        'package_tl_gamma0_a1_a2': np.array([0.0, 1.734e-3, 1.455e-4]),
        'package_tl_tau': np.array([6.141e-3]),
        'package_Z_c': np.array([[78.2, 78.2]]),
    }


def _param_block_4col():
    return {
        'C_p': np.array([0.1, 0.1, 0.1, 0.1]),
        'R_d': np.array([50.0, 50.0, 50.0, 50.0]),
        'A_v': np.array([0.5, 0.5, 0.5, 0.5]),
        'A_fe': np.array([0.3, 0.3, 0.3, 0.3]),
        'A_ne': np.array([0.2, 0.2, 0.2, 0.2]),
        # 4 segments (rows) x 1 case (col) -> stored (1 case, 4 elements)
        'z_p (TX)': np.array([[5.0], [10.0], [15.0], [20.0]]),
        'z_p (NEXT)': np.array([[3.0], [6.0], [9.0], [12.0]]),
        'z_p (FEXT)': np.array([[4.0], [8.0], [12.0], [16.0]]),
        'z_p (RX)': np.array([[2.0], [4.0], [6.0], [8.0]]),
        'package_tl_gamma0_a1_a2': np.array([0.0, 1.734e-3, 1.455e-4]),
        'package_tl_tau': np.array([6.141e-3]),
        'package_Z_c': np.array([[78.2, 78.2, 78.2, 78.2]]),
    }


def test_flex2_set():
    """mele=2 sets flex=2."""
    p = read_package_parameters(_param_block_2col())
    assert p.flex == 2


def test_flex4_set():
    """mele=4 sets flex=4."""
    p = read_package_parameters(_param_block_4col())
    assert p.flex == 4


def test_2col_expands_to_4():
    """mele=2 expands z_p_tx_cases to 4 columns."""
    p = read_package_parameters(_param_block_2col())
    assert p.z_p_tx_cases.shape[1] == 4


def test_invalid_mele_raises():
    """mele=3 raises ValueError."""
    block = _param_block_2col()
    three = np.array([[1.0], [2.0], [3.0]])   # 3 segments -> mele=3, invalid
    block['z_p (TX)'] = three
    block['z_p (NEXT)'] = three
    block['z_p (FEXT)'] = three
    block['z_p (RX)'] = three
    block['package_Z_c'] = np.array([[78.2, 78.2, 78.2]])
    with pytest.raises(ValueError):
        read_package_parameters(block)


def test_returns_namespace():
    """Returns a SimpleNamespace."""
    p = read_package_parameters(_param_block_2col())
    assert hasattr(p, 'z_p_tx_cases')
    assert hasattr(p, 'R_diepad')


def test_appends_to_existing():
    """Can append to an existing param_struct."""
    existing = SimpleNamespace(other_field=42)
    p = read_package_parameters(_param_block_2col(), existing)
    assert p.flex == 2
    assert p.other_field == 42


# ---------------------------------------------------------------------------
# Adversarial orientation fixtures (added 2026-08-18).
#
# The tests above check flex, column count, and the invalid-mele raise, but
# none of them checks WHERE THE VALUES LAND. Engine defect #1 of the 208-case
# MATLAB correlation was exactly that: MATLAB transposes all four z_p keywords
# (L10678/10689/10695/10701 each end in .') while Python transposed only TX, so
# the RX package was built from a matrix ROW instead of a case COLUMN -- 111 mm
# of package instead of 13.8 mm, roughly 15 dB of spurious loss. Every test
# above still passed.
#
# The fixture is deliberately NON-SQUARE (2 segments x 3 cases) and gives every
# keyword distinct values, so a dropped or extra transpose changes both the
# shape and the contents. A square fixture cannot catch this.
# ---------------------------------------------------------------------------
def _param_block_3cases():
    """2 segments (rows) x 3 package cases (cols), spreadsheet orientation."""
    return {
        'C_p': np.array([0.1, 0.1]),
        'R_d': np.array([50.0, 50.0]),
        'A_v': np.array([0.5, 0.5]),
        'A_fe': np.array([0.3, 0.3]),
        'A_ne': np.array([0.2, 0.2]),
        'z_p (TX)':   np.array([[5.0, 6.0, 7.0], [10.0, 11.0, 12.0]]),
        'z_p (NEXT)': np.array([[3.0, 3.1, 3.2], [6.0, 6.1, 6.2]]),
        'z_p (FEXT)': np.array([[4.0, 4.1, 4.2], [8.0, 8.1, 8.2]]),
        'z_p (RX)':   np.array([[2.0, 2.1, 2.2], [4.0, 4.1, 4.2]]),
        'package_Z_c': np.array([[78.2, 78.2]]),
    }


@pytest.mark.parametrize('attr,expected', [
    ('z_p_tx_cases',   [[5.0, 10.0], [6.0, 11.0], [7.0, 12.0]]),
    ('z_p_next_cases', [[3.0, 6.0], [3.1, 6.1], [3.2, 6.2]]),
    ('z_p_fext_cases', [[4.0, 8.0], [4.1, 8.1], [4.2, 8.2]]),
    ('z_p_rx_cases',   [[2.0, 4.0], [2.1, 4.1], [2.2, 4.2]]),
])
def test_all_four_z_p_keywords_are_transposed(attr, expected):
    """Each z_p keyword must be stored (ncases, mele), i.e. row = one case.

    Dropping the transpose on any one of these reintroduces engine defect #1.
    """
    p = read_package_parameters(_param_block_3cases())
    got = np.asarray(getattr(p, attr))
    assert got.shape == (3, 4), (
        '%s should be (ncases=3, 4) after transpose and 2->4 expansion, got %s '
        '-- a missing transpose gives (2, 4)' % (attr, got.shape))
    np.testing.assert_allclose(got[:, :2], np.array(expected))
    np.testing.assert_allclose(got[:, 2:], 0.0)


def test_z_p_case_rows_are_independent():
    """Row i must carry case i only -- catches a transpose that happens to fit.

    With 2 segments and 3 cases the matrix is non-square, so an un-transposed
    read cannot produce three distinct rows at all.
    """
    p = read_package_parameters(_param_block_3cases())
    tx = np.asarray(p.z_p_tx_cases)
    assert len({tuple(r) for r in tx}) == 3, (
        'the three package cases collapsed to %d distinct rows'
        % len({tuple(r) for r in tx}))
    # Case 0 must not contain case 1's or case 2's segment lengths.
    assert 6.0 not in tx[0] and 7.0 not in tx[0]
