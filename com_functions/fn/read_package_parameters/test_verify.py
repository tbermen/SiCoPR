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
        # package_Z_c is (segments, [Tx Rx]) in the sheet; MATLAB transposes
        # it, so it must have mele rows.  COM Octave rejects a 1-row
        # package_Z_c against a 2-element package: 'tx rx pairs must have
        # thesame number element entries as TX, NEXT, FEXT, Rx'.
        'package_Z_c': np.array([[78.2, 78.2], [78.2, 78.2]]),
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
        'package_Z_c': np.array([[78.2, 78.2]] * 4),   # 4 segments
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
    block['package_Z_c'] = np.array([[78.2, 78.2]] * 3)
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
        'package_Z_c': np.array([[78.2, 78.2], [78.2, 78.2]]),
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


# ---------------------------------------------------------------------------
# COM Octave oracle tests.
#
# Values below come from running read_package_parameters verbatim out of
# octave/com_ieee8023_4p16p0_octave_compat.m (with xls_parameter and
# missingParameter) through tools/octave_oracle.py, on a `parameter` cell
# array holding the same key/value pairs each fixture below carries.  The
# extracted body was first checked identical to matlab/com_ieee8023_4p16p0.m.
#
# Shape convention: MATLAB row vectors come back (1,N) and the port stores
# them 1-D, as it does everywhere else.  The 2-D fields -- the four
# z_p_*_cases and pkg_Z_c -- are compared shape-for-shape, because
# make_full_pkg indexes pkg_Z_c(1,:) for TX and (2,:) for RX.
# ---------------------------------------------------------------------------

_OCT_COMMON = {
    'C_p': np.array([0.018, 0.018]),
    'R_d': np.array([50.0, 50.0]),
    'A_v': 0.4, 'A_fe': 0.4, 'A_ne': 0.6,
    'package_tl_gamma0_a1_a2': np.array([0.0, 1.734e-3, 1.455e-4]),
    'package_tl_tau': 6.141e-3,
}


def _oct_block(z_p, package_Z_c=None):
    b = dict(_OCT_COMMON)
    for k in ('z_p (TX)', 'z_p (NEXT)', 'z_p (FEXT)', 'z_p (RX)'):
        b[k] = np.asarray(z_p, float)
    if package_Z_c is not None:
        b['package_Z_c'] = np.asarray(package_Z_c, float)
    return b


_OCT_Z4 = [[12.0, 1.8, 0.0, 0.0]] * 4
_OCT_ZC4 = [[92.0, 92.0], [70.0, 70.0], [80.0, 80.0], [100.0, 100.0]]


def test_oracle_pkg_z_c_is_transposed():
    """package_Z_c is read with .' like the four z_p keywords.

    COM Octave on the shipped [92 92 ; 70 70; 80 80; 100 100] with a
    four-element package: pkg_Z_c is (2,4) = [92 70 80 100 ; 92 70 80 100].
    The port kept the sheet's (4,2) orientation, which both put the values in
    the wrong place for make_full_pkg's pkg_Z_c(1,:)/(2,:) reads and made the
    mele check compare the wrong axis -- it raised 'tx rx pairs must have the
    same number element entries' on the shipped configuration.
    """
    p = read_package_parameters(_oct_block(_OCT_Z4, _OCT_ZC4))
    zc = np.asarray(p.pkg_Z_c)
    assert zc.shape == (2, 4)
    np.testing.assert_array_equal(zc, [[92.0, 70.0, 80.0, 100.0],
                                       [92.0, 70.0, 80.0, 100.0]])
    assert int(np.asarray(p.flex).ravel()[0]) == 4
    np.testing.assert_array_equal(
        np.asarray(p.z_p_tx_cases),
        [[12.0, 12.0, 12.0, 12.0], [1.8, 1.8, 1.8, 1.8],
         [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])


def test_oracle_flex_two_pkg_z_c_expansion():
    """mele=2: [pkg_Z_c' ; [100 100 ; 100 100]]' -- (2,2) becomes (2,4).

    COM Octave with z_p = [12 1.8 ; 30 1.8] and package_Z_c = [92 92 ; 70 70]:
    pkg_Z_c = [92 70 100 100 ; 92 70 100 100].  Stacking on the untransposed
    array gave (4,2) = [92 92 ; 70 70 ; 100 100 ; 100 100].
    """
    p = read_package_parameters(
        _oct_block([[12.0, 1.8], [30.0, 1.8]], [[92.0, 92.0], [70.0, 70.0]]))
    zc = np.asarray(p.pkg_Z_c)
    assert zc.shape == (2, 4)
    np.testing.assert_array_equal(zc, [[92.0, 70.0, 100.0, 100.0],
                                       [92.0, 70.0, 100.0, 100.0]])
    np.testing.assert_array_equal(np.asarray(p.z_p_tx_cases),
                                  [[12.0, 30.0, 0.0, 0.0],
                                   [1.8, 1.8, 0.0, 0.0]])
    assert int(np.asarray(p.flex).ravel()[0]) == 2


def test_oracle_flex_one_and_the_scalar_default():
    """mele=1, and the package_Z_c default is the scalar 78.2, not a pair.

    COM Octave with z_p = [12 30] and package_Z_c = [92 70]: pkg_Z_c is (2,1)
    = [92 ; 70].  With package_Z_c left out entirely it is (1,1) = 78.2, so
    the default must be the scalar; the port defaulted to [[78.2 78.2]].
    """
    p = read_package_parameters(_oct_block([12.0, 30.0], [92.0, 70.0]))
    np.testing.assert_array_equal(np.asarray(p.pkg_Z_c), [[92.0], [70.0]])
    np.testing.assert_array_equal(np.asarray(p.z_p_tx_cases),
                                  [[12.0], [30.0]])
    assert int(np.asarray(p.flex).ravel()[0]) == 1
    d = read_package_parameters(_oct_block([12.0, 30.0]))
    np.testing.assert_array_equal(np.asarray(d.pkg_Z_c), [[78.2]])


@pytest.mark.parametrize('key', ['C_p', 'R_d', 'A_v', 'A_fe', 'A_ne',
                                 'z_p (TX)', 'z_p (NEXT)', 'z_p (FEXT)',
                                 'z_p (RX)'])
def test_oracle_mandatory_parameters_have_no_default(key):
    """xls_parameter is called with no fourth argument for these nine keys.

    COM Octave with the key removed from the parameter cell: "error: The data
    for mandatory parameter <key> is missing or incorrect".  The port supplied
    an invented default for every one of them and answered.
    """
    block = _oct_block(_OCT_Z4, _OCT_ZC4)
    del block[key]
    with pytest.raises(KeyError, match='mandatory parameter'):
        read_package_parameters(block)


@pytest.mark.parametrize('key,default', [
    ('package_tl_gamma0_a1_a2', None),
    ('package_tl_tau', 6.141e-3),
    ('package_Z_c', None),
])
def test_oracle_optional_parameters_have_defaults(key, default):
    """These three do have a fourth argument, so leaving them out is legal."""
    block = _oct_block(_OCT_Z4, _OCT_ZC4)
    block.pop(key, None)
    if key == 'package_Z_c':
        # the scalar default makes mele1=1, which a 4-element package rejects
        with pytest.raises(ValueError, match='tx rx pairs'):
            read_package_parameters(block)
        return
    p = read_package_parameters(block)
    if key == 'package_tl_gamma0_a1_a2':
        np.testing.assert_array_equal(np.asarray(p.pkg_gamma0_a1_a2).ravel(),
                                      [0.0, 1.734e-3, 1.455e-4])
    else:
        assert float(np.asarray(p.pkg_tau).ravel()[0]) == default


@pytest.mark.parametrize('key', ['z_p (NEXT)', 'z_p (FEXT)', 'z_p (RX)'])
def test_oracle_disagreeing_case_counts(key):
    """COM Octave: "error: All TX, NEXT, FEXT, Rx cases must agree"."""
    block = _oct_block(_OCT_Z4, _OCT_ZC4)
    block[key] = np.array([[12.0, 1.8, 0.0, 0.0]] * 2)
    with pytest.raises(ValueError, match='cases must agree'):
        read_package_parameters(block)


def test_oracle_three_element_package_is_a_syntax_error():
    """COM Octave with z_p = [12 1.8 ; 0 0 ; 1 1] (mele=3):
    "error: config file syntax error"."""
    with pytest.raises(ValueError, match='config file syntax error'):
        read_package_parameters(
            _oct_block([[12.0, 1.8], [0.0, 0.0], [1.0, 1.0]],
                       [[92.0, 92.0], [70.0, 70.0], [80.0, 80.0]]))


def test_oracle_pkg_z_c_width_must_match_mele():
    """COM Octave, a 2-segment package_Z_c against a 4-element package:
    "error: tx rx pairs must have thesame number element entries as TX, NEXT,
    FEXT, Rx"."""
    with pytest.raises(ValueError, match='tx rx pairs'):
        read_package_parameters(
            _oct_block(_OCT_Z4, [[92.0, 92.0], [70.0, 70.0]]))
    # and a 3-segment package_Z_c against a 2-element package
    with pytest.raises(ValueError, match='tx rx pairs'):
        read_package_parameters(
            _oct_block([[12.0, 1.8], [30.0, 1.8]],
                       [[92.0, 92.0], [70.0, 70.0], [80.0, 80.0]]))
