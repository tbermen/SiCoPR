"""Verification tests for parameter_size_adjustment().

# ============================================================
# MATLAB GROUND TRUTH (lines 8794-8845)
# Expands scalar parameters to arrays of required length.
# make_length2 fields → length 2
# make_length_GDC fields → length of ctle_gdc_values
# make_length_DCHP fields → length of g_DC_HP_values
# make_length_WCPORTZ fields → length per WC_PORTZ flag
# ============================================================
"""
import numpy as np
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.parameter_size_adjustment.py_impl import parameter_size_adjustment


def _param():
    return SimpleNamespace(
        C_pkg_board=1.0, C_diepad=1.0, L_comp=1.0, C_bump=1.0, tfx=1.0,
        C_v=1.0, C_0=1.0, C_1=1.0, pkg_Z_c=50.0, brd_Z_c=50.0, R_diepad=1.0,
        a_thru=0.5, a_fext=0.5, a_next=0.5, SNDR=10.0,
        CTLE_fp1=1e9, CTLE_fp2=5e9, CTLE_fz=0.5e9, f_HP_Z=2e9, f_HP_P=8e9,
        f_HP=1e9,
        AC_CM_RMS=0.01,
        ctle_gdc_values=np.array([-6.0, -3.0, 0.0]),
        g_DC_HP_values=np.array([0.0, 3.0]),
        z_p_tx_cases=np.ones((4, 3)),
    )


def _op():
    return SimpleNamespace(pkg_len_select=[1, 2], WC_PORTZ=False)


def test_length2_expanded():
    """Scalar make_length2 fields become length-2 arrays."""
    p = parameter_size_adjustment(_param(), _op())
    assert len(p.C_pkg_board) == 2
    assert len(p.brd_Z_c) == 2


def test_length2_already_right():
    """Already-length-2 fields are not changed."""
    param = _param()
    param.C_0 = np.array([1.0, 2.0])
    p = parameter_size_adjustment(param, _op())
    assert list(p.C_0) == [1.0, 2.0]


def test_gdc_expanded_to_ctle_length():
    """CTLE_fp1 scalar → same length as ctle_gdc_values."""
    p = parameter_size_adjustment(_param(), _op())
    assert len(p.CTLE_fp1) == 3


def test_dchp_expanded_to_g_DC_HP_length():
    """f_HP scalar → same length as g_DC_HP_values."""
    p = parameter_size_adjustment(_param(), _op())
    assert len(p.f_HP) == 2


def test_WCPORTZ_mult_no_wc():
    """WC_PORTZ=False → PORTZ_mult = ones(max(pkg_len_select))."""
    p = parameter_size_adjustment(_param(), _op())
    assert len(p.a_thru) == max(_op().pkg_len_select)


def test_returns_param():
    """Returns the modified param object."""
    param = _param()
    result = parameter_size_adjustment(param, _op())
    assert result is param
