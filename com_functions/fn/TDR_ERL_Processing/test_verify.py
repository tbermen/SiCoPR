"""Verification tests for TDR_ERL_Processing().

# ============================================================
# MATLAB GROUND TRUTH (lines 4498-4592)
# Fills TDR/ERL data in output_args.
# package_testcase_i==1: fills Z11est, Z22est, ERL11, ERL22.
# package_testcase_i!=1: skips TDR/ERL fill (no-op for first block).
# OP.ERL=False: ERL fields set to [].
# ERL_ONLY: sets file_names and Z_t.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.TDR_ERL_Processing.py_impl import TDR_ERL_Processing


def _TDR():
    return SimpleNamespace(avgZport=95.0, ERL=8.0, ERL_CD=7.0, ERL_DC=6.0, ERL_CC=5.0)


def _chdata(s2p=False):
    ch = SimpleNamespace(
        TDR11=_TDR(),
        TDR22=_TDR(),
        base='test.s4p',
        type='THRU',
    )
    return [ch]


def _param(s2p=False):
    return SimpleNamespace(
        FLAG=SimpleNamespace(S2P=s2p),
        tfx=np.array([0.0, 5.0]),
        Z_t=50.0,
    )


def _OP(erl=True, tdr=True, erl_only=False):
    return SimpleNamespace(
        TDR=tdr, ERL=erl, ERL_ONLY=erl_only,
        TDR_W_TXPKG=False,
        BREAD_CRUMBS=False,
        AUTO_TFX=False,
        Report_Modal_ERL='disable',
    )


def test_returns_three_values():
    """Returns (output_args, ERL, min_ERL) tuple."""
    result = TDR_ERL_Processing(SimpleNamespace(), _OP(), 1, _chdata(), _param())
    assert len(result) == 3


def test_erl11_set_on_testcase1():
    """ERL11 is populated when package_testcase_i==1 and ERL=True."""
    out, ERL, min_ERL = TDR_ERL_Processing(SimpleNamespace(), _OP(), 1, _chdata(), _param())
    assert out.ERL11 == pytest.approx(8.0)


def test_erl_empty_when_disabled():
    """ERL fields are [] when OP.ERL=False."""
    out, ERL, min_ERL = TDR_ERL_Processing(SimpleNamespace(), _OP(erl=False), 1, _chdata(), _param())
    assert out.ERL == [] or out.ERL is None or len(out.ERL) == 0


def test_z11est_set_when_tdr():
    """Z11est is set when OP.TDR=True."""
    out, _, _ = TDR_ERL_Processing(SimpleNamespace(), _OP(), 1, _chdata(), _param())
    assert out.Z11est == pytest.approx(95.0)


def test_testcase2_no_override():
    """package_testcase_i==2 does not set ERL11 (skips first block)."""
    out = SimpleNamespace()
    out.ERL11 = 'previous'
    TDR_ERL_Processing(out, _OP(), 2, _chdata(), _param())
    assert out.ERL11 == 'previous'


def test_erl_only_sets_file_names():
    """ERL_ONLY=True sets output_args.file_names."""
    op = _OP(erl_only=True)
    out, _, _ = TDR_ERL_Processing(SimpleNamespace(), op, 1, _chdata(), _param())
    assert hasattr(out, 'file_names')
    assert 'test.s4p' in out.file_names
