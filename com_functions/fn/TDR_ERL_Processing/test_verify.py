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
    # Numeric sentinel: the ERL summary block below runs for every testcase and
    # concatenates ERL11 into a numeric vector, exactly as MATLAB's [a b] does.
    out = SimpleNamespace()
    out.ERL11 = -99.0
    TDR_ERL_Processing(out, _OP(), 2, _chdata(), _param())
    assert out.ERL11 == -99.0


def test_erl_only_sets_file_names():
    """ERL_ONLY=True sets output_args.file_names."""
    op = _OP(erl_only=True)
    out, _, _ = TDR_ERL_Processing(SimpleNamespace(), op, 1, _chdata(), _param())
    assert hasattr(out, 'file_names')
    assert 'test.s4p' in out.file_names


# ============================================================
# COM Octave oracle values — min(), the [a b] concatenation and str2csv
# evaluated under Octave on the same operands (str2csv taken verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m).  Pinned 2026-09-22.
#
# Three divergences these pin:
#  * min(ERL11, ERL22).  MATLAB's two-argument min SKIPS NaN; Python's min()
#    only compares, so it returned NaN or not depending on argument order.
#  * [nan ERL22] / [ERL11 nan].  An EMPTY operand contributes NOTHING to a
#    MATLAB concatenation; the port built a 2-element list holding [] itself,
#    which np.asarray in the caller then could not turn into a float vector.
#  * str2csv({chdata(1).base}).  MATLAB passes a ONE-element cell; the port
#    joined every chdata entry, so a crosstalk run reported the aggressor file
#    names alongside the victim's.
#
# COM Octave:
#   min(8,5)=5  min(NaN,5)=5  min(5,NaN)=5  min(NaN,NaN)=NaN
#   [nan, 5] -> [NaN 5] numel 2      [nan, []] -> NaN numel 1
#   [5, nan] -> [5 NaN] numel 2      [[], nan] -> NaN numel 1
#   str2csv({'a.s4p'}) -> 'a.s4p' ;  sprintf('"%s"','a.s4p') -> '"a.s4p"'
#   str2csv({'a.s4p','b.s4p','c.s4p'}) -> 'a.s4p,b.s4p,c.s4p'
# ============================================================

def _chdata_with_xtalk():
    ch0 = SimpleNamespace(TDR11=_TDR(), TDR22=_TDR(), base='victim.s4p',
                          type='THRU')
    ch1 = SimpleNamespace(TDR11=_TDR(), TDR22=_TDR(), base='next1.s4p',
                          type='NEXT')
    ch2 = SimpleNamespace(TDR11=_TDR(), TDR22=_TDR(), base='fext1.s4p',
                          type='FEXT')
    return [ch0, ch1, ch2]


@pytest.mark.parametrize('e11,e22,expected', [
    (8.0, 5.0, 5.0),
    (float('nan'), 5.0, 5.0),
    (5.0, float('nan'), 5.0),
])
def test_octave_min_erl_skips_nan(e11, e22, expected):
    ch = _chdata()
    ch[0].TDR11 = SimpleNamespace(avgZport=95.0, ERL=e11, ERL_CD=0, ERL_DC=0, ERL_CC=0)
    ch[0].TDR22 = SimpleNamespace(avgZport=95.0, ERL=e22, ERL_CD=0, ERL_DC=0, ERL_CC=0)
    out, ERL, min_ERL = TDR_ERL_Processing(SimpleNamespace(), _OP(), 1, ch, _param())
    assert min_ERL == expected
    assert out.ERL == expected


def test_octave_min_erl_both_nan():
    ch = _chdata()
    nan = float('nan')
    ch[0].TDR11 = SimpleNamespace(avgZport=95.0, ERL=nan, ERL_CD=0, ERL_DC=0, ERL_CC=0)
    ch[0].TDR22 = SimpleNamespace(avgZport=95.0, ERL=nan, ERL_CD=0, ERL_DC=0, ERL_CC=0)
    out, ERL, min_ERL = TDR_ERL_Processing(SimpleNamespace(), _OP(), 1, ch, _param())
    assert np.isnan(min_ERL)


def test_octave_erl_vector_drops_empty_operand():
    """S2P leaves ERL22 empty, so MATLAB's [ERL11 nan] is 2 long but
    [nan ERL22] (TDR_W_TXPKG) collapses to a single NaN."""
    out, ERL, min_ERL = TDR_ERL_Processing(SimpleNamespace(), _OP(), 1,
                                           _chdata(), _param(s2p=True))
    arr = np.asarray(ERL, dtype=float).ravel()
    np.testing.assert_array_equal(arr, np.array([8.0, np.nan]))

    op = _OP()
    op.TDR_W_TXPKG = True
    out2, ERL2, _ = TDR_ERL_Processing(SimpleNamespace(), op, 1,
                                       _chdata(), _param(s2p=True))
    arr2 = np.asarray(ERL2, dtype=float).ravel()
    assert arr2.size == 1, "an empty operand must contribute nothing"
    assert np.isnan(arr2[0])


def test_octave_erl_vector_both_present():
    """COM Octave: [8 5] -> numel 2."""
    ch = _chdata()
    ch[0].TDR22 = SimpleNamespace(avgZport=95.0, ERL=5.0, ERL_CD=0, ERL_DC=0, ERL_CC=0)
    out, ERL, min_ERL = TDR_ERL_Processing(SimpleNamespace(), _OP(), 1, ch, _param())
    np.testing.assert_array_equal(np.asarray(ERL, dtype=float).ravel(),
                                  np.array([8.0, 5.0]))


def test_octave_file_names_is_first_base_only():
    """str2csv receives {chdata(1).base}, so only the victim is named."""
    op = _OP(erl_only=True)
    out, _, _ = TDR_ERL_Processing(SimpleNamespace(), op, 1,
                                   _chdata_with_xtalk(), _param())
    assert out.file_names == '"victim.s4p"'
    assert 'next1.s4p' not in out.file_names
    assert 'fext1.s4p' not in out.file_names
