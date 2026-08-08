"""Verification tests for OptFom_Compute_DFE().

# ============================================================
# MATLAB GROUND TRUTH (lines 3219-3304)
# Returns (THIS, param) with dfetaps, excess_dfe_cursors, tail_RSS.
# cursor_i is 0-based. dfecursors at cursor_i+M:M:cursor_i+ndfe*M+1.
# Floating DFE path off; do_C2M=False tested here.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.OptFom_Compute_DFE.py_impl import OptFom_Compute_DFE


def _param(M=8, ndfe=3):
    bmax = np.ones(ndfe) * 0.5
    bmin = -bmax
    return SimpleNamespace(
        samples_per_ui=M,
        ndfe=ndfe,
        dfe_delta=0,
        Floating_DFE=False,
        use_bmax=bmax,
        use_bmin=bmin,
        bmax=bmax,
        bmin=bmin,
        N_tail_start=0,
        B_float_RSS_MAX=0.5,
    )


def _THIS(cursor_i=40):
    return SimpleNamespace(cursor_i=cursor_i)


def _sbr(cursor_i=40, M=8, ndfe=3, N=200):
    sbr = np.zeros(N)
    sbr[cursor_i] = 1.0
    for n in range(1, ndfe + 1):
        sbr[cursor_i + n * M] = 0.3 * (0.5 ** n)
    return sbr


def test_returns_two_outputs():
    """OptFom_Compute_DFE returns (THIS, param)."""
    sbr = _sbr()
    THIS, param = OptFom_Compute_DFE(sbr, _THIS(), _param(), False, 0)
    assert hasattr(THIS, 'dfetaps')


def test_dfetaps_length():
    """dfetaps has length == ndfe."""
    sbr = _sbr()
    THIS, param = OptFom_Compute_DFE(sbr, _THIS(), _param(ndfe=3), False, 0)
    assert len(THIS.dfetaps) == 3


def test_dfetaps_within_bmax():
    """dfetaps values are clipped to bmax."""
    sbr = _sbr()
    THIS, param = OptFom_Compute_DFE(sbr, _THIS(), _param(), False, 0)
    assert np.all(np.asarray(THIS.dfetaps) <= 0.5 + 1e-12)
    assert np.all(np.asarray(THIS.dfetaps) >= -0.5 - 1e-12)


def test_excess_dfe_cursors_shape():
    """excess_dfe_cursors has same length as ndfe."""
    sbr = _sbr()
    THIS, param = OptFom_Compute_DFE(sbr, _THIS(), _param(ndfe=3), False, 0)
    assert len(THIS.excess_dfe_cursors) == 3


def test_tail_rss_zero_when_N_tail_start_zero():
    """tail_RSS=0 when N_tail_start=0 (no tail computation)."""
    sbr = _sbr()
    THIS, param = OptFom_Compute_DFE(sbr, _THIS(), _param(), False, 0)
    assert THIS.tail_RSS == 0.0


def test_floating_tap_coef_empty_no_floating():
    """floating_tap_coef is empty when Floating_DFE=False."""
    sbr = _sbr()
    THIS, param = OptFom_Compute_DFE(sbr, _THIS(), _param(), False, 0)
    assert len(THIS.floating_tap_coef) == 0
