"""Verification tests for get_pdf().

# ============================================================
# MATLAB GROUND TRUTH (lines 7377-7472)
# Computes ISI PDF from equalised pulse response.
# THRU type: removes cursor and DFE postcursors before computing PDF.
# Non-THRU type: uses all phases.
# Returns pdf SimpleNamespace with y, x, BinSize fields.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.get_pdf.py_impl import get_pdf


def _param(M=8, ndfe=3, levels=4):
    return SimpleNamespace(
        samples_per_ui=M, ndfe=ndfe, levels=levels,
        N_bmax=ndfe,
        Floating_DFE=False,
        dfe_delta=0,
        bmax=np.ones(ndfe) * 0.9,
        bmin=-np.ones(ndfe) * 0.9,
        use_bmax=np.ones(ndfe) * 0.9,
        use_bmin=-np.ones(ndfe) * 0.9,
    )


def _OP(method='', rxffe=False):
    return SimpleNamespace(
        FFE_OPT_METHOD=method,
        RxFFE=rxffe,
        DISPLAY_WINDOW=False,
    )


def _thru_chdata(M=8, cursor_i=40, N=200):
    sbr = np.zeros(N)
    sbr[cursor_i] = 1.0
    for k in range(1, 4):
        sbr[cursor_i + k * M] = 0.2 / k
    sbr[cursor_i - M] = 0.05
    return SimpleNamespace(eq_pulse_response=sbr, type='THRU')


def _xtalk_chdata(M=8, N=200):
    sbr = np.random.default_rng(42).normal(0, 0.01, N)
    return SimpleNamespace(eq_pulse_response=sbr, type='NEXT')


def test_returns_pdf():
    """get_pdf returns a PDF struct with y, x, BinSize."""
    ch = _thru_chdata()
    pdf = get_pdf(ch, 1e-3, 40, _param(), _OP())
    assert hasattr(pdf, 'y')
    assert hasattr(pdf, 'x')
    assert hasattr(pdf, 'BinSize')


def test_pdf_y_sums_to_one():
    """PDF.y sums to 1."""
    ch = _thru_chdata()
    pdf = get_pdf(ch, 1e-3, 40, _param(), _OP())
    assert float(np.sum(pdf.y)) == pytest.approx(1.0, abs=1e-6)


def test_xtalk_type_uses_all_phases():
    """Non-THRU type returns a valid PDF."""
    ch = _xtalk_chdata()
    pdf = get_pdf(ch, 1e-3, 40, _param(), _OP())
    assert len(pdf.y) > 0


def test_pdf_x_y_same_length():
    """PDF.x and PDF.y have the same length."""
    ch = _thru_chdata()
    pdf = get_pdf(ch, 1e-3, 40, _param(), _OP())
    assert len(pdf.x) == len(pdf.y)


def test_all_zero_sbr_returns_delta():
    """All-zero SBR returns trivial delta PDF."""
    ch = SimpleNamespace(eq_pulse_response=np.zeros(200), type='THRU')
    pdf = get_pdf(ch, 1e-3, 40, _param(), _OP())
    assert len(pdf.y) >= 1
    assert float(np.sum(pdf.y)) == pytest.approx(1.0, abs=1e-6)
