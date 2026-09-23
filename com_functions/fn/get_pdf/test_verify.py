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


# ============================================================
# COM Octave oracle values — tools/octave_oracle.py runs get_pdf verbatim out of
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to the body in
# matlab/com_ieee8023_4p16p0.m), with dfe_clipper / d_cpdf / Init_PDF_Fast /
# conv_fct from the same file and get_pdf_from_sampled_signal from matlab/ (the
# compat copy is a rewritten Octave speed variant).  Pinned 2026-09-22.
#
# Four divergences these pin:
#  * strcmp(OP.FFE_OPT_METHOD,'MMSE') is CASE SENSITIVE, so 'mmse' takes the
#    phase-loop path.  The port compared .upper() and took the MMSE path.
#  * mxV is sized from `phases`, which for THRU is a SCALAR.  Writing mxV(k) at
#    the 1-based phase auto-grows it with leading zeros, so when that phase's
#    sigma is 0 max() returns index 1 -- a phase pdf_samples never got, and the
#    reference hands back an unset struct with every field [].  The port
#    returned the delta pdf.
#  * SBR(t_s + M*(0:ndfe)) past the end of the pulse response: MATLAB refuses,
#    the port silently used fewer DFE taps.
#  * residual_response(start_cancel:end_cancel) off either end: MATLAB refuses,
#    the port silently shifted/shortened the cancellation window.
#
# Six nominal cases (THRU/NEXT, FOM/MMSE, dfe_delta, Floating_DFE) come back
# bit-for-bit identical, so only the edges above are pinned here.
# ============================================================

def _octave_param(M=8, ndfe=3, levels=4):
    return _param(M=M, ndfe=ndfe, levels=levels)


def test_octave_ffe_opt_method_is_case_sensitive():
    """OP.FFE_OPT_METHOD='mmse' must NOT take the MMSE path.

    COM Octave: NEXT, 200-sample N(0,0.01) response (numpy default_rng(1)),
    t_s=41 (1-based), ixphase=4, delta_y=1e-3, M=8, ndfe=3, levels=4 ->
      numel(pdf.y)=407, sum=1, max=0.010023007630934444 at 1-based 204,
      pdf.y(1)=pdf.y(end)=2.2737367544323206e-13,
      pdf.x(1)=-0.20300000000000001, pdf.Min=-203
    Taking the MMSE/ixphase path instead gave a 295-bin pdf.
    """
    sbr = np.random.default_rng(1).normal(0, 0.01, 200)
    ch = SimpleNamespace(eq_pulse_response=sbr, type='NEXT', base='b')
    pdf = get_pdf(ch, 1e-3, 40, _octave_param(), _OP('mmse', True), 3)
    assert len(pdf.y) == 407
    assert float(np.sum(pdf.y)) == 1.0
    assert float(np.max(pdf.y)) == 0.010023007630934444
    assert int(np.argmax(pdf.y)) == 203
    assert float(pdf.y[0]) == 2.2737367544323206e-13
    assert float(pdf.y[-1]) == 2.2737367544323206e-13
    assert float(pdf.x[0]) == -0.20300000000000001
    assert int(pdf.Min) == -203


def _thru_sbr(N=200, cursor=40, M=8):
    sbr = np.zeros(N)
    sbr[cursor] = 1.0
    for k in range(1, 4):
        sbr[cursor + k * M] = 0.2 / k
    sbr[cursor - M] = 0.05
    sbr[cursor + 5 * M] = -0.03
    sbr[cursor + 9 * M] = 0.02
    return sbr


def test_octave_thru_zero_sigma_phase_returns_unset_struct():
    """THRU whose chosen phase has sigma 0 and 1-based phase > 1.

    COM Octave: the SBR below with t_s=40 (1-based; phase mod(40,8)==0 -> 8)
    returns numel(pdf.x)==0, numel(pdf.y)==0, pdf.BinSize==[] -- max() of the
    zero-padded mxV picks index 1, which pdf_samples never received.  The port
    returned a 1-bin delta pdf with BinSize 1e-3.
    """
    ch = SimpleNamespace(eq_pulse_response=_thru_sbr(), type='THRU', base='b')
    pdf = get_pdf(ch, 1e-3, 39, _octave_param(), _OP())
    assert np.asarray(pdf.y).size == 0
    assert np.asarray(pdf.x).size == 0
    assert np.asarray(pdf.BinSize).size == 0


def test_octave_thru_zero_sigma_at_phase_1_is_not_affected():
    """Guard the other side: phase 1 IS assigned, so it must still be returned.

    COM Octave: all-zero 200-sample THRU response, t_s=41 (phase mod(41,8)==1)
    -> numel(pdf.y)==1, sum==1, BinSize==1e-3.
    """
    ch = SimpleNamespace(eq_pulse_response=np.zeros(200), type='THRU', base='b')
    pdf = get_pdf(ch, 1e-3, 40, _octave_param(), _OP())
    assert len(pdf.y) == 1
    assert float(np.sum(pdf.y)) == 1.0
    assert float(pdf.BinSize) == 1e-3


def test_octave_postcursor_past_end_raises():
    """COM Octave: SBR 1x60, t_s=41, ndfe=3 ->
    "error: SBR(65): out of bound 60 (dimensions are 1x60)".
    The port silently dropped the two unreachable DFE taps."""
    ch = SimpleNamespace(eq_pulse_response=_thru_sbr()[:60], type='THRU', base='b')
    with pytest.raises(IndexError):
        get_pdf(ch, 1e-3, 40, _octave_param(), _OP())


def test_octave_cancellation_window_before_start_raises():
    """COM Octave: t_s=3 with M=8 -> start_cancel is -1 ->
    "error: residual_response(-1): subscripts must be either integers 1 to
    (2^63)-1 or logicals".  The port shifted the window instead."""
    ch = SimpleNamespace(eq_pulse_response=_thru_sbr(), type='THRU', base='b')
    with pytest.raises(IndexError):
        get_pdf(ch, 1e-3, 2, _octave_param(), _OP())


def test_octave_cancellation_window_past_end_raises():
    """66 samples: the last DFE postcursor (1-based 65) is in range but the
    cancellation window ends at 68.

    COM Octave, this exact input with t_s=41:
    "error: residual_response(68): out of bound 66 (dimensions are 1x66)".
    The port shortened the window instead."""
    sbr = np.zeros(66)
    sbr[40] = 1.0
    for k in range(1, 4):
        sbr[40 + k * 8] = 0.2 / k
    sbr[40 - 8] = 0.05
    ch = SimpleNamespace(eq_pulse_response=sbr, type='THRU', base='b')
    with pytest.raises(IndexError):
        get_pdf(ch, 1e-3, 40, _octave_param(), _OP())


def test_caller_eq_pulse_response_is_not_modified():
    """MATLAB passes chdata by value: get_pdf must not write the caller's SBR.

    SBR is np.asarray(chdata.eq_pulse_response).ravel(), and for an array that
    is already float64 and 1-D both of those return a VIEW, not a copy. So SBR
    aliases the caller's chdata, and residual_response = SBR.copy() is the only
    thing standing between the subtraction loop and the caller's pulse
    response.

    This is the same class as the OptFom_Calc_FOM leak, where a LOSING EQ
    candidate's response was written back into chdata and carried into the next
    tick. Every other test in this file reads only the returned pdf, so all of
    them pass with that copy removed.

    Pinned by tests/test_mutation_score.py: the drop_dot_copy mutant at
    py_impl line 142 is caught only by this test.
    """
    ch = _thru_chdata()
    before = ch.eq_pulse_response.copy()

    get_pdf(ch, 1e-3, 40, _param(), _OP())

    np.testing.assert_array_equal(
        ch.eq_pulse_response, before,
        err_msg='get_pdf modified the caller\'s chdata.eq_pulse_response')
