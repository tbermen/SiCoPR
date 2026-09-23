"""Verification tests for adjust_Rx_noise_for_quantization().

# ============================================================
# MATLAB GROUND TRUTH (lines 4858-4896)
# Computes ADC clip, quantization noise, and convolves into
# the combined_interference_and_noise_pdf.
# NS gets sigma_Q, sigma_before_clip, peak_clip, p2ptosigma_clip,
# quantization_noise_pdf fields.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.adjust_Rx_noise_for_quantization.py_impl import adjust_Rx_noise_for_quantization


def _gaussian_pdf(BinSize=0.01, sigma=0.1):
    x = np.arange(-200, 201) * BinSize
    y = np.exp(-0.5 * (x / sigma) ** 2)
    y /= y.sum()
    return SimpleNamespace(BinSize=BinSize, Min=-200, x=x, y=y)


def _param():
    return SimpleNamespace(
        levels=4,
        delta_y=0.01,
        P_qc=1e-4,
        N_qb=6,
        ffe_pre_tap_len=1,
    )


def _fom_result():
    return SimpleNamespace(RxFFE=np.array([1.0, 0.0]))


def _chdata():
    return [SimpleNamespace(pulse_sampled_w_tx_ffe_ctle=np.array([0.3, -0.2, 0.1]))]


def test_ns_sigma_q_set():
    """NS.sigma_Q is set and positive."""
    pdf = _gaussian_pdf()
    NS = SimpleNamespace()
    cd, NS2, _ = adjust_Rx_noise_for_quantization(pdf, NS, _chdata(), _fom_result(), _param(), None)
    assert hasattr(NS2, 'sigma_Q')
    assert NS2.sigma_Q > 0


def test_ns_peak_clip_set():
    """NS.peak_clip is set and positive."""
    NS = SimpleNamespace()
    _, NS2, _ = adjust_Rx_noise_for_quantization(_gaussian_pdf(), NS, _chdata(), _fom_result(), _param(), None)
    assert NS2.peak_clip > 0


def test_output_pdf_has_fields():
    """Output combined pdf has BinSize, Min, x, y."""
    NS = SimpleNamespace()
    _, _, out_pdf = adjust_Rx_noise_for_quantization(_gaussian_pdf(), NS, _chdata(), _fom_result(), _param(), None)
    for f in ('BinSize', 'Min', 'x', 'y'):
        assert hasattr(out_pdf, f)


def test_ns_quantization_noise_pdf():
    """NS.quantization_noise_pdf is set."""
    NS = SimpleNamespace()
    _, NS2, _ = adjust_Rx_noise_for_quantization(_gaussian_pdf(), NS, _chdata(), _fom_result(), _param(), None)
    assert hasattr(NS2, 'quantization_noise_pdf')


def test_output_pdf_y_sums_to_one():
    """Output PDF y sums approximately to 1."""
    NS = SimpleNamespace()
    _, _, out_pdf = adjust_Rx_noise_for_quantization(_gaussian_pdf(), NS, _chdata(), _fom_result(), _param(), None)
    assert np.sum(out_pdf.y) == pytest.approx(1.0, rel=1e-4)


# ============================================================
# COM Octave oracle values (tools/octave_oracle.py). The compat file's
# get_pdf_from_sampled_signal is a rewritten speed variant, so that one body
# was taken verbatim from matlab/com_ieee8023_4p16p0.m; every other
# subfunction (conv_fct, CDF_inv_ev, scalePDF, d_cpdf, Init_PDF_Fast) is
# byte-identical between the two files.
#
# Fixture: combined pdf Min=-20, BinSize=0.01, x=(-20:20)*0.01,
#   y = exp(-(x/0.06)^2/2) normalised;
#   pulse_sampled_w_tx_ffe_ctle = [0.42 -0.05 0.031 0.022 -0.013 0.004];
#   RxFFE = [0.05 -0.1 0.2 1.0 -0.3 0.1];
#   levels=4, delta_y=0.01, P_qc=1e-4, N_qb=8, ffe_pre_tap_len=3.
#
# What the empty-pulse case catches: the inlined copy of
# get_pdf_from_sampled_signal here had not tracked the canonical's guard for
# an empty input, so np.max raised where MATLAB's `if []` is simply false.
# ============================================================

def _oracle_pdf():
    x = np.arange(-20, 21) * 0.01
    y = np.exp(-(x / 0.06) ** 2 / 2)
    return SimpleNamespace(BinSize=0.01, Min=-20, x=x, y=y / y.sum())


def _oracle_param():
    return SimpleNamespace(levels=4, delta_y=0.01, P_qc=1e-4, N_qb=8,
                           ffe_pre_tap_len=3)


_ORACLE_RXFFE = np.array([0.05, -0.1, 0.2, 1.0, -0.3, 0.1])

# cin.y[50:61] -- the same eleven samples in both cases below, because an
# all-sub-bin pulse and an empty one both fall through to the delta pdf.
_CIN_Y_50_60 = [
    0.046880990870615716, 0.053001181468932901, 0.058308461396681208,
    0.062421681865330682, 0.065027585198255952, 0.065920170006215945,
    0.065027585198255938, 0.062421681865330682, 0.058308461396681208,
    0.053001181468932901, 0.046880990870615716]


def test_octave_nominal_case():
    """COM Octave, the fixture above."""
    NS = SimpleNamespace()
    chdata = [SimpleNamespace(
        pulse_sampled_w_tx_ffe_ctle=np.array([0.42, -0.05, 0.031, 0.022,
                                              -0.013, 0.004]))]
    fom = SimpleNamespace(RxFFE=_ORACLE_RXFFE)
    _, NS2, cin = adjust_Rx_noise_for_quantization(
        _oracle_pdf(), NS, chdata, fom, _oracle_param(), None)
    assert NS2.sigma_Q == pytest.approx(0.0014943183437849136, rel=1e-13)
    assert NS2.sigma_before_clip == pytest.approx(0.32221364059106311, rel=1e-13)
    assert NS2.peak_clip == pytest.approx(0.65999999999999992, rel=1e-13)
    assert NS2.p2ptosigma_clip == pytest.approx(4.0966608290655069, rel=1e-13)
    assert cin.Min == -55 and len(cin.y) == 111
    np.testing.assert_allclose(cin.y[50:61], _CIN_Y_50_60, rtol=1e-12)
    # the cursor tap is skipped, so quantization_noise_pdf is the delta
    # convolved with the five remaining scaled uniform blocks
    assert NS2.quantization_noise_pdf.Min == -35
    assert np.flatnonzero(NS2.quantization_noise_pdf.y).tolist() == [34, 35, 36]


def test_octave_empty_sampled_pulse_falls_through_to_the_delta_pdf():
    """An empty pulse_sampled_w_tx_ffe_ctle is not an error in MATLAB.

    COM Octave with chdata(1).pulse_sampled_w_tx_ffe_ctle = []:
      NS.peak_clip = 0.2, NS.sigma_Q = 4.5282374054088305e-04,
      NS.sigma_before_clip = 0.059763117245897099,
      NS.p2ptosigma_clip = 6.6930912983368707, cin.Min = -55, 111 bins.
    np.max raised "zero-size array to reduction operation maximum" instead.
    """
    NS = SimpleNamespace()
    chdata = [SimpleNamespace(pulse_sampled_w_tx_ffe_ctle=np.zeros(0))]
    fom = SimpleNamespace(RxFFE=_ORACLE_RXFFE)
    _, NS2, cin = adjust_Rx_noise_for_quantization(
        _oracle_pdf(), NS, chdata, fom, _oracle_param(), None)
    assert NS2.peak_clip == pytest.approx(0.20000000000000001, rel=1e-13)
    assert NS2.sigma_Q == pytest.approx(0.00045282374054088305, rel=1e-13)
    assert NS2.sigma_before_clip == pytest.approx(0.059763117245897099, rel=1e-13)
    assert NS2.p2ptosigma_clip == pytest.approx(6.6930912983368707, rel=1e-13)
    assert cin.Min == -55 and len(cin.y) == 111
    np.testing.assert_allclose(cin.y[50:61], _CIN_Y_50_60, rtol=1e-12)
