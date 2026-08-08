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
