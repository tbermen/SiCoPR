"""Verification tests for get_pdf_from_sampled_signal().

# ============================================================
# MATLAB GROUND TRUTH (lines 7473-7518)
# if max(|x|) <= BinSize: return delta pdf at 0
# else: sort descending by |value|, convolve delta PDFs for each ISI sample
# L: number of PAM levels; values = 2*(0..L-1)/(L-1)-1
# Output pdf struct: BinSize, Min, x, y
# FAST_NOISE_CONV: small (<0.001) residual taps approximated as one Gaussian (normal_dist),
#   convolved in via conv_fct (MATLAB conv_fct_TEST is undefined; conv_fct per L7516 intent)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.get_pdf_from_sampled_signal.py_impl import get_pdf_from_sampled_signal


def test_small_input_returns_delta():
    """All |values| <= BinSize → delta PDF at 0."""
    iv = np.array([0.001, 0.002, -0.001])
    pdf = get_pdf_from_sampled_signal(iv, 4, 0.01)
    assert pdf.Min == 0
    assert len(pdf.y) == 1
    assert pdf.y[0] == pytest.approx(1.0)


def test_output_is_pdf_struct():
    """Output has BinSize, Min, x, y fields."""
    iv = np.array([0.5, -0.3, 0.2, -0.1])
    pdf = get_pdf_from_sampled_signal(iv, 4, 0.05)
    assert hasattr(pdf, 'BinSize')
    assert hasattr(pdf, 'Min')
    assert hasattr(pdf, 'x')
    assert hasattr(pdf, 'y')


def test_y_sums_to_one():
    """PDF y values sum to approximately 1."""
    iv = np.array([0.4, -0.3, 0.2, -0.15])
    pdf = get_pdf_from_sampled_signal(iv, 4, 0.05)
    assert np.sum(pdf.y) == pytest.approx(1.0, rel=1e-6)


def test_binsize_preserved():
    """Output BinSize matches input BinSize."""
    iv = np.array([0.3, -0.2])
    pdf = get_pdf_from_sampled_signal(iv, 2, 0.025)
    assert pdf.BinSize == 0.025


def test_fast_noise_conv_runs_and_close_to_exact():
    """FAST_NOISE_CONV=1: small (<0.001) residual taps approximated as one Gaussian;
    yields a valid normalized PDF whose overall spread (std) matches the exact path
    (the dominant large taps are convolved identically; the small residual is tiny)."""
    BinSize = 0.0001
    iv = np.concatenate([[0.5, -0.3, 0.2],
                         np.full(150, 0.0005), np.full(150, -0.0004)])  # large + many small
    exact = get_pdf_from_sampled_signal(iv.copy(), 4, BinSize, FAST_NOISE_CONV=0)
    fast = get_pdf_from_sampled_signal(iv.copy(), 4, BinSize, FAST_NOISE_CONV=1)
    assert np.sum(fast.y) == pytest.approx(1.0, rel=1e-6)
    assert len(fast.x) == len(fast.y)

    def std(p):
        m = np.sum(p.x * p.y)
        return np.sqrt(np.sum((p.x - m) ** 2 * p.y))
    assert std(fast) == pytest.approx(std(exact), rel=0.02)


def test_fast_noise_conv_no_small_taps_is_exact():
    """With no taps below 0.001, FAST_NOISE_CONV=1 keeps all taps exact (== exact path)."""
    iv = np.array([0.5, -0.3, 0.2, -0.15])
    exact = get_pdf_from_sampled_signal(iv.copy(), 4, 0.05, FAST_NOISE_CONV=0)
    fast = get_pdf_from_sampled_signal(iv.copy(), 4, 0.05, FAST_NOISE_CONV=1)
    assert fast.Min == exact.Min
    np.testing.assert_allclose(fast.y, exact.y)


def test_x_y_length_match():
    """len(x) == len(y)."""
    iv = np.array([0.4, -0.3, 0.2])
    pdf = get_pdf_from_sampled_signal(iv, 4, 0.05)
    assert len(pdf.x) == len(pdf.y)
