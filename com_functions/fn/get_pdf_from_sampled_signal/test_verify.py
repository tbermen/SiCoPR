"""Verification tests for get_pdf_from_sampled_signal().

# ============================================================
# MATLAB GROUND TRUTH (lines 7473-7518)
# if max(|x|) <= BinSize: return delta pdf at 0
# else: sort descending by |value|, convolve delta PDFs for each ISI sample
# L: number of PAM levels; values = 2*(0..L-1)/(L-1)-1
# Output pdf struct: BinSize, Min, x, y
# FAST_NOISE_CONV: ends in conv_fct_TEST, which is defined NOWHERE in the
#   reference, so that branch cannot return a pdf at all
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


def test_fast_noise_conv_cannot_complete():
    """FAST_NOISE_CONV=1 ends in conv_fct_TEST, which the reference never defines.

    COM Octave, get_pdf_from_sampled_signal([0.03 0.02], 4, 1e-3, 1):
        "error: 'conv_fct_TEST' undefined near line 45".
    Substituting conv_fct (the commented-out L7516) produced a pdf that the
    reference cannot produce, so nothing could check it.  Every call site in
    the reference hard-codes FAST_NOISE_CONV=0.
    """
    iv = np.array([0.5, -0.3, 0.2, -0.15])
    with pytest.raises(NameError, match='conv_fct_TEST'):
        get_pdf_from_sampled_signal(iv.copy(), 4, 0.05, FAST_NOISE_CONV=1)


def test_fast_noise_conv_still_takes_the_early_return():
    """The delta-pdf return happens BEFORE the FAST_NOISE_CONV block, so it is
    reached even with the flag set.

    COM Octave with FAST_NOISE_CONV=1: every tap below BinSize -> pdf.Min 0,
    one bin; an empty input -> the same.  Neither errors.
    """
    iv = np.array([0.031, -0.012, 0.0004, 0.0219, -0.0451, 0.0007, -0.0033, 0.019])
    pdf = get_pdf_from_sampled_signal(iv, 4, 1.0, FAST_NOISE_CONV=1)
    assert pdf.Min == 0 and len(pdf.y) == 1
    pdf = get_pdf_from_sampled_signal(np.array([]), 4, 1e-3, FAST_NOISE_CONV=1)
    assert pdf.Min == 0 and len(pdf.y) == 1


def test_x_y_length_match():
    """len(x) == len(y)."""
    iv = np.array([0.4, -0.3, 0.2])
    pdf = get_pdf_from_sampled_signal(iv, 4, 0.05)
    assert len(pdf.x) == len(pdf.y)


# ============================================================
# COM Octave oracle values (2026-09-22)
# get_pdf_from_sampled_signal taken from matlab/com_ieee8023_4p16p0.m (the
# copy in the Octave compat file is an inlined speed variant) and run under
# Octave with d_cpdf, Init_PDF_Fast and conv_fct.  The numerics agreed
# bit-for-bit on all ten probes; the divergences were at the edges.
# ============================================================

OCT_IV = np.array([0.031, -0.012, 0.0004, 0.0219, -0.0451, 0.0007, -0.0033, 0.019])


def test_oracle_nominal_pdf():
    """L=4, BinSize=1e-3.  COM Octave: 265 bins, Min -132, peak at index 118."""
    pdf = get_pdf_from_sampled_signal(OCT_IV, 4, 1e-3)
    assert len(pdf.y) == 265
    assert pdf.Min == -132
    assert int(np.argmax(pdf.y)) == 118
    np.testing.assert_array_equal(
        np.asarray(pdf.y)[[0, 66, 118, 132, 198, 264]],
        [0.000244140625, 0.00390625, 0.008056640625, 0.0078125, 0.00390625,
         0.000244140625])
    assert pdf.x[0] == -0.13200000000000001
    assert pdf.x[-1] == 0.13200000000000001


def test_oracle_empty_input_returns_the_delta_pdf():
    """MATLAB max([]) is [], and `if []` is false.

    COM Octave, get_pdf_from_sampled_signal([],4,1e-3) -> Min 0, one bin, y=1.
    np.max raised "zero-size array to reduction operation maximum".
    """
    pdf = get_pdf_from_sampled_signal(np.array([]), 4, 1e-3)
    assert pdf.Min == 0
    assert len(pdf.y) == 1
    assert pdf.y[0] == 1.0


def test_oracle_max_exactly_at_binsize_is_not_greater():
    """The test is `> BinSize`, not `>=`.

    COM Octave, input [0.01 -0.01] with BinSize 0.01 -> the delta pdf.
    """
    pdf = get_pdf_from_sampled_signal(np.array([0.01, -0.01]), 4, 0.01)
    assert pdf.Min == 0 and len(pdf.y) == 1


def test_oracle_nan_is_skipped_by_max_and_dropped_by_the_filter():
    """MATLAB max() ignores a NaN, and abs(NaN) > BinSize is false.

    COM Octave, [0.03 NaN 0.01] with BinSize 1e-3 -> 81 bins, Min -40, the
    same pdf the two real taps alone would give.
    """
    pdf = get_pdf_from_sampled_signal(np.array([0.03, np.nan, 0.01]), 4, 1e-3)
    assert len(pdf.y) == 81
    assert pdf.Min == -40
    np.testing.assert_array_equal(
        np.asarray(pdf.y)[[0, 20, 40, 60, 80]],
        [0.0625, 0.125, 0.125, 0.125, 0.0625])


def test_oracle_all_nan_input_returns_the_delta_pdf():
    """max() of an all-NaN vector is NaN, and NaN > BinSize is false.

    COM Octave, [NaN NaN] -> Min 0, one bin.
    """
    pdf = get_pdf_from_sampled_signal(np.array([np.nan, np.nan]), 4, 1e-3)
    assert pdf.Min == 0 and len(pdf.y) == 1


def test_oracle_column_input_matches_the_row():
    """COM Octave gives the same 265-bin pdf for v(:) as for v(:).'."""
    pdf = get_pdf_from_sampled_signal(OCT_IV.reshape(-1, 1), 4, 1e-3)
    assert len(pdf.y) == 265 and pdf.Min == -132


def test_oracle_single_tap():
    """COM Octave, a scalar 0.03 with L=4, BinSize=1e-3 -> 61 bins, Min -30,
    the four PAM levels at 0.25 each (the two ends land on bins 0 and 60)."""
    pdf = get_pdf_from_sampled_signal(np.array([0.03]), 4, 1e-3)
    assert len(pdf.y) == 61 and pdf.Min == -30
    np.testing.assert_array_equal(
        np.asarray(pdf.y)[[0, 15, 30, 45, 60]], [0.25, 0, 0, 0, 0.25])
