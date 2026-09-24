"""Verification tests for conv_fct_MeanNotZero().

# ============================================================
# MATLAB GROUND TRUTH
# Identical logic to conv_fct: p.Min=round(p1.Min+p2.Min),
# p.y=conv(p1.y,p2.y), p.x=(p.Min*BinSize:BinSize:pMax*BinSize).
# Name indicates use for non-zero-mean PDFs (non-symmetric Min).
#
# p1.Min=2, p1.y=[0.6,0.4]; p2.Min=1, p2.y=[0.7,0.3]; BinSize=0.1
#   p.Min = 3
#   p.y = conv([0.6,0.4],[0.7,0.3]) = [0.42,0.46,0.12]
#   pMax = 3+3-1 = 5
#   p.x = (3:5)*0.1 = [0.3, 0.4, 0.5]
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.conv_fct_MeanNotZero.py_impl import conv_fct_MeanNotZero


def make_pdf(min_idx, y, binsize=0.1):
    return SimpleNamespace(Min=min_idx, BinSize=binsize, y=np.array(y, dtype=float))


def test_positive_min_convolve():
    """p1.Min=2, p2.Min=1: p.Min=3, y=conv([0.6,0.4],[0.7,0.3])=[0.42,0.46,0.12]."""
    out = conv_fct_MeanNotZero(make_pdf(2, [0.6, 0.4]), make_pdf(1, [0.7, 0.3]))
    assert out.Min == 3
    np.testing.assert_allclose(out.y, [0.42, 0.46, 0.12], atol=1e-12)
    np.testing.assert_allclose(out.x, [0.3, 0.4, 0.5], atol=1e-12)


def test_x_length_matches_y():
    out = conv_fct_MeanNotZero(make_pdf(1, [0.5, 0.5]), make_pdf(2, [0.4, 0.6]))
    assert len(out.x) == len(out.y)


def test_min_additive():
    out = conv_fct_MeanNotZero(make_pdf(-3, [1.0]), make_pdf(5, [1.0]))
    assert out.Min == 2


def test_output_length():
    out = conv_fct_MeanNotZero(make_pdf(0, [0.3, 0.4, 0.3]), make_pdf(0, [0.5, 0.5]))
    assert len(out.y) == 4


def test_binsize_mismatch_raises():
    p1 = make_pdf(1, [1.0], binsize=0.1)
    p2 = make_pdf(1, [1.0], binsize=0.2)
    with pytest.raises(ValueError):
        conv_fct_MeanNotZero(p1, p2)


# ============================================================
# Divergences found by EXECUTING the reference (COM Octave oracle,
# tools/octave_oracle.py). Same three as conv_fct, since the bodies match:
#
# 1. p.x is a FLOATING-POINT COLON. The MATLAB comment immediately above the
#    line claims it is "equivalent to (p.Min:p.Min+length(p.y)-1)*p.BinSize";
#    executing it shows otherwise. Over 7920 (Min, length, BinSize) cases the
#    product form got 249387 of 1013684 elements wrong, each by 1 ulp.
# 2. p.Min used Python round() (half-to-even) where MATLAB rounds half away
#    from zero.
# 3. An empty operand: conv2 returns empty, np.convolve raised ValueError.
# ============================================================

def test_x_axis_is_a_colon_not_a_product():
    """COM Octave: BinSize=1e-4, p.Min=-5, 9 bins.

    Exact equality on purpose -- the old form was within 1 ulp everywhere, so
    assert_allclose passes on the defect.
    """
    out = conv_fct_MeanNotZero(make_pdf(-2, [1, 2, 3, 2, 1], binsize=1e-4),
                               make_pdf(-3, [1, 2, 3, 2, 1], binsize=1e-4))
    assert out.Min == -5
    assert list(out.x) == [-0.00050000000000000001, -0.00040000000000000002,
                           -0.00030000000000000003, -0.00019999999999999998,
                           -9.9999999999999991e-05, 0,
                           0.00010000000000000005, 0.00019999999999999998,
                           0.00030000000000000003]
    assert list(out.x) != list(np.arange(-5, 4) * 1e-4)


def test_x_last_element_pinned_only_when_it_overshoots():
    """COM Octave: BinSize=0.1, p.Min=-3, 2 bins -> ends -0.20000000000000004."""
    out = conv_fct_MeanNotZero(make_pdf(-3, [1, 1]), make_pdf(0, [1]))
    assert list(out.x) == [-0.30000000000000004, -0.20000000000000004]
    assert out.x[-1] != -2 * 0.1


def test_min_round_is_half_away_from_zero():
    """COM Octave: round(-0.5) = -1; Python round() gave 0."""
    assert conv_fct_MeanNotZero(make_pdf(-0.5, [1, 1]), make_pdf(0, [1])).Min == -1
    assert conv_fct_MeanNotZero(make_pdf(0.5, [1, 1]), make_pdf(0, [1])).Min == 1


def test_empty_operand_returns_empty():
    """COM Octave: conv2([1 2 3], []) is 0x0; np.convolve raised instead."""
    out = conv_fct_MeanNotZero(make_pdf(-1, [1, 2, 3]), make_pdf(0, []))
    assert out.Min == -1
    assert len(out.y) == 0
    assert len(out.x) == 0


# ============================================================
# COM Octave, 4p15p0 and 4p16p0 alike: p1 = p2 = BinSize 1e-5, Min 0,
# y = 2.^-(0:199). conv2 is a direct convolution, so every output bin is
# exact: y(n+1) = (min(n, 398-n)+1) * 2^-n, down to 1.5e-120 at the far end.
# An FFT convolution buries everything below ~eps of the peak in round-off
# (fftconvolve: y[150] = 4.0e-18 for 1.06e-43, y[300] = -2.5e-17, NEGATIVE),
# and the far tail of the noise CDF is where DER_DFE and DER_MLSE are read.
# ============================================================

def test_pdf_tail_is_exact_as_conv2_is():
    a = make_pdf(0, 2.0 ** -np.arange(200), binsize=1e-5)
    out = conv_fct_MeanNotZero(a, a)
    assert out.Min == 0 and len(out.y) == 399
    pinned = {0: 1.0, 1: 1.0, 50: 4.5297099404706387e-14,
              150: 1.0579803405652369e-43, 199: 2.4892061111444567e-58,
              250: 8.2354503341380624e-74, 300: 4.8600025306447493e-89,
              398: 1.5490367659397273e-120}
    for n, v in pinned.items():
        assert out.y[n] == v, (n, out.y[n], v)
    assert np.all(out.y > 0)


# ============================================================
# Span convolution (2026-09-24, owner's equivalence-class rule). The result is
# np.convolve of each operand's NONZERO SPAN, with the exact zeros restored
# around it. The dropped products all have a zero factor, so this differs from
# convolving the full arrays only in BLAS summation order -- on these operands
# in 168 of 608 elements, by at most 1 ulp -- while COM Octave's conv_fct is
# matched to 1.2e-15 either way, with its 152 exact zeros in the same places.
# Operands: a signal-like PDF with zero margins, and a Gaussian whose tail has
# underflowed to zero, the shape of get_PSDs' ADC-clip convolution.
# ============================================================

def _span_operands():
    rng = np.random.default_rng(0)
    a = np.concatenate([np.zeros(37), rng.random(260) ** 3, np.zeros(11)])
    x = np.linspace(-4, 4, 301)
    b = np.exp(-x ** 2 * 6)
    b[b < 1e-18] = 0.0
    return a, b


def test_convolves_the_nonzero_span_and_restores_exact_zeros():
    a, b = _span_operands()
    out = conv_fct_MeanNotZero(make_pdf(0, a, binsize=1.0), make_pdf(0, b, binsize=1.0))
    ia, ib = np.flatnonzero(a), np.flatnonzero(b)
    want = np.zeros(a.size + b.size - 1)
    want[ia[0] + ib[0]:ia[-1] + ib[-1] + 1] = np.convolve(
        a[ia[0]:ia[-1] + 1], b[ib[0]:ib[-1] + 1])
    np.testing.assert_array_equal(out.y, want)
    # COM Octave 4p15p0 conv_fct on the same operands: 152 exact zeros
    assert int(np.sum(np.asarray(out.y) == 0)) == 152
    idx = [200, 400, 544]
    oct_vals = [6.9534220007283762, 6.8446278412357691, 7.859092101412288e-19]
    np.testing.assert_allclose(np.asarray(out.y)[idx], oct_vals, rtol=1e-14, atol=0)
