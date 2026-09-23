"""Verification tests for scaleCDF().

# ============================================================
# MATLAB GROUND TRUTH
# scale_factor = 1/10^(-delta_com/20) = 10^(delta_com/20)
#   delta_com=0  → scale_factor=1.0
#   delta_com=20 → scale_factor=10.0
#   delta_com=40 → scale_factor=100.0
#
# Test PDF: BinSize=0.1, Min=-2
#   x=[-0.2,-0.1,0.0,0.1,0.2], y=[0.20,0.30,0.30,0.15,0.05]
#   P=cumsum(y)=[0.20,0.50,0.80,0.95,1.00]
#   DER0=0.4, A_s=1.0:
#     ider0 = first index where P>=0.4 = index 1 (P[1]=0.50)
#     anias = x[1]/1.0 = -0.10
#     scale_factor = 10^(delta_com/20)
#
# cdf_out = cumsum(pdf_out.y)
# cdf_out[-1] = 1.0 (pdf_out.y is normalized)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.scaleCDF.py_impl import scaleCDF


def make_pdf():
    return SimpleNamespace(
        BinSize=0.1, Min=-2,
        x=np.array([-0.2, -0.1, 0.0, 0.1, 0.2]),
        y=np.array([0.20, 0.30, 0.30, 0.15, 0.05]),
    )


def test_scale_factor_zero_delta():
    """delta_com=0 → scale_factor = 1.0."""
    _, _, sf = scaleCDF(make_pdf(), 0.0, 0.4, 1.0)
    assert sf == pytest.approx(1.0, rel=1e-10)


def test_scale_factor_20dB():
    """delta_com=20 → scale_factor = 10.0."""
    _, _, sf = scaleCDF(make_pdf(), 20.0, 0.4, 1.0)
    assert sf == pytest.approx(10.0, rel=1e-6)


def test_scale_factor_40dB():
    """delta_com=40 → scale_factor = 100.0."""
    _, _, sf = scaleCDF(make_pdf(), 40.0, 0.4, 1.0)
    assert sf == pytest.approx(100.0, rel=1e-6)


def test_cdf_out_ends_at_one():
    """cdf_out[-1] = 1.0 (pdf_out is normalised)."""
    _, cdf_out, _ = scaleCDF(make_pdf(), 0.0, 0.4, 1.0)
    assert cdf_out[-1] == pytest.approx(1.0, rel=1e-10)


def test_cdf_out_monotone():
    """cdf_out is non-decreasing."""
    _, cdf_out, _ = scaleCDF(make_pdf(), 6.0, 0.4, 1.0)
    assert np.all(np.diff(cdf_out) >= -1e-14)


# ============================================================
# COM Octave oracle values (tools/octave_oracle.py, scaleCDF + scalePDF
# extracted verbatim from octave/com_ieee8023_4p16p0_octave_compat.m).
#
# What these catch:
#   1. find(P>=DER0,1,'first') is EMPTY when the CDF never reaches DER0, and
#      `-1/[]` on the next line is an error the reference refuses to pass.
#      np.argmax on an all-False mask silently returned bin 0.
#   2. interp1's default is linear with NaN OUTSIDE the data range; np.interp
#      clamps to the end values. The "NAN interp work around" in scalePDF
#      only hides that for a grid symmetric about zero.
# ============================================================

_YSYM = [0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.08, 0.05, 0.02]


def _pdf(Min, n, y, BinSize=0.05):
    """pdf whose x runs Min..Min+n-1 bins (not necessarily symmetric)."""
    return SimpleNamespace(Min=Min, BinSize=BinSize,
                           x=np.arange(Min, Min + n) * BinSize,
                           y=np.array(y, dtype=float))


def test_octave_empty_find_is_an_error():
    """DER0 above sum(pdf.y): MATLAB errors, it does not pick a bin.

    COM Octave, x=(-4:4)*0.05, y=_YSYM, DER0=5:
      "error: operator /: nonconformant arguments (op1 is 1x1, op2 is 1x0)"
      called from scaleCDF line 7 (new_db = 20*log10(-1/anias)-delta_com).
    """
    with pytest.raises(ValueError, match='DER0'):
        scaleCDF(_pdf(-4, 9, _YSYM), 1.0, 5.0, 0.5)


def test_octave_left_heavy_pdf_interpolates_to_nan():
    """max(pdf.x) < -min(pdf.x): the new grid runs off the data and interp1
    returns NaN, so the normalised y and the CDF are NaN throughout.

    COM Octave, Min=-8, x=(-8:0)*0.05, y=_YSYM, delta_com=1, DER0=0.4,
    A_s=0.5 -> pdf_out.Min = -9 and all 19 samples of pdf_out.y and cdf_out
    are NaN. np.interp clamped instead and produced 19 finite values.
    """
    pdf_out, cdf_out, sf = scaleCDF(_pdf(-8, 9, _YSYM), 1.0, 0.4, 0.5)
    assert pdf_out.Min == -9
    assert len(pdf_out.y) == 19 and len(cdf_out) == 19
    assert np.all(np.isnan(pdf_out.y))
    assert np.all(np.isnan(cdf_out))


def test_octave_right_heavy_pdf_matches_reference():
    """max(pdf.x) > -min(pdf.x): only the two end samples fall off the data,
    which is the case the neighbour-copy workaround was written for.

    COM Octave, Min=-4, x=(-4:6)*0.05, delta_com=1, DER0=0.4, A_s=0.5.
    """
    y = _YSYM + [0.01, 0.005]
    pdf_out, cdf_out, sf = scaleCDF(_pdf(-4, 11, y), 1.0, 0.4, 0.5)
    assert pdf_out.Min == -5
    assert sf == pytest.approx(1.1220184543019633, rel=1e-15)
    np.testing.assert_allclose(pdf_out.y, [
        0.059929527894748126, 0.059929527894748126, 0.097150703944833092,
        0.13437187999491809, 0.16705138567607045, 0.16705138567607045,
        0.12983020962598546, 0.079537231303439523, 0.049937853083096333,
        0.027605147453045351, 0.027605147453045351], rtol=1e-13)
    np.testing.assert_allclose(cdf_out, [
        0.059929527894748126, 0.11985905578949625, 0.21700975973432934,
        0.35138163972924741, 0.51843302540531788, 0.68548441108138836,
        0.81531462070737382, 0.8948518520108133, 0.94478970509390958,
        0.9723948525469549, 1.0000000000000002], rtol=1e-13)
