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
