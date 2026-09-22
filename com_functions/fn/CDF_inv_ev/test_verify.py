"""Verification tests for CDF_inv_ev().

# ============================================================
# MATLAB GROUND TRUTH
# index = find(CDF >= val, 1, 'first')   (1-based)
# if isempty(index): return PDF.x(end)
# else: return PDF.x(index)
#
# PDF.x = [-0.3,-0.2,-0.1,0,0.1,0.2,0.3]
# CDF   = [0.05,0.15,0.35,0.65,0.85,0.95,1.0]
#
# val=0.5:  first CDF>=0.5 is CDF[3]=0.65 (index 3); return x[3]=0.0
# val=0.9:  first CDF>=0.9 is CDF[5]=0.95 (index 5); return x[5]=0.2
# val=1.5:  no CDF>=1.5 → isempty → return x[-1]=0.3
# val=0.05: first CDF>=0.05 is CDF[0]=0.05; return x[0]=-0.3
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.CDF_inv_ev.py_impl import CDF_inv_ev


def make_pdf_cdf():
    PDF = SimpleNamespace(x=np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]))
    CDF = np.array([0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 1.0])
    return PDF, CDF


def test_midrange_val():
    """val=0.5 → first CDF>=0.5 is index 3; return x[3]=0.0."""
    PDF, CDF = make_pdf_cdf()
    assert CDF_inv_ev(0.5, PDF, CDF) == pytest.approx(0.0)


def test_high_val():
    """val=0.9 → first CDF>=0.9 is index 5; return x[5]=0.2."""
    PDF, CDF = make_pdf_cdf()
    assert CDF_inv_ev(0.9, PDF, CDF) == pytest.approx(0.2)


def test_no_match_returns_last_x():
    """val=1.5 → no CDF>=1.5 → return PDF.x[-1]=0.3."""
    PDF, CDF = make_pdf_cdf()
    assert CDF_inv_ev(1.5, PDF, CDF) == pytest.approx(0.3)


def test_at_left_edge():
    """val=0.05 → first CDF>=0.05 is index 0; return x[0]=-0.3."""
    PDF, CDF = make_pdf_cdf()
    assert CDF_inv_ev(0.05, PDF, CDF) == pytest.approx(-0.3)


# ============================================================
# Executed-reference checks — values produced by running the reference
# function under Octave (tools/octave_oracle.py, COM Octave 4p16p0), not by
# reading the MATLAB.  PDF.x=[-0.3:0.1:0.3], CDF=[0.05 0.15 0.35 0.65 0.85
# 0.95 1.0] unless stated.
#
#   CDF_inv_ev(NaN, PDF, CDF)   -> 0.3    (nothing compares >= NaN, isempty
#                                          branch, PDF.x(end))
#   CDF_inv_ev(-Inf, PDF, CDF)  -> -0.3   (first element already qualifies)
#   CDF_inv_ev(0.65, PDF, CDF)  -> 0      (>= is inclusive at an exact hit)
#   CDF shortened to [0.05 0.15 0.35], val=0.99 -> 0.3  (isempty branch takes
#                                          PDF.x(end), not CDF's own end)
#   CDF with a NaN at index 2, val=0.5 -> 0  (NaN never satisfies >=)
#   PDF.x = [-0.3 -0.2] with the full 7-long CDF, val=0.5 -> errors
#     "PDF(4): out of bound 2 (dimensions are 1x2)"
#
# No divergence was found in this function: the Octave run agreed with the
# Python for every probe above, including the column-vector PDF.x form.
# ============================================================


def test_nan_val_takes_last_x():
    """val=NaN: nothing compares >= NaN, so the isempty branch runs."""
    PDF, CDF = make_pdf_cdf()
    assert CDF_inv_ev(float('nan'), PDF, CDF) == pytest.approx(0.3)


def test_neg_inf_val_takes_first_x():
    PDF, CDF = make_pdf_cdf()
    assert CDF_inv_ev(float('-inf'), PDF, CDF) == pytest.approx(-0.3)


def test_exact_cdf_hit_is_inclusive():
    """val exactly equal to a CDF entry selects that entry, not the next."""
    PDF, CDF = make_pdf_cdf()
    assert CDF_inv_ev(0.65, PDF, CDF) == pytest.approx(0.0)


def test_short_cdf_no_match_uses_pdf_x_end():
    """isempty branch returns PDF.x(end) even when CDF is shorter than PDF.x."""
    PDF, _ = make_pdf_cdf()
    assert CDF_inv_ev(0.99, PDF, np.array([0.05, 0.15, 0.35])) == pytest.approx(0.3)


def test_nan_in_cdf_is_skipped():
    PDF, _ = make_pdf_cdf()
    CDF = np.array([0.05, np.nan, 0.35, 0.65, 0.85, 0.95, 1.0])
    assert CDF_inv_ev(0.5, PDF, CDF) == pytest.approx(0.0)


def test_index_past_pdf_x_raises():
    """CDF longer than PDF.x: MATLAB errors on the out-of-bound PDF.x(index)."""
    PDF = SimpleNamespace(x=np.array([-0.3, -0.2]))
    _, CDF = make_pdf_cdf()
    with pytest.raises(IndexError):
        CDF_inv_ev(0.5, PDF, CDF)
