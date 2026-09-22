"""Verification tests for CDF_ev().

# ============================================================
# MATLAB GROUND TRUTH
# index = find(PDF.x >= -val, 1, 'first')   (1-based)
# CDF_ev = CDF(index)
# Python: index = argmax(PDF.x >= -val)  (0-based, same element)
#
# PDF.x = [-0.3,-0.2,-0.1,0,0.1,0.2,0.3]
# CDF   = [0.05,0.15,0.35,0.65,0.85,0.95,1.0]
#
# val=0.15: -val=-0.15; first x>=-0.15 is x[2]=-0.1; CDF[2]=0.35
# val=0.0:  -val=0;     first x>=0 is x[3]=0;         CDF[3]=0.65
# val=0.3:  -val=-0.3;  first x>=-0.3 is x[0]=-0.3;   CDF[0]=0.05
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.CDF_ev.py_impl import CDF_ev


def make_pdf_cdf():
    PDF = SimpleNamespace(x=np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]))
    CDF = np.array([0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 1.0])
    return PDF, CDF


def test_val_midrange():
    """val=0.15 → CDF at first x>=-0.15 (index 2) = 0.35."""
    PDF, CDF = make_pdf_cdf()
    assert CDF_ev(0.15, PDF, CDF) == pytest.approx(0.35)


def test_val_zero():
    """val=0.0 → CDF at first x>=0 (index 3) = 0.65."""
    PDF, CDF = make_pdf_cdf()
    assert CDF_ev(0.0, PDF, CDF) == pytest.approx(0.65)


def test_val_at_left_edge():
    """val=0.3 → CDF at first x>=-0.3 (index 0) = 0.05."""
    PDF, CDF = make_pdf_cdf()
    assert CDF_ev(0.3, PDF, CDF) == pytest.approx(0.05)


def test_val_small():
    """val=0.05 → first x>=-0.05 is x[3]=0 (index 3); CDF=0.65."""
    PDF, CDF = make_pdf_cdf()
    assert CDF_ev(0.05, PDF, CDF) == pytest.approx(0.65)


# ============================================================
# Executed-reference checks — values produced by running the reference
# function under Octave (tools/octave_oracle.py, COM Octave 4p16p0), not by
# reading the MATLAB.  PDF.x=[-0.3:0.1:0.3], CDF=[0.05 0.15 0.35 0.65 0.85
# 0.95 1.0] throughout.
#
# find() vs argmax, the no-match case:
#   find(PDF.x >= 0.5, 1, 'first')  ->  []   (val = -0.5)
#   so MATLAB's CDF(index) is an empty 1x0 and there is no crossing.
#   np.argmax on an all-False mask answers 0, which handed back CDF(1)=0.05.
#   The Octave compat build, which uses lookup() instead of find(), refuses
#   outright: "CDF(8): out of bound 7 (dimensions are 1x7)".
#   Same for val=NaN and val=-Inf, where nothing compares >=.
#
# find() vs Octave's lookup(), the exact-grid case — a divergence in the
# *Octave* build, recorded here so it is not mistaken for a Python bug:
#   val=0.1 -> -val = -0.1 = PDF.x(3) exactly.
#   MATLAB find(PDF.x >= -0.1, 1) = 3        -> CDF(3) = 0.35
#   Octave lookup(PDF.x,-0.1)+1   = 3+1 = 4  -> CDF(4) = 0.65
#   lookup() is `find(PDF.x > -val, 1)`, strict where the reference is not.
#   This port follows the MATLAB reference, so 0.35 is the expected value.
# ============================================================


def test_no_match_refuses():
    """-val past the top of PDF.x: find() is empty, so there is no answer."""
    PDF, CDF = make_pdf_cdf()
    with pytest.raises(IndexError):
        CDF_ev(-0.5, PDF, CDF)


def test_nan_val_refuses():
    """val=NaN: nothing compares >= NaN, so find() is empty."""
    PDF, CDF = make_pdf_cdf()
    with pytest.raises(IndexError):
        CDF_ev(float('nan'), PDF, CDF)


def test_neg_inf_val_refuses():
    """val=-Inf: -val=+Inf is above every PDF.x."""
    PDF, CDF = make_pdf_cdf()
    with pytest.raises(IndexError):
        CDF_ev(float('-inf'), PDF, CDF)


def test_val_at_exact_grid_point_is_inclusive():
    """-val exactly on a PDF.x point: `>=` takes that bin, not the next one."""
    PDF, CDF = make_pdf_cdf()
    assert CDF_ev(0.1, PDF, CDF) == pytest.approx(0.35)   # not 0.65
    assert CDF_ev(0.2, PDF, CDF) == pytest.approx(0.15)   # not 0.35
    assert CDF_ev(-0.1, PDF, CDF) == pytest.approx(0.85)  # not 0.95


def test_val_below_range_takes_first_bin():
    """-val below PDF.x(1): find() returns 1, so CDF(1)."""
    PDF, CDF = make_pdf_cdf()
    assert CDF_ev(0.5, PDF, CDF) == pytest.approx(0.05)
    assert CDF_ev(float('inf'), PDF, CDF) == pytest.approx(0.05)
