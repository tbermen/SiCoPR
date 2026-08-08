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
