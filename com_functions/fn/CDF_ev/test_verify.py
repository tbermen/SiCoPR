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
