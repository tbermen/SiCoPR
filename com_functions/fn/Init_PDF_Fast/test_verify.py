"""Verification tests for Init_PDF_Fast().

# ============================================================
# MATLAB GROUND TRUTH
# EmptyPDF with BinSize=0.1; values=[-0.1,0,0.1]; probs=[0.25,0.5,0.25]
#   rvd = round([-0.1,0,0.1]/0.1) = [-1,0,1]
#   pdf.x = arange(-1,2)*0.1 = [-0.1,0,0.1]
#   pdf.Min = -1
#   bin_placement = [-1,0,1]-(-1) = [0,1,2]
#   pdf.y[0] = 0.25 (direct assign)
#   pdf.y[1] += 0.5 → 0.5
#   pdf.y[2] += 0.25 → 0.25
#   pdf.y = [0.25, 0.5, 0.25]
#
# No normalization; caller responsibility.
# Duplicate values → same bin: probabilities accumulate.
#   values=[0,0], probs=[0.3,0.7]: rvd=[0,0], bin_placement=[0,0]
#   pdf.y[0]=0.3 then pdf.y[0]+=0.7 → 1.0
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.Init_PDF_Fast.py_impl import Init_PDF_Fast


def make_empty(binsize=0.1):
    return SimpleNamespace(BinSize=binsize, Min=0,
                           y=np.array([0.0]), x=np.array([0.0]))


def test_three_point_symmetric():
    """values=[-0.1,0,0.1]: y=[0.25,0.5,0.25], x=[-0.1,0,0.1], Min=-1."""
    out = Init_PDF_Fast(make_empty(), np.array([-0.1, 0.0, 0.1]),
                        np.array([0.25, 0.5, 0.25]))
    np.testing.assert_allclose(out.y, [0.25, 0.5, 0.25], atol=1e-12)
    np.testing.assert_allclose(out.x, [-0.1, 0.0, 0.1], atol=1e-12)
    assert out.Min == -1


def test_no_normalization():
    """No normalization: sum(y) = sum(probs) if no duplicates."""
    out = Init_PDF_Fast(make_empty(), np.array([-0.1, 0.1]),
                        np.array([0.3, 0.7]))
    assert np.sum(out.y) == pytest.approx(1.0, rel=1e-12)


def test_duplicate_values_accumulate():
    """Duplicate values → same bin: probabilities accumulate to 1.0."""
    out = Init_PDF_Fast(make_empty(), np.array([0.0, 0.0]),
                        np.array([0.3, 0.7]))
    assert out.y[0] == pytest.approx(1.0, rel=1e-12)


def test_binsize_preserved():
    out = Init_PDF_Fast(make_empty(binsize=0.05),
                        np.array([-0.05, 0.0, 0.05]),
                        np.array([0.2, 0.6, 0.2]))
    assert out.BinSize == pytest.approx(0.05)


def test_x_length_matches_y():
    out = Init_PDF_Fast(make_empty(), np.array([-0.2, -0.1, 0.0, 0.1, 0.2]),
                        np.array([0.1, 0.2, 0.4, 0.2, 0.1]))
    assert len(out.x) == len(out.y)
