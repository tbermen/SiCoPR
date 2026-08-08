"""Verification tests for d_cpdf().

# ============================================================
# MATLAB GROUND TRUTH
# Special case all(values==0):
#   pdf.BinSize=binsize, pdf.Min=0, pdf.y=1, pdf.x=0
#
# Symmetric 3-point: values=[-0.1,0,0.1], probs=[0.25,0.5,0.25], binsize=0.1
#   t = [-0.1, 0, 0.1]  (after snap, same)
#   pdf.Min = -1
#   k=0: bin=0 → pdf.y[0]+=0.25
#   k=1: argmin(|t-0|)=1 → pdf.y[1]+=0.5
#   k=2: bin=last=2 → pdf.y[2]+=0.25
#   pdf.y = [0.25,0.5,0.25] (sum=1, already normalized)
#   support = [0,1,2] → no trim
#   pdf.Min = -1+(0) = -1
#   pdf.x = (-1:1)*0.1 = [-0.1,0,0.1]
#
# Normalization: probs=[1,3] → pdf.y=[0.25,0,0.75] after normalize
#   (middle bin stays 0 since only first and last bin are populated)
#
# Unsorted input: same result as sorted (argsort applied)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.d_cpdf.py_impl import d_cpdf


def test_all_zeros_special_case():
    """all(values==0) → scalar y=1, x=0."""
    out = d_cpdf(0.1, np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.3, 0.2]))
    assert out.Min == 0
    assert out.BinSize == pytest.approx(0.1)
    np.testing.assert_allclose(out.y, [1.0])
    np.testing.assert_allclose(out.x, [0.0])


def test_symmetric_three_point():
    """values=[-0.1,0,0.1]: pdf.y=[0.25,0.5,0.25], pdf.x=[-0.1,0,0.1]."""
    out = d_cpdf(0.1, np.array([-0.1, 0.0, 0.1]), np.array([0.25, 0.5, 0.25]))
    np.testing.assert_allclose(out.y, [0.25, 0.5, 0.25], atol=1e-12)
    np.testing.assert_allclose(out.x, [-0.1, 0.0, 0.1], atol=1e-12)
    assert out.Min == -1


def test_normalization():
    """Unnormalized probs → pdf.y sums to 1."""
    out = d_cpdf(0.1, np.array([-0.1, 0.0, 0.1]), np.array([1.0, 2.0, 1.0]))
    assert np.sum(out.y) == pytest.approx(1.0, rel=1e-12)


def test_unsorted_same_as_sorted():
    """Unsorted values should give same result as sorted."""
    out_sorted = d_cpdf(0.1, np.array([-0.1, 0.0, 0.1]), np.array([0.25, 0.5, 0.25]))
    out_unsorted = d_cpdf(0.1, np.array([0.1, -0.1, 0.0]), np.array([0.25, 0.25, 0.5]))
    np.testing.assert_allclose(out_sorted.y, out_unsorted.y, atol=1e-12)


def test_binsize_stored():
    out = d_cpdf(0.05, np.array([-0.1, 0.0, 0.1]), np.array([1.0, 2.0, 1.0]))
    assert out.BinSize == pytest.approx(0.05)
