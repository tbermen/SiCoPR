"""Verification tests for applyDFEbk().

# ============================================================
# MATLAB GROUND TRUTH (0-based idx in Python)
# rng = idx:idx+tap_bk-1  (MATLAB 1-based)
# flt_curval_q = hisi[rng]  when dfe_delta==0
# tap_coef = min(|flt_curval_q/curval|, bmaxg) * sign(flt_curval_q)
# hisi[rng] -= curval * tap_coef
# hisi_ref[rng] = 0
#
# Example: hisi=[0,0.3,0.4,0.2,0], idx=1, tap_bk=2, curval=0.5, bmaxg=1.0
#   flt_curval = [0.3, 0.4]
#   tap_coef = min(|[0.3,0.4]/0.5|, 1)*sign = [0.6, 0.8]
#   hisi[1:3] = [0.3,0.4] - 0.5*[0.6,0.8] = [0.0, 0.0]
#
# With dfe_delta=0.2: quantize tap to nearest multiple of dfe_delta
#   flt_curval_q = floor(|0.3/0.5|/0.2)*0.2*sign(0.3)*0.5 = floor(0.6/0.2)*0.2*0.5 = 3*0.2*0.5=0.3
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.applyDFEbk.py_impl import applyDFEbk


def test_basic_no_delta():
    """dfe_delta=0: tap_coef = min(|hisi/curval|,bmaxg)*sign; hisi reduced."""
    hisi = np.array([0.0, 0.3, 0.4, 0.2, 0.0])
    hisi_ref = np.ones(5)
    hisi_out, tap_coef, hisi_ref_out = applyDFEbk(hisi, hisi_ref, 1, 2, 0.5, 1.0, 0)
    np.testing.assert_allclose(tap_coef, [0.6, 0.8], atol=1e-12)
    np.testing.assert_allclose(hisi_out[1:3], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(hisi_ref_out[1:3], [0.0, 0.0])


def test_bmaxg_clipping():
    """Tap value exceeds bmaxg → clipped to bmaxg * sign."""
    hisi = np.array([0.0, 1.0, 0.0])
    hisi_ref = np.ones(3)
    hisi_out, tap_coef, _ = applyDFEbk(hisi, hisi_ref, 1, 1, 0.5, 0.5, 0)
    # |1.0/0.5|=2, clipped to bmaxg=0.5 → tap_coef=0.5, hisi[1]-=0.5*0.5=0.25
    assert tap_coef[0] == pytest.approx(0.5)
    assert hisi_out[1] == pytest.approx(1.0 - 0.5 * 0.5)


def test_hisi_ref_zeroed():
    """hisi_ref is zeroed at rng positions."""
    hisi = np.array([1.0, 2.0, 3.0, 4.0])
    hisi_ref = np.ones(4)
    _, _, hisi_ref_out = applyDFEbk(hisi, hisi_ref, 0, 2, 1.0, 5.0, 0)
    np.testing.assert_allclose(hisi_ref_out[:2], [0.0, 0.0])
    np.testing.assert_allclose(hisi_ref_out[2:], [1.0, 1.0])


def test_outside_range_unchanged():
    """Taps outside rng are not modified."""
    hisi = np.array([1.0, 0.2, 0.3, 1.0])
    hisi_ref = np.ones(4)
    hisi_out, _, _ = applyDFEbk(hisi, hisi_ref, 1, 2, 0.5, 1.0, 0)
    assert hisi_out[0] == pytest.approx(1.0)
    assert hisi_out[3] == pytest.approx(1.0)


def test_dfe_delta_quantization():
    """dfe_delta != 0: tap is quantized to nearest dfe_delta multiple.

    hisi[1]=0.3, curval=0.5, dfe_delta=0.2:
      |0.3/0.5|/0.2 = 0.6/0.2; due to float repr floor gives 2 not 3
      flt_curval_q = 2 * 0.2 * 1 * 0.5 = 0.2
      tap_coef = min(0.2/0.5, 1.0) = 0.4
    """
    hisi = np.array([0.0, 0.3, 0.0])
    hisi_ref = np.zeros(3)
    hisi_out, tap_coef, _ = applyDFEbk(hisi, hisi_ref, 1, 1, 0.5, 1.0, 0.2)
    flt_q = np.floor(np.abs(0.3 / 0.5) / 0.2) * 0.2 * np.sign(0.3) * 0.5
    expected = min(abs(flt_q / 0.5), 1.0) * np.sign(flt_q)
    assert tap_coef[0] == pytest.approx(expected, rel=1e-8)
