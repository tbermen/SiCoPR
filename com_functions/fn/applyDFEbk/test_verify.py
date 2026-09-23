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


# ============================================================
# Divergences found by EXECUTING the reference (COM Octave oracle,
# tools/octave_oracle.py). Octave values pinned as literals below.
#
# 1. hisi(rng) is a READ, so MATLAB errors when the bank runs off either end;
#    a Python slice silently shortened and answered on the wrong bank.
# 2. hisi_ref(rng)=0 is an ASSIGNMENT, so MATLAB GROWS hisi_ref with zeros;
#    the Python slice assignment wrote nothing past the end.
# 3. MATLAB min() drops a NaN operand; np.minimum propagated it, so a NaN
#    bmaxg poisoned every tap instead of disabling the clip.
# 4. The reference guards dfe_delta with `if nargin<6`, but dfe_delta is the
#    SEVENTH argument, so seven is the only arity MATLAB accepts.
# ============================================================

def test_bank_past_end_raises():
    """COM Octave: hisi=[1 2 3 4 5 6], idx=5 (1-based), tap_bk=4 ->
    'hisi(8): out of bound 6'.  Python returned a 2-tap answer
    ([1 2 3 4 3 4], tap_coef [0.5 0.5])."""
    hisi = np.arange(1.0, 7.0)
    with pytest.raises(IndexError):
        applyDFEbk(hisi, np.full(6, 9.0), 4, 4, 4.0, 0.5, 0)


def test_negative_idx_raises():
    """COM Octave: idx=0 (1-based) ->
    'hisi(0): subscripts must be either integers 1 to (2^63)-1 or logicals'.
    Python's idx=-1 indexed from the end and answered."""
    hisi = np.arange(1.0, 7.0)
    with pytest.raises(IndexError):
        applyDFEbk(hisi, np.full(6, 9.0), -1, 2, 4.0, 0.5, 0)


def test_empty_bank_is_allowed_anywhere():
    """COM Octave: tap_bk=0 makes rng empty, which MATLAB accepts; hisi and
    hisi_ref come back untouched and tap_coef is 1x0."""
    hisi = np.arange(1.0, 7.0)
    h, tc, hr = applyDFEbk(hisi, np.full(6, 9.0), 1, 0, 4.0, 0.5, 0)
    np.testing.assert_array_equal(h, np.arange(1.0, 7.0))
    np.testing.assert_array_equal(hr, np.full(6, 9.0))
    assert len(tc) == 0


def test_hisi_ref_grows_when_shorter_than_bank():
    """COM Octave: hisi_ref=[9 9 9], idx=4 (1-based), tap_bk=3 ->
    hisi_ref becomes [9 9 9 0 0 0].  Python returned [9 9 9]."""
    h, tc, hr = applyDFEbk(np.arange(1.0, 7.0), np.array([9.0, 9.0, 9.0]),
                           3, 3, 4.0, 0.5, 0)
    np.testing.assert_array_equal(hr, [9.0, 9.0, 9.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(h, [1.0, 2.0, 3.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(tc, [0.5, 0.5, 0.5])


def test_hisi_ref_grows_from_empty():
    """COM Octave: hisi_ref=[], idx=2 (1-based), tap_bk=3 -> [0 0 0 0]."""
    _, _, hr = applyDFEbk(np.arange(1.0, 7.0), np.array([]), 1, 3, 4.0, 0.5, 0)
    np.testing.assert_array_equal(hr, [0.0, 0.0, 0.0, 0.0])


def test_nan_bmaxg_disables_the_clip():
    """COM Octave: bmaxg=NaN -> tap_coef [0.5 0.75 1] and hisi [1 0 0 0 5 6].
    np.minimum propagated the NaN and gave tap_coef [nan nan nan]."""
    h, tc, hr = applyDFEbk(np.arange(1.0, 7.0), np.full(6, 9.0),
                           1, 3, 4.0, float('nan'), 0)
    np.testing.assert_array_equal(tc, [0.5, 0.75, 1.0])
    np.testing.assert_array_equal(h, [1.0, 0.0, 0.0, 0.0, 5.0, 6.0])
    np.testing.assert_array_equal(hr, [9.0, 0.0, 0.0, 0.0, 9.0, 9.0])


def test_nan_bmaxg_with_quantization():
    """COM Octave: hisi=[1 -2.3 3.7 -0.9 5 6], curval=4, bmaxg=NaN,
    dfe_delta=0.05 -> tap_coef [-0.55 0.9 -0.2]."""
    h, tc, _ = applyDFEbk(np.array([1.0, -2.3, 3.7, -0.9, 5.0, 6.0]),
                          np.full(6, 9.0), 1, 3, 4.0, float('nan'), 0.05)
    np.testing.assert_array_equal(
        tc, [-0.55000000000000004, 0.90000000000000002, -0.20000000000000001])


def test_nan_inside_the_bank_still_yields_nan_tap():
    """COM Octave: hisi=[1 NaN 3 4 5 6] -> tap_coef [NaN 0.5 0.5].
    The NaN must survive the NaN-dropping min, because sign(NaN) is NaN."""
    _, tc, _ = applyDFEbk(np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0]),
                          np.full(6, 9.0), 1, 3, 4.0, 0.5, 0)
    assert np.isnan(tc[0])
    np.testing.assert_array_equal(tc[1:], [0.5, 0.5])


def test_seven_arguments_are_required():
    """COM Octave: applyDFEbk(hisi,href,2,3,4,0.5) -> "'dfe_delta' undefined".
    The reference's `if nargin<6` guard never fires for the 7th argument, so
    a 6-argument call is not something the reference can answer; Python used
    to supply dfe_delta=0 and answer anyway."""
    with pytest.raises(TypeError):
        applyDFEbk(np.arange(1.0, 7.0), np.full(6, 9.0), 1, 3, 4.0, 0.5)


def test_quantization_and_negative_curval_bit_exact():
    """COM Octave, hisi=[1 -2.3 3.7 -0.9 5 6], idx=2 (1-based), tap_bk=3:
      curval=4,  dfe_delta=0.05 -> hisi [1 -0.29999999999999982
                                         1.7000000000000002
                                         -0.099999999999999978 5 6]
      curval=-4, dfe_delta=0    -> tap_coef [-0.5 0.5 -0.22500000000000001]
    """
    h, tc, _ = applyDFEbk(np.array([1.0, -2.3, 3.7, -0.9, 5.0, 6.0]),
                          np.full(6, 9.0), 1, 3, 4.0, 0.5, 0.05)
    np.testing.assert_array_equal(
        h, [1, -0.29999999999999982, 1.7000000000000002,
            -0.099999999999999978, 5, 6])
    _, tc2, _ = applyDFEbk(np.array([1.0, -2.3, 3.7, -0.9, 5.0, 6.0]),
                           np.full(6, 9.0), 1, 3, -4.0, 0.5, 0)
    np.testing.assert_array_equal(tc2, [-0.5, 0.5, -0.22500000000000001])


def test_caller_arrays_are_not_modified():
    """MATLAB passes by value: applyDFEbk must not write the caller's arrays.

    The function reduces hisi and zeroes hisi_ref over the bank, and it does
    that on a .copy() of each. Drop either copy and the writes land on the
    array the CALLER still holds. That aliasing class accounted for five of the
    eight defects found in 2026-08, yet every test above reads only the RETURN
    values, so all of them pass with both copies removed.

    Pinned by tests/test_mutation_score.py: with this test in place the
    drop_dot_copy mutants at py_impl lines 18 and 19 are caught.
    """
    hisi = np.array([0.0, 0.3, 0.4, 0.2, 0.0])
    hisi_ref = np.ones(5)
    hisi_before = hisi.copy()
    hisi_ref_before = hisi_ref.copy()

    applyDFEbk(hisi, hisi_ref, 1, 2, 0.5, 1.0, 0)

    np.testing.assert_array_equal(
        hisi, hisi_before,
        err_msg='applyDFEbk modified the caller\'s hisi in place')
    np.testing.assert_array_equal(
        hisi_ref, hisi_ref_before,
        err_msg='applyDFEbk modified the caller\'s hisi_ref in place')
