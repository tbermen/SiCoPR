"""Verification tests for Raised_Cosine_Filter().

# ============================================================
# MATLAB GROUND TRUTH
# Thin wrapper: calls Tukey_Window(f, param, param.RC_Start, param.RC_end)
#
# use_RC=False → H = ones(1,N)
#
# use_RC=True, RC_Start=1e9, RC_end=3e9:
#   f=0   → H=1  (below RC_Start)
#   f=1e9 → H=1  (at RC_Start, pass edge)
#   f=2e9 → H=0.5 (midpoint)
#   f=3e9 → H=0  (at RC_end, stop edge)
#   f=5e9 → H=0  (above RC_end)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.Raised_Cosine_Filter.py_impl import Raised_Cosine_Filter


def make_param(start=1e9, end=3e9):
    return SimpleNamespace(RC_Start=start, RC_end=end)


def test_passthrough_when_disabled():
    """use_RC=False → all ones."""
    f = np.array([0.0, 1e9, 10e9])
    H = Raised_Cosine_Filter(make_param(), f, use_RC=False)
    np.testing.assert_array_equal(H, np.ones(3))


def test_passband_below_start():
    """f < RC_Start → H = 1."""
    f = np.array([0.0, 0.5e9])
    H = Raised_Cosine_Filter(make_param(), f, use_RC=True)
    np.testing.assert_allclose(H, 1.0, atol=1e-14)


def test_stopband_above_end():
    """f > RC_end → H = 0."""
    f = np.array([4e9, 10e9])
    H = Raised_Cosine_Filter(make_param(), f, use_RC=True)
    np.testing.assert_allclose(H, 0.0, atol=1e-14)


def test_midpoint_half():
    """At midpoint f = (RC_Start+RC_end)/2 → H = 0.5."""
    param = make_param(start=1e9, end=3e9)
    H = Raised_Cosine_Filter(param, np.array([2e9]), use_RC=True)
    assert H[0] == pytest.approx(0.5, rel=1e-10)


def test_output_length_matches_input():
    f = np.linspace(0, 5e9, 20)
    H = Raised_Cosine_Filter(make_param(), f, use_RC=True)
    assert len(H) == 20


# ============================================================
# COM Octave oracle values (2026-09-22)
# Raised_Cosine_Filter extracted verbatim from the reference and run under
# Octave with param.RC_Start=20e9, param.RC_end=40e9.  Third of the family
# after Butterworth_Filter and Bessel_Thomson_Filter, and it had both of the
# traps they had, plus the stale element-wise Tukey copy.
# ============================================================

OCT_F = np.array([0.0, 5e9, 10e9, 20e9, 25e9, 30e9, 35e9, 40e9, 45e9, 50e9])
OCT_H = np.array([1, 1, 1, 1, 0.85355339059327373, 0.49999999999999989,
                  0.14644660940672616, 0, 0, 0])


def oct_param():
    return SimpleNamespace(RC_Start=20e9, RC_end=40e9)


def test_oracle_values():
    """Pin the COM Octave response on a sorted axis."""
    H = Raised_Cosine_Filter(oct_param(), OCT_F, 1)
    np.testing.assert_allclose(H, OCT_H, rtol=1e-13, atol=0)


def test_use_rc_any_zero_is_false():
    """MATLAB `if [1 0]` is FALSE — every element must be non-zero.

    COM Octave: use_RC=[1 0] returned ones(1,10).  Python list/tuple
    truthiness called it true and ran the Tukey window.
    """
    for flag in (np.array([1, 0]), [1, 0], (1, 0)):
        H = Raised_Cosine_Filter(oct_param(), OCT_F, flag)
        np.testing.assert_array_equal(H, np.ones(10), err_msg=repr(flag))


def test_use_rc_empty_is_false():
    """MATLAB `if []` is false.  COM Octave: use_RC=[] returned ones(1,10)."""
    H = Raised_Cosine_Filter(oct_param(), OCT_F, np.array([]))
    np.testing.assert_array_equal(H, np.ones(10))


def test_use_rc_all_nonzero_is_true():
    """MATLAB `if [1 1]` is true.  COM Octave: same answer as use_RC=1."""
    H = Raised_Cosine_Filter(oct_param(), OCT_F, np.array([1, 1]))
    np.testing.assert_allclose(H, OCT_H, rtol=1e-13, atol=0)


def test_length_is_longest_dimension_not_first():
    """ones(1,length(f)) for a 3x5 f is FIVE ones, not three.

    COM Octave, f=reshape((0:14)*1e9,3,5), use_RC=0 -> 1x5 of ones.
    """
    f = (np.arange(15, dtype=float) * 1e9).reshape(3, 5, order='F')
    H = Raised_Cosine_Filter(oct_param(), f, 0)
    assert H.size == 5
    np.testing.assert_array_equal(H, np.ones(5))


def test_scalar_f_disabled_returns_one():
    """length() of a scalar is 1; len() raised TypeError on an unsized float."""
    H = Raised_Cosine_Filter(oct_param(), 40e9, 0)
    assert np.asarray(H).size == 1
    assert float(np.asarray(H).ravel()[0]) == 1.0


def test_unsorted_f_is_grouped_not_elementwise():
    """Tukey_Window concatenates three counted pieces, so an unsorted axis
    does NOT come back element-wise.

    COM Octave, f=[30e9 5e9 45e9 25e9], RC_Start=20e9, RC_end=40e9:
        [1, 0.49999999999999989, 0.85355339059327373, 0]
    The element-wise np.where copy that used to be inlined here returned
    [0.5, 1, 0, 0.85355339059327373].
    """
    f = np.array([30e9, 5e9, 45e9, 25e9])
    H = Raised_Cosine_Filter(oct_param(), f, 1)
    np.testing.assert_allclose(
        H, [1, 0.49999999999999989, 0.85355339059327373, 0],
        rtol=1e-13, atol=0)


def test_nan_in_f_is_an_out_of_bound_read():
    """A NaN lands in none of the three pieces, so H_tw comes up short and
    MATLAB's H_tw(1:length(f)) reads past the end.

    COM Octave, f=[0 25e9 NaN 45e9]: "H_tw(4): out of bound 3".
    The old copy answered 0 for the NaN.
    """
    with pytest.raises(IndexError):
        Raised_Cosine_Filter(oct_param(), np.array([0.0, 25e9, np.nan, 45e9]), 1)
