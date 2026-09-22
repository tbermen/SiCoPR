"""Verification tests for H_interp().

# ============================================================
# MATLAB GROUND TRUTH (lines 2180-2198)
# mag_db_new = pchip(f_old, 20*log10(|S21|), f_new)
# ph_new = pchip(f_old, unwrap(angle(S21)), f_new)
# H_new = 10^(mag_db_new/20) * exp(j*ph_new)
# H_new(isinf(H_new)) = 0
# inq = last index where f_new <= fb/2
# H_new(inq+1:end) = 0
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.H_interp.py_impl import H_interp


def test_output_length():
    """Output length == len(f_new)."""
    f_old = np.linspace(0.1e9, 50e9, 50)
    S21 = np.ones(50)
    f_new = np.linspace(0.1e9, 50e9, 200)
    H = H_interp(S21, f_old, f_new, 50e9)
    assert len(H) == 200


def test_above_fb2_zeroed():
    """Points above fb/2 are zero."""
    fb = 20e9
    f_old = np.linspace(1e9, 30e9, 30)
    S21 = np.ones(30) + 0j
    f_new = np.linspace(1e9, 30e9, 60)
    H = H_interp(S21, f_old, f_new, fb)
    above = f_new > fb / 2
    np.testing.assert_allclose(H[above], 0.0, atol=1e-12)


def test_no_inf_in_output():
    """No infinite values in output."""
    f_old = np.linspace(1e9, 40e9, 40)
    S21 = np.ones(40) * 0.5
    f_new = np.linspace(1e9, 40e9, 80)
    H = H_interp(S21, f_old, f_new, 40e9)
    assert np.all(np.isfinite(np.abs(H)))


def test_unity_magnitude_preserved_below_fb2():
    """Flat unity magnitude → interpolated magnitude ≈ 1 below fb/2."""
    fb = 40e9
    f_old = np.linspace(1e9, 40e9, 40)
    S21 = np.ones(40) + 0j
    f_new = np.linspace(1e9, 40e9, 40)
    H = H_interp(S21, f_old, f_new, fb)
    below = f_new <= fb / 2
    np.testing.assert_allclose(np.abs(H[below]), 1.0, atol=1e-6)


def test_phase_linear_interpolated():
    """Linear phase S21 → output phase is also linear (pchip = linear for linear data)."""
    f_old = np.linspace(1e9, 20e9, 20)
    phase = -np.pi * f_old / 20e9
    S21 = np.exp(1j * phase)
    f_new = np.linspace(1e9, 20e9, 40)
    H = H_interp(S21, f_old, f_new, 40e9)
    expected_phase = -np.pi * f_new / 20e9
    below = f_new <= 20e9
    np.testing.assert_allclose(np.angle(H[below]) % (2 * np.pi),
                                expected_phase[below] % (2 * np.pi), atol=1e-5)


# ============================================================
# COM Octave oracle — H_interp extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m and run by tools/octave_oracle.py.
#   f_old = [1 5 10 20 30 40]e9
#   S21   = [1 .9 .7 .4 .2 .05] .* exp(1j*[0 -.5 -1.2 -2.5 -3.9 -5.4])
#   f_new = [2 7 15 25 35]e9
#
#   f_b=100e9 (nothing zeroed) ->
#       [0.97423226153237874-0.11744772561742448j,
#        0.58817510290352548-0.57758819096020542j,
#        -0.14788859181041519-0.51449156980994903j,
#        -0.29352240905039173+0.013479613151685488j,
#        -0.0083015232762606913+0.11032506580256483j]
#   f_b=1e6 AND f_b=0 -> the SAME five values.  Nothing qualifies for
#       find(f_new<=f_b/2,1,'last'), so inq is [], []+1:end is an empty index
#       list, and H_new(...) = 0 touches nothing.  Zeroing the whole vector
#       threw the result away.
#   f_b=40e9 -> the first three values, then 0, 0
#   f_new=[10 20 30]e9, f_b=40e9 -> [0.2536504281336715-0.65242736017705838j,
#       -0.32045744621877348-0.23938885764158255j, 0]  (f_b/2 itself is kept)
#   f_new=[25 5 30 7]e9, f_b=40e9 -> 'last' keeps index 4, so nothing is zeroed:
#       [-0.29352240905039173+0.013479613151685488j,
#        0.78982430570133555-0.43148298474378272j,
#        -0.14518646084002806+0.13755323183679477j,
#        0.58817510290352548-0.57758819096020542j]
#   f_old and S21 both REVERSED -> the ascending answer to ~4e-16: interp1
#       sorts its sample points, PchipInterpolator refused them
#   f_old with a repeated point -> error "discontinuities not supported for
#       METHOD 'pchip'";  a single point -> "minimum of 2 points required"
#
# Extrapolation: MATLAB's interp1 extrapolates for 'pchip' BY DEFAULT (that is
# what the "magnitude can explode and go to Inf" comment in the reference is
# about), while Octave's interp1 returns NaN unless 'extrap' is passed.  The
# values below come from the same statements with 'extrap' spelled out, i.e.
# MATLAB's default, and agree with this port to 6.2e-15 relative:
#   f_new=[0 .5e9 2e9 35e9 45e9 60e9], f_b=200e9 ->
#       [1.0053626086079435+0.11786957640864702j,
#        1.0052789443036036+0.059137579586922055j,
#        0.97423226153237874-0.11744772561742448j,
#        -0.0083015232762606913+0.11032506580256483j,
#        0.020037794661881235+0.0019362711371921336j,
#        -0.001148079361870877-0.0010598679299073685j]
#   f_new=[2e9 1e13], f_b=1e18 -> [0.97423226153237874-0.11744772561742448j, 0]
#       (the extrapolated magnitude overflows, and H_new(isinf(H_new))=0 fires)
# ============================================================

OCT_F_OLD = np.array([1e9, 5e9, 10e9, 20e9, 30e9, 40e9])
OCT_S21 = np.array([1.0, 0.9, 0.7, 0.4, 0.2, 0.05]) * np.exp(
    1j * np.array([0.0, -0.5, -1.2, -2.5, -3.9, -5.4]))
OCT_F_NEW = np.array([2e9, 7e9, 15e9, 25e9, 35e9])
OCT_H = np.array([0.97423226153237874 - 0.11744772561742448j,
                  0.58817510290352548 - 0.57758819096020542j,
                  -0.14788859181041519 - 0.51449156980994903j,
                  -0.29352240905039173 + 0.013479613151685488j,
                  -0.0083015232762606913 + 0.11032506580256483j])


def test_oracle_values():
    H = H_interp(OCT_S21, OCT_F_OLD, OCT_F_NEW, 100e9)
    np.testing.assert_allclose(H, OCT_H, rtol=1e-13, atol=0)


def test_nothing_qualifies_zeroes_nothing():
    """inq=[] makes H_new(inq+1:end)=0 an empty assignment, not a wipe."""
    for f_b in (1e6, 0.0, -10e9):
        H = H_interp(OCT_S21, OCT_F_OLD, OCT_F_NEW, f_b)
        np.testing.assert_allclose(H, OCT_H, rtol=1e-13, atol=0,
                                   err_msg='f_b=%g' % f_b)


def test_partial_zeroing_above_fb_over_2():
    H = H_interp(OCT_S21, OCT_F_OLD, OCT_F_NEW, 40e9)
    np.testing.assert_allclose(H[:3], OCT_H[:3], rtol=1e-13, atol=0)
    assert H[3] == 0 and H[4] == 0


def test_fb_over_2_itself_is_kept():
    H = H_interp(OCT_S21, OCT_F_OLD, np.array([10e9, 20e9, 30e9]), 40e9)
    np.testing.assert_allclose(
        H, [0.2536504281336715 - 0.65242736017705838j,
            -0.32045744621877348 - 0.23938885764158255j, 0], rtol=1e-13, atol=0)


def test_last_qualifying_index_wins_on_an_unsorted_axis():
    """find(...,1,'last') looks at the LAST index, not the first breach."""
    H = H_interp(OCT_S21, OCT_F_OLD, np.array([25e9, 5e9, 30e9, 7e9]), 40e9)
    np.testing.assert_allclose(
        H, [-0.29352240905039173 + 0.013479613151685488j,
            0.78982430570133555 - 0.43148298474378272j,
            -0.14518646084002806 + 0.13755323183679477j,
            0.58817510290352548 - 0.57758819096020542j], rtol=1e-13, atol=0)


def test_descending_f_old_is_sorted_like_interp1():
    """interp1 sorts its sample points; PchipInterpolator refused them."""
    H = H_interp(OCT_S21[::-1], OCT_F_OLD[::-1], OCT_F_NEW, 100e9)
    np.testing.assert_allclose(H, OCT_H, rtol=1e-12, atol=0)


def test_extrapolates_outside_f_old():
    """MATLAB's interp1 extrapolates for 'pchip' by default."""
    f_new = np.array([0.0, 0.5e9, 2e9, 35e9, 45e9, 60e9])
    H = H_interp(OCT_S21, OCT_F_OLD, f_new, 200e9)
    np.testing.assert_allclose(
        H, [1.0053626086079435 + 0.11786957640864702j,
            1.0052789443036036 + 0.059137579586922055j,
            0.97423226153237874 - 0.11744772561742448j,
            -0.0083015232762606913 + 0.11032506580256483j,
            0.020037794661881235 + 0.0019362711371921336j,
            -0.001148079361870877 - 0.0010598679299073685j], rtol=1e-13, atol=0)


def test_exploded_extrapolation_is_zeroed():
    """H_new(isinf(H_new))=0 catches the overflow the reference warns about."""
    H = H_interp(OCT_S21, OCT_F_OLD, np.array([2e9, 1e13]), 1e18)
    np.testing.assert_allclose(H[0], OCT_H[0], rtol=1e-13, atol=0)
    assert H[1] == 0


def test_repeated_or_single_sample_point_is_an_error():
    with pytest.raises(ValueError):
        H_interp(OCT_S21, np.array([1e9, 5e9, 10e9, 10e9, 30e9, 40e9]),
                 OCT_F_NEW, 100e9)
    with pytest.raises(ValueError):
        H_interp(OCT_S21[:1], OCT_F_OLD[:1], OCT_F_NEW, 100e9)
