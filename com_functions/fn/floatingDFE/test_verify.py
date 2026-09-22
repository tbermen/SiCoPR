"""Verification tests for floatingDFE().

# ============================================================
# MATLAB GROUND TRUTH
# [tap_loc, tap_coef, hisi, b] = floatingDFE(hisi, N_b, N_bf, N_bg, N_bmax, bmaxg, curval)
# tap_loc: sorted 0-based indices of applied floating taps
# tap_coef: full-length array with tap values at tap_loc
# hisi: modified (floating ISI subtracted)
# b: full-length array; b[tap_loc] = bmaxg
#
# b(tap_loc)=bmaxg always (independent of actual tap values)
# tap_loc is sorted ascending
# len(tap_coef) == len(hisi_in)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.floatingDFE.py_impl import floatingDFE


def test_returns_four_outputs():
    hisi = np.array([0.0, 0.3, 0.1, 0.5, 0.2, 0.0])
    tap_loc, tap_coef, hisi_out, b = floatingDFE(hisi, 1, 1, 1, 5, 0.8, 0.5)
    assert len(tap_loc) >= 1
    assert len(tap_coef) == len(hisi)
    assert len(hisi_out) == len(hisi)
    assert len(b) == len(hisi)


def test_tap_loc_sorted():
    """tap_loc is sorted ascending."""
    hisi = np.array([0.0, 0.4, 0.1, 0.5, 0.2, 0.3, 0.0])
    tap_loc, _, _, _ = floatingDFE(hisi, 1, 1, 2, 6, 0.8, 0.5)
    assert np.all(np.diff(tap_loc) >= 0)


def test_b_at_tap_loc_equals_bmaxg():
    """b[tap_loc] == bmaxg for all tap locations."""
    hisi = np.array([0.0, 0.3, 0.5, 0.2, 0.0, 0.0])
    bmaxg = 0.7
    tap_loc, _, _, b = floatingDFE(hisi, 1, 1, 1, 5, bmaxg, 0.5)
    np.testing.assert_allclose(b[tap_loc], bmaxg)


def test_tap_coef_at_non_tap_is_zero():
    """tap_coef is zero at all positions not in tap_loc."""
    hisi = np.array([0.0, 0.3, 0.5, 0.2, 0.1, 0.0])
    tap_loc, tap_coef, _, _ = floatingDFE(hisi, 1, 1, 1, 5, 0.8, 0.5)
    non_tap = np.setdiff1d(np.arange(len(hisi)), tap_loc)
    np.testing.assert_allclose(tap_coef[non_tap], 0.0)


def test_hisi_reduced_at_tap_loc():
    """hisi is modified (reduced) at tap_loc positions."""
    hisi = np.array([0.0, 0.0, 0.8, 0.0, 0.0])
    tap_loc, _, hisi_out, _ = floatingDFE(hisi, 1, 1, 1, 4, 0.9, 0.5)
    # At the chosen tap position, hisi_out should be smaller in magnitude
    assert abs(hisi_out[tap_loc[0]]) < abs(hisi[tap_loc[0]]) + 1e-12


def test_reference_config_params_valid_placement():
    """V-FDFE: with the reference config's actual floating-DFE params
    (N_b=12, N_bf=4, N_bg=3, N_bmax=40, bmaxg=0.1), the placement must be
    valid (count, in-range, non-overlapping) and land on the largest ISI banks.
    The default C2M run uses exactly these (Floating_DFE auto-on since N_bg>0)."""
    N_b, N_bf, N_bg, N_bmax, bmaxg, curval = 12, 4, 3, 40, 0.1, 1.0
    rng = np.random.default_rng(0)
    hisi = np.abs(rng.normal(0, 0.02, 50))          # small background ISI, len > N_bmax
    # plant three strong 4-tap clusters inside the floating region [N_b, N_bmax-1]
    for start, amp in [(14, 0.40), (20, 0.50), (30, 0.45)]:
        hisi[start:start + N_bf] = amp
    tap_loc, tap_coef, hisi_out, b = floatingDFE(hisi, N_b, N_bf, N_bg, N_bmax, bmaxg, curval)

    assert len(tap_loc) == N_bg * N_bf                       # 12 taps placed
    assert np.all(tap_loc >= N_b) and np.all(tap_loc <= N_bmax - 1)   # within floating span
    assert len(np.unique(tap_loc)) == len(tap_loc)          # no overlap
    assert np.all(np.diff(tap_loc) >= 0)                    # sorted
    # the three strongest banks must be covered
    for start in (14, 20, 30):
        assert set(range(start, start + N_bf)).issubset(set(tap_loc.tolist())), start
    # coefficients clamped to bmaxg and ISI reduced there
    np.testing.assert_allclose(np.abs(tap_coef[tap_loc]), bmaxg)
    np.testing.assert_allclose(b[tap_loc], bmaxg)
    assert np.all(np.abs(hisi_out[tap_loc]) < np.abs(hisi[tap_loc]))


# ============================================================
# COM Octave 4p16p0 oracle pins.
# floatingDFE and findbankloc extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m (both byte-identical to
# matlab/com_ieee8023_4p16p0.m) and run under Octave on the inputs below.
# Octave's tap_loc is 1-based; every _OCT_LOC here is Octave's minus one.
#
# The inputs are chosen for EXACT TIES, because findbankloc sorts ndiff and
# an ISI tail is mostly zeros: MATLAB sort keeps equal elements in their
# original order, and np.sort does not unless kind='stable' is asked for.
# Octave and the port agreed bit for bit on tap_loc, tap_coef, hisi and b
# across these cases and a further 40 randomised quantised-ISI trials, so no
# tie-breaking divergence was found.
# ============================================================

_BMAXG = 0.1

# two identical clusters: ndiff ties exactly between them
_TIE_HISI = [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0]
_TIE_LOC = [3, 4, 8, 9]
_TIE_COEF = [0, 0, 0, 0.10000000000000001, 0.10000000000000001, 0, 0, 0,
             0.10000000000000001, 0.10000000000000001, 0, 0, 0, 0, 0, 0]
_TIE_HISI_OUT = [0, 0, 0, 0.40000000000000002, 0.40000000000000002, 0, 0, 0,
                 0.40000000000000002, 0.40000000000000002, 0, 0, 0, 0, 0, 0]

# a single strong bank at the far end: the bank straddles the last taps, so
# most of its coefficients are zero while b is bmaxg across the whole bank
_END_HISI = [0.0] * 16 + [0.9] * 4
_END_LOC = [2, 3, 4, 5, 13, 14, 15, 16]
_END_COEF = [0] * 16 + [0.10000000000000001, 0, 0, 0]
_END_HISI_OUT = [0] * 16 + [0.80000000000000004, 0.9, 0.9, 0.9]

# dfe_delta quantisation: floor(|h/curval|/delta)*delta*sign(h)*curval
_Q_HISI = [0.0, 0.37, 0.11, 0.53, 0.22, 0.0, 0.0, 0.0]
_Q_LOC = [1, 2, 3, 4]
_Q_COEF = [0, 0.35000000000000003, 0.10000000000000001, 0.5,
           0.20000000000000001, 0, 0, 0]
_Q_HISI_OUT = [0, 0.019999999999999962, 0.009999999999999995,
               0.030000000000000027, 0.01999999999999999, 0, 0, 0]

# mixed signs with curval 0.5: the coefficient is clamped to bmaxg in
# magnitude and carries sign(h)
_SGN_HISI = [0.0, -0.4, 0.1, -0.5, 0.2, 0.3, 0.0, 0.0]
_SGN_LOC = [1, 2, 3, 4]
_SGN_COEF = [0, -0.2, 0.2, -0.2, 0.2, 0, 0, 0]
_SGN_HISI_OUT = [0, -0.30000000000000004, 0, -0.40000000000000002,
                 0.10000000000000001, 0.29999999999999999, 0, 0]


def test_octave_tied_banks_pick_the_earlier_pair():
    """COM Octave: with two identical clusters and two groups wanted, ndiff
    ties and the sort keeps the earlier start first."""
    tap_loc, tap_coef, hisi_out, b = floatingDFE(
        np.array(_TIE_HISI), 2, 2, 2, 16, _BMAXG, 1.0)
    np.testing.assert_array_equal(tap_loc, _TIE_LOC)
    np.testing.assert_allclose(tap_coef, _TIE_COEF, rtol=1e-15)
    np.testing.assert_allclose(hisi_out, _TIE_HISI_OUT, rtol=1e-15)
    np.testing.assert_allclose(b[tap_loc], _BMAXG, rtol=1e-15)
    assert np.count_nonzero(b) == len(_TIE_LOC)


def test_octave_bank_straddling_the_end_of_the_isi():
    """COM Octave: banks are chosen by their START position, so a bank may run
    past the last non-zero ISI sample.  b is bmaxg across the whole bank while
    tap_coef is non-zero only where the ISI is."""
    tap_loc, tap_coef, hisi_out, b = floatingDFE(
        np.array(_END_HISI), 2, 4, 2, 20, _BMAXG, 1.0)
    np.testing.assert_array_equal(tap_loc, _END_LOC)
    np.testing.assert_allclose(tap_coef, _END_COEF, rtol=1e-15)
    np.testing.assert_allclose(hisi_out, _END_HISI_OUT, rtol=1e-15)
    np.testing.assert_allclose(b[tap_loc], _BMAXG, rtol=1e-15)


def test_octave_dfe_delta_quantisation():
    """COM Octave, dfe_delta=0.05: the coefficient is floored onto the delta
    grid before the bmaxg clamp, so 0.37 becomes 0.35 and 0.53 becomes 0.50."""
    tap_loc, tap_coef, hisi_out, _ = floatingDFE(
        np.array(_Q_HISI), 1, 2, 2, 8, 0.8, 1.0, dfe_delta=0.05)
    np.testing.assert_array_equal(tap_loc, _Q_LOC)
    np.testing.assert_allclose(tap_coef, _Q_COEF, rtol=1e-14)
    np.testing.assert_allclose(hisi_out, _Q_HISI_OUT, rtol=1e-12, atol=1e-18)


def test_octave_sign_and_clamp_with_a_non_unit_cursor():
    """COM Octave, curval=0.5 and bmaxg=0.2: applied_coef is
    min(|h/curval|, bmaxg)*sign(h) and hisi loses curval*applied_coef."""
    tap_loc, tap_coef, hisi_out, b = floatingDFE(
        np.array(_SGN_HISI), 1, 2, 2, 8, 0.2, 0.5)
    np.testing.assert_array_equal(tap_loc, _SGN_LOC)
    np.testing.assert_allclose(tap_coef, _SGN_COEF, rtol=1e-14)
    np.testing.assert_allclose(hisi_out, _SGN_HISI_OUT, rtol=1e-14,
                               atol=1e-18)
    # b carries bmaxg at every tap regardless of the sign of the coefficient
    np.testing.assert_allclose(b[tap_loc], 0.2, rtol=1e-15)


def test_octave_ties_with_opposite_signs():
    """COM Octave: ndiff is built from |hisi|, so two clusters of equal
    magnitude and opposite sign tie, and the coefficients carry the signs."""
    hisi = np.array([0.0, 0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0])
    tap_loc, tap_coef, _, _ = floatingDFE(hisi, 1, 2, 2, 10, _BMAXG, 1.0)
    np.testing.assert_array_equal(tap_loc, [1, 2, 4, 5])
    np.testing.assert_allclose(tap_coef[[1, 2, 4, 5]],
                               [0.1, -0.1, 0.1, -0.1], rtol=1e-14)


def test_octave_reference_config_tap_locations():
    """COM Octave, the reference config's floating-DFE parameters
    (N_b=12, N_bf=4, N_bg=3, N_bmax=40, bmaxg=0.1) on three planted banks."""
    rng = np.random.default_rng(0)
    hisi = np.abs(rng.normal(0, 0.02, 50))
    for start, amp in [(14, 0.40), (20, 0.50), (30, 0.45)]:
        hisi[start:start + 4] = amp
    tap_loc, _, _, _ = floatingDFE(hisi, 12, 4, 3, 40, 0.1, 1.0)
    np.testing.assert_array_equal(
        tap_loc, [14, 15, 16, 17, 20, 21, 22, 23, 30, 31, 32, 33])


def test_octave_tap_loc_is_sorted_after_the_strongest_bank_is_later():
    """COM Octave: findbankloc returns the groups strongest-first, so with the
    strong bank at 10 and the weaker at 3 the raw order is [10 11 4 5] (Octave
    1-based) and the closing sort(...,'ascend') puts it back in index order."""
    hisi = np.zeros(16)
    hisi[10] = hisi[11] = 0.9
    hisi[3] = hisi[4] = 0.5
    tap_loc, tap_coef, hisi_out, b = floatingDFE(hisi, 2, 2, 2, 16, 0.1, 1.0)
    np.testing.assert_array_equal(tap_loc, [3, 4, 10, 11])
    np.testing.assert_allclose(tap_coef[[3, 4, 10, 11]], 0.1, rtol=1e-15)
    np.testing.assert_allclose(hisi_out[[3, 4, 10, 11]],
                               [0.4, 0.4, 0.8, 0.8], rtol=1e-14)
    np.testing.assert_allclose(b[[3, 4, 10, 11]], 0.1, rtol=1e-15)


def test_octave_first_floating_tap_starts_after_the_fixed_taps():
    """COM Octave: findbankloc is given idx_st = N_b+1, so the bank may not
    start on a fixed tap.  With N_b=4 and ISI at samples 3 and 4, the bank
    starts at 4 and sample 3 keeps its 0.9 -- passing N_b would let the search
    start at 3 and take both."""
    hisi = np.zeros(14)
    hisi[3] = hisi[4] = 0.9
    tap_loc, tap_coef, hisi_out, _ = floatingDFE(hisi, 4, 2, 1, 14, 0.1, 1.0)
    np.testing.assert_array_equal(tap_loc, [4, 5])
    np.testing.assert_allclose(tap_coef[4], 0.1, rtol=1e-15)
    assert tap_coef[3] == 0.0
    np.testing.assert_allclose(hisi_out[[3, 4]], [0.9, 0.8], rtol=1e-14)
