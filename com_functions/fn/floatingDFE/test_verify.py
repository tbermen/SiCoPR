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
