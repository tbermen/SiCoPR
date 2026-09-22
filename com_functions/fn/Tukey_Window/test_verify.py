"""Verification tests for Tukey_Window().

# ============================================================
# MATLAB GROUND TRUTH
# Three-region raised-cosine window with pass edge fr, stop edge fb:
#   f < fr  → H = 1
#   fr≤f≤fb → H = 0.5*cos(2π*(f-fb)/(2*(fb-fr)) - π) + 0.5
#   f > fb  → H = 0
#
# Boundary values (fr=1e9, fb=3e9, fperiod=4e9):
#   f=0   (below fr): H = 1
#   f=fr  (border): cos(2π*(fr-fb)/fperiod - π) = cos(-2π) = 1 → H=1
#   f=(fr+fb)/2 (midpoint, f=2e9): cos(2π*(-1/4) - π) = cos(-3π/2) = 0 → H=0.5
#   f=fb  (stop edge): cos(-π) = -1 → H=0
#   f=4e9 (above fb): H=0
#
# Optional args: fr=None, fb=None → use param.f_r*param.fb and param.fb
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.Tukey_Window.py_impl import Tukey_Window


def test_passband_below_fr():
    """f < fr → H = 1."""
    param = SimpleNamespace(fb=3e9, f_r=1/3)  # fr=1e9
    f = np.array([0.0, 0.5e9, 0.99e9])
    H = Tukey_Window(f, param, fr=1e9, fb=3e9)
    np.testing.assert_allclose(H, 1.0, atol=1e-14)


def test_stopband_above_fb():
    """f > fb → H = 0."""
    f = np.array([3.01e9, 4e9, 10e9])
    H = Tukey_Window(f, None, fr=1e9, fb=3e9)
    np.testing.assert_allclose(H, 0.0, atol=1e-14)


def test_at_fr_equals_one():
    """At f = fr exactly, H = 1 (pass edge)."""
    H = Tukey_Window(np.array([1e9]), None, fr=1e9, fb=3e9)
    assert H[0] == pytest.approx(1.0, abs=1e-14)


def test_at_fb_equals_zero():
    """At f = fb exactly, H = 0 (stop edge)."""
    H = Tukey_Window(np.array([3e9]), None, fr=1e9, fb=3e9)
    assert H[0] == pytest.approx(0.0, abs=1e-14)


def test_midpoint_equals_half():
    """At midpoint f = (fr+fb)/2 = 2e9, H = 0.5."""
    H = Tukey_Window(np.array([2e9]), None, fr=1e9, fb=3e9)
    assert H[0] == pytest.approx(0.5, rel=1e-10)


def test_default_params():
    """fr=None, fb=None → uses param.f_r*param.fb and param.fb."""
    param = SimpleNamespace(fb=4e9, f_r=0.25)   # fr=1e9, fb=4e9
    f = np.array([0.0, 1e9, 2.5e9, 4e9, 5e9])
    H = Tukey_Window(f, param)
    assert H[0] == pytest.approx(1.0)     # below fr
    assert H[1] == pytest.approx(1.0)     # at fr
    assert H[3] == pytest.approx(0.0)     # at fb
    assert H[4] == pytest.approx(0.0)     # above fb


def test_output_length_matches_input():
    f = np.linspace(0, 5e9, 30)
    H = Tukey_Window(f, None, fr=1e9, fb=4e9)
    assert len(H) == 30


# ============================================================
# COM Octave oracle — Tukey_Window extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m and run by tools/octave_oracle.py.
# MATLAB builds H_tw by CONCATENATING three counted pieces, so the answer is
# grouped by category and only lines up with f while f ascends.  fr=1e9, fb=3e9
# unless noted:
#
#   f=[0 .5 1 1.5 2 2.5 3 3.5]e9 -> [1 1 1 0.85355339059327373
#                                    0.49999999999999989 0.14644660940672616 0 0]
#   f=[2.5 0 3.5 1.5]e9          -> [1 0.14644660940672616
#                                    0.85355339059327373 0]
#        (element-wise would be    [0.14644660940672616 1 0 0.85355339059327373])
#   f=[1 1 3 3 0 9]e9            -> [1 1 1 0 0 0]
#        (element-wise would be    [1 1 0 0 1 0])
#   f=[0 1.5 NaN 3.5]e9          -> error "H_tw(4): out of bound 3": a NaN
#        falls in no category, so the pieces are short and H_tw(1:length(f))
#        reads past the end
#   COLUMN f, 5 values in band   -> error "horizontal dimensions mismatch
#        (1x2 vs 5x1)": only the middle piece keeps f's orientation
#   COLUMN f, <=1 value in band  -> answers, as a row
#   f=2e9 scalar                 -> 0.49999999999999989
#   f=[] -> []   f=[-Inf 2e9 Inf] -> [1 0.49999999999999989 0]
#   fr=fb=2e9, f=[0 2e9 3e9]     -> [1 NaN 0]   (fperiod is 0)
#   fr=3e9 > fb=1e9, f=[0 2 4]e9 -> [1 1 0]: the categories overlap, the pieces
#        run long, and H_tw(1:length(f)) truncates
# ============================================================

RC15, RC20, RC25 = (0.85355339059327373, 0.49999999999999989, 0.14644660940672616)


def test_oracle_ascending_axis():
    f = np.array([0.0, 0.5e9, 1.0e9, 1.5e9, 2.0e9, 2.5e9, 3.0e9, 3.5e9])
    H = Tukey_Window(f, None, fr=1e9, fb=3e9)
    np.testing.assert_allclose(H, [1, 1, 1, RC15, RC20, RC25, 0, 0],
                               rtol=1e-15, atol=0)


def test_unsorted_axis_is_grouped_not_element_wise():
    """The concatenation is by category count, not by position."""
    H = Tukey_Window(np.array([2.5e9, 0.0, 3.5e9, 1.5e9]), None, fr=1e9, fb=3e9)
    np.testing.assert_allclose(H, [1, RC25, RC15, 0], rtol=1e-15, atol=0)


def test_ties_at_both_edges_are_grouped():
    """Duplicated edge values with one late low value: [1 1 1 0 0 0]."""
    H = Tukey_Window(np.array([1e9, 1e9, 3e9, 3e9, 0.0, 9e9]), None, fr=1e9, fb=3e9)
    np.testing.assert_allclose(H, [1, 1, 1, 0, 0, 0], rtol=1e-15, atol=0)


def test_nan_in_f_is_out_of_bound():
    """A NaN is in no category, so MATLAB's H_tw(1:length(f)) errors."""
    with pytest.raises(IndexError):
        Tukey_Window(np.array([0.0, 1.5e9, np.nan, 3.5e9]), None, fr=1e9, fb=3e9)


def test_column_f_is_a_concatenation_error():
    """[ones(1,2), 5x1, zeros(1,1)] is a horizontal dimension mismatch."""
    f = np.array([0.0, 0.5e9, 1.0e9, 1.5e9, 2.0e9, 2.5e9, 3.0e9, 3.5e9])
    with pytest.raises(ValueError):
        Tukey_Window(f.reshape(-1, 1), None, fr=1e9, fb=3e9)
    with pytest.raises(ValueError):
        Tukey_Window(np.array([[0.0, 2.0e9, 4.0e9], [0.5e9, 2.5e9, 5e9]]),
                     None, fr=1e9, fb=3e9)


def test_column_f_with_one_in_band_value_is_legal():
    """A 1x1 middle piece concatenates, so MATLAB answers these two."""
    H = Tukey_Window(np.array([[0.0], [2.0e9], [4.0e9]]), None, fr=1e9, fb=3e9)
    np.testing.assert_allclose(H, [1, RC20, 0], rtol=1e-15, atol=0)
    H = Tukey_Window(np.array([[0.0], [0.5e9], [4.0e9]]), None, fr=1e9, fb=3e9)
    np.testing.assert_allclose(H, [1, 1, 0], rtol=1e-15, atol=0)


def test_column_f_all_in_band_is_legal():
    """With both outer pieces empty there is nothing to mismatch."""
    f = np.array([[1.0e9], [1.5e9], [2.0e9], [2.5e9], [3.0e9]])
    H = Tukey_Window(f, None, fr=1e9, fb=3e9)
    np.testing.assert_allclose(H.ravel(), [1, RC15, RC20, RC25, 0],
                               rtol=1e-15, atol=0)


def test_scalar_f():
    H = Tukey_Window(2e9, None, fr=1e9, fb=3e9)
    assert np.asarray(H).size == 1
    assert float(np.asarray(H).ravel()[0]) == pytest.approx(RC20, rel=1e-15)


def test_empty_f():
    assert Tukey_Window(np.array([]), None, fr=1e9, fb=3e9).size == 0


def test_infinite_f_values():
    H = Tukey_Window(np.array([-np.inf, 2e9, np.inf]), None, fr=1e9, fb=3e9)
    np.testing.assert_allclose(H, [1, RC20, 0], rtol=1e-15, atol=0)


def test_zero_width_window_is_nan_in_band():
    H = Tukey_Window(np.array([0.0, 2e9, 3e9]), None, fr=2e9, fb=2e9)
    assert H[0] == 1.0
    assert np.isnan(H[1])
    assert H[2] == 0.0


def test_reversed_edges_truncate():
    """fr>fb makes the categories overlap; H_tw(1:length(f)) cuts it back."""
    H = Tukey_Window(np.array([0.0, 2e9, 4e9]), None, fr=3e9, fb=1e9)
    np.testing.assert_allclose(H, [1, 1, 0], rtol=1e-15, atol=0)
