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
