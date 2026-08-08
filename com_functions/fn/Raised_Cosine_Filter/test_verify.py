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
