"""Verification tests for find_eye_width().

# ============================================================
# MATLAB GROUND TRUTH (lines 5723-5785)
# Left_EW = min(L1, L0), Right_EW = min(R1, R0).
# Open eye: EW = half_UI (left) or samples_per_UI-half_UI (right).
# Closed eye: EW = 0.
# Interpolation uses vref_intersect for normal case.
# ============================================================
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.find_eye_width.py_impl import find_eye_width


def _open_eye(N=32, half_UI=16):
    """Eye contour where top is always above vref and bottom always below."""
    ec = np.zeros((N, 2))
    ec[:, 0] = 1.0   # top eye always above vref=0
    ec[:, 1] = -1.0  # bottom eye always below vref=0
    return ec


def _closed_eye(N=32, half_UI=16):
    """Eye contour where top is always below vref (closed eye)."""
    ec = np.zeros((N, 2))
    ec[:, 0] = -1.0  # top below vref=0 → closed
    ec[:, 1] = 1.0   # bottom above vref=0 → closed
    return ec


def test_open_eye_left():
    """Fully open eye returns Left_EW = half_UI."""
    half_UI = 16
    L, R = find_eye_width(_open_eye(32, half_UI), half_UI, 32, 0.0)
    assert L == half_UI


def test_open_eye_right():
    """Fully open eye returns Right_EW = samples_per_UI - half_UI."""
    half_UI = 16
    samples_per_UI = 32
    L, R = find_eye_width(_open_eye(samples_per_UI, half_UI), half_UI, samples_per_UI, 0.0)
    assert R == samples_per_UI - half_UI


def test_closed_eye_returns_zero():
    """Fully closed eye → Left_EW = 0 and Right_EW = 0."""
    L, R = find_eye_width(_closed_eye(32, 16), 16, 32, 0.0)
    assert L == 0
    assert R == 0


def test_returns_two_values():
    """Function returns a 2-tuple."""
    result = find_eye_width(_open_eye(), 16, 32, 0.0)
    assert len(result) == 2


def test_symmetric_eye():
    """Symmetric eye → Left_EW == Right_EW."""
    N = 32
    half_UI = 16
    ec = np.zeros((N, 2))
    # Linear ramp from +1 down to -1 and back (symmetric crossing)
    ec[:, 0] = np.concatenate([np.linspace(1, -1, half_UI + 1), np.linspace(-1, 1, N - half_UI - 1)])
    ec[:, 1] = -ec[:, 0]
    L, R = find_eye_width(ec, half_UI, N, 0.0)
    assert abs(L - R) <= 2  # symmetric to within 2 samples


# ---------------------------------------------------------------------------
# Against COM Octave. The assertions above check ordering and bounds; these
# pin the interpolated crossing, which is the part that can be quietly wrong.
#
# Note the convention: find_eye_width takes half_UI as MATLAB gives it, 1-based
# (17 for 32 samples per UI), unlike get_center_of_UI which returns 0-based.
# ---------------------------------------------------------------------------

_EW_M = 32
_EW_HALF = _EW_M // 2 + 1
_OCT_LEFT = 9.6918426845953825
_OCT_RIGHT = 8.6918426845953789


def _contour():
    """A partly open eye: the top contour crosses vref inside the UI on both
    sides, so the interpolating branch runs and not the open/closed shortcuts."""
    t = np.linspace(-1.0, 1.0, _EW_M)
    top = 0.25 * np.cos(np.pi * t / 1.25) - 0.02
    bot = -0.25 * np.cos(np.pi * t / 1.25) + 0.02
    return np.column_stack([top, bot])


def test_matches_com_octave():
    L, R = find_eye_width(_contour(), _EW_HALF, _EW_M, 0.0)
    for name, got, want in (('Left_EW', float(L), _OCT_LEFT),
                            ('Right_EW', float(R), _OCT_RIGHT)):
        assert abs(got - want) <= 1e-13, (
            '%s is %.17g, COM Octave gives %.17g' % (name, got, want))


def test_interpolating_branch_is_the_one_exercised():
    """Guard the guard: a fully open or fully closed eye takes a shortcut that
    returns an integer, and would pin nothing about the interpolation."""
    L, R = find_eye_width(_contour(), _EW_HALF, _EW_M, 0.0)
    for v in (float(L), float(R)):
        assert abs(v - round(v)) > 1e-6, (
            'eye width %r is an integer, so the vref interpolation did not run'
            % v)


def test_open_and_closed_eyes_take_the_shortcuts():
    """The two special cases, values from COM Octave: a contour that never
    crosses vref gives (17, 15) for 32 samples per UI, and one that never
    clears it gives (0, 0)."""
    open_eye = np.column_stack([np.full(_EW_M, 0.5), np.full(_EW_M, -0.5)])
    L, R = find_eye_width(open_eye, _EW_HALF, _EW_M, 0.0)
    assert (float(L), float(R)) == (17.0, 15.0), (
        'a fully open eye gave L=%r R=%r; COM Octave gives 17 and 15'
        % (float(L), float(R)))

    closed = np.column_stack([np.full(_EW_M, -0.5), np.full(_EW_M, 0.5)])
    L, R = find_eye_width(closed, _EW_HALF, _EW_M, 0.0)
    assert (float(L), float(R)) == (0.0, 0.0), (
        'a fully closed eye gave L=%r R=%r; COM Octave gives 0 and 0'
        % (float(L), float(R)))
