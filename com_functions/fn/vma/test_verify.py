"""Verification tests for vma().

# ============================================================
# MATLAB GROUND TRUTH (lines 11337-11365)
# Uses PRBS13Q sequence to find patterns: 7 consecutive 3s and 6 consecutive 0s.
# PR is upsampled, convolved with PRBS signal, then P_3/P_0 measured.
# VMA = P_3 - P_0
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.vma.py_impl import vma


def test_returns_namespace_with_fields():
    """Returns SimpleNamespace with P_3, P_0, VMA."""
    M = 4
    N = 50000
    PR = np.zeros(N)
    PR[0] = 1.0  # impulse at 0
    result = vma(PR, M)
    assert hasattr(result, 'P_3')
    assert hasattr(result, 'P_0')
    assert hasattr(result, 'VMA')


def test_vma_equals_p3_minus_p0():
    """VMA = P_3 - P_0."""
    M = 4
    N = 50000
    PR = np.zeros(N)
    PR[0] = 1.0
    result = vma(PR, M)
    assert result.VMA == pytest.approx(result.P_3 - result.P_0)


def test_output_is_finite():
    """P_3, P_0, VMA are all finite."""
    M = 4
    N = 50000
    PR = np.zeros(N)
    PR[0] = 1.0
    result = vma(PR, M)
    assert np.isfinite(result.VMA)
    assert np.isfinite(result.P_3)
    assert np.isfinite(result.P_0)


def test_p3_greater_p0_for_unity_pulse():
    """For a unit impulse response, P_3 > P_0 (higher PAM level → higher mean)."""
    M = 4
    N = 50000
    PR = np.zeros(N)
    PR[0] = 1.0
    result = vma(PR, M)
    assert result.P_3 > result.P_0


# ============================================================
# COM Octave 4p16p0 oracle pins.
# vma, PRBS13Q, pam and LFSR extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m (vma byte-identical to
# matlab/com_ieee8023_4p16p0.m) and run under Octave.
#
# One substitution was needed, and it is not in vma: Octave's strfind refuses
# a numeric PATTERN, where MATLAB's accepts numeric arrays.  The probe
# supplied a strfind.m with the MATLAB semantics (return the 1-based start of
# every occurrence of the pattern in the vector), so
# strfind(syms, ones(1,7)*3) works as it does in MATLAB and the rest of vma
# ran unmodified.
#
# Fixture pulse: n = 0..63,
#   PR = exp(-((n-8)/3).^2) + 0.2*exp(-((n-16)/5).^2).
# ============================================================

_N64 = np.arange(64)
_PULSE = (np.exp(-((_N64 - 8.0) / 3.0) ** 2)
          + 0.2 * np.exp(-((_N64 - 16.0) / 5.0) ** 2))


def _impulse(N=64, at=0):
    pr = np.zeros(N)
    pr[at] = 1.0
    return pr


def test_octave_unit_impulse_M4():
    """COM Octave, a unit impulse at sample 1 with M=4: the bit-stream
    response is the PRBS13Q symbol sequence itself, so P_3 and P_0 are the
    mean of 2M+1 samples spanning one symbol either side of the run centre."""
    r = vma(_impulse(), 4)
    assert r.P_3 == pytest.approx(0.33333333333333331, rel=1e-14)
    assert r.P_0 == pytest.approx(-0.22222222222222221, rel=1e-14)
    assert r.VMA == pytest.approx(0.55555555555555558, rel=1e-14)


def test_octave_smooth_pulse_M4():
    """COM Octave, the fixture pulse at M=4."""
    r = vma(_PULSE, 4)
    assert r.P_3 == pytest.approx(1.7313856773680385, rel=1e-13)
    assert r.P_0 == pytest.approx(-1.7008626223418035, rel=1e-13)
    assert r.VMA == pytest.approx(3.4322482997098422, rel=1e-13)


def test_octave_smooth_pulse_M8_and_M32():
    """COM Octave: the window is 2M+1 wide and the run centres move with M."""
    r8 = vma(_PULSE, 8)
    assert r8.P_3 == pytest.approx(0.90658242435219871, rel=1e-13)
    assert r8.P_0 == pytest.approx(-0.90039098452487221, rel=1e-13)
    assert r8.VMA == pytest.approx(1.806973408877071, rel=1e-13)
    r32 = vma(_PULSE, 32)          # the production samples_per_ui
    assert r32.P_3 == pytest.approx(0.23376625968918108, rel=1e-13)
    assert r32.P_0 == pytest.approx(-0.23203104657925827, rel=1e-13)
    assert r32.VMA == pytest.approx(0.46579730626843935, rel=1e-13)


def test_octave_window_follows_the_pulse_peak():
    """COM Octave: every index is offset by imaxPR, so moving the peak of PR
    moves the measurement window with it and the answer does not change."""
    early = vma(_impulse(at=0), 4)
    late = vma(_impulse(at=60), 4)
    assert late.P_3 == pytest.approx(early.P_3, rel=1e-15)
    assert late.P_0 == pytest.approx(early.P_0, rel=1e-15)
    assert late.VMA == pytest.approx(early.VMA, rel=1e-15)


def test_octave_negative_pulse_response():
    """COM Octave, PR = -pulse: imaxPR is the LEAST negative sample, and both
    levels come out negative while VMA stays positive."""
    r = vma(-_PULSE, 4)
    assert r.P_3 == pytest.approx(-0.34317217752276274, rel=1e-13)
    assert r.P_0 == pytest.approx(-1.0843316087466099, rel=1e-13)
    assert r.VMA == pytest.approx(0.7411594312238472, rel=1e-13)


def test_octave_window_past_the_end_of_the_bit_stream_is_refused():
    """COM Octave: Bit_stream_response is indexed directly, so a PR whose peak
    pushes the window past the end errors --
    'Bit_stream_response(11732): out of bound 4095' for a 9000-sample PR
    peaking at sample 8901 with M=1.  A numpy slice clipped instead and
    returned NaN for both levels."""
    pr = np.zeros(9000)
    pr[8900] = 1.0
    with pytest.raises(IndexError):
        vma(pr, 1)
