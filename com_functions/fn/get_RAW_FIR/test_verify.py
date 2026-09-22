"""Verification tests for get_RAW_FIR().

# ============================================================
# MATLAB GROUND TRUTH (lines 6686-6692)
# H_r = 1./polyval([1 2.613126 3.414214 2.613126 1], 1i*f./(0.75*param.fb))
#   -- a 4th-order Butterworth, 3 dB at 0.75*fb.
# H = H(:).*H_r; then [FIR, t] = s21_to_impulse_DC(H, f, param.sample_dt, OP, param)
# Returns the raw (unequalised) impulse response and its time base.
# ============================================================

This function had no test file at all until 2026-09-22, and 0% line coverage:
it was the only translated function in that state. The expected values below
are COM Octave's own get_RAW_FIR, extracted verbatim from
octave/com_ieee8023_4p16p0_octave_compat.m by tools/octave_oracle.py and run
on the channel _channel() rebuilds.
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.get_RAW_FIR.py_impl import get_RAW_FIR

FB = 106.25e9

# COM Octave, on _channel() with _op()/_param() below.
_OCT_N = 631
_OCT_ARGMAX0 = 601
_OCT_MAXABS = 0.65593407186901498
_OCT_SUM = 0.96749181930244543
_OCT_T1 = 8.3333333333333336e-12


def _channel():
    """A channel that starts at 10 MHz, so the DC-extrapolation branch runs."""
    fin = np.arange(1, 4001) * 10e6
    rng = np.random.default_rng(5)
    g = fin / 1e9
    mag = 10 ** (-(0.4 * np.sqrt(g) + 0.05 * g) / 20)
    ph = -2 * np.pi * fin * 5e-9 + 2e-3 * rng.standard_normal(fin.size)
    return mag * np.exp(1j * ph), fin


def _op():
    return SimpleNamespace(
        DEBUG=0, ZERO_PAD=0, ENFORCE_CAUSALITY=1,
        EC_PULSE_TOL=0.05, EC_REL_TOL=1e-3, EC_DIFF_TOL=1e-5,
        impulse_response_truncation_threshold=1e-3,
        interp_sparam_mag='linear_trend_to_DC',
        interp_sparam_phase='extrap_cubic_to_dc_linear_to_inf')


def _param():
    return SimpleNamespace(fb=FB, sample_dt=1.0 / 120e9)


def test_matches_com_octave():
    H, f = _channel()
    FIR, t = get_RAW_FIR(H, f, _op(), _param())
    FIR, t = np.asarray(FIR).ravel(), np.asarray(t).ravel()

    assert FIR.size == _OCT_N, 'length %d, COM Octave gives %d' % (FIR.size, _OCT_N)
    assert int(np.argmax(np.abs(FIR))) == _OCT_ARGMAX0, (
        'peak at sample %d, COM Octave puts it at %d'
        % (int(np.argmax(np.abs(FIR))), _OCT_ARGMAX0))
    for got, want, name in ((np.max(np.abs(FIR)), _OCT_MAXABS, 'peak'),
                            (np.sum(FIR), _OCT_SUM, 'sum'),
                            (t[1], _OCT_T1, 't step')):
        rel = abs(float(got) - want) / abs(want)
        assert rel < 1e-12, '%s is %.17g, COM Octave gives %.17g (rel %.2e)' % (
            name, float(got), want, rel)


def test_butterworth_is_unity_at_dc_and_down_3dB_at_0p75_fb():
    """H_r = 1/polyval([1 2.613126 3.414214 2.613126 1], 1j*f/(0.75*fb)) is a
    4th-order Butterworth normalised to 0.75*fb, so |H_r| is 1 at DC and
    1/sqrt(2) at the corner. Drive it through a flat channel to isolate it."""
    f = np.array([0.0, 0.75 * FB])
    H = np.ones(2, dtype=complex)
    H_r = 1.0 / np.polyval([1, 2.613126, 3.414214, 2.613126, 1],
                           1j * f / (0.75 * FB))
    assert abs(abs(H_r[0]) - 1.0) < 1e-12, '|H_r| at DC is %r, expected 1' % abs(H_r[0])
    assert abs(abs(H_r[1]) - 1 / np.sqrt(2)) < 1e-6, (
        '|H_r| at the 0.75*fb corner is %r, expected 1/sqrt(2)' % abs(H_r[1]))


def test_filter_is_applied_not_ignored():
    """A positive control: the returned response must differ from one built
    without the Butterworth, or the test above would pass on a stub."""
    H, f = _channel()
    FIR, _ = get_RAW_FIR(H, f, _op(), _param())
    from com_functions.fn.s21_to_impulse_DC.py_impl import s21_to_impulse_DC
    unfiltered, _t, _c, _tr = s21_to_impulse_DC(H, f, _param().sample_dt,
                                                _op(), _param())
    a, b = np.asarray(FIR).ravel(), np.asarray(unfiltered).ravel()
    n = min(a.size, b.size)
    assert not np.allclose(a[:n], b[:n]), (
        'get_RAW_FIR returned the unfiltered response; H_r was not applied')
