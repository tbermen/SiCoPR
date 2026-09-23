"""Tests for plot_modal (MATLAB lines ~8914).

MATLAB GROUND TRUTH:
  CM_MASK_REPORT=True, all S-params = 0 → all margins large positive → no fail.
  CM_MASK_REPORT=False → returns None.
  S11 = -40 dB flat → Rlcc margin = mask_value - (-40).
"""
import numpy as np
import pytest
from types import SimpleNamespace
from com_functions.fn.plot_modal.py_impl import plot_modal


def _op(cm_mask=True, show_brd=False):
    return SimpleNamespace(
        CM_MASK_REPORT=cm_mask,
        DISPLAY_WINDOW=False,
        PLOT_CM=False,
        SHOW_BRD=show_brd,
        DEBUG=0,
        TDR=False,
        Report_Modal_ERL='disable',
    )


def _chdata(n_f=200, scc11_val=1e-6):
    f = np.linspace(0.05e9, 67e9, n_f)
    ch = SimpleNamespace()
    ch.faxis = f
    ch.scc11_orig = np.ones(n_f, dtype=complex) * scc11_val
    ch.scd22_orig = np.ones(n_f, dtype=complex) * scc11_val
    ch.sdc22_orig = np.ones(n_f, dtype=complex) * scc11_val
    ch.scc11_raw = ch.scc11_orig
    ch.scd22_raw = ch.scd22_orig
    ch.sdc22_raw = ch.sdc22_orig
    return ch


def test_cm_mask_report_returns_struct():
    rs = plot_modal(SimpleNamespace(), _op(cm_mask=True), [_chdata()])
    assert rs is not None


def test_no_cm_mask_returns_none():
    rs = plot_modal(SimpleNamespace(), _op(cm_mask=False), [_chdata()])
    assert rs is None


def test_required_fields():
    rs = plot_modal(SimpleNamespace(), _op(cm_mask=True), [_chdata()])
    for f in ('Rlcc_179mm', 'Rlcc_178mm', 'Rlcd_179mm', 'Rldc_179mm',
              'Rlcc_179mm_fail', 'Rlcc_178mm_fail', 'Rlcd_179mm_fail', 'Rldc_179mm_fail'):
        assert hasattr(rs, f), f'missing field: {f}'


def test_small_scc11_no_fail():
    # Very small scc11 → dB is very negative → mask - dB is large positive → no fail
    rs = plot_modal(SimpleNamespace(), _op(cm_mask=True), [_chdata(scc11_val=1e-6)])
    assert not rs.Rlcc_179mm_fail
    assert not rs.Rlcc_178mm_fail


def test_fail_detection():
    # Large scc11 → dB ~ 0 → mask (negative) - 0 < 0 → fail
    rs = plot_modal(SimpleNamespace(), _op(cm_mask=True), [_chdata(scc11_val=1.0)])
    # At 0 dB, mask is -2 to -4 → margin < 0 → fail expected
    assert rs.Rlcc_179mm_fail or rs.Rlcc_178mm_fail


# ============================================================
# COM Octave oracle values -- the reference plot_modal (extracted verbatim
# from octave/com_ieee8023_4p16p0_octave_compat.m) run on the fixture below.
# Pinned 2026-09-23.
#
# One divergence this pins.  The port's dB helper was
#     20*log10(abs(x) + np.finfo(float).eps)
# where the reference (MATLAB L9407) is
#     dB=@(x) 20*log10(squeeze(abs(x)))
# with no epsilon.  The floor shifted every margin by 20/ln(10)*eps/|S| dB --
# 3.9e-14 dB at |S|=0.05, 9.6e-13 dB at |S|=0.002 -- and, where |S| is exactly
# zero, replaced the reference's +Inf margin with a finite 313 dB.
#
# The S-parameters are real-valued on purpose: Octave's abs() of a COMPLEX
# number and numpy's differ by an ULP (different hypot), a library difference
# and not the port's, which would otherwise stop these being pinned exactly.
#
# Fixture: f = [0.05 2 4 30 44 53.125 60 67 70] GHz -- one point in every
#   segment of all three masks -- with
#   scc11 = [0 0.1 0.15 0.9 0.25 0.3 0.35 0.4 0.45]   (first point EXACTLY 0)
#   scd22 = [0.002 0.005 0.008 0.011 0.014 0.017 0.02 0.023 0.026]
#   sdc22 = [0.004 0.007 0.01 0.013 0.016 0.019 0.022 0.025 0.028]
#
# COM Octave, return_struct:
#   Rlcc_179mm(1) = Inf,  Rlcc_178mm(1) = Inf   (the epsilon gave ~313 dB)
#   Rlcc_179mm_fail = 1   Rlcc_178mm_fail = 1
#   Rlcd_179mm_fail = 0   Rldc_179mm_fail = 0
# ============================================================

_OCT_F_GHZ = np.array([0.05, 2.0, 4.0, 30.0, 44.0, 53.125, 60.0, 67.0, 70.0])
_OCT_SCC11 = np.array([0.0, 0.1, 0.15, 0.9, 0.25, 0.3, 0.35, 0.4, 0.45],
                      dtype=complex)
_OCT_SCD22 = np.array([0.002, 0.005, 0.008, 0.011, 0.014, 0.017, 0.02,
                       0.023, 0.026], dtype=complex)
_OCT_SDC22 = np.array([0.004, 0.007, 0.01, 0.013, 0.016, 0.019, 0.022,
                       0.025, 0.028], dtype=complex)

_OCT_RS = {
    'Rlcc_179mm': [np.inf, 18, 14.478174818886377, -2.3848501887864977,
                   8.0411998265592484, 7.5981999056067515,
                   7.1186391129944884, 5.9588001734407516,
                   4.9357497244931263],
    'Rlcc_178mm': [np.inf, 16.75, 13.228174818886377, -2.3348501887864979,
                   8.7911998265592484, 7.2075749056067515,
                   5.8686391129944884, 4.7088001734407516,
                   3.6857497244931263],
    'Rlcd_179mm': [30.989753027896846, 23.434717560338449,
                   19.766435554278772, 22.383911002717852,
                   23.18802752172936, 23.391021572434525,
                   21.979400086720375, 20.765443279648146,
                   19.70053304058364],
    'Rldc_179mm': [24.969153114617221, 20.512156846773685,
                   17.828235294117647, 20.93289765974562,
                   22.028188582175623, 22.424927980943423,
                   21.151546383555875, 20.041199826559243,
                   19.056839373155615],
}
_OCT_RS_FAIL = {'Rlcc_179mm_fail': True, 'Rlcc_178mm_fail': True,
                'Rlcd_179mm_fail': False, 'Rldc_179mm_fail': False}


def _chdata_oracle():
    ch = SimpleNamespace()
    ch.faxis = _OCT_F_GHZ * 1e9
    ch.scc11_orig = _OCT_SCC11.copy()
    ch.scd22_orig = _OCT_SCD22.copy()
    ch.sdc22_orig = _OCT_SDC22.copy()
    return ch


@pytest.mark.parametrize('field', sorted(_OCT_RS))
def test_octave_mask_margins(field):
    """Each margin array, bit for bit against the reference."""
    rs = plot_modal(SimpleNamespace(), _op(cm_mask=True), [_chdata_oracle()])
    got = np.asarray(getattr(rs, field), dtype=float).ravel()
    np.testing.assert_array_equal(got, np.asarray(_OCT_RS[field]))


def test_octave_dB_of_zero_is_minus_inf_not_a_floor():
    """|S| = 0 gives the reference an Inf margin, not a 313 dB one.

    This is the visible face of the missing epsilon: with the floor in place
    the first element read 313.44..., a finite number, and no arithmetic
    downstream could tell it from a real measurement.
    """
    rs = plot_modal(SimpleNamespace(), _op(cm_mask=True), [_chdata_oracle()])
    assert np.isinf(rs.Rlcc_179mm[0]) and rs.Rlcc_179mm[0] > 0
    assert np.isinf(rs.Rlcc_178mm[0]) and rs.Rlcc_178mm[0] > 0


def test_octave_mask_fail_flags():
    rs = plot_modal(SimpleNamespace(), _op(cm_mask=True), [_chdata_oracle()])
    for field, expected in _OCT_RS_FAIL.items():
        assert bool(getattr(rs, field)) is expected, field
