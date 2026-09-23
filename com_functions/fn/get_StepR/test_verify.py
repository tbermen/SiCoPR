"""Verification tests for get_StepR().

# ============================================================
# MATLAB GROUND TRUTH (lines 6876-6902)
# cb_step=False: pulse = cumsum(ir)
# cb_step=True:  pulse = filter(drive_pulse, 1, ir) [shaped edge]
# TDR_response = (1+pulse)./(1-pulse) * ZT * 2
# result.ZSR, result.pulse
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.get_StepR.py_impl import get_StepR


def _param(M=4, fb=50e9, TR_TDR=10.0):
    return SimpleNamespace(samples_per_ui=M, fb=fb, TR_TDR=TR_TDR)


def test_no_cbstep_pulse_is_cumsum():
    """cb_step=False: pulse = cumsum(ir)."""
    ir = np.array([0.0, 0.1, 0.2, 0.0, 0.0])
    result = get_StepR(ir, _param(), False, 50.0)
    np.testing.assert_allclose(result.pulse, np.cumsum(ir))


def test_output_length_matches_input():
    """Output length == len(ir)."""
    ir = np.zeros(20)
    result = get_StepR(ir, _param(), False, 50.0)
    assert len(result.ZSR) == 20
    assert len(result.pulse) == 20


def test_zsr_formula():
    """ZSR = (1+pulse)/(1-pulse)*ZT*2."""
    ir = np.zeros(8)
    result = get_StepR(ir, _param(), False, 50.0)
    expected = (1 + result.pulse) / (1 - result.pulse) * 50.0 * 2
    np.testing.assert_allclose(result.ZSR, expected)


def test_returns_namespace_with_fields():
    """Result has ZSR and pulse fields."""
    ir = np.ones(8) * 0.01
    result = get_StepR(ir, _param(), False, 25.0)
    assert hasattr(result, 'ZSR')
    assert hasattr(result, 'pulse')


def test_cbstep_true_output_length():
    """cb_step=True: output still matches ir length."""
    ir = np.zeros(40)
    result = get_StepR(ir, _param(), True, 50.0)
    assert len(result.ZSR) == 40


# ============================================================
# COM Octave oracle values (tools/octave_oracle.py, get_StepR extracted
# verbatim from octave/com_ieee8023_4p16p0_octave_compat.m).
#
# What these catch:
#   1. `tedge=0:dt:edge_time*2` is a MATLAB colon, which STOPS at the limit.
#      np.arange(0, edge_time*2+dt, dt) adds one more point whenever
#      edge_time*2/dt is not an integer, lengthening drive_pulse by a tap.
#      Over 660 (fb, samples_per_ui, TR_TDR) combinations the arange form
#      disagreed with Octave's numel on 583 of them.
#   2. ir was cast with dtype=float, which discarded the imaginary part of a
#      complex impulse response on a ComplexWarning only.
# ============================================================

_IR6 = [6.2e-05, 0.014937, -0.013707, -0.04453, -0.022734, -0.049582]
_IR40 = _IR6 + [
    0.003007, 0.067011, -0.02461, -0.031024, 0.024492, 0.017844, 0.005271,
    -0.046523, -0.001463, 0.034765, -0.067211, -0.022881, -0.095061,
    -0.064477, -0.092087, -0.011755, -0.063372, 0.013563, 0.007838,
    -0.009347, -0.125838, -0.026935, -0.002425, 0.005665, -0.076507,
    -0.023888, -0.048926, -0.040442, 0.053045, -0.040377, -0.001626,
    0.044219, -0.02918, -0.005585]


def test_octave_cbstep_edge_taps_stop_at_the_limit_M1():
    """samples_per_ui=1: tedge holds ONE point, so drive_pulse = [edge0, 1].

    COM Octave, fb=53.125e9, samples_per_ui=1, TR_TDR=8e-3, ZT=50, cb_step=1.
    dt = 1.882e-11 exceeds edge_time*2 = 1.6e-11, so the colon yields [0]
    alone; np.arange added a second point and pulse(2) came out 4% low.
    """
    result = get_StepR(np.array(_IR6), _param(M=1, fb=53.125e9, TR_TDR=8e-3),
                       True, 50.0)
    np.testing.assert_allclose(result.pulse, [
        1.3766765505351942e-20, 6.2000000000003323e-05, 0.014936999999999997,
        -0.013707000000000011, -0.044530000000000007, -0.022734000000000011],
        rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(result.ZSR, [
        100.0, 100.01240076884767, 103.03269943140691, 97.29566827495519,
        91.473677156232952, 95.554269243028983], rtol=1e-12)


def test_octave_cbstep_default_com_config():
    """COM defaults fb=53.125 GHz, 32 spui, TR_TDR=8e-3 ns, ZT=50.

    tedge holds 28 points, not the 29 np.arange produced, so the divergence
    first bites at pulse(29).
    """
    result = get_StepR(np.array(_IR40),
                       _param(M=32, fb=53.125e9, TR_TDR=8e-3), True, 50.0)
    assert len(result.pulse) == 40
    np.testing.assert_allclose(result.pulse[27:], [
        -0.26065234107230867, -0.2900808409697847, -0.31866600135503187,
        -0.34582733338850852, -0.37629374171039692, -0.40704975187448883,
        -0.43954385636442139, -0.4732854461561502, -0.50269134963666362,
        -0.53293424182340765, -0.56175945341123867, -0.58646857944775732,
        -0.61114480062107035], rtol=1e-12)
    np.testing.assert_allclose(result.ZSR[27:], [
        58.648021729670795, 55.029044419925825, 51.668428392393864,
        48.607473661901572, 45.317815476984485, 42.141384647954027,
        38.932898164771082, 35.751018597113358, 33.094530722066168,
        30.468740630454118, 28.060694342623894, 26.066158883284729,
        24.135335273963722], rtol=1e-12)


def test_octave_complex_ir_keeps_its_imaginary_part():
    """MATLAB cumsum() carries a complex ir through; the float cast did not.

    COM Octave, ir = _IR6 + 1j*_IR6[::-1], cb_step=0, ZT=50.
    """
    ir = np.array(_IR6) + 1j * np.array(_IR6)[::-1]
    result = get_StepR(ir, _param(M=32, fb=53.125e9, TR_TDR=8e-3), False, 50.0)
    np.testing.assert_allclose(result.pulse, [
        6.2e-05 - 0.049582j, 0.014999 - 0.072316j, 0.001292 - 0.116846j,
        -0.043238 - 0.130553j, -0.065972 - 0.115616j,
        -0.115554 - 0.115554j], rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(result.ZSR, [
        99.521840491708232 - 9.8933052801872492j,
        101.95691461799493 - 14.82710803086994j,
        97.554541436910753 - 23.113320358640639j,
        88.754804665202215 - 23.621173704807671j,
        85.440718240973908 - 20.113018053146277j,
        77.379880395046115 - 18.373789793384415j], rtol=1e-12)


def test_colon_is_not_one_short_just_below_an_integer_quotient():
    """The floor() spelling fails when limit/step lands a hair below an integer.

    fb=106.25e9, samples_per_ui=32, TR_TDR=0.5325 ns gives a quotient of
    3620.9999999999995, about 2000 eps below 3621. COM Octave returns 3622
    points; floor()+1 returns 3621. This case was missed by a 660-combination
    sweep that compared only numel against Octave, which is precisely the
    quantity it gets wrong -- the sweep never included a below-integer
    quotient. Pinned so that spelling cannot come back.
    """
    from com_functions.fn.get_StepR.py_impl import _colon
    dt = 1.0 / 106.25e9 / 32
    limit = 0.5325e-9 * 2
    assert int(np.floor(limit / dt)) + 1 == 3621      # the wrong spelling
    assert _colon(dt, limit).size == 3622              # what Octave returns


def test_colon_matches_get_PulseR_exactly():
    """Same reference line, so the two helpers must not drift apart."""
    from com_functions.fn.get_StepR.py_impl import _colon as s_colon
    from com_functions.fn.get_PulseR.py_impl import _colon as p_colon
    for fb, M, tr in ((106.25e9, 32, 0.5325e-9), (53.125e9, 32, 8e-12),
                      (112.5e9, 16, 0.5325e-9), (42.5e9, 64, 1e-12)):
        dt = 1.0 / fb / M
        np.testing.assert_array_equal(s_colon(dt, tr * 2), p_colon(dt, tr * 2))
