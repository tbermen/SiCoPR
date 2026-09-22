"""Verification tests for TD_CTLE().

# ============================================================
# MATLAB GROUND TRUTH
# p1_ctle = -2*pi*f_p1  (analog pole frequency, rad/s)
# p2_ctle = -2*pi*f_p2
# z_ctle  = -2*pi*f_z * 10^(kacdc_dB/20)
#
# f_p1=1e9, f_p2=2e9, f_z=0.5e9:
#   p1_ctle = -2*pi*1e9 ≈ -6.28318e9
#   p2_ctle = -2*pi*2e9 ≈ -12.56637e9
#   z_ctle (kacdc_dB=0) = -2*pi*0.5e9 ≈ -3.14159e9
#   z_ctle (kacdc_dB=20) = -2*pi*0.5e9*10 = -2*pi*5e9 ≈ -31.4159e9
#
# Bilinear poles/zeros are inside unit circle (stable) for
#   any f_p,f_z << fb*oversampling.
#
# filter applied to delta [1,0,...,0]:
#   output[0] = B_filt[0] / A_filt[0] = B_filt[0]  (A monic)
#   output length = input length
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.TD_CTLE.py_impl import TD_CTLE


def delta(n=32):
    ir = np.zeros(n)
    ir[0] = 1.0
    return ir


def test_p1_ctle_value():
    """p1_ctle = -2*pi*f_p1."""
    _, p1_ctle, _, _ = TD_CTLE(delta(), 25e9, 0.5e9, 1e9, 2e9, 0.0, 2)
    assert p1_ctle == pytest.approx(-2 * np.pi * 1e9, rel=1e-12)


def test_p2_ctle_value():
    """p2_ctle = -2*pi*f_p2."""
    _, _, p2_ctle, _ = TD_CTLE(delta(), 25e9, 0.5e9, 1e9, 2e9, 0.0, 2)
    assert p2_ctle == pytest.approx(-2 * np.pi * 2e9, rel=1e-12)


def test_z_ctle_zero_dB():
    """kacdc_dB=0: z_ctle = -2*pi*f_z."""
    _, _, _, z_ctle = TD_CTLE(delta(), 25e9, 0.5e9, 1e9, 2e9, 0.0, 2)
    assert z_ctle == pytest.approx(-2 * np.pi * 0.5e9, rel=1e-12)


def test_z_ctle_positive_dB():
    """kacdc_dB=20: z_ctle = -2*pi*f_z*10."""
    _, _, _, z_ctle = TD_CTLE(delta(), 25e9, 0.5e9, 1e9, 2e9, 20.0, 2)
    assert z_ctle == pytest.approx(-2 * np.pi * 0.5e9 * 10, rel=1e-12)


def test_output_length_matches_input():
    """Output length equals input length."""
    ir = np.zeros(50)
    ir[0] = 1.0
    ir_out, _, _, _ = TD_CTLE(ir, 25e9, 5e9, 10e9, 20e9, 0.0, 2)
    assert len(ir_out) == 50


def test_output_is_finite():
    """No NaN or inf in output for reasonable parameters."""
    ir = np.zeros(64)
    ir[0] = 1.0
    ir_out, _, _, _ = TD_CTLE(ir, 25e9, 5e9, 10e9, 20e9, -3.0, 2)
    assert np.all(np.isfinite(ir_out))


# ============================================================
# COM Octave oracle — TD_CTLE extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m and run by tools/octave_oracle.py.
# fb=106.25e9, f_z=6e9, f_p1=15e9, f_p2=30e9, kacdc_dB=-5, oversampling=2.
#
#   ir = [1 0 0 0 0 0 0 0]  ->
#       [0.66006149825361637, 0.7376254397746439, -0.0052120685212741552,
#        -0.18645989371443616, -0.18937285517349406, -0.14784426986338875,
#        -0.10466654748288841, -0.070715630008877789]
#       p1_ctle = -94247779607.693787
#       p2_ctle = -188495559215.38757
#       z_ctle  = -21199768512.335392
#
#   ir 4x2 = [1 0; 0 1; 0 0; 0 0]  ->  filter() runs down the COLUMNS:
#       column 1 = the impulse response above (first four samples)
#       column 2 = that response delayed by one sample
#       (scipy's lfilter defaults to axis=-1 and filtered along the rows)
#
#   ir = 1 (scalar)         -> 0.66006149825361637
#   ir = [1 0 0 0 0 0 0 1j] -> real part as above, and the imaginary delta
#       re-appears as ...+0.66006149825361637j in the last sample.  Casting the
#       input with dtype=float DISCARDED that, on a ComplexWarning alone.
#
#   f_z=f_p1=0.66e9, f_p2=100e100 (the engine's high-pass call) ->
#       [0.99577084733306243, -0.0083765716299195464, -0.0082146838288203322,
#        -0.0080559247134534655, -0.0079002338180251264, -0.0077475518453190388,
#        -0.0075978206441124551, -0.0074509831870281642]
#       p1_ctle = -4146902302.7385268, p2_ctle = -6.2831853071795862e+102,
#       z_ctle  = -2331974536.3568935
# ============================================================

OCT_IR = [0.66006149825361637, 0.7376254397746439, -0.0052120685212741552,
          -0.18645989371443616, -0.18937285517349406, -0.14784426986338875,
          -0.10466654748288841, -0.070715630008877789]
OCT_ARGS = (106.25e9, 6e9, 15e9, 30e9, -5.0, 2)


def test_oracle_impulse_response():
    ir_out, p1, p2, z = TD_CTLE(delta(8), *OCT_ARGS)
    np.testing.assert_allclose(ir_out, OCT_IR, rtol=1e-13, atol=1e-16)
    assert p1 == pytest.approx(-94247779607.693787, rel=1e-15)
    assert p2 == pytest.approx(-188495559215.38757, rel=1e-15)
    assert z == pytest.approx(-21199768512.335392, rel=1e-15)


def test_oracle_highpass_call():
    ir_out, p1, p2, z = TD_CTLE(delta(8), 106.25e9, 0.66e9, 0.66e9, 100e100, -5.0, 2)
    np.testing.assert_allclose(
        ir_out, [0.99577084733306243, -0.0083765716299195464,
                 -0.0082146838288203322, -0.0080559247134534655,
                 -0.0079002338180251264, -0.0077475518453190388,
                 -0.0075978206441124551, -0.0074509831870281642],
        rtol=1e-13, atol=1e-16)
    assert p1 == pytest.approx(-4146902302.7385268, rel=1e-15)
    assert p2 == pytest.approx(-6.2831853071795862e+102, rel=1e-15)
    assert z == pytest.approx(-2331974536.3568935, rel=1e-15)


def test_matrix_input_is_filtered_down_the_columns():
    """MATLAB filter() uses the first non-singleton dimension."""
    ir = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    out, _, _, _ = TD_CTLE(ir, *OCT_ARGS)
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out[:, 0], OCT_IR[:4], rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(out[:, 1], [0.0] + OCT_IR[:3], rtol=1e-13, atol=1e-16)


def test_row_vector_is_filtered_along_itself():
    """A 1xN row keeps filtering along the row, not down its one element."""
    out, _, _, _ = TD_CTLE(delta(8).reshape(1, -1), *OCT_ARGS)
    assert out.shape == (1, 8)
    np.testing.assert_allclose(out[0], OCT_IR, rtol=1e-13, atol=1e-16)


def test_complex_input_keeps_its_imaginary_part():
    """dtype=float silently dropped it; MATLAB filters the complex signal."""
    ir = np.zeros(8, dtype=complex)
    ir[0] = 1.0
    ir[7] = 1j
    out, _, _, _ = TD_CTLE(ir, *OCT_ARGS)
    assert np.iscomplexobj(out)
    np.testing.assert_allclose(out.real, OCT_IR, rtol=1e-13, atol=1e-16)
    assert out[7].imag == pytest.approx(0.66006149825361637, rel=1e-13)
    np.testing.assert_allclose(out[:7].imag, 0.0, atol=0)


def test_scalar_input_is_filtered():
    """MATLAB filters a 1x1; lfilter raised "selected axis is out of range"."""
    out, _, _, _ = TD_CTLE(1.0, *OCT_ARGS)
    assert np.asarray(out).size == 1
    assert float(np.asarray(out).ravel()[0]) == pytest.approx(
        0.66006149825361637, rel=1e-13)


def test_empty_input():
    out, _, _, _ = TD_CTLE(np.array([]), *OCT_ARGS)
    assert out.size == 0
