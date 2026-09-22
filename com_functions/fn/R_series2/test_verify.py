"""Verification tests for R_series2() — 2-port S-params for a series resistor.

# ============================================================
# MATLAB GROUND TRUTH
# s11=s22=R/(R+2*zref),  s21=s12=2*zref/(R+2*zref)
#
# zref=50, R=50:  s11=50/150=1/3,  s21=100/150=2/3
# zref=50, R=0:   s11=0,           s21=1   (short: lossless thru)
# zref=50, R→∞:   s11→1,           s21→0   (open: full reflection)
# zref=50, R=100: s11=100/200=0.5, s21=100/200=0.5
#
# Note: s11+s21 = R/(R+2Z) + 2Z/(R+2Z) = (R+2Z)/(R+2Z) = 1  (always)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.R_series2.py_impl import R_series2


def test_known_values():
    S = R_series2(50.0, np.array([1e9]), 50.0)
    assert S.Parameters[0, 0, 0] == pytest.approx(1/3, rel=1e-10)
    assert S.Parameters[1, 0, 0] == pytest.approx(2/3, rel=1e-10)


def test_zero_resistance_thru():
    S = R_series2(50.0, np.array([1e9]), 0.0)
    assert S.Parameters[0, 0, 0] == pytest.approx(0.0)
    assert S.Parameters[1, 0, 0] == pytest.approx(1.0)


def test_symmetry_s11_eq_s22():
    S = R_series2(50.0, np.array([1e9, 2e9]), 75.0)
    np.testing.assert_array_equal(S.Parameters[0, 0, :], S.Parameters[1, 1, :])


def test_symmetry_s12_eq_s21():
    S = R_series2(50.0, np.array([1e9, 2e9]), 75.0)
    np.testing.assert_array_equal(S.Parameters[0, 1, :], S.Parameters[1, 0, :])


def test_s11_plus_s21_equals_one():
    """s11 + s21 = 1 always (energy conservation for this lossless-style formula)."""
    S = R_series2(50.0, np.array([1e9]), 100.0)
    assert (S.Parameters[0, 0, 0] + S.Parameters[1, 0, 0]).real == pytest.approx(1.0)


def test_output_shape():
    S = R_series2(50.0, np.linspace(1e9, 10e9, 7), 50.0)
    assert S.Parameters.shape == (2, 2, 7)


def test_frequency_independent():
    """Result is constant across frequencies."""
    f = np.array([1e9, 5e9, 10e9])
    S = R_series2(50.0, f, 50.0)
    assert np.all(S.Parameters[0, 0, :] == S.Parameters[0, 0, 0])


# ---------------------------------------------------------------------------
# Values below came from running R_series2() out of
# octave/com_ieee8023_4p16p0_octave_compat.m under COM Octave via
# tools/octave_oracle.py, rather than from reading the MATLAB.
# ---------------------------------------------------------------------------

_F = np.array([1e9, 2e9, 3e9])


def test_oracle_nominal():
    """COM Octave: zref=50, R=50 -> S11 1/3, S21 2/3, flat over frequency."""
    P = R_series2(50.0, _F, 50.0).Parameters
    assert P.shape == (2, 2, 3)
    np.testing.assert_allclose(P[:, :, 0].ravel(order='F'),
                               [1 / 3, 2 / 3, 2 / 3, 1 / 3])
    np.testing.assert_allclose(P[:, :, 2], P[:, :, 0])


def test_oracle_zero_and_infinite_R():
    """COM Octave: R=0 -> S11 0, S21 1.  R=Inf -> S11 NaN, S21 0.

    The open case is worth pinning because the intuitive answer is S11=1: the
    reference computes Inf/(Inf+100), which is NaN.
    """
    P0 = R_series2(50.0, _F, 0.0).Parameters
    assert P0[0, 0, 0] == 0.0 and P0[1, 0, 0] == 1.0
    Pi = R_series2(50.0, _F, np.inf).Parameters
    assert np.isnan(Pi[0, 0, 0].real) and Pi[1, 0, 0] == 0.0


def test_oracle_negative_R():
    """COM Octave: R=-25, zref=50 -> S11 -1/3, S21 4/3."""
    P = R_series2(50.0, _F, -25.0).Parameters
    np.testing.assert_allclose(P[:, :, 0].ravel(order='F'),
                               [-1 / 3, 4 / 3, 4 / 3, -1 / 3])


def test_oracle_empty_frequency_axis():
    """COM Octave returns a 2x2x0 rather than erroring."""
    assert R_series2(50.0, np.array([]), 50.0).Parameters.shape == (2, 2, 0)


def test_oracle_vector_R_is_rejected():
    """COM Octave: R_series2(50, [1e9 2e9 3e9], [1 2 3]) errors with
    'operator *: nonconformant arguments (op1 is 1x3, op2 is 1x3)'.

    MATLAB builds r as ones(1,length(f))*R, a matrix product. numpy would
    broadcast and return a per-frequency answer the reference cannot produce.
    """
    with pytest.raises(ValueError):
        R_series2(50.0, _F, np.array([1.0, 2.0, 3.0]))
