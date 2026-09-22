"""Verification tests for r_parrelell2() — 2-port S-params for a shunt resistor.

# ============================================================
# MATLAB GROUND TRUTH
# s11=s22 = -zref/(rpad*(zref/rpad+2)) = -zref/(zref+2*rpad)
# s21=s12 = 2/(zref/rpad+2) = 2*rpad/(zref+2*rpad)
#
# zref=50, rpad=50:   denom=150, s11=-50/150=-1/3, s21=100/150=2/3
# zref=50, rpad→∞:    s11→0, s21→1  (open circuit: no effect)
# zref=50, rpad=25:   denom=100, s11=-50/100=-0.5, s21=50/100=0.5
# Lossless check: |s11|^2 + |s21|^2 = 1/9+4/9 = 5/9 (lossy network)
# Energy conservation: s11+s21 = (-zref+2rpad)/(zref+2rpad) ≠ 1 (shunt load)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.r_parrelell2.py_impl import r_parrelell2


def test_known_values():
    S = r_parrelell2(50.0, np.array([1e9]), 50.0)
    assert S.Parameters[0, 0, 0] == pytest.approx(-1/3, rel=1e-10)
    assert S.Parameters[1, 0, 0] == pytest.approx(2/3, rel=1e-10)


def test_large_rpad_approaches_thru():
    """Very large rpad → open circuit → s11≈0, s21≈1."""
    S = r_parrelell2(50.0, np.array([1e9]), 1e12)
    assert abs(S.Parameters[0, 0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert abs(S.Parameters[1, 0, 0]) == pytest.approx(1.0, rel=1e-6)


def test_quarter_ohm():
    S = r_parrelell2(50.0, np.array([1e9]), 25.0)
    assert S.Parameters[0, 0, 0] == pytest.approx(-0.5, rel=1e-10)
    assert S.Parameters[1, 0, 0] == pytest.approx(0.5, rel=1e-10)


def test_symmetry():
    S = r_parrelell2(50.0, np.array([1e9, 2e9]), 75.0)
    np.testing.assert_array_equal(S.Parameters[0, 0, :], S.Parameters[1, 1, :])
    np.testing.assert_array_equal(S.Parameters[0, 1, :], S.Parameters[1, 0, :])


def test_s11_negative():
    """Shunt resistor always produces negative s11 (unlike series)."""
    S = r_parrelell2(50.0, np.array([1e9]), 100.0)
    assert S.Parameters[0, 0, 0].real < 0


def test_output_shape():
    S = r_parrelell2(50.0, np.linspace(1e9, 10e9, 5), 50.0)
    assert S.Parameters.shape == (2, 2, 5)


# ---------------------------------------------------------------------------
# Values below came from running r_parrelell2() out of
# octave/com_ieee8023_4p16p0_octave_compat.m under COM Octave via
# tools/octave_oracle.py, rather than from reading the MATLAB.
#
# These pin the EXPRESSION, not just the result. MATLAB writes
#     S11 = -zref/(rpad*(zref/rpad + 2))      S21 = 2/(zref/rpad + 2)
# and the algebraically equal -zref/(zref+2*rpad) and 2*rpad/(zref+2*rpad)
# are not equal in floating point.
# ---------------------------------------------------------------------------

_F = np.array([1e9, 2e9, 3e9])


def test_oracle_nominal_is_bit_identical():
    """COM Octave: zref=50, rpad=1000 -> S11 -0.024390243902439025,
    S21 0.97560975609756106. Exact equality, not allclose: the simplified
    form agrees to every printed digit and still differs in the last bit.
    """
    P = r_parrelell2(50.0, _F, 1000.0).Parameters
    assert P[0, 0, 0].real == -0.024390243902439025
    assert P[1, 0, 0].real == 0.97560975609756106
    assert P[1, 1, 0] == P[0, 0, 0] and P[0, 1, 0] == P[1, 0, 0]


def test_oracle_infinite_rpad_passes_through():
    """COM Octave: rpad=Inf -> S11 -0, S21 1.

    This is the "no pad resistor" setting. The simplified form computes S21 as
    2*Inf/(50+2*Inf) = Inf/Inf = NaN and puts it straight into the package
    cascade.
    """
    P = r_parrelell2(50.0, _F, np.inf).Parameters
    assert P[1, 0, 0] == 1.0, 'an absent shunt resistor must pass the wave through'
    assert P[0, 0, 0] == 0.0
    assert np.signbit(P[0, 0, 0].real), 'COM Octave gives negative zero here'


def test_oracle_zero_rpad_is_nan_not_minus_one():
    """COM Octave: rpad=0 -> S11 NaN, S21 0.

    zref/0 is Inf, 0*(Inf+2) is NaN, and -50/NaN is NaN. The simplified form
    reduces to -zref/zref = -1, a clean-looking answer MATLAB never gives.
    """
    P = r_parrelell2(50.0, _F, 0.0).Parameters
    assert np.isnan(P[0, 0, 0].real)
    assert P[1, 0, 0] == 0.0


def test_oracle_negative_rpad():
    """COM Octave: rpad=-1000 -> S11 0.02564102564102564, S21 1.0256410256410258."""
    P = r_parrelell2(50.0, _F, -1000.0).Parameters
    assert P[0, 0, 0].real == 0.02564102564102564
    assert P[1, 0, 0].real == 1.0256410256410258
