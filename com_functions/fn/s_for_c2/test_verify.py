"""Verification tests for s_for_c2() — 2-port S-params for a shunt capacitor.

# ============================================================
# MATLAB GROUND TRUTH
# Y = jωC,  s11=s22=-Y*Z/(2+Y*Z),  s21=s12=2/(2+Y*Z)
# where Y*Z = j*2π*f*cpad*zref
#
# f=0 (DC): Y*Z=0, s11=0, s21=1
# f=∞:      s11→-1, s21→0
#
# Specific: zref=50, cpad=1e-12, f=1e9
#   Y*Z = j*2π*1e9*1e-12*50 = j*2π*0.05 ≈ j*0.3142
#   denom = 2 + j*0.3142
#   s21 = 2/(2+j*0.3142), |s21| ≈ 2/|2+j0.3142| = 2/2.0246 ≈ 0.9878
#   s11 = -j*0.3142/(2+j*0.3142), |s11| ≈ 0.3142/2.0246 ≈ 0.1552
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.s_for_c2.py_impl import s_for_c2


def test_dc_thru():
    S = s_for_c2(50.0, np.array([0.0]), 1e-12)
    assert S.Parameters[1, 0, 0] == pytest.approx(1.0, abs=1e-12)
    assert S.Parameters[0, 0, 0] == pytest.approx(0.0, abs=1e-12)


def test_symmetry():
    S = s_for_c2(50.0, np.array([1e9, 5e9]), 1e-12)
    np.testing.assert_array_equal(S.Parameters[0, 0, :], S.Parameters[1, 1, :])
    np.testing.assert_array_equal(S.Parameters[0, 1, :], S.Parameters[1, 0, :])


def test_high_freq_approaches_short():
    S = s_for_c2(50.0, np.array([1e15]), 1e-12)
    assert abs(S.Parameters[1, 0, 0]) == pytest.approx(0.0, abs=1e-3)
    assert abs(S.Parameters[0, 0, 0]) == pytest.approx(1.0, abs=1e-3)


def test_s11_purely_imaginary_contribution():
    """s11 = -jωCZ/(2+jωCZ): numerator is purely imaginary."""
    S = s_for_c2(50.0, np.array([1e9]), 1e-12)
    ycz = 1j * 2 * np.pi * 1e9 * 1e-12 * 50
    expected_s11 = -ycz / (2 + ycz)
    assert S.Parameters[0, 0, 0] == pytest.approx(expected_s11, rel=1e-10)


def test_output_shape():
    S = s_for_c2(50.0, np.linspace(0, 10e9, 11), 1e-12)
    assert S.Parameters.shape == (2, 2, 11)


def test_passive_network():
    """|s11|^2 + |s21|^2 ≤ 1 (passivity) for all frequencies."""
    S = s_for_c2(50.0, np.linspace(1e8, 100e9, 50), 1e-12)
    power = np.abs(S.Parameters[0, 0, :])**2 + np.abs(S.Parameters[1, 0, :])**2
    assert np.all(power <= 1.0 + 1e-12)


# ============================================================
# COM Octave oracle — values from the reference body run under Octave
# (tools/octave_oracle.py, `sparameters` stubbed to a struct, which is all
# the one caller s_for_c4 uses: it reads S2.Parameters and nothing else).
#
#   s_for_c2(50, [0 1e9 1e10 2.5e10 5.3e10], 1e-12)
#
# These are pinned with `==`, not a tolerance, because the point is the
# arithmetic FORM: MATLAB evaluates the numerator left to right as
# ((((-1i*2)*pi).*f)*cpad)*zref and divides once, at the end. Any
# algebraically equivalent rearrangement — folding the sign into the
# quotient, s11 = s21 - 1, dividing by (1 + jwCZ/2) — lands on different
# last bits and fails here. The Octave run confirms the numerator and the
# denominator are bit-identical to the Python intermediates.
#
# Signed zero is part of the contract: at f=0 the numerator is -0-0j, so
# s11 is -0.0+0.0j, not +0.0. `1 - s21` would give +0.0.
# ============================================================
def test_octave_oracle_nominal_bit_exact():
    S = s_for_c2(50.0, np.array([0.0, 1e9, 1e10, 2.5e10, 5.3e10]), 1e-12)
    s11 = np.array([-0 + 0j,
                    -0.024079864169266819 - 0.1532971764608092j,
                    -0.71159956085799903 - 0.45301835045029026j,
                    -0.93910332153571241 - 0.23914069711428193j,
                    -0.98577712985342891 - 0.11840853056838824j])
    s21 = np.array([1 + 0j,
                    0.97592013583073312 - 0.1532971764608092j,
                    0.28840043914200103 - 0.45301835045029026j,
                    0.060896678464287556 - 0.23914069711428193j,
                    0.014222870146571157 - 0.11840853056838824j])
    np.testing.assert_array_equal(S.Parameters[0, 0, :], s11)
    np.testing.assert_array_equal(S.Parameters[1, 1, :], s11)
    np.testing.assert_array_equal(S.Parameters[1, 0, :], s21)
    np.testing.assert_array_equal(S.Parameters[0, 1, :], s21)


def test_octave_oracle_dc_signed_zero():
    """Octave: s11(f=0) is -0+0j — the minus sign survives the divide."""
    S = s_for_c2(50.0, np.array([0.0]), 1e-12)
    assert np.signbit(S.Parameters[0, 0, 0].real)
    assert not np.signbit(S.Parameters[0, 0, 0].imag)
    assert not np.signbit(S.Parameters[1, 0, 0].real)
