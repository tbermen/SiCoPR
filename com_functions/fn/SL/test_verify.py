"""Verification tests for SL().

# ============================================================
# MATLAB GROUND TRUTH (lines 4368-4403)
# S = sparameters struct; S.Parameters shape (2, 2, N)
# R=0: return S unchanged (with warning)
# R=R_0: return S unchanged (else branch)
# R > R_0: cascade series resistor then S
# R < R_0: cascade shunt resistor then S
#
# Series: s_series = R/(R+2*Z0); s21_series = 2*Z0/(R+2*Z0)
# Shunt:  s11_shunt = -Z0/(Z0+2*Rpar); s21_shunt = 2*Rpar/(Z0+2*Rpar)
#          where Rpar = -R*Z0/(R-Z0) (from R < Z0 branch)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.SL.py_impl import SL


def _sparams(N=10, val=0.5):
    """Simple sparameters struct with Parameters of shape (2,2,N)."""
    p = np.zeros((2, 2, N), dtype=complex)
    p[0, 1, :] = p[1, 0, :] = val  # S12=S21=val
    return SimpleNamespace(Parameters=p, Frequencies=np.linspace(1e9, 20e9, N),
                           Impedance=50.0, NumPorts=2)


def test_r_equals_r0_returns_unchanged():
    """R=R_0: identity cascade → Parameters unchanged."""
    S = _sparams()
    SLD = SL(S, np.linspace(1e9, 20e9, 10), 50.0, 50.0)
    np.testing.assert_array_equal(SLD.Parameters, S.Parameters)


def test_r_zero_returns_s_unchanged():
    """R=0: returns S without cascade (with MATLAB warning in original)."""
    S = _sparams()
    SLD = SL(S, np.linspace(1e9, 20e9, 10), 0.0, 50.0)
    np.testing.assert_array_equal(SLD.Parameters, S.Parameters)


def test_r_greater_modifies_s11():
    """R > R_0: series cascade changes S11."""
    S = _sparams()
    f = np.linspace(1e9, 20e9, 10)
    SLD = SL(S, f, 100.0, 50.0)
    # S11 is no longer zero
    assert not np.allclose(SLD.Parameters[0, 0, :], 0.0)


def test_r_less_modifies_s11():
    """R < R_0: shunt cascade changes S11."""
    S = _sparams()
    f = np.linspace(1e9, 20e9, 10)
    SLD = SL(S, f, 25.0, 50.0)
    assert not np.allclose(SLD.Parameters[0, 0, :], 0.0)


def test_input_s_not_mutated():
    """Original S.Parameters is not modified."""
    S = _sparams()
    original = S.Parameters.copy()
    SL(S, np.linspace(1e9, 20e9, 10), 100.0, 50.0)
    np.testing.assert_array_equal(S.Parameters, original)


def _thru(N=3):
    """S11=S22=0, S12=S21=1. combines4p's N = 1-s22*s11 is then exactly 1,
    so SL hands back the pad's own S-parameters with no rounding in between."""
    p = np.zeros((2, 2, N), dtype=complex)
    p[0, 1, :] = p[1, 0, :] = 1.0
    return SimpleNamespace(Parameters=p, Frequencies=np.linspace(1e9, 20e9, N),
                           Impedance=50.0, NumPorts=2)


# ============================================================
# COM Octave oracle — SL run under Octave with R_series2,
# r_parrelell2 and combines4p, all verbatim from the reference.
#
# Cascaded through a lossless thru the divide in combines4p is by exactly 1,
# so these pin the pad arithmetic to the bit. That matters because
# r_parrelell2's denominator is written -zref/(rpad*(zref/rpad + 2)) and the
# algebraically equal -zref/(zref + 2*rpad) rounds differently:
#
#   SL(thru, f, -30, 50)   rpad = -18.75   Octave s11 -4.0000000000000009
#                                                  s21 -3.0000000000000009
#                          simplified form gives   s11 -4, s21 -3
#   SL(thru, f,  49.9, 50) rpad = 24950    Octave s21  0.99899899899899891
#                          simplified form gives       0.99899899899899902
#
# SL used to carry its own copy of both pads, and that copy was the
# simplified one; it now calls the shared functions.
# ============================================================
def test_octave_oracle_shunt_pad_form_negative_r():
    SLD = SL(_thru(), np.linspace(1e9, 20e9, 3), -30.0, 50.0)
    assert SLD.Parameters[0, 0, 0] == -4.0000000000000009 + 0j
    assert SLD.Parameters[1, 1, 0] == -4.0000000000000009 + 0j
    assert SLD.Parameters[1, 0, 0] == -3.0000000000000009 + 0j
    assert SLD.Parameters[0, 1, 0] == -3.0000000000000009 + 0j


def test_octave_oracle_shunt_pad_form_large_rpad():
    SLD = SL(_thru(), np.linspace(1e9, 20e9, 3), 49.9, 50.0)
    assert SLD.Parameters[0, 0, 0] == -0.0010010010010010151 + 0j
    assert SLD.Parameters[1, 0, 0] == 0.99899899899899891 + 0j


def test_octave_oracle_pads_at_round_values():
    """R<zref shunt and R>zref series, where both forms happen to agree."""
    shunt = SL(_thru(), np.linspace(1e9, 20e9, 3), 25.0, 50.0)
    assert shunt.Parameters[0, 0, 0] == -0.33333333333333331 + 0j
    assert shunt.Parameters[1, 0, 0] == 0.66666666666666663 + 0j
    series = SL(_thru(), np.linspace(1e9, 20e9, 3), 100.0, 50.0)
    assert series.Parameters[0, 0, 0] == 0.33333333333333331 + 0j
    assert series.Parameters[1, 0, 0] == 0.66666666666666663 + 0j


def test_octave_oracle_cascade_series_and_shunt():
    """A reflective S cascaded with each pad. The residual against Octave is
    ~6e-17, from complex division in combines4p (numpy's Smith form against
    libstdc++'s), so this is pinned to 1e-15 rather than to the bit."""
    P = np.array([
        0.00068385534506368339 - 0.0059593902221289421j,
        0.027194950806199235 - 0.010547683860668504j,
        0.024494421571718647 + 0.011394527151439203j,
        -0.01020614153575335 - 0.0011212887809123519j,
        0.69999999999999996 + 0j,
        0.13722191776007717 - 0.56696377579526569j,
        -0.415019065166086 - 0.21339389243004561j,
        -0.22877526730226416 + 0.26488087335777483j,
        0.69999999999999996 + 0j,
        0.13722191776007717 - 0.56696377579526569j,
        -0.415019065166086 - 0.21339389243004561j,
        -0.22877526730226416 + 0.26488087335777483j,
        0.022406568487696316 + 0.020411353598224383j,
        -0.055419743969223285 - 0.004096990019304832j,
        0.046996463240985616 - 0.011372957012245598j,
        -0.0028929648046686165 + 0.013893304757927601j]).reshape(2, 2, 4)
    f = np.linspace(1e9, 25e9, 4)

    S = SimpleNamespace(Parameters=P.copy(), Frequencies=f, Impedance=50.0, NumPorts=2)
    got = SL(S, f, 100.0, 50.0).Parameters
    np.testing.assert_allclose(got[0, 0, :], [
        0.33363207337936146 - 0.0026498153428000403j,
        0.34551360738058851 - 0.0047739599220362108j,
        0.34428964655391431 + 0.0051478797378522161j,
        0.32881246562975419 - 0.00049497690177688672j], rtol=0, atol=1e-15)
    np.testing.assert_allclose(got[1, 0, :], [
        0.46677122568277646 - 0.00092743536998001396j,
        0.090963647617075483 - 0.38175628358010938j,
        -0.27840365316511578 - 0.14449983423397045j,
        -0.15193415855249656 + 0.17604512244885995j], rtol=0, atol=1e-15)
    np.testing.assert_allclose(got[1, 1, :], [
        0.18577649747666805 + 0.02008675121873138j,
        -0.15739963286906331 - 0.056076201250759627j,
        0.08935018413734877 + 0.04831695565579703j,
        -0.0088290688293878403 - 0.026366306519953236j], rtol=0, atol=1e-15)

    S = SimpleNamespace(Parameters=P.copy(), Frequencies=f, Impedance=50.0, NumPorts=2)
    got = SL(S, f, 25.0, 50.0).Parameters
    np.testing.assert_allclose(got[0, 0, :], [
        -0.33302420912594388 - 0.0026474003329527013j,
        -0.32133922788023922 - 0.0046039547088531172j,
        -0.32251631829729055 + 0.0049824684712854754j,
        -0.33788469263839271 - 0.00050175870545816751j], rtol=0, atol=1e-15)
    np.testing.assert_allclose(got[1, 0, :], [
        0.46655847319408028 + 0.00092659011653344536j,
        0.091963489203333915 - 0.37425985712547605j,
        -0.27496635721381646 - 0.14007454277826978j,
        -0.15310391723103625 + 0.17713263792804659j], rtol=0, atol=1e-15)


# ============================================================
# COM Octave oracle — the two pass-through branches.
# Octave returns SLD.Parameters identical to S.Parameters for R==0 and for
# R==R_0. MATLAB assigns `SLD=S`, which is a value copy; the port returned
# the caller's object, so a later write through SLD reached back into S.
# ============================================================
def test_passthrough_branches_return_a_copy():
    for R in (0.0, 50.0):
        S = _sparams()
        before = S.Parameters.copy()
        SLD = SL(S, np.linspace(1e9, 20e9, 10), R, 50.0)
        np.testing.assert_array_equal(SLD.Parameters, before)
        assert SLD is not S
        assert SLD.Parameters is not S.Parameters
        SLD.Parameters[0, 0, :] = 99.0
        np.testing.assert_array_equal(S.Parameters, before)
