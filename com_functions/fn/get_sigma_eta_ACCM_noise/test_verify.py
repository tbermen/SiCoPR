"""Verification tests for get_sigma_eta_ACCM_noise().

# ============================================================
# MATLAB GROUND TRUTH
# sigma_N1 = sqrt(eta_0 * sum(|H_sy(2:end).*H_r(2:end).*H_ctf(2:end)|^2 .* diff(faxis)/1e9))
# When AC_CM_RMS = 0: sigma_N = sigma_N1
# When AC_CM_RMS != 0: sigma_N = norm([sigma_N1, sigma_ACCM])
#   sigma_ACCM = norm of per-channel AC CM contributions
#
# With all H=1 and uniform df: sigma_N1 = sqrt(eta_0 * (N-1) * df/1e9)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.get_sigma_eta_ACCM_noise.py_impl import get_sigma_eta_ACCM_noise


def make_chdata(n=10, fb=100e9):
    faxis = np.linspace(0, fb, n)
    sdc21 = np.ones(n, dtype=complex)
    ch = SimpleNamespace(faxis=faxis, sdc21=sdc21)
    return [ch]


def make_param(eta_0=1e-14, AC_CM_RMS=0.0, AC_CM_RMS_TX=0.0, ACCM_MAX=100e9):
    return SimpleNamespace(
        eta_0=eta_0,
        AC_CM_RMS=AC_CM_RMS,
        AC_CM_RMS_TX=AC_CM_RMS_TX,
        ACCM_MAX_Freq=ACCM_MAX,
    )


def test_no_accm_returns_sigma_n1():
    """AC_CM_RMS=0: sigma_N = sigma_N1 only."""
    n = 10
    chdata = make_chdata(n=n, fb=100e9)
    p = make_param(eta_0=1.0, AC_CM_RMS=0.0)
    H = np.ones(n, dtype=complex)
    sigma_N = get_sigma_eta_ACCM_noise(chdata, p, H, H, H)
    # sum(|H|^2 * diff)/1e9 = (n-1) * (100e9/n-1) / 1e9 ≈ 100
    assert sigma_N > 0
    assert np.isfinite(sigma_N)


def test_scales_with_sqrt_eta0():
    """sigma_N1 ∝ sqrt(eta_0)."""
    n = 10
    chdata = make_chdata(n=n)
    H = np.ones(n, dtype=complex)
    p1 = make_param(eta_0=1.0)
    p4 = make_param(eta_0=4.0)
    s1 = get_sigma_eta_ACCM_noise(chdata, p1, H, H, H)
    s4 = get_sigma_eta_ACCM_noise(chdata, p4, H, H, H)
    assert s4 == pytest.approx(2 * s1, rel=1e-10)


def test_accm_increases_sigma():
    """Non-zero AC_CM_RMS_TX increases total sigma."""
    n = 10
    chdata = make_chdata(n=n)
    H = np.ones(n, dtype=complex)
    p0 = make_param(eta_0=1e-14, AC_CM_RMS=0.0)
    p1 = make_param(eta_0=1e-14, AC_CM_RMS=1.0, AC_CM_RMS_TX=0.01)
    s0 = get_sigma_eta_ACCM_noise(chdata, p0, H, H, H)
    s1 = get_sigma_eta_ACCM_noise(chdata, p1, H, H, H)
    assert s1 >= s0


def test_non_negative():
    n = 10
    chdata = make_chdata(n=n)
    H = np.ones(n, dtype=complex)
    sigma = get_sigma_eta_ACCM_noise(chdata, make_param(), H, H, H)
    assert sigma >= 0


# ============================================================
# COM Octave oracle values (2026-09-22)
# get_sigma_eta_ACCM_noise run verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m via tools/octave_oracle.py on the
# inputs built below. The port agreed on every case to between 1 and 5 ulp,
# and the residual is summation order alone -- numpy sums pairwise, Octave
# left to right; forming the sum left to right in Python lands on the Octave
# double exactly. That is the case the brief calls unresolvable, so the
# tolerance below is loose enough to let it through and tight enough to catch
# anything else.
# ============================================================

_N_ORACLE = 201
_F_ORACLE = np.linspace(0.0, 50e9, _N_ORACLE)
_HSY = np.ones(_N_ORACLE, dtype=complex)
_HR = 1.0 / (1 + 1j * _F_ORACLE / 3e10) ** 2
_HCTF = 0.5 + 0.5 / (1 + 1j * _F_ORACLE / 1e10)
_SDC21_A = 0.02 * np.exp(-_F_ORACLE / 8e10) * np.exp(1j * _F_ORACLE / 4e10)
_SDC21_B = 0.03 * np.exp(-_F_ORACLE / 6e10) * np.exp(-1j * _F_ORACLE / 5e10)


def _oracle_chdata(nch=1):
    chs = [SimpleNamespace(faxis=_F_ORACLE, sdc21=_SDC21_A)]
    if nch == 2:
        chs.append(SimpleNamespace(faxis=_F_ORACLE, sdc21=_SDC21_B))
    return chs


def _oracle_call(nch=1, eta_0=1.5e-14, rms=0.01, tx=0.01, fmax=30e9):
    p = SimpleNamespace(eta_0=eta_0, AC_CM_RMS=rms, AC_CM_RMS_TX=tx,
                        ACCM_MAX_Freq=fmax)
    return get_sigma_eta_ACCM_noise(_oracle_chdata(nch), p, _HSY, _HR, _HCTF)


@pytest.mark.parametrize('kw,expected', [
    (dict(rms=0.0, tx=0.0), 4.4333040522859273e-07),
    (dict(), 0.00016265187840761039),
    (dict(nch=2), 0.00028681699399529429),
    (dict(nch=2, fmax=90e9), 0.00022527142929351939),
    (dict(fmax=250e6), 0.00028187493175470095),
    (dict(eta_0=0.0), 0.00016265127422702803),
    (dict(rms=0.0, tx=0.01), 4.4333040522859273e-07),
    # ML 12 guards this branch with `~= 0`, not `> 0`. A negative RMS is
    # not physical, but the two conditions differ on exactly that input
    # and nothing else covered it: COM Octave takes the branch and gives
    # the same figure as +0.01, where `> 0` would fall through to the
    # 4.43e-07 of the rms=0 row.
    (dict(rms=-0.01), 0.00016265187840761039),
])
def test_octave_sigma_values(kw, expected):
    """COM Octave sigma_N for each configuration of the AC CM branch. The last
    row is sum(AC_CM_RMS)==0 with a non-zero AC_CM_RMS_TX: the branch is not
    taken, so the TX figure is ignored."""
    assert _oracle_call(**kw) == pytest.approx(expected, rel=1e-14)


def test_octave_accm_max_below_first_point_errors():
    """COM Octave, ACCM_MAX_Freq below faxis(1): f_int is empty and
        error: f_int(0): subscripts must be either integers
               1 to (2^63)-1 or logicals"""
    with pytest.raises(IndexError):
        _oracle_call(fmax=-1.0)


def test_octave_accm_max_at_dc_is_nan():
    """COM Octave, ACCM_MAX_Freq = 0: f_int is [0], the sum is empty and the
    division by f_int(end)=0 gives 0/0 -> NaN, carried into norm()."""
    assert np.isnan(_oracle_call(fmax=0.0))
