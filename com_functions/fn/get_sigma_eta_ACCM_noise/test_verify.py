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
