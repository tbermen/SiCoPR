"""Verification tests for OptFom_Calc_Noise_XC().

# ============================================================
# MATLAB GROUND TRUTH (lines 3010-3033)
# WIENER-HOPF only:
#   H_ctf_xc = H_low * ctle_gain; H_rx_ctle = H_r * H_ctf
#   P = H_rx_ctle .* conj(H_rx_ctle)
#   XC = ifft(P, 2*N, 'symmetric')  → real(ifft([P, zeros(N)]))
#   Noise_XC = eta_0*f_end/1e9 * XC(1:spu:N_fft_by2)
# Other methods → empty list
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Calc_Noise_XC.py_impl import OptFom_Calc_Noise_XC


def _op(method='WIENER-HOPF', white=False):
    return SimpleNamespace(FFE_OPT_METHOD=method, Do_White_Noise=white)


def _settings(N=32, M=4):
    N_fft_by2 = N
    return SimpleNamespace(H_r_xc=np.ones(N), f_xc=np.linspace(0, 10e9, N),
                           N_fft_by2=N_fft_by2)


def _param(eta_0=1e-13, M=4):
    return SimpleNamespace(eta_0=eta_0, samples_per_ui=M)


def test_non_wiener_hopf_returns_empty():
    """Non-WIENER-HOPF method returns empty list."""
    S = _settings()
    out = OptFom_Calc_Noise_XC(np.ones(32), np.ones(32), S, _param(), _op('MMSE'))
    assert len(out) == 0


def test_output_length_wiener_hopf():
    """WIENER-HOPF output has correct length: ceil(N_fft_by2 / spu)."""
    N, M = 32, 4
    S = _settings(N, M)
    out = OptFom_Calc_Noise_XC(np.ones(N), np.ones(N), S, _param(M=M), _op())
    expected_len = len(range(0, N, M))
    assert len(out) == expected_len


def test_do_white_noise_returns_single():
    """Do_White_Noise=True → output is length 1."""
    N, M = 32, 4
    S = _settings(N, M)
    out = OptFom_Calc_Noise_XC(np.ones(N), np.ones(N), S, _param(M=M), _op(white=True))
    assert len(out) == 1


def test_output_is_real():
    """Output array contains only real (or nearly real) values."""
    N, M = 64, 8
    S = _settings(N, M)
    rng = np.random.default_rng(42)
    H = rng.random(N) + 1j * rng.random(N)
    out = OptFom_Calc_Noise_XC(H, np.ones(N), S, _param(M=M), _op())
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(np.imag(out), 0.0, atol=1e-10)


def test_case_insensitive_method():
    """Method name comparison is case-insensitive."""
    N, M = 16, 4
    S = _settings(N, M)
    out = OptFom_Calc_Noise_XC(np.ones(N), np.ones(N), S, _param(M=M), _op('wiener-hopf'))
    assert len(out) > 0
