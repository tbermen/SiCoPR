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


# ============================================================
# COM Octave oracle values (2026-09-22)
# OptFom_Calc_Noise_XC run verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m via tools/octave_oracle.py.
#
# The WIENER-HOPF branch itself is NOT oracle-able: it calls
# ifft(X,n,'symmetric'), an option Octave does not implement, so the reference
# stops there with "invalid conversion from string to real scalar" at line 16.
# That error is itself the evidence for the case-insensitivity tests below:
# reaching line 16 proves the case label was taken.
# ============================================================


def test_octave_settings_read_before_the_switch():
    """COM Octave: H_r_xc/f_xc/N_fft_by2 are read above the switch, so a
    SETTINGS without them errors for every method:
        OP.FFE_OPT_METHOD='MMSE', SETTINGS without H_r_xc ->
        error: structure has no member 'H_r_xc'
        OptFom_Calc_Noise_XC at line 4 column 1
    """
    S = SimpleNamespace(f_xc=np.linspace(0, 10e9, 8), N_fft_by2=8)
    with pytest.raises(AttributeError):
        OptFom_Calc_Noise_XC(np.ones(8), np.ones(8), S, _param(), _op('MMSE'))


def test_octave_mmse_returns_empty():
    """COM Octave: OP.FFE_OPT_METHOD='MMSE' -> Noise_XC = [] (0x0)."""
    out = OptFom_Calc_Noise_XC(np.ones(8), np.ones(8), _settings(8, 2),
                               _param(M=2), _op('MMSE'))
    assert len(out) == 0


def test_octave_empty_method_returns_empty():
    """COM Octave: OP.FFE_OPT_METHOD='' -> Noise_XC = [] (0x0)."""
    out = OptFom_Calc_Noise_XC(np.ones(8), np.ones(8), _settings(8, 2),
                               _param(M=2), _op(''))
    assert len(out) == 0


@pytest.mark.parametrize('method', ['wiener-hopf', 'Wiener-Hopf', 'WIENER-HOPF'])
def test_octave_method_label_is_case_insensitive(method):
    """COM Octave: each spelling reaches the ifft on line 16, so upper() makes
    the case label case-insensitive and none of them fall through to []."""
    out = OptFom_Calc_Noise_XC(np.ones(8), np.ones(8), _settings(8, 2),
                               _param(M=2), _op(method))
    assert len(out) > 0


# ---------------------------------------------------------------------------
# COM Octave, 2026-09-22.
#
# This branch used to be recorded as "not oracle-able": Octave's ifft has no
# 'symmetric' flag and rejects it, so the reference line could not run. That
# was a failure of spelling, not of capability. ifft(X,n,'symmetric') pads X to
# n, keeps the first n/2+1 entries and infers the rest by conjugate symmetry,
# and building that mirrored spectrum explicitly reproduces it exactly. The
# construction is now in octave/patches/OptFom_Calc_Noise_XC.m, so the branch
# is a first-class oracle again.
#
# It also showed the port was WRONG: real(ifft([P zeros])) is the ifft of the
# HERMITIAN PART of the padded vector, which disagrees by a non-constant ratio
# (1.45, 1.28 on the N=4 probe), not by a factor of 2. The port now uses
# np.fft.irfft([P, 0], 2N) and agrees with Octave to 4.8e-16 relative.
# ---------------------------------------------------------------------------

def test_octave_wiener_hopf_branch_matches_the_reference():
    """COM Octave: Noise_XC[:3] on a 64-point deterministic spectrum."""
    rng = np.random.default_rng(11)
    N = 64
    H_r = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * 0.1
    H_low = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * 0.1
    SETTINGS = SimpleNamespace(H_r_xc=H_r, f_xc=np.linspace(0, 53.125e9, N),
                               N_fft_by2=32)
    param = SimpleNamespace(eta_0=1.2e-9, samples_per_ui=8)
    OP = SimpleNamespace(FFE_OPT_METHOD='WIENER-HOPF', Do_White_Noise=0)
    got = np.asarray(OptFom_Calc_Noise_XC(H_low, 1.7, SETTINGS, param, OP)).ravel()
    np.testing.assert_allclose(
        got[:3],
        [5.1337666600188375e-11, -1.0688349862241651e-11, -2.8214400159142196e-12],
        rtol=1e-14)


def test_symmetric_ifft_is_not_the_hermitian_part():
    """The two forms differ by a NON-constant ratio, so this separates them.

    real(ifft([P zeros])) was the port's old line. If it ever comes back, the
    values above move by ~30-45%, not by a clean factor anyone would notice as
    a unit error.
    """
    P = np.array([0.85, 0.5, 0.1625, 0.0404])
    n = 2 * P.size
    symmetric = np.fft.irfft(np.concatenate([P, [0.0]]), n)
    hermitian = np.real(np.fft.ifft(np.concatenate([P, np.zeros(P.size)]), n))
    ratio = symmetric[:2] / hermitian[:2]
    assert not np.allclose(ratio[0], ratio[1], rtol=1e-3), \
        'the two forms must differ by a non-constant ratio, else this test is vacuous'
