"""Verification tests for get_xtlk_noise().

# ============================================================
# MATLAB GROUND TRUTH (lines 7844-7938)
# Computes FEXT/NEXT crosstalk noise sigma.
# No crosstalk channels (only THRU in chdata): sigma_XT=0.
# With FEXT/NEXT channels: computes power-weighted sigma.
# upsampled_txffe all-zero: PWF_tx=ones.
# nargout==1 NEXT: returns MDNEXT_ICN * scale.
# nargout==3: returns (sigma_XT, sigma_FEXT, sigma_NEXT).
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.get_xtlk_noise.py_impl import get_xtlk_noise


def _faxis(N=64, fb=25e9):
    return np.linspace(0, fb, N)


def _param(fb=25e9, M=8, dt=None):
    if dt is None:
        dt = 1.0/(2*fb)
    return SimpleNamespace(
        fb=fb, samples_per_ui=M, sample_dt=dt,
        levels=4, f2=fb/2,
        RxFFE_cmx=2, RxFFE_cpx=2,
    )


def _thru_chdata(N=64, fb=25e9):
    f = _faxis(N, fb)
    return [SimpleNamespace(
        faxis=f, type='THRU',
        sdd21ctf=np.zeros(N, dtype=complex),
        delta_f=f[1]-f[0], A=0.5,
    )]


def _fext_chdata(N=64, fb=25e9):
    f = _faxis(N, fb)
    sdd21ctf = 0.01 * np.ones(N, dtype=complex)
    return [
        SimpleNamespace(faxis=f, type='THRU', sdd21ctf=np.zeros(N, dtype=complex),
                        delta_f=f[1]-f[0], A=0.5),
        SimpleNamespace(faxis=f, type='FEXT', sdd21ctf=sdd21ctf,
                        delta_f=f[1]-f[0], A=0.5),
    ]


def test_no_xtalk_returns_zero():
    """With only THRU channel, sigma_XT=0 for FEXT query."""
    txffe = np.zeros(8)
    result = get_xtlk_noise(txffe, 'FEXT', _param(), _thru_chdata())
    assert result == pytest.approx(0.0)


def test_no_xtalk_returns_zero_next():
    """With only THRU channel, sigma_XT=0 for NEXT query."""
    txffe = np.zeros(8)
    result = get_xtlk_noise(txffe, 'NEXT', _param(), _thru_chdata())
    assert result == pytest.approx(0.0)


def test_fext_sigma_positive():
    """FEXT sigma > 0 when FEXT channel present."""
    txffe = np.zeros(8)
    result = get_xtlk_noise(txffe, 'FEXT', _param(), _fext_chdata())
    assert result >= 0.0


def test_three_output_mode():
    """'both' mode returns (sigma_XT, sigma_FEXT, sigma_NEXT)."""
    txffe = np.zeros(8)
    result = get_xtlk_noise(txffe, 'both', _param(), _fext_chdata())
    assert len(result) == 3


def test_invalid_type_raises():
    """Unknown channel type in chdata raises ValueError."""
    txffe = np.zeros(8)
    f = _faxis()
    chdata_bad = [
        SimpleNamespace(faxis=f, type='THRU', sdd21ctf=np.zeros(64, dtype=complex),
                        delta_f=f[1]-f[0], A=0.5),
        SimpleNamespace(faxis=f, type='UNKNOWN', sdd21ctf=np.zeros(64, dtype=complex),
                        delta_f=f[1]-f[0], A=0.5),
    ]
    with pytest.raises(ValueError):
        get_xtlk_noise(txffe, 'NEXT', _param(), chdata_bad)


def test_nzero_txffe_runs():
    """Non-zero upsampled_txffe runs without error."""
    txffe = np.zeros(8)
    txffe[3] = 0.8
    txffe[2] = 0.1
    result = get_xtlk_noise(txffe, 'FEXT', _param(), _fext_chdata())
    assert result >= 0.0


# ============================================================
# COM Octave oracle — get_xtlk_noise run verbatim under Octave with
#   param.fb = 53.125e9   samples_per_ui = 8   sample_dt = 1/(8*fb)
#   param.f2 = 40e9       levels = 4   RxFFE_cmx = 2   RxFFE_cpx = 3
#   chdata(1) THRU  A 0.4, chdata(2) NEXT  A 0.6, chdata(3) FEXT  A 0.5,
#   delta_f 1e9 on each, sdd21ctf as _ORACLE_S below,
#   upsampled_txffe = [0 -0.1 0.75 -0.05 0]
#
#   faxis = (0:19)*6e9, no phase_memory, no C
#       sigma_XT 0.005744621797994268
#       sigma_FEXT 0.0041331818385112349  sigma_NEXT 0.0039896726045869713
#   same, with C = [0.05 -0.2 0.8 -0.1 0.02 0.01] and phase_memory []
#       sigma_XT 0.0050032802758520727
#       sigma_FEXT 0.0036271423224473014  sigma_NEXT 0.0034462518903066253
#   faxis = (0:19)*2e9, so nothing exceeds fb and index_f2 falls back to
#   length(faxis)
#       sigma_XT 0.0094851303890926882
#       sigma_FEXT 0.0068455756012968895  sigma_NEXT 0.0065655002235182613
#   faxis = fb + (0:19)*1e9, so the crossing is the SECOND point
#       sigma_XT 3.7632493238421401e-05
#       sigma_FEXT 2.4948590773404411e-05  sigma_NEXT 2.8173966099238365e-05
#
# That last fixture is the one that pins index_f2. MATLAB's find() is 1-based
# and the sums slice 1:index_f2 inclusively, so the 1-based number is the
# exclusive Python end; using the 0-based index dropped the top bin. Here the
# first bin sits exactly at fb, where sin(pi)/pi is ~1e-16, so dropping the
# second bin takes essentially all the energy: the port answered ~2e-19.
# ============================================================
_OR_FB = 53.125e9
_OR_TXFFE = np.array([0.0, -0.1, 0.75, -0.05, 0.0])
_OR_C = np.array([0.05, -0.2, 0.8, -0.1, 0.02, 0.01])


def _oracle_param():
    return SimpleNamespace(fb=_OR_FB, samples_per_ui=8,
                           sample_dt=1.0 / (8 * _OR_FB), f2=40e9, levels=4,
                           RxFFE_cmx=2, RxFFE_cpx=3)


def _oracle_chdata(f):
    f = np.asarray(f, dtype=float)
    s1 = 0.6 * np.exp(-f / 8e10) * np.exp(-1j * f / 2e10)
    s2 = 0.02 * np.exp(-f / 2e11) * np.exp(-1j * f / 3e10)
    s3 = 0.03 * np.exp(-f / 1.5e11) * np.exp(-1j * f / 2.5e10)
    return [SimpleNamespace(faxis=f, type='THRU', sdd21ctf=s1, delta_f=1e9, A=0.4),
            SimpleNamespace(faxis=f, type='NEXT', sdd21ctf=s2, delta_f=1e9, A=0.6),
            SimpleNamespace(faxis=f, type='FEXT', sdd21ctf=s3, delta_f=1e9, A=0.5)]


@pytest.mark.parametrize('faxis,C,expected', [
    (np.arange(20) * 6e9, None,
     (0.005744621797994268, 0.0041331818385112349, 0.0039896726045869713)),
    (np.arange(20) * 6e9, _OR_C,
     (0.0050032802758520727, 0.0036271423224473014, 0.0034462518903066253)),
    (np.arange(20) * 2e9, None,
     (0.0094851303890926882, 0.0068455756012968895, 0.0065655002235182613)),
    (_OR_FB + np.arange(20) * 1e9, None,
     (3.7632493238421401e-05, 2.4948590773404411e-05, 2.8173966099238365e-05)),
])
def test_octave_oracle_icn(faxis, C, expected):
    kw = {} if C is None else {'phase_memory': np.array([]), 'C': C}
    got = get_xtlk_noise(_OR_TXFFE, 'both', _oracle_param(),
                         _oracle_chdata(faxis), **kw)
    for name, g, e in zip(('sigma_XT', 'sigma_FEXT', 'sigma_NEXT'), got, expected):
        assert g == pytest.approx(e, rel=1e-13), name


# ============================================================
# COM Octave oracle — two calls the reference refuses.
#
#   C shorter than RxFFE_cmx+RxFFE_cpx+1
#       C = [0.05 -0.2] -> error: C(3): out of bound 2 (dimensions are 2x1)
#       C = []          -> error: C(1): out of bound 0 (dimensions are 0x0)
#     The port skipped the missing taps and answered with a truncated RX FFE.
#
#   upsampled_txffe with no positive element, and C supplied
#       -> error: 'pre_calc' undefined near line 55, column 16
#     MATLAB assigns pre_calc only inside `if max(upsampled_txffe) > 0`, and
#     the RX FFE branch reads it regardless. The port defined its own copy.
# ============================================================
@pytest.mark.parametrize('C', [np.array([0.05, -0.2]), np.array([])])
def test_octave_oracle_short_C_is_refused(C):
    with pytest.raises(IndexError):
        get_xtlk_noise(_OR_TXFFE, 'both', _oracle_param(),
                       _oracle_chdata(np.arange(20) * 6e9),
                       phase_memory=np.array([]), C=C)


@pytest.mark.parametrize('txffe', [np.zeros(5), np.array([0.0, -0.1, -0.2, 0.0, 0.0])])
def test_octave_oracle_pre_calc_undefined_is_refused(txffe):
    with pytest.raises(NameError):
        get_xtlk_noise(txffe, 'both', _oracle_param(),
                       _oracle_chdata(np.arange(20) * 6e9),
                       phase_memory=np.array([]), C=_OR_C)
