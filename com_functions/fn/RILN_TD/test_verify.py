"""Verification tests for RILN_TD().

# ============================================================
# MATLAB GROUND TRUTH (lines 4226-4305)
# Computes ILN = FIT.PR - REF.PR over a range from ipeak.
# All-zero sdd21/RIL: uses eps-valued impulse response (s21_to_impulse_DC zero path).
# Non-zero sdd21: raises NotImplementedError (interp_Sparam pending).
# Returns struct with REF, FIT, ILN, FOM, FOM_PDF, SNR_ISI_FOM.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import sicopr  # assembled module (provides s21_to_impulse_DC, Bessel/Butterworth filters)
import com_functions.fn.RILN_TD.py_impl as _mod
from com_functions.fn.RILN_TD.py_impl import RILN_TD

# Inject the real top-level dependencies RILN_TD calls as bare globals.
for _n in dir(sicopr):
    _v = getattr(sicopr, _n)
    if callable(_v) and not _n.startswith('__') and not hasattr(_mod, _n):
        setattr(_mod, _n, _v)


def _op():
    return SimpleNamespace(
        transmitter_transition_time=10e-12,
        EC_PULSE_TOL=0.01,
        EC_REL_TOL=1e-4,
        EC_DIFF_TOL=1e-6,
        ENFORCE_CAUSALITY=0,
        impulse_response_truncation_threshold=1e-3,
        BinSize=1e-4,
        DEBUG=True,  # downgrade anti-causal check to a warning for synthetic channels
    )


def _param(M=8, fb=25e9):
    return SimpleNamespace(
        fb=fb, ui=1.0/fb, samples_per_ui=M,
        fb_BW_cutoff=1.0, BTorder=2, fb_BT_cutoff=1.0,
        sample_dt=1.0/(2*fb),
        levels=4,
        specBER=1e-6,
    )


def _freq(N=100, fmax=50e9):
    return np.linspace(0, fmax, N)


def test_returns_struct():
    """RILN_TD returns a SimpleNamespace with REF, FIT, ILN."""
    f = _freq()
    sdd21 = np.zeros(len(f))
    RIL = np.zeros(len(f))
    result = RILN_TD(sdd21, RIL, f, _op(), _param())
    assert hasattr(result, 'REF')
    assert hasattr(result, 'FIT')
    assert hasattr(result, 'ILN')


def test_ILN_zero_for_identical_channels():
    """ILN ≈ 0 when sdd21 == RIL (both zero)."""
    f = _freq()
    sdd21 = np.zeros(len(f))
    result = RILN_TD(sdd21, sdd21.copy(), f, _op(), _param())
    assert np.allclose(result.ILN, 0.0, atol=1e-10)


def test_FOM_nonnegative():
    """FOM is non-negative."""
    f = _freq()
    sdd21 = np.zeros(len(f))
    result = RILN_TD(sdd21, sdd21.copy(), f, _op(), _param())
    assert result.FOM >= 0


def test_REF_PR_length_positive():
    """REF.PR has positive length."""
    f = _freq()
    sdd21 = np.zeros(len(f))
    result = RILN_TD(sdd21, sdd21.copy(), f, _op(), _param())
    assert len(result.REF.PR) > 0


def _lossy(f, tau=150e-12, loss_db_per_ghz=0.6, scale=1.0):
    mag = scale * np.exp(-(loss_db_per_ghz / 20.0) * np.log(10) * f / 1e9)
    return mag * np.exp(-1j * 2 * np.pi * f * tau)


def test_nonzero_channel_computes_riln():
    """Realistic lossy sdd21/RIL: RILN_TD runs end-to-end (s21_to_impulse_DC is
    implemented now) and returns finite REF/FIT/ILN/FOM (no NotImplementedError)."""
    f = _freq()
    sdd21 = _lossy(f, tau=150e-12)
    RIL = _lossy(f, tau=300e-12, scale=0.2)  # weaker reflective path
    result = RILN_TD(sdd21, RIL, f, _op(), _param())
    assert hasattr(result, 'REF') and hasattr(result, 'FIT') and hasattr(result, 'ILN')
    assert len(result.REF.PR) > 0
    assert np.isfinite(float(result.FOM))


def test_uses_real_s21_to_impulse_DC():
    assert _mod.s21_to_impulse_DC is sicopr.s21_to_impulse_DC


# ===========================================================================
# COM Octave oracle — the whole of RILN_TD executed under the reference
# ===========================================================================
# Every number below came out of the REFERENCE function itself:
# com_ieee8023_4p16p0_octave_compat.m's RILN_TD, run under Octave by
# tools/octave_oracle.py on the inputs _riln_channels()/_riln_run() build here.
# Nothing was computed by the port. Reproduce with:
#
#   from octave_oracle import call
#   call('RILN_TD', args=['sdd21','RIL','faxis_f2','OP','param'],
#        inputs={'sdd21': sdd21, 'RIL': RIL, 'faxis_f2': f},
#        outputs=['R'], setup=<the same struct fields as below, with
#                              faxis_f2 = faxis_f2(:).'>,
#        needs=['RILN_TD','Bessel_Thomson_Filter','Butterworth_Filter','bessel',
#               'Tukey_Window','s21_to_impulse_DC','interp_Sparam',
#               'get_pdf_from_sampled_signal','d_cpdf','conv_fct',
#               'Init_PDF_Fast','normal_dist'])
#
# Two things the workdir also needs. RILN_TD plots unconditionally
# (print_for_codereview = 1), so no-op .m shims for figure/gcf/set/semilogy/
# plot/hold/ylim/xlim/title/grid/legend are required or Octave hangs; and
# `sinc` lives in the signal package, not core Octave, so it needs a shim too
# (its result, X, is never used). octave/patches/com_octave_accel_on.m as well.
#
# faxis_f2 must be a ROW under Octave: sdd21 is forced to a row by the
# function's own iscolumn test, so a COLUMN frequency vector makes
# sdd21.*H_bw.*H_t broadcast to an N-by-N matrix and interp1 then dies inside
# s21_to_impulse_DC. The port ravels everything and is orientation-agnostic.
#
# The channel carries linear phase on purpose. With zero phase the pulse
# response has a numerically flat top and `find(PR==max(PR),1)` picks whichever
# sample the last bits favour, which moves ipeak and with it the whole ILN
# window. The tolerances that remain are the ifft (FFTW vs pocketfft).

_RILN_F = np.arange(0, 41) * 0.5e9          # 0 .. 20 GHz, DC included


def _riln_channels(ripple=0.08, fr=1.2e-9, tau=180e-12,
                   a0=0.5, a1=0.35, a2=0.02):
    """A smooth fitted IL, and a 'measured' IL that is the fit plus ripple."""
    g = _RILN_F / 1e9
    mag = 10 ** (-(a0 + a1 * np.sqrt(g) + a2 * g) / 20.0)
    ph = np.exp(-1j * 2 * np.pi * _RILN_F * tau)
    return mag * (1.0 + ripple * np.cos(2 * np.pi * _RILN_F * fr)) * ph, mag * ph


def _riln_run(sdd21, RIL, fb=10e9, M=4, dt=2.5e-11, BinSize=1e-3,
              levels=4, specBER=1e-4, ttt=0.008):
    param = SimpleNamespace(
        fb=fb, ui=1.0 / fb, samples_per_ui=M, sample_dt=dt, levels=levels,
        specBER=specBER, fb_BT_cutoff=0.75, fb_BW_cutoff=0.75, BTorder=4,
        f_r=0.75)
    OP = SimpleNamespace(
        transmitter_transition_time=ttt, BinSize=BinSize, DISPLAY_WINDOW=False,
        DEBUG=False, EC_PULSE_TOL=0.01, EC_REL_TOL=1e-4, EC_DIFF_TOL=1e-6,
        ENFORCE_CAUSALITY=0, impulse_response_truncation_threshold=1e-3,
        interp_sparam_mag='linear_trend_to_DC',
        interp_sparam_phase='extrap_cubic_to_dc_linear_to_inf', ZERO_PAD=0)
    return RILN_TD(sdd21, RIL, _RILN_F, OP, param)


def test_oracle_base_channel_impulse_and_pulse():
    """COM Octave: REF/FIT impulse and pulse responses, and the causality dB."""
    sdd21, RIL = _riln_channels()
    r = _riln_run(sdd21, RIL)
    assert len(r.REF.FIR) == 80 and len(r.FIT.FIR) == 80     # COM Octave
    np.testing.assert_allclose(r.REF.FIR[0], 0.0011514230238655499, rtol=1e-11)
    np.testing.assert_allclose(r.REF.FIR[12], 0.003342881653556379, rtol=1e-11)
    np.testing.assert_allclose(r.REF.causality_correction_dB,
                               -24.350052983800442, rtol=1e-13)
    np.testing.assert_allclose(r.FIT.causality_correction_dB,
                               -34.879420078653602, rtol=1e-13)
    # norm of the truncated tail is 0 -> 20*log10(0); the reference has no
    # epsilon floor here either, so both sides report -Inf.
    assert r.REF.truncation_dB == -np.inf                    # COM Octave
    assert r.FIT.truncation_dB == -np.inf                    # COM Octave
    ipeak = int(np.argmax(r.REF.PR))
    assert ipeak == 11                                       # COM Octave
    np.testing.assert_allclose(r.REF.PR[ipeak], 0.96714074825492535, rtol=1e-13)
    np.testing.assert_allclose(r.FIT.PR[ipeak], 0.96705383420692392, rtol=1e-13)


def test_oracle_base_ILN_window_and_FOM():
    """COM Octave: the ILN window runs ipeak..range_end INCLUSIVE, 69 samples.

    MATLAB `range = ipeak:range_end` with range_end = min(ipeak+M*1000,
    min(len(FIT.FIR),len(REF.FIR))) is 1-based and inclusive at both ends, so
    with ipeak = 12 (1-based) and 80 samples the window is 69 long.
    """
    sdd21, RIL = _riln_channels()
    r = _riln_run(sdd21, RIL)
    assert len(r.ILN) == 69                                  # COM Octave
    assert len(r.t) == 69                                    # COM Octave
    np.testing.assert_allclose(r.t[0], 2.7499999999999998e-10, rtol=1e-15)
    np.testing.assert_allclose(r.t[-1], 1.9749999999999999e-09, rtol=1e-15)
    np.testing.assert_allclose(r.ILN[0], -8.6914048001429656e-05, rtol=1e-11)
    np.testing.assert_allclose(r.ILN[1], -8.789977536483029e-05, rtol=1e-11)
    np.testing.assert_allclose(r.ILN[-1], -0.00010284345707397636, rtol=1e-11)
    np.testing.assert_allclose(r.FOM, 0.055015401343908435, rtol=1e-13)
    np.testing.assert_allclose(r.FOM_PDF, 0.085000000000000006, rtol=1e-15)
    np.testing.assert_allclose(r.SNR_ISI_FOM, 24.899327312767717, rtol=1e-13)
    np.testing.assert_allclose(r.SNR_ISI_FOM_PDF, 21.120634509219748, rtol=1e-13)


def test_oracle_base_PDF():
    """COM Octave: the ISI PDF kept for the worst rms phase."""
    sdd21, RIL = _riln_channels()
    r = _riln_run(sdd21, RIL)
    assert r.PDF.Min == -85                                  # COM Octave
    assert len(r.PDF.y) == 171                               # COM Octave
    np.testing.assert_allclose(r.PDF.x[0], -0.085000000000000006, rtol=1e-15)
    np.testing.assert_allclose(r.PDF.y[0], 0.0009765625, rtol=1e-14)
    np.testing.assert_allclose(r.PDF.BinSize, 1e-3, rtol=1e-15)


def test_oracle_lossier_channel_bigger_ripple():
    """COM Octave: a lossier channel with 25% ripple moves ipeak to 15 (1-based)."""
    sdd21, RIL = _riln_channels(ripple=0.25, fr=0.6e-9, a1=0.9, a2=0.05,
                                tau=260e-12)
    r = _riln_run(sdd21, RIL)
    assert int(np.argmax(r.REF.PR)) == 14                    # COM Octave
    assert len(r.ILN) == 66                                  # COM Octave
    np.testing.assert_allclose(r.REF.PR[14], 0.85085105951743789, rtol=1e-13)
    np.testing.assert_allclose(r.FIT.PR[14], 0.85003912808031756, rtol=1e-13)
    np.testing.assert_allclose(r.REF.causality_correction_dB,
                               -13.69877999769313, rtol=1e-13)
    np.testing.assert_allclose(r.ILN[0], -0.00081193143712032789, rtol=1e-11)
    np.testing.assert_allclose(r.ILN[-1], -0.0017626814509356173, rtol=1e-11)
    np.testing.assert_allclose(r.FOM, 0.15109453847877821, rtol=1e-13)
    np.testing.assert_allclose(r.FOM_PDF, 0.23399999999999999, rtol=1e-15)
    np.testing.assert_allclose(r.SNR_ISI_FOM, 15.003803013999107, rtol=1e-13)
    np.testing.assert_allclose(r.SNR_ISI_FOM_PDF, 11.204461194747919, rtol=1e-13)
    assert r.PDF.Min == -244 and len(r.PDF.y) == 489         # COM Octave
    np.testing.assert_allclose(r.PDF.y[0], 3.7252902984619141e-09, rtol=1e-12)


def test_oracle_samples_per_ui_8():
    """COM Octave: 8 samples per UI — the phase loop and the M-tap pulse filter."""
    sdd21, RIL = _riln_channels()
    r = _riln_run(sdd21, RIL, fb=5e9, M=8)
    assert int(np.argmax(r.REF.PR)) == 16                    # COM Octave
    assert len(r.ILN) == 64                                  # COM Octave
    np.testing.assert_allclose(r.REF.PR[16], 0.98955002216309551, rtol=1e-13)
    np.testing.assert_allclose(r.FIT.PR[16], 0.98939421659118132, rtol=1e-13)
    np.testing.assert_allclose(r.t[0], 4.0000000000000001e-10, rtol=1e-15)
    np.testing.assert_allclose(r.ILN[0], -0.00015580557191419153, rtol=1e-11)
    np.testing.assert_allclose(r.ILN[-1], -0.0010888948106986172, rtol=1e-11)
    np.testing.assert_allclose(r.FOM, 0.056723737341683005, rtol=1e-13)
    np.testing.assert_allclose(r.FOM_PDF, 0.087000000000000008, rtol=1e-15)
    np.testing.assert_allclose(r.SNR_ISI_FOM, 24.832090600941218, rtol=1e-13)
    np.testing.assert_allclose(r.SNR_ISI_FOM_PDF, 21.117002295806046, rtol=1e-13)
    assert r.PDF.Min == -87 and len(r.PDF.y) == 175          # COM Octave


def test_oracle_inverted_fit_gives_negative_SNR_not_nan():
    """COM Octave: MATLAB's db() takes abs(), so a sign-flipped fit is finite.

    RILN_TD's local `db = @(x) 20*log10(abs(x))` is applied to
    FIT.PR(ipeak)/FOM. With an inverted fit that ratio is negative and the
    reference reports -6.0602241763211184 dB. The port dropped the abs() and
    guarded with `FOM > 0` instead, so np.log10 of a negative gave nan for both
    SNR_ISI_FOM and SNR_ISI_FOM_PDF.
    """
    sdd21, RIL = _riln_channels()
    r = _riln_run(sdd21, -RIL)
    ipeak = int(np.argmax(r.REF.PR))
    assert ipeak == 11
    np.testing.assert_allclose(r.FIT.PR[ipeak], -0.96705383420692392, rtol=1e-13)
    np.testing.assert_allclose(r.FIT.causality_correction_dB,
                               -34.87000143628017, rtol=1e-13)
    np.testing.assert_allclose(r.ILN[0], -1.9341945824618492, rtol=1e-12)
    np.testing.assert_allclose(r.FOM, 1.9429510531785648, rtol=1e-13)
    np.testing.assert_allclose(r.FOM_PDF, 2.242, rtol=1e-15)
    np.testing.assert_allclose(r.SNR_ISI_FOM, -6.0602241763211184, rtol=1e-13)
    np.testing.assert_allclose(r.SNR_ISI_FOM_PDF, -7.3036991416734809, rtol=1e-13)
    assert r.PDF.Min == -2271                                # COM Octave


def test_oracle_identical_channels_give_infinite_SNR():
    """COM Octave: sdd21 == RIL gives FOM = 0, FOM_PDF = -0 and SNR = +Inf.

    20*log10(abs(x/0)) is +Inf in MATLAB; the port's `FOM > 0` guard happened to
    reach the same answer here, but only by short-circuiting.
    """
    sdd21, _ = _riln_channels()
    r = _riln_run(sdd21, sdd21.copy())
    assert not np.any(r.ILN)                                 # COM Octave
    assert r.FOM == 0.0                                      # COM Octave
    assert r.FOM_PDF == 0.0                                  # COM Octave (-0)
    assert r.SNR_ISI_FOM == np.inf                           # COM Octave
    assert r.SNR_ISI_FOM_PDF == np.inf                       # COM Octave
    assert r.PDF.Min == 0 and len(r.PDF.y) == 1              # COM Octave
    np.testing.assert_allclose(r.PDF.y[0], 1.0, rtol=1e-15)
