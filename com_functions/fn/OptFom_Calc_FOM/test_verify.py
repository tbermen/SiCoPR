"""Verification tests for OptFom_Calc_FOM().

# ============================================================
# MATLAB GROUND TRUTH (lines 2817-2873)
# do_C2M=False: FOM = 20*log10(A_s / total_noise_rms).
# do_C2M=True: raises NotImplementedError (pending dependencies).
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.OptFom_Calc_FOM.py_impl import OptFom_Calc_FOM


def _THIS(A_s=1.0, total_noise_rms=0.1, sigma_N=0.05, sigma_TX=0.03, cursor_i=40):
    return SimpleNamespace(A_s=A_s, total_noise_rms=total_noise_rms,
                           sigma_N=sigma_N, sigma_TX=sigma_TX, cursor_i=cursor_i)


def _param():
    return SimpleNamespace(
        Noise_Crest_Factor=0.0,
        specBER=1e-6,
        Min_VEO_Test=10,
    )


def _op():
    return SimpleNamespace(force_pdf_bin_size=False, BinSize=1e-4)


def test_fom_formula():
    """do_C2M=False: FOM = 20*log10(A_s/total_noise_rms)."""
    THIS = _THIS(A_s=1.0, total_noise_rms=0.1)
    FOM, skip = OptFom_Calc_FOM(None, False, THIS, _param(), _op(), None)
    assert FOM == pytest.approx(20 * np.log10(1.0 / 0.1))


def test_skip_loop_zero_for_no_c2m():
    """skip_loop=0 when do_C2M=False."""
    THIS = _THIS()
    _, skip = OptFom_Calc_FOM(None, False, THIS, _param(), _op(), None)
    assert skip == 0


def test_higher_snr_higher_fom():
    """Higher A_s/total_noise_rms → higher FOM."""
    THIS1 = _THIS(A_s=1.0, total_noise_rms=0.1)
    THIS2 = _THIS(A_s=1.0, total_noise_rms=0.5)
    FOM1, _ = OptFom_Calc_FOM(None, False, THIS1, _param(), _op(), None)
    FOM2, _ = OptFom_Calc_FOM(None, False, THIS2, _param(), _op(), None)
    assert FOM1 > FOM2


def _chdata_1ch():
    """A minimal one-channel chdata. MATLAB L3120 writes chdata(1) before the
    EH_1st skip test, so the C2M path needs a real struct even when it skips."""
    return [SimpleNamespace(eq_pulse_response=np.zeros(4), A=1.0)]


def test_c2m_skips_when_noise_exceeds_signal():
    """do_C2M=True: returns (None, 1) when EH_1st <= threshold (noise >> signal)."""
    # A_s=0.1, total_noise_rms=0.1 → EH_1st ≈ -0.75 << Min_VEO_Test/1000 - 0.001
    THIS = _THIS(A_s=0.1, total_noise_rms=0.1)
    FOM, skip = OptFom_Calc_FOM(_chdata_1ch(), True, THIS, _param(), _op(), None)
    assert skip == 1
    assert FOM is None


def test_callee_sees_sbr_but_caller_chdata_is_untouched():
    """MATLAB passes chdata BY VALUE and returns only [FOM, skip_loop] (L3097).

    So COM_eye_width must see chdata(1).eq_pulse_response == sbr, while the
    caller's chdata must come back exactly as it went in. Getting the second
    half wrong is the process_sxp defect class: five of the eight engine
    defects found by the 208-case correlation were by-reference leaks.
    """
    chdata = _chdata_1ch()
    original = chdata[0]
    sbr = np.array([0.1, 0.9, 0.3, 0.05])
    seen = {}

    def _stub_eye_width(cd, delta_y, res, param, OP, noise, flag):
        seen['eq'] = np.asarray(cd[0].eq_pulse_response, dtype=float).copy()
        return 0.0, 0.0, [], 0.1, -0.1

    FOM, skip = OptFom_Calc_FOM(chdata, True, _THIS(A_s=1.0, total_noise_rms=0.1),
                                _param(), _op(), sbr,
                                _COM_eye_width_fn=_stub_eye_width)
    assert skip == 0
    # the callee sees the write
    np.testing.assert_allclose(seen['eq'], sbr)
    # the caller does not
    assert chdata[0] is original
    np.testing.assert_allclose(chdata[0].eq_pulse_response, np.zeros(4))


def test_caller_chdata_untouched_on_the_skip_path():
    """The skip path returns early; the caller's chdata must still be clean.

    MATLAB writes chdata(1) BEFORE the skip test, so an implementation that
    copies only on the non-skip path would leak here.
    """
    chdata = _chdata_1ch()
    sbr = np.array([0.4, 0.4, 0.4, 0.4])
    FOM, skip = OptFom_Calc_FOM(chdata, True, _THIS(A_s=0.1, total_noise_rms=0.1),
                                _param(), _op(), sbr)
    assert (FOM, skip) == (None, 1)
    np.testing.assert_allclose(chdata[0].eq_pulse_response, np.zeros(4))


def test_unit_snr_gives_zero_db():
    """A_s == total_noise_rms → FOM = 0 dB."""
    THIS = _THIS(A_s=0.5, total_noise_rms=0.5)
    FOM, _ = OptFom_Calc_FOM(None, False, THIS, _param(), _op(), None)
    assert FOM == pytest.approx(0.0)
