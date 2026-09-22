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


# ============================================================
# COM Octave oracle values — 20*log10 evaluated under Octave on the same
# operands.  Pinned 2026-09-22.
#
# Two divergences these pin, both on the 20*log10(A_s/x) sites (equation
# 93A-36 and the C2M A_s/N_i site):
#  * a zero divisor.  MATLAB carries Inf (and NaN for 0/0); the port did the
#    division in Python floats and raised ZeroDivisionError.
#  * a negative ratio.  MATLAB's log10 goes COMPLEX; numpy's on a float is NaN.
# The C2M path also carried an `if N_i <= 0: return None, 1` guard that the
# reference does not have, turning those two cases into a skip.
#
# COM Octave:
#   20*log10( 0.5 /  0   ) -> Inf
#   20*log10( 0.0 /  0   ) -> NaN
#   20*log10(-0.5 /  0.05) -> 20 + 27.287527076836827i
#   20*log10( 0.5 / -0.05) -> 20 + 27.287527076836827i
#   20*log10(-0.25/  0.05) -> 13.979400086720377 + 27.287527076836827i
#   20*log10( 0.5 /  0.05) -> 20            (unchanged, real)
# ============================================================

def test_octave_zero_total_noise_rms_is_inf():
    FOM, skip = OptFom_Calc_FOM(None, False, _THIS(A_s=0.5, total_noise_rms=0.0),
                                _param(), _op(), None)
    assert FOM == float('inf')
    assert skip == 0


def test_octave_zero_over_zero_is_nan():
    FOM, skip = OptFom_Calc_FOM(None, False, _THIS(A_s=0.0, total_noise_rms=0.0),
                                _param(), _op(), None)
    assert np.isnan(FOM)
    assert skip == 0


@pytest.mark.parametrize('A_s,tnr,expected', [
    (-0.5, 0.05, complex(20.0, 27.287527076836827)),
    (0.5, -0.05, complex(20.0, 27.287527076836827)),
    (-0.25, 0.05, complex(13.979400086720377, 27.287527076836827)),
])
def test_octave_negative_ratio_is_complex(A_s, tnr, expected):
    FOM, skip = OptFom_Calc_FOM(None, False, _THIS(A_s=A_s, total_noise_rms=tnr),
                                _param(), _op(), None)
    assert isinstance(FOM, complex)
    # numpy's complex log10 and Octave's round a few ulps apart (the imaginary
    # part is 20*pi/ln(10)), so both parts are pinned relatively.  What matters
    # here is that the result is COMPLEX at all, where the port gave NaN.
    assert FOM.real == pytest.approx(expected.real, rel=1e-15)
    assert FOM.imag == pytest.approx(expected.imag, rel=1e-15)
    assert skip == 0


def test_octave_positive_ratio_stays_a_real_float():
    """Guard the other side: the ordinary case must not become complex."""
    FOM, skip = OptFom_Calc_FOM(None, False, _THIS(A_s=0.5, total_noise_rms=0.05),
                                _param(), _op(), None)
    assert isinstance(FOM, float)
    assert FOM == 20.0
