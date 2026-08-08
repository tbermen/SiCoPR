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


def test_c2m_skips_when_noise_exceeds_signal():
    """do_C2M=True: returns (None, 1) when EH_1st <= threshold (noise >> signal)."""
    # A_s=0.1, total_noise_rms=0.1 → EH_1st ≈ -0.75 << Min_VEO_Test/1000 - 0.001
    THIS = _THIS(A_s=0.1, total_noise_rms=0.1)
    FOM, skip = OptFom_Calc_FOM(None, True, THIS, _param(), _op(), None)
    assert skip == 1
    assert FOM is None


def test_unit_snr_gives_zero_db():
    """A_s == total_noise_rms → FOM = 0 dB."""
    THIS = _THIS(A_s=0.5, total_noise_rms=0.5)
    FOM, _ = OptFom_Calc_FOM(None, False, THIS, _param(), _op(), None)
    assert FOM == pytest.approx(0.0)
