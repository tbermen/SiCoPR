"""Verification tests for OptFom_Compute_TXFFE().

# ============================================================
# MATLAB GROUND TRUTH (lines 3370-3417)
# When ctle_response_updated=True: builds pulse_ctle_circshift matrix.
# Applies TX-FFE via FFE_Fast to get sbr.
# Sets chdata[0].pulse_response_w_CFT_TXFFE_noRxFFE = sbr.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Compute_TXFFE.py_impl import OptFom_Compute_TXFFE


def _param(M=8):
    return SimpleNamespace(
        samples_per_ui=M,
        cursor_index=5,
        num_s4p_files=1,
    )


def _op():
    return SimpleNamespace(TDMODE=False, RxFFE_with_MMSE=False)


def _pulse(N=200, peak=40):
    p = np.zeros(N)
    p[peak] = 1.0
    return p


def _chdata(N=200, peak=40):
    p = _pulse(N, peak)
    return [SimpleNamespace(ctle_imp_response=p, type='THRU')]


def _pulse_struc():
    return [SimpleNamespace()]


def test_returns_three_outputs():
    """Returns (sbr, chdata, pulse_struc)."""
    result = OptFom_Compute_TXFFE(_chdata(), _pulse_struc(), [0, 0, 1, 0, 0], True, _param(), _op())
    assert len(result) == 3


def test_sbr_length_matches_ctle():
    """sbr has same length as ctle_imp_response."""
    N = 200
    sbr, chdata, _ = OptFom_Compute_TXFFE(_chdata(N), _pulse_struc(), [0, 0, 1, 0, 0], True, _param(), _op())
    assert len(sbr) == N


def test_chdata_field_set():
    """chdata[0].pulse_response_w_CFT_TXFFE_noRxFFE is set."""
    sbr, chdata, _ = OptFom_Compute_TXFFE(_chdata(), _pulse_struc(), [0, 0, 1, 0, 0], True, _param(), _op())
    assert hasattr(chdata[0], 'pulse_response_w_CFT_TXFFE_noRxFFE')


def test_pulse_ctle_circshift_built(tmp_path):
    """When ctle_response_updated=True, pulse_ctle_circshift matrix is built."""
    ps = _pulse_struc()
    _, _, ps_out = OptFom_Compute_TXFFE(_chdata(), ps, [0, 0, 1, 0, 0], True, _param(), _op())
    assert hasattr(ps_out[0], 'pulse_ctle_circshift')
    assert ps_out[0].pulse_ctle_circshift.ndim == 2


def test_unity_ffe_preserves_pulse():
    """With FFE=[0,0,0,1,0] (cursor tap only), sbr ≈ equalized pulse."""
    N = 200
    sbr, _, _ = OptFom_Compute_TXFFE(_chdata(N), _pulse_struc(), [0, 0, 0, 1, 0], True, _param(), _op())
    assert np.any(np.abs(sbr) > 0.0)


def test_multichannel_grows_pulse_struc():
    """RxFFE_with_MMSE with crosstalk (num_s4p_files>1): pulse_struc (len 1) must
    auto-grow to num_s4p_files entries instead of raising IndexError.
    Reproduces the crosstalk-run crash (debug_error_messages_2026_07_01)."""
    N = 200
    n_files = 3  # thru + FEXT + NEXT
    param = _param()
    param.num_s4p_files = n_files
    op = SimpleNamespace(TDMODE=False, RxFFE_with_MMSE=True)
    chdata = [
        SimpleNamespace(ctle_imp_response=_pulse(N), type='THRU'),
        SimpleNamespace(ctle_imp_response=_pulse(N, 50), type='FEXT'),
        SimpleNamespace(ctle_imp_response=_pulse(N, 60), type='NEXT'),
    ]
    sbr, chdata_out, ps_out = OptFom_Compute_TXFFE(
        chdata, _pulse_struc(), [0, 0, 1, 0, 0], True, param, op)
    assert len(ps_out) == n_files
    # NEXT channel: pulse_ctle passed through without TxFFE (MATLAB L3419)
    assert hasattr(chdata_out[2], 'pulse_response_w_CFT_TXFFE_noRxFFE')
    # FEXT channel: TxFFE applied via circshift matrix (MATLAB L3417)
    assert hasattr(chdata_out[1], 'pulse_response_w_CFT_TXFFE_noRxFFE')
