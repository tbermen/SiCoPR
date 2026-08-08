"""Verification tests for OptFom_Update_Best_Setttings() (note: MATLAB typo).

# ============================================================
# MATLAB GROUND TRUTH (lines 3935-3975)
# Copies THIS/param fields to BEST. Computes IR via FFE when not TDMODE.
# ============================================================
"""
import numpy as np
from types import SimpleNamespace
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Update_Best_Setttings.py_impl import OptFom_Update_Best_Setttings


def _THIS():
    return SimpleNamespace(
        txffe=[0, 0, 1, 0, 0],
        tx_index_vector=[0, 0, 1, 0, 0],
        ctle_index=1,
        g_dc=-3.0,
        g_LP_index=0,
        FOM=-25.0,
        cursor_i=40,
        itick=5,
        sigma_N=0.01,
        h_J=0.001,
        A_s=0.9,
        A_p=1.0,
        ISI_N=0.05,
        tail_RSS=0.001,
        dfetaps=np.array([0.1, 0.05]),
    )


def _param(M=8):
    return SimpleNamespace(
        current_ffegain=1.0,
        cursor_index=5,
        samples_per_ui=M,
        use_bmax=[1.0],
        use_bmin=[-1.0],
        Floating_DFE=False,
        Floating_RXFFE=False,
    )


def _op(tdmode=False, rxffe=False):
    return SimpleNamespace(TDMODE=tdmode, RxFFE=rxffe, RxFFE_with_MMSE=False)


def _chdata(M=8):
    ir = np.zeros(200)
    ir[40] = 1.0
    return [SimpleNamespace(ctle_imp_response=ir)]


def test_fom_set():
    """BEST.FOM is updated from THIS.FOM."""
    BEST = SimpleNamespace()
    result = OptFom_Update_Best_Setttings(BEST, _THIS(), np.zeros(200), _chdata(), _param(), _op())
    assert result.FOM == _THIS().FOM


def test_txffe_set():
    """BEST.txffe is copied from THIS."""
    BEST = SimpleNamespace()
    result = OptFom_Update_Best_Setttings(BEST, _THIS(), np.zeros(200), _chdata(), _param(), _op())
    assert result.txffe == _THIS().txffe


def test_ir_set_when_not_tdmode():
    """BEST.IR is computed via FFE when not TDMODE."""
    BEST = SimpleNamespace()
    result = OptFom_Update_Best_Setttings(BEST, _THIS(), np.zeros(200), _chdata(), _param(), _op(tdmode=False))
    assert hasattr(result, 'IR')
    assert len(result.IR) == 200


def test_ir_not_set_in_tdmode():
    """BEST.IR is not set when TDMODE=True."""
    BEST = SimpleNamespace()
    result = OptFom_Update_Best_Setttings(BEST, _THIS(), np.zeros(200), _chdata(), _param(), _op(tdmode=True))
    assert not hasattr(result, 'IR')


def test_sigma_n_set():
    """BEST.sigma_N is copied from THIS."""
    BEST = SimpleNamespace()
    result = OptFom_Update_Best_Setttings(BEST, _THIS(), np.zeros(200), _chdata(), _param(), _op())
    assert result.sigma_N == _THIS().sigma_N
