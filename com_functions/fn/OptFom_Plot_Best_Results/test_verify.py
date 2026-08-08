"""Tests for OptFom_Plot_Best_Results (MATLAB lines ~3674).

MATLAB GROUND TRUTH:
  When OP.DEBUG=0: no output, no error.
  When OP.DEBUG=1: prints debug text with FOM, TXFFE, SNR values.
  Function returns None.
"""
import numpy as np
import pytest
from com_functions.fn.OptFom_Plot_Best_Results.py_impl import OptFom_Plot_Best_Results
from types import SimpleNamespace


def _make_best():
    B = SimpleNamespace()
    B.FOM = 3.14
    B.txffe = np.array([0.9, -0.05, -0.05])
    B.A_p = 0.5
    B.ISI = 0.1
    B.gdc = -6.0
    B.ctle_gain = np.array([1.0, 1.2, 0.8])
    B.cursor = 0.5
    B.sbr = np.zeros(100)
    B.sbr[10] = 1.0
    return B


def _make_chdata():
    ch = SimpleNamespace()
    ch.uneq_imp_response = np.zeros(100)
    ch.uneq_imp_response[5] = 1.0
    return ch


def _param():
    p = SimpleNamespace()
    p.samples_per_ui = 4
    p.levels = 4
    return p


def test_returns_none_no_debug():
    result = OptFom_Plot_Best_Results(
        _make_best(), np.linspace(0, 1, 20), np.linspace(0, 1e10, 20),
        [_make_chdata()], _param(),
        SimpleNamespace(DEBUG=0, DISPLAY_WINDOW=False)
    )
    assert result is None


def test_no_error_with_debug(capsys):
    OptFom_Plot_Best_Results(
        _make_best(), np.linspace(0, 1, 20), np.linspace(0, 1e10, 20),
        [_make_chdata()], _param(),
        SimpleNamespace(DEBUG=1, DISPLAY_WINDOW=False)
    )
    out = capsys.readouterr().out
    assert len(out) > 0


def test_no_error_with_zero_isi(capsys):
    B = _make_best()
    B.ISI = 0.0
    OptFom_Plot_Best_Results(
        B, np.linspace(0, 1, 20), np.linspace(0, 1e10, 20),
        [_make_chdata()], _param(),
        SimpleNamespace(DEBUG=1, DISPLAY_WINDOW=False)
    )


def test_no_error_minimal_inputs():
    B = SimpleNamespace(FOM=1.0, sbr=np.zeros(4))
    ch = SimpleNamespace(uneq_imp_response=np.zeros(4))
    p = SimpleNamespace(samples_per_ui=2, levels=2)
    OP = SimpleNamespace(DEBUG=0, DISPLAY_WINDOW=False)
    result = OptFom_Plot_Best_Results(B, np.array([0.0, 1.0]), np.array([0.0, 1e9]),
                                      [ch], p, OP)
    assert result is None


def test_prints_fom_when_debug(capsys):
    B = _make_best()
    B.FOM = 42.0
    OptFom_Plot_Best_Results(
        B, np.linspace(0, 1, 20), np.linspace(0, 1e10, 20),
        [_make_chdata()], _param(),
        SimpleNamespace(DEBUG=1, DISPLAY_WINDOW=False)
    )
    out = capsys.readouterr().out
    assert 'FOM' in out or '42' in out
