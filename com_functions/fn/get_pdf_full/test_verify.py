"""Tests for get_pdf_full (MATLAB lines ~7520).

MATLAB GROUND TRUTH:
  THRU channel, flat pulse → pdf_list has entry at the cursor phase.
  h_j_full shape: (n_ui-2, samp_UI) ~ (n_rows, samp_UI).
  A_s_vec: length samp_UI (one UI of signal).
  Non-THRU (FEXT/NEXT): all phases included, A_s_vec=None.
"""
import numpy as np
import pytest
from types import SimpleNamespace
from com_functions.fn.get_pdf_full.py_impl import get_pdf_full


def _param(samp_UI=8):
    p = SimpleNamespace()
    p.samples_per_ui = 4
    p.samples_for_C2M = samp_UI
    p.levels = 4
    p.R_LM = 1.0
    p.ndfe = 2
    p.N_bmax = 2
    p.Floating_DFE = False
    p.bmax = np.array([0.9, 0.9])
    p.bmin = np.array([-0.9, -0.9])
    p.dfe_delta = 0
    p.specBER = 1e-4
    return p


def _make_thru_chdata(pulse_len=64, samp_UI=4):
    ch = SimpleNamespace()
    pulse = np.zeros(pulse_len)
    pulse[pulse_len // 2] = 1.0
    ch.eq_pulse_response = pulse
    ch.type = 'THRU'
    return ch


def test_thru_returns_a_s_vec():
    samp_UI = 8
    p = _param(samp_UI)
    p.samples_per_ui = 4
    chdata = _make_thru_chdata(128, p.samples_per_ui)
    cursor_i = 64
    pdf_list, h_j, A_s_vec = get_pdf_full(chdata, 1e-4, cursor_i, p, SimpleNamespace(), None)
    assert A_s_vec is not None
    assert len(A_s_vec) == samp_UI


def test_fext_returns_none_a_s():
    samp_UI = 8
    p = _param(samp_UI)
    p.samples_per_ui = 4
    chdata = SimpleNamespace()
    pulse = np.zeros(128)
    pulse[40] = 0.5
    chdata.eq_pulse_response = pulse
    chdata.type = 'FEXT'
    cursor_i = 40
    pdf_list, h_j, A_s_vec = get_pdf_full(chdata, 1e-4, cursor_i, p, SimpleNamespace(), None)
    assert A_s_vec is None


def test_h_j_full_shape():
    samp_UI = 8
    p = _param(samp_UI)
    p.samples_per_ui = 4
    chdata = _make_thru_chdata(128, p.samples_per_ui)
    cursor_i = 64
    pdf_list, h_j, A_s_vec = get_pdf_full(chdata, 1e-4, cursor_i, p, SimpleNamespace(), None)
    assert h_j.shape[1] == samp_UI
    assert h_j.ndim == 2


def test_pdf_list_length():
    samp_UI = 8
    p = _param(samp_UI)
    p.samples_per_ui = 4
    chdata = _make_thru_chdata(128, p.samples_per_ui)
    cursor_i = 64
    pdf_list, h_j, A_s_vec = get_pdf_full(chdata, 1e-4, cursor_i, p, SimpleNamespace(), None)
    assert len(pdf_list) == samp_UI


def test_thru_all_phases_computed():
    samp_UI = 8
    p = _param(samp_UI)
    p.samples_per_ui = 4
    chdata = _make_thru_chdata(128, p.samples_per_ui)
    cursor_i = 64
    pdf_list, h_j, A_s_vec = get_pdf_full(chdata, 1e-4, cursor_i, p, SimpleNamespace(), None)
    non_none = [x for x in pdf_list if x is not None]
    # MATLAB loops all pdf_range (1:samp_UI) for both THRU and non-THRU
    assert len(non_none) == samp_UI, "all phases should be computed"
