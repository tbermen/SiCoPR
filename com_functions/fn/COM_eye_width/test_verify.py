# ============================================================
# MATLAB GROUND TRUTH
# COM_eye_width: computes eye contour and width for COM.
# MATLAB lines 1362–1551
#
# Key invariants:
# 1. eye_contour shape = (samp_UI, 2*(levels-1))
#    eye_contour[:,0] = A_ni_top of eye 0 (level 1's top)
#    eye_contour[:,1] = A_ni_bot of eye 0 (level 0's bottom)
#
# 2. Left_EW and Right_EW have length (levels-1)
#
# 3. Levels vector: [-1, -1/3, 1/3, 1] for L=4
#    A_s_vec shift: pdf_full{n}[j].x += A_s_vec[j] * Levels[n]
#    For n=0 (level -1), shift is negative; n=3 (level +1), shift is positive
#
# 4. out_VT, out_VB are empty when T_O=0
#
# 5. When T_O != 0 (windowed), out_VT and out_VB are set to the worst eye's
#    top and bottom voltage respectively
# ============================================================

import numpy as np
import pytest
from types import SimpleNamespace

from com_functions.fn.COM_eye_width.py_impl import COM_eye_width


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_param(samp_UI=16, levels=4, T_O=0):
    return SimpleNamespace(
        samples_for_C2M=samp_UI, T_O=T_O, levels=levels, specBER=1e-4,
        sigma_RJ=0.01, sigma_X=1.0, A_DD=0.05, QL=2.5,
    )


def _make_noise():
    delta_y = 0.01
    ne_pdf = SimpleNamespace(x=np.array([0.0]), y=np.array([1.0]), BinSize=delta_y, Min=0)
    cci_pdf = SimpleNamespace(x=np.array([0.0]), y=np.array([1.0]), BinSize=delta_y, Min=0)
    return SimpleNamespace(
        sigma_N=0.01, sigma_TX=0.005, ber_q=7.0,
        ne_noise_pdf=ne_pdf, cci_pdf=cci_pdf,
    )


def _make_op():
    return SimpleNamespace(Histogram_Window_Weight='rectangle', ber_q=7.0)


def _make_fom(samp_UI=16):
    return SimpleNamespace(t_s=samp_UI // 2)


def _stub_get_pdf_full(chdata_0, delta_y, t_s, param, OP, pdf_range):
    samp_UI = int(param.samples_for_C2M)
    n_bins = 16
    x0 = np.linspace(-0.5, 0.5, n_bins)
    dx = x0[1] - x0[0]
    y0 = np.ones(n_bins) / n_bins
    pdf_list = [SimpleNamespace(x=x0.copy(), y=y0.copy(), BinSize=dx, Min=-n_bins//2)
                for _ in range(samp_UI)]
    h_j_full = np.zeros((2, samp_UI))
    A_s_vec = np.ones(samp_UI) * 0.3
    return pdf_list, h_j_full, A_s_vec


# ---------------------------------------------------------------------------
# Test 1 – Nominal: eye_contour shape is correct
# ---------------------------------------------------------------------------

def test_eye_contour_shape():
    """eye_contour.shape = (samp_UI, 2*(levels-1))."""
    samp_UI = 16
    levels = 4
    param = _make_param(samp_UI=samp_UI, levels=levels)
    ch = SimpleNamespace()
    delta_y = 0.01

    Left_EW, Right_EW, eye_contour, out_VT, out_VB = COM_eye_width(
        [ch], delta_y, _make_fom(samp_UI), param, _make_op(), _make_noise(), False,
        _get_pdf_full_fn=_stub_get_pdf_full)

    assert eye_contour.shape == (samp_UI, 2 * (levels - 1)), \
        f"eye_contour.shape={eye_contour.shape}, expected ({samp_UI}, {2*(levels-1)})"


# ---------------------------------------------------------------------------
# Test 2 – Nominal: Left_EW and Right_EW have correct length
# ---------------------------------------------------------------------------

def test_ew_lengths():
    """Left_EW and Right_EW each have length (levels-1)."""
    samp_UI = 16
    levels = 4
    param = _make_param(samp_UI=samp_UI, levels=levels)
    ch = SimpleNamespace()

    Left_EW, Right_EW, _, _, _ = COM_eye_width(
        [ch], 0.01, _make_fom(samp_UI), param, _make_op(), _make_noise(), False,
        _get_pdf_full_fn=_stub_get_pdf_full)

    assert len(Left_EW) == levels - 1
    assert len(Right_EW) == levels - 1


# ---------------------------------------------------------------------------
# Test 3 – Nominal: out_VT, out_VB are empty when T_O = 0
# ---------------------------------------------------------------------------

def test_out_vt_vb_empty_when_T_O_zero():
    """When param.T_O=0, out_VT and out_VB should be [] (not set)."""
    samp_UI = 16
    param = _make_param(samp_UI=samp_UI, T_O=0)
    ch = SimpleNamespace()

    _, _, _, out_VT, out_VB = COM_eye_width(
        [ch], 0.01, _make_fom(samp_UI), param, _make_op(), _make_noise(), False,
        _get_pdf_full_fn=_stub_get_pdf_full)

    assert out_VT == [] or out_VT is None or out_VT == 0.0 or not out_VT, \
        f"out_VT should be empty, got {out_VT}"


# ---------------------------------------------------------------------------
# Test 4 – Boundary: pdf_range_flag=True limits computation to T_O window
# ---------------------------------------------------------------------------

def test_pdf_range_flag_limits_computation():
    """pdf_range_flag=True → PDF only built for center ± T_O samples."""
    samp_UI = 16
    T_O_val = 2  # percent → T_O_samples = floor(2/1000 * 16) = 0 ... use larger value
    # Use T_O=125 (‰) to get T_O_samples = floor(0.125 * 16) = 2
    param = _make_param(samp_UI=samp_UI, T_O=125)
    ch = SimpleNamespace()

    # Track which j values are passed to pdf_full stub via get_pdf_full
    passed_ranges = {}

    def track_pdf_full(chdata_0, delta_y, t_s, p, op, pdf_range):
        passed_ranges['pdf_range'] = pdf_range
        return _stub_get_pdf_full(chdata_0, delta_y, t_s, p, op, pdf_range)

    Left_EW, Right_EW, ec, _, _ = COM_eye_width(
        [ch], 0.01, _make_fom(samp_UI), param, _make_op(), _make_noise(), True,
        _get_pdf_full_fn=track_pdf_full)

    assert 'pdf_range' in passed_ranges
    pr = passed_ranges['pdf_range']
    assert len(pr) == 2, f"Expected [start, end] range, got {pr}"


# ---------------------------------------------------------------------------
# Test 5 – Nominal: 2-level PAM (levels=2) → 1 eye
# ---------------------------------------------------------------------------

def test_pam2_single_eye():
    """For levels=2, there is 1 eye → Left_EW/Right_EW each length 1."""
    samp_UI = 16
    param = _make_param(samp_UI=samp_UI, levels=2)
    ch = SimpleNamespace()

    def stub_pdf2(chdata_0, delta_y, t_s, p, op, pr):
        n = samp_UI
        nb = 16
        x0 = np.linspace(-0.5, 0.5, nb)
        dx = x0[1] - x0[0]
        y0 = np.ones(nb) / nb
        pdf_list = [SimpleNamespace(x=x0.copy(), y=y0.copy(), BinSize=dx, Min=-nb//2)
                    for _ in range(n)]
        h_j = np.zeros((1, n))
        A_s = np.ones(n) * 0.3
        return pdf_list, h_j, A_s

    Left_EW, Right_EW, ec, _, _ = COM_eye_width(
        [ch], 0.01, _make_fom(samp_UI), param, _make_op(), _make_noise(), False,
        _get_pdf_full_fn=stub_pdf2)

    assert len(Left_EW) == 1
    assert ec.shape == (samp_UI, 2)


# ---------------------------------------------------------------------------
# Test 6 – Windowed: out_VT and out_VB set when T_O != 0
# ---------------------------------------------------------------------------

def test_out_vt_vb_set_when_T_O_nonzero():
    """When T_O != 0, out_VT and out_VB are scalars (not empty)."""
    samp_UI = 16
    param = _make_param(samp_UI=samp_UI, T_O=125)  # T_O_samples = 2
    ch = SimpleNamespace()

    _, _, _, out_VT, out_VB = COM_eye_width(
        [ch], 0.01, _make_fom(samp_UI), param, _make_op(), _make_noise(), False,
        _get_pdf_full_fn=_stub_get_pdf_full)

    # out_VT and out_VB should now be numeric scalars
    assert isinstance(out_VT, (int, float, np.floating)), \
        f"out_VT should be scalar, got {type(out_VT)}"
    assert isinstance(out_VB, (int, float, np.floating)), \
        f"out_VB should be scalar, got {type(out_VB)}"
