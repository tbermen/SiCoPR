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


# --------------------------------------------------------------------------
# Against COM Octave's own get_pdf_full.  HELD FOR REVIEW 2026-09-22.
#
# Seven of the eight per-phase PDFs reproduce COM Octave exactly. Two things
# do not, and they are recorded here rather than fixed, because the right fix
# depends on which index convention this function is meant to take and that is
# the owner's call:
#
#   * A_s_vec starts one sample early. Given the cursor that the PDFs agree
#     with (0-based 40 for a MATLAB t_s of 41), SiCoPR returns
#     [0.0331795, 0.0568476, 0.0932374, ...] where COM Octave returns
#     [0.0568476, 0.0932374, ...]. The values are identical, only the starting
#     index differs, so the resampled SBR agrees and the slice does not.
#   * the eighth PDF differs (Min -2830 against -1692, 5661 bins against
#     3385), which is consistent with residual_response being zeroed over that
#     same shifted window.
#
# Neither reaches a result today: get_pdf_full and its only caller
# COM_eye_width have no call sites in the engine. That is not a reason to
# leave it wrong -- it is a reason it can wait for a decision rather than be
# guessed at tonight.
# --------------------------------------------------------------------------

_M, _LEVELS, _NDFE, _T_S1, _DELTA_Y = 8, 4, 2, 41, 1e-4
_OCT_PDF = [(-1719, 3439), (-1028, 2057), (-737, 1475), (-637, 1275),
            (-546, 1093), (-702, 1405), (-1050, 2101), (-1692, 3385)]
_OCT_AS = [0.056847563875723262, 0.093237397139956157, 0.13633762878863839,
           0.17280366024604291]
_OCT_HJ0 = 8.2192832372489316e-08


def _pulse():
    n = _M * 12
    t = np.arange(n, dtype=float)
    p = 0.55 * np.exp(-((t - (_T_S1 - 1)) / 3.5) ** 2)
    p += 0.15 * np.exp(-((t - (_T_S1 - 1 + _M)) / 5.0) ** 2)
    p += 0.05 * np.exp(-((t - (_T_S1 - 1 + 2 * _M)) / 6.0) ** 2)
    p += 0.04 * np.exp(-((t - (_T_S1 - 1 - _M)) / 5.0) ** 2)
    return p


def _run(t_s):
    ch = SimpleNamespace(eq_pulse_response=_pulse(), type='THRU', base='synthetic')
    p = SimpleNamespace(samples_per_ui=_M, levels=_LEVELS, ndfe=_NDFE,
                        bmax=np.full(_NDFE, 0.85), bmin=np.full(_NDFE, -0.85),
                        use_bmax=np.ones(_NDFE), use_bmin=np.ones(_NDFE),
                        R_LM=1, dfe_delta=0, Floating_DFE=0, N_bmax=_NDFE,
                        samples_for_C2M=_M)
    return get_pdf_full(ch, _DELTA_Y, t_s, p,
                        SimpleNamespace(force_pdf_bin_size=0), None)


def test_first_seven_phase_pdfs_match_com_octave():
    pdfs, _h, _a = _run(_T_S1 - 1)
    bad = ['[%d] Min %s/%s bins %d/%d' % (i, pdfs[i].Min, mn,
                                          np.asarray(pdfs[i].y).size, ln)
           for i, (mn, ln) in enumerate(_OCT_PDF[:7])
           if pdfs[i].Min != mn or np.asarray(pdfs[i].y).size != ln]
    assert not bad, 'phases disagreeing with COM Octave: %s' % bad


def test_h_j_full_matches_com_octave():
    _p, h_j, _a = _run(_T_S1 - 1)
    got = float(np.asarray(h_j).ravel()[0])
    assert abs(got - _OCT_HJ0) < 1e-20, (
        'h_j_full[0] is %r, COM Octave gives %r' % (got, _OCT_HJ0))


@pytest.mark.xfail(strict=True, reason=(
    'HELD FOR REVIEW 2026-09-22: A_s_vec starts one sample early. Given the '
    'cursor the PDFs agree with, SiCoPR returns [0.0331795, 0.0568476, ...] '
    'where COM Octave returns [0.0568476, 0.0932374, ...] -- the same values, '
    'a shifted starting index. get_pdf_full has no call site in the engine, so '
    'this reaches no result today; the fix depends on which index convention '
    'the function is meant to take.'))
def test_A_s_vec_starts_at_the_matlab_sample():
    _p, _h, A_s = _run(_T_S1 - 1)
    assert np.allclose(np.asarray(A_s).ravel()[:4], _OCT_AS, rtol=0, atol=1e-15)


@pytest.mark.xfail(strict=True, reason=(
    'HELD FOR REVIEW 2026-09-22: the eighth phase PDF has Min -2830 and 5661 '
    'bins where COM Octave gives -1692 and 3385, consistent with '
    'residual_response being zeroed over the same shifted window as A_s_vec.'))
def test_eighth_phase_pdf_matches_com_octave():
    pdfs, _h, _a = _run(_T_S1 - 1)
    assert pdfs[7].Min == _OCT_PDF[7][0]
    assert np.asarray(pdfs[7].y).size == _OCT_PDF[7][1]
