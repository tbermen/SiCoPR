"""Audit batch B09: G7 cursor and sample indexing.

Functions under audit (MATLAB com_ieee8023_4p15p0.m -> sicopr.py):
  cursor_sample_index  ML 5537-5611 -> py 8130-8192
  get_center_of_UI     ML 7450-7460 -> py 10596-10606 (+ 2 inlined copies)
  COM_eye_width        ML 1367-1556 -> py 1607-1879
  vma                  ML 11479-11507 -> py 17664-17706

Traps checked (audit prompt section 3 item 7, section 4 item 1):
  - 1-based MATLAB indices vs 0-based Python at every seam (peak search, zero
    crossing, Muller-Mueller offset, cursor).
  - get_center_of_UI return-base convention across its three copies.
  - eye width/height nonnegative and bounded by 1 UI / peak signal.
  - vma index formulas and P_3 > P_0.

Oracles transcribed from the cited MATLAB lines; cursor / eye bounds are physics
invariants. Run: python tests/test_cursor_indexing.py
"""
import os
import sys
import copy
from types import SimpleNamespace

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, finish  # noqa: E402
import sicopr  # noqa: E402


def rel_err(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = max(np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(a - b)) / denom)


# ===========================================================================
# 1. get_center_of_UI: three copies, return-base conventions (ML 7450-7460)
# ===========================================================================
for su in (16, 32, 30, 31, 64):
    top = sicopr.get_center_of_UI(su)                     # 0-based (argmin)
    gpf = sicopr._get_pdf_full__get_center_of_UI(su)      # 1-based (M//2+1)
    cew = sicopr._COM_eye_width__get_center_of_UI(su)     # 0-based (M//2)
    ui = np.arange(su) / su
    argmin0 = int(np.argmin(np.abs(ui - 0.5)))         # true 0-based center
    ok = (top == argmin0) and (gpf == top + 1) and (cew == su // 2) and (cew == argmin0)
    check("get_center_of_UI_conventions_su%d" % su, ok,
          "su=%d: top=%d(exp %d) gpf=%d(exp %d) cew=%d" % (su, top, argmin0, gpf, argmin0 + 1, cew))

# ===========================================================================
# 2. cursor_sample_index (ML 5537-5611 vs py 8130-8192)
# ===========================================================================
M = 16
P = 12 * M                                              # peak index
n = 30 * M
tt = np.arange(n)
# asymmetric pulse: fast rise, slower decay (COM-like main cursor)
sbr = np.where(tt <= P, np.exp(-((tt - P) / (1.5 * M)) ** 2),
               np.exp(-((tt - P) / (2.5 * M)) ** 2)) * 1.0
sbr += 1e-3 * np.sin(tt / 3.0)                          # small ripple
psr = np.arange(P - 4 * M, P + 4 * M)                   # 0-based search window

param_c = SimpleNamespace(samples_per_ui=M, ndfe=1, bmax=np.array([0.2, 0.1]))
OP_c = SimpleNamespace(CDR='MM')
cursor_i, nzc, peak_i, zxi_arr = sicopr.cursor_sample_index(sbr, param_c, OP_c, psr)


def cursor_oracle(sbr, M, ndfe, bmax, CDR, psr):
    """Transcription of ML 5558-5611 with 1-based -> 0-based conversion."""
    sub = sbr[psr]
    peak_tmp = int(np.argmax(sub))
    max_of = float(sub[peak_tmp])
    peak_i = peak_tmp + int(psr[0])
    ss = max(0, peak_i - 4 * M)
    seg = sbr[ss:peak_i + 1]
    ds = np.diff(np.sign(seg - 0.01 * max_of))
    loc = np.where(ds >= 1)[0]
    zxi_a = loc + ss
    if len(zxi_a) == 0:
        return None, 1, peak_i, zxi_a
    zxi = int(zxi_a[-1]) if len(zxi_a) > 1 else int(zxi_a[0])
    mdfe = 0.0 if ndfe == 0 else float(bmax[0])
    mm = np.arange(zxi, zxi + 2 * M + 1)
    if CDR == 'Mod-MM':
        metric = np.abs(sbr[mm + M] - mdfe * sbr[mm])
    else:
        metric = np.abs(sbr[mm - M] - np.maximum(sbr[mm + M] - mdfe * sbr[mm], 0.0))
    return zxi + int(np.argmin(metric)), 0, peak_i, zxi_a


cur_ml, nzc_ml, peak_ml, zxi_ml = cursor_oracle(sbr, M, 1, param_c.bmax, 'MM', psr)
check("cursor_sample_index_peak_is_argmax",
      peak_i == int(np.argmax(sbr[psr])) + int(psr[0]) and peak_i == P,
      "peak index %d != argmax %d" % (peak_i, P))
check("cursor_sample_index_matches_matlab",
      cursor_i == cur_ml and nzc == nzc_ml and peak_i == peak_ml
      and len(zxi_arr) == len(zxi_ml) and (len(zxi_ml) == 0 or zxi_arr[-1] == zxi_ml[-1]),
      "cursor_sample_index diverges from transcription: py cursor=%s ml=%s" % (cursor_i, cur_ml))
check("cursor_sample_index_in_mm_window",
      nzc == 0 and int(zxi_arr[-1]) <= cursor_i <= int(zxi_arr[-1]) + 2 * M,
      "cursor not inside [zxi, zxi+2M]")
# MM criterion: first precursor near zero after balancing (physics).
check("cursor_sample_index_precursor_small",
      abs(sbr[cursor_i - M]) <= 0.6 * sbr[cursor_i],
      "first precursor %g not small vs cursor %g" % (sbr[cursor_i - M], sbr[cursor_i]))
# No-zero-crossing flag: a constant-positive signal never rises through threshold.
flat = np.ones(n) * 0.5
cur2, nzc2, _, zxi2 = sicopr.cursor_sample_index(flat, param_c, OP_c, psr)
check("cursor_sample_index_no_zero_crossing_flag",
      nzc2 == 1 and cur2 is None and len(zxi2) == 0,
      "constant signal should set no_zero_crossing=1")

# ===========================================================================
# 3. vma (ML 11479-11507 vs py 17664-17706)
# ===========================================================================
Mv = 32
# A clean, mostly-single-cursor pulse response so the bit stream tracks symbols.
pr = np.zeros(6 * Mv)
pr[2 * Mv] = 1.0
pr[2 * Mv + 1: 2 * Mv + Mv] = np.linspace(0.9, 0.0, Mv - 1)   # short tail
res = sicopr.vma(pr, Mv)
check("vma_P3_greater_than_P0",
      res.P_3 > res.P_0 and res.VMA > 0,
      "VMA=%g, P_3=%g P_0=%g (expected P_3>P_0)" % (res.VMA, res.P_3, res.P_0))
check("vma_VMA_is_P3_minus_P0",
      abs(res.VMA - (res.P_3 - res.P_0)) <= 1e-12,
      "VMA != P_3 - P_0")
# VMA bounded by the full-scale swing of the bit stream response.
seq, syms, _ = sicopr.PRBS13Q()
bsr = sicopr.lfilter(pr, [1.0], np.kron(seq, np.concatenate([[1.0], np.zeros(Mv - 1)])))
check("vma_bounded_by_swing",
      res.VMA <= (np.max(bsr) - np.min(bsr)) + 1e-9,
      "VMA %g exceeds bit-stream swing" % res.VMA)

# ===========================================================================
# 4. COM_eye_width (ML 1367-1556): eye width/height bounds + half_UI indexing
# ===========================================================================
samp_UI = 32
levels = 4
delta_y = 1e-3
half_UI = samp_UI // 2

# Controlled get_pdf_full: narrow (delta) ISI, no jitter, chosen A_s_vec.
_A_S_VEC = None


def fake_gpf(chdata0, delta_y_, t_s, param_, OP_, pdf_range_):
    su = int(param_.samples_for_C2M)
    pdf1 = [sicopr.d_cpdf(delta_y_, 0, 1) for _ in range(su)]
    hjf = np.zeros((8, su))
    return pdf1, hjf, np.asarray(_A_S_VEC, dtype=float)


def run_eye(A_s_vec, sigma_N):
    global _A_S_VEC
    _A_S_VEC = A_s_vec
    delta0 = sicopr.d_cpdf(delta_y, 0, 1)
    ber_q = float(np.sqrt(2) * sicopr.erfcinv(2 * 1e-4))
    Struct_Noise = SimpleNamespace(sigma_N=sigma_N, sigma_TX=0.0,
                                   ne_noise_pdf=delta0, cci_pdf=delta0, ber_q=ber_q)
    param = SimpleNamespace(samples_for_C2M=samp_UI, T_O=0, levels=levels,
                            specBER=1e-4, sigma_RJ=0.0, sigma_X=0.30, A_DD=0.0,
                            QL=2.5, R_LM=1.0)
    OP = SimpleNamespace(Histogram_Window_Weight='rectangle', ber_q=ber_q)
    fom_result = SimpleNamespace(t_s=samp_UI * 4)
    chd = [SimpleNamespace(type='THRU', base='thru')]
    return sicopr.COM_eye_width(chd, delta_y, fom_result, param, OP, Struct_Noise, False,
                             _get_pdf_full_fn=fake_gpf,
                             _normal_dist_fn=sicopr.normal_dist,
                             _conv_fct_fn=sicopr.conv_fct,
                             _conv_fct_MNZ_fn=sicopr.conv_fct_MeanNotZero,
                             _get_pdf_ss_fn=sicopr.get_pdf_from_sampled_signal,
                             _pdf_to_cdf_fn=sicopr.pdf_to_cdf,
                             _cdf_to_ber_fn=sicopr.cdf_to_ber_contour,
                             _find_eye_width_fn=sicopr.find_eye_width,
                             _combine_pdf_fn=sicopr.combine_pdf_same_voltage_axis)


# Open eye: raised-cosine signal amplitude peaking at the UI center.
jj = np.arange(samp_UI)
A_open = 0.35 * np.cos(np.pi * (jj - half_UI) / samp_UI) ** 2 + 0.02
Left_EW, Right_EW, eye_contour, out_VT, out_VB = run_eye(A_open, 0.004)

check("COM_eye_width_contour_shape",
      eye_contour.shape == (samp_UI, 2 * (levels - 1)),
      "eye_contour shape %s != (%d,%d)" % (eye_contour.shape, samp_UI, 2 * (levels - 1)))
check("COM_eye_width_EW_nonneg_bounded",
      np.all(Left_EW >= 0) and np.all(Right_EW >= 0)
      and np.all(Left_EW + Right_EW <= samp_UI + 1e-9),
      "eye widths out of [0, 1 UI]: L=%s R=%s" % (Left_EW, Right_EW))
# Eye height at the center for the middle eye must be positive (open).
mid = levels // 2 - 1
EH_center = eye_contour[half_UI, mid * 2] - eye_contour[half_UI, mid * 2 + 1]
check("COM_eye_width_EH_open_positive",
      EH_center > 0,
      "center eye height %g not positive for an open eye" % EH_center)
check("COM_eye_width_out_VT_VB_empty_when_TO_zero",
      list(out_VT) == [] and list(out_VB) == [],
      "out_VT/out_VB should be [] when T_O==0")

# Closed eye: heavy noise collapses the eye height at center.
_, _, ec_closed, _, _ = run_eye(A_open, 0.06)
EH_closed = ec_closed[half_UI, mid * 2] - ec_closed[half_UI, mid * 2 + 1]
check("COM_eye_width_EH_shrinks_with_noise",
      EH_closed < EH_center,
      "eye height did not shrink under heavier noise (%g >= %g)" % (EH_closed, EH_center))

# half_UI indexing: the EW/EH use the center sample. Verify the center column of
# the open eye is the widest-open (top-bottom largest) across sampled phases.
mid_eye = eye_contour[:, mid * 2] - eye_contour[:, mid * 2 + 1]
check("COM_eye_width_center_is_max_open",
      int(np.argmax(mid_eye)) in (half_UI - 1, half_UI, half_UI + 1),
      "widest eye opening at phase %d, expected ~center %d" % (int(np.argmax(mid_eye)), half_UI))


# ---------------------------------------------------------------------------
# BEST.ctle / BEST.G_high_pass index base (regression, 2026-08-21)
#
# THIS.ctle_index and THIS.g_LP_index are set 1-BASED to match MATLAB, and
# OptFom_Update_Best_Setttings copies them straight into BEST.ctle and
# BEST.G_high_pass. OptFom_Update_BEST_Post_Optimize then used them to index
# param.CTLE_fz / f_HP / g_DC_HP_values with a comment claiming 0-based, so it
# read one entry too high on every run.
#
# It hid for two reasons, both worth knowing: the fields feed only
# OptFom_Plot_Best_Results (the COM path recomputes ctle_gain in optimize_fom,
# which does convert), and an over-long CTLE list turns the error into a silently
# wrong lookup instead of an IndexError. It only raised when the winning CTLE was
# the LAST in the list -- which is exactly what a truncated --max-ctle sweep does.
# The 208-case corpus cannot reach it.
_ctle_n = 3
_param = SimpleNamespace(
    samples_per_ui=32, ndfe=0, ui=1.0 / 106.25e9,
    CTLE_fz=np.array([1e9, 2e9, 3e9]),
    CTLE_fp1=np.array([4e9, 5e9, 6e9]),
    CTLE_fp2=np.array([7e9, 8e9, 9e9]),
    CTLE_type='CL93',
)
# OP must disable all three filters, or OptFom_Calc_Hr raises before the CTLE
# lookup is ever reached and the check passes without testing anything. An
# earlier version of this test did exactly that; mutation testing caught it.
_OP = SimpleNamespace(Butterworth=False, Bessel_Thomson=False, Raised_Cosine=False)
_BEST = SimpleNamespace(
    sbr=np.zeros(64), cursor_i=32, ctle=_ctle_n, gdc=-5.0, G_high_pass=1,
    bmax=np.array([]), bmin=np.array([]),
)
_f = np.linspace(1e8, 50e9, 64)

_raised = None
try:
    _got = sicopr.OptFom_Update_BEST_Post_Optimize(copy.deepcopy(_BEST), _f, _param, _OP)
except Exception as _e:          # noqa: BLE001 - any failure here is a real result
    _raised = _e
    _got = None

check("post_optimize_ctle_index_is_1_based",
      _raised is None,
      "OptFom_Update_BEST_Post_Optimize failed for the LAST CTLE entry "
      "(BEST.ctle=%d, len(CTLE_fz)=%d). BEST.ctle is 1-based and must be "
      "converted before indexing -- %s: %s"
      % (_ctle_n, _ctle_n, type(_raised).__name__, _raised))

# Value check, not just absence-of-crash: the winning CTLE is the LAST entry, so
# the pole/zero used must be the last of each list.
_expect = _FD_CTLE_ref = None
if _got is not None:
    num = 10 ** (-5.0 / 20) + 1j * _f / 3e9
    den = (1 + 1j * _f / 6e9) * (1 + 1j * _f / 9e9)
    _expect = num / den
    check("post_optimize_ctle_uses_the_selected_entry",
          np.allclose(np.asarray(_got.ctle_gain1), _expect, rtol=0, atol=0),
          "ctle_gain1 was not built from CTLE_fz/fp1/fp2[-1]; an off-by-one here "
          "silently uses the neighbouring CTLE setting whenever the list is long "
          "enough to stay in range")

finish()
