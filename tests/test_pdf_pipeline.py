"""Audit batch B07: G6 PDF/CDF noise pipeline, leaf level.

Functions under audit (MATLAB com_ieee8023_4p15p0.m -> sicopr.py):
  normal_dist    ML 8540-8545 -> py 12400-12421
  pdf_to_cdf     ML 8983-8994 -> py 12973-12989
  conv_fct       ML 5503-5517 -> py 8085-8095
  d_cpdf         ML 5612-5662 -> py 8220-8269
  Init_PDF_Fast  ML 2204-2270 -> py 2649-2666
  get_pdf        ML 7507-7602 -> py 10864-10962

Invariants checked (audit prompt section 3 item 6, section 5 Tier 2/3):
  - normal_dist integrates to 1, symmetric, centered, correct sigma.
  - pdf_to_cdf: yB monotone up, yT monotone down, y=min, ends near 0.
  - conv_fct / Init_PDF_Fast / d_cpdf conserve probability mass and preserve the
    bin axis; convolution is associative.
  - get_pdf applies the DFE cancellation EXACTLY ONCE over the correct window,
    and the THRU phase-column selection matches MATLAB (1-based col vs 0-based).

Oracles transcribed from the cited MATLAB lines. Run:
python tests/test_pdf_pipeline.py
"""
import os
import sys
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
# 1. normal_dist (ML 8540-8545 vs py 12400-12421)
# ===========================================================================
sigma, nsigma, binsize = 5e-3, 5, 1e-4
nd = sicopr.normal_dist(sigma, nsigma, binsize)
# Min = -round(2*nsigma*sigma/binsize) (MATLAB round half-away; matches for this value).
expect_min = -round(2 * nsigma * sigma / binsize)
check("normal_dist_min_formula",
      nd.Min == expect_min and len(nd.x) == 2 * (-nd.Min) + 1,
      "Min/x-length wrong: Min=%d len=%d" % (nd.Min, len(nd.x)))
check("normal_dist_integrates_to_one",
      abs(np.sum(nd.y) - 1.0) <= 1e-12,
      "sum(pdf.y)=%g != 1" % np.sum(nd.y))
check("normal_dist_symmetric_centered",
      rel_err(nd.y, nd.y[::-1]) <= 1e-12 and abs(nd.x[len(nd.x) // 2]) <= 1e-15,
      "Gaussian PDF not symmetric/centered")
# Achieved sigma: sqrt(sum(x^2 * y)) ~ sigma (discretisation makes it slightly off).
achieved = np.sqrt(np.sum(nd.x ** 2 * nd.y))
check("normal_dist_sigma",
      abs(achieved - sigma) <= 1e-3 * sigma,
      "achieved sigma %g != %g" % (achieved, sigma))

# ===========================================================================
# 2. pdf_to_cdf (ML 8983-8994 vs py 12973-12989)
# ===========================================================================
cdf = sicopr.pdf_to_cdf(nd)
check("pdf_to_cdf_yB_monotone_up",
      np.all(np.diff(cdf.yB) >= -1e-15) and abs(cdf.yB[-1] - 1.0) <= 1e-12,
      "yB not monotone nondecreasing ending at 1")
check("pdf_to_cdf_yT_monotone_down",
      np.all(np.diff(cdf.yT) <= 1e-15) and abs(cdf.yT[0] - 1.0) <= 1e-12,
      "yT not monotone nonincreasing starting at 1")
check("pdf_to_cdf_y_is_min_and_tails_small",
      rel_err(cdf.y, np.minimum(cdf.yB, cdf.yT)) <= 1e-15
      and cdf.y[0] <= 1e-6 and cdf.y[-1] <= 1e-6,
      "cdf.y != min(yB,yT) or tails not small")

# ===========================================================================
# 3. conv_fct (ML 5503-5517 vs py 8085-8095): mass + axis + associativity
# ===========================================================================
def mk_pdf(binsize, minbin, y):
    p = SimpleNamespace()
    p.BinSize = binsize
    p.Min = minbin
    p.y = np.asarray(y, dtype=float)
    p.x = np.arange(minbin, minbin + len(p.y)) * binsize
    return p


p1 = mk_pdf(1e-3, -2, [0.1, 0.2, 0.4, 0.2, 0.1])
p2 = mk_pdf(1e-3, -1, [0.25, 0.5, 0.25])
pc = sicopr.conv_fct(p1, p2)
check("conv_fct_full_convolution",
      rel_err(pc.y, np.convolve(p1.y, p2.y)) <= 1e-15
      and len(pc.y) == len(p1.y) + len(p2.y) - 1,
      "conv_fct is not a full convolution")
check("conv_fct_min_and_axis",
      pc.Min == p1.Min + p2.Min
      and abs(pc.x[0] - pc.Min * pc.BinSize) <= 1e-18
      and abs(pc.x[-1] - (pc.Min + len(pc.y) - 1) * pc.BinSize) <= 1e-18,
      "conv_fct Min/x-axis wrong")
check("conv_fct_conserves_mass",
      abs(np.sum(pc.y) - np.sum(p1.y) * np.sum(p2.y)) <= 1e-14,
      "conv_fct did not conserve mass")
# Associativity within tolerance.
p3 = mk_pdf(1e-3, 0, [0.6, 0.4])
left = sicopr.conv_fct(sicopr.conv_fct(p1, p2), p3)
right = sicopr.conv_fct(p1, sicopr.conv_fct(p2, p3))
check("conv_fct_associative",
      rel_err(left.y, right.y) <= 1e-13 and left.Min == right.Min,
      "convolution not associative")

# ===========================================================================
# 4. d_cpdf (ML 5612-5662 vs py 8220-8269)
# ===========================================================================
bs = 1e-3
vals = np.array([-3e-3, -1e-3, 1e-3, 3e-3])       # symmetric PAM-like values
prob = np.array([0.25, 0.25, 0.25, 0.25])
dp = sicopr.d_cpdf(bs, vals, prob)
check("d_cpdf_mass_one",
      abs(np.sum(dp.y) - 1.0) <= 1e-14,
      "d_cpdf mass != 1")
check("d_cpdf_symmetric_axis",
      len(dp.x) == 2 * (-dp.Min) + 1 and abs(dp.x[0] + dp.x[-1]) <= 1e-18,
      "d_cpdf x-axis (Min:-Min) not symmetric")
# All-zero fast path.
dz = sicopr.d_cpdf(bs, np.zeros(4), prob)
check("d_cpdf_all_zero_shortcut",
      dz.Min == 0 and np.array_equal(dz.y, [1.0]) and np.array_equal(dz.x, [0.0]),
      "all-zero values did not give the delta PDF")
# Bin placement: probability mass lands on the snapped bins.
check("d_cpdf_bins_place_mass",
      abs(dp.y[np.argmin(np.abs(dp.x - (-3e-3)))] - 0.25) <= 1e-12
      and abs(dp.y[np.argmin(np.abs(dp.x - 3e-3))] - 0.25) <= 1e-12,
      "d_cpdf did not place mass on the expected bins")

# ===========================================================================
# 5. Init_PDF_Fast (ML 2204-2270 vs py 2649-2666)
# ===========================================================================
empty = SimpleNamespace(BinSize=bs, Min=0, x=np.array([0.0]), y=np.array([1.0]))
ipf = sicopr.Init_PDF_Fast(empty, vals, prob)
check("Init_PDF_Fast_mass_preserved",
      abs(np.sum(ipf.y) - np.sum(prob)) <= 1e-14,
      "Init_PDF_Fast changed total mass")
# Bin placement matches rounded_values_div_binsize - Min.
rvd = np.round(vals / bs).astype(int)
oracle_y = np.zeros(rvd[-1] - rvd[0] + 1)
for k in range(len(vals)):
    oracle_y[rvd[k] - rvd[0]] += prob[k]
check("Init_PDF_Fast_bin_placement",
      ipf.Min == rvd[0] and rel_err(ipf.y, oracle_y) <= 1e-15
      and abs(ipf.x[0] - rvd[0] * bs) <= 1e-18,
      "Init_PDF_Fast bin placement/axis wrong")

# ===========================================================================
# 6. get_pdf (ML 7507-7602 vs py 10864-10962): DFE applied EXACTLY once
# ===========================================================================
M = 16
ndfe = 2
levels = 4
delta_y = 1e-3
bmax = np.array([0.7, 0.3])
bmin = np.array([-0.7, -0.3])

nUI = 24
SBR = np.zeros(nUI * M)
rng = np.random.default_rng(70707)
SBR += rng.standard_normal(len(SBR)) * 2e-3        # small ISI everywhere
t_s = 5 * M + M // 2                                # cursor at 88
SBR[t_s] = 1.0                                      # main cursor
SBR[t_s + M] = 0.2                                  # postcursor 1: within bmax[0]*1=0.7
SBR[t_s + 2 * M] = 0.5                              # postcursor 2: exceeds bmax[1]*1=0.3

chdata = SimpleNamespace(eq_pulse_response=SBR.copy(), type='THRU', base='thru')
param = SimpleNamespace(samples_per_ui=M, ndfe=ndfe, Floating_DFE=False,
                        dfe_delta=0.0, bmax=bmax, bmin=bmin, N_bmax=ndfe,
                        use_bmax=bmax, use_bmin=bmin, levels=levels)
OP = SimpleNamespace(FFE_OPT_METHOD='none', RxFFE=False, DISPLAY_WINDOW=False)

pdf = sicopr.get_pdf(chdata, delta_y, t_s, param, OP)


def get_pdf_oracle(SBR, t_s, param):
    """Faithful transcription of the THRU path (ML 7511-7594): cancel cursor+DFE
    ONCE over [t_s-M/2, t_s+(1/2+ndfe)M-1], then sample the cursor phase."""
    M = param.samples_per_ui
    residual = np.asarray(SBR, dtype=float).copy()
    ndfe = param.ndfe
    idx = t_s + M * np.arange(ndfe + 1)
    icc = residual[idx]
    cursor = residual[t_s]
    bmax_v = cursor * np.concatenate([[1.0], param.bmax])
    bmin_v = cursor * np.concatenate([[1.0], param.bmin])
    ecc = sicopr.dfe_clipper(icc, bmax_v, bmin_v)      # B04-verified clipper
    ecs = np.repeat(ecc, M)                          # kron(ecc, ones(1,M))
    start = t_s - M // 2
    residual[start:start + (ndfe + 1) * M] -= ecs    # SINGLE subtraction
    nui = round(len(residual) / M)
    vs = np.zeros((nui - 2, M))
    for i in range(M):
        vs[:, i] = residual[M * np.arange(1, nui - 1) + i]
    ph = t_s % M                                     # 0-based cursor phase
    return sicopr.get_pdf_from_sampled_signal(vs[:, ph], param.levels, delta_y), residual


pdf_ml, residual_ml = get_pdf_oracle(SBR, t_s, param)
check("get_pdf_matches_matlab_transcription",
      rel_err(pdf.y, pdf_ml.y) <= 1e-12 and pdf.Min == pdf_ml.Min,
      "get_pdf diverges from the DFE-cancellation transcription")

# DFE applied EXACTLY once: postcursor within bound fully cancelled (residual~0);
# clipped postcursor leaves exactly (tap - clip) residual (double subtraction would
# give tap - 2*clip).
check("get_pdf_dfe_applied_once_inbound_tap",
      abs(residual_ml[t_s + M]) <= 1e-12,
      "in-bound postcursor not fully cancelled once (residual=%g)" % residual_ml[t_s + M])
check("get_pdf_dfe_applied_once_clipped_tap",
      abs(residual_ml[t_s + 2 * M] - (0.5 - 0.3)) <= 1e-12,
      "clipped postcursor residual=%g, expected 0.2 (single subtraction of clip=0.3)"
      % residual_ml[t_s + 2 * M])
check("get_pdf_cursor_cancelled",
      abs(residual_ml[t_s]) <= 1e-12,
      "cursor not removed by the cancellation")
check("get_pdf_valid_distribution",
      abs(np.sum(pdf.y) - 1.0) <= 1e-9 and np.all(pdf.y >= -1e-15),
      "get_pdf output is not a valid probability mass function")

# Phase selection: the returned PDF must be the cursor-phase PDF (0-based t_s%M),
# i.e. it equals get_pdf_from_sampled_signal on that exact column.
nui = round(len(residual_ml) / M)
vs_check = np.zeros((nui - 2, M))
for i in range(M):
    vs_check[:, i] = residual_ml[M * np.arange(1, nui - 1) + i]
pdf_cursor_phase = sicopr.get_pdf_from_sampled_signal(vs_check[:, t_s % M], levels, delta_y)
check("get_pdf_selects_cursor_phase",
      rel_err(pdf.y, pdf_cursor_phase.y) <= 1e-12,
      "get_pdf did not select the cursor sub-phase column")

finish()
