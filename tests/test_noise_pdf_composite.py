"""Audit batch B08: G6 PDF/CDF noise pipeline, composite.

Functions under audit (MATLAB com_ieee8023_4p15p0.m -> com.py):
  Create_Noise_PDF              ML 1557-1685 -> py 1995-2125
  combine_pdf_same_voltage_axis ML 5418-5458 -> py 8009-8046
  comb_fct                      ML 5390-5415 -> py 7960-7984
  conv_fct_MeanNotZero          ML 5521-5536 -> py 8114-8124
  get_pdf_full                  ML 7650-7793 -> py 11197-11348
  get_pdf_from_sampled_signal   ML 7603-7649 -> py 11037-11085

Invariants (audit prompt section 3 item 6, section 5 Tier 2/3):
  - conv_fct_MeanNotZero agrees with conv_fct (redundant-helper agreement).
  - comb_fct / combine_pdf_same_voltage_axis align axes and sum probabilities.
  - get_pdf_from_sampled_signal conserves mass, symmetric input -> zero mean,
    total variance = sum(v_k^2) * sigma_PAM^2 (independent ISI terms).
  - Create_Noise_PDF: combined PDF conserves mass, CDF monotone to ~1, sigma_TX.
  - get_pdf_full compared against a MATLAB-faithful oracle (resampling grid,
    DFE cancellation, reshape, circshift, phase selection).

Run: python tests/test_noise_pdf_composite.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy.special import erfcinv

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, xcheck, finish  # noqa: E402
import com  # noqa: E402


def rel_err(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = max(np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(a - b)) / denom)


def mk_pdf(binsize, minbin, y):
    p = SimpleNamespace()
    p.BinSize = binsize
    p.Min = minbin
    p.y = np.asarray(y, dtype=float)
    p.x = np.arange(minbin, minbin + len(p.y)) * binsize
    return p


rng = np.random.default_rng(80808)

# ===========================================================================
# 1. conv_fct_MeanNotZero == conv_fct (redundant-helper agreement)
# ===========================================================================
p1 = mk_pdf(1e-3, -2, [0.1, 0.2, 0.4, 0.2, 0.1])
p2 = mk_pdf(1e-3, -1, [0.25, 0.5, 0.25])
cmnz = com.conv_fct_MeanNotZero(p1, p2)
cf = com.conv_fct(p1, p2)
check("conv_fct_MeanNotZero_agrees_with_conv_fct",
      rel_err(cmnz.y, cf.y) <= 1e-15 and cmnz.Min == cf.Min
      and rel_err(cmnz.x, cf.x) <= 1e-15,
      "conv_fct_MeanNotZero diverges from conv_fct")
check("conv_fct_MeanNotZero_conserves_mass",
      abs(np.sum(cmnz.y) - np.sum(p1.y) * np.sum(p2.y)) <= 1e-14,
      "conv_fct_MeanNotZero did not conserve mass")

# ===========================================================================
# 2. comb_fct (ML 5390-5415): aligned addition of two PDFs
# ===========================================================================
pa = mk_pdf(1e-3, -3, [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])   # x=-3..3
pb = mk_pdf(1e-3, -2, [0.2, 0.3, 0.3, 0.2])                     # Min=-2, len 4
# Oracle (ML 5398-5412): Min=min, shift pb into pa's range, add.
pc = com.comb_fct(pa, pb)
difsz = abs(pa.Min - pb.Min)
oracle_pb = np.zeros(len(pa.y))
oracle_pb[difsz:difsz + len(pb.y)] = pb.y
check("comb_fct_aligned_addition",
      pc.Min == min(pa.Min, pb.Min)
      and rel_err(pc.y, pa.y + oracle_pb) <= 1e-15,
      "comb_fct did not align-and-add correctly")
check("comb_fct_mass_is_sum",
      abs(np.sum(pc.y) - (np.sum(pa.y) + np.sum(pb.y))) <= 1e-14,
      "comb_fct mass != sum of input masses")

# ===========================================================================
# 3. combine_pdf_same_voltage_axis (ML 5418-5458)
# ===========================================================================
# pdf1 wider on the left, pdf2 wider on the right.
pdf1 = SimpleNamespace(BinSize=1e-3, x=np.arange(-4, 3) * 1e-3,
                       y=np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1]))
pdf2 = SimpleNamespace(BinSize=1e-3, x=np.arange(-2, 5) * 1e-3,
                       y=np.array([0.15, 0.2, 0.3, 0.2, 0.1, 0.03, 0.02]))
out = com.combine_pdf_same_voltage_axis(pdf1, pdf2)
check("combine_pdf_mass_is_sum",
      abs(np.sum(out.y) - (np.sum(pdf1.y) + np.sum(pdf2.y))) <= 1e-13,
      "combine_pdf mass != sum of input masses")
check("combine_pdf_common_axis",
      abs(out.x[0] - min(pdf1.x[0], pdf2.x[0])) <= 1e-18
      and abs(out.x[-1] - max(pdf1.x[-1], pdf2.x[-1])) <= 1e-18
      and len(out.x) == len(out.y),
      "combine_pdf did not build the union voltage axis")
# Equal-min shortcut (shift_amount=0).
pdf3 = SimpleNamespace(BinSize=1e-3, x=np.arange(-3, 4) * 1e-3, y=np.ones(7) / 7)
out2 = com.combine_pdf_same_voltage_axis(pdf3, pdf3)
check("combine_pdf_equal_min",
      rel_err(out2.y, 2 * pdf3.y) <= 1e-15,
      "combine_pdf of identical PDFs != 2x")

# ===========================================================================
# 4. get_pdf_from_sampled_signal (ML 7603-7649): mass + variance invariant
# ===========================================================================
L = 4
BinSize = 1e-4
values = 2 * np.arange(L) / (L - 1) - 1.0
sigma_PAM2 = float(np.sum(values ** 2) / L)          # variance of uniform PAM levels

isi = np.array([0.03, -0.02, 0.015, -0.008, 0.005])
p_isi = com.get_pdf_from_sampled_signal(isi, L, BinSize)
check("gpfss_conserves_mass",
      abs(np.sum(p_isi.y) - 1.0) <= 1e-9,
      "get_pdf_from_sampled_signal mass != 1")
# Zero mean (symmetric PAM, symmetric result).
mean_isi = float(np.sum(p_isi.x * p_isi.y))
check("gpfss_zero_mean",
      abs(mean_isi) <= 1e-6,
      "ISI PDF mean %g not ~0 (symmetric noise should be zero-mean)" % mean_isi)
# Total variance = sum(v_k^2) * sigma_PAM^2 (independent convolution).
var_out = float(np.sum(p_isi.x ** 2 * p_isi.y) - mean_isi ** 2)
var_expect = float(np.sum(isi ** 2) * sigma_PAM2)
check("gpfss_variance_is_sum_of_terms",
      abs(var_out - var_expect) <= 1e-3 * var_expect,
      "ISI PDF variance %g != sum(v^2)*sigma_PAM^2 %g" % (var_out, var_expect))
# Tiny input (all < BinSize) -> delta at 0.
p_tiny = com.get_pdf_from_sampled_signal(np.array([1e-9, -1e-9]), L, BinSize)
check("gpfss_tiny_input_delta",
      len(p_tiny.y) == 1 and abs(p_tiny.y[0] - 1.0) <= 1e-15,
      "sub-bin input did not give the delta PDF")

# ===========================================================================
# 5. Create_Noise_PDF (ML 1557-1685): composite mass + monotone CDF + sigma_TX
# ===========================================================================
delta_y = 1e-4
levels = 4
A_s = 0.5
# Build ISI/crosstalk PDFs.
sci = com.get_pdf_from_sampled_signal(np.array([0.02, -0.015, 0.01]), levels, delta_y)
next_pdf = com.get_pdf_from_sampled_signal(np.array([0.008, -0.005]), levels, delta_y)
fext_pdf = com.get_pdf_from_sampled_signal(np.array([0.006, 0.004]), levels, delta_y)
chdata = [SimpleNamespace(type='THRU', pdfr=sci, faxis=np.linspace(0, 40e9, 64)),
          SimpleNamespace(type='NEXT', pdfr=next_pdf),
          SimpleNamespace(type='FEXT', pdfr=fext_pdf)]

h_J = rng.standard_normal(40) * 1e-3
param = SimpleNamespace(delta_y=delta_y, levels=levels, R_LM=0.95, SNR_TX=32.0,
                        sigma_RJ=0.01, sigma_X=0.30, A_DD=0.02, specBER=1e-5,
                        Noise_Crest_Factor=0, N_qb=0, number_of_s4p_files=3)
fom_result = SimpleNamespace(sigma_N=1.5e-3, h_J=h_J, ctle=1, txffe=[1.0], cur=1)
OP = SimpleNamespace(RX_CALIBRATION=False, FFE_OPT_METHOD='none', RxFFE=False,
                     SNR_TXwC0=False, force_BBN_Q_factor=False, PSDRXCAL=False)

PDF, CDF, NS = com.Create_Noise_PDF(A_s, param, fom_result, chdata, OP, 0.0, None)

check("Create_Noise_PDF_combined_mass_one",
      abs(np.sum(PDF.y) - 1.0) <= 1e-6,
      "combined interference+noise PDF mass=%g != 1" % np.sum(PDF.y))
check("Create_Noise_PDF_cdf_monotone_to_one",
      np.all(np.diff(CDF) >= -1e-15) and abs(CDF[-1] - 1.0) <= 1e-6,
      "CDF not monotone nondecreasing ending at ~1")
# sigma_TX closed form (ML 1584): (L-1)*A_s/R_LM * 10^(-SNR_TX/20).
expect_sigma_TX = (levels - 1) * A_s / param.R_LM * 10 ** (-param.SNR_TX / 20)
check("Create_Noise_PDF_sigma_TX_formula",
      abs(NS.sigma_TX - expect_sigma_TX) <= 1e-12 * expect_sigma_TX,
      "sigma_TX=%g != %g" % (NS.sigma_TX, expect_sigma_TX))
# sigma_G = norm([sigma_rjit, sigma_N, sigma_TX]) (ML 1588).
sigma_rjit = param.sigma_RJ * param.sigma_X * np.linalg.norm(h_J)
expect_sigma_G = np.linalg.norm([sigma_rjit, fom_result.sigma_N, expect_sigma_TX])
check("Create_Noise_PDF_sigma_G_formula",
      abs(NS.sigma_G - expect_sigma_G) <= 1e-12 * expect_sigma_G,
      "sigma_G=%g != %g" % (NS.sigma_G, expect_sigma_G))
check("Create_Noise_PDF_sci_is_thru_pdfr",
      NS.sci_pdf is chdata[0].pdfr,
      "sci_pdf is not chdata[0].pdfr")
# ber_q crest factor.
check("Create_Noise_PDF_ber_q",
      abs(NS.ber_q - np.sqrt(2) * erfcinv(2 * param.specBER)) <= 1e-12,
      "ber_q crest factor wrong")

# ===========================================================================
# 6. get_pdf_full (ML 7650-7793) vs MATLAB-faithful oracle
# ===========================================================================
M_orig = 16
samp_UI = 32                                   # C2M upsample ratio 2
ndfe = 2
levels_f = 4
delta_yf = 1e-3
bmax = np.array([0.7, 0.3])
bmin = np.array([-0.7, -0.3])

nUI = 20
pulse = rng.standard_normal(nUI * M_orig) * 2e-3
t_s_orig = 6 * M_orig                           # 0-based cursor (peak)
pulse[t_s_orig] = 1.0
pulse[t_s_orig + M_orig] = 0.15
pulse[t_s_orig + 2 * M_orig] = 0.4

chdata_f = SimpleNamespace(eq_pulse_response=pulse.copy(), type='THRU', base='thru')
param_f = SimpleNamespace(samples_per_ui=M_orig, samples_for_C2M=samp_UI,
                          ndfe=ndfe, Floating_DFE=False, dfe_delta=0.0,
                          bmax=bmax, bmin=bmin, N_bmax=ndfe, use_bmax=bmax,
                          use_bmin=bmin, levels=levels_f, R_LM=0.95)
OP_f = SimpleNamespace(DISPLAY_WINDOW=False)

pdf_list, h_j_full, A_s_vec = com.get_pdf_full(chdata_f, delta_yf, t_s_orig, param_f, OP_f, None)


def gpf_oracle(pulse, t_s_orig, param):
    """Faithful transcription of get_pdf_full THRU path (ML 7651-7793) using the
    MATLAB colon-based new_time and 1-based half_UI / shift_amount."""
    M_orig = param.samples_per_ui
    samp_UI = param.samples_for_C2M
    old_time = np.arange(len(pulse)) / M_orig
    old_time = old_time - old_time[t_s_orig]
    mn, mx = float(np.min(old_time)), float(np.max(old_time))
    n_neg = int(np.floor(-mn * samp_UI + 1e-9))     # MATLAB colon 0:-1/samp_UI:mn count
    n_pos = int(np.floor(mx * samp_UI + 1e-9))
    timea = -np.arange(0, n_neg + 1) / samp_UI       # [0,-1/s,...,-n_neg/s]
    new_time = np.concatenate([timea[::-1], (np.arange(1, n_pos + 1) / samp_UI)])
    SBR = np.interp(new_time, old_time, pulse)        # in-range: no extrapolation
    t_s = int(np.argmin(np.abs(new_time)))            # 0-based
    residual = SBR.copy()
    ui = np.arange(samp_UI) / samp_UI
    half_UI_M = int(np.argmin(np.abs(ui - 0.5))) + 1  # 1-based
    ndfe = param.ndfe
    post = t_s + samp_UI * np.arange(1, ndfe + 1)
    icc = SBR[post]
    cursor = residual[t_s]
    ecc = com.dfe_clipper(icc, cursor * param.bmax[:ndfe], cursor * param.bmin[:ndfe])
    ecs = np.repeat(ecc, samp_UI)
    start_cancel = t_s - half_UI_M + samp_UI + 1      # ML 1-based -> 0-based
    residual[start_cancel:start_cancel + ndfe * samp_UI] -= ecs
    uiv_start = start_cancel - samp_UI
    A_s_vec = param.R_LM * SBR[uiv_start:uiv_start + samp_UI] / (param.levels - 1)
    residual[uiv_start:uiv_start + samp_UI] = 0.0
    nui = round(len(residual) / samp_UI)
    vs = residual[samp_UI:samp_UI * (nui - 1)].reshape(nui - 2, samp_UI)
    t_s_1b = t_s + 1
    phases_M = t_s_1b % samp_UI
    if phases_M == 0:
        phases_M = samp_UI
    shift = half_UI_M - phases_M
    vs_shift = np.roll(vs, shift, axis=1)
    pdfs = [None] * samp_UI
    for k in range(1, samp_UI + 1):
        pdfs[k - 1] = com.get_pdf_from_sampled_signal(vs_shift[:, k - 1], param.levels, delta_yf)
    return pdfs, A_s_vec


pdfs_ml, A_s_vec_ml = gpf_oracle(pulse, t_s_orig, param_f)

# Compare every column that both produced.
cols_common = [k for k in range(samp_UI)
               if pdf_list[k] is not None and pdfs_ml[k] is not None]
max_col_err = 0.0
mismatch_cols = []
for k in cols_common:
    a = np.asarray(pdf_list[k].y, dtype=float)
    b = np.asarray(pdfs_ml[k].y, dtype=float)
    n = min(len(a), len(b))
    # align by Min (both are symmetric about 0; compare overlapping support)
    e = 1.0 if (len(a) != len(b) or pdf_list[k].Min != pdfs_ml[k].Min) else rel_err(a, b)
    if e > 1e-9:
        mismatch_cols.append(k)
    max_col_err = max(max_col_err, e)

xcheck("get_pdf_full_matches_matlab_oracle",
      len(mismatch_cols) == 0,
      "DIVERGENT (D12, C2M-only): get_pdf_full diverges from the MATLAB-faithful "
      "oracle in %d/%d phase columns (max relerr %.3g). Cause: py 11218-11222 "
      "new_time uses arange(...,floor(x)+2), adding an extra out-of-range point "
      "per side vs MATLAB colon (ML 7665-7667); the cursor t_s then shifts by one "
      "sample, moving the phase and centering circshift. (half_UI is correctly "
      "1-based via _get_pdf_full__get_center_of_UI=M//2+1.)"
      % (len(mismatch_cols), len(cols_common), max_col_err))

finish()
