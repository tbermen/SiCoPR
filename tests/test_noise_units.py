"""Audit batch B01: G1 unit conversions in noise integrals.

Functions under audit (MATLAB com_ieee8023_4p15p0.m -> sicopr.py):
  S_RN                      ML 4479-4502  -> py 6594-6610
  N_s                       ML 2698-2738  -> py 3391-3415 (+ copy _S_IN__N_s py 6546-6566)
  get_sigma_noise           ML 7950-7973  -> py 11489-11521
  get_sigma_eta_ACCM_noise  ML 7926-7948  -> py 11447-11483
  get_PSDs                  ML 6512-6784  -> py 9494-9789 (+ stubs py 9443-9456)

Every oracle below is transcribed directly from the cited MATLAB lines with
1-based indexing converted at the array-access boundary only. Oracles do NOT
call the sicopr.py code paths they are checking (exception: the ADC 'slow' clip
oracle uses sicopr.py's public get_pdf_from_sampled_signal / conv_fct / CDF_inv_ev
as the best available stand-ins for the MATLAB PDF machinery; those functions
are audited separately in batches B07/B08).

Checks that are EXPECTED to FAIL document divergences; the FAIL rows in
tests/results.csv are the ledger evidence. Run: python tests/test_noise_units.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, xcheck, finish  # noqa: E402
import sicopr  # noqa: E402


def rel_err(a, b):
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    denom = max(np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(a - b)) / denom)


# ---------------------------------------------------------------------------
# Shared MATLAB-transcribed helpers (from the cited lines, not from sicopr.py)
# ---------------------------------------------------------------------------

def ml_double_sided(S):
    """ML 6552 pattern: [real(S(1)), S(2:end-1), real(S(end)), conj(S(end-1:-1:2))]."""
    S = np.asarray(S)
    return np.concatenate(([np.real(S[0])], S[1:-1], [np.real(S[-1])],
                           np.conj(S[-2:0:-1])))


def ml_fold(psd, num_ui, M):
    """ML 6555: sum(reshape(psd, num_ui, M).').

    MATLAB reshape is column-major: element (k,m) = psd[k + num_ui*m] (0-based),
    so the fold is out[k] = sum_m psd[k + num_ui*m]. Implemented index-wise to
    stay independent of numpy reshape order semantics.
    """
    out = np.zeros(num_ui, dtype=float)
    for m in range(M):
        out += np.real(psd[m * num_ui:(m + 1) * num_ui])
    return out


def ml_S_RN(f, G_DC, G_DC2, p):
    """Transcription of ML 4480-4490 (S_RN body)."""
    f = np.asarray(f, dtype=float)
    p1, z1, p2 = p.CTLE_fp1, p.CTLE_fz, p.CTLE_fp2
    zlf = plf = p.f_HP
    H_CTF = ((10 ** (G_DC / 20) + 1j * f / z1) * (10 ** (G_DC2 / 20) + 1j * f / zlf)
             / ((1 + 1j * f / p1) * (1 + 1j * f / p2) * (1 + 1j * f / plf)))
    H_R = 1.0 / np.polyval([1, 2.613126, 3.414214, 2.613126, 1],
                           1j * f / (p.f_r * p.fb))
    return p.eta_0 / 2 * np.abs(H_CTF * H_R) ** 2


# ===========================================================================
# 1. S_RN (ML 4479-4502 vs py 6594-6610)  Tier 2 analytic
# ===========================================================================

par = SimpleNamespace(CTLE_fp1=1.2e10, CTLE_fz=4e9, CTLE_fp2=3.5e10,
                      f_HP=1e8, f_r=0.75, fb=2.5e10, eta_0=8.2e-9)

# DC gain: at f=0, H_CTF = 10^(G_DC/20)*10^(G_DC2/20) and H_R = 1, so
# S_RN(0) = eta_0/2 * 10^((G_DC+G_DC2)/10). Pure closed form -> tol 1e-12.
srn0 = float(sicopr.S_RN(np.array([0.0]), -6.0, -1.0, par)[0])
expect0 = par.eta_0 / 2 * 10 ** ((-6.0 - 1.0) / 10)
check("S_RN_dc_gain_closed_form", abs(srn0 - expect0) <= 1e-12 * expect0,
      "S_RN(0)=%g expected %g" % (srn0, expect0))

# Bessel-Thomson 3 dB point: with H_CTF forced to exactly 1 (z1==p1 and
# G_DC=G_DC2=0 make both CTLE factors cancel; p2 -> 1e30 removes the last
# pole), |H_R|^2 at f = f_r*fb is 1/|polyval(...,1j)|^2 = 1/2.0000011,
# so S_RN(f_r*fb)/S_RN(0) = 0.4999997. Tol 1e-5 covers the non-ideal
# polynomial coefficients (they are 6-digit truncations of the exact ones).
par_flat = SimpleNamespace(CTLE_fp1=1e10, CTLE_fz=1e10, CTLE_fp2=1e30,
                           f_HP=1e8, f_r=0.75, fb=2.5e10, eta_0=8.2e-9)
sr = sicopr.S_RN(np.array([0.0, par_flat.f_r * par_flat.fb]), 0.0, 0.0, par_flat)
ratio = float(sr[1] / sr[0])
check("S_RN_bessel_thomson_3dB", abs(ratio - 0.5) <= 1e-5,
      "S_RN(fr*fb)/S_RN(0)=%g expected ~0.5" % ratio)

# Full-formula agreement on a dense random grid, tol 1e-13 (identical algebra
# evaluated in float64; only ordering differences allowed).
fgrid = np.linspace(0, 1.5 * par.fb, 601)
check("S_RN_matches_matlab_formula",
      rel_err(sicopr.S_RN(fgrid, -6.0, -1.0, par), ml_S_RN(fgrid, -6.0, -1.0, par)) <= 1e-13,
      "S_RN transcription mismatch")

# ===========================================================================
# 2. N_s (ML 2698-2738 vs py 3391-3415)  Tier 2 analytic
# ===========================================================================

fb = 2.5e10
df = 1e7
f = np.arange(0, 3.0e10 + df / 2, df)   # fb/2 = 12.5e9 lands exactly on index 1250
i_half = 1250
sig_ns = 3e-3

par_ns = SimpleNamespace(fb=fb, f_hp=1e9)
OP_178 = SimpleNamespace(RIT_REF_PTR='Clause_178')   # mixed case exercises lower()
OP_179 = SimpleNamespace(RIT_REF_PTR='clause_179')

ns178 = sicopr.N_s(f, par_ns, sig_ns, OP_178)
# ML 2705: inq = find(f<=fb/2,1,'last') INCLUDES the point at exactly fb/2.
check("N_s_178_edge_inclusive",
      ns178[i_half] > 0 and ns178[i_half + 1] == 0,
      "edge at fb/2: got Ns[%d]=%g Ns[%d]=%g" % (i_half, ns178[i_half],
                                                 i_half + 1, ns178[i_half + 1]))
# ML 2710: flat level is exactly 2*sigma^2/fb.
check("N_s_178_level_exact",
      rel_err(ns178[:i_half + 1], 2 * sig_ns ** 2 / fb * np.ones(i_half + 1)) <= 1e-15,
      "flat level wrong")
# Single-sided PSD integrates to sigma^2 (2*s^2/fb * fb/2). Riemann sum on a
# flat function is exact except for the inclusive edge bin: (1251/1250) - 1 =
# 8e-4, so tol 1e-3.
mass178 = float(np.sum(ns178) * df)
check("N_s_178_mass_sigma2", abs(mass178 - sig_ns ** 2) <= 1e-3 * sig_ns ** 2,
      "mass=%g expected %g" % (mass178, sig_ns ** 2))

ns179 = sicopr.N_s(f, par_ns, sig_ns, OP_179)
# ML 2715-2716 transcription at every kept point, tol 1e-15 (same algebra).
beta = 1 - (2 * par_ns.f_hp / fb) * np.arctan(fb / (2 * par_ns.f_hp))
ref179 = np.zeros(len(f))
ref179[:i_half + 1] = ((2 * sig_ns ** 2) / (beta * fb)
                       * (f[:i_half + 1] / par_ns.f_hp) ** 2
                       / (1 + (f[:i_half + 1] / par_ns.f_hp) ** 2))
check("N_s_179_matches_matlab_formula", rel_err(ns179, ref179) <= 1e-15,
      "clause_179 transcription mismatch")
# Analytic: integral of eq 179-23 over [0, fb/2] is exactly sigma^2 (the beta
# normalisation is designed for that). Riemann-sum discretisation error is
# O(df/fb) ~ 4e-4, tol 1e-3.
mass179 = float(np.sum(ns179) * df)
check("N_s_179_mass_sigma2", abs(mass179 - sig_ns ** 2) <= 1e-3 * sig_ns ** 2,
      "mass=%g expected %g" % (mass179, sig_ns ** 2))

# ML 2712-2714: f_hp<=0 must raise for clause_179.
try:
    sicopr.N_s(f, SimpleNamespace(fb=fb, f_hp=0.0), sig_ns, OP_179)
    raised = False
except Exception:
    raised = True
check("N_s_179_fhp_zero_raises", raised, "no error for f_hp=0")

# Cross-path consistency: the inlined copy _S_IN__N_s (py 6546) must agree
# with top-level N_s on shared inputs. Identical code -> tol 1e-15.
for op, tag in ((OP_178, '178'), (OP_179, '179')):
    check("N_s_copy_SIN_agrees_%s" % tag,
          rel_err(sicopr._S_IN__N_s(f, par_ns, sig_ns, op),
                  sicopr.N_s(f, par_ns, sig_ns, op)) <= 1e-15,
          "_S_IN__N_s diverges from N_s")

# ===========================================================================
# 3. get_sigma_noise (ML 7950-7973 vs py 11489-11521)  Tier 2/3
# ===========================================================================

df2 = 1e8
fax = np.arange(0, 2.5e10 + df2 / 2, df2)   # 251 pts; fb/2=12.5e9 at index 125
sigma_bn = 5e-3

def mk_chdata2(sdd21):
    return [SimpleNamespace(), SimpleNamespace(faxis=fax, sdd21=sdd21)]

# Flat channel, H_r ~ 1 (f_r tiny denominator scale -> huge corner), f_hp=0:
# sigma_NE = sigma_bn*sqrt(mean(1)) = sigma_bn, sigma_HP = sigma_bn*mean(1).
par_gs = SimpleNamespace(f_r=1e3, fb=fb, f_hp=0.0)
s_ne, s_hp = sicopr.get_sigma_noise(np.ones(len(fax)), par_gs,
                                 mk_chdata2(np.ones(len(fax))), sigma_bn)
check("get_sigma_noise_flat_unity",
      abs(s_ne - sigma_bn) <= 1e-9 * sigma_bn and abs(s_hp - sigma_bn) <= 1e-12 * sigma_bn,
      "sigma_NE=%g sigma_HP=%g expected %g" % (s_ne, s_hp, sigma_bn))

# MATLAB quirk (ML 7973): sigma_HP = sigma_bn * MEAN (no sqrt). Verify the
# port reproduces the quirk: with f_hp=5e9, c=mean(|H_hp|^2) is well below 1,
# so sigma_bn*c and sigma_bn*sqrt(c) differ by >1%.
f_hp_q = 5e9
H_hp_q = (-1j * fax / f_hp_q) / (1 + 1j * fax / f_hp_q)
c_q = float(np.mean(np.abs(H_hp_q[:126]) ** 2))       # ML H_hp(1:idxfbby2), idx=126
par_gsq = SimpleNamespace(f_r=1e3, fb=fb, f_hp=f_hp_q)
_, s_hp_q = sicopr.get_sigma_noise(np.ones(len(fax)), par_gsq,
                                mk_chdata2(np.ones(len(fax))), sigma_bn)
check("get_sigma_noise_hp_no_sqrt_quirk",
      abs(s_hp_q - sigma_bn * c_q) <= 1e-12 * sigma_bn
      and abs(s_hp_q - sigma_bn * np.sqrt(c_q)) > 1e-2 * sigma_bn,
      "sigma_HP=%g, sigma_bn*c=%g, sigma_bn*sqrt(c)=%g"
      % (s_hp_q, sigma_bn * c_q, sigma_bn * np.sqrt(c_q)))

# Full transcription of ML 7954-7973 with random complex inputs, tol 1e-13.
rng = np.random.default_rng(20260709)
sdd21_r = (rng.standard_normal(len(fax)) + 1j * rng.standard_normal(len(fax))) * 0.3
H_ctf_r = (rng.standard_normal(len(fax)) + 1j * rng.standard_normal(len(fax))) * 0.5
par_gr = SimpleNamespace(f_r=0.75, fb=fb, f_hp=3e9)

H_r_ml = 1.0 / np.polyval([1, 2.613126, 3.414214, 2.613126, 1],
                          1j * fax / (par_gr.f_r * par_gr.fb))          # ML 7954
idxfbby2_ml = int(np.nonzero(fax >= fb / 2)[0][0]) + 1                  # ML 7955, 1-based
H_hp_ml = (-1j * fax / par_gr.f_hp) / (1 + 1j * fax / par_gr.f_hp)      # ML 7964
H_np_ml = sdd21_r * H_ctf_r * H_r_ml * H_hp_ml                          # ML 7969
sne_ml = sigma_bn * np.sqrt(np.mean(np.abs(H_np_ml[:idxfbby2_ml] ** 2)))  # ML 7972
shp_ml = sigma_bn * np.mean(np.abs(H_hp_ml[:idxfbby2_ml] ** 2))         # ML 7973
sne_py, shp_py = sicopr.get_sigma_noise(H_ctf_r, par_gr, mk_chdata2(sdd21_r), sigma_bn)
check("get_sigma_noise_matches_matlab_formula",
      abs(sne_py - sne_ml) <= 1e-13 * abs(sne_ml)
      and abs(shp_py - shp_ml) <= 1e-13 * abs(shp_ml),
      "NE %g vs %g, HP %g vs %g" % (sne_py, sne_ml, shp_py, shp_ml))

# Index-window poison: ML averages H_np(1:idxfbby2) where idxfbby2 is the
# FIRST index with f >= fb/2 (=126 1-based = 125 0-based, inclusive).
sd_in = np.zeros(len(fax), dtype=complex); sd_in[125] = 1.0   # last included bin
sd_out = np.zeros(len(fax), dtype=complex); sd_out[126] = 1.0  # first excluded bin
ne_in, _ = sicopr.get_sigma_noise(np.ones(len(fax)), par_gs, mk_chdata2(sd_in), sigma_bn)
ne_out, _ = sicopr.get_sigma_noise(np.ones(len(fax)), par_gs, mk_chdata2(sd_out), sigma_bn)
check("get_sigma_noise_idx_window_inclusive",
      ne_in > 0 and ne_out == 0,
      "bin at fb/2 included=%g, bin above=%g (expected >0 and ==0)" % (ne_in, ne_out))

# ===========================================================================
# 4. get_sigma_eta_ACCM_noise (ML 7926-7948 vs py 11447-11483)  Tier 2
#    THE /1e9 unit-factor test for eta_0 in V^2/GHz (ML 6526, 7936).
# ===========================================================================

df3 = 2e8
fax3 = np.arange(0, 6.0e10 + df3 / 2, df3)      # 301 pts, 0..60 GHz
eta_0 = 8.2e-9                                   # V^2/GHz
ones3 = np.ones(len(fax3))

def mk_ch(faxis, sdc21_list):
    return [SimpleNamespace(faxis=faxis, sdc21=s) for s in sdc21_list]

# Thermal only: H_sy=H_r=H_ctf=1 -> sigma_N1 = sqrt(eta_0 * sum(diff(f))/1e9)
# = sqrt(eta_0 * 60e9/1e9) = sqrt(eta_0*60). Closed form, tol 1e-12.
par_a = SimpleNamespace(eta_0=eta_0, AC_CM_RMS=0.0, AC_CM_RMS_TX=0.0,
                        ACCM_MAX_Freq=3.0e10)
sN = sicopr.get_sigma_eta_ACCM_noise(mk_ch(fax3, [ones3]), par_a, ones3, ones3, ones3)
expect_sN = np.sqrt(eta_0 * 60.0)
check("ACCM_thermal_1e9_unit_factor", abs(sN - expect_sN) <= 1e-12 * expect_sN,
      "sigma_N=%g expected %g (eta_0 V^2/GHz => /1e9 on Hz axis)" % (sN, expect_sN))

# ML 7936 uses H(2:end): a poison in element 1 (0-based 0) must not matter.
poison_sy = ones3.copy(); poison_sy[0] = 1e6
sN_p = sicopr.get_sigma_eta_ACCM_noise(mk_ch(fax3, [ones3]), par_a, poison_sy, ones3, ones3)
check("ACCM_thermal_excludes_dc_bin", abs(sN_p - sN) <= 1e-14 * sN,
      "DC bin leaked into the integral: %g vs %g" % (sN_p, sN))

# AC-CM path (ML 7937-7945): sdc21=1, f_int = f<=30 GHz (nf=151, f_int(end)=30e9):
# per channel sigma_acc = sqrt(2*TX^2*(f_int(end)-0)/f_int(end)) = sqrt(2)*TX;
# two channels -> norm -> 2*TX; total = hypot(sigma_N1, 2*TX). Tol 1e-12.
TX = 0.02
par_b = SimpleNamespace(eta_0=eta_0, AC_CM_RMS=1.0, AC_CM_RMS_TX=TX,
                        ACCM_MAX_Freq=3.0e10)
sN_cm = sicopr.get_sigma_eta_ACCM_noise(mk_ch(fax3, [ones3, ones3]), par_b,
                                     ones3, ones3, ones3)
expect_cm = float(np.hypot(expect_sN, 2 * TX))
check("ACCM_cm_path_closed_form", abs(sN_cm - expect_cm) <= 1e-12 * expect_cm,
      "sigma_N=%g expected %g" % (sN_cm, expect_cm))

# Window poison via sdc21 (only enters the ACCM term): ML 7942 uses
# H_dc(2:length(f_int)) = 1-based 2..151 = 0-based 1..150. Poison at 0-based
# 150 (last included) must change the result; at 0-based 151 (first excluded)
# must not.
sdc_in = ones3.copy(); sdc_in[150] = 100.0
sdc_out = ones3.copy(); sdc_out[151] = 100.0
sN_in = sicopr.get_sigma_eta_ACCM_noise(mk_ch(fax3, [sdc_in, ones3]), par_b,
                                     ones3, ones3, ones3)
sN_out = sicopr.get_sigma_eta_ACCM_noise(mk_ch(fax3, [sdc_out, ones3]), par_b,
                                      ones3, ones3, ones3)
check("ACCM_window_index_mapping",
      abs(sN_in - sN_cm) > 1e-6 and abs(sN_out - sN_cm) <= 1e-14 * sN_cm,
      "in-window poison delta=%g (expect >0), out-of-window delta=%g (expect 0)"
      % (abs(sN_in - sN_cm), abs(sN_out - sN_cm)))

# ===========================================================================
# 5. get_PSDs (ML 6512-6784 vs py 9494-9789)  Tier 3 composite
# ===========================================================================

M = 8
num_ui = 64
NM = num_ui * M                                   # 512
fbp = 2.5e10
delta_f = fbp / num_ui
n_fvec = NM // 2 + 1
fvec = np.arange(n_fvec) * delta_f
L = 4
sigma_X2 = (L ** 2 - 1) / (3.0 * (L - 1) ** 2)
SNR_TX = 32.5
dw = 1
Nb = 4
bmax = np.array([0.7, 0.2, 0.1, 0.05])
bmin = np.array([-0.7, -0.2, -0.1, -0.05])
A_DD, sigma_RJ = 0.02, 0.01
N_qb = 6

param = SimpleNamespace(
    num_ui_RXFF_noise=num_ui, samples_per_ui=M, levels=L, fb=fbp,
    SNR_TX=SNR_TX, RxFFE_cmx=dw, bmax=bmax, bmin=bmin, ndfe=Nb,
    eta_0=8.2e-9, A_DD=A_DD, sigma_RJ=sigma_RJ, N_qb=N_qb, S_tn_w_AM=1,
    clip_method='Fast', P_qc=1e-5,
    # CTLE fields for the injected real S_RN:
    CTLE_fp1=1.2e10, CTLE_fz=4e9, CTLE_fp2=3.5e10, f_HP=1e8, f_r=0.75)
OP = SimpleNamespace(COMPUTE_COM=False, WO_TXFFE=True, PSDRXCAL=False,
                     TDMODE=False, LIMIT_JITTER_CONTRIB_TO_DFE_SPAN=False,
                     BinSize=1e-4)

cursor_py = 5 * M + 3                             # 0-based cursor, = MATLAB 44
h = rng.standard_normal(520) * 1e-3
h[cursor_py] = 0.8
h[cursor_py - M] = 0.1
h[cursor_py + 1 * M] = 0.6                        # >= bmax(1)*cursor -> subtract
h[cursor_py + 2 * M] = -0.5                       # <= bmin(2)*cursor -> subtract
h[cursor_py + 3 * M] = 0.05                       # inside band -> zero
h[cursor_py + 4 * M] = 0.06                       # >= bmax(4)*cursor -> subtract

ctle_imp = rng.standard_normal(300) * 0.02
ctle_imp[40:60] += 0.15
thru_pulse = rng.standard_normal(480) * 1e-3
thru_pulse[cursor_py] = 0.75
agg_pulse = rng.standard_normal(700) * 0.01
agg_pulse[100:104] += 0.05

chdata = [SimpleNamespace(ctle_imp_response=ctle_imp,
                          pulse_response_w_CFT_TXFFE_noRxFFE=thru_pulse,
                          faxis=np.arange(0, 4e10, 1e8)),
          SimpleNamespace(pulse_response_w_CFT_TXFFE_noRxFFE=agg_pulse)]

G_DC, G_DC2 = -6.0, -1.0

# ---- call 1: WO_TXFFE=1, COMPUTE_COM=0 -> S_rn, S_in (ML 6550-6579) ----
res = sicopr.get_PSDs(None, [], [], [], G_DC, G_DC2, param, chdata, OP,
                   _S_RN_fn=sicopr.S_RN, _S_IN_fn=sicopr.S_IN, _H_interp_fn=sicopr.H_interp)

check("get_PSDs_fvec_grid",
      len(res.fvec) == n_fvec and rel_err(res.fvec, fvec) <= 1e-15
      and abs(res.fvec[-1] - M * fbp / 2) <= 1e-3,
      "fvec length/step/endpoint mismatch (ML 6528-6529)")

# Oracle ML 6551-6559 (uses ml_S_RN, itself verified in section 1).
S_RN_of_f = ml_S_RN(fvec, G_DC, G_DC2, param)
rxn_psd = np.real(ml_double_sided(S_RN_of_f)) / 1e9          # ML 6552-6553
rxn_rms = np.sqrt(np.sum(rxn_psd) * delta_f)                 # ML 6554
S_rn_fold = ml_fold(rxn_psd, num_ui, M)                      # ML 6555
S_rn_ml = np.real(ml_double_sided(S_rn_fold[:num_ui // 2 + 1]))  # ML 6556-6557
S_rn_rms_ml = np.sqrt(np.sum(S_rn_ml) * delta_f)             # ML 6559

check("get_PSDs_S_rn_matches_matlab",
      len(np.atleast_1d(res.S_rn)) == num_ui and rel_err(res.S_rn, S_rn_ml) <= 1e-13,
      "S_rn transcription mismatch (ML 6551-6558)")
check("get_PSDs_S_rn_rms_matches_matlab",
      abs(res.S_rn_rms - S_rn_rms_ml) <= 1e-13 * S_rn_rms_ml,
      "S_rn_rms mismatch")
# Physics invariant: the fold conserves total power, and the folded spectrum
# is symmetric, so S_rn_rms equals the unfolded rxn_rms. Tol 1e-12 (pure
# rearrangement of float sums).
check("get_PSDs_fold_conserves_power",
      abs(res.S_rn_rms - rxn_rms) <= 1e-12 * rxn_rms,
      "fold lost/gained power: %g vs %g" % (res.S_rn_rms, rxn_rms))
check("get_PSDs_S_in_zero_wo_psdrxcal",
      res.S_in == 0 and res.S_in_rms == 0, "S_in should be 0 (ML 6577-6578)")

# ---- call 2: WO_TXFFE=0, COMPUTE_COM=0 -> S_xn,S_tn,S_jn,S_qn,S_n ----
OP.WO_TXFFE = False
res = sicopr.get_PSDs(res, h, cursor_py, [], G_DC, G_DC2, param, chdata, OP,
                   _S_RN_fn=sicopr.S_RN, _S_IN_fn=sicopr.S_IN, _H_interp_fn=sicopr.H_interp)

# Oracle S_xn, ML 6600-6621 (single aggressor, xchan ML 2).
k = agg_pulse.copy()
k = np.concatenate([k, np.zeros(NM - len(k))]) if NM > len(k) else k[:NM]
hxn = np.array([np.linalg.norm(k[i1 - 1::M]) for i1 in range(1, M + 1)])  # ML 6609-6611
iphase_ml = int(np.argmax(hxn)) + 1                                       # ML 6612, 1-based
hrn_ml = k[iphase_ml - 1::M]                                              # ML 6613
S_xn_ml = sigma_X2 * np.abs(np.fft.fft(hrn_ml)) ** 2 / fbp                # ML 6615
S_xn_rms_ml = np.sqrt(np.sum(S_xn_ml) * delta_f)
check("get_PSDs_S_xn_matches_matlab", rel_err(res.S_xn, S_xn_ml) <= 1e-13,
      "S_xn transcription mismatch (ML 6600-6616)")
check("get_PSDs_S_xn_rms_matches_matlab",
      abs(res.S_xn_rms - S_xn_rms_ml) <= 1e-13 * S_xn_rms_ml, "S_xn_rms mismatch")
# Seam: Python stores iphase 0-based (ML 1-based). Equivalence holds if the
# selected decimated response is identical and the consumer (get_pdf) treats
# ixphase as 0-based (it documents that it does).
check("get_PSDs_iphase_seam_selects_same_phase",
      int(res.iphase[1]) == iphase_ml - 1 and rel_err(res.hk[1].hrn, hrn_ml) <= 1e-15,
      "phase pick differs: py %s vs ML %d" % (res.iphase[1], iphase_ml))

# Oracle S_tn, ML 6636-6660. filter(ones(1,M),1,x) is a causal moving sum.
htn = np.convolve(ctle_imp, np.ones(M))[:len(ctle_imp)]                   # ML 6637
cursor_ml = cursor_py + 1
htn = htn[(np.mod(cursor_ml - 1, M) + 1) - 1:]                            # ML 6647
htn = htn[::M]                                                            # ML 6649
hext = (np.concatenate([htn, np.zeros(num_ui - len(htn))])
        if num_ui > len(htn) else htn[:num_ui])                           # ML 6651-6655
S_tn_ml = (sigma_X2 ** param.S_tn_w_AM * 10 ** (-SNR_TX / 10)
           * np.abs(np.fft.fft(hext)) ** 2 / fbp)                         # ML 6659
check("get_PSDs_S_tn_matches_matlab", rel_err(res.S_tn, S_tn_ml) <= 1e-13,
      "S_tn transcription mismatch (ML 6636-6659)")

# Oracle S_jn non-LIMIT branch, ML 6670-6694.
so = np.mod(cursor_ml, M)                                                 # ML 6670
if so <= 1:
    so += M                                                               # ML 6672-6674
early = h[(so - 1) - 1::M]                                                # ML 6679
late = h[(so + 1) - 1::M]                                                 # ML 6680
early = early[:len(late)]                                                 # ML 6683
h_J = (late - early) / 2 * M                                              # ML 6684
h_J = (np.concatenate([h_J, np.zeros(num_ui - len(h_J))])
       if num_ui > len(h_J) else h_J[:num_ui])                            # ML 6686-6690
S_jn_ml = sigma_X2 * (A_DD ** 2 + sigma_RJ ** 2) * np.abs(np.fft.fft(h_J)) ** 2 / fbp
S_rj_ml = sigma_X2 * (sigma_RJ ** 2) * np.abs(np.fft.fft(h_J)) ** 2 / fbp
check("get_PSDs_S_jn_matches_matlab", rel_err(res.S_jn, S_jn_ml) <= 1e-13,
      "S_jn transcription mismatch (ML 6670-6691)")
check("get_PSDs_S_rj_jn_matches_matlab", rel_err(res.S_rj_jn, S_rj_ml) <= 1e-13,
      "S_rj_jn transcription mismatch (ML 6693)")

# Oracle S_qn fast path, ML 6705-6734.
sample_idx_ml = np.mod(cursor_ml - 1, M) + 1                              # ML 6706
spr = thru_pulse[sample_idx_ml - 1:]                                      # ML 6707
spr = spr[::M]                                                            # ML 6709
spr = (np.concatenate([spr, np.zeros(num_ui - len(spr))])
       if num_ui > len(spr) else spr[:num_ui])                            # ML 6710-6714
adc_clip_ml = float(np.sum(np.abs(spr)))                                  # ML 6727
adc_lsb = 2 * adc_clip_ml / (2 ** N_qb - 1)                               # ML 6729
sigma_Q = adc_lsb / np.sqrt(12)                                           # ML 6730
S_qn_ml = sigma_Q ** 2 / fbp * np.ones(num_ui)                            # ML 6731
check("get_PSDs_S_qn_fast_matches_matlab",
      abs(res.adc_clip - adc_clip_ml) <= 1e-13 * adc_clip_ml
      and rel_err(res.S_qn, S_qn_ml) <= 1e-13,
      "S_qn fast-path transcription mismatch (ML 6705-6734)")

# Total, ML 6739-6740.
S_n_ml = S_rn_ml + S_tn_ml + S_xn_ml + S_jn_ml + S_qn_ml + 0.0
check("get_PSDs_S_n_total_matches_matlab", rel_err(res.S_n, S_n_ml) <= 1e-12,
      "S_n sum mismatch (ML 6739)")

# ---- LIMIT_JITTER branch (ML 6676-6677): EXPECTED FAIL, +1 sample shift ----
OP_lim = SimpleNamespace(**vars(OP))
OP_lim.LIMIT_JITTER_CONTRIB_TO_DFE_SPAN = True
res_lim = sicopr.get_PSDs(SimpleNamespace(S_rn=res.S_rn.copy(), S_in=0,
                                       S_rn_rms=res.S_rn_rms, S_in_rms=0),
                       h, cursor_py, [], G_DC, G_DC2, param, chdata, OP_lim,
                       _S_RN_fn=sicopr.S_RN, _S_IN_fn=sicopr.S_IN, _H_interp_fn=sicopr.H_interp)
kvec = np.arange(-1, Nb + 1)
early_lim = h[(cursor_ml - 1 + M * kvec) - 1]                             # ML 6676
late_lim = h[(cursor_ml + 1 + M * kvec) - 1]                              # ML 6677
early_lim = early_lim[:len(late_lim)]
h_J_lim = (late_lim - early_lim) / 2 * M
h_J_lim = np.concatenate([h_J_lim, np.zeros(num_ui - len(h_J_lim))])
S_jn_lim_ml = (sigma_X2 * (A_DD ** 2 + sigma_RJ ** 2)
               * np.abs(np.fft.fft(h_J_lim)) ** 2 / fbp)
check("get_PSDs_S_jn_LIMIT_matches_matlab",
      rel_err(res_lim.S_jn, S_jn_lim_ml) <= 1e-13,
      "DIVERGENT: py idx_early/idx_late (py 9673-9674) are +1 sample late vs "
      "ML 6676-6677; slope sampled at cursor+1 instead of cursor")

# ---- ADC 'slow' clip path (ML 6716-6725): EXPECTED FAIL, not ported ----
param_slow = SimpleNamespace(**vars(param))
param_slow.clip_method = 'Slow'
res_slow = sicopr.get_PSDs(SimpleNamespace(S_rn=res.S_rn.copy(), S_in=0,
                                        S_rn_rms=res.S_rn_rms, S_in_rms=0),
                        h, cursor_py, [], G_DC, G_DC2, param_slow, chdata, OP,
                        _S_RN_fn=sicopr.S_RN, _S_IN_fn=sicopr.S_IN, _H_interp_fn=sicopr.H_interp)
# Oracle: ML 6717-6724 using sicopr.py's public PDF machinery as stand-in
# (get_pdf_from_sampled_signal/conv_fct/CDF_inv_ev are audited in B07/B08).
sig_pdf = sicopr.get_pdf_from_sampled_signal(spr, L, OP.BinSize)             # ML 6717
sigma_noise_ml = float(np.sqrt(res_slow.S_in_rms ** 2 + res_slow.S_rn_rms ** 2
                               + res_slow.S_xn_rms ** 2 + res_slow.S_tn_rms ** 2
                               + res_slow.S_rj_rms ** 2))                 # ML 6719
noise_pdf = SimpleNamespace(**vars(sig_pdf))
noise_pdf.y = (1 / (np.sqrt(2 * np.pi) * sigma_noise_ml)
               * np.exp(-np.asarray(sig_pdf.x) ** 2 / (2 * sigma_noise_ml ** 2))
               * OP.BinSize)                                              # ML 6720
snp = sicopr.conv_fct(sig_pdf, noise_pdf)                                    # ML 6721
snp_cdf = np.cumsum(snp.y)                                                # ML 6722
adc_clip_slow_ml = -sicopr.CDF_inv_ev(param.P_qc, snp, snp_cdf)              # ML 6724
check("get_PSDs_adc_clip_slow_matches_matlab",
      abs(res_slow.adc_clip - adc_clip_slow_ml) <= 1e-6 * abs(adc_clip_slow_ml),
      "DIVERGENT: py 9721-9730 substitutes max(abs)+3*sigma (=%g) for the ML "
      "PDF/CDF_inv_ev clip (=%g)" % (res_slow.adc_clip, adc_clip_slow_ml))
check("get_PSDs_slow_sets_ctle_signal_sigma",
      hasattr(res_slow, 'ctle_signal_sigma'),
      "DIVERGENT: ML 6725 sets result.ctle_signal_sigma; py slow path never does")

# ---- call 3: COMPUTE_COM=1, WO_TXFFE=1 -> H_rxffe_2 and S_rn scaling ----
OP.COMPUTE_COM = True
OP.WO_TXFFE = True
w = np.array([-0.1, 0.85, 0.15, -0.05])
res.w = w
S_rn_before = np.array(res.S_rn, dtype=float).copy()
res = sicopr.get_PSDs(res, h, cursor_py, [], G_DC, G_DC2, param, chdata, OP,
                   _S_RN_fn=sicopr.S_RN, _S_IN_fn=sicopr.S_IN, _H_interp_fn=sicopr.H_interp)

# Oracle ML 6536-6542 with 1-based nn. The Python loop uses 0-based nn with
# the same (nn-dw-1) exponent, a pure one-UI linear phase that cancels in
# |H|^2, so this must PASS if that reasoning is right.
T_b = 1.0 / fbp
H_ml = np.zeros(n_fvec, dtype=complex)
for nn in range(1, len(w) + 1):                                           # ML 6537
    H_ml += w[nn - 1] * np.exp(-1j * 2 * np.pi * fvec * T_b * (nn - dw - 1))
H2_ml = np.abs(H_ml) ** 2                                                 # ML 6540
H2_ml = np.real(ml_double_sided(H2_ml[:num_ui // 2 + 1]))                 # ML 6541-6542
check("get_PSDs_H_rxffe2_matches_matlab", rel_err(res.H_rxffe_2, H2_ml) <= 1e-12,
      "|H_rxffe|^2 mismatch (ML 6536-6542); enumerate offset does NOT cancel")
check("get_PSDs_S_rn_scaled_by_H2", rel_err(res.S_rn, S_rn_before * H2_ml) <= 1e-12,
      "S_rn *= |H_rxffe|^2 mismatch (ML 6582)")

# ---- call 4: COMPUTE_COM=1, WO_TXFFE=0 -> S_isi, S_G, Sn_rho (ML 6744-6783) ----
OP.WO_TXFFE = False
res = sicopr.get_PSDs(res, h, cursor_py, [], G_DC, G_DC2, param, chdata, OP,
                   _S_RN_fn=sicopr.S_RN, _S_IN_fn=sicopr.S_IN, _H_interp_fn=sicopr.H_interp)

# Oracle ML 6755-6776.
samp_idx_ml = np.arange((np.mod(cursor_ml - 1, M) + 1), len(h) + 1, M)    # ML 6755, 1-based
cursor_n_ml = int(np.nonzero(samp_idx_ml == cursor_ml)[0][0]) + 1         # ML 6756, 1-based
hisi = h[samp_idx_ml - 1].copy()
hisi = np.concatenate([hisi, np.zeros(max(0, num_ui - len(hisi)))])[:num_ui]  # ML 6758-6759
for ii in range(1, len(hisi) + 1):                                        # ML 6761, 1-based
    if ii == cursor_n_ml:
        cursor_val = hisi[ii - 1]
        hisi[ii - 1] = 0.0
    elif cursor_n_ml + 1 <= ii <= cursor_n_ml + Nb:
        ib = ii - cursor_n_ml                                             # 1-based
        if hisi[ii - 1] >= bmax[ib - 1] * cursor_val:
            hisi[ii - 1] -= bmax[ib - 1] * cursor_val
        elif hisi[ii - 1] <= bmin[ib - 1] * cursor_val:
            hisi[ii - 1] -= bmin[ib - 1] * cursor_val
        else:
            hisi[ii - 1] = 0.0
S_isi_ml = sigma_X2 * np.abs(np.fft.fft(hisi)) ** 2 / fbp                 # ML 6776
check("get_PSDs_S_isi_matches_matlab", rel_err(res.S_isi, S_isi_ml) <= 1e-13,
      "S_isi/DFE-clip transcription mismatch (ML 6755-6776)")

S_G_ml = S_tn_ml * H2_ml + S_rj_ml * H2_ml + S_rn_before * H2_ml + 0.0 * H2_ml
check("get_PSDs_S_G_matches_matlab", rel_err(res.S_G, S_G_ml) <= 1e-12,
      "S_G mismatch (ML 6779)")
S_n4_ml = (S_rn_before * H2_ml + S_tn_ml * H2_ml + S_xn_ml * H2_ml
           + S_jn_ml * H2_ml + S_qn_ml + 0.0 * H2_ml)
check("get_PSDs_Sn_rho_matches_matlab",
      rel_err(res.Sn_rho, S_isi_ml + S_n4_ml) <= 1e-12,
      "Sn_rho mismatch (ML 6781)")

# ---- Stub defaults (py 9443-9456): EXPECTED FAIL, API-level trap ----
# When get_PSDs is called without injection (as MATLAB callers would), the
# defaults are stubs: flat eta_0 (no /2, no CTLE/BT filter) and zero S_IN.
stub_srn = sicopr._get_PSDs__S_RN(fvec, G_DC, G_DC2, param)
real_srn = sicopr.S_RN(fvec, G_DC, G_DC2, param)
xcheck("get_PSDs_stub_SRN_agrees_with_S_RN", rel_err(stub_srn, real_srn) <= 1e-6,
      "DIVERGENT default: _get_PSDs__S_RN is a flat-eta_0 stub, not ML S_RN "
      "(in-repo call sites inject the real S_RN, so mainline is unaffected)")
par_sin = SimpleNamespace(**vars(param))
par_sin.sigma_ns = 2e-3
par_sin.f_hp = 1e9
stub_sin = sicopr._get_PSDs__S_IN(fvec, np.ones(n_fvec), G_DC, G_DC2, par_sin, OP_179)
real_sin = sicopr.S_IN(fvec, np.ones(n_fvec), G_DC, G_DC2, par_sin, OP_179)
xcheck("get_PSDs_stub_SIN_agrees_with_S_IN", rel_err(stub_sin, real_sin) <= 1e-6,
      "DIVERGENT default: _get_PSDs__S_IN returns zeros, not ML S_IN "
      "(in-repo call sites inject the real S_IN, so mainline is unaffected)")

finish()
