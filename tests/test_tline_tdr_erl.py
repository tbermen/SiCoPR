"""Audit batch B06: G5 S-parameter synthesis + TDR/ERL.

Functions under audit (MATLAB com_ieee8023_4p15p0.m -> sicopr.py):
  synth_tline          ML 11434-11458 -> py 17475-17524
  get_TDR              ML 7034-7345   -> py 10231-10487 (+ helper 10212-10224)
  TDR_ERL_Processing   ML 4503-4597   -> py 6632-6725
  R_series2            ML 4359-4364   -> py 6348-6367
  r_parrelell2         ML 9520-9524   -> py 13577-13597

Traps checked (audit prompt section 3 item 5, section 4 items 8/13):
  - complex propagation constant gamma: sqrt/log branch, f=0 fix (no NaN leak).
  - MATLAB left-division (mldivide) vs Python `/` in the s2p RL renorm.
  - reflection renormalisation (4-port TDR_RL) and matched-load -> zero reflection.
  - ERL summary/min and ERL_ONLY file_names.

Oracles transcribed from the cited MATLAB lines. Matched-load reflection and
passivity/reciprocity are analytic invariants. FAIL rows document divergences.
Run: python tests/test_tline_tdr_erl.py
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


# ===========================================================================
# 1. synth_tline (ML 11434-11458 vs py 17475-17524)
# ===========================================================================
f = np.linspace(0, 40e9, 128)          # includes f=0
Z_c, Z_0 = 90.0, 50.0
gc = np.array([0.0, 1.734e-3, 1.455e-4])
tau = 6.141e-3
d = 0.15

s11, s12, s21, s22 = sicopr.synth_tline(f, Z_c, Z_0, gc, tau, d)


def synth_oracle(f, Z_c, Z_0, gc, tau, d):
    """Transcription of ML 11435-11458."""
    f_GHz = f / 1e9
    g1 = gc[1] * (1 + 1j)
    with np.errstate(divide='ignore', invalid='ignore'):
        g2 = gc[2] * (1 - 2j / np.pi * np.log(f_GHz)) + 2j * np.pi * tau
    gamma = gc[0] + g1 * np.sqrt(f_GHz) + g2 * f_GHz
    gamma[f_GHz == 0] = gc[0]
    rho = 0.0 if d == 0 else (Z_c - 2 * Z_0) / (Z_c + 2 * Z_0)
    egd = np.exp(-d * gamma)
    egd2 = egd ** 2
    den = 1 - rho ** 2 * egd2
    o11 = rho * (1 - egd2) / den
    o21 = (1 - rho ** 2) * egd / den
    return o11, o21, o21, o11


e11, e12, e21, e22 = synth_oracle(f, Z_c, Z_0, gc, tau, d)
check("synth_tline_matches_matlab",
      rel_err(s11, e11) <= 1e-12 and rel_err(s21, e21) <= 1e-12,
      "synth_tline diverges from the 93A tline transcription")
check("synth_tline_no_nan_at_dc",
      np.all(np.isfinite(s11)) and np.all(np.isfinite(s21)),
      "NaN/Inf leaked at f=0 (log(0) not overridden)")
# Reciprocal (s12=s21) and symmetric (s11=s22).
check("synth_tline_reciprocal_symmetric",
      rel_err(s12, s21) <= 1e-15 and rel_err(s11, s22) <= 1e-15,
      "tline not reciprocal/symmetric")
# Passive lossy line: |s21| <= 1.
check("synth_tline_passive",
      np.max(np.abs(s21)) <= 1.0 + 1e-9,
      "|s21| > 1 (non-passive tline)")
# d=0 -> ideal through (rho=0 -> s11=0, s21=1).
z11, _, z21, _ = sicopr.synth_tline(f, Z_c, Z_0, gc, tau, 0.0)
check("synth_tline_zero_length_is_through",
      np.max(np.abs(z11)) <= 1e-15 and rel_err(z21, np.ones_like(z21)) <= 1e-15,
      "d=0 did not give an ideal through")

# ===========================================================================
# 2. R_series2 / r_parrelell2 (ML 4359-4364, 9520-9524)
# ===========================================================================
zref = 50.0
fr = np.linspace(0, 40e9, 16)
Rs = 20.0
Ss = sicopr.R_series2(zref, fr, Rs).Parameters
check("R_series2_matches_matlab",
      rel_err(Ss[0, 0], Rs / (Rs + 2 * zref)) <= 1e-14
      and rel_err(Ss[1, 0], 2 * zref / (Rs + 2 * zref)) <= 1e-14,
      "R_series2 != series-resistor S-parameter transcription")
check("R_series2_limits",
      abs(sicopr.R_series2(zref, fr, 0.0).Parameters[1, 0, 0] - 1.0) <= 1e-14
      and abs(sicopr.R_series2(zref, fr, 1e12).Parameters[0, 0, 0] - 1.0) <= 1e-6,
      "series R=0 not through, or R=inf not open")

rpad = 200.0
Sp = sicopr.r_parrelell2(zref, fr, rpad).Parameters
# MATLAB literal: -zref/(rpad*(zref/rpad+2)) == -zref/(zref+2*rpad).
check("r_parrelell2_matches_matlab",
      rel_err(Sp[0, 0], -zref / (rpad * (zref / rpad + 2))) <= 1e-13
      and rel_err(Sp[1, 0], 2 / (zref / rpad + 2)) <= 1e-13,
      "r_parrelell2 != shunt-resistor S-parameter transcription")
# Passive + reciprocal for both.
psv_s = float(np.max(np.abs(Ss[0, 0]) ** 2 + np.abs(Ss[1, 0]) ** 2))
psv_p = float(np.max(np.abs(Sp[0, 0]) ** 2 + np.abs(Sp[1, 0]) ** 2))
check("R_series2_r_parrelell2_passive_reciprocal",
      psv_s <= 1 + 1e-12 and psv_p <= 1 + 1e-12
      and rel_err(Ss[0, 1], Ss[1, 0]) <= 1e-15 and rel_err(Sp[0, 1], Sp[1, 0]) <= 1e-15,
      "series/shunt R network not passive or not reciprocal")

# ===========================================================================
# 3. get_TDR 4-port reflection kernel (_get_TDR__TDR_RL, py 10212-10224)
# ===========================================================================
Zin, Zout = 100.0, 100.0
# matched-through 4-port (s11=s22=0), Zin==Zout -> zero reflection.
rl_matched = sicopr._get_TDR__TDR_RL(Zin, Zout, 0.0, 1.0, 1.0, 0.0)
check("TDR_RL_matched_is_zero",
      abs(rl_matched) <= 1e-12,
      "matched through with Zin==Zout gave nonzero reflection: %g" % abs(rl_matched))
# Transcription check on a reflective network.
s11r, s12r, s21r, s22r = 0.3 + 0.1j, 0.05, 0.05, 0.2 - 0.05j
Zi, Zo = 100.0, 141.4
num = (Zi**2 * s11r + Zi**2 * s22r + Zo**2 * s11r + Zo**2 * s22r + Zi**2 - Zo**2
       + Zi * Zo * s11r * 2 - Zi * Zo * s22r * 2 + Zi**2 * s11r * s22r
       - Zi**2 * s12r * s21r - Zo**2 * s11r * s22r + Zo**2 * s12r * s21r)
den = (Zi * Zo * 2 + Zi**2 * s11r + Zi**2 * s22r - Zo**2 * s11r - Zo**2 * s22r
       + Zi**2 + Zo**2 + Zi**2 * s11r * s22r - Zi**2 * s12r * s21r
       + Zo**2 * s11r * s22r - Zo**2 * s12r * s21r
       - Zi * Zo * s11r * s22r * 2 + Zi * Zo * s12r * s21r * 2)
check("TDR_RL_matches_matlab_formula",
      rel_err(sicopr._get_TDR__TDR_RL(Zi, Zo, s11r, s12r, s21r, s22r), num / den) <= 1e-12,
      "4-port TDR_RL rational function diverges from ML line 7065")

# ===========================================================================
# 4. get_TDR s2p RL renorm: MATLAB `\` (left div) vs Python `/`  (EXPECTED FAIL)
# ===========================================================================
df = 1e8
faxis = np.arange(0, 40e9 + df / 2, df)
Nf = len(faxis)
tau_s = 0.5e-9
S11 = 0.2 * np.exp(-1j * 2 * np.pi * faxis * tau_s)
Params = np.zeros((Nf, 2, 2), dtype=complex)
Params[:, 0, 0] = S11
S = SimpleNamespace(Frequencies=faxis, Parameters=Params, Impedance=95.0, NumPorts=1)

ZT = 50.0
fb = 40e9
M = 32
param_tdr = SimpleNamespace(
    FLAG=SimpleNamespace(S2P=1), RL_sel=0, TR_TDR=6.7e-3, tfx=[0.0, 0.0],
    ui=1.0 / fb, sample_dt=1.0 / (fb * M), samples_per_ui=M, Tukey_Window=0.0,
    fb=fb, fb_BT_cutoff=1.0, N_bx=0, ndfe=0, levels=4, Grr=1, rho_x=0.0,
    beta_x=0.0, specBER=1e-4, Z_t=100.0)
OP_tdr = SimpleNamespace(
    TDR=True, PTDR=False, DISPLAY_WINDOW=False, cb_Guassian=1, N=60, T_k=1e-9,
    interp_sparam_mag='linear_trend_to_DC',
    interp_sparam_phase='extrap_cubic_to_dc_linear_to_inf',
    EC_PULSE_TOL=0.01, EC_REL_TOL=1e-2, EC_DIFF_TOL=1e-3, ENFORCE_CAUSALITY=0,
    impulse_response_truncation_threshold=1e-3, ZERO_PAD=False, cb_step=0,
    DEBUG=True)

res_s2p = sicopr.get_TDR(S, OP_tdr, param_tdr, ZT, 0,
                      _Bessel_Thomson_Filter_fn=sicopr.Bessel_Thomson_Filter,
                      _Butterworth_Filter_fn=sicopr.Butterworth_Filter,
                      _Tukey_Window_fn=sicopr.Tukey_Window,
                      _s21_to_impulse_DC_fn=sicopr.s21_to_impulse_DC,
                      _get_StepR_fn=sicopr.get_StepR,
                      _get_PulseR_fn=sicopr.get_PulseR,
                      _get_pdf_fn=sicopr.get_pdf_from_sampled_signal)

rho = (2 * ZT - S.Impedance) / (2 * ZT + S.Impedance)
RL_matlab = (S11 - rho) / (1 - rho * S11)                 # ML: interim cancels

# B06-D9 (get_TDR s2p RL used '/' where MATLAB 7080 uses '\' left division) was
# FIXED in the 8-defect correlation commit. A second check used to sit here asserting sicopr.py still
# produced the WRONG formula, interim^2/((s11-rho)*(1-rho*s11)); it inverted the
# moment the bug was fixed and had been failing ever since. Removed -- the check
# below is the one that carries meaning.
check("get_TDR_s2p_RL_matches_matlab",
      rel_err(res_s2p.RL, RL_matlab) <= 1e-9,
      "py s2p RL diverges from ML 7080 RL=(s11-rho)/(1-rho*s11). "
      "max|RL_py|=%.3g (reflection>1 is non-physical)" % np.max(np.abs(res_s2p.RL)))

# ===========================================================================
# 5. get_TDR s4p end-to-end: matched line -> avgZport ~ 2*ZT (physics)
# ===========================================================================
S11_4 = np.zeros(Nf, dtype=complex)                       # no reflection
S21_4 = np.exp(-1j * 2 * np.pi * faxis * tau_s)           # ideal delay
P4 = np.zeros((Nf, 2, 2), dtype=complex)
P4[:, 0, 0] = S11_4; P4[:, 1, 1] = S11_4
P4[:, 0, 1] = S21_4; P4[:, 1, 0] = S21_4
S4 = SimpleNamespace(Frequencies=faxis, Parameters=P4, Impedance=2 * ZT, NumPorts=2)
param4 = SimpleNamespace(**vars(param_tdr))
param4.FLAG = SimpleNamespace(S2P=0)

res_s4p = sicopr.get_TDR(S4, OP_tdr, param4, ZT, 0,
                      _Bessel_Thomson_Filter_fn=sicopr.Bessel_Thomson_Filter,
                      _Butterworth_Filter_fn=sicopr.Butterworth_Filter,
                      _Tukey_Window_fn=sicopr.Tukey_Window,
                      _s21_to_impulse_DC_fn=sicopr.s21_to_impulse_DC,
                      _get_StepR_fn=sicopr.get_StepR,
                      _get_PulseR_fn=sicopr.get_PulseR,
                      _get_pdf_fn=sicopr.get_pdf_from_sampled_signal)
check("get_TDR_s4p_matched_zero_reflection",
      np.max(np.abs(res_s4p.RL)) <= 1e-9,
      "matched s4p (Zin==Zout, s11=0) produced nonzero RL: %g" % np.max(np.abs(res_s4p.RL)))
check("get_TDR_s4p_avgZport_is_2ZT",
      abs(res_s4p.avgZport - 2 * ZT) <= 1e-3 * (2 * ZT),
      "matched line avgZport=%.3f expected ~%.3f" % (res_s4p.avgZport, 2 * ZT))

# ===========================================================================
# 6. TDR_ERL_Processing (ML 4503-4597 vs py 6632-6725)
# ===========================================================================
ch0 = SimpleNamespace(
    base='thru',
    TDR11=SimpleNamespace(ERL=15.0, avgZport=101.0),
    TDR22=SimpleNamespace(ERL=18.0, avgZport=99.0))
ch1 = SimpleNamespace(base='fext', TDR11=SimpleNamespace(ERL=30.0, avgZport=100.0),
                      TDR22=SimpleNamespace(ERL=31.0, avgZport=100.0))
chdata_e = [ch0, ch1]
oa = SimpleNamespace()
OP_e = SimpleNamespace(TDR=True, ERL=True, AUTO_TFX=False, ERL_ONLY=True,
                       BREAD_CRUMBS=False, TDR_W_TXPKG=False,
                       Report_Modal_ERL='disable', DISPLAY_WINDOW=False)
param_e = SimpleNamespace(FLAG=SimpleNamespace(S2P=False), Z_t=100.0)

oa, ERL, min_ERL = sicopr.TDR_ERL_Processing(oa, OP_e, 1, chdata_e, param_e)
check("TDR_ERL_min_and_array",
      min_ERL == 15.0 and ERL[0] == 15.0 and ERL[1] == 18.0,
      "min_ERL/ERL array wrong: min=%s ERL=%s" % (min_ERL, ERL))
check("TDR_ERL_Z11est_copied",
      oa.Z11est == 101.0 and oa.Z22est == 99.0,
      "Z11est/Z22est not copied from TDR structs")
# EXPECTED FAIL: ML str2csv({chdata(1).base}) uses ONLY the first base; py joins all.
xcheck("TDR_ERL_file_names_first_base_only",
      oa.file_names == '"thru"',
      "DIVERGENT (cosmetic, ERL_ONLY): py 6721-6722 joins ALL channel bases "
      "(%s) but ML 4592 str2csv({chdata(1).base}) uses only the first base" % oa.file_names)

finish()
