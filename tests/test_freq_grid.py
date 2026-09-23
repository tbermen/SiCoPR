"""Audit batch B02: G2 frequency grid construction.

Functions under audit (MATLAB com_ieee8023_4p15p0.m -> sicopr.py):
  FD_Processing    ML 1689-2030 -> py 2177-2394 (+ helpers 2160-2174)
  interp_Sparam    ML 8080-8295 -> py 11739-11971
  read_s4p_files   ML 10729-10909 -> py 16060-16203 (+ s2p inline 15971-16055)

Grid traps checked (audit prompt section 3 item 2, section 4 items 2/7):
  - MATLAB find(f>=x,1,'first') / find(f<=x,1,'last') vs 0-based searchsorted,
    including exact-match and boundary inclusivity.
  - delta_f = faxis(11)-faxis(10) (1-based) -> faxis[10]-faxis[9] (0-based).
  - interp1(...,'linear','extrap') EXTRAPOLATES; np.interp CLAMPS. Where the
    port pre-extends the grid this is harmless; where it does not it diverges.
  - MATLAB eps (2.2e-16) vs np.finfo(float).tiny (2.2e-308) as the magnitude floor.
  - Native Touchstone grid pass-through (no resampling in read_s4p_files).

Oracles are transcribed from the cited MATLAB lines; 1-based indices converted
only at the array-access boundary. FAIL rows document divergences and are the
ledger evidence. Run: python tests/test_freq_grid.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import tempfile
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
# 1. FD_Processing grid helpers (py 2165-2174 vs ML find semantics)
# ===========================================================================

fax = np.arange(0.0, 50e9 + 1.0, 1e8)   # 0..50 GHz, 100 MHz step, endpoint incl.
N = len(fax)

# ML index_f2: a=find(faxis>=f2,1,'first'); if empty -> length(faxis).
# 40 GHz lands exactly on a grid point (index 400 0-based). find(>=) returns it.
check("FD_find_ge_exact_on_grid",
      sicopr._FD_Processing__find_idx_ge(fax, 40e9) == 400,
      "find_idx_ge(40GHz) != 400")
# Between grid points: 40.05 GHz -> first strictly-greater point at 40.1 GHz (401).
check("FD_find_ge_between_points",
      sicopr._FD_Processing__find_idx_ge(fax, 40.05e9) == 401,
      "find_idx_ge(40.05GHz) != 401")
# No point >= f: ML resets to length(faxis) (last index). Python clamps to N-1.
check("FD_find_ge_none_clamps_last",
      sicopr._FD_Processing__find_idx_ge(fax, 99e9) == N - 1,
      "find_idx_ge(99GHz) != N-1")
# ML index_f1: b=find(faxis<=f1,1,'last'); exact match INCLUDED.
check("FD_find_le_exact_on_grid",
      sicopr._FD_Processing__find_idx_le(fax, 10e9) == 100,
      "find_idx_le(10GHz) != 100")
# Between points: 10.05 GHz -> last point <= is 10.0 GHz (100).
check("FD_find_le_between_points",
      sicopr._FD_Processing__find_idx_le(fax, 10.05e9) == 100,
      "find_idx_le(10.05GHz) != 100")
# No point <= f (f below faxis[0]): ML resets index_f1=1 -> 0-based 0.
check("FD_find_le_none_clamps_first",
      sicopr._FD_Processing__find_idx_le(fax, -1.0) == 0,
      "find_idx_le(-1) != 0")

# W (eq 93A-57) transcription: 1/fb * sinc(f/fb)^2 / (1+(f/ft)^4) / (1+(f/fr)^8).
fb = 50e9
ftr = 40e9
fr = 0.75 * fb
ftest = np.linspace(1e8, 60e9, 257)   # avoid f=0 and integer multiples of fb
W_ml = (1.0 / fb * (np.sin(np.pi * ftest / fb) / (np.pi * ftest / fb)) ** 2
        * 1.0 / (1 + (ftest / ftr) ** 4) * 1.0 / (1 + (ftest / fr) ** 8))
check("FD_W_matches_matlab_eq93A57",
      rel_err(sicopr._FD_Processing__W(ftest, ftr, fr, fb), W_ml) <= 1e-12,
      "W transcription mismatch")

# ===========================================================================
# 2. FD_Processing end-to-end on a synthetic THRU + FEXT (Tier 3/4)
# ===========================================================================

spui = 32
sample_dt = 1.0 / (fb * spui)
f1 = 0.05e9
f2 = 40e9
f2_ild = 40e9
df = float(fax[10] - fax[9])

# THRU: two-pole low-pass magnitude, mild ripple; linear phase (pure delay).
fp = 12e9
tau = 2e-9
mag_thru = 1.0 / np.abs(1 + 1j * fax / fp) ** 2 * (1 + 0.02 * np.cos(2 * np.pi * fax / 6e9))
sdd21f_thru = mag_thru * np.exp(-1j * 2 * np.pi * fax * tau)
# FEXT: high-pass-ish weak coupling.
sdd21f_fext = 0.02 * (fax / fb) / (1 + (fax / fb) ** 2) * np.exp(-1j * 2 * np.pi * fax * 1e-9)

thru = SimpleNamespace(type='THRU', faxis=fax, ftr=ftr, sdd21f=sdd21f_thru,
                       sdd21=sdd21f_thru.copy(), sdd21p=sdd21f_thru.copy(),
                       sdd21p_nodie=sdd21f_thru.copy(),
                       scd21_orig=1e-4 * np.ones(N), sdc21_orig=1e-4 * np.ones(N))
fext = SimpleNamespace(type='FEXT', faxis=fax, ftr=ftr, sdd21f=sdd21f_fext)
chdata = [thru, fext]

param = SimpleNamespace(number_of_s4p_files=2, package_testcase_i=1,
                        a_thru=[1.0], a_fext=[1.0], a_next=[1.0],
                        a_icn_fext=0.6, a_icn_next=0.6,
                        f1=f1, f2=f2, f2_ild=f2_ild, f_r=0.75, fb=fb,
                        samples_per_ui=spui, sample_dt=sample_dt, ui=1.0 / fb,
                        sigma_X=0.30, P_peak=1e-5)
OP = SimpleNamespace(WC_PORTZ=False, TDMODE=False, GET_FD=True,
                     INCLUDE_FILTER=False, DISPLAY_WINDOW=False,
                     COMPUTE_TDILN=False, COMPUTE_RILN=False,
                     include_pcb=False, DEBUG=False, pkg_len_select=[1])

out = SimpleNamespace()
chdata, out = sicopr.FD_Processing(chdata, out, param, OP, None, True,
                                _get_ILN_fn=sicopr.get_ILN)

# --- Oracle index selection + delta_f ---
idx_f1 = int(np.nonzero(fax >= 0)[0][0])          # find(<=f1,1,last): last<=0.05e9
idx_f1 = int(np.nonzero(fax <= f1)[0][-1])
idx_f2 = int(np.nonzero(fax >= f2)[0][0])
idx_f2_ild = int(np.nonzero(fax >= f2_ild)[0][0])
check("FD_delta_f_uses_idx10_minus_idx9",
      abs(thru.delta_f - df) <= 1e-6 and abs(df - 1e8) <= 1e-3,
      "delta_f=%g expected %g" % (thru.delta_f, df))

# --- Oracle P_signal (ML 1834-1835) ---
temp_angle = spui * sample_dt * np.pi * fax
temp_angle[0] = 1e-20
SINC = np.sin(temp_angle) / temp_angle
PWF = SINC ** 2 / (1 + (fax / ftr) ** 4) / (1 + (fax / fr) ** 8)
Il_dB = -20 * np.log10(np.abs(sdd21f_thru))
W_sl = sicopr._FD_Processing__W(fax[idx_f1:idx_f2 + 1], ftr, fr, fb)
P_signal_ml = 2 * df * np.sum(W_sl * 10 ** (-Il_dB[idx_f1:idx_f2 + 1] / 10))
check("FD_P_signal_matches_matlab",
      abs(out.P_signal_FD - P_signal_ml) <= 1e-10 * P_signal_ml,
      "P_signal=%g expected %g" % (out.P_signal_FD, P_signal_ml))
check("FD_P_signal_sigma_is_sqrt",
      abs(out.P_signal_sigma_FD - np.sqrt(P_signal_ml)) <= 1e-10 * np.sqrt(P_signal_ml),
      "P_signal_sigma mismatch")

# --- Oracle FOM_ILD (ML 1940) ---
ILD_magft, _ = sicopr.get_ILN(sdd21f_thru[idx_f1:idx_f2_ild + 1], fax[idx_f1:idx_f2_ild + 1])
FOM_ILD_ml = np.sqrt(df / fb * np.sum(PWF[idx_f1:idx_f2_ild + 1] * np.asarray(ILD_magft) ** 2))
check("FD_FOM_ILD_matches_matlab",
      abs(out.FOM_ILD - FOM_ILD_ml) <= 1e-9 * max(FOM_ILD_ml, 1e-12),
      "FOM_ILD=%g expected %g" % (out.FOM_ILD, FOM_ILD_ml))
check("FD_FOM_ILD_nonneg", out.FOM_ILD >= 0, "FOM_ILD negative")

# --- Oracle ICN / MDFEXT_ICN (ML 1970, 1986-1987) ---
MDFEXT = np.sqrt(np.abs(sdd21f_fext) ** 2)
MDFEXT_ICN_ml = np.sqrt(2 * df / fb * np.sum(
    param.a_icn_fext ** 2 * PWF[idx_f1:idx_f2 + 1] * np.abs(MDFEXT[idx_f1:idx_f2 + 1]) ** 2))
check("FD_MDFEXT_ICN_matches_matlab",
      abs(out.MDFEXT_ICN_92_47_mV / 1000 - MDFEXT_ICN_ml) <= 1e-9 * max(MDFEXT_ICN_ml, 1e-15),
      "MDFEXT_ICN mismatch")
PSXT = np.sqrt((np.abs(sdd21f_fext) * param.a_icn_fext) ** 2)
ICN_ml = np.sqrt(2 * df / fb * np.sum(PWF[idx_f1:idx_f2 + 1] * np.abs(PSXT[idx_f1:idx_f2 + 1]) ** 2))
check("FD_ICN_matches_matlab",
      abs(out.ICN_mV / 1000 - ICN_ml) <= 1e-9 * max(ICN_ml, 1e-15),
      "ICN mismatch")

# --- Nyquist-loss interpolation (ML 2025, linear interp1, within range) ---
fnq = 1.0 / (param.ui * 2.0)
vtf_ml = float(np.interp(fnq, fax, -20 * np.log10(np.abs(chdata[0].sdd21))))
check("FD_VTF_loss_at_Fnq_matches_matlab",
      abs(out.VTF_loss_dB_at_Fnq - vtf_ml) <= 1e-9 * max(abs(vtf_ml), 1e-9),
      "VTF loss mismatch")

# --- Index-window inclusive edge: poison the bin exactly at f2 must count ---
thru2 = SimpleNamespace(type='THRU', faxis=fax, ftr=ftr,
                        sdd21f=sdd21f_thru.copy(),
                        sdd21=sdd21f_thru.copy(), sdd21p=sdd21f_thru.copy(),
                        sdd21p_nodie=sdd21f_thru.copy(),
                        scd21_orig=1e-4 * np.ones(N), sdc21_orig=1e-4 * np.ones(N))
thru2.sdd21f[idx_f2] *= 0.5                # change IL exactly at f2
out2 = SimpleNamespace()
_, out2 = sicopr.FD_Processing([thru2], SimpleNamespace(), param, OP, None, True,
                            _get_ILN_fn=sicopr.get_ILN)
check("FD_f2_bin_included_in_P_signal",
      abs(out2.P_signal_FD - out.P_signal_FD) > 1e-12 * out.P_signal_FD,
      "bin at exactly f2 did not affect P_signal (should be inclusive)")

# ===========================================================================
# 3. interp_Sparam (ML 8080-8295 vs py 11739-11971)
# ===========================================================================

OPi = SimpleNamespace(DEBUG=True, ZERO_PAD=False)     # DEBUG=True: anti-causal warns not raises
pari = SimpleNamespace(fb=fb, zero_pad_tukey_window_in_fb=0.0)

# --- Default mag path 'linear_trend_to_DC' matches interp1-extrap in interior ---
# The port pre-extends fin_x with 0 and fout[-1], so np.interp never clamps here.
fin = np.arange(0.0, 20e9 + 1.0, 1e8)
tau2 = 1e-9
Sin_lin = (0.9 / (1 + 1j * fin / 8e9)) * np.exp(-1j * 2 * np.pi * fin * tau2)
fout_wide = np.arange(0.0, 30e9 + 1.0, 1e8)   # extends beyond fin[-1]=20 GHz
Sout = sicopr.interp_Sparam(Sin_lin, fin, fout_wide, 'linear_trend_to_DC',
                         'trend_and_shift_to_DC', OPi, pari)
# Oracle for the magnitude in the DATA band (fout <= fin[-1]): linear interp of |Sin|.
mag_in = np.abs(Sin_lin)
band = fout_wide <= fin[-1]
mag_oracle_band = np.interp(fout_wide[band], fin, mag_in)
check("interp_Sparam_linear_trend_mag_interior",
      rel_err(np.abs(Sout)[band], mag_oracle_band) <= 1e-9,
      "linear_trend_to_DC magnitude diverges inside the data band")

# --- FINDING B: base phase interp CLAMPS where MATLAB extrapolates (py 11855) ---
# Pure delay so unwrapped phase is exactly linear: MATLAB interp1(...'extrap')
# continues the line beyond fin[-1]. The port used np.interp, which HOLDS IT
# FLAT, until 51f30f2 converted every interp1 site in interp_Sparam to the
# file's _interp_extrap helper. Both checks below now pass.
Sin_del = np.exp(-1j * 2 * np.pi * fin * tau2)   # |S|=1, phase = -2*pi*f*tau2
Sout_del = sicopr.interp_Sparam(Sin_del, fin, fout_wide, 'old',
                             'extrap_cubic_to_dc_linear_to_inf', OPi, pari)
ph_py = np.unwrap(np.angle(Sout_del))
ph_oracle = -2 * np.pi * fout_wide * tau2       # MATLAB linear extrapolation
band_in = fout_wide <= fin[-1]
band_out = fout_wide > fin[-1]
# Interior must match to 1e-9 (linear interp of a line).
check("interp_Sparam_phase_interior_matches",
      np.max(np.abs(ph_py[band_in] - ph_oracle[band_in])) <= 1e-6,
      "interior phase diverges from linear")
# Extrapolation region: ML 8185 continues the end-segment slope.
ph_gap = float(np.max(np.abs(ph_py[band_out] - ph_oracle[band_out])))
check("interp_Sparam_phase_extrap_matches_matlab",
      ph_gap <= 1e-3,
      "phase beyond fin[-1] does not follow ML 8185's interp1 'extrap' linear "
      "ramp; max gap %.3f rad at 30 GHz" % ph_gap)
# There used to be a companion xcheck here, "interp_Sparam_phase_extrap_is_flat",
# asserting that the port held the phase FLAT. It was written with INVERTED
# polarity: an xcheck's condition is meant to be the same "agrees with MATLAB"
# condition a check takes, so that resolving the divergence makes it XPASS and
# fails the run until the ledger is updated. This one's condition was the
# DIVERGENCE itself, so once the port started extrapolating correctly it
# XFAILed -- reporting correct behaviour as an accepted divergence -- and it
# would have XPASSed only if the port regressed. Deleted rather than inverted,
# because the check above already asserts the agreement directly.

# --- FINDING A: magnitude floor uses tiny (2.2e-308) not MATLAB eps (2.2e-16) ---
fin_e = np.arange(0.0, 20e9 + 1.0, 1e8)
mag_e = 0.5 * np.ones(len(fin_e))
mag_e[50] = 1e-20                               # deep null well below MATLAB eps
Sin_e = mag_e * np.exp(-1j * 2 * np.pi * fin_e * tau2)
Sout_e = sicopr.interp_Sparam(Sin_e, fin_e, fin_e, 'old', 'zero_DC', OPi, pari)
mag_at_null = float(np.abs(Sout_e[50]))
matlab_eps = np.finfo(float).eps                # 2.220446e-16
check("interp_Sparam_mag_floor_matlab_eps",
      abs(mag_at_null - matlab_eps) <= 1e-3 * matlab_eps,
      "DIVERGENT: py 11756 sets eps=np.finfo.tiny (2.2e-308); MATLAB eps=2.2e-16 "
      "floors |S|. Port left null at %.3g, MATLAB would floor to %.3g"
      % (mag_at_null, matlab_eps))

# --- ZERO_PAD tail zeroing (ML 8292 -> py 11969) ---
OPz = SimpleNamespace(DEBUG=True, ZERO_PAD=True)
Sout_z = sicopr.interp_Sparam(Sin_del, fin, fout_wide, 'old', 'zero_DC', OPz, pari)
check("interp_Sparam_zero_pad_zeros_beyond_fin_end",
      np.all(np.abs(Sout_z[fout_wide > fin[-1]]) <= 1e-300 * 10),
      "ZERO_PAD did not zero the tail beyond fin[-1]")

# ===========================================================================
# 4. read_s4p_files native-grid pass-through (ML 10729-10909 vs py 16060)
# ===========================================================================

def write_s2p(path, fghz, s21_mag):
    """Write a minimal 2-port GHz/MA/50 Touchstone file."""
    with open(path, 'w') as fh:
        fh.write('! synthetic test s2p\n')
        fh.write('# GHz S MA R 50\n')
        for f, m in zip(fghz, s21_mag):
            # cols: f  |S11| ang11  |S21| ang21  |S12| ang12  |S22| ang22
            fh.write('%.6f 0.02 10 %.6f -%.3f 0.02 10 0.02 10\n'
                     % (f, m, 3.0 * f))


tmpd = tempfile.mkdtemp(prefix='b02_s2p_')
fghz = np.round(np.arange(0.0, 20.0 + 1e-9, 0.1), 6)   # 0..20 GHz, 0.1 GHz step
s21_mag = 1.0 / (1.0 + (fghz / 8.0) ** 2)
s2p_path = os.path.join(tmpd, 'thru.s2p')
write_s2p(s2p_path, fghz, s21_mag)

param_r = SimpleNamespace(Z0=50.0, flim=float('inf'), fb=50e9,
                          max_start_freq=0.15e9, snpPortsOrder=[1, 2],
                          package_testcase_i=1, FLAG=SimpleNamespace(S2P=True))
OP_r = SimpleNamespace(DISPLAY_WINDOW=False, ZERO_PAD=False,
                       INC_PACKAGE=0, RX_CALIBRATION=0, include_pcb=0)
ch_r = SimpleNamespace(filename=s2p_path, ext='.s2p', type='THRU')
chdata_r, _, _, _ = sicopr.read_s4p_files(param_r, OP_r, [ch_r])

got = np.asarray(chdata_r[0].faxis, dtype=float).ravel()
expect_hz = fghz * 1e9
check("read_s4p_grid_length_native",
      len(got) == len(expect_hz),
      "faxis length %d != file %d (resampled?)" % (len(got), len(expect_hz)))
check("read_s4p_grid_values_native_hz",
      len(got) == len(expect_hz) and np.max(np.abs(got - expect_hz)) <= 1.0,
      "faxis values differ from native GHz->Hz grid by > 1 Hz")
check("read_s4p_grid_endpoints",
      abs(got[0] - 0.0) <= 1.0 and abs(got[-1] - 20e9) <= 1.0,
      "grid endpoints wrong: [%g, %g]" % (got[0], got[-1]))
step = np.diff(got)
check("read_s4p_grid_uniform_step",
      np.max(step) - np.min(step) <= 1.0 and abs(np.mean(step) - 1e8) <= 1.0,
      "grid step not uniform 100 MHz")

# Crosstalk axis-consistency validation (ML 10871-10879): a different-length
# second file must raise.
fghz2 = np.round(np.arange(0.0, 20.0 + 1e-9, 0.2), 6)   # coarser -> different length
s2p_path2 = os.path.join(tmpd, 'fext.s2p')
write_s2p(s2p_path2, fghz2, 0.01 / (1.0 + (fghz2 / 8.0) ** 2))
ch_a = SimpleNamespace(filename=s2p_path, ext='.s2p', type='THRU')
ch_b = SimpleNamespace(filename=s2p_path2, ext='.s2p', type='FEXT')
try:
    sicopr.read_s4p_files(SimpleNamespace(**vars(param_r)), OP_r, [ch_a, ch_b])
    raised_axis = False
except Exception:
    raised_axis = True
check("read_s4p_crosstalk_axis_mismatch_raises", raised_axis,
      "different-length crosstalk axis did not raise (ML 10871-10872)")

finish()
