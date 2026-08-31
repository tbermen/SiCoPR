"""Audit batch B03: G3 impulse and spectrum handling (FFT prime suspects).

Functions under audit (MATLAB com_ieee8023_4p15p0.m -> sicopr.py):
  s21_to_impulse_DC   ML 11218-11288 -> py 16992-17075
  COM_FD_to_TD        ML 1206-1366   -> py 1346-1482

Spectrum/FFT traps checked (audit prompt section 3 item 3, section 4 item 5):
  - fout grid: 0:fmax/round(fmax/df):fmax  (endpoint inclusive, round trap).
  - Hermitian extension for a real ifft: DC and Nyquist forced real, conjugate
    mirror of the interior; exact structure and values.
  - ifft scaling (numpy and MATLAB both 1/N) and t_base = (0:L-1)/(df*L).
  - Truncation index (find(...,1,'last')) 1-based -> 0-based.
  - Alternating-Projections causality window indexing (start_ind, floor(L/2)).
  - COM_FD_to_TD: Gaussian Tx filter, pulse = filter(ones(1,M),1,ir) running
    sum, SCMR sampling phase/stride, amplitude scaling.

Oracles are transcribed from the cited MATLAB lines; 1-based indices converted
only at the array-access boundary. Interp is made a near-identity by putting the
input on the exact fout grid, so the FFT machinery is isolated from the B02
interp_Sparam divergences. FAIL rows document divergences. Run:
python tests/test_impulse_spectrum.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

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
# Common setup: input IL already on the exact fout grid so interp is identity.
# fmax = 1/(2*dt); df = 1e8; n_steps = round(fmax/df) = 400 -> fout = 0..40 GHz.
# ---------------------------------------------------------------------------
df = 1e8
n_steps = 400
sample_dt = 1.0 / (2.0 * n_steps * df)          # fmax = n_steps*df = 40 GHz
freq_array = np.arange(0, n_steps + 1) * df      # 0..40 GHz, exactly the fout grid
fmax = 1.0 / sample_dt / 2.0

tau = 0.3e-9
mag = np.exp(-3.0 * np.sqrt(freq_array / 40e9)) * (1 + 0.2 * np.cos(2 * np.pi * freq_array / 8e9))
IL = mag * np.exp(-1j * 2 * np.pi * freq_array * tau)   # decreasing phase -> causal-ish

OP = SimpleNamespace(
    interp_sparam_mag='linear_trend_to_DC',
    interp_sparam_phase='extrap_cubic_to_dc_linear_to_inf',
    ZERO_PAD=False, DEBUG=True,                  # DEBUG: anti-causal warns not raises
    EC_PULSE_TOL=0.01, EC_REL_TOL=1e-2, EC_DIFF_TOL=1e-3,
    ENFORCE_CAUSALITY=0, impulse_response_truncation_threshold=1e-12)
param = SimpleNamespace(fb=50e9, zero_pad_tukey_window_in_fb=0.0)


def matlab_core(IL_on_grid, freq_step, trunc_thresh):
    """Faithful transcription of s21_to_impulse_DC's ENFORCE_CAUSALITY=0 path
    (ML 11243-11285): Hermitian extension, real ifft, t_base, truncation."""
    ILc = np.asarray(IL_on_grid, dtype=complex).ravel()
    IL_sym = np.concatenate([
        [np.real(ILc[0])], ILc[1:-1], [np.real(ILc[-1])],
        np.conj(ILc[1:-1])[::-1]])
    ir = np.real(np.fft.ifft(IL_sym))            # ML 11247
    L = len(ir)
    t_base = np.arange(L) / (freq_step * L)      # ML 11249
    ir_peak = np.max(np.abs(ir))                 # ML 11281
    last = np.nonzero(np.abs(ir) > ir_peak * trunc_thresh)[0][-1]  # ML 11282 (1-based last)
    return ir[:last + 1], t_base[:last + 1], IL_sym, ir


# ===========================================================================
# 1. fout grid construction (ML 11230-11232)
# ===========================================================================
# Reproduce the grid the function builds and confirm endpoint inclusivity/step.
freq_step_chk = freq_array[2] - freq_array[1]
n_chk = max(1, round(fmax / freq_step_chk))
fout_chk = np.arange(0, n_chk + 1) * (fmax / n_chk)
check("s21_fout_grid_endpoint_and_step",
      len(fout_chk) == n_steps + 1 and abs(fout_chk[-1] - fmax) <= 1e-3
      and abs((fout_chk[1] - fout_chk[0]) - df) <= 1e-3,
      "fout grid length/endpoint/step wrong")

# ===========================================================================
# 2. Core impulse path (ENFORCE_CAUSALITY=0): Hermitian + ifft + truncation
# ===========================================================================
volt, tb, ccdB, trdB = sicopr.s21_to_impulse_DC(IL, freq_array, sample_dt, OP, param)
v_ml, tb_ml, IL_sym_ml, ir_full_ml = matlab_core(IL, freq_step_chk,
                                                 OP.impulse_response_truncation_threshold)

# Hermitian structure: length 2N, DC and Nyquist real, conjugate-symmetric.
check("s21_hermitian_length_2N",
      len(IL_sym_ml) == 2 * n_steps,
      "IL_symmetric length %d != 2N=%d" % (len(IL_sym_ml), 2 * n_steps))
check("s21_hermitian_dc_nyquist_real",
      abs(IL_sym_ml[0].imag) == 0 and abs(IL_sym_ml[n_steps].imag) == 0,
      "DC or Nyquist bin not real")
kk = np.arange(1, n_steps)
check("s21_hermitian_conjugate_symmetry",
      rel_err(IL_sym_ml[2 * n_steps - kk], np.conj(IL_sym_ml[kk])) <= 1e-13,
      "bin k and 2N-k are not conjugates")

# ifft output is real (imag part negligible) and matches oracle.
check("s21_impulse_matches_matlab_core",
      len(volt) == len(v_ml) and rel_err(volt, v_ml) <= 1e-11,
      "impulse response diverges from ML Hermitian-ifft transcription")
check("s21_t_base_matches_matlab",
      len(tb) == len(tb_ml) and rel_err(tb, tb_ml) <= 1e-13,
      "t_base mismatch")

# Parseval: sum(ir^2) == sum(|IL_sym|^2)/L  (numpy ifft convention).
L_full = len(ir_full_ml)
parseval_lhs = float(np.sum(ir_full_ml ** 2))
parseval_rhs = float(np.sum(np.abs(IL_sym_ml) ** 2)) / L_full
check("s21_parseval_energy_consistency",
      abs(parseval_lhs - parseval_rhs) <= 1e-10 * parseval_rhs,
      "Parseval mismatch: %g vs %g" % (parseval_lhs, parseval_rhs))

# Physics: causal delay channel -> peak near tau, negligible pre-cursor before t=0
# region. Peak sample index should be close to tau/dt.
dt_actual = tb_ml[1] - tb_ml[0]
peak_idx = int(np.argmax(np.abs(ir_full_ml)))
check("s21_peak_near_delay",
      abs(peak_idx - tau / dt_actual) <= 5,
      "impulse peak at index %d, expected ~%d" % (peak_idx, round(tau / dt_actual)))
# Pre-cursor energy before the leading edge is a small fraction of total.
pre = float(np.sum(ir_full_ml[:max(1, peak_idx - 40)] ** 2))
tot = float(np.sum(ir_full_ml ** 2))
check("s21_small_precursor_energy",
      pre / tot < 1e-3,
      "pre-cursor energy fraction %.3g too large for a causal channel" % (pre / tot))

# all-zero IL branch (ML 11233-11235): floor value is MATLAB eps (2.2e-16).
volt_z, _, _, _ = sicopr.s21_to_impulse_DC(np.zeros(51, dtype=complex),
                                        np.arange(51) * df, sample_dt, OP, param)
check("s21_all_zero_uses_matlab_eps",
      np.max(np.abs(volt_z)) <= 1e-15,
      "all-zero IL did not produce an eps-level (2.2e-16) response")

# ===========================================================================
# 3. Alternating-Projections causality window indexing (ML 11255-11261)
# ===========================================================================
# EXPECTED FAIL: py 17040 start_ind = candidates[0] (0-based) used in [:start_ind]
# zeros one FEWER leading sample than ML impulse_response(1:a(1)); and py 17045
# impulse_response[half:] omits the 0-based floor(L/2)-1 sample that ML zeros.

def matlab_apm(IL_on_grid, freq_step, op):
    """Faithful transcription of the full s21_to_impulse_DC APM (ML 11246-11285)
    with MATLAB-correct 1-based indexing, ENFORCE_CAUSALITY honoured."""
    ILc = np.asarray(IL_on_grid, dtype=complex).ravel()
    IL_sym = np.concatenate([
        [np.real(ILc[0])], ILc[1:-1], [np.real(ILc[-1])],
        np.conj(ILc[1:-1])[::-1]])
    ir = np.real(np.fft.ifft(IL_sym))
    L = len(ir)
    t_base = np.arange(L) / (freq_step * L)
    ir_orig = ir.copy()
    abs_ir = np.abs(ir)
    half = L // 2
    a = np.nonzero(abs_ir[:half] > np.max(abs_ir[:half]) * op.EC_PULSE_TOL)[0]
    start_ind = int(a[0]) + 1                     # ML a(1) is 1-based
    err = np.inf
    while not np.all(ir == 0):
        ir[:start_ind] = 0                        # ML impulse_response(1:start_ind)=0 (inclusive)
        ir[half - 1:] = 0                         # ML impulse_response(floor(L/2):end)=0 -> 0-based half-1
        IL_mod = np.abs(IL_sym) * np.exp(1j * np.angle(np.fft.fft(ir)))
        ir_mod = np.real(np.fft.ifft(IL_mod))
        delta = np.abs(ir - ir_mod)
        err_prev = err
        signed_peak = np.max(ir)                  # ML max(impulse_response) is signed
        err = np.max(delta) / signed_peak if signed_peak != 0 else 0.0
        if err < op.EC_REL_TOL or abs(err_prev - err) < op.EC_DIFF_TOL:
            break
        ir = ir_mod
    ir_norm = np.linalg.norm(ir)
    ccdB = 20 * np.log10(np.linalg.norm(ir - ir_orig) / ir_norm) if ir_norm > 0 else 0.0
    if not op.ENFORCE_CAUSALITY:
        ir = ir_orig
    ir_peak = np.max(np.abs(ir))
    last = np.nonzero(np.abs(ir) > ir_peak * op.impulse_response_truncation_threshold)[0][-1]
    return ir[:last + 1], t_base[:last + 1], ccdB

# Build an IL with a mild acausal component (magnitude ripple) so the APM has
# something to correct, and enforce causality.
OPc = SimpleNamespace(**vars(OP))
OPc.ENFORCE_CAUSALITY = 1
mag2 = np.exp(-2.0 * np.sqrt(freq_array / 40e9)) * (1 + 0.4 * np.cos(2 * np.pi * freq_array / 4e9))
IL2 = mag2 * np.exp(-1j * 2 * np.pi * freq_array * tau)

volt_c, _, ccdB_c = None, None, None
volt_c, _, ccdB_c, _ = sicopr.s21_to_impulse_DC(IL2, freq_array, sample_dt, OPc, param)
v_apm_ml, _, ccdB_ml = matlab_apm(IL2, freq_step_chk, OPc)

nlen = min(len(volt_c), len(v_apm_ml))
apm_gap = rel_err(volt_c[:nlen], v_apm_ml[:nlen]) if nlen > 0 else 1.0
cc_gap = abs(ccdB_c - ccdB_ml)
check("s21_causality_window_indexing_matches_matlab",
      len(volt_c) == len(v_apm_ml) and apm_gap <= 1e-9 and cc_gap <= 1e-6,
      "DIVERGENT: py 17040/17045 causality window off by one vs ML 11256/11261 "
      "(start_ind and floor(L/2) zeroing); voltage relerr %.3g, causality dB gap "
      "%.4f (py=%.3f, ML=%.3f dB)" % (apm_gap, cc_gap, ccdB_c, ccdB_ml))

# ENFORCE_CAUSALITY=0 default: the returned voltage IS the plain Hermitian-ifft
# (original_impulse_response), unaffected by the APM off-by-one. Confirm.
volt_off, _, _, _ = sicopr.s21_to_impulse_DC(IL2, freq_array, sample_dt, OP, param)
v_off_ml, _, _, _ = matlab_core(IL2, freq_step_chk, OP.impulse_response_truncation_threshold)
check("s21_default_path_unaffected_by_apm",
      len(volt_off) == len(v_off_ml) and rel_err(volt_off, v_off_ml) <= 1e-11,
      "default (ENFORCE_CAUSALITY=0) impulse response should equal plain ifft")

# ===========================================================================
# 4. fout grid round(): banker's vs MATLAB half-away (ML 11232, py 17009)
# ===========================================================================
# RESOLVED. This was recorded as a known divergence, and the record outlived the
# defect: the condition was `round(200.5) == 201`, computed with Python's builtin
# INSIDE THE TEST FILE. That can never be true whatever the engine does, so the
# xcheck reported "divergent" permanently -- including after the engine was fixed
# (B03-D7). It said nothing about sicopr at any point.
#
# The engine now uses its half-away helper at the fout-grid site. Assert that,
# through the engine, so this resolves if it regresses.
_mr = sicopr._s21_to_impulse_DC__mround
check("s21_grid_round_half_away_from_zero",
      _mr(200.5) == 201 and _mr(-200.5) == -201,
      "the fout grid must use MATLAB's half-away round; the helper gave "
      "%d for 200.5 (MATLAB 201, banker's 200)" % _mr(200.5))

import inspect as _inspect
_fout_src = [l.strip() for l in _inspect.getsource(sicopr.s21_to_impulse_DC).splitlines()
             if 'n_steps' in l and '=' in l]
check("s21_fout_grid_site_uses_the_helper",
      any('mround' in l for l in _fout_src),
      "the n_steps site in s21_to_impulse_DC must call the half-away helper, "
      "not builtin round(); found: %s" % (_fout_src[:1] or '<not found>'))

# ===========================================================================
# 5. COM_FD_to_TD driver (ML 1206-1366 vs py 1346-1482)
# ===========================================================================
M = 32
faxis2 = np.arange(0, 401) * 1e8                 # 0..40 GHz
dt2 = 1.0 / (2.0 * 40e9)                          # fmax = 40 GHz -> identity grid
tau2 = 0.25e-9
mag_thru = np.exp(-2.5 * np.sqrt(faxis2 / 40e9))
sdd21 = mag_thru * np.exp(-1j * 2 * np.pi * faxis2 * tau2)
scd21 = 5e-4 * (faxis2 / 40e9) * np.exp(-1j * 2 * np.pi * faxis2 * tau2)
sdc21 = 5e-4 * (faxis2 / 40e9) * np.exp(-1j * 2 * np.pi * faxis2 * tau2)

thru = SimpleNamespace(faxis=faxis2, sdd21=sdd21.copy(), sdd21_raw=sdd21.copy(),
                       sdd21_orig=sdd21.copy(), scd21_orig=scd21, sdc21_orig=sdc21,
                       A=0.75, type='THRU', base='thru')
chdata = [thru]

param2 = SimpleNamespace(fb=40e9, samples_per_ui=M, sample_dt=dt2,
                         number_of_s4p_files=1, package_testcase_i=1, sigma_X=0.30,
                         f1=0.05e9, f2=39e9, levels=4, P_peak=1e-5, ndfe=0,
                         zero_pad_tukey_window_in_fb=0.0)
ttr = 6.9e-3   # ns
OP2 = SimpleNamespace(
    Bessel_Thomson=False, Butterworth=False, transmitter_transition_time=ttr,
    RX_CALIBRATION=False, PSDRXCAL=False, DEBUG=False, DISPLAY_WINDOW=False,
    ENFORCE_CAUSALITY=0, interp_sparam_mag='linear_trend_to_DC',
    interp_sparam_phase='extrap_cubic_to_dc_linear_to_inf', ZERO_PAD=False,
    EC_PULSE_TOL=0.01, EC_REL_TOL=1e-2, EC_DIFF_TOL=1e-3,
    impulse_response_truncation_threshold=1e-3)

chdata = sicopr.COM_FD_to_TD(chdata, param2, OP2,
                          _s21_to_impulse_DC_fn=sicopr.s21_to_impulse_DC,
                          _Bessel_Thomson_Filter_fn=sicopr.Bessel_Thomson_Filter,
                          _Butterworth_Filter_fn=sicopr.Butterworth_Filter,
                          _get_cm_noise_fn=sicopr.get_cm_noise)

ch = chdata[0]
# Gaussian Tx filter H_t (ML 1217): exp(-(pi*f/1e9*ttr/1.6832)^2). Reconstruct and
# verify the *_filtered impulse came from sdd21_raw*H_bt*H_bw*H_t (BT/BW off -> 1).
H_t = np.exp(-(np.pi * faxis2 / 1e9 * ttr / 1.6832) ** 2)
v_filt_ml, _, _, _ = sicopr.s21_to_impulse_DC(sdd21 * H_t, faxis2, dt2, OP2, param2)
check("FDTD_H_filters_gaussian_tx_applied",
      len(ch.uneq_imp_response_raw_filtered) == len(v_filt_ml)
      and rel_err(ch.uneq_imp_response_raw_filtered, v_filt_ml) <= 1e-10,
      "raw_filtered impulse != s21(sdd21_raw*H_t)")

# pulse = filter(ones(1,M),1,ir) is a causal moving-sum (ML 1234).
imp_raw = ch.uneq_imp_response_raw
pulse_oracle = np.convolve(imp_raw, np.ones(M))[:len(imp_raw)]
check("FDTD_pulse_is_running_sum",
      rel_err(ch.uneq_pulse_response_raw, pulse_oracle) <= 1e-11,
      "uneq_pulse_response_raw != filter(ones(M),1,ir)")

# Amplitude scaling (ML 1267-1269): uneq_imp_response *= A when USE_channel_amplitude.
v_main_ml, _, _, _ = sicopr.s21_to_impulse_DC(sdd21, faxis2, dt2, OP2, param2)
check("FDTD_amplitude_scaling_applied",
      rel_err(ch.uneq_imp_response, v_main_ml * thru.A) <= 1e-10,
      "uneq_imp_response not scaled by channel amplitude A")

# SCMR P_signal sampling (ML 1314-1321): phase = mod(ipeak-1,M), stride M.
PR = ch.uneq_pulse_response_orig_filtered
ipeak = int(np.argmax(PR))
istart = ipeak % M
q = len(PR) // M
sampled = PR[istart:istart + q * M:M]
P_signal_ml = float(np.linalg.norm(sampled) ** 2)
check("FDTD_SCMR_P_signal_sampling_matches_matlab",
      abs(ch.P_signal - P_signal_ml) <= 1e-9 * max(P_signal_ml, 1e-30),
      "P_signal=%g expected %g" % (ch.P_signal, P_signal_ml))

# Physics: through pulse response is real, peaked, and positive main cursor.
check("FDTD_pulse_real_and_positive_peak",
      np.max(ch.uneq_pulse_response) > 0
      and np.max(np.abs(ch.uneq_pulse_response)) == np.max(ch.uneq_pulse_response),
      "through pulse main cursor is not a positive peak")

finish()
