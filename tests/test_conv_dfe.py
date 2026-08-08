"""Audit batch B04: G4 convolution and DFE.

Functions under audit (MATLAB com_ieee8023_4p15p0.m -> com.py):
  FFE          ML 2031-2047 -> py 2417-2428
  FFE_Fast     ML 2054-2067 -> py 2449-2456
  Fract_T_FFE  ML 2114-2123 -> py 2544-2549
  TD_CTLE      ML 4598-4613 -> py 6748-6764
  dfe_clipper  ML 5663-5675 -> py 8290-8316

Traps checked (audit prompt section 3 item 4, section 4 items 6/8/9):
  - MATLAB circshift([ishift,0]) vs np.roll direction and the (i-1-cmx) -> (i-cmx)
    1-based/0-based tap-shift index remap.
  - IIR direct-form filter(B,A,x) vs scipy.lfilter; np.poly vs MATLAB poly order.
  - Element-wise clip (logical-mask assignment) row/column orientation.
  - DFE taps stay within [bmin, bmax] (Tier 3 invariant).

Oracles are transcribed from the cited MATLAB lines; tap placement, cross-path
FFE/FFE_Fast agreement, and the CTLE DC-gain invariant are independent semantic
checks (not just transcription echoes). Run: python tests/test_conv_dfe.py
"""
import os
import sys

import numpy as np
from scipy.signal import lfilter

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, finish  # noqa: E402
import com  # noqa: E402


def rel_err(a, b):
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    denom = max(np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(a - b)) / denom)


# ===========================================================================
# 1. FFE (ML 2031-2047 vs py 2417-2428)
# ===========================================================================
spui = 4
cmx = 2                         # 2 precursor taps
C = np.array([-0.05, 0.12, 0.85, -0.15, 0.04])   # 5 taps, cursor at index cmx=2
N = 48
rng = np.random.default_rng(40404)
V = rng.standard_normal(N)

V0 = com.FFE(C, cmx, spui, V)


def ffe_oracle(C, cmx, spui, V):
    """Transcription of ML 2040-2047: sum_i circshift(V, (i-1-cmx)*spui)*C(i)."""
    V = np.asarray(V, dtype=float).ravel()
    out = np.zeros_like(V)
    for i in range(len(C)):                # 0-based i -> ishift=(i-cmx)*spui
        if C[i] != 0:
            out = np.roll(V, (i - cmx) * spui) * C[i] + out
    return out


check("FFE_matches_matlab_transcription",
      rel_err(V0, ffe_oracle(C, cmx, spui, V)) <= 1e-13,
      "FFE diverges from circshift-sum transcription")

# Independent semantics: FFE of a unit impulse places C[i] at ((i-cmx)*spui) mod N.
delta = np.zeros(N); delta[0] = 1.0
V0_d = com.FFE(C, cmx, spui, delta)
placed = np.zeros(N)
for i in range(len(C)):
    placed[((i - cmx) * spui) % N] += C[i]
check("FFE_tap_placement_on_impulse",
      rel_err(V0_d, placed) <= 1e-13,
      "FFE tap placement wrong: precursor taps must wrap to the tail")
# Cursor-only tap (C=[1] at cmx=0) returns the signal unchanged.
check("FFE_cursor_only_identity",
      rel_err(com.FFE(np.array([1.0]), 0, spui, V), V) <= 1e-15,
      "single cursor tap did not return the input unchanged")
# Direction: a postcursor tap (i>cmx) shifts the impulse to LATER samples.
V0_post = com.FFE(np.array([0.0, 1.0]), 0, spui, delta)  # tap index1, cmx0 -> +spui
check("FFE_postcursor_shifts_later",
      abs(V0_post[spui] - 1.0) <= 1e-15 and abs(V0_post[0]) <= 1e-15,
      "positive-lag tap did not move energy to +spui (roll direction wrong)")

# ===========================================================================
# 2. FFE_Fast (ML 2054-2067 vs py 2449-2456) + cross-path agreement with FFE
# ===========================================================================
# Build V_shift so column i = circshift(V, (i-cmx)*spui); then FFE_Fast == FFE.
V_shift = np.column_stack([np.roll(V, (i - cmx) * spui) for i in range(len(C))])
V0_fast = com.FFE_Fast(C, V_shift)
check("FFE_Fast_matches_column_weighted_sum",
      rel_err(V0_fast, sum(V_shift[:, i] * C[i] for i in range(len(C)))) <= 1e-13,
      "FFE_Fast != sum_i V_shift[:,i]*C[i]")
check("FFE_Fast_agrees_with_FFE",
      rel_err(V0_fast, V0) <= 1e-13,
      "FFE_Fast with pre-shifted columns disagrees with FFE (cross-path)")
# Zero-tap skip must not change the result (0*column contributes nothing).
C_z = C.copy(); C_z[1] = 0.0
check("FFE_Fast_zero_tap_skip",
      rel_err(com.FFE_Fast(C_z, V_shift),
              sum(V_shift[:, i] * C_z[i] for i in range(len(C_z)))) <= 1e-13,
      "zero-tap handling changed the sum")

# ===========================================================================
# 3. Fract_T_FFE (ML 2114-2123 vs py 2544-2549)
# ===========================================================================
skew = 2
Vf = com.Fract_T_FFE(V, skew)
check("Fract_T_FFE_matches_matlab",
      rel_err(Vf, (np.roll(V, skew) + V) / 2) <= 1e-14,
      "Fract_T_FFE != (circshift(V,skew)+V)/2")
# skew=0 is the identity (average of V with itself).
check("Fract_T_FFE_zero_skew_identity",
      rel_err(com.Fract_T_FFE(V, 0), V) <= 1e-15,
      "zero skew did not return the input")
# Averaging halves a DC level and preserves it (both terms equal).
Vdc = np.full(N, 0.7)
check("Fract_T_FFE_preserves_dc",
      rel_err(com.Fract_T_FFE(Vdc, skew), Vdc) <= 1e-15,
      "DC level not preserved by the 2-tap average")

# ===========================================================================
# 4. TD_CTLE (ML 4598-4613 vs py 6748-6764): bilinear IIR
# ===========================================================================
# np.poly convention matches MATLAB poly (monic, highest-degree first).
check("np_poly_matches_matlab_poly",
      np.allclose(np.poly([2.0, 3.0]), [1.0, -5.0, 6.0]),
      "np.poly root->coeff convention differs from MATLAB poly")

fb = 53.125e9
f_z, f_p1, f_p2 = 6e9, 12e9, 20e9
kacdc_dB = -6.0
oversampling = 4
ir_in = np.zeros(200); ir_in[0] = 1.0     # unit impulse -> filter impulse response

ir_out, p1c, p2c, zc = com.TD_CTLE(ir_in, fb, f_z, f_p1, f_p2, kacdc_dB, oversampling)

# Oracle B/A transcription (ML 4600-4613) and lfilter == MATLAB filter.
p1_ctle = -2 * np.pi * f_p1
p2_ctle = -2 * np.pi * f_p2
z_ctle = -2 * np.pi * f_z * 10 ** (kacdc_dB / 20)
k_ctle = -p2_ctle
fs = 2 * fb * oversampling
p2d = (1 + p2_ctle / fs) / (1 - p2_ctle / fs)
p1d = (1 + p1_ctle / fs) / (1 - p1_ctle / fs)
zd = (1 + z_ctle / fs) / (1 - z_ctle / fs)
kd = (fs - z_ctle) / ((fs - p1_ctle) * (fs - p2_ctle)) * f_p1 / f_z
B = k_ctle * kd * np.poly([zd, -1])
A = np.poly([p1d, p2d])
ir_ml = lfilter(B, A, ir_in)
check("TD_CTLE_matches_matlab_iir",
      rel_err(ir_out, ir_ml) <= 1e-12,
      "TD_CTLE lfilter output != MATLAB filter(B,A,ir) transcription")

# Analytic invariant: bilinear maps s=0 <-> z=1, so the discrete DC gain equals
# the analog CTLE DC gain, which for eq 93A-22 is exactly 10^(kacdc_dB/20).
dc_gain = float(np.sum(B) / np.sum(A))       # = B(z=1)/A(z=1)
expect_dc = 10 ** (kacdc_dB / 20)
check("TD_CTLE_dc_gain_is_kacdc",
      abs(dc_gain - expect_dc) <= 1e-9 * expect_dc,
      "CTLE DC gain %g != 10^(kacdc/20)=%g" % (dc_gain, expect_dc))
# Impulse-response sum equals the DC gain (filter steady-state identity).
check("TD_CTLE_impulse_sum_is_dc_gain",
      abs(np.sum(ir_out) - dc_gain) <= 1e-9 * abs(dc_gain),
      "sum(impulse_response) %g != DC gain %g" % (np.sum(ir_out), dc_gain))
# Stability: bilinear-mapped poles of an LHP analog filter are inside |z|<1.
check("TD_CTLE_stable_poles",
      abs(p1d) < 1.0 and abs(p2d) < 1.0,
      "discrete poles not inside the unit circle: |p1d|=%.4f |p2d|=%.4f" % (abs(p1d), abs(p2d)))
check("TD_CTLE_real_output",
      np.all(np.isreal(ir_out)) and np.all(np.isfinite(ir_out)),
      "CTLE impulse response is not real/finite")

# ===========================================================================
# 5. dfe_clipper (ML 5663-5675 vs py 8290-8316)
# ===========================================================================
inp = np.array([-2.0, -0.3, 0.0, 0.4, 1.5, 0.9])
hi = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.5])     # per-element upper bounds
lo = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -0.5])
out = com.dfe_clipper(inp, hi, lo)


def clip_oracle(inp, hi, lo):
    """Transcription of ML 5672-5674: clip to per-element [lo, hi], masks on
    the ORIGINAL input."""
    inp = np.asarray(inp, dtype=float)
    hi = np.asarray(hi, dtype=float)
    lo = np.asarray(lo, dtype=float)
    o = inp.copy()
    o[inp > hi] = hi[inp > hi]
    o[inp < lo] = lo[inp < lo]
    return o


check("dfe_clipper_matches_matlab",
      rel_err(out, clip_oracle(inp, hi, lo)) <= 1e-15,
      "dfe_clipper != element-wise clip transcription")
# Explicit expected values.
check("dfe_clipper_values",
      np.allclose(out, [-1.0, -0.3, 0.0, 0.4, 1.0, 0.5]),
      "clip produced %s" % out)
# Masks reference the ORIGINAL input, not the sequentially clipped output.
check("dfe_clipper_in_bounds",
      np.all(out <= hi + 1e-15) and np.all(out >= lo - 1e-15),
      "clip output escaped [lo, hi]")

# Tier 3: DFE taps constrained to [bmin*cursor, bmax*cursor].
cursor = 0.8
bmax = np.array([0.7, 0.3, 0.15, 0.05])
bmin = -bmax
ideal_cursors = np.array([0.9, -0.5, 0.05, 0.2]) * cursor   # some exceed bounds
eff = com.dfe_clipper(ideal_cursors, bmax * cursor, bmin * cursor)
check("dfe_clipper_taps_within_dfe_bounds",
      np.all(eff <= bmax * cursor + 1e-15) and np.all(eff >= bmin * cursor - 1e-15),
      "clipped DFE taps escaped [bmin*cursor, bmax*cursor]")
# A tap already inside the band is untouched.
check("dfe_clipper_passes_inband_unchanged",
      abs(eff[2] - ideal_cursors[2]) <= 1e-15,
      "in-band tap was modified")

finish()
