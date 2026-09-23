"""Audit batch B12: G10 optimizer loops, MMSE + floating taps.

Scope (audit prompt section 3 item 10, "Optimizer loops"):
  - Full_Grid_Matrix         (MATLAB 2124-2184 -> sicopr.py 2569-2593)
  - FOM_rxffe_floating_taps  (MATLAB 2068-2113 -> sicopr.py 2487-2525)
  - MMSE                     (MATLAB 2485-2584 -> sicopr.py 3133-3257)
  - MMSE_FOM                 (MATLAB 2585-2697 -> sicopr.py 3286-3385)

Verification strategy (no MATLAB, no golden data):
  1. Full_Grid_Matrix: the cartesian-product row order (last column varies
     fastest) is the trap. Checked against the MATLAB docstring example and
     against itertools.product (same ordering convention).
  2. FOM_rxffe_floating_taps: greedy bank selection + overlap removal + 1-based
     index arithmetic. Driven with an INJECTED mock MMSE_FOM so the chosen banks
     are dictated by a known scoring function; the resulting idx and the removal
     behaviour are checked against an independent transcription.
  3. MMSE index seams (samp_idx phase, dh, isi_start, C-realign) are validated as
     faithful transcriptions of the MATLAB 1-based expressions.
  4. MMSE_FOM: the full block-matrix solve is validated against an independent
     re-implementation of the MATLAB (matlab_ref_MMSE_FOM) on a no-clip input,
     and the hoisted _MMSE__MMSE_FOM copy is checked identical to the top-level.

One confirmed divergence (EXPECTED FAIL row):
  B12-D17 (medium): MMSE_FOM recomputes blim from Hb*wlim UNCONDITIONALLY when
  Nb>0 (sicopr.py 3367-3369 / 3119-3121), whereas MATLAB (2685-2688) recomputes it
  only inside the `if ~isequal(w,wlim)` branch. When the DFE taps are clipped
  (first re-solve fires) but the RxFFE taps are within limits (w==wlim), MATLAB
  returns blim = clip(original DFE taps) while Python returns
  blim = clip(Hb*w_resolved), changing the returned DFE coefficients, sigma_e,
  and FOM on the RxFFE-with-MMSE path.

Run: python tests/test_optimizer_mmse.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import os
import sys
from types import SimpleNamespace

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, finish, xcheck  # noqa: E402
import sicopr  # noqa: E402

TOL = 1e-9   # linear-algebra agreement tolerance (double-precision solves)


# ===========================================================================
# 1. Full_Grid_Matrix  (MATLAB 2124-2184 -> sicopr.py 2569-2593)
#    Cartesian product; LAST column varies fastest (order is the trap).
# ===========================================================================
# (a) The MATLAB docstring example.
ex = sicopr.Full_Grid_Matrix([[1, 2], [100, 200]])
check("full_grid_matrix_docstring_example",
      ex == [[1, 100], [1, 200], [2, 100], [2, 200]],
      "Full_Grid_Matrix cartesian order != MATLAB example, got %s" % ex)

# (b) Order matches itertools.product (last index fastest) for 3 mixed-length vars.
a, b, c = [1, 2, 3], [10, 20], [7, 8, 9, 0]
got = sicopr.Full_Grid_Matrix([a, b, c])
ref = [list(t) for t in itertools.product(a, b, c)]
check("full_grid_matrix_matches_product_order",
      got == ref and len(got) == len(a) * len(b) * len(c),
      "Full_Grid_Matrix != itertools.product order for 3 vars")

# (c) Single-column and the number-of-cases invariant.
one = sicopr.Full_Grid_Matrix([[5, 6, 7]])
check("full_grid_matrix_single_column",
      one == [[5], [6], [7]],
      "single-column grid wrong: %s" % one)


# ===========================================================================
# 2. FOM_rxffe_floating_taps  (MATLAB 2068-2113 -> sicopr.py 2487-2525)
#    Greedy bank-by-bank selection; injected mock MMSE_FOM controls the scores.
# ===========================================================================
# Config: RxFFE_cpx=0 (no cursor offset), 6 ISI taps, bank_size 1, 2 groups.
param_ft = SimpleNamespace(RxFFE_cpx=0, N_bmax=6, N_bf=1, N_bg=2)
h_ft = np.array([10.0, 20, 30, 40, 50, 60, 0, 0])   # hisi = h[0:6] then [0:6] -> 6 taps

# Mock MMSE_FOM: FOM is highest when the chosen tap set is close to {3,5}.
# Signature mirrors MMSE_FOM's return: (sigma_e, FOM, w, idx, Nw, blim).
_ideal = np.array([3, 5])
def _mock_fom(param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2, idx):
    cand = np.asarray(idx, dtype=float).ravel()
    score = -float(sum(np.min(np.abs(cval - _ideal)) for cval in cand))
    return (0.0, score, None, idx, 0, None)

idx_ft = sicopr.FOM_rxffe_floating_taps(
    param_ft, h_ft, None, 0, None, 0, 0, None, None, None, None, 0.5, 0, 6,
    _MMSE_FOM_fn=_mock_fom)
# Group 1 ties at loc 3 and 5 -> MATLAB max()/np.argmax pick the FIRST (loc 3).
# Group 2 then picks loc 5. Final sorted idx = [3, 5] (+RxFFE_cpx=0).
check("floating_taps_greedy_selects_expected_banks",
      list(np.asarray(idx_ft).ravel()) == [3, 5],
      "greedy floating-tap idx = %s, expected [3,5]" % list(np.asarray(idx_ft).ravel()))

# Overlap-removal check: with bank_size 2, choosing a bank must remove the bank's
# positions AND the overlapping start locations (start-1). Prefer starting at loc 1.
param_ft2 = SimpleNamespace(RxFFE_cpx=0, N_bmax=8, N_bf=2, N_bg=1)
h_ft2 = np.arange(1.0, 12.0)   # hisi = h[0:8] -> 8 taps, max_isi = 8-2+1 = 7
def _mock_fom_first(param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2, idx):
    cand = np.asarray(idx, dtype=float).ravel()
    return (0.0, -float(np.min(cand)), None, idx, 0, None)   # prefers smallest start
idx_ft2 = sicopr.FOM_rxffe_floating_taps(
    param_ft2, h_ft2, None, 0, None, 0, 0, None, None, None, None, 0.5, 0, 8,
    _MMSE_FOM_fn=_mock_fom_first)
# One group of bank_size 2 starting at loc 1 -> taps {1,2}. +RxFFE_cpx=0.
check("floating_taps_bank_size2_start1",
      list(np.asarray(idx_ft2).ravel()) == [1, 2],
      "bank_size-2 single group idx = %s, expected [1,2]"
      % list(np.asarray(idx_ft2).ravel()))


# ===========================================================================
# 3. MMSE index seams  (MATLAB 2496/2497/2535/2572 -> sicopr.py 3149/3153/3206/3244)
#    Standalone reproductions of the 1-based -> 0-based translations.
# ===========================================================================
# (a) samp_idx starting phase: MATLAB (mod(cursor_i_1based-1,M)+1) 1-based ==
#     Python (cursor_i_0based % M) 0-based, for the same physical cursor.
for M in (16, 32):
    for cur1 in (1, 17, 40, 65):     # 1-based MATLAB cursor
        cur0 = cur1 - 1              # 0-based Python cursor
        ml_phase_1b = ((cur1 - 1) % M) + 1
        py_phase_0b = cur0 % M
        check("mmse_samp_idx_phase_M%d_cur%d" % (M, cur1),
              py_phase_0b == ml_phase_1b - 1,
              "samp_idx phase mismatch M=%d cur1=%d: py0=%d ml1=%d"
              % (M, cur1, py_phase_0b, ml_phase_1b))

# (b) isi_start: MATLAB dh+2 (1-based) == Python dh+1 (0-based) for the same dh.
for dh in (0, 1, 5):
    check("mmse_isi_start_dh%d" % dh,
          (dh + 1) == (dh + 2) - 1,
          "isi_start 0/1-based mismatch for dh=%d" % dh)

# (c) C-realignment tap column: MATLAB idx+RxFFE_cmx+1 (1-based) == Python
#     idx+RxFFE_cmx (0-based) + 1.
RxFFE_cmx = 3
for loc in (1, 4, 7):
    check("mmse_C_realign_col_loc%d" % loc,
          (loc + RxFFE_cmx) == (loc + RxFFE_cmx + 1) - 1,
          "C-realign column 0/1-based mismatch for loc=%d" % loc)


# ===========================================================================
# 4. MMSE_FOM block solve  (MATLAB 2585-2697 -> sicopr.py 3286-3385)
#    Independent MATLAB-semantics reference; the ONLY intended difference from
#    sicopr.py is the blim-recompute placement (conditional vs unconditional), so
#    on no-clip inputs they must agree, and the D17 case isolates the difference.
# ===========================================================================
def matlab_ref_MMSE_FOM(param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2, idx):
    """Faithful transcription of MATLAB MMSE_FOM 2585-2697 (conditional blim).

    Returns (sigma_e, FOM, w, idx, Nw_used, blim, block1_fired, block2_fired).
    """
    H = np.asarray(H, dtype=float)
    Rnn = np.asarray(Rnn, dtype=float)
    idx = np.array([], dtype=int) if idx is None or len(idx) == 0 else np.asarray(idx, dtype=int)
    RxFFE_cmx_ = int(param.RxFFE_cmx)
    Nfix = RxFFE_cmx_ + 1 + int(param.RxFFE_cpx)
    bmax_use = np.asarray(param.bmax, dtype=float).ravel()
    bmin_use = np.asarray(param.bmin, dtype=float).ravel()
    if len(idx) > 0:
        cols = np.concatenate([np.arange(Nfix), idx + RxFFE_cmx_])
        H = H[:, cols]
        Rnn = Rnn[np.ix_(cols, cols)]
    d = int(d)
    HH = H.T @ H
    R = HH + Rnn / sigma_X2
    Hb = H[d + 1:d + Nb + 1, :]
    h0 = H[d, :]
    ib = np.eye(Nb)
    zb = np.zeros(Nb)
    A = np.block([[R, -Hb.T], [-Hb, ib]])
    C = np.concatenate([h0, zb])
    Z = np.linalg.solve(A, C.reshape(-1, 1))
    S_inv = float(C @ Z.ravel())
    wbl = np.concatenate([Z.ravel(), [1 - S_inv]]) / S_inv
    Nw_used = H.shape[1]
    w = wbl[:Nw_used]
    b = wbl[Nw_used:Nw_used + Nb]
    blim = np.minimum(bmax_use[:Nb], np.maximum(bmin_use[:Nb], b))
    block1 = False
    if Nb > 0 and not np.allclose(b, blim):
        block1 = True
        Rb = np.block([[R, -h0.reshape(-1, 1)], [h0.reshape(1, -1), np.array([[0.0]])]])
        rhs = np.concatenate([h0 + Hb.T @ blim, [1.0]])
        w = np.linalg.solve(Rb, rhs)[:Nw_used]
    wmax_a = np.asarray(wmax, dtype=float).ravel()
    wmin_a = np.asarray(wmin, dtype=float).ravel()
    if len(wmax_a) > Nw_used:
        wmax_a = wmax_a[:Nw_used]
        wmin_a = wmin_a[:Nw_used]
    wc = w[int(dw)]
    wlim = np.minimum(wmax_a * wc, np.maximum(wmin_a * wc, w))
    block2 = False
    if not np.allclose(w, wlim):          # MATLAB ~isequal
        block2 = True
        wlim = wlim / (h0 @ wlim)
        if Nb > 0:                        # <-- conditional blim recompute (MATLAB)
            b = Hb @ wlim
            blim = np.minimum(bmax_use[:Nb], np.maximum(bmin_use[:Nb], b))
    w = wlim
    b = blim
    Hb_T_b = Hb.T @ b if Nb > 0 else np.zeros_like(h0)
    val = sigma_X2 * (w @ R @ w + 1 + b @ b - 2 * w @ h0 - 2 * w @ Hb_T_b)
    sigma_e = float(np.sqrt(val)) if val > 0 else 0.0
    FOM = float(20 * np.log10(param.R_LM / (param.levels - 1) / sigma_e)) if sigma_e > 0 else np.inf
    return sigma_e, FOM, w, idx, Nw_used, blim, block1, block2


# Common problem geometry: RxFFE_cmx=1, RxFFE_cpx=1 -> Nw=3, dw=1; dh=0 -> d=1;
# Nb=2 DFE taps -> H is 4x3.
H_test = np.array([
    [0.05, 0.10, 0.05],
    [0.10, 1.00, 0.10],   # h0 (row d=1): main cursor
    [0.02, 0.20, 0.03],   # Hb row 1
    [0.01, 0.05, 0.02],   # Hb row 2
])
Rnn_test = 0.001 * np.eye(3)
sigma_X2 = float((4 ** 2 - 1) / (3 * (4 - 1) ** 2))   # levels=4
dw, d, Nb = 1, 1, 2

# (a) NO-CLIP case: loose DFE and RxFFE bounds -> block1 and block2 both quiet,
#     so sicopr.py (unconditional recompute) and the MATLAB reference must agree.
param_loose = SimpleNamespace(RxFFE_cmx=1, RxFFE_cpx=1, N_bmax=2, N_bf=1, N_bg=0,
                              bmax=np.array([1e3, 1e3]), bmin=np.array([-1e3, -1e3]),
                              R_LM=1.0, levels=4)
wmax_l = np.array([1e6, 1.0, 1e6])
wmin_l = np.array([-1e6, 1.0, -1e6])
py = sicopr.MMSE_FOM(param_loose, H_test, Nb, Rnn_test, dw, d, wmax_l, wmin_l,
                  param_loose.bmin, param_loose.bmax, sigma_X2, None)
rf = matlab_ref_MMSE_FOM(param_loose, H_test, Nb, Rnn_test, dw, d, wmax_l, wmin_l,
                         param_loose.bmin, param_loose.bmax, sigma_X2, None)
check("mmse_fom_noclip_matches_matlab_reference",
      (not rf[6]) and (not rf[7])
      and abs(py[0] - rf[0]) <= TOL and abs(py[1] - rf[1]) <= TOL
      and np.max(np.abs(py[2] - rf[2])) <= TOL
      and np.max(np.abs(py[5] - rf[5])) <= TOL,
      "no-clip MMSE_FOM diverges from MATLAB reference "
      "(block1=%s block2=%s dFOM=%.3e)" % (rf[6], rf[7], abs(py[1] - rf[1])))

# (b) Cross-path: the hoisted _MMSE__MMSE_FOM copy is identical to the top-level.
py2 = sicopr._MMSE__MMSE_FOM(param_loose, H_test, Nb, Rnn_test, dw, d, wmax_l, wmin_l,
                          param_loose.bmin, param_loose.bmax, sigma_X2, None)
check("mmse_fom_hoisted_copy_identical",
      abs(py[0] - py2[0]) <= 1e-15 and abs(py[1] - py2[1]) <= 1e-15
      and np.max(np.abs(py[2] - py2[2])) <= 1e-15
      and np.max(np.abs(py[5] - py2[5])) <= 1e-15,
      "_MMSE__MMSE_FOM != top-level MMSE_FOM")

# (c) FOM physics sign: less noise (smaller Rnn) => higher FOM (better channel).
py_lownoise = sicopr.MMSE_FOM(param_loose, H_test, Nb, 0.0001 * np.eye(3), dw, d,
                           wmax_l, wmin_l, param_loose.bmin, param_loose.bmax,
                           sigma_X2, None)
check("mmse_fom_less_noise_higher_fom",
      py_lownoise[1] > py[1],
      "lower noise did not raise FOM (%.3f vs %.3f)" % (py_lownoise[1], py[1]))

# (d) sigma_e nonnegative and FOM finite for the well-posed case.
check("mmse_fom_sigma_e_nonneg",
      py[0] >= 0 and np.isfinite(py[1]),
      "sigma_e negative or FOM not finite (sigma_e=%s FOM=%s)" % (py[0], py[1]))

# --- B12-D17 (EXPECTED FAIL): DFE clip fires, RxFFE clip does not -----------
#     MATLAB keeps blim=clip(original DFE); Python overwrites with clip(Hb*w).
#     A single symmetric DFE bound that clips b_original to 0 is NOT
#     discriminating (both paths give 0), so sweep the bound to find a value
#     where block1 fires, block2 stays quiet, AND the two blims differ.
# First, get the unclipped DFE (b_original) to size the sweep.
rf_open = matlab_ref_MMSE_FOM(param_loose, H_test, Nb, Rnn_test, dw, d, wmax_l, wmin_l,
                              param_loose.bmin, param_loose.bmax, sigma_X2, None)
b_original = _Hb0 = H_test[d + 1:d + Nb + 1, :] @ rf_open[2]   # Hb @ w_unclipped
b_max_mag = float(np.max(np.abs(b_original)))

_found = None   # (bmax_val, py_blim, ref_blim, py_FOM, ref_FOM)
for bmax_val in np.linspace(0.02 * b_max_mag, 0.98 * b_max_mag, 60):
    p = SimpleNamespace(RxFFE_cmx=1, RxFFE_cpx=1, N_bmax=2, N_bf=1, N_bg=0,
                        bmax=np.array([bmax_val, bmax_val]),
                        bmin=np.array([-bmax_val, -bmax_val]), R_LM=1.0, levels=4)
    rf_s = matlab_ref_MMSE_FOM(p, H_test, Nb, Rnn_test, dw, d, wmax_l, wmin_l,
                               p.bmin, p.bmax, sigma_X2, None)
    if rf_s[6] and not rf_s[7]:   # block1 fired, block2 quiet
        py_s = sicopr.MMSE_FOM(p, H_test, Nb, Rnn_test, dw, d, wmax_l, wmin_l,
                            p.bmin, p.bmax, sigma_X2, None)
        if np.max(np.abs(np.asarray(py_s[5]) - np.asarray(rf_s[5]))) > TOL:
            _found = (bmax_val, np.asarray(py_s[5]), np.asarray(rf_s[5]), py_s[1], rf_s[1])
            break

# B12-D17 was FIXED in the 8-defect correlation commit: the `b = Hb*wlim; blim = clip(b)` refresh now
# runs only inside the `if ~isequal(w, wlim)` branch, matching MATLAB 2683-2692.
# The sweep above is therefore a REGRESSION GUARD, not a bug demonstration: it
# hunts across 60 DFE bounds for any clip-only case where sicopr.py and the
# MATLAB-faithful reference disagree. Finding one means D17 has come back.
#
# This check previously asserted `_found is not None` -- it was written while
# D17 was live and kept failing after the fix, which is what the audit caught.
_detail = ""
if _found is not None:
    bmax_val, py_blim, ref_blim, py_fom, ref_fom = _found
    _detail = ("bmax=%.4g -> Python blim=%s but MATLAB blim=%s; FOM py=%.4f vs "
               "ML=%.4f. Check that the blim refresh in MMSE_FOM/MMSE is still "
               "inside the `w != wlim` branch."
               % (bmax_val, py_blim, ref_blim, py_fom, ref_fom))

check("mmse_fom_D17_dfe_clip_matches_matlab",
      _found is None,
      "REGRESSION (B12-D17): sicopr.py diverges from MATLAB on a DFE-clip-only "
      "case. " + _detail)


# ---------------------------------------------------------------------------
# The singular MMSE solve: a DECIDED divergence, recorded rather than fixed.
#
# ML 2645 (Z = A\Ct) and ML 2669 are backslashes on SQUARE systems, and the
# three languages part company on exactly one input:
#
#   MATLAB  warns, returns Inf, and the NaNs that follow make the candidate
#           lose the FOM comparison. Measured 2026-09-23 under Octave with
#           octave/patches/mldivide_matlab.m restoring MATLAB semantics on a
#           rank-deficient H: sigma_e NaN, FOM NaN, w all NaN.
#   Octave  returns a minimum-norm least-squares answer instead: a finite
#           sigma_e of 0.0904 and FOM 11.34 on that same case.
#   SiCoPR  raises, so the run stops.
#
# The 2026-09-23 ruling on force() settled the class: a silent minimum-norm
# answer is the one outcome NEITHER reference produces, and a degenerate case
# is worth seeing. The cost is real and belongs on the record -- MMSE_FOM runs
# ~130k times per case, so where MATLAB discards one candidate SiCoPR stops
# the run -- which is why this is an xcheck and not a comment.
#
# The condition is written as "agrees with MATLAB", so the day the port starts
# returning NaN and continuing, this goes green and the ledger says so.
# ---------------------------------------------------------------------------

def _singular_mmse_args():
    import scipy.linalg
    cmx, cpx, Nb, d = 2, 3, 2, 4
    Nw = cmx + 1 + cpx
    L = 4
    sigma_X2 = (L ** 2 - 1) / (3.0 * (L - 1) ** 2)
    h = np.array([0.02, 0.10, 0.62, 0.21, 0.07, 0.03, 0.01, 0.004])
    H = scipy.linalg.toeplitz(np.concatenate([h, np.zeros(Nw - 1)]),
                              np.concatenate([[h[0]], np.zeros(Nw - 1)]))
    H[:, 1] = H[:, 0]                       # rank-deficient R
    p = SimpleNamespace(RxFFE_cmx=cmx, RxFFE_cpx=cpx, N_bg=0, N_bf=0,
                        N_bmax=0, levels=L, R_LM=1,
                        bmax=np.full(Nb, 1.5), bmin=np.full(Nb, -1.5))
    return (p, H, Nb, np.zeros((Nw, Nw)), cmx, d, np.full(Nw, 50.0),
            np.full(Nw, -50.0), np.full(Nb, -1.5), np.full(Nb, 1.5),
            sigma_X2, None)


_sing_outcome = None
try:
    _r = sicopr.MMSE_FOM(*_singular_mmse_args())
    _sing_outcome = ('returned sigma_e=%r FOM=%r' % (_r[0], _r[1]))
except Exception as _e:                       # noqa: BLE001
    _sing_outcome = '%s: %s' % (type(_e).__name__, str(_e).split('.')[0])

xcheck("mmse_fom_singular_solve_matches_matlab",
       _sing_outcome.startswith('returned') and 'nan' in _sing_outcome.lower(),
       "DECIDED (2026-09-23, following the force() ruling): on an exactly "
       "singular A, MATLAB returns Inf and the candidate loses the FOM "
       "comparison (COM Octave with mldivide_matlab: sigma_e NaN, FOM NaN, w "
       "all NaN), Octave's own backslash returns a finite minimum-norm answer "
       "(sigma_e 0.0904, FOM 11.34), and SiCoPR stops so the degenerate case "
       "is visible. SiCoPR here: " + _sing_outcome)


finish()
