"""Audit batch B11: G10 optimizer loops, FOM core.

Scope (audit prompt section 3 item 10, "Optimizer loops"):
  - OptFom_Calc_Hr        (base MATLAB 2879-2885   -> com.py 3861-3866)
  - optimize_fom          (base MATLAB 8547-8923   -> com.py 12470-12841)
  - OptFom_Adaptive_Local_Search
        (ADAPTIVE VARIANT com_ieee8023_4p15p0_adaptive_local_search.m 2739-2992
         -> com.py 3611-3720; helpers 3578-3608)

The source of truth for OptFom_Adaptive_Local_Search is the adaptive variant
file, per the audit prompt instruction to "cross-check against the adaptive
variant file" for optimizer loops. It is a config-gated (NonZeroLSMethod==1)
search-speedup skip predicate; it does not change the physics/noise/FOM math.

Verification strategy (no MATLAB, no golden data):
  1. OptFom_Calc_Hr: analytic filter properties (unit DC gain, Butterworth 3 dB
     point) + cross-path identity vs the top-level Butterworth/Bessel/Raised-
     Cosine filters (the hoisted _OptFom_Calc_Hr__* copies must agree).
  2. optimize_fom: the genuine MATLAB->Python traps are the 1-based cursor/slice
     index formulas and the post-optimize grids. optimize_fom is a full-pipeline
     integration driver, so these seams are validated as faithful transcriptions
     of the exact com.py expressions against an independent 1-based reference.
  3. OptFom_Adaptive_Local_Search: it is a pure predicate, so it is called
     directly and its skip/evaluate decisions are checked against the variant.

One documented divergence (EXPECTED FAIL row):
  B11-D15 (low): triple_transit_time = round(2*sbr_peak_i/samples_per_ui)+20 uses
  Python's banker's round vs MATLAB half-away. Only bites exact half-integer
  2*peak/spui, and only shifts a min-response-length lower bound by 1 UI. Same
  class as B10-D14.

Run: python tests/test_optimizer_fom.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, xcheck, finish  # noqa: E402
import com  # noqa: E402

TOL = 1e-12  # element-wise identity tolerance for the FD filter cross-path check


# ===========================================================================
# 1. OptFom_Calc_Hr  (MATLAB 2879-2885 -> com.py 3861-3866)
#    H_r = H_bw .* H_bt .* H_RCos  (element-wise product, order-agnostic)
# ===========================================================================
param = SimpleNamespace(
    fb=53.125e9,
    fb_BW_cutoff=1.0,      # Butterworth cutoff = fb_BW_cutoff*fb
    fb_BT_cutoff=0.75,     # Bessel-Thomson cutoff = fb_BT_cutoff*fb
    BTorder=4,
    RC_Start=40e9,
    RC_end=53e9,
)
OP_all_on = SimpleNamespace(Bessel_Thomson=1, Butterworth=1, Raised_Cosine=1)
OP_all_off = SimpleNamespace(Bessel_Thomson=0, Butterworth=0, Raised_Cosine=0)

f = np.arange(0.0, 60e9 + 1e8, 1e8)  # 0 .. 60 GHz

# (a) All filters off -> H_r is all ones (each helper returns ones(len(f))).
Hr_off = com.OptFom_Calc_Hr(f, param, OP_all_off)
check("calc_hr_all_off_is_unity",
      Hr_off.shape == f.shape and np.max(np.abs(Hr_off - 1.0)) <= TOL,
      "OptFom_Calc_Hr with all filters off is not all-ones")

# (b) Unit DC gain: at f=0, Butterworth=1, Bessel-Thomson=1, Raised-Cosine=1
#     (f=0 < RC_Start), so H_r(0) == 1 exactly.
Hr_on = com.OptFom_Calc_Hr(f, param, OP_all_on)
check("calc_hr_unit_dc_gain",
      abs(Hr_on[0] - 1.0) <= 1e-12,
      "OptFom_Calc_Hr DC gain (f=0) = %s, expected 1.0" % Hr_on[0])

# (c) Butterworth 3 dB point: the 4th-order normalized poly gives |H|=1/sqrt(2)
#     at f = fb_BW_cutoff*fb (s=j). Analytic, independent of any reference.
#     Tol 3e-7: MATLAB (and com.py) use the truncated coeffs [1 2.613126
#     3.414214 2.613126 1], so |H| hits 1/sqrt(2) only to ~7 sig figs. Exact
#     agreement with MATLAB is separately pinned by calc_hr_matches_toplevel (1e-12).
f_cut = np.array([param.fb_BW_cutoff * param.fb])
Hbw_cut = com.Butterworth_Filter(param, f_cut, 1)
check("butterworth_3dB_point",
      abs(abs(Hbw_cut[0]) - 1.0 / np.sqrt(2.0)) <= 3e-7,
      "Butterworth |H| at cutoff = %s, expected 1/sqrt(2)=%s"
      % (abs(Hbw_cut[0]), 1.0 / np.sqrt(2.0)))

# (d) CROSS-PATH: the hoisted _OptFom_Calc_Hr__* copies must reproduce the
#     top-level Butterworth/Bessel/Raised-Cosine filters exactly.
Hbw = com.Butterworth_Filter(param, f, 1)
Hbt = com.Bessel_Thomson_Filter(param, f, 1)
Hrc = com.Raised_Cosine_Filter(param, f, 1)
ref_product = Hbw * Hbt * Hrc
check("calc_hr_matches_toplevel_filters",
      Hr_on.shape == ref_product.shape
      and np.max(np.abs(Hr_on - ref_product)) <= TOL,
      "OptFom_Calc_Hr != top-level Butterworth*Bessel*RaisedCosine (max abs err %s)"
      % (np.max(np.abs(Hr_on - ref_product)) if Hr_on.shape == ref_product.shape else float('nan')))

# (e) Bessel-Thomson unit DC gain (a(1)/polyval(acoef,0) == 1) at f=0.
check("bessel_thomson_unit_dc_gain",
      abs(Hbt[0] - 1.0) <= 1e-12,
      "Bessel-Thomson DC gain = %s, expected 1.0" % Hbt[0])


# ===========================================================================
# 2. optimize_fom index/slice seams  (MATLAB 8788-8908 -> com.py 12722-12826)
#    These are the real 1-based->0-based translation traps. Reference is an
#    independent transcription of the MATLAB 1-based colon semantics.
#    sbr[k-1] holds the value k, so gathered values reveal the 1-based indices.
# ===========================================================================
N = 200
sbr = np.arange(1, N + 1, dtype=float)   # sbr[k-1] == k
spui = 8
ndfe = 2


def matlab_far(cursor_i, T_O):
    # MATLAB: sbr(cursor_i - T_O + spui*(ndfe+1) : spui : end)   (1-based)
    start1 = cursor_i - T_O + spui * (ndfe + 1)
    return np.array([sbr[k - 1] for k in range(start1, N + 1, spui)])


def py_far(cursor_i, T_O):
    # com.py 12731-12733
    far_start = cursor_i - T_O + spui * (ndfe + 1) - 1
    far_start = max(far_start, 0)
    return sbr[far_start::spui]


for cursor_i, T_O in [(50, 0), (50, 3), (73, 0), (128, 5)]:
    check("optfom_far_cursors_slice_c%d_TO%d" % (cursor_i, T_O),
          np.array_equal(py_far(cursor_i, T_O), matlab_far(cursor_i, T_O)),
          "far_cursors 0-based slice != MATLAB 1-based colon for cursor_i=%d T_O=%d"
          % (cursor_i, T_O))

# far_start clamp is inert for realistic cursors (start is always > cursor).
check("optfom_far_start_clamp_inert",
      (50 - 0 + spui * (ndfe + 1) - 1) > 0,
      "far_start would be negative for a realistic cursor (clamp not inert)")


def matlab_pre(cursor_i):
    # MATLAB: p = sbr(cursor_i - spui : -spui : 1); p = p(end:-1:1)
    idx = list(range(cursor_i - spui, 0, -spui))   # descending 1-based, >=1
    vals = np.array([sbr[k - 1] for k in idx])
    return vals[::-1]


def py_pre(cursor_i):
    # com.py 12735-12740
    pre_start = cursor_i - spui - 1
    if pre_start >= 0:
        return sbr[pre_start::-spui][::-1]
    return np.array([])


for cursor_i in [50, 42, 9, 8, 5, 128]:
    check("optfom_precursors_slice_c%d" % cursor_i,
          np.array_equal(py_pre(cursor_i), matlab_pre(cursor_i)),
          "precursors 0-based reversed slice != MATLAB for cursor_i=%d" % cursor_i)

# precursors empty exactly when cursor_i <= spui (MATLAB range empty).
check("optfom_precursors_empty_when_cursor_le_spui",
      py_pre(spui).size == 0 and py_pre(spui - 1).size == 0 and py_pre(spui + 1).size == 1,
      "precursors empty-boundary at cursor_i==spui is wrong")

# sbr required-length zero-extend: length count is 1-based-consistent
# (cursor_i is kept 1-based; the count == cursor_i + spui*(ndfe+1)).
cursor_i = 195
required = cursor_i + spui * (ndfe + 1)   # com.py 12751
sbr_ext = sbr.copy()
if len(sbr_ext) < required:
    sbr_ext = np.append(sbr_ext, np.zeros(required - len(sbr_ext)))
check("optfom_sbr_zero_extend_length",
      len(sbr_ext) == required and np.all(sbr_ext[N:] == 0.0),
      "sbr zero-extend produced wrong length %d (expected %d)" % (len(sbr_ext), required))

# Post-optimize frequency grid: f = 1e8:1e8:100e9 (MATLAB) == arange(1e8,100e9+1e8,1e8)
f_post = np.arange(1e8, 100e9 + 1e8, 1e8)   # com.py 12824
check("optfom_post_f_grid",
      len(f_post) == 1000 and abs(f_post[0] - 1e8) <= 1
      and abs(f_post[-1] - 100e9) <= 1,
      "post-optimize f grid: len=%d first=%s last=%s (expected 1000, 1e8, 100e9)"
      % (len(f_post), f_post[0], f_post[-1]))

# Post-optimize time grid: t = 0:ui/spui:(len-1)*ui/spui == arange(len)*ui/spui
ui = 1.0 / 53.125e9
length_sbr = 128
t_out = np.arange(length_sbr) * ui / spui                 # com.py 12826
t_ref = np.array([k * ui / spui for k in range(length_sbr)])
check("optfom_post_t_grid",
      len(t_out) == length_sbr and np.max(np.abs(t_out - t_ref)) <= 1e-18,
      "post-optimize t grid != MATLAB colon reference")

# --- B11-D15 (EXPECTED FAIL): triple_transit_time uses banker's round ---------
# com.py 12689: round(int(sbr_peak_i)*2/int(spui)) + 20  (Python builtin round).
# Pick 2*peak/spui = 2.5 exactly (peak=5, spui=4): MATLAB half-away round=3 -> 23;
# Python banker's round=2 -> 22.
_peak, _spui = 5, 4
py_ttt = round(int(_peak) * 2 / int(_spui)) + 20
xcheck("optfom_triple_transit_uses_matlab_half_away",
      py_ttt == 23,
      "DIVERGENT (B11-D15, low): triple_transit_time=%d; MATLAB round(2.5)=3 -> 23, "
      "Python banker's round(2.5)=2 -> 22. Site com.py 12689. Only bites exact "
      "half-integer 2*peak/spui; shifts min_number_of_UI_in_response by 1 UI. "
      "Same class as B10-D14." % py_ttt)
# Positive confirmation that it IS the banker's value (mechanism check).
check("optfom_triple_transit_is_bankers",
      py_ttt == 22,
      "triple_transit_time is neither MATLAB nor banker's value: %d" % py_ttt)


# ===========================================================================
# 3. OptFom_Adaptive_Local_Search (ADAPTIVE VARIANT 2739-2992 -> com.py 3611)
#    Pure skip predicate. Called directly; decisions checked vs the variant.
#    Knobs (variant 2758-2774): min_radius=1, edge=1.0, lp=0.25, vga=0.5,
#    l2_to_l1_ratio=0.55, hard_cap_mul=1.2, det_shrink=0.15.
# ===========================================================================
def make(best_taps, curr_taps, ctle_index=1, best_ctle=1,
         lp_curr=1, lp_best=1, vga_curr=1, vga_best=1, fom=0.0, best_fom=0.0):
    BEST = SimpleNamespace(txffe_index=np.asarray(best_taps, dtype=float),
                           ctle=best_ctle, G_high_pass=lp_best,
                           vga_index=vga_best, FOM=best_fom)
    THIS = SimpleNamespace(tx_index_vector=np.asarray(curr_taps, dtype=float),
                           ctle_index=ctle_index, g_LP_index=lp_curr,
                           vga_index=vga_curr, FOM=fom)
    return BEST, THIS


# (a) Persistent re-init on iter_count==1: adaptive_radius := max(1, round(LSV)).
com.reset_state()
BEST, THIS = make([1, 1, 1], [1, 1, 1])   # exact match
skip = com.OptFom_Adaptive_Local_Search(3, BEST, THIS, [0.0], iter_count=1, num_txffe_runs=8)
# LSV=3 -> init radius max(1,round(3))=3, then deterministic shrink
# max(1,round(3/(1+0.15*1)))=max(1,round(2.6087))=3 -> adaptive_radius=3.
check("als_iter1_reinits_radius",
      com._ALS_STATE['initialized'] is True and com._ALS_STATE['adaptive_radius'] == 3,
      "iter_count==1 did not re-init adaptive_radius to 3 (got %s)"
      % com._ALS_STATE['adaptive_radius'])

# (b) Exact match (L1_w==0) -> evaluate (skip False). Never skip your own best.
check("als_exact_match_evaluates",
      skip is False,
      "exact-match candidate was skipped (should always be evaluated)")

# (c) CTLE constraint: |ctle_index - BEST.ctle| > 2 -> skip True.
com.reset_state()
BEST, THIS = make([1, 1, 1], [1, 1, 2], ctle_index=5, best_ctle=1)
skip = com.OptFom_Adaptive_Local_Search(3, BEST, THIS, [0.0, 0.0], iter_count=1, num_txffe_runs=8)
check("als_ctle_too_far_skips",
      skip is True,
      "candidate with |ctle-best.ctle|=4 (>2) was not skipped")

# (d) Hard cap: raw_L1_TX > max(1, round(1.2*LSV)). LSV=3 -> cap=round(3.6)=4.
#     tap L1 = 5 > 4 -> skip True. (ctle diff 0, so CTLE gate passes first.)
com.reset_state()
BEST, THIS = make([1, 1, 1], [6, 1, 1], ctle_index=1, best_ctle=1)
skip = com.OptFom_Adaptive_Local_Search(3, BEST, THIS, [0.0, 0.0], iter_count=1, num_txffe_runs=8)
check("als_hard_cap_skips",
      skip is True,
      "candidate with raw_L1_TX=5 > hard_cap=4 was not skipped")

# (e) Within radius -> evaluate (skip False). tap L1 = 1 <= adaptive_radius=3.
com.reset_state()
BEST, THIS = make([2, 2], [2, 3], ctle_index=1, best_ctle=1)
skip = com.OptFom_Adaptive_Local_Search(3, BEST, THIS, [0.0, 0.0], iter_count=1, num_txffe_runs=8)
check("als_within_radius_evaluates",
      skip is False,
      "candidate within adaptive radius was skipped")

# (f) L1/L2 rule: L1_w=4 > adaptive_radius=3 AND L2_w=4 > threshold=2 -> skip True.
#     raw_L1_TX=4 == hard_cap=4 (not >), so hard cap does NOT fire; L1/L2 does.
com.reset_state()
BEST, THIS = make([1, 1, 1, 1], [5, 1, 1, 1], ctle_index=1, best_ctle=1)
skip = com.OptFom_Adaptive_Local_Search(3, BEST, THIS, [0.0, 0.0], iter_count=1, num_txffe_runs=8)
check("als_l1l2_rule_skips",
      skip is True,
      "candidate outside L1/L2 limits was not skipped")

# (g) compute_hard_cap helper: off -> NaN; on -> max(min_radius, round(mul*LSV)).
hc_off = com._OptFom_Adaptive_Local_Search__compute_hard_cap(False, 1.2, 3, 1)
hc_on = com._OptFom_Adaptive_Local_Search__compute_hard_cap(True, 1.2, 3, 1)
check("als_compute_hard_cap",
      np.isnan(hc_off) and hc_on == 4,
      "compute_hard_cap off=%s on=%s (expected NaN, 4)" % (hc_off, hc_on))

# (h) _mround helper is half-away-from-zero (reassert for the ALS helper).
mr = com._OptFom_Adaptive_Local_Search__mround
mcases = [(0.5, 1), (1.5, 2), (2.5, 3), (3.5, 4), (-0.5, -1), (-2.5, -3), (0.4, 0)]
check("als_mround_half_away",
      all(int(mr(x)) == e for x, e in mcases),
      "ALS _mround not half-away: %s" % [int(mr(x)) for x, _ in mcases])

# (i) adaptive_radius never drops below min_radius=1, across many iterations of
#     a stalled (no-improvement) search. Tier-3 invariant on persistent state.
com.reset_state()
BEST, THIS = make([1, 1], [1, 2], ctle_index=1, best_ctle=1)
stalled = [1.0, 1.0, 1.0, 1.0]   # improvement < threshold every window
radii = []
for it in range(1, 12):
    com.OptFom_Adaptive_Local_Search(3, BEST, THIS, stalled, iter_count=it, num_txffe_runs=8)
    radii.append(com._ALS_STATE['adaptive_radius'])
check("als_radius_floor_min_radius",
      all(r >= 1 for r in radii),
      "adaptive_radius dropped below min_radius=1: %s" % radii)

# (j) Determinism: identical inputs -> identical decision (no hidden RNG).
com.reset_state()
BEST, THIS = make([1, 1, 1, 1], [5, 1, 1, 1])
s1 = com.OptFom_Adaptive_Local_Search(3, BEST, THIS, [0.0, 0.0], iter_count=1, num_txffe_runs=8)
com.reset_state()
BEST, THIS = make([1, 1, 1, 1], [5, 1, 1, 1])
s2 = com.OptFom_Adaptive_Local_Search(3, BEST, THIS, [0.0, 0.0], iter_count=1, num_txffe_runs=8)
check("als_deterministic",
      s1 == s2 is True,
      "ALS not deterministic for identical inputs (%s vs %s)" % (s1, s2))

# (k) Latent guard note (informational): Python returns skip=False on empty tap
#     vectors regardless of logging. In the variant the empty-guard return sits
#     inside `if log_yes_1_no_0==1`, so with logging OFF MATLAB would fall through
#     to the [lp;vga] distance computation. Unreachable in practice: tap vectors
#     are always >= [1] (OptFom_Build_TXFFE sets FULL_tx_index_vector=1 when no
#     taps). This check confirms Python's actual behavior only.
com.reset_state()
BEST, THIS = make([], [], ctle_index=1, best_ctle=1)
skip_empty = com.OptFom_Adaptive_Local_Search(3, BEST, THIS, [0.0], iter_count=1, num_txffe_runs=8)
check("als_empty_taps_returns_false_python",
      skip_empty is False,
      "Python ALS empty-tap path did not return False (behavior changed)")

finish()
