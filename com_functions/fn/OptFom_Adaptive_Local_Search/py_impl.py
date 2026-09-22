# ============================================================
# MATLAB→Python translation notes for OptFom_Adaptive_Local_Search
# MATLAB lines: 2739-2992 (com_ieee8023_4p15p0_adaptive_local_search.m)
# NEW in Hansel D'silva's adaptive-local-search branch.
# ============================================================
# A skip/prune predicate for the EQ search loop. Given the current
# candidate EQ setting (THIS) vs. the best-so-far (BEST), the FOM history
# and the iteration count, it keeps a *persistent* adaptive_radius that
# grows when the FOM is improving and shrinks when it stalls, then skips
# candidates that are too far from BEST in a weighted L1/L2 tap-space
# distance (TX-FFE taps + CTLE low-pass index + VGA index), subject to a
# hard cap on the raw TX-tap L1 distance and a CTLE-index window.
#
# persistent vs Python: MATLAB's `persistent adaptive_radius
# no_improve_count initialized` is emulated with a module-level dict that
# resets whenever iter_count == 1 (start of an optimize_fom search), which
# is exactly when the MATLAB code re-initialises.
#
# 1-based/0-based: indices here are compared as values (THIS.ctle_index,
# BEST.ctle, etc.), not used to index arrays, so no offset is needed.
# round() is MATLAB round (half away from zero).
#
# Optional logging: set ALS_LOG_CSV to a path to append one row per
# candidate (the data source for the sweep-vs-adaptive comparison plot).
# Default None == Hansel's log_yes_1_no_0 = 0 (logging off; no effect on
# the skip decision).
# ============================================================

import numpy as np
from com_functions.fn.compute_hard_cap.py_impl import compute_hard_cap as _compute_hard_cap

def _mextreme_complex(a, take):
    """MATLAB orders complex values by magnitude, then by angle; numpy orders
    them lexicographically by real part, so max([3+4i, 5]) is 3+4i in MATLAB
    and 5 in numpy. take is -1 for max, 0 for min."""
    f = np.asarray(a).ravel()
    good = ~np.isnan(np.abs(f))
    if not good.any():
        return f[0]
    g = f[good]
    return g[np.lexsort((np.angle(g), np.abs(g)))[take]]


def _mmax(a):
    """MATLAB max(): a NaN is skipped unless every element is NaN, and complex
    values are ordered by magnitude then angle.

    np.max propagates a NaN, so one bad sample swallows the result where MATLAB
    ignores it. np.nanmax matches MATLAB but warns on an all-NaN input, where
    MATLAB quietly returns NaN. The isnan test also keeps the ordinary no-NaN
    case on np.max's faster path.
    """
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, -1)
    if a.dtype.kind != 'f':
        return np.max(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.max(a)
    return np.nanmax(a)


def _mmin(a):
    """MATLAB min(): the mirror of _mmax."""
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, 0)
    if a.dtype.kind != 'f':
        return np.min(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.min(a)
    return np.nanmin(a)


# --- persistent state (MATLAB `persistent`) and optional logging knob ---
_ALS_STATE = {'adaptive_radius': None, 'no_improve_count': 0, 'initialized': False}
ALS_LOG_CSV = None  # path -> enable per-candidate CSV logging (feature 3)

_ALS_HEADER = ['iter', 'adaptive_radius', 'deterministic_radius', 'raw_L1_TX',
               'L1_w', 'L2_w', 'hard_cap', 'curr_taps', 'ctle_index', 'lp_idx',
               'best_taps', 'best_ctle', 'best_lp', 'vga_index', 'best_vga_index',
               'THIS_FOM', 'BEST_FOM', 'FOM_end', 'skip_it', 'skip_reason']


def _mround(x):
    """MATLAB round(): half away from zero."""
    x = float(x)
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    # Off a tie round() is exact, and unlike floor(x + 0.5) it does not
    # send 0.49999999999999994 to 1: that sum is exactly 1.0 in binary.
    return int(round(x))


def _append_csv_row(file_path, header_cells, row_cells):
    """Inlined append_csv_row (MATLAB lines 5157-5187)."""
    import os
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as fid:
        if not file_exists:
            fid.write(','.join(str(h) for h in header_cells) + '\n')
        if row_cells:
            out = []
            for v in row_cells:
                if isinstance(v, (bool, np.bool_)):
                    out.append(f'{float(v):.6g}')
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    out.append(f'{v:.6g}')
                elif isinstance(v, str):
                    out.append(f'"{v}"')
                else:
                    out.append('""')
            fid.write(','.join(out) + '\n')


def OptFom_Adaptive_Local_Search(LocalSearch_Value, BEST, THIS, FOM_history,
                                 iter_count, num_txffe_runs,
                                 Overwrite_Min_Radius=None,
                                 matlab_version='4p15p0'):
    """Adaptive local-search skip predicate.

    4p15p0 baseline: Hansel's branch file, com_ieee8023_4p15p0_adaptive_local_search.m
    4p16p0: mainline, L2759-2922 -- the method was adopted upstream.

    The mainline signature puts Overwrite_Min_Radius second:
        OptFom_Adaptive_Local_Search(LocalSearch_Value, Overwrite_Min_Radius,
                                     BEST, THIS, FOM_history, iter_count,
                                     num_txffe_runs)
    Here it is a trailing keyword instead, so existing 4p15p0 callers and tests
    are unaffected by argument order.

    Returns skip_it (bool): True -> skip evaluating this candidate.
    """
    _v416 = str(matlab_version) == '4p16p0'

    # ---- Tuned knobs (PATCHED values from Hansel's branch) ----
    min_improvement_threshold = 0.002
    adaptation_window = 2
    radius_shrink_factor = 0.60
    deterministic_shrink_rate = 0.15
    # ML 4p16p0 L2782-2786: the radius floor is 1 only when there is a single
    # TXFFE candidate, otherwise 2.
    #
    # This rule is applied on BOTH version paths, which needs saying because the
    # 4p15p0 adaptive-search branch source we hold forces 1 unconditionally.
    # The reference workbooks do not behave that way. With the Tx FFE grid the
    # reference actually swept (1584 candidates), a floor of 1 reproduces 5 of
    # the 10 crosstalk cases that turn on it and a floor of 2 reproduces 10 of
    # 10 -- see docs/MIN_RADIUS_ASSUMPTION.md. num_txffe_runs is the only
    # quantity in the 4p16p0 delta that this floor depends on, so the mainline
    # rule is both the simplest statement of the observed behaviour and one that
    # already exists in published MATLAB. Modelling the branch's forced 1 would
    # mean modelling a source revision that demonstrably did not produce the
    # reference results.
    #
    # The single-candidate arm is unchanged by this: 1 either way.
    min_radius = 1 if int(num_txffe_runs) == 1 else 2
    # ML L2788-2792: a positive config value overrides the rule; empty leaves it
    # alone. Honoured on both paths -- it used to be read only under 4p16p0, so a
    # config setting it while emulating 4p15p0 had it silently discarded.
    if Overwrite_Min_Radius is not None:
        try:
            _omr = float(np.asarray(Overwrite_Min_Radius).ravel()[0])
        except (TypeError, ValueError, IndexError):
            _omr = None
        if _omr is not None and _omr > 0:
            min_radius = _omr
    edge_weight = 1.0
    lp_weight = 0.25
    vga_weight = 0.5
    l2_to_l1_ratio = 0.55
    use_hard_cap = True
    hard_cap_multiplier = 1.2

    st = _ALS_STATE

    # ---- Persistent memory (re-init at the start of each search) ----
    # 4p16p0 L2794-2796 drops the `initialized` persistent and keys the reset on
    # iter_count == 1 alone. Left as-is because optimize_fom increments
    # iter_count from 0 before the first call, so the first call of every search
    # always has iter_count == 1 -- the two forms cannot diverge here, and a
    # version branch would add a path with no behavioural difference to test.
    if (not st['initialized']) or iter_count == 1:
        st['adaptive_radius'] = max(min_radius, _mround(LocalSearch_Value))
        st['no_improve_count'] = 0
        st['initialized'] = True

    # ---- Current FOM value ----
    FOM_history = np.asarray(FOM_history, dtype=float).ravel()
    FOM = float(FOM_history[-1]) if FOM_history.size > 0 else float('nan')

    # ---- ADAPTIVE shrink ----
    if FOM_history.size >= adaptation_window:
        recent = FOM_history[-adaptation_window:]
        improvement = float(_mmax(recent) - _mmin(recent))
        if improvement < min_improvement_threshold:
            st['no_improve_count'] += 1
        else:
            st['no_improve_count'] = 0
            st['adaptive_radius'] = min(st['adaptive_radius'] + 1, LocalSearch_Value)
        # shrink only when CTLE is "stable"
        if st['no_improve_count'] >= 1 and abs(THIS.ctle_index - BEST.ctle) <= 1:
            st['adaptive_radius'] = max(min_radius, _mround(st['adaptive_radius'] * radius_shrink_factor))
            st['no_improve_count'] = 0

    # ---- Deterministic shrink ----
    deterministic_radius = max(min_radius, _mround(LocalSearch_Value / (1 + deterministic_shrink_rate * iter_count)))
    st['adaptive_radius'] = max(min_radius, min(st['adaptive_radius'], deterministic_radius))
    adaptive_radius = st['adaptive_radius']

    # ---- Extract tap vectors ----
    best_taps = np.asarray(BEST.txffe_index, dtype=float).ravel()
    curr_taps = np.asarray(THIS.tx_index_vector, dtype=float).ravel()
    ctle_index = THIS.ctle_index
    lp_curr = THIS.g_LP_index
    lp_best = BEST.G_high_pass
    if _v416:
        # ML 4p16p0 L2825-2827 writes THIS.vga_index = 1 and BEST.vga_index = 1
        # and then reads them back, replacing the branch's isfield fallbacks.
        # The net effect is that both are 1 regardless of what the caller set.
        #
        # Deliberately NOT writing to THIS/BEST here. MATLAB passes structs by
        # value, so those two assignments are local and the caller never sees
        # them; in Python they would escape and silently pin the caller's
        # vga_index. That is the defect class behind five of the eight engine
        # bugs, and tests/test_reference_leaks.py flags it -- it caught this
        # exact code on the first run.
        vga_curr = 1
        vga_best = 1
    else:
        vga_curr = getattr(THIS, 'vga_index', 1)
        vga_best = getattr(BEST, 'vga_index', 1)

    def _finish(skip_it, reason, raw_L1_TX=float('nan'), L1_w=float('nan'),
                L2_w=float('nan'), hard_cap=float('nan')):
        if ALS_LOG_CSV is not None:
            _append_csv_row(ALS_LOG_CSV, _ALS_HEADER, [
                iter_count, adaptive_radius, deterministic_radius, raw_L1_TX,
                L1_w, L2_w, hard_cap, np.array2string(curr_taps), ctle_index, lp_curr,
                np.array2string(best_taps), BEST.ctle, lp_best, vga_curr, vga_best,
                float(THIS.FOM), float(BEST.FOM), FOM, bool(skip_it), reason])
        return bool(skip_it)

    if best_taps.size == 0 or curr_taps.size == 0:
        return _finish(False, 'Skip: Empty BEST.txffe_index or THIS.tx_index_vector')

    # ---- Build weighted vectors ----
    best_vec = np.concatenate([best_taps, [lp_best], [vga_best]])
    this_vec = np.concatenate([curr_taps, [lp_curr], [vga_curr]])
    num_taps = curr_taps.size

    if ctle_index > 1:
        w_lp = lp_weight * (1 + 0.5 * (ctle_index - 1))
        w_vga = vga_weight * (1 + 0.3 * (ctle_index - 1))
    else:
        w_lp = lp_weight
        w_vga = vga_weight
    weights = np.concatenate([np.ones(num_taps) * edge_weight, [w_lp], [w_vga]])

    # ---- Distances ----
    diff_vec = this_vec - best_vec
    weighted_diff = weights * diff_vec
    L1_w = float(np.sum(np.abs(weighted_diff)))
    L2_w = float(np.sqrt(np.sum(weighted_diff ** 2)))
    raw_L1_TX = float(np.sum(np.abs(diff_vec[:num_taps])))

    # ---- CTLE constraint (±2) ----
    if abs(ctle_index - BEST.ctle) > 2:
        return _finish(True, 'Skip: CTLE_idx too far', raw_L1_TX, L1_w, L2_w)

    # ---- Early exact match ----
    if L1_w == 0:
        return _finish(False, 'Evaluate Candidate: Exact Match', raw_L1_TX, L1_w, L2_w)

    # ---- Hard cap ----
    hard_cap = _compute_hard_cap(use_hard_cap, hard_cap_multiplier, LocalSearch_Value, min_radius)
    if use_hard_cap and raw_L1_TX > hard_cap:
        return _finish(True, 'Skip: TX Exceeds Cap', raw_L1_TX, L1_w, L2_w, hard_cap)

    # ---- L1/L2 skip rule ----
    L2_threshold = max(min_radius, int(np.ceil(l2_to_l1_ratio * adaptive_radius)))
    skip_it = (L1_w > adaptive_radius) and (L2_w > L2_threshold)
    reason = 'Skip: Outside L1/L2 Limits' if skip_it else 'Evaluate Candidate'
    return _finish(skip_it, reason, raw_L1_TX, L1_w, L2_w, hard_cap)


def reset_state():
    """Clear persistent state (call before an independent search run)."""
    _ALS_STATE.update(adaptive_radius=None, no_improve_count=0, initialized=False)


if __name__ == '__main__':
    from types import SimpleNamespace
    BEST = SimpleNamespace(txffe_index=np.array([0, 0, 0]), ctle=3, G_high_pass=2, FOM=5.0)
    THIS = SimpleNamespace(tx_index_vector=np.array([0, 0, 0]), ctle_index=3,
                           g_LP_index=2, FOM=4.0)
    reset_state()
    print('exact match -> skip?', OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0], 1, 5))
    THIS.tx_index_vector = np.array([9, 9, 9])  # far away
    print('far -> skip?', OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0, 4.0, 4.0], 5, 5))
