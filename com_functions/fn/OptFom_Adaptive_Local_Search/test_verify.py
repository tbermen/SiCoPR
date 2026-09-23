# ============================================================
# MATLAB GROUND TRUTH for OptFom_Adaptive_Local_Search (L2739-2992)
# ------------------------------------------------------------
# The radius floor is NOT a constant: min_radius = 1 if num_txffe_runs == 1
# else 2 (ML 4p16p0 L2782-2786), applied on both version paths -- see
# docs/MIN_RADIUS_ASSUMPTION.md. Cases that exercise the shrink behaviour
# therefore pass num_txffe_runs=1 so the floor is 1 and the radius is free to
# fall; the rule itself is pinned separately at the bottom of this file.
#
# Decision branches (LocalSearch_Value=2, num_txffe_runs=1 -> min_radius=1,
# hard_cap = max(1, round(1.2*2)) = 2):
#   * exact match (L1_w == 0)                 -> skip = False
#   * |ctle_index - BEST.ctle| > 2            -> skip = True
#   * raw TX L1 distance > hard_cap (=2)      -> skip = True
#   * within radius                           -> skip = False
#   * L1_w > adaptive_radius AND L2_w > thr   -> skip = True
#   * empty tap vectors                       -> skip = False
# Deterministic shrink: adaptive_radius <= round(2/(1+0.15*iter)).
# These are hand-derived from the algorithm (no MATLAB run needed).
# ============================================================
import os
import sys
import numpy as np
import pytest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import com_functions.fn.OptFom_Adaptive_Local_Search.py_impl as py_impl
from com_functions.fn.OptFom_Adaptive_Local_Search.py_impl import (
    OptFom_Adaptive_Local_Search, reset_state)


def _mk(best_taps, this_taps, ctle_index, best_ctle=3, lp=2, best_lp=2,
        this_fom=4.0, best_fom=5.0):
    BEST = SimpleNamespace(txffe_index=np.asarray(best_taps), ctle=best_ctle,
                           G_high_pass=best_lp, FOM=best_fom)
    THIS = SimpleNamespace(tx_index_vector=np.asarray(this_taps), ctle_index=ctle_index,
                           g_LP_index=lp, FOM=this_fom)
    return BEST, THIS


def setup_function(_):
    reset_state()


def test_exact_match_not_skipped():
    BEST, THIS = _mk([0, 0, 0], [0, 0, 0], ctle_index=3)
    assert OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0], 1, 5) is False


def test_ctle_too_far_skipped():
    BEST, THIS = _mk([0, 0, 0], [1, 0, 0], ctle_index=10, best_ctle=3)
    assert OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0], 1, 5) is True


def test_hard_cap_exceeded_skipped():
    # raw TX L1 = 5 > hard_cap = 2
    BEST, THIS = _mk([0, 0, 0], [5, 0, 0], ctle_index=3, best_ctle=3)
    assert OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0], 1, 5) is True


def test_within_radius_not_skipped():
    # raw TX L1 = 1 <= hard_cap; L1_w = 1 <= adaptive_radius (=2 at iter 1)
    BEST, THIS = _mk([0, 0, 0], [1, 0, 0], ctle_index=3, best_ctle=3)
    assert OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0], 1, 5) is False


def test_outside_l1_l2_skipped():
    # num_txffe_runs=1 -> min_radius=1, so the radius is free to shrink.
    # iter 10 -> deterministic_radius = round(2/(1+1.5)) = 1, so adaptive_radius=1.
    # taps [1,1,0] vs [0,0,0]: raw_L1_TX=2 (==hard_cap, not >), L1_w=2>1, L2_w=1.41>1 -> skip.
    BEST, THIS = _mk([0, 0, 0], [1, 1, 0], ctle_index=3, best_ctle=3)
    assert OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0, 4.0, 4.0], 10, 1) is True


def test_min_radius_floor_follows_the_txffe_candidate_count():
    """The floor rule itself: 1 for a single TXFFE candidate, 2 for a real grid.

    Same candidate, same iteration, only num_txffe_runs differs. With one
    candidate the floor is 1, the radius shrinks to 1 by iteration 10, and the
    candidate falls outside it. With a multi-candidate grid the floor is 2, the
    radius cannot shrink past it, and the same candidate stays in.

    This is the setting deduced for the MATLAB reference runs (5 of 10 crosstalk
    cases reproduce at a floor of 1, 10 of 10 at 2), so it is pinned here rather
    than left to emerge from a correlation run.
    """
    BEST, THIS = _mk([0, 0, 0], [1, 1, 0], ctle_index=3, best_ctle=3)
    reset_state()
    assert OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0, 4.0, 4.0], 10, 1) is True
    reset_state()
    assert OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0, 4.0, 4.0], 10, 2) is False


def test_min_radius_rule_is_version_independent():
    """4p15p0 and 4p16p0 apply the same floor.

    The 4p15p0 adaptive-search branch source forces 1 unconditionally; the port
    deliberately does not, because the reference workbooks do not behave that
    way. If someone 'restores' the branch behaviour on the 4p15p0 path, this
    fails.
    """
    BEST, THIS = _mk([0, 0, 0], [1, 1, 0], ctle_index=3, best_ctle=3)
    for ver in ('4p15p0', '4p16p0'):
        reset_state()
        assert OptFom_Adaptive_Local_Search(
            2, BEST, THIS, [4.0, 4.0, 4.0], 10, 5, matlab_version=ver) is False


def test_overwrite_min_radius_wins_on_both_paths():
    """A positive config value overrides the rule (ML 4p16p0 L2788-2792).

    It used to be read only under 4p16p0, so a config setting it while emulating
    4p15p0 had it silently discarded.
    """
    BEST, THIS = _mk([0, 0, 0], [1, 1, 0], ctle_index=3, best_ctle=3)
    for ver in ('4p15p0', '4p16p0'):
        reset_state()
        # multi-candidate grid would give 2; the override forces it back to 1
        assert OptFom_Adaptive_Local_Search(
            2, BEST, THIS, [4.0, 4.0, 4.0], 10, 5,
            Overwrite_Min_Radius=1, matlab_version=ver) is True


def test_empty_taps_not_skipped():
    BEST, THIS = _mk([], [], ctle_index=3, best_ctle=3)
    assert OptFom_Adaptive_Local_Search(2, BEST, THIS, [], 1, 5) is False


def test_returns_python_bool():
    BEST, THIS = _mk([0, 0, 0], [1, 0, 0], ctle_index=3)
    out = OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0], 1, 5)
    assert isinstance(out, bool)


def test_optional_logging_writes_trajectory(tmp_path):
    log = str(tmp_path / 'ALS_log.csv')
    py_impl.ALS_LOG_CSV = log
    try:
        BEST, THIS = _mk([0, 0, 0], [1, 0, 0], ctle_index=3, best_ctle=3)
        OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0], 1, 5)
        OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0, 4.1], 2, 5)
    finally:
        py_impl.ALS_LOG_CSV = None
    lines = open(log).read().splitlines()
    assert lines[0].startswith('iter,adaptive_radius')
    assert len(lines) == 3  # header + 2 candidate rows
    assert 'skip_reason' in lines[0]


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))


# ---------------------------------------------------------------------------
# COM Octave oracle tests.
#
# skip_it values below come from running OptFom_Adaptive_Local_Search verbatim
# out of octave/com_ieee8023_4p16p0_octave_compat.m (with compute_hard_cap and
# append_csv_row) through Octave.  The extracted body was first checked
# identical to matlab/com_ieee8023_4p16p0.m.
#
# `persistent adaptive_radius no_improve_count` only means anything across
# calls, so every probe is a SEQUENCE run in one Octave session, started with
# `clear -f` and with iter_count==1 on the first call -- which is what
# optimize_fom does, since it increments iter_count from 0.  A first call with
# iter_count != 1 leaves the persistents empty, and `[] && x` is not something
# Octave can be trusted on, so that state is deliberately not pinned.
#
# These pin the 4p16p0 path.  The port's default, matlab_version='4p15p0',
# follows Hansel's branch file, which is not in the compat build and therefore
# has no oracle here; the only behavioural difference is that 4p16p0 forces
# vga_index to 1 on both sides.
# ---------------------------------------------------------------------------

_OCT_VER = dict(matlab_version='4p16p0')


def _oct_mk(this_taps, best_taps=(0.0, 0.0, 0.0), ctle_index=3, best_ctle=3,
            lp=2.0, best_lp=2.0, **extra):
    BEST = SimpleNamespace(txffe_index=np.asarray(best_taps, float), ctle=best_ctle,
                           G_high_pass=best_lp, FOM=5.0)
    THIS = SimpleNamespace(tx_index_vector=np.asarray(this_taps, float),
                           ctle_index=ctle_index, g_LP_index=lp, FOM=4.0)
    for k, v in extra.items():
        (BEST if k.startswith('best_') else THIS).__dict__[
            k[5:] if k.startswith('best_') else k] = v
    return BEST, THIS


def _seq(calls):
    """calls: (LSV, this_taps, FOM_history, iter, nruns, kwargs) tuples."""
    reset_state()
    out = []
    for LSV, taps, fom, it, nruns, kw in calls:
        mkkw = {k: v for k, v in kw.items()
                if k in ('best_taps', 'ctle_index', 'best_ctle', 'lp', 'best_lp',
                         'vga_index', 'best_vga_index')}
        callkw = {k: v for k, v in kw.items() if k == 'Overwrite_Min_Radius'}
        BEST, THIS = _oct_mk(taps, **mkkw)
        out.append(OptFom_Adaptive_Local_Search(LSV, BEST, THIS, fom, it, nruns,
                                                **callkw, **_OCT_VER))
    return out


def _warm(LSV=4, nruns=5, **kw):
    return (LSV, (0.0, 0.0, 0.0), [1.0], 1, nruns, kw)


def test_oracle_radius_trajectory_over_twelve_iterations():
    """The grow/shrink loop, pinned end to end.

    LocalSearch_Value=8, taps [3 1 0] against [0 0 0], FOM improving by 0.5 a
    step for four steps and then flat.  COM Octave skip_it, iterations 1..12:
    0 0 0 0 0 1 1 1 1 1 1 1.
    """
    calls = [(8, (3.0, 1.0, 0.0),
              [1.0 + 0.5 * min(j, 4) for j in range(k + 1)], k + 1, 5, {})
             for k in range(12)]
    assert _seq(calls) == [False] * 5 + [True] * 7


def test_oracle_widening_tap_distance_sequence():
    """LocalSearch_Value=4, taps [k 0 0] on iteration k+1, flat FOM.

    COM Octave skip_it for k=0..5: 0 0 0 1 1 1.
    """
    calls = [(4, (float(k), 0.0, 0.0), [4.0] * (k + 1), k + 1, 5, {})
             for k in range(6)]
    assert _seq(calls) == [False, False, False, True, True, True]


def test_oracle_nan_local_search_value_still_decides():
    """round(NaN) is NaN and max(min_radius, NaN) is min_radius in MATLAB.

    COM Octave, LocalSearch_Value=NaN with taps [3 0 0]: skip_it is 1 on both
    iterations, because hard_cap falls back to min_radius=2 and raw_L1_TX=3
    exceeds it.  The port raised ValueError out of _mround.
    """
    assert _seq([(float('nan'), (3.0, 0.0, 0.0), [1.0], 1, 5, {}),
                 (float('nan'), (3.0, 0.0, 0.0), [4.0, 4.0], 2, 5, {})]) \
        == [True, True]


def test_oracle_inf_local_search_value_still_decides():
    """round(Inf) is Inf, so every radius is Inf and nothing is ever skipped.

    COM Octave, LocalSearch_Value=Inf with taps [3 0 0]: skip_it is 0 on both
    iterations.  The port raised OverflowError out of _mround, and would have
    raised again at int(ceil(0.55*Inf)).
    """
    assert _seq([(float('inf'), (3.0, 0.0, 0.0), [1.0], 1, 5, {}),
                 (float('inf'), (3.0, 0.0, 0.0), [4.0, 4.0], 2, 5, {})]) \
        == [False, False]


def test_oracle_fom_history_tail_is_column_major():
    """`FOM_history(end-1:end)` indexes linearly, which is COLUMN-major.

    COM Octave with FOM_history = [1 1 ; 3 1.0005] on iterations 3, 4 and 5:
    skip_it is 1, 1, 1.  The window is [1 1.0005], improvement 5e-4, below the
    0.002 threshold, so the radius shrinks.  A row-major ravel takes
    [3 1.0005], improvement ~2, treats it as improving and gave 0, 0, 1.
    """
    H = [[1.0, 1.0], [3.0, 1.0005]]
    calls = [_warm()] + [(4, (3.0, 0.0, 0.0), H, it, 5, {}) for it in (3, 4, 5)]
    assert _seq(calls) == [False, True, True, True]
    # a 3x2 history, same reasoning
    H2 = [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0002]]
    calls2 = [_warm(6)] + [(6, (4.0, 0.0, 0.0), H2, it, 5, {}) for it in (2, 3)]
    assert _seq(calls2) == [False, True, True]


@pytest.mark.parametrize('taps_this,taps_best', [
    ((), (0.0, 0.0, 0.0)),            # empty THIS.tx_index_vector
    ((0.0, 0.0, 0.0), ()),            # empty BEST.txffe_index
])
def test_oracle_one_sided_empty_taps_is_an_error(taps_this, taps_best):
    """MATLAB's empty-vector early return sits inside `if log_yes_1_no_0 == 1`.

    With logging off -- the shipped state and this port's default -- it does
    not fire, and the mismatched vectors reach `this_vec - best_vec`.  COM
    Octave: "error: operator -: nonconformant arguments (op1 is 2x1, op2 is
    5x1)".  The port returned False for every empty case.
    """
    reset_state()
    BEST, THIS = _oct_mk(taps_this, best_taps=taps_best)
    OptFom_Adaptive_Local_Search(4, *_oct_mk((0.0, 0.0, 0.0)), [1.0], 1, 5,
                                 **_OCT_VER)
    with pytest.raises(ValueError):
        OptFom_Adaptive_Local_Search(4, BEST, THIS, [4.0, 4.0], 3, 5,
                                     **_OCT_VER)


def test_oracle_both_taps_empty_is_an_exact_match():
    """With both empty the vectors are both [lp ; vga], so L1_w is 0.

    COM Octave returns skip_it=0, which is also what the removed early return
    happened to give -- pinned so the removal is shown not to have moved it.
    """
    assert _seq([_warm(), (4, (), [4.0, 4.0], 3, 5, {'best_taps': ()})]) \
        == [False, False]


@pytest.mark.parametrize('kw,expected', [
    ({}, True),                                  # min_radius = 2
    ({'Overwrite_Min_Radius': 5}, False),        # floor raised to 5
    ({'Overwrite_Min_Radius': -1}, True),        # not positive, so ignored
])
def test_oracle_overwrite_min_radius(kw, expected):
    """COM Octave with LocalSearch_Value=4, taps [3 0 0] on iteration 4:
    skip_it is 1 by default, 0 with Overwrite_Min_Radius=5, and 1 again with
    -1, which `if Overwrite_Min_Radius > 0` rejects."""
    calls = [_warm(4, 5, **kw), (4, (3.0, 0.0, 0.0), [4.0, 4.0], 4, 5, kw)]
    assert _seq(calls) == [False, expected]


def test_oracle_nan_in_fom_history_is_skipped_by_max_min():
    """MATLAB max()/min() ignore NaN, so the window is judged on the rest.

    COM Octave, FOM_history = [4 NaN 4.5] on iteration 3: skip_it=1.  An
    all-NaN window gives NaN-NaN = NaN, which is not < 0.002, so it counts as
    improvement: skip_it=0.
    """
    assert _seq([_warm(), (4, (3.0, 0.0, 0.0), [4.0, float('nan'), 4.5],
                          3, 5, {})]) == [False, True]
    assert _seq([_warm(), (4, (3.0, 0.0, 0.0),
                           [float('nan'), float('nan')], 3, 5, {})]) \
        == [False, False]


def test_oracle_ctle_index_one_weight_branch():
    """ctle_index == 1 takes the unscaled lp/vga weights.

    COM Octave, ctle_index=1 with BEST.ctle=1 on iteration 3: skip_it=1.
    """
    assert _seq([_warm(), (4, (3.0, 0.0, 0.0), [4.0, 4.0], 3, 5,
                           {'ctle_index': 1, 'best_ctle': 1})]) == [False, True]


def test_oracle_vga_index_is_forced_to_one_on_4p16p0():
    """ML 4p16p0 writes THIS.vga_index = BEST.vga_index = 1 before reading them.

    COM Octave with the caller setting THIS.vga_index=9 and BEST.vga_index=7:
    skip_it=1 on iteration 3, the same as if neither had been set, because
    both assignments are local to the function.
    """
    reset_state()
    OptFom_Adaptive_Local_Search(4, *_oct_mk((0.0, 0.0, 0.0)), [1.0], 1, 5,
                                 **_OCT_VER)
    BEST, THIS = _oct_mk((3.0, 0.0, 0.0))
    BEST.vga_index, THIS.vga_index = 7, 9
    assert OptFom_Adaptive_Local_Search(4, BEST, THIS, [4.0, 4.0], 3, 5,
                                        **_OCT_VER) is True
    # and the caller's structs are not written through
    assert BEST.vga_index == 7 and THIS.vga_index == 9


@pytest.mark.parametrize('LSV,taps,kw', [
    (2.5, (2.0, 0.0, 0.0), {}),                       # round() tie on the radius
    (-3, (2.0, 0.0, 0.0), {}),                        # negative LocalSearch_Value
    (6, (3.0, 0.0, 0.0), {'Overwrite_Min_Radius': 2.5}),  # fractional floor
])
def test_oracle_odd_radius_inputs_do_not_skip(LSV, taps, kw):
    """COM Octave returns skip_it = 0, 0 for each of these two-call runs."""
    assert _seq([(LSV, taps, [1.0], 1, 5, kw),
                 (LSV, taps, [4.0, 4.0], 2, 5, kw)]) == [False, False]


def test_oracle_inlined_logger_matches_append_csv_row(tmp_path):
    """The inlined _append_csv_row follows the corrected canonical function.

    The CTLE-too-far return leaves hard_cap unset, so that column logs NaN.
    COM Octave, append_csv_row(f, {'h'}, {NaN}) writes NaN; the inlined copy
    wrote Python's lower-case nan.  skip_it must stay numeric: MATLAB logs
    double(skip_it), and isnumeric() is false for a logical, so passing the
    bool through the corrected formatter would write "" instead of 1.
    """
    log = str(tmp_path / 'ALS_log.csv')
    py_impl.ALS_LOG_CSV = log
    try:
        reset_state()
        BEST, THIS = _oct_mk((1.0, 0.0, 0.0), ctle_index=9, best_ctle=3)
        assert OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0], 1, 5,
                                            **_OCT_VER) is True
    finally:
        py_impl.ALS_LOG_CSV = None
    header, row = open(log).read().splitlines()
    cells = dict(zip(header.split(','), row.split(',')))
    assert cells['hard_cap'] == 'NaN'
    assert cells['skip_it'] == '1'
