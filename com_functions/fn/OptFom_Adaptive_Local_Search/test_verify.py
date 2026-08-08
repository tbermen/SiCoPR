# ============================================================
# MATLAB GROUND TRUTH for OptFom_Adaptive_Local_Search (L2739-2992)
# ------------------------------------------------------------
# Decision branches (LocalSearch_Value=2, min_radius=1 forced,
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
    # iter 10 -> deterministic_radius = round(2/(1+1.5)) = 1, so adaptive_radius=1.
    # taps [1,1,0] vs [0,0,0]: raw_L1_TX=2 (==hard_cap, not >), L1_w=2>1, L2_w=1.41>1 -> skip.
    BEST, THIS = _mk([0, 0, 0], [1, 1, 0], ctle_index=3, best_ctle=3)
    assert OptFom_Adaptive_Local_Search(2, BEST, THIS, [4.0, 4.0, 4.0], 10, 5) is True


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
