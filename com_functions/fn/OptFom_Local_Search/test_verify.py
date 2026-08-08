"""Verification tests for OptFom_Local_Search().

# ============================================================
# MATLAB GROUND TRUTH (lines 3618-3673)
# Returns skip_it=1 if any tx index differs from best by more
# than LocalSearch_Value steps. Also checks g_LP vs G_high_pass.
# ============================================================
"""
import numpy as np
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.OptFom_Local_Search.py_impl import OptFom_Local_Search


def _BEST(txffe_index=None, G_high_pass=1):
    if txffe_index is None:
        txffe_index = [1, 1, 1, 1, 1]
    return SimpleNamespace(txffe_index=txffe_index, G_high_pass=G_high_pass)


def _THIS(tx_index_vector=None, ctle_index=1, g_LP_index=1):
    if tx_index_vector is None:
        tx_index_vector = [1, 1, 1, 1, 1]
    return SimpleNamespace(tx_index_vector=tx_index_vector, ctle_index=ctle_index, g_LP_index=g_LP_index)


def test_no_skip_close_enough():
    """When all indices match best, skip_it=0."""
    BEST = _BEST(txffe_index=[1, 2, 3, 1, 1])
    THIS = _THIS(tx_index_vector=[1, 2, 3, 1, 1])
    result = OptFom_Local_Search(2, BEST, THIS, [2, 3, 4])
    assert result == 0


def test_skip_when_too_far():
    """previous_loop_val>1 and current tap far from best → skip_it=1."""
    # kv=2: previous_loop_val = tx_index_vector[0]=3 (>1); current = tx_index_vector[1]=8
    # best_txffe_index[1]=1; |8-1|=7 > LocalSearch_Value=2 → skip
    BEST = _BEST(txffe_index=[1, 1, 1, 1, 1])
    THIS = _THIS(tx_index_vector=[3, 8, 1, 1, 1])
    result = OptFom_Local_Search(2, BEST, THIS, [2])
    assert result == 1


def test_skip_on_ctle_mismatch():
    """ctle_index>1 and g_LP differs from best_G_high_pass by >LocalSearch → skip."""
    BEST = _BEST(G_high_pass=1)
    THIS = _THIS(ctle_index=2, g_LP_index=5)
    result = OptFom_Local_Search(2, BEST, THIS, [])
    assert result == 1


def test_no_skip_empty_sweep_indices():
    """Empty txffe_sweep_indices with ctle_index=1 → no skip."""
    BEST = _BEST()
    THIS = _THIS()
    result = OptFom_Local_Search(2, BEST, THIS, [])
    assert result == 0


def test_kv1_uses_g_LP():
    """kv=1 branch uses g_LP_index, not tx_index_vector."""
    BEST = _BEST(txffe_index=[1, 1, 1, 1, 1])
    THIS = _THIS(tx_index_vector=[1, 1, 1, 1, 1], g_LP_index=1)
    result = OptFom_Local_Search(2, BEST, THIS, [1])
    assert result == 0
