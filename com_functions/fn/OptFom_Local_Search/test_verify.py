"""Verification tests for OptFom_Local_Search().

# ============================================================
# MATLAB GROUND TRUTH (lines 3618-3673)
# Returns skip_it=1 if any tx index differs from best by more
# than LocalSearch_Value steps. Also checks g_LP vs G_high_pass.
# ============================================================
"""
import numpy as np
import pytest
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


# ============================================================
# COM Octave oracle — OptFom_Local_Search run verbatim under Octave, with
#   BEST.txffe_index = [1 1 3 1 1]   BEST.G_high_pass = 1
#   THIS.tx_index_vector = [3 8 1 1 1]   ctle_index = 1   g_LP_index = 1
#   LocalSearch_Value = 2
#
#   txffe_sweep_indices = [2]      -> skip_it 1
#   txffe_sweep_indices = [2 3 4]  -> skip_it 0   (on the matching vectors)
#   txffe_sweep_indices = [1]      -> skip_it 0   (kv==1 reads g_LP_index)
#   txffe_sweep_indices = [2;3]    -> skip_it 1   (column, same as row)
#   txffe_sweep_indices = []       -> skip_it 0
#   LocalSearch_Value = 0          -> skip_it 1
#   LocalSearch_Value = -1, all indices equal -> skip_it 0
#   a NaN in tx_index_vector       -> skip_it 0   (NaN > 2 is false)
# ============================================================
def test_octave_oracle_grid_decisions():
    BEST = _BEST(txffe_index=[1, 1, 3, 1, 1], G_high_pass=1)
    THIS = _THIS(tx_index_vector=[3, 8, 1, 1, 1], ctle_index=1, g_LP_index=1)
    assert OptFom_Local_Search(2, BEST, THIS, [2]) == 1
    assert OptFom_Local_Search(2, BEST, THIS, [1]) == 0
    assert OptFom_Local_Search(2, BEST, THIS, [2, 3]) == 1
    assert OptFom_Local_Search(2, BEST, THIS, []) == 0
    assert OptFom_Local_Search(0, BEST, THIS, [2, 3]) == 1

    far = _THIS(tx_index_vector=[9, 1, 1, 1, 1], ctle_index=1, g_LP_index=4)
    assert OptFom_Local_Search(2, BEST, far, [1]) == 1

    flat = _THIS(tx_index_vector=[1, 1, 1, 1, 1], ctle_index=1, g_LP_index=1)
    assert OptFom_Local_Search(-1, _BEST(), flat, [2]) == 0

    nan = _THIS(tx_index_vector=[3, np.nan, 1, 1, 1], ctle_index=1, g_LP_index=1)
    assert OptFom_Local_Search(2, BEST, nan, [2]) == 0


# ============================================================
# COM Octave oracle — txffe_sweep_indices are MATLAB subscripts.
#
#   [0]   -> error: tx_index_vector(-1): subscripts must be either integers
#            1 to (2^63)-1 or logicals
#   [-1]  -> error: tx_index_vector(-2): subscripts must be ...
#   [2.5] -> error: tx_index_vector(1.5): subscripts must be ...
#   [9]   -> error: tx_index_vector(8): out of bound 5 (dimensions are 1x5)
#
# The port used int(kv), so 0 and -1 indexed from the end of the array and
# returned skip_it 0, and 2.5 became tap 2 and returned skip_it 1.
# ============================================================
@pytest.mark.parametrize('bad', [0, -1, 2.5, -0.5])
def test_non_subscript_sweep_index_refused(bad):
    BEST = _BEST(txffe_index=[1, 1, 3, 1, 1])
    THIS = _THIS(tx_index_vector=[3, 8, 1, 1, 1])
    with pytest.raises(IndexError):
        OptFom_Local_Search(2, BEST, THIS, [bad])


def test_sweep_index_past_the_end_refused():
    BEST = _BEST(txffe_index=[1, 1, 3, 1, 1])
    THIS = _THIS(tx_index_vector=[3, 8, 1, 1, 1])
    with pytest.raises(IndexError):
        OptFom_Local_Search(2, BEST, THIS, [9])


# ============================================================
# COM Octave oracle — which element each 1-based subscript reads.
#
# The three subscripts in the loop are tx_index_vector(kv-1) for the gate,
# tx_index_vector(kv) for the candidate and best_txffe_index(kv) for the
# reference, i.e. [kv-2], [kv-1] and [kv-1] in Python. Every earlier test in
# this file passes with the gate read one element late, so none of them can
# catch that slip. These three cases each flip on exactly one of the offsets:
#
#   best=[1 1 1 1 1] this=[1 1 9 1 1] sweep=[3] -> skip_it 0
#       gate is this(2)=1, so the far tap is never compared
#   best=[1 9 1 1 1] this=[5 1 4 1 1] sweep=[2] -> skip_it 1
#       reference is best(2)=9, not best(1)=1
#   best=[1 1 1 1 1] this=[5 1 9 1 1] sweep=[2] -> skip_it 0
#       candidate is this(2)=1, not this(3)=9
# ============================================================
def test_octave_oracle_gate_is_the_previous_tap():
    BEST = _BEST(txffe_index=[1, 1, 1, 1, 1])
    THIS = _THIS(tx_index_vector=[1, 1, 9, 1, 1])
    assert OptFom_Local_Search(2, BEST, THIS, [3]) == 0


def test_octave_oracle_reference_index_is_read_at_kv():
    BEST = _BEST(txffe_index=[1, 9, 1, 1, 1])
    THIS = _THIS(tx_index_vector=[5, 1, 4, 1, 1])
    assert OptFom_Local_Search(2, BEST, THIS, [2]) == 1


def test_octave_oracle_candidate_index_is_read_at_kv():
    BEST = _BEST(txffe_index=[1, 1, 1, 1, 1])
    THIS = _THIS(tx_index_vector=[5, 1, 9, 1, 1])
    assert OptFom_Local_Search(2, BEST, THIS, [2]) == 0
