# ============================================================
# MATLAB GROUND TRUTH for compute_hard_cap (L5788-5794)
#   compute_hard_cap(true, 1.2, 2, 1) -> max(1, round(2.4)) = 2
#   compute_hard_cap(true, 1.2, 5, 2) -> max(2, round(6.0)) = 6
#   compute_hard_cap(true, 0.1, 2, 3) -> max(3, round(0.2)) = 3  (floor by min_radius)
#   compute_hard_cap(false, ...)      -> NaN
#   round() is half-away-from-zero: round(2.5)=3
# ============================================================
import os
import sys
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.compute_hard_cap.py_impl import compute_hard_cap


def test_nominal():
    assert compute_hard_cap(True, 1.2, 2, 1) == 2


def test_min_radius_floor():
    assert compute_hard_cap(True, 0.1, 2, 3) == 3


def test_larger_values():
    assert compute_hard_cap(True, 1.2, 5, 2) == 6


def test_round_half_away():
    # round(2.5) -> 3 (MATLAB), not 2 (banker's)
    assert compute_hard_cap(True, 0.5, 5, 1) == 3


def test_disabled_returns_nan():
    assert math.isnan(compute_hard_cap(False, 1.2, 2, 1))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))


# ============================================================
# Divergences found by EXECUTING the reference (COM Octave oracle,
# tools/octave_oracle.py). Octave values pinned as literals.
#
# MATLAB round(NaN) is NaN and round(Inf) is Inf, and MATLAB max() DROPS a
# NaN operand. Python's int() raised on NaN/Inf, and builtin max() keeps a
# NaN it sees first. MATLAB's `if X` is also false for an empty X, where
# numpy raises. LocalSearch_Value reaching here as NaN is what makes this
# live: the reference caps and carries on, the port aborted the run.
# ============================================================

def test_nan_local_search_value():
    """COM Octave: compute_hard_cap(1, 1.2, NaN, 1) -> 1.
    Python raised ValueError: cannot convert float NaN to integer."""
    assert compute_hard_cap(True, 1.2, float('nan'), 1) == 1


def test_nan_multiplier():
    """COM Octave: compute_hard_cap(1, NaN, 2, 3) -> 3."""
    assert compute_hard_cap(True, float('nan'), 2, 3) == 3


def test_nan_min_radius_is_dropped():
    """COM Octave: compute_hard_cap(1, 1.2, 2, NaN) -> 2.
    builtin max(nan, 2) kept the nan."""
    assert compute_hard_cap(True, 1.2, 2, float('nan')) == 2


def test_all_nan_stays_nan():
    """COM Octave: compute_hard_cap(1, NaN, NaN, NaN) -> NaN."""
    assert math.isnan(compute_hard_cap(True, float('nan'),
                                       float('nan'), float('nan')))


def test_infinite_local_search_value():
    """COM Octave: LSV=Inf -> Inf, LSV=-Inf -> min_radius (1).
    Python raised OverflowError on both."""
    assert math.isinf(compute_hard_cap(True, 1.2, float('inf'), 1))
    assert compute_hard_cap(True, 1.2, float('-inf'), 1) == 1


def test_empty_use_hard_cap_is_false():
    """COM Octave: `if []` is false, so compute_hard_cap([], 1.2, 2, 1) -> NaN.
    numpy raised: truth value of an empty array is ambiguous."""
    import numpy as np
    assert math.isnan(compute_hard_cap(np.array([]), 1.2, 2, 1))


def test_round_half_away_from_zero_negative():
    """COM Octave: compute_hard_cap(1, -0.5, 5, -10) -> -3, i.e. round(-2.5)
    is -3, not the -2 that half-to-even gives."""
    assert compute_hard_cap(True, -0.5, 5, -10) == -3
