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
