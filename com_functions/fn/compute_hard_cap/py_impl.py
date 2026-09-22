# ============================================================
# MATLAB→Python translation notes for compute_hard_cap
# MATLAB lines: 5788-5794 (com_ieee8023_4p15p0_adaptive_local_search.m)
# NEW in Hansel D'silva's adaptive-local-search branch.
# ============================================================
# Tiny helper: the hard cap on the raw TX-tap L1 distance used by the
# adaptive local search. round() is MATLAB round (half away from zero).
# Returns NaN when capping is disabled, matching the MATLAB return.
# ============================================================

import numpy as np


def _mround(x):
    """MATLAB round(): half away from zero."""
    x = float(x)
    # MATLAB round(NaN) is NaN and round(Inf) is Inf; int() raises on both, so
    # a NaN local-search value used to abort here instead of being capped.
    # COM Octave: compute_hard_cap(1, 1.2, NaN, 1) -> 1
    #             compute_hard_cap(1, 1.2, Inf, 1) -> Inf
    if not np.isfinite(x):
        return x
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    # Off a tie round() is exact, and unlike floor(x + 0.5) it does not
    # send 0.49999999999999994 to 1: that sum is exactly 1.0 in binary.
    return int(round(x))


def _mmax2(a, b):
    """MATLAB max(a,b): a NaN operand is dropped, not propagated.

    Python's max() compares with > and keeps whichever it saw first, so
    max(NaN, 2) is NaN where MATLAB gives 2.
    COM Octave: compute_hard_cap(1, 1.2, 2, NaN) -> 2   (Python gave nan)
                compute_hard_cap(1, NaN, 2, 3)   -> 3   (Python raised)
                compute_hard_cap(1, NaN, NaN, NaN) -> NaN
    """
    if np.isnan(a):
        return b
    if np.isnan(b):
        return a
    return a if a > b else b


def compute_hard_cap(use_hard_cap, mul, LSV, min_radius):
    """Hard cap for the raw TX L1 distance (MATLAB lines 5788-5794)."""
    # MATLAB `if X` is false for an empty X and true only when every element is
    # non-zero; `if np.array([])` raises in numpy.
    # COM Octave: compute_hard_cap([], 1.2, 2, 1) -> NaN
    if np.size(use_hard_cap) and np.all(use_hard_cap):
        return _mmax2(min_radius, _mround(mul * LSV))
    return float('nan')


if __name__ == '__main__':
    print(compute_hard_cap(True, 1.2, 2, 1))   # max(1, round(2.4)) = 2
    print(compute_hard_cap(False, 1.2, 2, 1))  # nan
