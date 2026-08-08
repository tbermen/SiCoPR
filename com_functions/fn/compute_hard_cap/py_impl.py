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
    return int(np.floor(float(x) + 0.5)) if x >= 0 else int(np.ceil(float(x) - 0.5))


def compute_hard_cap(use_hard_cap, mul, LSV, min_radius):
    """Hard cap for the raw TX L1 distance (MATLAB lines 5788-5794)."""
    if use_hard_cap:
        return max(min_radius, _mround(mul * LSV))
    return float('nan')


if __name__ == '__main__':
    print(compute_hard_cap(True, 1.2, 2, 1))   # max(1, round(2.4)) = 2
    print(compute_hard_cap(False, 1.2, 2, 1))  # nan
