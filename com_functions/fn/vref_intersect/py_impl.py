# ============================================================
# MATLAB→Python translation notes for vref_intersect
# MATLAB lines: 11366–11378
# ============================================================
# 1-based vs 0-based indexing: x_in is a 1-based MATLAB row index.
#   MATLAB eye_contour(x_in,   1) → Python eye_contour[x_in-1, 0]
#   MATLAB eye_contour(x_in-1, 1) → Python eye_contour[x_in-2, 0]
#   x_in is also used as a real coordinate in b1 = y - m1*x_in, so it
#   stays as-is in the arithmetic (1-based convention preserved throughout).
#   The returned line_intersection is therefore also in 1-based index space.
# Column indexing: eye_contour(:,1) in MATLAB = eye_contour[:,0] in Python.
# Output shape: scalar float (fractional index, 1-based).
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
import numpy as np


def vref_intersect(eye_contour, x_in, vref):
    """Find x where the eye contour linearly crosses vref.

    Uses the line through (x_in-1, y_prev) and (x_in, y_curr) to find
    the fractional 1-based index where y == vref.

    Parameters
    ----------
    eye_contour : 2-D array, first column holds y-values
    x_in        : 1-based integer index of the upper bracket point
    vref        : target voltage level
    """
    ec = np.asarray(eye_contour, dtype=float)
    x_in = int(x_in)

    y_curr = ec[x_in - 1, 0]          # MATLAB eye_contour(x_in, 1)
    y_prev = ec[x_in - 2, 0]          # MATLAB eye_contour(x_in-1, 1)

    m1 = y_curr - y_prev               # slope in index-space (step = 1)
    b1 = y_curr - m1 * x_in           # y-intercept of the contour line
    # m2 = 0, b2 = vref  →  intersection = (b2 - b1) / (m1 - m2)
    line_intersection = (vref - b1) / m1   # MATLAB line 11378
    return float(line_intersection)


if __name__ == "__main__":
    import numpy as np
    ec = np.array([[0.1], [0.3], [0.7], [0.9]])
    print("vref_intersect(ec, x_in=3, vref=0.5) =",
          vref_intersect(ec, 3, 0.5))   # expected 2.5
