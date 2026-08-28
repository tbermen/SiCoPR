# ============================================================
# MATLAB→Python translation notes for cdf_to_ber_contour
# MATLAB lines: 5246–5257
# ============================================================
# 1-based vs 0-based indexing: MATLAB find() returns 1-based indices.
#   np.argmax on a boolean mask returns the 0-based index of the first True.
#   Index conversion for the top crossing:
#     MATLAB: nidx_true = length(y) - nidx_flipped + 1  (1-based)
#     Python:  nidx_true = len(y) - 1 - nidx_flipped     (0-based)
#   where nidx_flipped is 0-based in the flipped array.
# fliplr(cdf.y(:)'): flatten to row then flip — np.flip(y.ravel()).
# find(..., 1, 'first'): np.argmax on boolean mask (returns 0 if no match;
#   MATLAB returns [] — callers must ensure specBER is in range).
# Output shape: two scalars (float).
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
import numpy as np


def cdf_to_ber_contour(cdf, specBER):
    """Find the bottom and top voltage crossings of specBER in the CDF.

    noise_bottom: leftmost x where cdf.y first exceeds specBER
    noise_top:    rightmost x where cdf.y last exceeds specBER
    """
    y = np.asarray(cdf.y, dtype=float).ravel()
    x = np.asarray(cdf.x, dtype=float).ravel()

    # Bottom eye: first index where cdf.y > specBER (MATLAB lines 5251-5252)
    nidx = int(np.argmax(y > specBER))           # 0-based
    noise_bottom = x[nidx]

    # Top eye: search flipped CDF for first crossing, then un-flip (lines 5254-5257)
    nidx_flipped = int(np.argmax(np.flip(y) > specBER))  # 0-based in flipped
    nidx = len(y) - 1 - nidx_flipped                      # 0-based in original
    noise_top = x[nidx]

    return float(noise_bottom), float(noise_top)


if __name__ == "__main__":
    from types import SimpleNamespace
    import numpy as np
    c = SimpleNamespace()
    c.y = np.array([0.1, 0.3, 0.7, 0.3, 0.1])
    c.x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("cdf_to_ber_contour(specBER=0.25):", cdf_to_ber_contour(c, 0.25))
    # expected: (-1.0, 1.0)
