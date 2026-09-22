# ============================================================
# MATLAB→Python translation notes for normal_dist
# MATLAB lines: 8410–8416
# ============================================================
# 1-based vs 0-based indexing: not applicable (range arithmetic only).
# Column-major vs row-major: MATLAB (pdf.Min:-pdf.Min) produces a row
#   vector; np.arange produces a 1-D array — equivalent for our purposes.
# MATLAB `end` keyword: not used.
# Output shape: pdf.x and pdf.y are 1-D arrays of length 2*|pdf.Min|+1.
# eps: MATLAB `eps` == np.finfo(float).eps (machine epsilon, ~2.22e-16).
# round() difference: MATLAB rounds 0.5 away from zero; Python rounds to
#   even. Inputs to round() here are 2*nsigma*sigma/binsize — in practice
#   these are integers or clearly non-half values, so the difference is moot.
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
from types import SimpleNamespace
import numpy as np

def _mround(x):
    """MATLAB round(): half away from zero, where Python's round() is banker's."""
    x = float(x)
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    # Off a tie round() is exact, and unlike floor(x + 0.5) it does not
    # send 0.49999999999999994 to 1: that sum is exactly 1.0 in binary.
    return int(round(x))



def normal_dist(sigma, nsigma, binsize):
    """Return a normalised Gaussian PDF struct.

    Fields:
      pdf.BinSize  — bin width (= binsize)
      pdf.Min      — negative integer; x-axis starts at pdf.Min*binsize
      pdf.x        — voltage axis, shape (2*|pdf.Min|+1,)
      pdf.y        — normalised Gaussian weights, sum == 1
    """
    pdf = SimpleNamespace()
    pdf.BinSize = binsize
    # pdf.Min is the (negative) index of the leftmost bin
    pdf.Min = -_mround(2 * nsigma * sigma / binsize)  # MATLAB line 8412

    # MATLAB: (pdf.Min:-pdf.Min) is range [pdf.Min, ..., -pdf.Min] step 1
    # np.arange upper bound is exclusive, so +1 is needed
    pdf.x = np.arange(pdf.Min, -pdf.Min + 1) * binsize  # MATLAB line 8413

    eps = np.finfo(float).eps
    pdf.y = np.exp(-pdf.x ** 2 / (2 * sigma ** 2 + eps))  # MATLAB line 8414
    pdf.y = pdf.y / np.sum(pdf.y)                          # MATLAB line 8415
    return pdf


if __name__ == "__main__":
    p = normal_dist(0.01, 5, 0.001)
    print(f"BinSize={p.BinSize}, Min={p.Min}, len(x)={len(p.x)}, sum(y)={sum(p.y):.6f}")
