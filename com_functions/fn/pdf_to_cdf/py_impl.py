# ============================================================
# MATLAB→Python translation notes for pdf_to_cdf
# MATLAB lines: 8853–8864
# ============================================================
# 1-based vs 0-based indexing: not applicable.
# cumsum direction: MATLAB cumsum on a row vector is element-wise left-to-
#   right — same as np.cumsum on a 1-D array.
# fliplr: MATLAB fliplr reverses a row vector; np.flip does the same for
#   1-D arrays.  (fliplr on a 2-D matrix reverses columns; that case does
#   not arise here because pdf.y is always 1-D.)
# min([yB(:) yT(:)], [], 2): MATLAB stacks two column vectors into n×2,
#   then takes the minimum across each row.  Equivalent to np.minimum(yB, yT).
# Output shape: cdf.y is 1-D (MATLAB produces a column vector, which we
#   store as a flat NumPy array consistent with our pdf convention).
# Known discrepancy from prior com.py attempt: none found.
# ============================================================
from types import SimpleNamespace
import numpy as np


def pdf_to_cdf(pdf):
    """Transform a PDF struct to a combined top/bottom CDF struct.

    Returns a SimpleNamespace with fields:
      cdf.yB  — left-to-right cumulative sum (bottom eye)
      cdf.yT  — right-to-left cumulative sum (top eye)
      cdf.y   — element-wise minimum of yB and yT
      cdf.x   — x-axis copied from pdf.x
    """
    cdf = SimpleNamespace()
    y = np.asarray(pdf.y, dtype=float).ravel()  # ensure 1-D

    cdf.yB = np.cumsum(y)                              # MATLAB line 8861
    cdf.yT = np.flip(np.cumsum(np.flip(y)))            # MATLAB line 8862
    cdf.y  = np.minimum(cdf.yB, cdf.yT)               # MATLAB line 8863
    cdf.x  = pdf.x                                     # MATLAB line 8864
    return cdf


if __name__ == "__main__":
    from types import SimpleNamespace
    import numpy as np
    p = SimpleNamespace()
    p.y = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    p.x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    c = pdf_to_cdf(p)
    print("yB =", c.yB)
    print("yT =", c.yT)
    print("y  =", c.y)
