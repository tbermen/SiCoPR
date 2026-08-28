# ============================================================
# MATLAB→Python translation notes for pdf2sgm
# MATLAB lines: 8846–8848
# ============================================================
# 1-based vs 0-based indexing: not applicable.
# Element-wise ops: .* and .^2 map directly to * and **2 on NumPy arrays.
# Output shape: scalar float.
# Formula: weighted mean then weighted standard deviation of the PDF.
#   avg = E[x]     = sum(x * y)
#   sgm = std(x)   = sqrt(sum((x - avg)^2 * y))
#   (This is the RMS voltage spread, i.e., std dev of the PDF.)
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
import numpy as np


def pdf2sgm(pdf):
    """Return the standard deviation (sigma) of a PDF struct.

    Computes:
      avg = sum(pdf.x * pdf.y)          weighted mean
      sgm = sqrt(sum((pdf.x-avg)^2 * pdf.y))  weighted std dev
    """
    x = np.asarray(pdf.x, dtype=float)
    y = np.asarray(pdf.y, dtype=float)
    avg = np.sum(x * y)                       # MATLAB line 8847
    sgm = np.sqrt(np.sum((x - avg) ** 2 * y)) # MATLAB line 8848
    return float(sgm)


if __name__ == "__main__":
    from types import SimpleNamespace
    import numpy as np
    p = SimpleNamespace()
    p.x = np.array([-1.0, 0.0, 1.0])
    p.y = np.array([0.25, 0.50, 0.25])
    print("pdf2sgm =", pdf2sgm(p))  # expected: sqrt(0.5) ≈ 0.7071
