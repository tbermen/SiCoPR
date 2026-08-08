# ============================================================
# MATLAB→Python translation notes for CDF_inv_ev
# MATLAB lines: 1142–1148
# ============================================================
# find(CDF >= val, 1, 'first'): 1-based index of first CDF ≥ val.
#   Python: np.where(CDF >= val)[0] → 0-based array of matching indices.
# isempty(index): no match → return PDF.x(end) = PDF.x[-1].
# else: return PDF.x(index) at 1-based index → PDF.x[index] at 0-based ✓
# ============================================================

import numpy as np
from types import SimpleNamespace


def CDF_inv_ev(val, PDF, CDF):
    x = np.asarray(PDF.x, dtype=float)
    CDF = np.asarray(CDF, dtype=float)
    indices = np.where(CDF >= val)[0]       # 0-based indices of CDF >= val
    if len(indices) == 0:
        return float(x[-1])                 # isempty(index) → PDF.x(end)
    return float(x[indices[0]])             # first match


if __name__ == "__main__":
    PDF = SimpleNamespace(x=np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]))
    CDF = np.array([0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 1.0])
    print("CDF_inv_ev(0.5):", CDF_inv_ev(0.5, PDF, CDF))    # 0.0
    print("CDF_inv_ev(1.5):", CDF_inv_ev(1.5, PDF, CDF))    # 0.3 (no match)
