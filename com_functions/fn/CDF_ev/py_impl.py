# ============================================================
# MATLAB→Python translation notes for CDF_ev
# MATLAB lines: 1139–1141
# ============================================================
# find(PDF.x >= -val, 1, 'first'): 1-based index of first element ≥ -val.
#   np.argmax(PDF.x >= -val): 0-based index of first True.
#   Both directly index CDF/PDF.x → equivalent result ✓
# CDF_ev = CDF(index): scalar probability at the voltage crossing.
# ============================================================

import numpy as np
from types import SimpleNamespace


def CDF_ev(val, PDF, CDF):
    x = np.asarray(PDF.x, dtype=float)
    CDF = np.asarray(CDF, dtype=float)
    index = int(np.argmax(x >= -val))   # first index where PDF.x >= -val
    return float(CDF[index])


if __name__ == "__main__":
    PDF = SimpleNamespace(x=np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]))
    CDF = np.array([0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 1.0])
    print("CDF_ev(0.15):", CDF_ev(0.15, PDF, CDF))   # 0.35
