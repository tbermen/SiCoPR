# ============================================================
# MATLAB→Python translation notes for FFE_Fast
# MATLAB lines: 2049–2062
# ============================================================
# V_shift: 2D matrix (N_samples × N_taps); each column is a pre-shifted
#   version of the signal (circshift already applied by caller).
# MATLAB V_shift(:,i): column i → Python V_shift[:, i] (0-based) ✓
# Skip zero taps: avoids multiply-by-zero accumulations.
# V0 starts as scalar 0; first non-zero accumulation produces an array
#   via 0 + array = array (numpy broadcasts scalar 0 correctly).
# 1-based MATLAB loop i=1:N → Python enumerate i=0..N-1 ✓
# ============================================================

import numpy as np


def FFE_Fast(C, V_shift):
    C = np.asarray(C, dtype=float)
    V_shift = np.asarray(V_shift, dtype=float)
    V0 = 0.0
    for i, c in enumerate(C):
        if c != 0:
            V0 = V_shift[:, i] * c + V0
    return V0


if __name__ == "__main__":
    C = np.array([0.5, 1.0, 0.0, -0.2])
    V_shift = np.column_stack([np.arange(4)*k for k in [1, 2, 3, 4]])
    print("V0:", FFE_Fast(C, V_shift))
