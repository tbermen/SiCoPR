# ============================================================
# MATLAB→Python translation notes for hrem
# MATLAB lines: 7939–7941
# ============================================================
# 1-based vs 0-based indexing: `index` is 1-based in MATLAB.
#   Python 0-based start = index - 1.
#   MATLAB h(1:index-1)         → Python h[0:index-1]
#   MATLAB h(index:index+N_bf-1)→ Python h[index-1:index-1+N_bf]
#   MATLAB h(index+N_bf:end)    → Python h[index-1+N_bf:]
# Operation on middle segment: x - sign(x)*min(bmaxg, |x|)
#   = soft-thresholding toward zero with threshold bmaxg.
#   If |x| <= bmaxg  → result is 0.
#   If |x| >  bmaxg  → magnitude reduced by bmaxg, sign preserved.
# Output shape: 1-D array, same length as h.
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
import numpy as np


def hrem(h, index, N_bf, bmaxg):
    """Remove up to bmaxg from N_bf tap values starting at 1-based index.

    Elements outside [index, index+N_bf-1] are unchanged.
    Middle segment: x → x - sign(x)*min(bmaxg, |x|)  (shrink toward zero).
    """
    h = np.asarray(h, dtype=float).ravel()
    i = int(index) - 1                        # convert 1-based → 0-based

    seg = h[i:i + N_bf]
    shrunk = seg - np.sign(seg) * np.minimum(bmaxg, np.abs(seg))  # MATLAB line 7941

    out = np.concatenate([h[:i], shrunk, h[i + N_bf:]])
    return out


if __name__ == "__main__":
    import numpy as np
    h = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    print("hrem([0,1,2,3,4], index=2, N_bf=3, bmaxg=1.5) =",
          hrem(h, 2, 3, 1.5))
    # expected: [0, 0, 0.5, 1.5, 4]
