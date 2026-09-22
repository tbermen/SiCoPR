# ============================================================
# MATLAB→Python translation notes for pam
# MATLAB lines: 4213–4225
# ============================================================
# 1-based vs 0-based indexing:
#   MATLAB loop i=1:2:N (odd 1-based): data(i:i+1) is a 2-element slice.
#   Python 0-based: i_py = 0, 2, 4, ...  → data[i_py:i_py+2]
#   MATLAB output index ceil(i/2) with i=1→1, 3→2, 5→3 (1-based)
#   Python output index: i_py // 2  (0-based)
# Loop bound: floor(length(data)/2)*2 ensures only complete pairs processed.
#   Python: range(0, (len(data)//2)*2, 2)
# Grey-coded PAM4 mapping:
#   [-1,-1] → -1,  [-1, 1] → -1/3,  [1, 1] → 1/3,  [1,-1] → 1
# Output shape: 1-D array of length floor(len(data)/2).
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
import numpy as np

_PAM_MAP = {
    (-1, -1): -1.0,
    (-1,  1): -1.0 / 3,
    ( 1,  1):  1.0 / 3,
    ( 1, -1):  1.0,
}


def pam(data):
    """Grey-coded PAM4 mapping: pairs of NRZ bits → PAM4 symbols.

    Input:  1-D array of ±1 values (NRZ bits), length must be even.
    Output: 1-D array of PAM4 symbols in {-1, -1/3, 1/3, 1}.
    """
    data = np.asarray(data, dtype=float).ravel()
    n_pairs = len(data) // 2

    # MATLAB assigns dataout(ceil(i/2)) only inside the four if/elseif arms. A
    # pair that matches none leaves that slot UNASSIGNED, and MATLAB's
    # auto-grow then fills it with 0 -- but only if some LATER index is
    # assigned, because the array only ever grows to the highest assigned
    # index. Verified against Octave:
    #     pam([0 0 1 1]) -> [0 1/3]      (slot 1 back-filled with 0)
    #     pam([1 1 0 0]) -> [1/3]        (length 1, NOT 2)
    #     pam([1]), pam([]) -> error: value on right hand side is undefined
    assigned = {}
    for k in range(n_pairs):                    # k = i_py // 2
        i = k * 2                               # 0-based start of pair
        key = (data[i], data[i + 1])            # exact ±1 comparison
        if key in _PAM_MAP:
            assigned[k] = _PAM_MAP[key]

    if not assigned:
        raise ValueError(
            'pam: no input pair matched a Grey-code symbol, so MATLAB never '
            'assigns dataout and errors with "Output argument dataout (and '
            'maybe others) not assigned". Got %d sample(s).' % data.size)

    dataout = np.zeros(max(assigned) + 1, dtype=float)
    for k, v in assigned.items():
        dataout[k] = v
    return dataout


if __name__ == "__main__":
    import numpy as np
    bits = np.array([-1, -1, -1, 1, 1, 1, 1, -1])
    print("pam([-1,-1,-1,1,1,1,1,-1]) =", pam(bits))
    # expected: [-1, -1/3, 1/3, 1]
