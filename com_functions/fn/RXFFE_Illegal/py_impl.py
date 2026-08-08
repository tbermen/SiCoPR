import numpy as np


def RXFFE_Illegal(C, param, last_index=None):
    """Return 1 if RxFFE tap vector C violates parameter constraints, else 0.

    C and all tap-limit parameters use 0-based Python indexing.
    param.RxFFE_cmx is the 0-based cursor position (MATLAB: RxFFE_cmx+1 was 1-based).
    last_index is the exclusive upper bound for postcursor tap checks (default len(C)).
    """
    C = np.asarray(C, dtype=float)
    if last_index is None:
        last_index = len(C)
    cursor = int(param.RxFFE_cmx)  # 0-based cursor index

    if C[cursor] < param.ffe_main_cursor_min:
        return 1

    if param.ffe_post_tap_len != 0:
        if abs(C[cursor + 1]) > param.ffe_post_tap1_max:
            return 1
        if param.ffe_post_tap_len > 1:
            if np.any(np.abs(C[cursor + 2:last_index]) > param.ffe_tapn_max):
                return 1

    if param.ffe_pre_tap_len != 0:
        if abs(C[cursor - 1]) > param.ffe_pre_tap1_max:
            return 1
        if param.ffe_pre_tap_len > 1:
            if np.any(np.abs(C[:cursor - 1]) > param.ffe_tapn_max):
                return 1

    return 0
