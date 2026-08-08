# ============================================================
# MATLAB→Python translation notes for OptFom_Local_Search
# MATLAB lines: 3618–3673
# ============================================================
# txffe_sweep_indices uses 1-based MATLAB convention:
#   kv==1 → g_LP parameter; kv>1 → tx_index_vector(kv-1) in MATLAB = [kv-2] in Python.
#   tx_index_vector(kv) in MATLAB = [kv-1] in Python.
#   best_txffe_index(kv) in MATLAB = [kv-1] in Python.
# ============================================================

import numpy as np


def OptFom_Local_Search(LocalSearch_Value, BEST, THIS, txffe_sweep_indices):
    best_txffe_index = np.asarray(BEST.txffe_index)
    best_G_high_pass = BEST.G_high_pass
    tx_index_vector = np.asarray(THIS.tx_index_vector)
    ctle_index = THIS.ctle_index
    g_LP_index = THIS.g_LP_index

    skip_it = 0
    for kv in txffe_sweep_indices:
        kv = int(kv)
        if kv == 1:
            previous_loop_val = g_LP_index
        else:
            previous_loop_val = tx_index_vector[kv - 2]  # MATLAB (kv-1) 1-based → 0-based [kv-2]
        if previous_loop_val > 1:
            best_index_this_tap = best_txffe_index[kv - 1]  # MATLAB 1-based → 0-based
            if abs(tx_index_vector[kv - 1] - best_index_this_tap) > LocalSearch_Value:
                skip_it = 1
                break

    if not skip_it and ctle_index > 1 and abs(g_LP_index - best_G_high_pass) > LocalSearch_Value:
        skip_it = 1

    return skip_it
