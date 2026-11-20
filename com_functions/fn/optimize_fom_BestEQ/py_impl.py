# ============================================================
# MATLAB->Python translation of optimize_fom_BestEQ
# MATLAB lines: 10311-10356 (com_ieee8023_4p18p0.m; new in 4p18p0)
# ============================================================
# Re-runs optimize_fom with every sweep variable fixed at the best value of an
# earlier full sweep (Gffe, CTLE gain, g_DC_HP, itick, each Tx FFE tap), then
# puts back the fields that are indices into the full sweep. com_ieee8023_
# calls it for Defer_Floating_RxFFE and for Clip Method = Fast (4p18p0).
#
# MATLAB passes param by value, so the narrowed sweep lives in a copy; here
# param is deep-copied for the same reason. The caller's param keeps its grids,
# which Apply_EQ and the output then index with the restored indices.
#
# Reproduced from the reference as it stands:
#   - With ctle_gdc_values narrowed to one entry, optimize_fom reads the CTLE
#     poles as CTLE_fz(1), CTLE_fp1(1), CTLE_fp2(1): the FIRST entry's poles,
#     not the winner's. Harmless when the poles are one value for every gain.
#   - The re-run's eq_failed is not checked.
# ============================================================

import copy

import numpy as np


def optimize_fom_BestEQ(fom_result, OP, param, chdata, sigma_bn, do_C2M,
                        _optimize_fom_fn=None):
    fom_result_original = fom_result

    param = copy.deepcopy(param)
    param.cursor_gain = fom_result.best_current_ffegain
    # x(k) is 1x1 in MATLAB: keep a one-element array (len() is taken downstream)
    param.ctle_gdc_values = np.atleast_1d(
        np.asarray(param.ctle_gdc_values).ravel()[int(fom_result.ctle) - 1])
    param.g_DC_HP_values = np.atleast_1d(
        np.asarray(param.g_DC_HP_values).ravel()[int(fom_result.best_G_high_pass) - 1])
    param.ts_sample_adj_range = [fom_result.itick, fom_result.itick]

    best_txffe = np.asarray(fom_result.txffe).ravel()
    cur = int(fom_result.cur)                      # 1-based, as in the reference
    for k in range(1, cur):                        # num_pre = cur-1
        setattr(param, 'tx_ffe_cm%d_values' % k, best_txffe[cur - k - 1])
    for k in range(1, len(best_txffe) - cur + 1):  # num_post = length - cur
        setattr(param, 'tx_ffe_cp%d_values' % k, best_txffe[cur + k - 1])

    fom_result = _optimize_fom_fn(OP, param, chdata, sigma_bn, do_C2M)

    # indices into the original sweep
    fom_result.ctle = fom_result_original.ctle
    fom_result.best_G_high_pass = fom_result_original.best_G_high_pass
    fom_result.cur = fom_result_original.cur
    fom_result.txffe = fom_result_original.txffe
    return fom_result
