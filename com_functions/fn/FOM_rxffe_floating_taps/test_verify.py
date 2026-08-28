"""Verification tests for FOM_rxffe_floating_taps().

# ============================================================
# MATLAB GROUND TRUTH (lines 2063-2108)
# Greedy floating-tap bank placement: choose N_bg banks (N_bf taps each), one bank
# at a time, each at the location maximising MMSE_FOM over the candidate tap set.
# Returns sorted idx = all_idx + RxFFE_cpx (1-based, untrimmed-hisi positions).
# ============================================================
"""
import numpy as np
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.FOM_rxffe_floating_taps.py_impl import FOM_rxffe_floating_taps


# ---- 1. Greedy-selection logic (stub kernel) ------------------------------
def test_greedy_picks_highest_isi_banks():
    """With FOM = -(uncovered ISI energy), greedy must cover the largest banks."""
    RxFFE_cpx = 1
    isi = np.array([0.02, 0.40, 0.35, 0.03, 0.05, 0.30, 0.28, 0.02])

    def stub(param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2, idx,
                 **_kw):   # **_kw: tolerate the optional HH_full fast path
        covered = set(int(i) - RxFFE_cpx - 1 for i in np.atleast_1d(idx))
        resid = sum(isi[j] ** 2 for j in range(len(isi)) if j not in covered)
        return (0.0, -resid, None, idx, 0, None)

    param = SimpleNamespace(RxFFE_cpx=RxFFE_cpx, N_bmax=len(isi) + RxFFE_cpx, N_bf=2, N_bg=2)
    h = np.concatenate([np.zeros(RxFFE_cpx), isi])
    idx = FOM_rxffe_floating_taps(param, h, None, 1, None, 0, 0, None, None, None, None,
                                  1.0, 0, RxFFE_cpx + len(isi), _MMSE_FOM_fn=stub)
    covered0 = sorted(int(i) - RxFFE_cpx - 1 for i in idx)
    assert covered0 == [1, 2, 5, 6], covered0


def test_idx_convention_and_no_overlap():
    """idx is sorted, count == N_bg*N_bf, values are untrimmed-hisi 1-based positions."""
    RxFFE_cpx = 2
    isi = np.array([0.5, 0.1, 0.05, 0.4, 0.05, 0.05, 0.3, 0.02])

    def stub(param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2, idx,
                 **_kw):   # **_kw: tolerate the optional HH_full fast path
        covered = set(int(i) - RxFFE_cpx - 1 for i in np.atleast_1d(idx))
        return (0.0, -sum(isi[j] ** 2 for j in range(len(isi)) if j not in covered), None, idx, 0, None)

    N_bf, N_bg = 1, 3
    param = SimpleNamespace(RxFFE_cpx=RxFFE_cpx, N_bmax=len(isi) + RxFFE_cpx, N_bf=N_bf, N_bg=N_bg)
    h = np.concatenate([np.zeros(RxFFE_cpx), isi])
    idx = FOM_rxffe_floating_taps(param, h, None, 1, None, 0, 0, None, None, None, None,
                                  1.0, 0, RxFFE_cpx + len(isi), _MMSE_FOM_fn=stub)
    assert list(idx) == sorted(idx)
    assert len(idx) == N_bg * N_bf
    assert all(RxFFE_cpx + 1 <= v <= RxFFE_cpx + len(isi) for v in idx)
    # single-tap banks pick the three largest ISI: positions 0,3,6 -> +RxFFE_cpx+1
    assert sorted(int(i) - RxFFE_cpx - 1 for i in idx) == [0, 3, 6]


# ---- 2. Integration with the assembled real MMSE_FOM ----------------------
def test_integration_real_mmse_fom():
    """End-to-end: real sicopr.MMSE_FOM selects valid floating taps and accepts them."""
    import importlib
    sicopr = importlib.import_module('sicopr')

    RxFFE_cmx, RxFFE_cpx = 2, 2
    N_bf, N_bg, N_bmax, ndfe = 1, 2, 6, 1
    dw = RxFFE_cmx
    Nw = dw + N_bmax + 1            # 9
    dh = 2
    d = dw + dh                    # 4
    param = SimpleNamespace(
        RxFFE_cmx=RxFFE_cmx, RxFFE_cpx=RxFFE_cpx, N_bf=N_bf, N_bg=N_bg, N_bmax=N_bmax,
        ndfe=ndfe, R_LM=1.0, levels=4, bmax=np.array([0.7]), bmin=np.array([-0.7]),
    )
    sigma_X2 = (param.levels ** 2 - 1) / (3 * (param.levels - 1) ** 2)

    h = np.array([0.05, 0.1, 1.0, 0.3, 0.05, 0.4, 0.35, 0.05, 0.3, 0.02, 0, 0, 0, 0, 0, 0])
    from scipy.linalg import toeplitz
    hc1 = np.concatenate([h, np.zeros(Nw - 1)])
    hr1 = np.concatenate([[h[0]], np.zeros(Nw - 1)])
    H = toeplitz(hc1, hr1)
    Rn = np.exp(-np.arange(Nw))    # decaying -> PD Toeplitz
    Rnn = toeplitz(Rn[:Nw], Rn[:Nw])
    wmax = np.ones(20) * 10.0
    wmin = -np.ones(20) * 10.0
    isi_start, isi_end = dh + 1, (dh - dw) + Nw   # 3, 9

    idx = sicopr.FOM_rxffe_floating_taps(param, h, H, ndfe, Rnn, dw, d, wmax, wmin,
                                      param.bmin, param.bmax, sigma_X2, isi_start, isi_end)
    # valid: sorted, right count, columns within H
    assert list(idx) == sorted(idx) and len(idx) == N_bg * N_bf
    assert all(0 <= v + RxFFE_cmx < H.shape[1] for v in idx)
    # the real kernel must produce a finite FOM for the selected taps
    sigma_e, FOM, w, idx_out, Nw_out, blim = sicopr.MMSE_FOM(
        param, H, ndfe, Rnn, dw, d, wmax, wmin, param.bmin, param.bmax, sigma_X2, idx)
    assert np.isfinite(FOM), FOM
    # the biggest ISI taps (h indices 5,6 -> trimmed positions 0,1) should be chosen
    chosen_trimmed = sorted(int(v) - RxFFE_cpx - 1 for v in idx)
    assert chosen_trimmed[0] == 0, chosen_trimmed


if __name__ == '__main__':
    test_greedy_picks_highest_isi_banks()
    test_idx_convention_and_no_overlap()
    test_integration_real_mmse_fom()
    print('FOM_rxffe_floating_taps: ALL TESTS PASSED')
