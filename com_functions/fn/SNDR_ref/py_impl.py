import numpy as np
from types import SimpleNamespace


def _FFE(C, cmx, spui, V):
    """Inline copy of FFE (MATLAB lines 2026-2048). cmx is 0-based cursor index."""
    C = np.asarray(C, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.ndim == 2 and V.shape[1] == 1:
        V = V.ravel()
    V0 = 0.0
    for i, c in enumerate(C):
        if c != 0:
            ishift = (i - cmx) * spui
            V0 = np.roll(V, ishift) * c + V0
    return V0


def SNDR_ref(PR_Ref, param):
    """Compute SNDR for 6 TX-FFE presets (MATLAB lines 4405-4442).

    Returns results struct with SNDR_ref (array), SNDR_ref_p1..p6, sigma_iL.
    """
    PR_Ref = np.asarray(PR_Ref, dtype=float).ravel()

    if not hasattr(param, 'preset') or param.preset is None or len(param.preset) == 0:
        param.preset = [
            SimpleNamespace(txffe=[0, 0, 0, 1, 0]),
            SimpleNamespace(txffe=[0, 0, 0, 0.5, 0]),
            SimpleNamespace(txffe=[0, 0, -0.075, 0.75, 0]),
            SimpleNamespace(txffe=[0, 0.05, -0.20, 0.75, 0]),
            SimpleNamespace(txffe=[-0.025, 0.075, -0.25, 0.65, 0]),
            SimpleNamespace(txffe=[0, 0, 0, 0.75, 0]),
        ]

    def ss(a): return float(np.sum(np.abs(np.asarray(a).ravel()) ** 2))

    SNR_TX = float(np.asarray(param.SNDR).ravel()[0])
    M = int(param.samples_per_ui)
    D_p = int(param.D_p)
    N_p = int(param.N_p)
    PR_noFFE = PR_Ref.copy()

    ipeak_0 = int(np.argmax(PR_noFFE))  # 0-based
    istart_0 = ipeak_0 % M              # 0-based start for subsampling
    iend_0 = (len(PR_noFFE) // M) * M  # exclusive end for Python slice

    PR_noFFE_sampled = PR_noFFE[istart_0:iend_0:M]
    sigma_tn_base = ss(PR_noFFE_sampled)

    n_presets = len(param.preset)
    SNDR_ref_arr = np.zeros(n_presets)
    sigma_iL_arr = np.zeros(n_presets)

    for ipst, preset in enumerate(param.preset):
        PR_FFE = _FFE(preset.txffe, 3, M, PR_noFFE)  # cmx=3: 0-based tap 4
        ipeak_0_ffe = int(np.argmax(PR_FFE))
        start_0 = -D_p * M + ipeak_0_ffe
        end_0 = N_p * M + ipeak_0_ffe
        hss = PR_FFE[start_0:end_0 + 1:M]
        ss_hss = ss(hss)
        sigma_iL_arr[ipst] = float(np.sqrt(ss_hss))
        sigma_ts = ss_hss * 10 ** (SNR_TX / 10)
        SNDR_ref_arr[ipst] = 10 * np.log10(sigma_ts / sigma_tn_base)

    results = SimpleNamespace(
        SNDR_ref=SNDR_ref_arr,
        SNDR_ref_p1=float(SNDR_ref_arr[0]),
        SNDR_ref_p2=float(SNDR_ref_arr[1]),
        SNDR_ref_p3=float(SNDR_ref_arr[2]),
        SNDR_ref_p4=float(SNDR_ref_arr[3]),
        SNDR_ref_p5=float(SNDR_ref_arr[4]),
        SNDR_ref_p6=float(SNDR_ref_arr[5]),
        sigma_iL=float(sigma_iL_arr[0]),
    )
    return results
