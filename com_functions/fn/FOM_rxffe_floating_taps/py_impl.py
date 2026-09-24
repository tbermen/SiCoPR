# ============================================================
# MATLAB→Python translation of FOM_rxffe_floating_taps
# MATLAB lines: 2063–2108
# ============================================================
# Greedy floating-tap bank placement for the RxFFE MMSE optimiser.
# Selects param.N_bg banks (each param.N_bf taps wide) one bank at a time,
# choosing each bank's start location to maximise the MMSE figure of merit
# (MMSE_FOM evaluated with the candidate tap set).
#
# Indexing (faithful to MATLAB):
#   hisi = h(isi_start:isi_end) then hisi(RxFFE_cpx+1:N_bmax)   (1-based MATLAB)
#     -> Python: h[isi_start:isi_end] then [RxFFE_cpx:N_bmax]   (isi_start/end already
#        passed in 0-based form by the MMSE caller)
#   valid_tap_locations: 1-based positions in the TRIMMED hisi (1..max_isi)
#   returned idx = sort(all_idx + RxFFE_cpx): 1-based positions in the UNtrimmed hisi,
#     matching findbankloc and MMSE_FOM column selection (cols = idx + RxFFE_cmx + 1
#     in MATLAB, idx + RxFFE_cmx in 0-based Python).
#
# MMSE_FOM is supplied via _MMSE_FOM_fn injection (the MMSE caller passes its own
# inlined _MMSE_FOM so the selection and the final solve use the same kernel).  In
# the assembled module it falls back to the top-level MMSE_FOM by bare name.
# ============================================================

import numpy as np


def FOM_rxffe_floating_taps(param, h, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax,
                            sigma_X2, isi_start, isi_end, _MMSE_FOM_fn=None):
    mmse_fom = _MMSE_FOM_fn if _MMSE_FOM_fn is not None else MMSE_FOM  # noqa: F821
    # H.T laid out contiguously, once for the whole bank search: MMSE_FOM then
    # gathers each candidate's columns as contiguous rows. Layout only -- the
    # Gram matrix is still H(:,sel)'*H(:,sel), formed per candidate, and
    # bit-identical to gathering from H.
    _H = np.asarray(H, dtype=float) if H is not None else None
    Ht = np.ascontiguousarray(_H.T) if _H is not None and _H.ndim == 2 else None

    h = np.asarray(h, dtype=float).ravel()
    RxFFE_cpx = int(param.RxFFE_cpx)
    N_bmax = int(param.N_bmax)
    bank_size = int(param.N_bf)
    num_groups = int(param.N_bg)

    hisi = h[isi_start:isi_end]
    hisi = hisi[RxFFE_cpx:N_bmax]
    num_isi = len(hisi)
    max_isi = num_isi - bank_size + 1

    valid = list(range(1, max_isi + 1))      # 1-based candidate bank-start locations
    all_idx = []                             # chosen tap locations (1-based, trimmed hisi)
    for _g in range(num_groups):
        if not valid:
            break
        best_FOM = np.full(len(valid), -np.inf)
        for k, loc in enumerate(valid):
            cand = sorted(all_idx + list(range(loc, loc + bank_size)))
            cand_idx = np.array(cand, dtype=int) + RxFFE_cpx
            if Ht is None:
                res = mmse_fom(param, H, Nb, Rnn, dw, d,
                               wmax, wmin, bmin, bmax, sigma_X2, cand_idx)
            else:
                res = mmse_fom(param, H, Nb, Rnn, dw, d,
                               wmax, wmin, bmin, bmax, sigma_X2, cand_idx, Ht=Ht)
            best_FOM[k] = res[1]             # FOM is the 2nd return value
        best_pos = int(np.argmax(best_FOM))  # 0-based position in valid
        start_tap = valid[best_pos]
        all_idx = all_idx + list(range(start_tap, start_tap + bank_size))
        # remove the chosen bank's positions (by index into valid)
        remove_pos = set(range(best_pos, min(best_pos + bank_size, len(valid))))
        # remove overlapping start locations (by value): start_tap-bank_size+1 .. start_tap-1
        bad_taps = set(t for t in range(start_tap - bank_size + 1, start_tap) if t >= 1)
        valid = [v for p, v in enumerate(valid)
                 if p not in remove_pos and v not in bad_taps]

    idx = np.sort(np.array(all_idx, dtype=int) + RxFFE_cpx)
    return idx


if __name__ == '__main__':
    from types import SimpleNamespace

    # ---- Unit test: greedy bank selection picks the highest-ISI locations ----
    # Stub MMSE_FOM: FOM = -(residual ISI energy at locations NOT covered by idx).
    # idx are 1-based untrimmed-hisi positions; map back to trimmed positions via
    # (idx - RxFFE_cpx).  Greedy should therefore choose banks over the big-ISI taps.
    RxFFE_cpx = 1
    isi_profile = np.array([0.02, 0.40, 0.35, 0.03, 0.05, 0.30, 0.28, 0.02])  # trimmed hisi tail

    def stub_MMSE_FOM(param, H, Nb, Rnn, dw, d, wmax, wmin, bmin, bmax, sigma_X2, idx):
        covered = set(int(i) - RxFFE_cpx - 1 for i in np.atleast_1d(idx))  # 0-based trimmed
        residual = sum(isi_profile[j] ** 2 for j in range(len(isi_profile)) if j not in covered)
        return (0.0, -residual, None, idx, 0, None)

    param = SimpleNamespace(RxFFE_cpx=RxFFE_cpx, N_bmax=len(isi_profile) + RxFFE_cpx,
                            N_bf=2, N_bg=2)
    # h such that h[isi_start:isi_end][RxFFE_cpx:N_bmax] == isi_profile
    isi_start, isi_end = 0, RxFFE_cpx + len(isi_profile)
    h = np.concatenate([np.zeros(RxFFE_cpx), isi_profile])

    idx = FOM_rxffe_floating_taps(param, h, None, 1, None, 0, 0, None, None,
                                  None, None, 1.0, isi_start, isi_end,
                                  _MMSE_FOM_fn=stub_MMSE_FOM)
    covered0 = sorted(int(i) - RxFFE_cpx - 1 for i in idx)   # 0-based trimmed positions
    print('selected (trimmed 0-based):', covered0)
    # Expect banks over {1,2} (0.40,0.35) and {5,6} (0.30,0.28) — the two largest pairs
    assert covered0 == [1, 2, 5, 6], f'unexpected selection {covered0}'
    print('FOM_rxffe_floating_taps greedy-selection test PASSED')
