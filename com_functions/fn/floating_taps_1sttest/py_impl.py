import numpy as np


def _hrem(h, index, N_bf, bmaxg):
    """Remove at most bmaxg from N_bf taps starting at 0-based index."""
    mid = (
        h[index:index + N_bf]
        - np.sign(h[index:index + N_bf])
        * np.minimum(bmaxg, np.abs(h[index:index + N_bf]))
    )
    return np.concatenate([h[:index], mid, h[index + N_bf:]])


def floating_taps_1sttest(hisi, N_b, N_bf, N_bg, N_bmax, bmaxg, COOP=0):
    """Find best floating DFE tap locations by exhaustive/sequential search.

    N_b: number of fixed taps (0-based: floating starts at index N_b).
    N_bmax: 1-based MATLAB max tap number; 0-based last position = N_bmax-1.
    COOP: 1 = co-optimise banks; 0 = sequential (default).
    Returns (bmax, floating_tap_locations) using 0-based Python indices.
    """
    hisi = np.asarray(hisi, dtype=float).ravel()
    if N_bg == 0:
        bmax = np.zeros(len(hisi))
        return bmax, np.array([], dtype=int)

    # MATLAB end1 = N_bmax - N_bf (1-based); Python end1_py = N_bmax - N_bf - 1 (0-based inclusive)
    # MATLAB loop: ig1 = N_b+1:end1  (1-based)  → Python: range(N_b, N_bmax - N_bf)
    if N_bg == 1:
        end1_py = N_bmax - N_bf      # exclusive upper bound for range
        end2_py = N_b + 1            # single iteration at N_b
        end3_py = N_b + 1
    elif N_bg == 2:
        end1_py = N_bmax - N_bf
        end2_py = N_bmax - N_bf
        end3_py = N_b + 1
    else:  # N_bg == 3
        end1_py = N_bmax - N_bf
        end2_py = N_bmax - N_bf
        end3_py = N_bmax - N_bf

    best_sigma = np.inf
    best_ig1 = N_b
    best_ig2 = N_b
    best_ig3 = N_b
    best_hcap = hisi.copy()

    if COOP:
        for ig1 in range(N_b, end1_py):
            hcap = _hrem(hisi, ig1, N_bf, bmaxg)
            for ig2 in range(N_b, end2_py):
                hcap2 = _hrem(hcap, ig2, N_bf, bmaxg) if N_bg >= 2 else hcap
                for ig3 in range(N_b, end3_py):
                    hcap3 = _hrem(hcap2, ig3, N_bf, bmaxg) if N_bg >= 3 else hcap2
                    sigma = float(np.linalg.norm(hcap3))
                    if sigma < best_sigma:
                        best_sigma = sigma
                        best_ig1, best_ig2, best_ig3 = ig1, ig2, ig3
                        best_hcap = hcap3
    else:
        # Sequential: first find best ig1
        for ig1 in range(N_b, end1_py):
            hcap = _hrem(hisi, ig1, N_bf, bmaxg)
            sigma = float(np.linalg.norm(hcap))
            if sigma < best_sigma:
                best_sigma = sigma
                best_ig1 = ig1
                best_hcap = hcap
        hisi = best_hcap
        # Then best ig2
        for ig2 in range(N_b, end2_py):
            hcap = _hrem(hisi, ig2, N_bf, bmaxg)
            sigma = float(np.linalg.norm(hcap))
            if sigma < best_sigma:
                best_sigma = sigma
                best_ig2 = ig2
                best_hcap = hcap
        hisi = best_hcap
        # Then best ig3
        for ig3 in range(N_b, end3_py):
            hcap = _hrem(hisi, ig3, N_bf, bmaxg)
            sigma = float(np.linalg.norm(hcap))
            if sigma < best_sigma:
                best_sigma = sigma
                best_ig3 = ig3
                best_hcap = hcap

    bmax = np.zeros(N_bmax)
    if N_bg == 1:
        bmax[best_ig1:best_ig1 + N_bf] = bmaxg
        locs = np.arange(best_ig1, best_ig1 + N_bf)
    elif N_bg == 2:
        bmax[best_ig1:best_ig1 + N_bf] = bmaxg
        bmax[best_ig2:best_ig2 + N_bf] = bmaxg
        locs = np.concatenate([
            np.arange(best_ig1, best_ig1 + N_bf),
            np.arange(best_ig2, best_ig2 + N_bf),
        ])
    else:  # N_bg == 3
        bmax[best_ig1:best_ig1 + N_bf] = bmaxg
        bmax[best_ig2:best_ig2 + N_bf] = bmaxg
        bmax[best_ig3:best_ig3 + N_bf] = bmaxg
        locs = np.concatenate([
            np.arange(best_ig1, best_ig1 + N_bf),
            np.arange(best_ig2, best_ig2 + N_bf),
            np.arange(best_ig3, best_ig3 + N_bf),
        ])

    return bmax, np.sort(locs).astype(int)
