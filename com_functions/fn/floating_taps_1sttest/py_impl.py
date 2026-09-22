import numpy as np


def _hrem(h, index, N_bf, bmaxg):
    """Remove at most bmaxg from N_bf taps starting at 0-based index.

    The index is 0-based here where the canonical hrem takes it 1-based: the
    caller below passes ig1/ig2/ig3 from range(N_b, ...), the 0-based form of
    MATLAB's `ig1 = N_b+1:end1`. Everything else matches the canonical.
    """
    h = np.asarray(h, dtype=float)
    # MATLAB L7941 builds the result with HORIZONTAL concatenation, so h must
    # be a row. A column makes the three pieces 1x1 / Nx1 / Mx1 and MATLAB
    # errors "horizontal dimensions mismatch". Verified against Octave.
    if h.ndim == 2 and h.shape[1] == 1 and h.shape[0] > 1:
        raise ValueError(
            'hrem: h must be a row vector. MATLAB concatenates the three '
            'pieces horizontally, so a %dx1 column errors there with '
            '"horizontal dimensions mismatch".' % h.shape[0])
    h = h.ravel()
    index = int(index)

    # MATLAB indexes h(index:index+N_bf-1) directly, so running past the end
    # errors ("h(7): out of bound 5"). A numpy slice silently returns a SHORTER
    # segment, handing back a result of the wrong length instead.
    if index < 0 or index + int(N_bf) > h.size:
        raise IndexError(
            'hrem: h(%d:%d) is out of bounds for a length-%d h; MATLAB errors '
            'here rather than shortening the result.'
            % (index + 1, index + int(N_bf), h.size))

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
        # MATLAB returns on `bmax=0` -- the SCALAR zero, not a vector, and
        # floating_tap_locations is never assigned.  This port always hands
        # back both, which is the two-output call.  COM Octave:
        #   [bmax,locs] = ... N_bg=0 -> "error: element number 2 undefined in
        #                                return list"
        #   bmax        = ... N_bg=0 -> 0   (1x1)
        # The old code returned zeros(len(hisi)) and an empty location list,
        # neither of which the reference ever produces.
        raise ValueError('floating_taps_1sttest: N_bg=0 returns bmax=0 with '
                         'floating_tap_locations unassigned; MATLAB errors on '
                         'a two-output call ("element number 2 undefined")')

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

    # MATLAB seeds best_ig1/2/3 = -1 and leaves best_hcap unset, so a group
    # that never improves on best_sigma is NOT silently placed at N_b: the
    # final bmax(best_ig:..) is an illegal subscript and MATLAB stops.
    # COM Octave, hisi 20 taps, N_b=2 N_bf=3 N_bmax=16, bmaxg=0 (so no
    # iteration can reduce the norm), N_bg=2 sequential:
    #   "error: bmax(-1): subscripts must be either integers 1 to (2^63)-1".
    # The pre-seeded N_b answered with a bank at 2..4 that was never evaluated.
    UNSET = -1
    best_sigma = np.inf
    best_ig1 = best_ig2 = best_ig3 = UNSET
    best_hcap = None

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
        # MATLAB reads best_hcap here unconditionally.  When the ig1 loop
        # never ran (N_bmax-N_bf < N_b+1) or every sigma was NaN, nothing
        # assigned it.  COM Octave, N_b=6 N_bf=3 N_bmax=8 and again with a NaN
        # in hisi: "error: 'best_hcap' undefined near line 69".  Python carried
        # on from a copy of hisi and returned a bank it had never scored.
        if best_hcap is None:
            raise ValueError("floating_taps_1sttest: 'best_hcap' undefined -- "
                             'no group improved on the initial sigma')
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

    # bmax(best_ig:...) with an unset best_ig is bmax(-1) in MATLAB, which
    # stops.  A negative Python slice bound would instead index from the END
    # of bmax and write the bank in the wrong place.
    for used, ig in ((1, best_ig1), (2, best_ig2), (3, best_ig3)):
        if used <= N_bg and ig == UNSET:
            raise IndexError(
                'floating_taps_1sttest: bmax(-1): group %d was never selected, '
                'so its location is unset' % used)

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
