import numpy as np

def _mextreme_complex(a, take):
    """MATLAB orders complex values by magnitude, then by angle; numpy orders
    them lexicographically by real part, so max([3+4i, 5]) is 3+4i in MATLAB
    and 5 in numpy. take is -1 for max, 0 for min."""
    f = np.asarray(a).ravel()
    good = ~np.isnan(np.abs(f))
    if not good.any():
        return f[0]
    g = f[good]
    return g[np.lexsort((np.angle(g), np.abs(g)))[take]]


def _mmax(a):
    """MATLAB max(): a NaN is skipped unless every element is NaN, and complex
    values are ordered by magnitude then angle.

    np.max propagates a NaN, so one bad sample swallows the result where MATLAB
    ignores it. np.nanmax matches MATLAB but warns on an all-NaN input, where
    MATLAB quietly returns NaN. The isnan test also keeps the ordinary no-NaN
    case on np.max's faster path.
    """
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, -1)
    if a.dtype.kind != 'f':
        return np.max(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.max(a)
    return np.nanmax(a)


def _mmin(a):
    """MATLAB min(): the mirror of _mmax."""
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, 0)
    if a.dtype.kind != 'f':
        return np.min(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.min(a)
    return np.nanmin(a)



def findbankloc(hisi, idx_st, idx_en, tap_bk, curval, bmaxg, N_bg):
    """Find optimal DFE bank locations within hisi[idx_st-1 : idx_en].

    Parameters use MATLAB 1-based convention (idx_st, idx_en).
    Returns 0-based indices into hisi (Python convention).

    Algorithm:
      Build ndiff[j] = sum of (h0^2 - h1^2) for tap bank starting at j,
      where h1 = max(0, h0 - bmaxg*curval).  Choose N_bg non-adjacent banks
      greedily by highest ndiff; enforce minimum gap of tap_bk between banks.
    """
    hisi = np.asarray(hisi, dtype=float).ravel()
    len_ = idx_en - idx_st + 1
    h0 = np.abs(hisi[idx_st - 1:idx_en])   # length len_
    h1 = np.maximum(0.0, h0 - bmaxg * curval)
    if curval < 0:
        h1 = np.zeros(len_)

    n_bins = len_ - tap_bk + 1             # number of valid bank start positions
    h0n = np.zeros(n_bins)
    h1n = np.zeros(n_bins)
    for ii in range(tap_bk):
        h0n += h0[ii:ii + n_bins] ** 2
        h1n += h1[ii:ii + n_bins] ** 2
    ndiff = h0n - h1n                       # 0-based indices 0..n_bins-1

    MIN_E = -np.inf
    # idx stores 0-based ndiff positions; -1 = not yet placed
    idx = np.full(tap_bk * N_bg, -1, dtype=int)
    # Shortcut check: are 0..(N_bg-1)*tap_bk the top (N_bg-1)*tap_bk+1 entries?
    ordered_set = np.arange((N_bg - 1) * tap_bk + 1)
    set_next_bank = -1   # -1 = not pending

    for k in range(N_bg):
        val_sort = np.argsort(-ndiff, kind='stable')       # 0-based, descending

        if k == 0:
            ns = len(ordered_set)
            if np.array_equal(np.sort(val_sort[:ns]), ordered_set):
                idx = np.arange(N_bg * tap_bk)
                break

        if set_next_bank >= 0:
            new_bank = np.arange(set_next_bank, set_next_bank + tap_bk)
            idx[tap_bk * k: tap_bk * (k + 1)] = new_bank
            set_next_bank = -1
            ndiff = _mask(ndiff, new_bank, MIN_E)
            b_start = new_bank[0] - tap_bk + 1
            b_end = new_bank[0] - 1
            badV = _make_badV(b_start, b_end)
            if len(badV):
                ndiff[badV] = MIN_E
            continue

        new_bank = np.arange(val_sort[0], val_sort[0] + tap_bk)
        if k == N_bg - 1:
            idx[tap_bk * k: tap_bk * (k + 1)] = new_bank
            break

        placed = idx[:tap_bk * k]           # already-placed 0-based positions
        badV = np.array([], dtype=int)
        do_it_again = True
        first_time = True
        num_loops = 0
        while do_it_again:
            do_it_again = False
            if num_loops > len(ndiff):
                break
            b_start = new_bank[0] - tap_bk + 1
            b_end = new_bank[0] - 1
            badV = _make_badV(b_start, b_end)
            # remove already-placed indices from badV
            if len(badV) and len(placed):
                badV = badV[~np.isin(badV, placed)]
            goodV_idx = new_bank[0] - tap_bk

            if len(badV) > 0:
                if not first_time:
                    val_sort = np.argsort(-ndiff, kind='stable')
                first_time = False
                checkV = np.concatenate([badV, new_bank])
                badV_pos = np.array(
                    [int(np.where(val_sort == v)[0][0]) for v in badV]
                )

                found_goodV = False
                ii_found = len(val_sort) - 1
                for ii_vs in range(len(val_sort)):
                    if val_sort[ii_vs] == goodV_idx:
                        found_goodV = True
                        ii_found = ii_vs
                        break
                    if not np.any(val_sort[ii_vs] == checkV):
                        ii_found = ii_vs
                        break

                if (not found_goodV) and (len(badV_pos) > 0) and (_mmin(badV_pos) < ii_found):
                    do_it_again = True
                    ndiff[new_bank[0]] = MIN_E
                    new_bank = np.arange(val_sort[1], val_sort[1] + tap_bk)
                if found_goodV:
                    set_next_bank = goodV_idx
            num_loops += 1

        ndiff = _mask(ndiff, new_bank, MIN_E)
        idx[tap_bk * k: tap_bk * (k + 1)] = new_bank
        if len(badV):
            ndiff[badV] = MIN_E

    # Convert 0-based ndiff positions → 0-based hisi positions
    return idx + (idx_st - 1)


def _mask(ndiff, positions, value):
    """ndiff[positions] = value, growing ndiff the way MATLAB would.

    ML 6260/6296: `ndiff(new_bank)=min_energy` where new_bank is
    `val_sort(1):val_sort(1)+tap_bk-1`. When the strongest bank start is within
    tap_bk-1 of the end, that range runs past the end of ndiff -- ndiff is
    indexed by bank START position and so is tap_bk-1 shorter than h0. MATLAB
    grows an array on out-of-range assignment; NumPy raises IndexError.

    On ~4.4% of random inputs the strongest start lands in that window, so the
    Python version crashed where MATLAB did not. The grown entries are -Inf, so
    they sort last and are never selected as a bank: the growth is inert, which
    is why the divergence is a crash rather than a wrong answer.

    Returns ndiff, which may be a new, longer array.
    """
    positions = np.asarray(positions, dtype=int).ravel()
    if positions.size == 0:
        return ndiff
    need = int(positions.max()) + 1
    if need > ndiff.size:
        # MATLAB zero-fills any gap; assignments here are contiguous from an
        # in-range start, so no gap arises, but fill for fidelity regardless.
        ndiff = np.concatenate([ndiff, np.zeros(need - ndiff.size)])
    ndiff[positions] = value
    return ndiff


def _make_badV(b_start, b_end):
    """Build array of 0-based indices from b_start to b_end inclusive (handles negatives)."""
    if b_end < 0:
        return np.array([], dtype=int)
    start = max(0, b_start)
    return np.arange(start, b_end + 1, dtype=int)
