# ============================================================
# MATLAB→Python translation notes for force
# MATLAB lines: 6090–6253
# ============================================================
# RxFFE tap solver. Inlines FFE helper.
# Inputs: V (pulse response row), param, OP, ix (0-based cursor), C (taps or []), return_V flag
#
# FFE: for each tap i (0-based, 0..len(C)-1):
#   ishift = (i - cmx) * spui
#   V0 += np.roll(V, ishift) * C[i]
#
# MATLAB vsampled: zero-padded to [zeros(num_taps), vsampled_raw, zeros(cpx)]
# In Python: same 0-based indexing.
# VV matrix: num_taps x num_taps
#   VV[:, i] = vsampled[ivs+i : ivs+i-num_taps+1 : -1]  (reverse stride)
# Solver: VV'\FV' → np.linalg.lstsq(VV.T, FV).
# WIENER_HOPF_MMSE path: undefined in the reference MATLAB -> raises NotImplementedError
#   (not a port gap; the reference itself has no such function). Use FFE_OPT_METHOD='MMSE'.
# Floating taps: uses _findbankloc (inlined).
# RXFFE_FLOAT_CTL: 'taps' uses Cmod for bank, else uses VV row.
# RXFFE_TAP_CONSTRAINT: 'unity cursor' → normalize by Cmod[cmx].
# ============================================================

import numpy as np
from com_functions.fn.FFE.py_impl import FFE as _FFE

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

from types import SimpleNamespace


def _fb_mask(ndiff, positions, value):
    """ndiff[positions] = value, growing ndiff the way MATLAB would.

    ML 6260/6296 assign `ndiff(new_bank)=min_energy` where new_bank can run
    past the end of ndiff: ndiff is indexed by bank START position, so it is
    tap_bk-1 shorter than h0. MATLAB grows on out-of-range assignment; NumPy
    raises IndexError. The grown entries are -Inf, sort last, and are never
    selected, so the growth is inert -- the divergence was a crash, not a wrong
    answer. See com_functions/fn/findbankloc/py_impl.py.
    """
    positions = np.asarray(positions, dtype=int).ravel()
    if positions.size == 0:
        return ndiff
    need = int(positions.max()) + 1
    if need > ndiff.size:
        ndiff = np.concatenate([ndiff, np.zeros(need - ndiff.size)])
    ndiff[positions] = value
    return ndiff


def _findbankloc(hisi, idx_st, idx_en, tap_bk, curval, bmaxg, N_bg):
    """Faithful port of MATLAB findbankloc (ML 5918-6069).

    This was previously a "pick the highest-power non-overlapping banks"
    approximation, which is not what MATLAB does. The real routine ranks bank
    start positions by ndiff = h0n - h1n and then runs a badV/goodV
    admissibility loop that can reject the strongest bank and pre-commit the
    next one, so the selected set differs from a plain energy ranking.

    idx_st/idx_en are MATLAB 1-based bounds. Returns MATLAB-convention 1-based
    positions in hisi, which is what the callers here expect (MMSE_FOM selects
    columns as idx + RxFFE_cmx; force places taps at cmx + 1 + idx - 1).
    """
    hisi = np.asarray(hisi, dtype=float).ravel()
    idx_st, idx_en = int(idx_st), int(idx_en)
    tap_bk, N_bg = int(tap_bk), int(N_bg)
    len_ = idx_en - idx_st + 1
    h0 = np.abs(hisi[idx_st - 1:idx_en])
    h1 = np.maximum(0.0, h0 - bmaxg * curval)
    if curval < 0:
        # ML 5934: a negative cursor would invert ndiff and make the WEAKEST isi
        # the most desirable, so h1 is forced flat.
        h1 = np.zeros(len_)

    n_bins = len_ - tap_bk + 1
    h0n = np.zeros(n_bins)
    h1n = np.zeros(n_bins)
    for _ii in range(tap_bk):
        h0n += h0[_ii:_ii + n_bins] ** 2
        h1n += h1[_ii:_ii + n_bins] ** 2
    ndiff = h0n - h1n

    def _bad_range(b_start, b_end):
        """Taps closer than one bank below new_bank[0] (ML 6002-6009)."""
        if b_end < 0:
            return np.array([], dtype=int)
        return np.arange(max(0, b_start), b_end + 1, dtype=int)

    MIN_E = -np.inf
    idx = np.full(tap_bk * N_bg, -1, dtype=int)
    ordered_set = np.arange((N_bg - 1) * tap_bk + 1)
    set_next_bank = -1

    for k in range(N_bg):
        # stable, to match MATLAB's sort(...,'descend'); ndiff ties are common
        # because the isi tail is mostly zeros.
        val_sort = np.argsort(-ndiff, kind='stable')

        if k == 0 and np.array_equal(np.sort(val_sort[:len(ordered_set)]), ordered_set):
            idx = np.arange(N_bg * tap_bk)
            break

        if set_next_bank >= 0:
            new_bank = np.arange(set_next_bank, set_next_bank + tap_bk)
            idx[tap_bk * k:tap_bk * (k + 1)] = new_bank
            set_next_bank = -1
            ndiff = _fb_mask(ndiff, new_bank, MIN_E)
            badV = _bad_range(new_bank[0] - tap_bk + 1, new_bank[0] - 1)
            if len(badV):
                ndiff[badV] = MIN_E
            continue

        new_bank = np.arange(val_sort[0], val_sort[0] + tap_bk)
        if k == N_bg - 1:
            idx[tap_bk * k:tap_bk * (k + 1)] = new_bank
            break

        placed = idx[:tap_bk * k]
        badV = np.array([], dtype=int)
        do_it_again, first_time, num_loops = True, True, 0
        while do_it_again:
            do_it_again = False
            if num_loops > len(ndiff):
                break
            badV = _bad_range(new_bank[0] - tap_bk + 1, new_bank[0] - 1)
            if len(badV) and len(placed):
                badV = badV[~np.isin(badV, placed)]
            goodV_idx = new_bank[0] - tap_bk

            if len(badV) > 0:
                if not first_time:
                    val_sort = np.argsort(-ndiff, kind='stable')
                first_time = False
                checkV = np.concatenate([badV, new_bank])
                badV_pos = np.array([int(np.where(val_sort == v)[0][0]) for v in badV])

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

                if (not found_goodV) and len(badV_pos) and _mmin(badV_pos) < ii_found:
                    do_it_again = True
                    ndiff[new_bank[0]] = MIN_E
                    new_bank = np.arange(val_sort[1], val_sort[1] + tap_bk)
                if found_goodV:
                    set_next_bank = goodV_idx
            num_loops += 1

        ndiff = _fb_mask(ndiff, new_bank, MIN_E)
        idx[tap_bk * k:tap_bk * (k + 1)] = new_bank
        if len(badV):
            ndiff[badV] = MIN_E

    # ML 6069 returns idx+idx_st-1 as 1-based; our idx is 0-based within the
    # window, so +idx_st lands on the same 1-based value.
    return idx + idx_st

def force(V, param, OP, ix=None, C=None, return_V=1, chdata=None, txffe=None, Noise_XC=None):
    """RxFFE tap solver (MATLAB lines 6090-6253).

    V: pulse response (1D array).
    ix: 0-based cursor index (default: argmax of V).
    C: pre-computed taps (if not None, skips optimization).
    return_V: if 0, skip computing Vfiltered (speed-up).

    Returns (Vfiltered, Cmod, idx).
    """
    V = np.asarray(V, dtype=float).ravel()

    if ix is None:
        ix = int(np.argmax(V))

    cmx = int(param.RxFFE_cmx)
    cpx = int(param.RxFFE_cpx)

    if int(getattr(param, 'N_bg', 0)) != 0:
        cpx = int(param.N_bmax)

    num_taps = cmx + cpx + 1
    cstep = float(getattr(param, 'RxFFE_stepz', 0))
    ndfe = int(param.ndfe)
    spui = int(param.samples_per_ui)
    # ML 6568 is `param.current_ffegain=0;` here, and ML 6634 reads it back a
    # few lines down, so inside force the gain is ALWAYS 0 in the reference.
    #
    # MATLAB passes param BY VALUE, so that write never reaches the caller.
    # This used to write the caller's object, zeroing a value the caller had
    # set. COM Octave, on its own force with param.current_ffegain = 7:
    #   caller param.current_ffegain BEFORE the call: 7
    #   caller param.current_ffegain AFTER  the call: 7
    # SiCoPR gave 0. Keeping the write local restores by-value semantics and
    # leaves the reference's own logic untouched, because the only reader is
    # inside this function.
    current_ffegain = 0.0

    idx = np.array([], dtype=int)

    if return_V and C is not None and len(np.asarray(C)) > 0:
        Vfiltered = _FFE(C, cmx, spui, V)
        return Vfiltered, np.asarray(C, dtype=float).ravel(), idx

    # Build vsampled_raw: samples at spui spacing starting at ix.
    #
    # MATLAB's ix is 1-BASED and ours is 0-based, so mod(ix,spui) there is
    # (ix+1) % spui here. Taking the modulus of the 0-based ix instead put the
    # first pre-cursor sample one UI late whenever mod(ix_MATLAB,spui)==1, and
    # dropped V(1) from the sampled vector. COM Octave, spui=4, cmx=2, cpx=3,
    # V = exp(-(t-17)^2/12) + 0.4*exp(-t/6) on t=0:39:
    #   ix=5  -> Cmod(1) = -341.8916096927, the port gave 1.3240864760e-16
    #   ix=9  -> Cmod(1) =   -1.2321309684, the port gave -1.6655157754
    #   ix=13 -> Cmod(1) =   -0.1151234261, the port gave 0.3764832703
    #   ix=17 -> Cmod(6) = -7.9658626843e-08, the port gave 9.0392051417e-05
    # Every ix with mod(ix,spui)~=1 already agreed to 1e-14.
    #
    # Both MATLAB branches end the pre-cursor run at 1-based ix-spui, because
    # mod(ix,spui) + spui*(floor(ix/spui)-1) is ix-spui either way.
    #
    # `ix < length(V)` is on the 1-BASED ix, so the last sample of V takes the
    # else branch in MATLAB and took the if branch here. COM Octave, same V and
    # params, ix = length(V) = 64:
    #   Cmod = [-0.180678956 0.117026343 1 -0.23555319 0.209619484 -0.239829445]
    #   the port gave [0.02850437 -0.19691076 1 -0.091935245 0.131560803 -0.04219104]
    N = len(V)
    ix = int(ix)
    mod_ix = (ix + 1) % spui                       # MATLAB mod(ix, spui)
    first = (spui if mod_ix == 0 else mod_ix) - 1  # 0-based first pre sample
    ix_before_end = (ix + 1) < N                   # MATLAB ix < length(V)
    if ix_before_end:
        pre_idx = np.arange(first, ix - spui + 1, spui)
        post_idx = np.arange(ix, N, spui)
        vsampled_raw = np.concatenate([V[pre_idx], V[post_idx]])
    else:
        vsampled_raw = V[first::spui]

    # Zero-pad: [zeros(num_taps), vsampled_raw, zeros(cpx)]
    vsampled = np.concatenate([np.zeros(num_taps), vsampled_raw, np.zeros(cpx)])

    # Find ivs: index in vsampled matching V[ix]
    if ix_before_end:
        matches = np.where(vsampled == V[ix])[0]
        ivs = int(matches[0]) if len(matches) > 0 else num_taps
    else:
        ivs = int(np.argmax(vsampled))

    # Build VV matrix (num_taps x num_taps).
    # A window that runs off either end of vsampled is a subscript error in
    # MATLAB, not a short column zero-filled to length. Silently padding meant
    # the solve ran on a VV the reference never produces. COM Octave, spui=4,
    # cmx=2, cpx=3, length(V)=64, ix=63:
    #   error: vsampled(26): out of bound 25 (dimensions are 1x25)
    #   force at line 90 column 5
    nvs = len(vsampled)
    VV = np.zeros((num_taps, num_taps))
    for i in range(num_taps):
        start_idx = ivs + i
        end_idx = start_idx - num_taps + 1
        if start_idx >= nvs:
            raise IndexError('vsampled(%d): out of bound %d (dimensions are '
                             '1x%d)' % (start_idx + 1, nvs, nvs))
        if end_idx < 0:
            raise IndexError('vsampled(%d): subscripts must be either integers '
                             '1 to (2^63)-1 or logicals' % (end_idx + 1))
        VV[:, i] = vsampled[start_idx:end_idx - 1 if end_idx > 0 else None:-1]

    if C is None or len(np.asarray(C)) == 0:
        ffe_opt = str(getattr(OP, 'FFE_OPT_METHOD', 'FORCE')).upper()
        if ffe_opt == 'WIENER-HOPF':
            # MATLAB L6188 calls WIENER_HOPF_MMSE(), but that function is NOT defined
            # anywhere in the reference (com_ieee8023_4p14p0.m or any matlab_source.m) —
            # the reference itself errors on this path.  We do not fabricate a solver
            # (would be speculative and, since Wiener-Hopf == MMSE normal equations, a
            # duplicate of the existing 'MMSE' path).  Use FFE_OPT_METHOD='MMSE' instead.
            raise NotImplementedError(
                "force: FFE_OPT_METHOD='WIENER-HOPF' is non-functional in the reference "
                "MATLAB (WIENER_HOPF_MMSE is undefined). Use 'MMSE'.")

        # Build forcing vector FV
        FV = np.zeros(num_taps)
        cursor_gain_dB = current_ffegain      # ML 6634, always 0: see L213
        FV[cmx] = vsampled[ivs] * 10 ** (cursor_gain_dB / 20.0)
        if ndfe != 0 and cpx > 0 and ivs + 1 < len(vsampled):
            bmax_arr = np.asarray(param.bmax, dtype=float).ravel()
            bmax_val = float(bmax_arr[0]) if len(bmax_arr) > 0 else 1.0
            FV[cmx + 1] = min(bmax_val * FV[cmx], abs(vsampled[ivs + 1])) * np.sign(vsampled[ivs + 1])

        VVt = VV.T
        if VV.shape[0] == VV.shape[1]:
            # RULED 2026-09-23 (owner): reproduce MATLAB's CONDITION, but make
            # its occurrence loud instead of silent.
            #
            # The three references disagreed:
            #   MATLAB  VV'\FV'  warns "Matrix is singular to working
            #                    precision" and returns Inf.
            #   Octave  VV'\FV'  returned a minimum-norm least-squares
            #                    solution: [1 2; 2 4] \ [1; 3] -> [0.28 0.56].
            #   here             LinAlgError -> lstsq, i.e. the OCTAVE answer.
            #
            # Octave was repaired in 4c73cba: force's backslash now goes
            # through octave/patches/mldivide_matlab.m, which returns inv(A)*b
            # for rcond(A)==0, so the oracle gives MATLAB's Inf. The oracle can
            # therefore adjudicate this now, and the port was the odd one out.
            #
            # A silent lstsq answer is the one outcome NEITHER reference
            # produces, and it hides a degenerate channel by quietly returning
            # something plausible. MATLAB's Inf poisons the taps and propagates;
            # knowing the case occurred is worth more than either. So: raise.
            #
            # VV is built square (zeros(num_taps,num_taps)), so the else-branch
            # below is dead in both languages and only this path matters.
            try:
                C_solved = np.linalg.solve(VVt, FV)
            except np.linalg.LinAlgError:
                raise ValueError(
                    'force: VV is singular to working precision, so the tap '
                    'solve has no unique answer. MATLAB returns Inf here and '
                    'carries it into the taps; SiCoPR stops instead, so the '
                    'degenerate channel is visible rather than silently '
                    'absorbed. num_taps=%d, cmx=%d.' % (num_taps, cmx))
        else:
            VVVt = VV @ VV.T
            try:
                C_solved = (np.linalg.solve(VVVt, VV) @ FV)
            except np.linalg.LinAlgError:
                C_solved, _, _, _ = np.linalg.lstsq(VV.T, FV, rcond=None)

        Cmod = C_solved[:num_taps]

        # Floating taps.
        # DIVERGENCE, REPORTED NOT FIXED (2026-09-22): MATLAB calls findbankloc
        # unconditionally, this guard skips it when N_bg==0. With N_bg==0 the
        # routine returns an empty idx, so the outputs agree -- unless
        # param.N_bmax exceeds the length of the vector it indexes, where the
        # reference stops and this does not. COM Octave, cmx=2, cpx=3
        # (num_taps=6), N_bg=0, N_bmax=10:
        #   error: hisi(10): out of bound 6 (dimensions are 6x1)
        #   findbankloc at line 12 column 1 / force at line 127 column 14
        # The bound belongs in findbankloc, which is a shared helper, so this
        # is for the owner of com_functions/fn/findbankloc to place.
        N_bg = int(getattr(param, 'N_bg', 0))
        if N_bg != 0:
            N_tail_start = int(getattr(param, 'N_tail_start', cpx))
            N_bmax = int(getattr(param, 'N_bmax', cpx))
            N_bf = int(getattr(param, 'N_bf', 1))
            bmaxg = float(getattr(param, 'bmaxg', 1.0))
            float_ctl = str(getattr(OP, 'RXFFE_FLOAT_CTL', 'isi')).lower()
            if float_ctl == 'taps':
                src = Cmod
            else:
                src = VV[cmx, :] if cmx < VV.shape[0] else Cmod
            idx = _findbankloc(src, N_tail_start, N_bmax, N_bf, float(Cmod[cmx]), bmaxg, N_bg)
            idx = np.sort(idx)

        tap_constraint = str(getattr(OP, 'RXFFE_TAP_CONSTRAINT', 'unity cursor')).lower()
        if tap_constraint == 'unity cursor':
            # RULED 2026-09-23 (owner), the same ruling as the singular VV
            # above: Cmod(cmx+1) only collapses when VV is degenerate.
            #
            # ML 6238 is `Cmod=Cmod/Cmod(cmx+1)` with NO guard. The 1e-12 floor
            # that used to stand here diverged twice over. For a cursor tap
            # below 1e-12 but LEGAL it substituted 1.0, leaving the taps
            # unnormalised and differing from MATLAB by the whole scale factor;
            # and for a cursor tap of exactly zero it silently returned
            # unnormalised taps where MATLAB returns Inf or NaN.
            #
            # So divide exactly as MATLAB does, however small the tap, and stop
            # only on the one value MATLAB cannot divide by. Small is not
            # degenerate; zero is.
            cursor_val = float(Cmod[cmx])
            if cursor_val == 0.0:
                raise ValueError(
                    'force: the cursor tap Cmod[%d] is exactly zero, so the '
                    'unity-cursor normalisation has no answer. MATLAB divides '
                    'anyway and carries Inf or NaN into every tap; SiCoPR '
                    'stops instead, so the degenerate case is visible.' % cmx)
            Cmod = Cmod / cursor_val
        else:
            Cmod = C_solved[:num_taps]

        if cstep != 0:
            Cmod = np.floor(np.abs(Cmod / cstep)) * np.sign(Cmod) * cstep

        if len(idx) > 0:
            N_fixed = cmx + int(getattr(param, 'RxFFE_cpx', cpx)) + 1
            C1 = Cmod.copy()
            C1[N_fixed:] = 0.0
            for j, k in enumerate(idx):
                pos = cmx + 1 + int(k) - 1  # idx is 1-based
                if pos < len(C1):
                    C1[pos] = Cmod[pos] if pos < len(Cmod) else 0.0
            Cmod = C1
    else:
        Cmod = np.asarray(C, dtype=float).ravel()

    if return_V:
        Vfiltered = _FFE(Cmod, cmx, spui, V)
    else:
        Vfiltered = np.array([])

    return Vfiltered, Cmod, idx
