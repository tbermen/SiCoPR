# ============================================================
# MATLAB→Python translation notes for get_xtlk_noise
# MATLAB lines: 7844–7938
# ============================================================
# index_f2: first index where faxis > fb → 0-based Python.
# temp_angle = M*dt*pi*f; at f(1)==0 → set temp_angle[0]=1e-20 to avoid /0.
# PWF_tx: 1D row vector length = len(f).
# When max(upsampled_txffe) > 0: build PWF_tx via TXFFE weights × exp(-j2π(ii-icur)f/fb).
#   ii: 1-based in MATLAB → Python 0-based (icur: argmax → 0-based).
#   phase_memory[:,ii]: MATLAB 1-based column ii → Python column ii-1 (0-based).
#   But note: MATLAB ii=1..N, Python ii=0..N-1, so MATLAB ii → Python column index ii-1.
#   However: the exponent uses (ii-icur) and Python ii runs 0..N-1 while MATLAB runs 1..N,
#   so MATLAB (ii-icur) = Python (ii - icur) since both shift the same way.
# PWF_rx: same pattern with C coefficients; MATLAB ii=-RxFFE_cmx..RxFFE_cpx (1-based column in C).
#   C(ii+RxFFE_cmx+1): MATLAB 1-based → Python C[ii+RxFFE_cmx] (0-based).
#   phase_memory col: MATLAB (ii+RxFFE_cmx+1+len(upsampled_txffe)) → Python same-1 = 0-based.
# For each chdata[ii] (ii=1..end in MATLAB → ii=1..N-1 in Python, 0-based loop over ii=1..):
#   FEXT: MDFEXT power sum; NEXT: MDNEXT power sum.
# nargout==1 NEXT: sigma_XT = MDNEXT_ICN * sqrt(...).
# nargout==1 FEXT: sigma_XT = MDFEXT_ICN * sqrt(...).
# nargout==3: sigma_XT = norm([MDNEXT_ICN, MDFEXT_ICN]) * sqrt(...).
# Python API: nargout parameter (1 or 3) and type ('NEXT' or 'FEXT').
# ============================================================

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



def get_xtlk_noise(upsampled_txffe, xtlk_type, param, chdata, phase_memory=None, C=None):
    """Compute crosstalk noise (MATLAB lines 7844-7938).

    Returns (sigma_XT,) for nargout=1, or (sigma_XT, sigma_FEXT, sigma_NEXT) for nargout=3.
    xtlk_type: 'NEXT', 'FEXT', or 'both' (both → 3-output mode).
    """
    upsampled_txffe = np.asarray(upsampled_txffe, dtype=float).ravel()
    f = np.asarray(chdata[0].faxis, dtype=float).ravel()

    # index_f2: MATLAB's find(f>fb,1,'first') is 1-based and the sums below
    # slice 1:index_f2 INCLUSIVELY, so the 1-based number is exactly the
    # exclusive Python end. Using the 0-based index instead dropped the top
    # bin of every ICN sum. COM Octave, faxis = fb + (0:19)*1e9 (so the very
    # first point is at fb and the crossing is the second):
    #     sigma_XT 4.9666884883754367e-05   the port answered 2.85e-19,
    # having summed one bin where the reference sums two, and the dropped bin
    # is the one that carries the energy.
    mask = f > param.fb
    index_f2 = int(np.argmax(mask)) + 1 if np.any(mask) else len(f)

    M = int(param.samples_per_ui)
    dt = float(param.sample_dt)

    temp_angle = M * dt * np.pi * f
    if f[0] == 0.0:
        temp_angle[0] = 1e-20

    # Build PWF_tx
    # pre_calc exists only if the TX branch below runs: MATLAB assigns it
    # inside `if max(upsampled_txffe) > 0` and the RX branch reads it anyway.
    pre_calc = None
    PWF_tx = np.ones(len(f), dtype=complex)
    if _mmax(upsampled_txffe) > 0:
        PWF_tx = np.zeros(len(f), dtype=complex)
        icur = int(np.argmax(upsampled_txffe))  # 0-based
        pre_calc = phase_memory is not None and len(phase_memory) > 0
        for ii in range(len(upsampled_txffe)):
            if upsampled_txffe[ii] == 0:
                continue
            if ii == icur:
                PWF_tx = PWF_tx + upsampled_txffe[ii]
            else:
                if pre_calc:
                    term = upsampled_txffe[ii] * phase_memory[:, ii]
                else:
                    term = upsampled_txffe[ii] * np.exp(-1j * 2 * np.pi * (ii - icur) * f / param.fb)
                PWF_tx = PWF_tx + term.ravel()

    # Build PWF_rx
    PWF_rx = np.ones(len(f), dtype=complex)
    if C is not None:
        C_arr = np.asarray(C, dtype=float).ravel()
        PWF_rx = np.zeros(len(f), dtype=complex)
        cmx = int(param.RxFFE_cmx)
        cpx = int(param.RxFFE_cpx)
        n_txffe = len(upsampled_txffe)
        for ii in range(-cmx, cpx + 1):
            c_idx = ii + cmx  # 0-based index into C
            # No length guard: MATLAB indexes C(ii+cmx+1) and stops if there
            # is no such element. COM Octave, C = [0.05 -0.2]:
            #     error: C(3): out of bound 2 (dimensions are 2x1)
            # and with C = []: error: C(1): out of bound 0. Skipping the
            # missing taps instead answered with a silently truncated RX FFE
            # (0.0025285640126039653 against an error, and 0 for an empty C).
            if C_arr[c_idx] == 0:
                continue
            if ii + 1 == 0:
                PWF_rx = PWF_rx + C_arr[c_idx]
            else:
                if pre_calc is None:
                    # COM Octave, all-zero upsampled_txffe with C supplied:
                    #     error: 'pre_calc' undefined near line 55, column 16
                    raise NameError(
                        "get_xtlk_noise: 'pre_calc' is undefined. MATLAB sets "
                        'it only when max(upsampled_txffe) > 0, so this call '
                        'errors in the reference.')
                if pre_calc:
                    pm_col = ii + cmx + n_txffe  # 0-based
                    term = C_arr[c_idx] * phase_memory[:, pm_col]
                else:
                    term = C_arr[c_idx] * np.exp(-1j * 2 * np.pi * (ii + 1) * f / param.fb)
                PWF_rx = PWF_rx + term.ravel()

    SINC = np.sin(temp_angle) / temp_angle  # sinc-like
    PWF_data = SINC ** 2

    MDFEXT = np.zeros(len(f))
    MDNEXT = np.zeros(len(f))
    MDFEXT_ICN = 0.0
    MDNEXT_ICN = 0.0

    for i in range(1, len(chdata)):
        ch = chdata[i]
        PWF_next_wt = np.abs(PWF_data * PWF_rx)            # no TX weighting for NEXT
        PWF_fext_wt = np.abs(PWF_data * PWF_rx * PWF_tx)   # TX weighting for FEXT
        sdd21ctf = np.asarray(ch.sdd21ctf, dtype=complex).ravel()
        delta_f = float(ch.delta_f)
        A = float(ch.A)
        f2 = float(param.f2)

        if ch.type == 'FEXT':
            MDFEXT = np.sqrt(np.abs(sdd21ctf) ** 2 + MDFEXT ** 2)  # power sum over channels
            MDFEXT_ICN = float(np.sqrt(
                2 * delta_f / f2 * np.sum(A ** 2 * PWF_fext_wt[:index_f2] * MDFEXT[:index_f2] ** 2)))
        elif ch.type == 'NEXT':
            MDNEXT = np.sqrt(np.abs(sdd21ctf) ** 2 + MDNEXT ** 2)  # power sum over channels
            MDNEXT_ICN = float(np.sqrt(
                2 * delta_f / f2 * np.sum(A ** 2 * PWF_next_wt[:index_f2] * MDNEXT[:index_f2] ** 2)))
        else:
            raise ValueError(
                f'get_xtlk_noise: unexpected channel type "{ch.type}"')

    scale = float(np.sqrt((param.levels ** 2 - 1) / (3 * (param.levels - 1) ** 2)))

    if xtlk_type.upper() == 'NEXT':
        return MDNEXT_ICN * scale
    elif xtlk_type.upper() == 'FEXT':
        return MDFEXT_ICN * scale
    else:
        sigma_XT = float(np.linalg.norm([MDNEXT_ICN, MDFEXT_ICN])) * scale
        sigma_NEXT = MDNEXT_ICN * scale
        sigma_FEXT = MDFEXT_ICN * scale
        return sigma_XT, sigma_FEXT, sigma_NEXT
