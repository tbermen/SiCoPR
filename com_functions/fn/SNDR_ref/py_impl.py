import numpy as np
from com_functions.fn.FFE.py_impl import FFE as _FFE
from types import SimpleNamespace


def _matlab_peak(x):
    """MATLAB `find(x==max(x),1,'first')`, 0-based, or None for no match.

    max() skips NaN where np.argmax returns the NaN's index.  COM Octave,
    SNDR_ref on a 400-point Gaussian with PR(101)=NaN: SNDR_ref(1) is
    19.999999999947136, not NaN -- the reference peaks on the largest real
    sample and carries on.  An all-NaN x matches nothing, and MATLAB then
    indexes with an empty subscript, which is legal and yields an empty
    result rather than an error.
    """
    if x.size == 0:
        return None
    with np.errstate(invalid='ignore'):
        mx = np.nanmax(x) if not np.all(np.isnan(x)) else np.nan
    hit = np.flatnonzero(x == mx)
    return int(hit[0]) if hit.size else None


def _matlab_range_index(x, first, step, last, expr):
    """MATLAB `x(first:step:last)` with MATLAB's subscript rules.

    An empty range indexes nothing and is legal; a non-empty one must be
    whole, >= 1 and <= numel(x).  COM Octave, SNDR_ref with the pulse peak
    inside D_p UI of the start: "error: PR_FFE(-5): subscripts must be either
    integers 1 to (2^63)-1 or logicals"; with N_p*M+ipeak past the end:
    "error: PR_FFE(541): out of bound 400 (dimensions are 400x1)".  A Python
    slice answers both silently -- a negative start reads from the tail and a
    long stop truncates, which returned a plausible wrong SNDR.
    """
    n = int(np.floor((last - first) / step)) + 1 if last >= first else 0
    if n <= 0:
        return x[:0]
    idx = first + step * np.arange(n, dtype=float)
    bad = idx[idx != np.round(idx)]
    if bad.size:
        # Octave accepts a non-integer RANGE as an index with a warning and
        # truncates toward zero; MATLAB refuses it.  The port used to refuse
        # neither and instead truncated D_p/N_p/M themselves, which agrees
        # with neither: COM Octave with D_p=2.1 gives SNDR_ref(1)
        # 19.948026128813808 against the port's 19.999999999629168.
        raise ValueError('SNDR_ref: %s(%g): subscripts must be positive '
                         'integers' % (expr, bad[0]))
    if idx[0] < 1:
        raise IndexError('SNDR_ref: %s(%d): subscripts must be either '
                         'integers 1 to (2^63)-1 or logicals'
                         % (expr, int(idx[0])))
    if idx[-1] > x.size:
        raise IndexError('SNDR_ref: %s(%d): out of bound %d'
                         % (expr, int(idx[-1]), x.size))
    return x[idx.astype(int) - 1]


def SNDR_ref(PR_Ref, param):
    """Compute SNDR for 6 TX-FFE presets (MATLAB lines 4405-4442).

    Returns results struct with SNDR_ref (array), SNDR_ref_p1..p6, sigma_iL.
    """
    PR_Ref = np.asarray(PR_Ref, dtype=float).ravel()

    # MATLAB tests `~isfield(param,'preset')` -- field existence only.  COM
    # Octave with param.preset=[]: the loop never runs, the local SNDR_ref is
    # never assigned, `results.SNDR_ref=SNDR_ref` resolves to the function and
    # recurses, "error: 'param' undefined near line 2".  Substituting the
    # defaults for a present-but-empty preset answered where MATLAB refuses.
    if not hasattr(param, 'preset'):
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
    # D_p and N_p stay as given: MATLAB uses them to form a subscript and
    # refuses a fractional one, so truncating them here would answer a call
    # the reference declines.  M has to be whole to index at all.
    M_f = float(param.samples_per_ui)
    if M_f != np.floor(M_f):
        raise ValueError('SNDR_ref: samples_per_ui=%g is not an integer'
                         % M_f)
    M = int(M_f)
    D_p = float(param.D_p)
    N_p = float(param.N_p)
    PR_noFFE = PR_Ref.copy()

    ipeak_0 = _matlab_peak(PR_noFFE)     # 0-based, or None for an all-NaN PR
    if ipeak_0 is None:
        PR_noFFE_sampled = PR_noFFE[:0]
    else:
        istart_0 = ipeak_0 % M           # 0-based start for subsampling
        iend_0 = (len(PR_noFFE) // M) * M  # exclusive end for Python slice
        PR_noFFE_sampled = PR_noFFE[istart_0:iend_0:M]
    sigma_tn_base = np.float64(ss(PR_noFFE_sampled))

    n_presets = len(param.preset)
    SNDR_ref_arr = np.zeros(n_presets)
    sigma_iL_arr = np.zeros(n_presets)

    for ipst, preset in enumerate(param.preset):
        PR_FFE = _FFE(preset.txffe, 3, M, PR_noFFE)  # cmx=3: 0-based tap 4
        ipeak_0_ffe = _matlab_peak(PR_FFE)
        if ipeak_0_ffe is None:
            hss = PR_FFE[:0]
        else:
            ipeak_1 = ipeak_0_ffe + 1    # MATLAB subscripts are 1-based
            hss = _matlab_range_index(PR_FFE, -D_p * M + ipeak_1, M_f,
                                      N_p * M + ipeak_1, 'PR_FFE')
        ss_hss = ss(hss)
        sigma_iL_arr[ipst] = float(np.sqrt(ss_hss))
        sigma_ts = np.float64(ss_hss * 10 ** (SNR_TX / 10))
        with np.errstate(divide='ignore', invalid='ignore'):
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
