import numpy as np


def applyDFEbk(hisi, hisi_ref, idx, tap_bk, curval, bmaxg, dfe_delta):
    """Apply a contiguous bank of DFE taps at 0-based position idx.

    idx is the 0-based start index (MATLAB used 1-based; callers must subtract 1).
    Returns (hisi_modified, tap_coef, hisi_ref_modified).

    dfe_delta has no default. The MATLAB guards it with `if nargin<6`, but
    dfe_delta is the SEVENTH argument, so a 6-argument call skips the guard and
    then reads an unset variable.
    COM Octave: applyDFEbk(hisi,href,2,3,4,0.5)   -> "'dfe_delta' undefined"
                applyDFEbk(hisi,href,2,3,4)       -> "'bmaxg' undefined"
    Seven arguments is the only arity the reference accepts, so Python must not
    answer a 6-argument call either (it used to, with dfe_delta=0).
    """
    hisi = np.asarray(hisi, dtype=float).copy()
    hisi_ref = np.asarray(hisi_ref, dtype=float).copy()

    # rng=idx:idx+tap_bk-1 is a READ of hisi, so MATLAB errors when it runs off
    # either end; a Python slice silently shortens and answers on the wrong bank.
    # COM Octave: hisi=[1 2 3 4 5 6], idx=5, tap_bk=4 ->
    #   "hisi(8): out of bound 6"      (Python returned a 2-tap answer)
    #             idx=0                -> "hisi(0): subscripts must be ...
    #                                      integers 1 to (2^63)-1"
    # An empty bank (tap_bk<=0) makes rng empty, which MATLAB accepts whatever
    # idx is, so only a non-empty bank is range-checked.
    if tap_bk > 0:
        if idx < 0:
            raise IndexError('hisi(%d): subscripts must be either integers '
                             '1 to (2^63)-1 or logicals' % (idx + 1))
        if idx + tap_bk > hisi.size:
            raise IndexError('hisi(%d): out of bound %d'
                             % (idx + tap_bk, hisi.size))

    sl = slice(idx, idx + tap_bk)
    flt_curval = hisi[sl].copy()

    if dfe_delta != 0:
        flt_curval_q = (
            np.floor(np.abs(flt_curval / curval) / dfe_delta)
            * dfe_delta * np.sign(flt_curval) * curval
        )
    else:
        flt_curval_q = flt_curval

    # MATLAB min() drops a NaN and returns the other operand; np.minimum
    # propagates it, so a NaN bmaxg poisoned every tap instead of disabling
    # the clip.
    # COM Octave: hisi=[1 2 3 4 5 6], idx=2, tap_bk=3, curval=4, bmaxg=NaN ->
    #   tap_coef = [0.5 0.75 1]        (Python returned [nan nan nan])
    tap_coef = np.fmin(np.abs(flt_curval_q / curval), bmaxg) * np.sign(flt_curval_q)
    hisi[sl] = hisi[sl] - curval * tap_coef

    # hisi_ref(rng)=0 is an ASSIGNMENT, which grows the array in MATLAB rather
    # than erroring; the Python slice assignment wrote nothing past the end.
    # COM Octave: hisi_ref=[9 9 9], idx=4, tap_bk=3 -> [9 9 9 0 0 0]
    #   (Python returned [9 9 9]).   hisi_ref=[] -> [0 0 0 0]
    if tap_bk > 0 and idx + tap_bk > hisi_ref.size:
        grown = np.zeros(idx + tap_bk)
        grown[:hisi_ref.size] = hisi_ref
        hisi_ref = grown
    hisi_ref[sl] = 0.0
    return hisi, tap_coef, hisi_ref
