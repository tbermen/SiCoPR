# ============================================================
# MATLAB→Python translation notes for OptFom_Create_Output
# MATLAB lines: 3418–3504
# ============================================================
# cursor_i is 0-based in Python.
# DFE_taps_i: MATLAB cursor_i+(1:ndfe)*M (1-based) = Python cursor_i+arange(1,ndfe+1)*M (0-based).
# BEST.IR padding: np.pad(..., (0, extra)).
# A_f (eq 163A-3): sum of truncated PR / M.
# its: 0-based argmax of uneq_pulse_response.
# iend/ibeg: derived from its (0-based); clamped to valid range.
# SRn circshift loop: np.roll + zero-fill.
# i20/i80: 0-based; difference i80-i20 is sample count (no unit issues).
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

from types import SimpleNamespace


def OptFom_Create_Output(result, BEST, t, chdata, param, OP):
    M = int(param.samples_per_ui)

    result.cur = param.cursor_index
    result.txffe = BEST.txffe
    result.ctle = BEST.ctle
    result.best_G_high_pass = BEST.G_high_pass
    # Genuine CTLE frequency response captured for the winning setting
    # (forwarded for the engineering .mat export; unused by COM math).
    result.H_ctf = getattr(BEST, 'H_ctf', None)
    result.sdd21ctf = getattr(BEST, 'sdd21ctf', None)
    result.DFE_taps = BEST.dfetaps
    result.DFE_taps_i = (int(BEST.cursor_i)
                         + np.arange(1, int(param.ndfe) + 1) * M)
    if param.Floating_DFE:
        result.floating_tap_locations = BEST.floating_tap_locations
        result.floating_tap_coef = BEST.floating_tap_coef
    if param.Floating_RXFFE:
        result.floating_tap_locations = BEST.floating_tap_locations
    result.A_s = BEST.A_s
    result.t_s = BEST.cursor_i
    result.itick = BEST.itick
    result.sigma_N = BEST.sigma_N
    result.h_J = BEST.h_J
    result.FOM = BEST.FOM
    if not OP.TDMODE:
        best_ir = np.asarray(BEST.IR, dtype=float)
        best_sbr = np.asarray(BEST.sbr, dtype=float)
        if len(best_ir) < len(best_sbr):
            best_ir = np.pad(best_ir, (0, len(best_sbr) - len(best_ir)))
        result.IR = best_ir
    result.t = t
    result.sbr = BEST.sbr
    if OP.RxFFE:
        result.RxFFE = BEST.RxFFE
        # MATLAB strcmp(OP.FFE_OPT_METHOD,'MMSE') is CASE SENSITIVE.  COM
        # Octave with OP.RxFFE=1: 'MMSE' adds result.PSD_results and
        # result.MMSE_results; 'mmse' and 'Mmse' add neither.  .upper() here
        # attached both fields on every spelling.
        if str(getattr(OP, 'FFE_OPT_METHOD', '')) == 'MMSE':
            result.PSD_results = BEST.PSD_results
            result.MMSE_results = BEST.MMSE_results

    PR = np.asarray(chdata[0].uneq_pulse_response, dtype=float)
    result.A_p = float(_mmax(PR))
    its = int(np.where(PR >= _mmax(PR))[0][0])  # 0-based

    # Equation 163A-3 window around peak
    iend = its + int(param.N_v) * M - M // 2
    ibeg = its - int(param.D_p) * M - M // 2
    iend = min(iend, len(PR) - 1)
    ibeg = max(ibeg, 0)
    PR = PR[ibeg:iend + 1]

    result.A_f = float(np.sum(PR / M))  # eq 163A-3

    SRn = PR.copy()
    for ik in range(1, len(PR) // M + 1):
        SPR = np.roll(PR, M * ik)
        SPR[:ik * M] = 0
        SRn = SRn + SPR

    A_f = result.A_f
    i20_arr = np.where(SRn >= 0.20 * A_f)[0]
    i80_arr = np.where(SRn >= 0.80 * A_f)[0]
    i20 = int(i20_arr[0]) if len(i20_arr) > 0 else 0
    i80 = int(i80_arr[0]) if len(i80_arr) > 0 else len(SRn) - 1
    result.Tr_measured_from_step = (i80 - i20) / (param.fb * M)
    # MATLAB divides straight through: A_p/0 is Inf and 0/0 is NaN, neither of
    # which is zero.  COM Octave, a PR window summing to zero:
    # result.Pmax_by_Vf = inf where this returned 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        result.Pmax_by_Vf = float(np.float64(result.A_p) / np.float64(result.A_f))

    result.ISI = BEST.ISI
    # Same for 20*log10(BEST.A_p/BEST.ISI).  COM Octave:
    #   A_p=1,  ISI=0 -> Inf                    (this agreed)
    #   A_p=0,  ISI=0 -> NaN                    (this returned Inf)
    #   A_p=-1, ISI=0.05 -> 26.020599913279625 + 27.287527076836827i,
    #                       because MATLAB log10 of a negative is complex
    #                       (np.log10 gives NaN).
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.float64(BEST.A_p) / np.float64(BEST.ISI)
        if ratio < 0:
            result.SNR_ISI = complex(20 * np.log10(np.complex128(ratio)))
        else:
            result.SNR_ISI = float(20 * np.log10(ratio))
    result.best_current_ffegain = BEST.ffegain
    result.best_bmax = BEST.bmax
    result.best_bmin = BEST.bmin
    result.tail_RSS = BEST.tail_RSS
    result.sampled_best_sbr_precursors_t = BEST.sampled_sbr_precursors_t
    result.sampled_best_sbr_postcursors_t = BEST.sampled_sbr_postcursors_t
    return result
