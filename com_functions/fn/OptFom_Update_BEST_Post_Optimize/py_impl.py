import math
import numpy as np


# --- inline from dfe_clipper (MATLAB 5531-5544) ---
def _dfe_clipper(input_arr, max_threshold, min_threshold):
    inp = np.asarray(input_arr, dtype=float)
    hi = np.asarray(max_threshold, dtype=float)
    lo = np.asarray(min_threshold, dtype=float)
    is_row = inp.ndim <= 1 or (inp.ndim == 2 and inp.shape[0] == 1)
    if is_row:
        hi = hi.ravel()
        lo = lo.ravel()
    else:
        hi = hi.ravel().reshape(-1, 1)
        lo = lo.ravel().reshape(-1, 1)
    out = inp.copy()
    out[inp > hi] = hi[inp > hi]
    out[inp < lo] = lo[inp < lo]
    return out


# --- inline from FD_CTLE (MATLAB 1681-1683) ---
def _FD_CTLE(freq, f_z, f_p1, f_p2, kacdc_dB):
    freq = np.asarray(freq, dtype=float)
    num = 10 ** (kacdc_dB / 20) + 1j * freq / f_z
    den = (1 + 1j * freq / f_p1) * (1 + 1j * freq / f_p2)
    return num / den


# --- inline from OptFom_Calc_Hr (MATLAB 2874-2880) with its helpers ---
_BW_POLY = [1, 2.613126, 3.414214, 2.613126, 1]


def _bessel_poly(n):
    a = np.zeros(n + 1)
    for ii in range(n + 1):
        a[ii] = (math.factorial(2 * n - ii)
                 / (2 ** (n - ii) * math.factorial(ii) * math.factorial(n - ii)))
    return a


def _tukey_window(f, fr, fb_top):
    fperiod = 2 * (fb_top - fr)
    return np.where(
        f < fr, 1.0,
        np.where((f >= fr) & (f <= fb_top),
                 0.5 * np.cos(2 * np.pi * (f - fb_top) / fperiod - np.pi) + 0.5,
                 0.0))


def _OptFom_Calc_Hr(f, param, OP):
    f = np.asarray(f, dtype=float)
    H_bw = (1.0 / np.polyval(_BW_POLY, 1j * f / (param.fb_BW_cutoff * param.fb))
            if OP.Butterworth else np.ones(len(f)))
    if OP.Bessel_Thomson:
        a = _bessel_poly(param.BTorder)
        s = 1j * f / (param.fb_BT_cutoff * param.fb)
        H_bt = a[0] / np.polyval(a[::-1], s)
    else:
        H_bt = np.ones(len(f))
    H_rc = (_tukey_window(f, param.RC_Start, param.RC_end)
            if OP.Raised_Cosine else np.ones(len(f)))
    return H_bw * H_bt * H_rc


def OptFom_Update_BEST_Post_Optimize(BEST, f, param, OP):
    """Update BEST after EQ optimization loops finish (MATLAB lines 3843-3890).

    Computes cursor, H_r, CTLE gains, sampled precursors/postcursors, DFE taps.
    Returns BEST.
    """
    sbr = np.asarray(BEST.sbr, dtype=float).ravel()
    length_sbr = len(sbr)
    M = int(param.samples_per_ui)
    ndfe = int(param.ndfe)
    cursor_i = int(BEST.cursor_i)  # 0-based

    BEST.cursor = float(sbr[cursor_i])

    BEST.H_r = _OptFom_Calc_Hr(f, param, OP)

    # BEST.ctle is 1-BASED: it is copied straight from THIS.ctle_index, which
    # optimize_fom sets as `ctle_index + 1  # 1-based to match MATLAB`. The
    # comment here used to claim 0-based and the value was used unadjusted, so
    # every lookup below read one CTLE entry too high. It stayed invisible
    # because these fields feed only OptFom_Plot_Best_Results -- the COM path
    # recomputes ctle_gain in optimize_fom, which does convert -- and it only
    # raises IndexError when the winning CTLE is the LAST in the list.
    ctle_idx = int(BEST.ctle) - 1
    BEST.ctle_gain1 = _FD_CTLE(
        f,
        float(np.asarray(param.CTLE_fz).ravel()[ctle_idx]),
        float(np.asarray(param.CTLE_fp1).ravel()[ctle_idx]),
        float(np.asarray(param.CTLE_fp2).ravel()[ctle_idx]),
        float(BEST.gdc))

    ctle_type = getattr(param, 'CTLE_type', 'CL93')
    if ctle_type == 'CL93':
        BEST.H_low = 1.0
    elif ctle_type == 'CL120d':
        hp_idx = int(BEST.G_high_pass) - 1  # 1-based, from THIS.g_LP_index
        BEST.H_low = _FD_CTLE(
            f,
            float(np.asarray(param.f_HP).ravel()[hp_idx]),
            float(np.asarray(param.f_HP).ravel()[hp_idx]),
            1e200,
            float(np.asarray(param.g_DC_HP_values).ravel()[hp_idx]))
    elif ctle_type == 'CL120e':
        BEST.H_low = _FD_CTLE(
            f,
            float(np.asarray(param.f_HP_Z).ravel()[ctle_idx]),
            float(np.asarray(param.f_HP_P).ravel()[ctle_idx]),
            1e200, 0.0)

    BEST.ctle_gain = BEST.H_low * BEST.ctle_gain1 * BEST.H_r

    # Precursor times: all samples from index 0..cursor_i-1
    # MATLAB: (cursor_i/M:-1:1/M)*ui reversed(end:-1:2) = [1/M .. (cursor_i-1)/M]*ui
    # Python 0-based: [0..cursor_i-1] → times [0 .. (cursor_i-1)/M]*ui
    BEST.sampled_sbr_precursors_t = np.arange(0, cursor_i) / M * param.ui
    BEST.sampled_sbr_precursors = sbr[0:cursor_i]

    # Postcursor times: UI-spaced after cursor (excluding cursor itself)
    # MATLAB: cursor_i+M, cursor_i+2M, ... (1-based) → Python: cursor_i+M, cursor_i+2M, ...
    post_indices = np.arange(cursor_i + M, length_sbr, M)
    if len(post_indices) > 0:
        BEST.sampled_sbr_postcursors_t = post_indices / M * param.ui
        BEST.sampled_sbr_postcursors = sbr[post_indices]
    else:
        BEST.sampled_sbr_postcursors_t = np.array([])
        BEST.sampled_sbr_postcursors = np.array([])

    # DFE cursor times: cursor + k*ui for k=1..ndfe_passed
    ndfe_passed = int(getattr(param, 'ndfe_passed', ndfe))
    BEST.sampled_sbr_dfecursors_t = ((cursor_i + 1) / M + np.arange(1, ndfe_passed + 1)) * param.ui

    if getattr(param, 'Floating_DFE', False) and hasattr(BEST, 'floating_tap_locations'):
        floc = np.asarray(BEST.floating_tap_locations)
        BEST.sampled_sbr_fdfecursors_t = ((cursor_i + 1) / M + floc) * param.ui

    # DFE tap clipping
    n_post = len(BEST.sampled_sbr_postcursors)
    dfe_cursors = BEST.sampled_sbr_postcursors[:min(ndfe, n_post)]
    dfe_SBRcursors = dfe_cursors.copy()

    bmax = np.asarray(BEST.bmax).ravel()
    bmin = np.asarray(BEST.bmin).ravel()
    BEST.bmax = bmax
    BEST.bmin = bmin

    BEST.DFE_taps_mV = _dfe_clipper(
        dfe_cursors,
        BEST.cursor * bmax[:min(ndfe, len(bmax))],
        BEST.cursor * bmin[:min(ndfe, len(bmin))])

    if getattr(param, 'Floating_DFE', False) and hasattr(BEST, 'floating_tap_locations'):
        # 1-based (see OptFom_Compute_DFE); MATLAB 4143 indexes DFE_taps_mV with
        # it directly because MATLAB is 1-based, so Python must convert.
        floc_arr = np.asarray(BEST.floating_tap_locations, dtype=int) - 1
        BEST.FDFE_taps_mV = BEST.DFE_taps_mV[floc_arr]
    else:
        BEST.FDFE_taps_mV = np.array([])

    BEST.sampled_sbr_postcursors[:min(ndfe, n_post)] = dfe_SBRcursors - BEST.DFE_taps_mV

    return BEST
