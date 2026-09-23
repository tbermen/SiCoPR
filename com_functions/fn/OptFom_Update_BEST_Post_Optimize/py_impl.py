import numpy as np
from com_functions.fn.FD_CTLE.py_impl import FD_CTLE as _FD_CTLE
from com_functions.fn.OptFom_Calc_Hr.py_impl import OptFom_Calc_Hr as _OptFom_Calc_Hr
from com_functions.fn.dfe_clipper.py_impl import dfe_clipper as _dfe_clipper


# --- inline from dfe_clipper (MATLAB 5531-5544) ---


# --- inline from FD_CTLE (MATLAB 1681-1683) ---


# OptFom_Calc_Hr is IMPORTED, not copied. The private copy that used to live
# here was the pre-oracle form and carried all three defects its own directory
# had already fixed: `_tukey_window` was an element-wise np.where where
# MATLAB's Tukey_Window CONCATENATES three counted pieces, `if OP.Butterworth`
# is not MATLAB's `if` on an array, and np.ones(len(f)) is not
# ones(1,length(f)). COM Octave, f=[25 0 35 15 40 20 45]*1e9 with
# RC_Start=20e9, RC_end=40e9: H_r(1) = -0.69940492382039898-0.65853321494420236j
# where the copy gave -0.59697944412453186-0.562093258433913j (5.8x relative).


def _mround_arr(x):
    """MATLAB round() on an array: halves go away from zero, where np.round
    takes them to even. Only exact ties are corrected; adding 0.5 and
    truncating would send 0.49999999999999994 to 1, since that sum is exactly
    1.0 in double precision."""
    x = np.asarray(x, dtype=float)
    tie = np.abs(x - np.trunc(x)) == 0.5
    return np.where(tie, np.trunc(x) + np.copysign(1.0, x), np.round(x))


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
    # A 1-based index of 0 or less is out of bounds for MATLAB; in Python it
    # wraps round to the LAST entry and answers with the wrong CTLE.
    # COM Octave, BEST.ctle=0: "param(0): subscripts must be either integers
    # 1 to (2^63)-1 or logicals".
    if ctle_idx < 0:
        raise IndexError('OptFom_Update_BEST_Post_Optimize: BEST.ctle(%r): '
                         'subscripts must be integers 1 or greater'
                         % (BEST.ctle,))
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
        # COM Octave, BEST.G_high_pass=0: "param(0): subscripts must be either
        # integers 1 to (2^63)-1 or logicals".
        if hp_idx < 0:
            raise IndexError('OptFom_Update_BEST_Post_Optimize: '
                             'BEST.G_high_pass(%r): subscripts must be '
                             'integers 1 or greater' % (BEST.G_high_pass,))
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

    cur1 = cursor_i + 1                      # MATLAB's 1-based cursor index

    # MATLAB `(cursor_i/M : -1 : 1/M)` steps by a whole UI, NOT by 1/M, so the
    # precursors are the UI-spaced samples before the cursor -- indices
    # cursor_i-M, cursor_i-2M, ... -- and `(end:-1:2)` drops the cursor and
    # puts them in ascending order. Taking every sample from 0 to cursor_i-1
    # returned M times too many and mislabelled their times.
    # COM Octave, cursor_i=33, samples_per_ui=8: four precursors at t/ui =
    # 0.125, 1.125, 2.125, 3.125 (1-based sbr indices 1, 9, 17, 25), not 32.
    n_colon = int(np.floor((cur1 - 1) / M)) + 1        # numel of that colon
    pre_t_ui = cur1 / M - np.arange(n_colon - 1, 0, -1)
    BEST.sampled_sbr_precursors_t = pre_t_ui * param.ui
    # MATLAB round() is half AWAY FROM ZERO; np.round is half-to-even, and
    # this is a real translation of `sbr(round(...))`, not an integrality
    # test, so the tie rule decides which sample is read.
    BEST.sampled_sbr_precursors = sbr[_mround_arr(pre_t_ui * M).astype(int) - 1]

    # Postcursor times: UI-spaced after cursor (excluding cursor itself).
    # MATLAB's t is the 1-BASED index over M, so the 0-based index needs the
    # +1 back. COM Octave, cursor_i=33, M=8: the first postcursor time is
    # 41/8*ui = 9.647e-11, not the 40/8*ui = 9.412e-11 the port returned.
    post_indices = np.arange(cursor_i + M, length_sbr, M)
    if len(post_indices) > 0:
        BEST.sampled_sbr_postcursors_t = (post_indices + 1) / M * param.ui
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
    # MATLAB `BEST.sampled_sbr_postcursors(1:param.ndfe)` is a READ, so it
    # errors as soon as there are fewer postcursors than taps.  COM Octave,
    # ndfe=30 with 20 postcursors: "BEST(30): out of bound 20 (dimensions are
    # 1x20)"; cursor_i=197 leaves none at all: "BEST(4): out of bound 0".
    # min(ndfe, n_post) shortened the tap set instead, and answered whenever
    # bmax happened to be short enough to line up.
    if ndfe > n_post:
        raise IndexError('OptFom_Update_BEST_Post_Optimize: '
                         'sampled_sbr_postcursors(%d): out of bound %d'
                         % (ndfe, n_post))
    dfe_cursors = BEST.sampled_sbr_postcursors[:ndfe]
    dfe_SBRcursors = dfe_cursors.copy()

    bmax = np.asarray(BEST.bmax).ravel()
    bmin = np.asarray(BEST.bmin).ravel()
    BEST.bmax = bmax
    BEST.bmin = bmin
    if ndfe > len(bmax) or ndfe > len(bmin):
        raise IndexError('OptFom_Update_BEST_Post_Optimize: bmax/bmin(%d): '
                         'out of bound %d/%d' % (ndfe, len(bmax), len(bmin)))

    BEST.DFE_taps_mV = _dfe_clipper(
        dfe_cursors,
        BEST.cursor * bmax[:ndfe],
        BEST.cursor * bmin[:ndfe])

    if getattr(param, 'Floating_DFE', False) and hasattr(BEST, 'floating_tap_locations'):
        # 1-based (see OptFom_Compute_DFE); MATLAB 4143 indexes DFE_taps_mV with
        # it directly because MATLAB is 1-based, so Python must convert.
        floc_arr = np.asarray(BEST.floating_tap_locations, dtype=int) - 1
        BEST.FDFE_taps_mV = BEST.DFE_taps_mV[floc_arr]
    else:
        BEST.FDFE_taps_mV = np.array([])

    BEST.sampled_sbr_postcursors[:ndfe] = dfe_SBRcursors - BEST.DFE_taps_mV

    return BEST
