"""
optimize_fom: EQ search loop — finds best CTLE/TXFFE/DFE/RxFFE settings for FOM.
MATLAB lines 8417–8793.
"""

import numpy as np
from types import SimpleNamespace



def _mround(x):
    """MATLAB round(): half away from zero, where Python's round() is banker's.

    Needed at the triple-transit-time site below because
    `2*sbr_peak_i / samples_per_ui` is a ratio of two INTEGERS, so it lands on
    .5 exactly whenever 2*sbr_peak_i is an odd multiple of samples_per_ui/2 --
    not the measure-zero case the conversion audit dismissed. Ledger #16 is the
    same shape (`nui = round(len/M)`) and was a live defect on 4 of 208 cases.
    Guarded by tests/test_integer_ratio_rounding.py.
    """
    import math
    x = float(x)
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    # Off a tie round() is exact, and unlike floor(x + 0.5) it does not
    # send 0.49999999999999994 to 1: that sum is exactly 1.0 in binary.
    return int(round(x))


# ── Optional EQ-search trajectory logging (off by default; no effect on COM) ──
# Set SWEEP_LOG_CSV to a path to capture one row per TX-FFE candidate considered
# (for any method: full grid / legacy local search / adaptive). Rows are buffered
# in memory and flushed to CSV when optimize_fom returns. This is the data source
# for the sweep-vs-adaptive comparison visualisation.
SWEEP_LOG_CSV = None
SWEEP_METHOD_LABEL = ''
_SWEEP_ROWS = []
_SWEEP_HEADER = ['method', 'gffe_index', 'ctle_index', 'lp_index', 'txffe_index',
                 'tx_taps', 'candidate_FOM', 'best_FOM', 'evaluated', 'eval_count',
                 'skip_reason']


def _sweep_record(row):
    if SWEEP_LOG_CSV is not None:
        _SWEEP_ROWS.append(row)


def _sweep_flush():
    if SWEEP_LOG_CSV is None:
        return

    def _cell(v):
        if isinstance(v, str):
            return '"' + v + '"'
        if isinstance(v, float):
            return 'nan' if v != v else f'{v:.6g}'
        return str(v)

    with open(SWEEP_LOG_CSV, 'w', newline='') as f:
        f.write(','.join(_SWEEP_HEADER) + '\n')
        for r in _SWEEP_ROWS:
            f.write(','.join(_cell(v) for v in r) + '\n')
    _SWEEP_ROWS.clear()


def optimize_fom(OP, param, chdata, sigma_bn, do_C2M,
                 _OptFom_Initialize_Loop_Struct_fn=None,
                 _OptFom_Build_TXFFE_fn=None,
                 _OptFom_Calculate_Settings_fn=None,
                 _FD_CTLE_fn=None,
                 _OptFom_Compute_CTLE_fn=None,
                 _OptFom_Calc_Noise_XC_fn=None,
                 _get_sigma_eta_ACCM_noise_fn=None,
                 _get_sigma_noise_fn=None,
                 _OptFom_Compute_TXFFE_fn=None,
                 _OptFom_Find_Sample_Point_fn=None,
                 _OptFom_Setup_Sampler_Sweep_fn=None,
                 _OptFom_Itick_BoxSearch_fn=None,
                 _OptFom_Itick_LocalSearch_fn=None,
                 _OptFom_Compute_RxFFE_fn=None,
                 _OptFom_Compute_DFE_fn=None,
                 _OptFom_Calc_Noise_fn=None,
                 _OptFom_Calc_FOM_fn=None,
                 _OptFom_Local_Search_fn=None,
                 _OptFom_Adaptive_Local_Search_fn=None,
                 _get_PSDs_fn=None,
                 _OptFom_Set_Best_Itick_fn=None,
                 _OptFom_Update_Best_Setttings_fn=None,
                 _OptFom_Update_Best_Settings_EQ_Failed_fn=None,
                 _OptFom_Update_BEST_Post_Optimize_fn=None,
                 _OptFom_Create_Output_fn=None):

    result = SimpleNamespace()

    # ── Initialize loop struct ─────────────────────────────────────────────
    THIS = _OptFom_Initialize_Loop_Struct_fn()

    # ── Initialize parameters ──────────────────────────────────────────────
    f = np.asarray(chdata[0].faxis).ravel()

    # Normalize ts_sample_adj_range: single value → [0, value]
    tsar = np.asarray(param.ts_sample_adj_range, dtype=float).ravel()
    if len(tsar) == 1:
        tsar = np.array([0.0, tsar[0]])
    param.ts_sample_adj_range = tsar
    full_sample_range = np.arange(int(tsar[0]), int(tsar[1]) + 1)

    param.ndfe_passed = int(param.ndfe)
    if getattr(param, 'Floating_DFE', False):
        param.ndfe = int(param.N_bmax)

    if getattr(OP, 'RxFFE', False) and getattr(OP, 'FFE_OPT_METHOD', '') == 'MMSE':
        OP.RxFFE_with_MMSE = 1
    else:
        OP.RxFFE_with_MMSE = 0

    OP.itick_box_size = 5

    Gffe_values = np.asarray(param.cursor_gain).ravel()
    if not getattr(OP, 'RxFFE', False):
        Gffe_values = np.array([0.0])

    if getattr(param, 'CTLE_type', '') in ('CL93', 'CL120e'):
        param.g_DC_HP_values = np.array([0.0])

    lf_indx = len(np.asarray(param.g_DC_HP_values).ravel())

    BEST = SimpleNamespace(FOM=float('-inf'), cursor_i=None)
    sbr = np.array([])

    # ── T_O ───────────────────────────────────────────────────────────────
    if do_C2M:
        loop_count = [1, 2]
        T_O = int(max(0, np.floor((float(param.T_O) / 1000.0) * int(param.samples_per_ui))))
    else:
        loop_count = [1]
        T_O = 0

    if getattr(OP, 'Optimize_loop_speed_up', 0) == 1:
        OP.BinSize = 1e-4
        OP.impulse_response_truncation_threshold = 1e-3

    # ── Used to speed up FFE ───────────────────────────────────────────────
    pulse_struc = [SimpleNamespace(pulse_ctle_circshift=None)]
    ctle_response_updated = 1

    # ── Build TXFFE values ─────────────────────────────────────────────────
    txffe_matrix, cur, txffe_sweep_indices, FULL_tx_index_vector, txffe_cursor_vector = \
        _OptFom_Build_TXFFE_fn(param)
    num_txffe_runs = txffe_matrix.shape[0]
    param.cursor_index = cur

    # ── Loop-independent settings ──────────────────────────────────────────
    SETTINGS = _OptFom_Calculate_Settings_fn(txffe_matrix, chdata, param, OP)
    if not hasattr(SETTINGS, 'delta_sbr') or SETTINGS.delta_sbr is None:
        SETTINGS.delta_sbr = None

    # FOM history for the adaptive local search (Hansel branch); harmless otherwise
    FOM_history = []
    iter_count = 0

    # Sweep-trajectory logging counters (only used when SWEEP_LOG_CSV is set)
    n_eval = 0
    if SWEEP_LOG_CSV is not None:
        _SWEEP_ROWS.clear()

    # ── EQ search dimensions ───────────────────────────────────────────────
    n_gffe = len(Gffe_values)
    n_ctle = len(np.asarray(param.ctle_gdc_values).ravel())
    n_samp = len(full_sample_range)
    FOM_TRACKER = np.zeros((n_gffe, n_ctle, lf_indx, num_txffe_runs, n_samp))

    pxi = 0
    runs = n_ctle * lf_indx * n_gffe * num_txffe_runs

    print(f'  optimize_fom: {n_gffe} gffe × {n_ctle} CTLE × {lf_indx} g_DC_HP × '
          f'{num_txffe_runs} TXFFE × {n_samp} itick = {runs * n_samp} evals', flush=True)

    # ── EQ Loop ────────────────────────────────────────────────────────────
    for i in loop_count:
        if do_C2M:
            print(f'  C2M pass {i}/{len(loop_count)}', flush=True)
        for Gffe_index, gffe_val in enumerate(Gffe_values):
            param.current_ffegain = gffe_val
            ctle_gdc_values = np.asarray(param.ctle_gdc_values).ravel()
            for ctle_index, g_dc in enumerate(ctle_gdc_values):
                print(f'    CTLE {ctle_index + 1}/{len(ctle_gdc_values)}  '
                      f'g_dc={g_dc:.2f}  best_FOM={BEST.FOM:.2f} dB', flush=True)
                THIS.ctle_index = ctle_index + 1  # 1-based to match MATLAB
                THIS.g_dc = float(g_dc)
                CTLE_fp1 = float(np.asarray(param.CTLE_fp1).ravel()[ctle_index])
                CTLE_fp2 = float(np.asarray(param.CTLE_fp2).ravel()[ctle_index])
                CTLE_fz = float(np.asarray(param.CTLE_fz).ravel()[ctle_index])
                ctle_gain = _FD_CTLE_fn(f, CTLE_fz, CTLE_fp1, CTLE_fp2, THIS.g_dc)
                ctle_gain_xc = _FD_CTLE_fn(
                    np.asarray(SETTINGS.f_xc).ravel(), CTLE_fz, CTLE_fp1, CTLE_fp2, THIS.g_dc)

                g_DC_HP_values = np.asarray(param.g_DC_HP_values).ravel()
                for g_LP_index, g_lp in enumerate(g_DC_HP_values):
                    THIS.g_DC_low = float(g_lp)
                    THIS.g_LP_index = g_LP_index + 1  # 1-based

                    qual_arr = np.atleast_2d(np.asarray(SETTINGS.qual, dtype=float))
                    g_row = min(g_LP_index, qual_arr.shape[0] - 1)
                    c_col = min(ctle_index, qual_arr.shape[1] - 1)
                    if float(qual_arr[g_row, c_col]) == 0.0:
                        pxi += num_txffe_runs
                        continue

                    if getattr(OP, 'INCLUDE_CTLE', 0) == 1:
                        ctle_response_updated = 1

                    chdata, THIS.H_ctf, H_low_xc, H_ctf2 = _OptFom_Compute_CTLE_fn(
                        chdata, ctle_gain, THIS, SETTINGS.f_xc, param, OP)

                    Noise_XC = _OptFom_Calc_Noise_XC_fn(H_low_xc, ctle_gain_xc, SETTINGS, param, OP)

                    THIS.sigma_N = _get_sigma_eta_ACCM_noise_fn(
                        chdata, param, SETTINGS.H_sy, SETTINGS.H_r, THIS.H_ctf)

                    if getattr(OP, 'RX_CALIBRATION', False):
                        # MATLAB: THIS.sigma_ne = get_sigma_noise(...) takes only the 1st
                        # output (sigma_NE); Python returns (sigma_NE, sigma_HP).
                        THIS.sigma_ne = _get_sigma_noise_fn(H_ctf2, param, chdata, sigma_bn)[0]
                    else:
                        THIS.sigma_ne = 0.0

                    gdc_min = float(getattr(param, 'GDC_MIN', 0))
                    if gdc_min != 0 and THIS.g_dc + THIS.g_DC_low > gdc_min:
                        pxi += num_txffe_runs
                        continue

                    THIS.PSD_results = None
                    if OP.RxFFE_with_MMSE:
                        OP.WO_TXFFE = 1
                        THIS.PSD_results = _get_PSDs_fn(
                            THIS.PSD_results, [], [], [], THIS.g_dc, THIS.g_DC_low,
                            param, chdata, OP)

                    # ── TXFFE Loop ─────────────────────────────────────────
                    _txffe_report_interval = max(1, num_txffe_runs // 10)
                    for TK in range(num_txffe_runs):
                        pxi += 1
                        if num_txffe_runs > 1 and TK % _txffe_report_interval == 0:
                            print(f'      TXFFE {TK + 1}/{num_txffe_runs}  '
                                  f'best={BEST.FOM:.2f} dB', flush=True)
                        txffe_cur = float(np.asarray(txffe_cursor_vector[TK]).ravel()[0])
                        if txffe_cur < float(getattr(param, 'tx_ffe_c0_min', -np.inf)):
                            continue

                        THIS.tx_index_vector = FULL_tx_index_vector[TK, :]

                        if getattr(param, 'LOCAL_SEARCH', 0) > 0 and not np.isinf(BEST.FOM):
                            iter_count += 1
                            # NonZeroLSMethod: 1 -> Hansel adaptive search, else legacy
                            if (getattr(param, 'NonZeroLSMethod', 0) == 1
                                    and _OptFom_Adaptive_Local_Search_fn is not None):
                                # 4p16p0 L9017 passes Overwrite_Min_Radius as the
                                # second positional argument; here it is a keyword
                                # so the 4p15p0 call shape is unchanged.
                                skip_it = _OptFom_Adaptive_Local_Search_fn(
                                    param.LOCAL_SEARCH, BEST, THIS, FOM_history,
                                    iter_count, num_txffe_runs,
                                    Overwrite_Min_Radius=getattr(
                                        param, 'Overwrite_Min_Radius', None),
                                    matlab_version=getattr(
                                        param, 'matlab_version', '4p15p0'))
                            else:
                                skip_it = _OptFom_Local_Search_fn(
                                    param.LOCAL_SEARCH, BEST, THIS, txffe_sweep_indices)
                            if skip_it:
                                if SWEEP_LOG_CSV is not None:
                                    _sweep_record([SWEEP_METHOD_LABEL, Gffe_index, ctle_index,
                                                   g_LP_index, TK,
                                                   np.array2string(np.asarray(THIS.tx_index_vector).ravel()),
                                                   float('nan'), float(BEST.FOM), 0, n_eval, 'search_skip'])
                                continue

                        cand_fom = float('-inf')  # best FOM over this candidate's itick sweep
                        THIS.txffe = txffe_matrix[TK, :]
                        sbr, chdata, pulse_struc = _OptFom_Compute_TXFFE_fn(
                            chdata, pulse_struc, THIS.txffe, ctle_response_updated, param, OP)
                        sbr_from_txffe = sbr.copy()
                        if ctle_response_updated:
                            ctle_response_updated = 0

                        raw_cursor_i, no_zero_crossing, sbr_peak_i = _OptFom_Find_Sample_Point_fn(
                            sbr, param, OP, SETTINGS.Peak_Search_Range)
                        if no_zero_crossing:
                            continue

                        triple_transit_time = _mround(int(sbr_peak_i) * 2 / int(param.samples_per_ui)) + 20
                        if SETTINGS.min_number_of_UI_in_response < triple_transit_time:
                            SETTINGS.min_number_of_UI_in_response = triple_transit_time

                        # ── ITICK loop ─────────────────────────────────────
                        loop_range_it, BEST, middle_search, box_search, cluster, box_mid = \
                            _OptFom_Setup_Sampler_Sweep_fn(full_sample_range, BEST, OP)

                        for itickn in loop_range_it:
                            if box_search:
                                THIS.itick, BEST, skip_it = _OptFom_Itick_BoxSearch_fn(
                                    itickn, cluster, BEST, box_mid, OP.itick_box_size)
                                if skip_it:
                                    continue
                            else:
                                THIS.itick = full_sample_range[itickn]

                            THIS.cursor_i = int(raw_cursor_i) + int(THIS.itick)
                            sbr = sbr_from_txffe.copy()

                            if getattr(param, 'LOCAL_SEARCH', 0) > 0:
                                skip_it = _OptFom_Itick_LocalSearch_fn(
                                    THIS.itick, middle_search, BEST, param.LOCAL_SEARCH)
                                if skip_it:
                                    continue

                            if getattr(OP, 'RxFFE', False):
                                sbr, THIS, skip_it = _OptFom_Compute_RxFFE_fn(
                                    sbr, THIS, Noise_XC, chdata, param, OP)
                                if skip_it:
                                    continue

                            # Cursor amplitude.
                            # AUDIT FINDING B16-D20: cursor_sample_index and
                            # OptFom_Find_Sample_Point return 0-BASED indices (the
                            # convention get_PSDs, get_pdf and OptFom_Compute_DFE use),
                            # so no -1 belongs here; MATLAB L8788-8801 reads
                            # sbr(cursor_i) with no offset.
                            #
                            # History: removing the -1 was tried on 2026-08-13 while the
                            # package die-network bug was still present and made COM
                            # agreement WORSE, so it was reverted. Once the die LC
                            # sections were fixed (make_full_pkg/read_s4p_files/s21_pkg
                            # kept only 1 of 3 sections) the two errors were shown to have
                            # been compensating, and D20 is now applied.
                            cursor = float(sbr[THIS.cursor_i])
                            THIS.A_p = float(sbr[int(sbr_peak_i)])
                            THIS.A_s = float(param.R_LM) * cursor / (int(param.levels) - 1)

                            if SETTINGS.delta_sbr is None:
                                SETTINGS.delta_sbr = sbr.copy()
                            sbr = sbr.ravel()

                            # Far cursors and precursors (eq 93A-27)
                            far_start = THIS.cursor_i - T_O + int(param.samples_per_ui) * (int(param.ndfe) + 1)
                            far_start = max(far_start, 0)
                            THIS.far_cursors = sbr[far_start::int(param.samples_per_ui)]

                            pre_start = THIS.cursor_i - int(param.samples_per_ui)
                            if pre_start >= 0:
                                pre_rev = sbr[pre_start::-int(param.samples_per_ui)]
                                THIS.precursors = pre_rev[::-1]
                            else:
                                THIS.precursors = np.array([])

                            # Early FOM bound check (skip if no chance of beating best)
                            if not OP.RxFFE_with_MMSE:
                                combined = np.concatenate([THIS.precursors, THIS.far_cursors])
                                sigma_ISI_ignoreDFE = float(param.sigma_X) * np.linalg.norm(combined)
                                if sigma_ISI_ignoreDFE > 0 and 20 * np.log10(
                                        abs(THIS.A_s) / sigma_ISI_ignoreDFE) < BEST.FOM:
                                    continue

                            # Extend sbr if too short
                            sbr_required_length = THIS.cursor_i + int(param.samples_per_ui) * (int(param.ndfe) + 1)
                            if len(sbr) < sbr_required_length:
                                sbr = np.append(sbr, np.zeros(sbr_required_length - len(sbr)))

                            # Solve DFE
                            THIS, param = _OptFom_Compute_DFE_fn(sbr, THIS, param, do_C2M, T_O)

                            # Calculate all noise
                            THIS, abort_status = _OptFom_Calc_Noise_fn(
                                THIS, BEST.FOM, sbr, SETTINGS, chdata, param, OP)
                            if abort_status == 1:
                                continue
                            elif abort_status == 2:
                                break

                            # Find FOM
                            if not OP.RxFFE_with_MMSE:
                                THIS.FOM, skip_loop = _OptFom_Calc_FOM_fn(
                                    chdata, do_C2M, THIS, param, OP, sbr)
                                if skip_loop:
                                    continue

                            # Update FOM history for the adaptive local search,
                            # capped at the itick sampler-sweep length (loop_range)
                            if getattr(param, 'LOCAL_SEARCH', 0) > 0:
                                FOM_history.append(float(THIS.FOM))
                                max_history_length = len(loop_range_it)
                                if len(FOM_history) > max_history_length:
                                    FOM_history = FOM_history[-max_history_length:]

                            # Track FOM
                            itick_matches = np.where(full_sample_range == int(THIS.itick))[0]
                            if len(itick_matches) > 0:
                                itick_idx = itick_matches[0]
                                FOM_TRACKER[Gffe_index, ctle_index, g_LP_index, TK, itick_idx] = THIS.FOM
                            if SWEEP_LOG_CSV is not None:
                                cand_fom = max(cand_fom, float(THIS.FOM))

                            # Update best settings
                            BEST = _OptFom_Set_Best_Itick_fn(THIS, BEST)
                            if THIS.FOM > BEST.FOM:
                                BEST = _OptFom_Update_Best_Setttings_fn(BEST, THIS, sbr, chdata, param, OP)
                                print(f'      ** new best FOM = {BEST.FOM:.3f} dB '
                                      f'(CTLE {ctle_index + 1}, TXFFE {TK + 1}, itick={THIS.itick})',
                                      flush=True)

                        # End of this candidate's itick sweep — record it as evaluated
                        if SWEEP_LOG_CSV is not None:
                            n_eval += 1
                            _sweep_record([SWEEP_METHOD_LABEL, Gffe_index, ctle_index,
                                           g_LP_index, TK,
                                           np.array2string(np.asarray(THIS.tx_index_vector).ravel()),
                                           float(cand_fom), float(BEST.FOM), 1, n_eval, ''])

        if do_C2M:
            if BEST.FOM == float('-inf'):
                param.Min_VEO_Test = 0
            else:
                break

    # ── Check EQ failure ──────────────────────────────────────────────────
    cursor_i_empty = (BEST.cursor_i is None or
                      (hasattr(BEST.cursor_i, '__len__') and len(np.asarray(BEST.cursor_i).ravel()) == 0))
    if cursor_i_empty:
        result.eq_failed = True
        BEST = _OptFom_Update_Best_Settings_EQ_Failed_fn(BEST, THIS, sbr, chdata, param, OP)
        if do_C2M:
            _sweep_flush()
            return result
    else:
        result.eq_failed = False

    # ── Post-optimize ─────────────────────────────────────────────────────
    f_post = np.arange(1e8, 100e9 + 1e8, 1e8)
    length_sbr = len(np.asarray(BEST.sbr).ravel())
    t_out = np.arange(length_sbr) * float(param.ui) / float(param.samples_per_ui)

    BEST = _OptFom_Update_BEST_Post_Optimize_fn(BEST, f_post, param, OP)

    # ── Create output ─────────────────────────────────────────────────────
    result = _OptFom_Create_Output_fn(result, BEST, t_out, chdata, param, OP)
    result.FOM_TRACKER = FOM_TRACKER  # full FOM landscape, for the convergence plot
    # The itick values indexing FOM_TRACKER's last axis. FOM_TRACKER was already
    # exported but the axis it is measured against was not, so a FOM-versus-
    # sampling-phase plot could not be drawn from it. Carrying the axis costs a
    # reference to an array the search already built and changes no result.
    result.sample_range = np.asarray(full_sample_range)
    # EQ sweep dimensions (for the engineering-export run summary; no effect on COM)
    result.sweep_dims = SimpleNamespace(
        n_gffe=int(n_gffe), n_ctle=int(n_ctle), n_g_DC_HP=int(lf_indx),
        n_txffe=int(num_txffe_runs), n_itick=int(n_samp),
        total_evals=int(runs * n_samp),
        local_search=int(getattr(param, 'LOCAL_SEARCH', 0)))

    _sweep_flush()  # write the per-candidate trajectory CSV when logging is enabled
    return result
