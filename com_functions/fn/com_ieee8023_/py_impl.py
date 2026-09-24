"""
com_ieee8023_: main COM computation entry point.
MATLAB lines 1–904.

Accepts pre-loaded param/OP/chdata (MATLAB file-I/O and GUI calls are skipped).
All callee functions are injected for testability.
"""

import numpy as np
from types import SimpleNamespace

# Opt-in diagnostic, off by default (same convention as SWEEP_LOG_CSV /
# ALS_LOG_CSV): compute the eye contour and timing bathtub for PLOTTING even
# when MLSE is enabled.
#
#   import sicopr; sicopr.EYE_PLOT_UNDER_MLSE = True
#
# MATLAB gates the eye on OP.MLSE == 0 (4p15p0 L620) and this port follows it,
# so by default neither tool emits an eye or bathtub under MLSE. But MLSE is
# applied later (L667) and the eye is computed from the pre-MLSE PDF/CDF, so the
# chart is perfectly well defined -- it is the DFE-only eye, which is what an
# eye diagram is regardless of what post-processing follows.
#
# Setting this changes NO reported value: the extra call's return values are
# discarded, and its only side effect is the plotting side-channel
# chdata[0].timing_bathtub. Verified against the full 208-case corpus.
EYE_PLOT_UNDER_MLSE = False


def _save_case_outputs(OP, param, chdata, fom_result, Noise_Struct, PDF, CDF,
                       COM_SNR_Struct, output_args, case_i):
    """Write one package-test-case's CSV results + standard figures.

    Output goes to <OP.RESULT_DIR>/case_<NN>/ — controlled by the config's
    SAVE_FIGURES / CSV_REPORT fields (no separate runner needed).
    """
    import os as _os
    result_dir = str(getattr(OP, 'RESULT_DIR', '') or 'results')
    case_dir = _os.path.join(result_dir, 'case_%02d' % int(case_i))
    _os.makedirs(case_dir, exist_ok=True)
    if getattr(OP, 'CSV_REPORT', 0):
        try:
            Write_CSV(output_args, _os.path.join(case_dir, 'results.csv'))
        except Exception as _e:
            print('    [save] CSV skipped: %s' % _e)
    if getattr(OP, 'SAVE_FIGURES', 0):
        try:
            import com_plots
            com_plots.save_case_figures(case_dir, param, OP, chdata, fom_result,
                                        Noise_Struct, PDF, CDF, COM_SNR_Struct)
        except Exception as _e:
            print('    [save] figures skipped: %s' % _e)


def _checkpoint(stage, package_testcase_i, **values):
    """Save every struct the pipeline holds at one stage boundary.

    The Python twin of octave/patches/com_checkpoint.m, for the checkpoint
    harness (tests/test_octave_checkpoints.py). A no-op unless the environment
    variable COM_CHECKPOINT_DIR names a directory, which nothing in a normal run
    sets, so it never changes a result.

    Each value is pickled on its own, at the moment of the call: the pipeline
    mutates chdata and its kin in place, and a snapshot taken later would
    record the later state -- the aliasing class that produced the
    BEST.PSD_results defect. A value that cannot be pickled is reported and
    skipped rather than stopping the run, because the result must not depend
    on whether it was being observed. File names follow com_checkpoint.m:
    <stage>_pc<k>.pkl, with _2, _3 for a stage reached twice.
    """
    import os as _os
    d = _os.environ.get('COM_CHECKPOINT_DIR')
    if not d:
        return
    import pickle as _pickle
    import sys as _sys
    base = _os.path.join(d, '%s_pc%d' % (stage, int(package_testcase_i)))
    f, n = base + '.pkl', 1
    while _os.path.exists(f):
        n += 1
        f = '%s_%d.pkl' % (base, n)
    saved, failed = {}, {}
    for k, v in values.items():
        try:
            saved[k] = _pickle.dumps(v, protocol=_pickle.HIGHEST_PROTOCOL)
        except Exception as e:                              # noqa: BLE001
            failed[k] = repr(e)[:200]
    try:
        with open(f, 'wb') as fh:
            _pickle.dump({'values': saved, 'failed': failed}, fh,
                         protocol=_pickle.HIGHEST_PROTOCOL)
    except Exception as e:                                  # noqa: BLE001
        print('com_checkpoint: %s not saved: %r' % (f, e), file=_sys.stderr)
    for k, why in failed.items():
        print('com_checkpoint: %s.%s not saved: %s' % (f, k, why),
              file=_sys.stderr)


def com_ieee8023_(param, OP, chdata, SDDp2p=None,
                  _parameter_size_adjustment_fn=None,
                  _process_sxp_fn=None,
                  _read_s4p_files_fn=None,
                  _FD_Processing_fn=None,
                  _COM_FD_to_TD_fn=None,
                  _optimize_fom_fn=None,
                  _Apply_EQ_fn=None,
                  _get_pdf_fn=None,
                  _Create_Noise_PDF_fn=None,
                  _TDR_ERL_Processing_fn=None,
                  _COM_eye_width_fn=None,
                  _get_PSDs_fn=None,
                  _Output_Arg_Fill_fn=None,
                  _MLSE_U1_c_178A_fn=None):

    # ── Derived parameter initialization (MATLAB lines 285-297) ────────────
    param.ui = 1.0 / float(param.fb)
    param.sample_dt = param.ui / float(param.samples_per_ui)
    L = int(param.levels)
    param.sigma_X = float(np.sqrt((L**2 - 1) / (3.0 * (L - 1)**2)))
    factor_3db = 0.473037
    param.fb_BT_cutoff = factor_3db * float(param.f_r)
    param.fb_BW_cutoff = float(param.f_r)
    param.Tx_rd_sel = 1
    param.Rx_rd_sel = 2
    # r4p15p0: empty / NaN / scalar-0 snpPortsOrder -> empty, which triggers the
    # auto port-order routine downstream (was: default to [1 3 2 4]).
    # A single s4p needs exactly one 4-element order; a multi-row / non-4-element
    # entry (e.g. a per-package "[1 2 3 4; 1 3 2 4]" matrix) is ambiguous for one
    # file, so we also defer to auto-detection in that case.
    _spo = getattr(param, 'snpPortsOrder', None)
    _spo_np = None if _spo is None else np.asarray(_spo)
    _use_auto = _spo_np is None or _spo_np.size == 0
    if not _use_auto:
        _flat = _spo_np.ravel()
        try:
            if bool(np.any(np.isnan(_flat.astype(float)))):
                _use_auto = True
        except (TypeError, ValueError):
            pass
        if np.array_equal(_flat, [0]):
            _use_auto = True
        if _spo_np.ndim > 1 or _flat.size != 4:
            print('INFO: snpPortsOrder is not a single 4-element vector '
                  f'(shape {_spo_np.shape}); using auto port-order detection.')
            _use_auto = True
    if _use_auto:
        param.snpPortsOrder = np.array([])

    if _parameter_size_adjustment_fn is not None:
        param = _parameter_size_adjustment_fn(param, OP)

    # ── S2P detection ──────────────────────────────────────────────────────
    param.FLAG = getattr(param, 'FLAG', SimpleNamespace())
    param.FLAG.S2P = 0
    for ch in chdata:
        if hasattr(ch, 'ext') and str(ch.ext).lower() == '.s2p':
            param.FLAG.S2P = 1
            break

    if param.FLAG.S2P:
        OP.ERL_ONLY = 1

    # ── Main calibration loop ──────────────────────────────────────────────
    results = [None] * len(np.asarray(OP.pkg_len_select).ravel())
    COM_global = float('inf')
    min_COM = float('inf')
    min_VEO_mV = float('inf')
    max_VEC_dB = float('-inf')
    threshold_DER = float('inf')
    threshold_DER_max = 0.0
    sigma_bn = 0.0
    DO_ONCE = True
    low_COM_found = 0
    output_args = SimpleNamespace()
    # ML 62-63: name_split = strsplit(mfilename,'com_ieee8023_');
    # output_args.code_revision = name_split{2}. The reference reports the
    # release it IS, read off its own file name; the port reports the release it
    # emulates, which is the same string for com_ieee8023_<ver>.m.
    output_args.code_revision = str(getattr(param, 'matlab_version', '4p15p0'))

    while getattr(OP, 'RX_CALIBRATION', False) or DO_ONCE or getattr(OP, 'PSDRXCAL', False):
        if not DO_ONCE:
            if abs(min_COM - float(param.pass_threshold)) < 0.1 or (
                    sigma_bn == 0 and min_COM < float(param.pass_threshold)):
                break
            elif min_COM > float(param.pass_threshold):
                if low_COM_found:
                    if getattr(OP, 'sigma_bn_STEP', 0) > 0:
                        OP.sigma_bn_STEP = OP.sigma_bn_STEP / 2
                    else:
                        OP.sigma_bn_STEP = -OP.sigma_bn_STEP / 2
            else:
                low_COM_found = 1
                if getattr(OP, 'sigma_bn_STEP', 0) > 0:
                    OP.sigma_bn_STEP = -OP.sigma_bn_STEP / 2
                else:
                    OP.sigma_bn_STEP = OP.sigma_bn_STEP / 2
            min_COM = float('inf')
            min_VEO_mV = float('inf')
            max_VEC_dB = float('-inf')
            sigma_bn = sigma_bn + float(getattr(OP, 'sigma_bn_STEP', 0))

        pkg_len_select = np.asarray(OP.pkg_len_select).ravel()
        for pkg_i, package_testcase in enumerate(pkg_len_select):
            package_testcase_i = pkg_i + 1  # 1-based
            param.package_testcase_i = package_testcase_i
            package_testcase = int(package_testcase)
            # Wall-clock start of *this* case, for the engineering-export per-case
            # run time (the .mat export reports time for this case alone).
            import time as _time
            OP.export_case_t_start = _time.time()

            # Package length parameters
            param.Pkg_len_TX = np.asarray(param.z_p_tx_cases)[package_testcase - 1, :]
            param.Pkg_len_NEXT = np.asarray(param.z_p_next_cases)[package_testcase - 1, :]
            param.Pkg_len_FEXT = np.asarray(param.z_p_fext_cases)[package_testcase - 1, :]
            param.Pkg_len_RX = np.asarray(param.z_p_rx_cases)[package_testcase - 1, :]
            param.AC_CM_RMS_TX = float(np.asarray(param.AC_CM_RMS).ravel()[package_testcase - 1])
            if getattr(param, 'PKG_Tx_FFE_preset', 0) != 0:
                param.Pkg_TXFFE_preset = np.asarray(param.PKG_Tx_FFE_preset)[package_testcase - 1, :]
            else:
                param.Pkg_TXFFE_preset = 0

            # ── Fill chdata with S-parameters ─────────────────────────────
            if not getattr(OP, 'TDMODE', False):
                if _read_s4p_files_fn is not None:
                    chdata, SDDch_local, SDDp2p_local, param = _read_s4p_files_fn(param, OP, chdata)
                    _checkpoint('01_read_s4p_files', package_testcase_i, param=param, OP=OP, chdata=chdata)
                    # r4p15p0: surface the (possibly auto-detected) port order
                    output_args.port_order = param.snpPortsOrder
                    # ML 394-395: both reported, straight after the read. The
                    # port had never set either, and the 208-case comparison
                    # could not see it: matlab_compare --report skips a MATLAB
                    # column the Python result lacks. The Octave checkpoint
                    # harness, which fails an absent field, found it.
                    output_args.fstop_GHz = min(
                        float(param.flim), float(np.asarray(chdata[0].faxis).ravel()[-1])) / 1e9
                    output_args.flim_GHz = float(param.flim) / 1e9
                else:
                    SDDch_local = None
                    SDDp2p_local = SDDp2p
                if _process_sxp_fn is not None:
                    chdata, _ = _process_sxp_fn(param, OP, chdata, SDDch_local)
                    # MATLAB L392-396: surface CM modal-mask pass/fail to the output
                    # when the mask test was run (OP.CM_MASK_REPORT in plot_modal).
                    if hasattr(chdata[0], 'Rlcc_179mm'):
                        output_args.Rlcc_179mm_fail = chdata[0].Rlcc_179mm_fail
                        output_args.Rlcc_178mm_fail = chdata[0].Rlcc_178mm_fail
                        output_args.Rlcd_179mm_fail = chdata[0].Rlcd_179mm_fail
                        output_args.Rldc_179mm_fail = chdata[0].Rldc_179mm_fail

            # ── TDR & ERL ─────────────────────────────────────────────────
            ERL = [float('inf'), float('inf')]
            min_ERL = float('inf')
            if _TDR_ERL_Processing_fn is not None:
                output_args, ERL, min_ERL = _TDR_ERL_Processing_fn(
                    output_args, OP, package_testcase_i, chdata, param)
                _checkpoint('02_TDR_ERL_Processing', package_testcase_i, output_args=output_args, ERL=ERL, min_ERL=min_ERL, chdata=chdata)

            if getattr(OP, 'ERL_ONLY', False):
                results = [None]
                results[0] = output_args
                break

            # ── FD processing ─────────────────────────────────────────────
            param.number_of_s4p_files = len(chdata)
            output_args.ICN_mV = 0.0
            output_args.MDNEXT_ICN_92_46_mV = 0.0
            output_args.MDFEXT_ICN_92_47_mV = 0.0
            if getattr(OP, 'WC_PORTZ', False):
                param.SNR_TX = float(np.asarray(param.SNDR).ravel()[int(param.Tx_rd_sel) - 1])
            else:
                param.SNR_TX = float(np.asarray(param.SNDR).ravel()[package_testcase - 1])

            chdata, output_args = _FD_Processing_fn(chdata, output_args, param, OP, SDDp2p, DO_ONCE)
            _checkpoint('03_FD_Processing', package_testcase_i, chdata=chdata, output_args=output_args)

            # ── FD to TD ──────────────────────────────────────────────────
            if DO_ONCE and not getattr(OP, 'TDMODE', False):
                chdata = _COM_FD_to_TD_fn(chdata, param, OP)
                _checkpoint('04_COM_FD_to_TD', package_testcase_i, chdata=chdata)
                output_args.VCM_CD_HF_mV = chdata[0].VCM_CD_HF_struct.CMn * 1000.0
                output_args.VCM_DC_HF_mV = chdata[0].VCM_DC_HF_struct.CMn * 1000.0
                output_args.SCMR_TD_CD_ch_dB = chdata[0].SCMR_CD_ch
                output_args.SCMR_CH = chdata[0].SCMR_CD_ch
                output_args.SCMR_TD_CD_ch_pk_dB_ = chdata[0].SCMR_CD_ch_pk
                output_args.SCMR_TD_DC_ch_dB = chdata[0].SCMR_DC_ch
                output_args.SCMR_TD_DC_ch_pk_dB_ = chdata[0].SCMR_DC_ch_pk
                output_args.P_signal_TD = chdata[0].P_signal
                output_args.Sigma_ts_sqrd = chdata[0].P_signal

            # ── EQ optimization ───────────────────────────────────────────
            do_C2M = 0
            if float(getattr(param, 'T_O', 0)) != 0 and float(getattr(param, 'Min_VEO_Test', 0)) != 0:
                do_C2M = 1

            if getattr(OP, 'DO_NOT_COMPUTE_COM', False):
                results[pkg_i] = output_args
                DO_ONCE = False
                break

            param.sigma_ns = sigma_bn
            OP.COMPUTE_COM = False  # MATLAB L483: false during optimize_fom (get_PSDs uses it)
            # MATLAB passes param BY VALUE to optimize_fom, so its Floating_DFE mutation
            # param.ndfe=N_bmax (L8445) is local. In Python param is shared by reference, so
            # without this restore the mutation leaks across package cases (case-2 ndfe_passed
            # becomes N_bmax -> findbankloc negative-dims crash) and into Apply_EQ.
            _ndfe_save = int(param.ndfe)
            fom_result = _optimize_fom_fn(OP, param, chdata, sigma_bn, do_C2M)
            _checkpoint('05_optimize_fom', package_testcase_i, fom_result=fom_result)
            param.ndfe = _ndfe_save
            if fom_result.eq_failed:
                return results

            OP.COMPUTE_COM = True  # MATLAB L505: true after optimization
            A_s = abs(float(fom_result.A_s))
            param.use_bmax = np.asarray(fom_result.best_bmax).ravel()
            param.use_bmin = np.asarray(fom_result.best_bmin).ravel()
            param.current_ffegain = fom_result.best_current_ffegain
            if getattr(OP, 'force_pdf_bin_size', False):
                param.delta_y = float(OP.BinSize)
            else:
                param.delta_y = min(A_s / 1000.0, float(OP.BinSize))

            if getattr(OP, 'RX_CALIBRATION', False):
                param.number_of_s4p_files = 1
            if getattr(OP, 'PSDRXCAL', False):
                param.number_of_s4p_files = param.number_of_s4p_files - 1

            chdata = _Apply_EQ_fn(param, fom_result, chdata, OP)
            _checkpoint('06_Apply_EQ', package_testcase_i, chdata=chdata)

            # ── PSD (MMSE RxFFE path) ─────────────────────────────────────
            PSD_results = SimpleNamespace()
            if getattr(OP, 'FFE_OPT_METHOD', '') == 'MMSE' and getattr(OP, 'RxFFE', False):
                OP.WO_TXFFE = 1
                PSD_results.w = fom_result.RxFFE
                PSD_results.S_rn = fom_result.PSD_results.S_rn
                PSD_results.S_in = fom_result.PSD_results.S_in
                g_dc = float(np.asarray(param.ctle_gdc_values).ravel()[fom_result.ctle - 1])
                g_hp = float(np.asarray(param.g_DC_HP_values).ravel()[fom_result.best_G_high_pass - 1])
                PSD_results = _get_PSDs_fn(
                    PSD_results, chdata[0].eq_pulse_response, fom_result.t_s,
                    fom_result.txffe, g_dc, g_hp, param, chdata, OP)
                _checkpoint('07_get_PSDs', package_testcase_i, PSD_results=PSD_results)
                OP.WO_TXFFE = 0
                for fld in ('S_xn', 'S_tn', 'S_jn', 'S_rj_jn'):
                    setattr(PSD_results, fld, getattr(fom_result.PSD_results, fld))
                PSD_results = _get_PSDs_fn(
                    PSD_results, chdata[0].eq_pulse_response, fom_result.t_s,
                    fom_result.txffe, g_dc, g_hp, param, chdata, OP)
                _checkpoint('07_get_PSDs', package_testcase_i, PSD_results=PSD_results)
                # r4p16p0 L554-559.  The port dropped these six assignments, so
                # output_args.noiseRMS_mV never existed on the MMSE+RxFFE path.
                # MATLAB writes .tn twice (L555 and L557) with the same value;
                # the field order it leaves behind is rn, tn, xn, jn, in.
                # 'in' is a Python keyword, so it is set by name rather than as
                # a keyword argument -- getattr(..., 'in') reads it back.
                output_args.noiseRMS_mV = SimpleNamespace(
                    rn=PSD_results.S_rn_rms * 1000,
                    tn=PSD_results.S_tn_rms * 1000,
                    xn=PSD_results.S_xn_rms * 1000,
                    jn=PSD_results.S_jn_rms * 1000,
                )
                setattr(output_args.noiseRMS_mV, 'in', PSD_results.S_in_rms * 1000)

            # ── Per-channel PDF ───────────────────────────────────────────
            for i, ch in enumerate(chdata[:param.number_of_s4p_files]):
                if getattr(OP, 'FFE_OPT_METHOD', '') == 'MMSE' and getattr(OP, 'RxFFE', False):
                    # MATLAB PSD_results.iphase(i): array per crosstalk channel, or scalar 1
                    # when there is no crosstalk (L6495). Scalar isn't indexable in Python.
                    _iphase = getattr(fom_result.PSD_results, 'iphase', [None] * len(chdata))
                    iphase = _iphase if np.ndim(_iphase) == 0 else _iphase[i]
                    pdf = _get_pdf_fn(ch, param.delta_y, fom_result.t_s, param, OP, iphase)
                else:
                    pdf = _get_pdf_fn(ch, param.delta_y, fom_result.t_s, param, OP, [])
                ch.pdfr = pdf
                nonzero = np.where(pdf.y > 1e-12)[0]
                ch.maxquickpdf = float(pdf.y[nonzero[0]]) if len(nonzero) > 0 else 0.0

            # ── Combined noise PDF ────────────────────────────────────────
            PDF, CDF, Noise_Struct = _Create_Noise_PDF_fn(
                A_s, param, fom_result, chdata, OP, sigma_bn, PSD_results)
            _checkpoint('08_Create_Noise_PDF', package_testcase_i, PDF=PDF, CDF=CDF, Noise_Struct=Noise_Struct)

            # ── COM, VEC, VEO ─────────────────────────────────────────────
            A_ni_ix_arr = np.where(CDF > float(param.specBER))[0]
            A_ni_ix = int(A_ni_ix_arr[0]) if len(A_ni_ix_arr) > 0 else 0
            A_ni = abs(float(PDF.x[A_ni_ix]))

            # Threshold DER
            thresh_x = -A_s / (10.0**(float(getattr(param, 'pass_threshold', 3.0)) / 20.0))
            th_arr = np.where(PDF.x > thresh_x)[0]
            if len(th_arr) > 0:
                threshold_DER = float(CDF[th_arr[0]])
            threshold_DER_max = max(threshold_DER_max, threshold_DER)

            # Eye width
            EW_UI = 0.0
            eye_contour = []
            EH_T_C2M = 0.0
            EH_B_C2M = 0.0
            MLSE_results = SimpleNamespace()

            if not getattr(OP, 'RX_CALIBRATION', False) and getattr(OP, 'EW', 0) == 1 and getattr(OP, 'MLSE', 0) == 0:
                Left_EW, Right_EW, eye_contour, EH_T_C2M, EH_B_C2M = _COM_eye_width_fn(
                    chdata, param.delta_y, fom_result, param, OP, Noise_Struct, 0)
                _checkpoint('09_COM_eye_width', package_testcase_i, Left_EW=Left_EW, Right_EW=Right_EW, eye_contour=eye_contour)
                EW_UI = float(np.floor(np.sum(Left_EW) + np.sum(Right_EW))) / float(param.samples_for_C2M)
            elif (EYE_PLOT_UNDER_MLSE
                  and not getattr(OP, 'RX_CALIBRATION', False)
                  and getattr(OP, 'EW', 0) == 1
                  and getattr(OP, 'MLSE', 0) != 0):
                # PLOTTING ONLY -- opt in with `sicopr.EYE_PLOT_UNDER_MLSE = True`.
                #
                # MATLAB (4p15p0 L620) gates the eye on OP.MLSE == 0, so with MLSE
                # enabled neither tool produces an eye contour or timing bathtub.
                # That is a reporting choice, not a computability limit: MLSE is
                # applied at L667, AFTER this point, so PDF/CDF/Noise_Struct here
                # are the pre-MLSE (DFE-only) quantities and the eye they describe
                # is well defined -- it is simply the eye before MLSE post-
                # processing, which is what an eye diagram means anyway.
                #
                # Every return value is DISCARDED, so EW_UI stays 0 and
                # eye_contour stays [] exactly as MATLAB leaves them. The only
                # effect is COM_eye_width's plotting side-channel,
                # chdata[0].timing_bathtub, which com_plots and the .mat export
                # read. No reported COM, VEC, VEO or EW value changes.
                _COM_eye_width_fn(chdata, param.delta_y, fom_result, param, OP,
                                  Noise_Struct, 0)

            eps_val = np.finfo(float).eps
            if getattr(OP, 'MLSE', 0) == 0:
                T_O_val = float(getattr(param, 'T_O', 0))
                if T_O_val != 0:
                    eye_opening = float(EH_T_C2M) - float(EH_B_C2M)
                    A_ni = 2.0 * A_s - eye_opening
                    vec_arg = 2.0 * A_s / max(eye_opening, eps_val)
                    VEC_dB = 20.0 * np.log10(max(vec_arg, eps_val))
                    COM = 20.0 * np.log10(2.0 * A_s / max(A_ni, eps_val))
                    VEO_mV = eye_opening * 1000.0
                else:
                    VEO_mV = 1000.0 * (A_s - A_ni) * 2.0
                    vec_arg = (A_s - A_ni) / max(A_s, eps_val)
                    VEC_dB = -20.0 * np.log10(max(vec_arg, eps_val))
                    COM = 20.0 * np.log10(A_s / max(A_ni, eps_val))
            else:
                MLSE_results = _MLSE_U1_c_178A_fn(
                    param, fom_result.MMSE_results.blim, A_s, A_ni, PDF, CDF, PSD_results)
                T_O_val = float(getattr(param, 'T_O', 0))
                _dcom = float(MLSE_results.delta_com)
                if T_O_val != 0:
                    eye_opening = float(EH_T_C2M) - float(EH_B_C2M)
                    A_ni = 2.0 * A_s - eye_opening
                    vec_arg = 2.0 * A_s / max(eye_opening, eps_val)
                    VEC_dB_orig = 20.0 * np.log10(max(vec_arg, eps_val))
                    # MATLAB L672/682
                    delta_VEC = _dcom - 20.0 * np.log10(
                        (10.0**(_dcom / 20.0) - 1.0) * 10.0**(VEC_dB_orig / 20.0) + 1.0)
                    VEC_dB = delta_VEC + VEC_dB_orig
                    COM_orig = 20.0 * np.log10(2.0 * A_s / max(A_ni, eps_val))  # MATLAB L673
                    COM = float(MLSE_results.COM)
                    VEO_mV = eye_opening * 1000.0
                else:
                    VEO_mV = 1000.0 * (A_s - A_ni) * 2.0
                    vec_arg = (A_s - A_ni) / max(A_s, eps_val)
                    VEC_dB_orig = -20.0 * np.log10(max(vec_arg, eps_val))
                    delta_VEC = _dcom - 20.0 * np.log10(
                        (10.0**(_dcom / 20.0) - 1.0) * 10.0**(VEC_dB_orig / 20.0) + 1.0)
                    VEC_dB = delta_VEC + VEC_dB_orig
                    COM_orig = 20.0 * np.log10(A_s / max(A_ni, eps_val))  # MATLAB L693
                    COM = float(MLSE_results.COM)

            min_COM = min(min_COM, COM)
            min_VEO_mV = min(min_VEO_mV, VEO_mV)
            max_VEC_dB = max(max_VEC_dB, VEC_dB)

            # ── COM_SNR_Struct ─────────────────────────────────────────────
            COM_SNR_Struct = SimpleNamespace(
                A_s=A_s, A_ni=A_ni, threshold_DER=threshold_DER,
                EW_UI=EW_UI, COM=COM, VEC_dB=VEC_dB, VEO_mV=VEO_mV,
                combined_interference_and_noise_pdf=PDF,
                combined_interference_and_noise_cdf=CDF,
                eye_contour=eye_contour,
            )
            if getattr(OP, 'MLSE', 0) == 0:
                COM_SNR_Struct.COM_orig = []
                COM_SNR_Struct.VEC_dB_orig = []
            else:
                # MATLAB L679-683/698-701/716-717: pre-MLSE COM_orig + MLSE deltas
                COM_SNR_Struct.COM_orig = COM_orig
                COM_SNR_Struct.VEC_dB_orig = VEC_dB_orig
                COM_SNR_Struct.delta_COM = MLSE_results.delta_com
                COM_SNR_Struct.DER_DFE = MLSE_results.DER_DFE
                COM_SNR_Struct.DER_MLSE = MLSE_results.DER_MLSE
                COM_SNR_Struct.delta_VEC = delta_VEC

            # ── Output filling ─────────────────────────────────────────────
            output_args = _Output_Arg_Fill_fn(
                output_args, sigma_bn, Noise_Struct, COM_SNR_Struct, param, chdata, fom_result, OP)
            _checkpoint('10_Output_Arg_Fill', package_testcase_i, output_args=output_args, COM_SNR_Struct=COM_SNR_Struct)

            if getattr(OP, 'BREAD_CRUMBS', False):
                output_args.OP = OP
                output_args.param = param
                output_args.chdata = chdata
                output_args.fom_result = fom_result
                output_args.PDF = PDF
                output_args.CDF = CDF
                output_args.MLSE_results = MLSE_results
                output_args.PSD_results = PSD_results

            # Snapshot this case (MATLAB results{i}=output_args is a value-copy;
            # Python aliases one object otherwise, so cases would overwrite).
            import copy as _copy
            results[pkg_i] = _copy.deepcopy(output_args)

            # ── Per-case results + figures (config-driven, MATLAB-style) ───
            if getattr(OP, 'SAVE_FIGURES', 0) or getattr(OP, 'CSV_REPORT', 0):
                _save_case_outputs(OP, param, chdata, fom_result, Noise_Struct,
                                   PDF, CDF, COM_SNR_Struct, output_args,
                                   param.package_testcase_i)

            # ── Engineering .mat export for R analysis (optional, --export-mat) ─
            # Captures the live pipeline objects already computed above; does not
            # alter any COM calculation. See com_mat_export.export_case_mat.
            if getattr(OP, 'EXPORT_MAT', 0):
                try:
                    import com_mat_export
                    com_mat_export.export_case_mat(
                        OP, param, chdata, fom_result, Noise_Struct,
                        PDF, CDF, COM_SNR_Struct, output_args,
                        param.package_testcase_i)
                except Exception as _e:
                    print('    [export-mat] skipped: %s' % _e)

        DO_ONCE = False

    # ── Unwrap single result ───────────────────────────────────────────────
    if len(results) == 1:
        results = results[0]

    return results
