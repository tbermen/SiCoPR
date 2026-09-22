# ============================================================
# MATLAB→Python translation notes for OptFom_Calc_Noise
# MATLAB lines: 2881–3009
# ============================================================
# cursor_i: 0-based in Python.
# sampling_offset: MATLAB mod(cursor_i_matlab, M) = mod(cursor_i_python+1, M).
# h_J early/late sample indexing: see detailed comments inline.
# sigma_TX: Equation 93A-30.  txffe(cursor_index) is 1-based → txffe[cursor_index-1].
# sigma_ISI: Equation 93A-31.  sigma_J: Equation 93A-32.
# sigma_XT: calls the real get_xtlk_noise (FEXT+NEXT); the RxFFE_with_MMSE path
#   instead uses PSD_results.S_xn_rms.
# H_Rx_FFE phase_memory column for RxFFE (non-MMSE): 0-based = ii+RxFFE_cmx+len(txffe).
# total_noise_rms: Equation 93A-36 denominator (MMSE vs non-MMSE path).
# ============================================================

import numpy as np
from com_functions.fn.get_xtlk_noise.py_impl import get_xtlk_noise as _get_xtlk_noise


def OptFom_Calc_Noise(THIS, Best_FOM, sbr, SETTINGS, chdata, param, OP):
    cursor_i = int(THIS.cursor_i)  # 0-based
    A_s = float(THIS.A_s)
    txffe = np.asarray(THIS.txffe, dtype=float)
    C = np.asarray(THIS.C, dtype=float) if hasattr(THIS, 'C') and THIS.C is not None else np.array([])
    PSD_results = getattr(THIS, 'PSD_results', None)
    sigma_ne = float(THIS.sigma_ne)
    precursors = np.asarray(THIS.precursors, dtype=float)
    far_cursors = np.asarray(THIS.far_cursors, dtype=float)
    excess_dfe_cursors = np.asarray(THIS.excess_dfe_cursors, dtype=float)
    H_ctf = np.asarray(THIS.H_ctf, dtype=complex)
    sigma_N = float(THIS.sigma_N)

    phase_memory = SETTINGS.phase_memory
    H_sy = np.asarray(SETTINGS.H_sy, dtype=float)
    H_r = np.asarray(SETTINGS.H_r, dtype=complex)
    f = np.asarray(chdata[0].faxis, dtype=float)
    M = int(param.samples_per_ui)
    sbr = np.asarray(sbr, dtype=float)

    abort_status = 0

    # Equation 93A-28: h_J — jitter contribution
    # MATLAB sampling_offset = mod(cursor_i_matlab, M), cursor_i_matlab = cursor_i_py + 1
    sampling_offset = (cursor_i + 1) % M
    if sampling_offset <= 1:
        sampling_offset += M

    if OP.LIMIT_JITTER_CONTRIB_TO_DFE_SPAN:
        # MATLAB: sbr(cursor_i - 1 + M*(-1:ndfe))  1-based
        # 0-based: cursor_i_py + M*k for k in range(-1, ndfe+1)
        ndfe = int(param.ndfe)
        ks = np.arange(-1, ndfe + 1)
        idx_early = cursor_i - 1 + M * ks  # 0-based
        idx_late = cursor_i + 1 + M * ks   # 0-based
        # MATLAB indexes sbr directly and refuses a span that runs off either
        # end.  COM Octave, 200-sample sbr:
        #   cursor_i=3   -> error: sbr(-6): subscripts must be either integers
        #                   1 to (2^63)-1 or logicals
        #   cursor_i=195 -> error: sbr(226): out of bound 200
        # Dropping the out-of-range entries produced a shorter h_J, and so a
        # quietly different sigma_J, where the reference declines to answer.
        span = np.concatenate([idx_early, idx_late])
        if span.min() < 0 or span.max() >= len(sbr):
            raise IndexError(
                'OptFom_Calc_Noise: jitter sample span reaches sbr(%d..%d), '
                'outside the %d samples available'
                % (int(span.min()) + 1, int(span.max()) + 1, len(sbr)))
        cursors_early = sbr[idx_early]
        cursors_late = sbr[idx_late]
    else:
        # MATLAB: sbr(sampling_offset-1:M:end) (1-based) → sbr[sampling_offset-2::M] (0-based)
        cursors_early = sbr[sampling_offset - 2 :: M]
        cursors_late = sbr[sampling_offset :: M]

    min_len = min(len(cursors_early), len(cursors_late))
    cursors_early = cursors_early[:min_len]
    cursors_late = cursors_late[:min_len]
    h_J = (cursors_late - cursors_early) / 2.0 * M

    # sigma_TX (Equation 93A-30)
    L = int(param.levels)
    R_LM = float(param.R_LM)
    SNR_TX_lin = 10 ** (-float(param.SNR_TX) / 20.0)
    if not OP.SNR_TXwC0:
        sigma_TX = (L - 1) * A_s / R_LM * SNR_TX_lin
    else:
        cur_idx = int(param.cursor_index) - 1  # 0-based
        # MATLAB txffe(0) is not the last tap, it is an error.  COM Octave,
        # OP.SNR_TXwC0=1 with param.cursor_index=0:
        #   "error: txffe(0): subscripts must be either integers 1 to (2^63)-1
        #    or logicals".
        # A negative Python index wraps to txffe[-1] and returns a sigma_TX
        # built from the wrong tap.
        if cur_idx < 0:
            raise IndexError('OptFom_Calc_Noise: param.cursor_index is %d, so '
                             'MATLAB txffe(%d) is not a valid subscript'
                             % (int(param.cursor_index), int(param.cursor_index)))
        sigma_TX = (L - 1) * A_s / float(txffe[cur_idx]) / R_LM * SNR_TX_lin

    # sigma_ISI (Equation 93A-31)
    sigma_X = float(param.sigma_X)
    isi_vec = np.concatenate([precursors.ravel(),
                               excess_dfe_cursors.ravel(),
                               far_cursors.ravel()])
    sigma_ISI = sigma_X * float(np.linalg.norm(isi_vec))
    ISI_N = sigma_X * float(np.linalg.norm(far_cursors.ravel()))

    # Quick-abort if FOM has no chance
    if not OP.RxFFE_with_MMSE:
        exe_mode = int(OP.EXE_MODE)
        if exe_mode in (1, 2) and sigma_ISI > 0:
            candidate = 20 * np.log10(A_s / sigma_ISI)
            if candidate < Best_FOM:
                abort_status = 1 if exe_mode == 1 else 2
                # MATLAB returns here with THIS untouched: h_J/sigma_TX/ISI_N are
                # still locals at this point and are only written into THIS at the
                # end of the function (ML 2935-2941). Python passes THIS by
                # reference, so writing them here would leave the ABORTED tick's
                # values visible to the caller where MATLAB leaves the last
                # successfully scored tick's. Leave THIS alone.
                return THIS, abort_status

    # sigma_J (Equation 93A-32)
    A_DD = float(param.A_DD)
    sigma_RJ = float(param.sigma_RJ)
    sigma_J = float(np.linalg.norm([A_DD, sigma_RJ])) * sigma_X * float(np.linalg.norm(h_J))

    # sigma_XT (MATLAB lines 2954-2967, Eq 93A-49/50)
    if OP.RX_CALIBRATION:
        sigma_XT = 0.0
    else:
        if not OP.RxFFE:
            # MATLAB: [sigma_XT,~,~] = get_xtlk_noise(txffe,'FEXT',param,chdata,phase_memory)
            sigma_XT, _, _ = _get_xtlk_noise(txffe, 'both', param, chdata, phase_memory)
        else:
            # MATLAB uses strcmp here, which is CASE SENSITIVE (the reference
            # is inconsistent: other sites wrap it in upper()).  COM Octave
            # with OP.RX_CALIBRATION=0, OP.RxFFE=1, OP.RxFFE_with_MMSE=0:
            #   FFE_OPT_METHOD='MMSE' -> takes PSD_results.S_xn_rms
            #   FFE_OPT_METHOD='mmse' -> calls get_xtlk_noise(...,C)
            # `.upper()` sent both to the PSD branch, where PSD_results is None
            # on that path.
            if not (hasattr(OP, 'FFE_OPT_METHOD')
                    and str(OP.FFE_OPT_METHOD) == 'MMSE'):
                # MATLAB: [sigma_XT,~,~] = get_xtlk_noise(txffe,'FEXT',param,chdata,phase_memory,C)
                sigma_XT, _, _ = _get_xtlk_noise(txffe, 'both', param, chdata, phase_memory, C)
            else:
                sigma_XT = float(PSD_results.S_xn_rms)

    # Update sigma_N for RxFFE (non-MMSE)
    if not OP.RxFFE_with_MMSE and OP.RxFFE:
        cmx = int(param.RxFFE_cmx)
        cpx = int(param.RxFFE_cpx)
        H_Rx_FFE = np.zeros(len(f), dtype=complex)
        n_txffe = len(txffe)
        for ii in range(-cmx, cpx + 1):
            c_idx = ii + cmx  # 0-based Python: MATLAB C(ii+cmx+1) → C[ii+cmx]
            # MATLAB indexes C directly, so a C shorter than
            # RxFFE_cmx+RxFFE_cpx+1 is an error, not a set of skipped taps.
            # COM Octave, RxFFE_cmx=RxFFE_cpx=1 with a two-element C:
            #   "error: C(3): out of bound 2 (dimensions are 1x2)".
            # `continue` here dropped the missing taps from H_Rx_FFE and
            # reported a sigma_N the reference declines to produce.
            if c_idx < 0 or c_idx >= len(C):
                raise IndexError('OptFom_Calc_Noise: C(%d) is out of bound %d'
                                 % (c_idx + 1, len(C)))
            if C[c_idx] == 0:
                continue
            if ii + 1 == 0:
                H_Rx_FFE = H_Rx_FFE + C[c_idx]
            else:
                pm_col = ii + cmx + n_txffe  # 0-based phase_memory column
                H_Rx_FFE = H_Rx_FFE + C[c_idx] * phase_memory[:, pm_col]
        eta_0 = float(param.eta_0)
        sigma_N = float(np.sqrt(
            eta_0 * np.sum(H_sy[1:] * np.abs(H_r[1:] * H_ctf[1:] * H_Rx_FFE[1:]) ** 2
                           * np.diff(f) / 1e9)))

    # total_noise_rms (Equation 93A-36 denominator)
    if OP.RxFFE_with_MMSE:
        S_n_rms = float(PSD_results.S_n_rms)
        total_noise_rms = float(np.linalg.norm([sigma_ISI, S_n_rms, sigma_ne]))
    else:
        total_noise_rms = float(np.linalg.norm(
            [sigma_ISI, sigma_J, sigma_XT, sigma_N, sigma_TX, sigma_ne]))

    THIS.h_J = h_J
    THIS.sigma_TX = sigma_TX
    THIS.ISI_N = ISI_N
    THIS.sigma_N = sigma_N
    THIS.total_noise_rms = total_noise_rms
    return THIS, abort_status
