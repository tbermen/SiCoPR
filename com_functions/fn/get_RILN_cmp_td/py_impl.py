# ============================================================
# MATLAB→Python translation notes for get_RILN_cmp_td
# MATLAB lines: 6694–6875
# ============================================================
# Row/col: MATLAB uses row vectors; Python uses 1D arrays (ravel).
# 1000-echo sum: geometric series in FD; looped faithfully.
# fmin_idx: MATLAB 1-based find(f>=fmin,1,'first') → np.searchsorted(f,fmin)
# fmbg least squares: MATLAB's normal equations ((fmbg'*fmbg)^-1)*fmbg'*LGw,
#   reproduced term for term.  np.linalg.lstsq is NOT a substitute: fmbg is
#   conditioned ~1e21 (its columns span sdd21 .. f^2*sdd21) and lstsq's SVD
#   cutoff throws away the very modes that carry the fit (see test_verify).
# filter(ones(1,M),1,x) → lfilter(ones(M),1,x)
# ipeak: MATLAB 1-based argmax → 0-based np.argmax; range ipeak:range_end
#   MATLAB inclusive → Python slice [ipeak:range_end] (exclusive upper OK since
#   MATLAB range_end = min(len,len) and slicing is safe)
# calculate_delay_CausalityEnforcement: MATLAB's `try ... catch end` leaves
#   delay_sec undefined, so the next line raises. Matched, not papered over.
# get_pdf_from_sampled_signal: returns SimpleNamespace with .x and .y
# Bessel_Thomson_Filter / Butterworth_Filter / s21_to_impulse_DC /
#   calculate_delay_CausalityEnforcement / get_pdf_from_sampled_signal /
#   normal_dist are top-level functions of the assembled module and are called
#   by name.  They used to be hand-written stubs here; the stubbed Butterworth
#   response alone (1/sqrt(1+(2f/fb)^8), magnitude only) put REF_noise.PR out
#   by a factor of 2.6 against the reference.  A stub is not a translation.
# ============================================================

import numpy as np
from scipy.signal import lfilter
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_RILN_cmp_td(sdd21, RIL_struct, faxis_f2, OP, param, A_T,
                    _Bessel_Thomson_Filter_fn=None,
                    _Butterworth_Filter_fn=None,
                    _s21_to_impulse_DC_fn=None,
                    _calculate_delay_fn=None,
                    _get_pdf_fn=None,
                    _normal_dist_fn=None):
    """Compute TD reflection and re-reflection noise (RILN) figure of merit.

    MATLAB lines 6694–6875.

    sdd21: complex insertion loss array (1D, length matching faxis_f2)
    RIL_struct: SimpleNamespace with fields RIL, rho_port1, rho_port2, freq
    faxis_f2: frequency axis (Hz, at least up to fb)
    OP: SimpleNamespace with Bessel_Thomson, Butterworth, transmitter_transition_time,
        BinSize, impulse_response_truncation_threshold, etc.
    param: SimpleNamespace with samples_per_ui, sample_dt, fb, fb_BT_cutoff,
           levels, specBER, etc.
    A_T: amplitude threshold (not used in this function body; kept for signature compat)

    Returns RILN_TD_struct SimpleNamespace with fields:
        REF, FIT, RIL, REF_noise — each with FIR, PR, t, causality/truncation dBs
        ILN, t, FOM, FOM_PDF, SNR_ISI_FOM, SNR_ISI_FOM_PDF, PDF
    """
    bt_fn = _Bessel_Thomson_Filter_fn or Bessel_Thomson_Filter
    bw_fn = _Butterworth_Filter_fn or Butterworth_Filter
    s21_fn = _s21_to_impulse_DC_fn or s21_to_impulse_DC
    delay_fn = _calculate_delay_fn or calculate_delay_CausalityEnforcement
    pdf_fn = _get_pdf_fn or get_pdf_from_sampled_signal
    norm_fn = _normal_dist_fn or normal_dist

    M = int(param.samples_per_ui)

    # ---- Flatten inputs to 1D row-like arrays ----
    sdd21 = np.asarray(sdd21, dtype=complex).ravel()
    RIL = np.asarray(RIL_struct.RIL, dtype=complex).ravel()
    rho_port1 = np.asarray(RIL_struct.rho_port1, dtype=complex).ravel()
    rho_port2 = np.asarray(RIL_struct.rho_port2, dtype=complex).ravel()
    RIL_f = np.asarray(RIL_struct.freq, dtype=float).ravel()
    faxis_f2 = np.asarray(faxis_f2, dtype=float).ravel()

    # ---- Override OP fields (matching MATLAB L6836-6838) ----
    # MATLAB passes OP BY VALUE, so these assignments are local to this function.
    # Python passes it by reference: mutating OP here leaked 'trend_to_DC' /
    # 'interp_to_DC' into every S-parameter interpolation performed afterwards,
    # replacing the configured 'linear_trend_to_DC' /
    # 'extrap_cubic_to_dc_linear_to_inf'. That changes the DC/low-frequency
    # extrapolation and hence the impulse and pulse responses — visible as a
    # low steady-state voltage, a high pulse peak, and a large ISI error, while
    # leaving Nyquist-band magnitude metrics (IL, ICN) untouched.
    # get_ILN_cmp_td already uses this OP_copy pattern; match it here.
    OP = SimpleNamespace(**vars(OP))
    OP.interp_sparam_mag = 'trend_to_DC'
    OP.interp_sparam_phase = 'interp_to_DC'
    OP.impulse_response_truncation_threshold = 1e-7

    # ---- Echo sum: port1 and port2 reflection/re-reflection noise ----
    number_of_echos = 1000
    fmin = 1e9
    port2_rn = np.zeros(len(RIL), dtype=complex)
    port1_rn = np.zeros(len(RIL), dtype=complex)
    abs_RIL = np.abs(RIL)
    for m in range(1, number_of_echos + 1):
        port2_rn += (abs_RIL * (RIL ** (2 * m)) *
                     (rho_port1 ** m) * (rho_port2 ** m) *
                     (1 + rho_port1) * (1 + rho_port2))
        port1_rn += (abs_RIL * (RIL ** (2 * m - 1)) *
                     (rho_port1 ** (m - 1)) * (rho_port2 ** m) *
                     (1 + rho_port1) * (1 + rho_port1))

    # ---- Remove data below fmin (bad TD conversion near DC) ----
    fmin_idx = int(np.searchsorted(RIL_f, fmin))
    port2_rn = port2_rn[fmin_idx:]
    port1_rn = port1_rn[fmin_idx:]
    f_rn = RIL_f[fmin_idx:]

    # ---- Log-domain polynomial fit to sdd21 (trend removal for ILN) ----
    f = faxis_f2
    N = len(f)
    sdd21_col = sdd21.reshape(-1, 1)  # N×1
    fmbg = np.column_stack([
        sdd21,
        np.sqrt(f) * sdd21,
        f * sdd21,
        f**2 * sdd21,
    ])  # N×4
    # MATLAB has no floor here: log(0) is -Inf and 0*-Inf is NaN, which then
    # poisons alpha and every output.  An eps floor quietly returns a plausible
    # number instead, so a channel with a null in it would look healthy.
    with np.errstate(divide='ignore', invalid='ignore'):
        unwraplog = np.log(np.abs(sdd21)) + 1j * np.unwrap(np.angle(sdd21))
        LGw = sdd21 * unwraplog  # N, complex
    # MATLAB L6755: alpha = ((fmbg'*fmbg)^-1)*fmbg'*LGw.  `'` is the conjugate
    # transpose and `^-1` is inv().  Spelled out because a least-squares solver
    # is NOT interchangeable with it here: cond(fmbg) ~ 1e21, so lstsq's SVD
    # cutoff discards the sqrt(f) and f modes and returns a different fit.
    fmbgH = fmbg.conj().T
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = np.linalg.inv(fmbgH @ fmbg) @ (fmbgH @ LGw)
    efit_C = alpha[0] + alpha[1] * np.sqrt(f) + alpha[2] * f + alpha[3] * f**2
    FIT = np.exp(efit_C)

    # ---- TD responses for REF, FIT ----
    H_bw = bw_fn(param, f, 1)
    H_t = np.exp(-(np.pi * f / 1e9 * OP.transmitter_transition_time / 1.6832) ** 2)
    H_tw = np.ones(len(f))  # Tukey window overridden to ones in MATLAB

    RILN_TD_struct = SimpleNamespace()

    RILN_TD_struct.REF = SimpleNamespace()
    (RILN_TD_struct.REF.FIR,
     RILN_TD_struct.REF.t,
     RILN_TD_struct.REF.causality_correction_dB,
     RILN_TD_struct.REF.truncation_dB) = s21_fn(sdd21 * H_bw * H_t * H_tw, f, param.sample_dt, OP, param)
    RILN_TD_struct.REF.PR = lfilter(np.ones(M), 1, RILN_TD_struct.REF.FIR)

    RILN_TD_struct.FIT = SimpleNamespace()
    (RILN_TD_struct.FIT.FIR,
     RILN_TD_struct.FIT.t,
     RILN_TD_struct.FIT.causality_correction_dB,
     RILN_TD_struct.FIT.truncation_dB) = s21_fn(FIT * H_bw * H_t * H_tw, f, param.sample_dt, OP, param)
    RILN_TD_struct.FIT.PR = lfilter(np.ones(M), 1, RILN_TD_struct.FIT.FIR)

    # ---- TD response for RIL ----
    H_bw_r = bw_fn(param, RIL_f, 1)
    H_t_r = np.exp(-(np.pi * RIL_f / 1e9 * OP.transmitter_transition_time / 1.6832) ** 2)
    H_tw_r = np.ones(len(RIL_f))

    RILN_TD_struct.RIL = SimpleNamespace()
    (RILN_TD_struct.RIL.FIR,
     RILN_TD_struct.RIL.t,
     RILN_TD_struct.RIL.causality_correction_dB,
     RILN_TD_struct.RIL.truncation_dB) = s21_fn(RIL * H_bw_r * H_t_r * H_tw_r, RIL_f, param.sample_dt, OP, param)
    RILN_TD_struct.RIL.PR = lfilter(np.ones(M), 1, RILN_TD_struct.RIL.FIR)

    # ---- Channel delay → apply to port2 noise ----
    # MATLAB: `try [delay_sec, delay_idx] = calculate_delay_...; catch end`.
    # The empty catch swallows the failure and leaves delay_sec UNDEFINED, so
    # the very next line dies with "'delay_sec' undefined" — confirmed by
    # running the reference under Octave against a failing delay function.
    # That is an upstream defect (the catch should set a fallback), but it is
    # the reference's behaviour; the port used to substitute a silent zero
    # delay, which is a different answer, not a recovered one.
    try:
        delay_sec, _ = delay_fn(f, sdd21, param, OP)
    except Exception as exc:
        raise NameError(
            "'delay_sec' undefined: calculate_delay_CausalityEnforcement "
            "failed and the reference's empty catch leaves delay_sec unset "
            '(upstream defect, COM 4p16p0 get_RILN_cmp_td)') from exc
    port2_rn = port2_rn * np.exp(-1j * 2 * np.pi * f_rn * delay_sec)

    # ---- TD response for REF_noise (port2 reflection/re-reflection) ----
    H_bw_n = bw_fn(param, f_rn, 1)
    H_t_n = np.exp(-(np.pi * f_rn / 1e9 * OP.transmitter_transition_time / 1.6832) ** 2)
    H_tw_n = np.ones(len(f_rn))

    RILN_TD_struct.REF_noise = SimpleNamespace()
    (RILN_TD_struct.REF_noise.FIR,
     RILN_TD_struct.REF_noise.t,
     RILN_TD_struct.REF_noise.causality_correction_dB,
     RILN_TD_struct.REF_noise.truncation_dB) = s21_fn(port2_rn * H_bw_n * H_t_n * H_tw_n,
                                                       f_rn, param.sample_dt, OP, param)
    RILN_TD_struct.REF_noise.PR = lfilter(np.ones(M), 1, RILN_TD_struct.REF_noise.FIR)

    # ---- Extract ILN from cursor of REF.PR ----
    REF_PR = RILN_TD_struct.REF.PR
    RN_PR = RILN_TD_struct.REF_noise.PR
    ipeak = int(np.argmax(REF_PR))  # 0-based
    range_end = min(len(REF_PR), len(RN_PR))
    RILN_TD_struct.ILN = RN_PR[ipeak:range_end]
    RILN_TD_struct.t = RILN_TD_struct.REF_noise.t[ipeak:range_end]
    RILN_TD_struct.FOM = -np.inf
    RILN_TD_struct.FOM_PDF = -np.inf
    rms_fom = -np.inf

    # ---- FOM: max norm over M sample phases ----
    bin_size = float(getattr(OP, 'BinSize', 1e-3))
    specBER = float(getattr(param, 'specBER', 1e-4))
    ILN = RILN_TD_struct.ILN

    best_pdf = None
    for im in range(M):
        phase_samples = ILN[im::M]
        if len(phase_samples) == 0:
            continue
        RILN_TD_struct.FOM = max(RILN_TD_struct.FOM, float(np.linalg.norm(phase_samples)))
        pdf = pdf_fn(phase_samples, param.levels, bin_size, 0)
        # rms from PDF: sqrt(sum(y * x^2))
        rms = float(np.sqrt(np.sum(pdf.y * pdf.x**2))) * np.sqrt(2)
        if rms > rms_fom:
            rms_fom = rms
            # CDF threshold
            cdf_y = np.cumsum(pdf.y)
            idx_ber = int(np.searchsorted(cdf_y, specBER))
            if idx_ber < len(pdf.x):
                RILN_TD_struct.FOM_PDF = -pdf.x[idx_ber]
            else:
                RILN_TD_struct.FOM_PDF = -np.inf
            best_pdf = pdf

    RILN_TD_struct.PDF = best_pdf

    # ---- SNR metrics ----
    # MATLAB: db = @(x) 20*log10(abs(x)), applied to FIT.PR(ipeak)/FOM.  No eps
    # floor on the denominator: a zero FOM gives +Inf, and the floored form
    # reported a plausible ~6000 dB instead.  FIT.PR(ipeak) is indexed with no
    # guard, exactly as MATLAB does — it raises when FIT.PR was truncated
    # shorter than REF.PR, and that is the reference's behaviour.
    FIT_peak = np.float64(RILN_TD_struct.FIT.PR[ipeak])
    with np.errstate(divide='ignore', invalid='ignore'):
        RILN_TD_struct.SNR_ISI_FOM = 20.0 * np.log10(
            np.abs(FIT_peak / np.float64(RILN_TD_struct.FOM)))
        RILN_TD_struct.SNR_ISI_FOM_PDF = 20.0 * np.log10(
            np.abs(FIT_peak / np.float64(RILN_TD_struct.FOM_PDF)))

    return RILN_TD_struct


if __name__ == '__main__':
    # The six dependencies are resolved as module globals, so this file is
    # not runnable on its own: import it and inject them (or the assembled
    # sicopr.py, where they are already top-level).  test_verify.py does
    # exactly that and checks the result against the Octave reference.
    import subprocess
    import sys
    import os
    sys.exit(subprocess.call(
        [sys.executable, '-m', 'pytest',
         os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'test_verify.py'), '-v']))
