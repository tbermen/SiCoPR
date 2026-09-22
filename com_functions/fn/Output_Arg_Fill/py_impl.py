# ============================================================
# MATLAB→Python translation notes for Output_Arg_Fill
# MATLAB lines: 3976–4173
# ============================================================
# vma, str2csv, Burst_Probability_Calc, pdf2sgm inlined as private helpers.
# {chdata.base} → [ch.base for ch in chdata].
# switch lower(OP.TDECQ): false/'none' → VMA=[]; 'vma' → call vma.
# fom_result.ctle/best_G_high_pass are 1-based → subtract 1 for array access.
# DFE4_RSS: norm(taps[3:]);  DFE2_RSS: norm(taps[1:]).
# steady_state_voltage_weq: its_eq 0-based, isumend exclusive.
# Pre2Pmax: -taps[-3]/taps[-2] when len>=3.
# ============================================================

import numpy as np
from com_functions.fn.Init_PDF_Fast.py_impl import Init_PDF_Fast as _Init_PDF_Fast_b
from com_functions.fn.conv_fct.py_impl import conv_fct as _conv_fct_b
from com_functions.fn.d_cpdf.py_impl import d_cpdf as _d_cpdf_b
from com_functions.fn.PRBS13Q.py_impl import PRBS13Q as _PRBS13Q
from com_functions.fn.pam.py_impl import pam as _pam
from com_functions.fn.pdf2sgm.py_impl import pdf2sgm as _pdf2sgm
from com_functions.fn.vma.py_impl import vma as _vma

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


from scipy.signal import lfilter, fftconvolve
from types import SimpleNamespace


# --- inline vma helpers ---

# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.


# --- inline str2csv ---
def _str2csv(c):
    return ','.join(c)


# --- inline pdf2sgm ---


# --- inline Burst_Probability_Calc helpers ---


def _get_pdf_b(iv, L, BinSize):
    iv = np.asarray(iv, dtype=float).ravel()
    if _mmax(np.abs(iv)) > BinSize:
        iv = iv[np.abs(iv) > BinSize]
    else:
        return _d_cpdf_b(BinSize, 0, 1)
    iv[np.abs(iv) < BinSize] = 0.0
    b = np.sign(iv)
    sort_idx = np.argsort(np.abs(iv), kind='stable')[::-1]
    iv = np.abs(iv[sort_idx]) * b[sort_idx]
    values = 2.0 * np.arange(L) / (L - 1) - 1.0
    prob = np.ones(L) / L
    pdf = _d_cpdf_b(BinSize, 0, 1)
    empty_pdf = pdf
    for v in iv:
        pdfn = _Init_PDF_Fast_b(empty_pdf, np.abs(v) * values, prob)
        pdf = _conv_fct_b(pdf, pdfn)
    return pdf


def _Burst_Probability_Calc(COM_SNR_Struct, DFE_taps, param, OP):
    A_s = float(COM_SNR_Struct.A_s)
    ep_noise_pdf = [COM_SNR_Struct.combined_interference_and_noise_pdf]
    error_threshold = A_s / (10 ** ((float(param.pass_threshold) - float(OP.COM_EP_margin)) / 20.0))
    pdf0 = ep_noise_pdf[0]
    idx = np.where(pdf0.x >= error_threshold)[0]
    p_ep_0 = 1e-20 if len(idx) == 0 else float(np.sum(pdf0.y[idx[0]:]))
    p_error_propagation = [p_ep_0]
    sorted_abs_taps = np.sort(np.abs(np.asarray(DFE_taps).ravel()))[::-1]
    nburst = int(OP.nburst)
    ndfe = int(param.ndfe)
    for k in range(1, min(ndfe, nburst)):
        if OP.use_simple_EP_model:
            tap_val = 2.0 * A_s * float(_mmax(sorted_abs_taps))
            post_pdf = _get_pdf_b(tap_val, param.levels, param.delta_y)
            new_pdf = _conv_fct_b(ep_noise_pdf[0], post_pdf)
        else:
            tap_val = 2.0 * A_s * float(sorted_abs_taps[k - 1])
            post_pdf = _get_pdf_b(tap_val, param.levels, param.delta_y)
            new_pdf = _conv_fct_b(ep_noise_pdf[k - 1], post_pdf)
        ep_noise_pdf.append(new_pdf)
        idx = np.where(new_pdf.x >= error_threshold)[0]
        p_ep_k = 1e-20 if len(idx) == 0 else float(np.sum(new_pdf.y[idx[0]:]))
        p_error_propagation.append(p_ep_k)
    p_burst = np.cumprod(p_error_propagation)
    return p_burst, p_error_propagation


def Output_Arg_Fill(output_args, sigma_bn, Noise_Struct, COM_SNR_Struct, param, chdata,
                    fom_result, OP):
    M = int(param.samples_per_ui)

    tdecq = OP.TDECQ
    tdecq_str = str(tdecq).lower() if not isinstance(tdecq, bool) else ''
    if tdecq is False or tdecq_str in ('false', 'none', '0'):
        output_args.VMA = []
    elif tdecq_str == 'vma':
        est_vma = _vma(fom_result.sbr, M)
        output_args.VMA = est_vma.VMA
    else:
        raise ValueError(f'{OP.TDECQ} not recognized for TDECQ')

    bases = [ch.base for ch in chdata]
    fileset_str = _str2csv(bases)
    output_args.file_names = f'"{fileset_str}"'

    for odt_param in ('R_diepad', 'C_diepad', 'L_comp', 'C_bump'):
        setattr(output_args, odt_param, getattr(param, odt_param))
    for pkg_param in ('levels', 'Pkg_len_TX', 'Pkg_len_NEXT', 'Pkg_len_FEXT',
                      'Pkg_len_RX', 'R_diepad', 'pkg_Z_c', 'C_v'):
        setattr(output_args, pkg_param, getattr(param, pkg_param))

    output_args.baud_rate_GHz = param.fb / 1e9
    output_args.f_Nyquist_GHz = param.fb / 2e9
    output_args.BER = param.specBER
    output_args.FOM = fom_result.FOM
    output_args.sigma_N = Noise_Struct.sigma_N
    dfe_taps = np.asarray(fom_result.DFE_taps).ravel()
    output_args.DFE4_RSS = float(np.linalg.norm(dfe_taps[3:]))  # MATLAB taps(4:end)
    output_args.DFE2_RSS = float(np.linalg.norm(dfe_taps[1:]))  # MATLAB taps(2:end)
    output_args.tail_RSS = fom_result.tail_RSS
    output_args.channel_operating_margin_dB = COM_SNR_Struct.COM
    output_args.available_signal_after_eq_mV = 1000 * COM_SNR_Struct.A_s
    output_args.peak_uneq_pulse_mV = 1000 * float(_mmax(np.abs(
        np.asarray(chdata[0].uneq_pulse_response, dtype=float))))

    try:
        uneq_ir = np.asarray(chdata[0].uneq_imp_response, dtype=float)
        t_arr = np.asarray(chdata[0].t, dtype=float)
        output_args.uneq_FIR_peak_time = float(t_arr[uneq_ir == _mmax(uneq_ir)][0])
    except Exception:
        output_args.uneq_FIR_peak_time = []

    output_args.steady_state_voltage_mV = 1000 * fom_result.A_f

    eq_pr = np.asarray(chdata[0].eq_pulse_response, dtype=float)
    its_eq = int(np.where(eq_pr >= _mmax(eq_pr))[0][0])  # 0-based
    # MATLAB: isumend = min(its+N_v*M, len); its is 1-based → its_py+1+N_v*M
    isumend = min(its_eq + int(param.N_v) * M + 1, len(eq_pr))
    output_args.steady_state_voltage_weq_mV = 1000 * float(np.sum(eq_pr[:isumend]) / M)

    if OP.RX_CALIBRATION == 1 or OP.PSDRXCAL == 1:
        output_args.sigma_bn = sigma_bn
    else:
        output_args.sigma_bn = []

    output_args.Peak_ISI_XTK_and_Noise_interference_at_BER_mV = 1000 * COM_SNR_Struct.A_ni
    output_args.peak_ISI_XTK_interference_at_BER_mV = 1000 * Noise_Struct.peak_interference_at_BER
    output_args.peak_ISI_interference_at_BER_mV = 1000 * Noise_Struct.thru_peak_interference_at_BER
    output_args.equivalent_ICI_sigma_assuming_PDF_is_Gaussian_mV = Noise_Struct.sci_sigma * 1000

    if OP.RX_CALIBRATION == 0:
        output_args.peak_MDXTK_interference_at_BER_mV = 1000 * Noise_Struct.crosstalk_peak_interference_at_BER
        output_args.peak_MDNEXT_interference_at_BER_mV = 1000 * Noise_Struct.MDNEXT_peak_interference
        output_args.peak_MDFEXT_interference_at_BER_mV = 1000 * Noise_Struct.MDFEXT_peak_interference
    else:
        output_args.peak_MDXTK_interference_at_BER_mV = []
        output_args.peak_MDNEXT_interference_at_BER_mV = []
        output_args.peak_MDFEXT_interference_at_BER_mV = []

    xtk = int(param.num_next) + int(param.num_fext)
    if xtk > 0 and OP.RX_CALIBRATION == 0 and OP.TDMODE == 0:
        output_args.equivalent_ICN_assuming_Gaussian_PDF_mV = Noise_Struct.cci_sigma * 1000
    else:
        output_args.MDNEXT_ICN_92_46_mV = 0
        output_args.MDFEXT_ICN_92_47_mV = 0
        output_args.equivalent_ICN_assuming_PDF_is_Gaussian_mV = 0

    if COM_SNR_Struct.A_s != 0 and Noise_Struct.peak_interference_at_BER != 0:
        ber = float(param.specBER)
        output_args.SNR_ISI_XTK_normalized_1_sigma = 20 * np.log10(
            COM_SNR_Struct.A_s
            / (Noise_Struct.peak_interference_at_BER / np.sqrt(2)
               / float(np.real(np.arccos(1 - 2 * ber) if False else 0) or
                       _erfc_inv_approx(2 * ber))))
    else:
        output_args.SNR_ISI_XTK_normalized_1_sigma = []
    output_args.SNR_ISI_est = fom_result.SNR_ISI
    output_args.Pmax_by_Vf_est = fom_result.Pmax_by_Vf
    output_args.Tr_measured_from_step_ps = fom_result.Tr_measured_from_step / 1e-12

    ctle_i = int(fom_result.ctle) - 1  # 0-based
    ctle_type = param.CTLE_type
    if ctle_type == 'CL93':
        output_args.CTLE_zero_poles = [
            float(np.asarray(param.CTLE_fz).ravel()[ctle_i]),
            float(np.asarray(param.CTLE_fp2).ravel()[ctle_i]),
            float(np.asarray(param.CTLE_fp1).ravel()[ctle_i])]
        output_args.CTLE_DC_gain_dB = float(np.asarray(param.ctle_gdc_values).ravel()[ctle_i])
        output_args.g_DC_HP = []
        output_args.HP_poles_zero = []
    elif ctle_type == 'CL120d':
        output_args.CTLE_zero_poles = [
            float(np.asarray(param.CTLE_fz).ravel()[ctle_i]),
            float(np.asarray(param.CTLE_fp2).ravel()[ctle_i]),
            float(np.asarray(param.CTLE_fp1).ravel()[ctle_i])]
        output_args.CTLE_DC_gain_dB = float(np.asarray(param.ctle_gdc_values).ravel()[ctle_i])
        ghp_i = int(fom_result.best_G_high_pass) - 1  # 0-based
        output_args.g_DC_HP = float(np.asarray(param.g_DC_HP_values).ravel()[ghp_i])
        output_args.HP_poles_zero = float(np.asarray(param.f_HP).ravel()[ghp_i])
    elif ctle_type == 'CL120e':
        output_args.CTLE_zero_poles = [
            float(np.asarray(param.CTLE_fz).ravel()[ctle_i]),
            float(np.asarray(param.f_HP_Z).ravel()[ctle_i]),
            float(np.asarray(param.CTLE_fp2).ravel()[ctle_i]),
            float(np.asarray(param.CTLE_fp1).ravel()[ctle_i]),
            float(np.asarray(param.f_HP_P).ravel()[ctle_i])]
        output_args.CTLE_DC_gain_dB = float(np.asarray(param.ctle_gdc_values).ravel()[ctle_i])
        output_args.g_DC_HP = []
        output_args.HP_poles_zero = []

    output_args.TXLE_taps = fom_result.txffe
    taps = np.asarray(fom_result.txffe).ravel()
    if len(taps) >= 3:
        output_args.Pre2Pmax = float(-taps[-3] / taps[-2]) if taps[-2] != 0 else []
    else:
        output_args.Pre2Pmax = []

    output_args.DFE_taps = fom_result.DFE_taps
    if param.Floating_DFE or param.Floating_RXFFE:
        output_args.floating_tap_locations = fom_result.floating_tap_locations
    else:
        output_args.floating_tap_locations = []

    if OP.RxFFE:
        output_args.RxFFE = fom_result.RxFFE
        output_args.RxFFEgain = param.current_ffegain
    else:
        output_args.RxFFE = []
        output_args.RxFFEgain = []

    output_args.itick = fom_result.itick

    if OP.nburst > 0:
        p_burst, p_error_propagation = _Burst_Probability_Calc(
            COM_SNR_Struct, fom_result.DFE_taps, param, OP)
        output_args.error_propagation_probability = p_error_propagation
        output_args.burst_probabilities = p_burst
    else:
        output_args.error_propagation_probability = []
        output_args.burst_probabilities = []

    output_args.sgm_Ani__isi_xt_noise = _pdf2sgm(COM_SNR_Struct.combined_interference_and_noise_pdf)
    output_args.sgm_isi_xt = _pdf2sgm(Noise_Struct.isi_and_xtalk_pdf)
    output_args.sgm_noise__gaussian_noise_p_DD = _pdf2sgm(Noise_Struct.noise_pdf)
    output_args.sgm_p_DD = _pdf2sgm(Noise_Struct.p_DD)
    output_args.sgm_gaussian_noise = _pdf2sgm(Noise_Struct.gaussian_noise_pdf)
    output_args.sgm_G = Noise_Struct.sigma_G
    output_args.sgm_rjit = Noise_Struct.sigma_rjit
    output_args.sgm_N = Noise_Struct.sigma_N
    output_args.sgm_TX = Noise_Struct.sigma_TX
    output_args.sgm_isi = _pdf2sgm(Noise_Struct.sci_pdf)
    if OP.RX_CALIBRATION == 0:
        output_args.sgm_xt = _pdf2sgm(Noise_Struct.cci_pdf)
    else:
        output_args.sgm_xt = []

    if int(param.N_qb) != 0:
        output_args.sgm_Q = Noise_Struct.sigma_Q
        output_args.sigma_before_clip = Noise_Struct.sigma_before_clip
        output_args.peak_clip = Noise_Struct.peak_clip
        output_args.P2ptopsigma_clip = Noise_Struct.p2ptosigma_clip

    output_args.VEC_dB = COM_SNR_Struct.VEC_dB
    output_args.VEO_mV = COM_SNR_Struct.VEO_mV

    if OP.RX_CALIBRATION == 0 and OP.EW == 1:
        output_args.EW_UI_est = COM_SNR_Struct.EW_UI
        output_args.eye_contour = COM_SNR_Struct.eye_contour
        output_args.VEO_window_mUI = param.T_O
    else:
        output_args.EW_UI_est = []
        output_args.eye_contour = []
        output_args.VEO_window_mUI = []

    ac_cm = np.asarray(param.AC_CM_RMS) if hasattr(param, 'AC_CM_RMS') else np.array([0])
    if np.sum(ac_cm) != 0:
        output_args.sigma_ACCM_at_tp0_mV = chdata[0].sigma_ACCM_at_tp0 * 1000
        output_args.sigma_AC_CCM_at_rxpkg_output_mV = chdata[0].CD_CM_RMS * 1000
    else:
        output_args.sigma_ACCM_at_tp0_mV = []
        output_args.sigma_AC_CCM_at_rxpkg_output_mV = []

    if OP.MLSE:
        output_args.COM_orig = COM_SNR_Struct.COM_orig
        output_args.delta_COM = COM_SNR_Struct.delta_COM
        output_args.DER_DFE = COM_SNR_Struct.DER_DFE
        output_args.DER_MLSE = COM_SNR_Struct.DER_MLSE
        if str(getattr(OP, 'PHY', '')).upper() == 'C2M':
            output_args.VEC_dB_orig = COM_SNR_Struct.VEC_dB_orig
            output_args.delta_VEC = COM_SNR_Struct.delta_VEC
            output_args.VEC_dB = COM_SNR_Struct.VEC_dB

    output_args.COM_dB = COM_SNR_Struct.COM
    output_args.DER_thresh = COM_SNR_Struct.threshold_DER
    return output_args


def _erfc_inv_approx(y):
    from scipy.special import erfcinv
    return float(erfcinv(y))
