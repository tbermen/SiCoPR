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


def _mround_arr(x):
    """MATLAB round() on an array: halves go away from zero, where np.round
    takes them to even.

    Only exact ties are corrected. Adding 0.5 and truncating would be wrong:
    0.49999999999999994 + 0.5 is exactly 1.0 in double precision, so that form
    rounds the largest double below a half up to 1 where MATLAB gives 0.
    """
    x = np.asarray(x, dtype=float)
    tie = np.abs(x - np.trunc(x)) == 0.5
    return np.where(tie, np.trunc(x) + np.copysign(1.0, x), np.round(x))

from scipy.signal import lfilter, fftconvolve
from types import SimpleNamespace


# --- inline vma helpers ---

# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.
_CONV_FFT_MIN = 128


def _conv1d(a, b):
    """Convolve two 1-D PDFs, choosing direct or FFT by operand size."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # conv2 with an empty operand returns empty; np.convolve raises instead.
    # COM Octave: p1.y=[1 2 3], p2.y=[] -> p.y is 0x0, p.x is 1x0, p.Min=-1.
    if a.size == 0 or b.size == 0:
        return np.zeros(0)
    if min(a.size, b.size) >= _CONV_FFT_MIN:
        return fftconvolve(a, b)
    return np.convolve(a, b)


def _mround(x):
    """MATLAB round(): half away from zero, where Python's round() is banker's.

    COM Octave: round(0.5)=1, round(-0.5)=-1, round(2.5)=3; Python gives
    0, 0, 2.
    """
    x = float(x)
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    return int(round(x))


def _colon_x(pmin, pmax, binsize):
    """MATLAB `pmin*binsize : binsize : pmax*binsize`.

    The colon accumulates from the first element as a+k*d and pins the last
    element to the limit only when accumulation overshoots it; the product form
    (pmin:pmax)*binsize builds each element as one product instead, and the two
    differ by 1 ulp on most bins.  Swept over 7920 (Min, length, BinSize)
    combinations against COM Octave, the product form got 24.6% of elements
    wrong; this form got none.
    """
    n = pmax - pmin + 1
    if n <= 0:
        return np.zeros(0)
    a = pmin * binsize
    b = pmax * binsize
    x = a + np.arange(n) * binsize
    if (binsize > 0 and x[-1] > b) or (binsize < 0 and x[-1] < b):
        x[-1] = b
    return x


def _lfsr(s, t):
    s = [int(b) for b in s]
    n = len(s)
    t = [int(x) - 1 for x in t]
    m = len(t)
    c = [s[:]]
    for _ in range(2 ** n - 2):
        b = [0] * m
        b[0] = s[t[0]] ^ s[t[1]]
        for i in range(m - 2):
            b[i + 1] = s[t[i + 2]] ^ b[i]
        for j in range(n - 1):
            s[n - 1 - j] = s[n - 2 - j]
        s[0] = b[m - 2]
        c.append(s[:])
    c_arr = np.array(c, dtype=int)
    return c_arr[:, n - 1]


def _pam(data):
    # MATLAB assigns dataout(ceil(i/2)) only inside the four if/elseif arms. A
    # pair that matches none leaves that slot UNASSIGNED, and MATLAB's
    # auto-grow then fills it with 0 -- but only if some LATER index is
    # assigned, because the array only ever grows to the highest assigned
    # index. Verified against Octave:
    #     pam([0 0 1 1]) -> [0 1/3]      (slot 1 back-filled with 0)
    #     pam([1 1 0 0]) -> [1/3]        (length 1, NOT 2)
    #     pam([1]), pam([]) -> error: value on right hand side is undefined
    data = np.asarray(data, dtype=float)
    n_pairs = int(np.floor(len(data) / 2))
    assigned = {}
    for i in range(n_pairs):
        pair = data[2 * i:2 * i + 2]
        if np.array_equal(pair, [-1, -1]):
            assigned[i] = -1.0
        elif np.array_equal(pair, [-1, 1]):
            assigned[i] = -1.0 / 3.0
        elif np.array_equal(pair, [1, 1]):
            assigned[i] = 1.0 / 3.0
        elif np.array_equal(pair, [1, -1]):
            assigned[i] = 1.0
    if not assigned:
        raise ValueError(
            'pam: no input pair matched a Grey-code symbol, so MATLAB never '
            'assigns dataout and errors with "Output argument dataout (and '
            'maybe others) not assigned". Got %d sample(s).' % len(data))
    out = np.zeros(max(assigned) + 1)
    for i, v in assigned.items():
        out[i] = v
    return out


def _PRBS13Q():
    seq_bits = _lfsr([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1], [13, 12, 2, 1])
    seq_nrz = 2.0 * (seq_bits - 0.5)
    seq = _pam(seq_nrz)
    syms = np.zeros(len(seq), dtype=int)
    syms[_mround_arr(2 * (seq + 1)) / 2 == 2] = 3
    syms[_mround_arr(2 * (seq + 1)) / 2 == 1.5] = 2
    syms[_mround_arr(2 * (seq + 1)) / 2 == 0.5] = 1
    return seq, syms, seq_nrz


def _strfind_int(arr, pattern):
    arr = np.asarray(arr)
    n, m = len(arr), len(pattern)
    return np.array([i for i in range(n - m + 1) if np.array_equal(arr[i:i + m], pattern)], dtype=int)


def _vma(PR, M):
    PR = np.asarray(PR, dtype=float).ravel()
    M = int(M)
    seq, syms, _ = _PRBS13Q()
    symbols = seq
    imaxPR = int(np.argmax(PR))
    pos_3x7 = _strfind_int(syms, [3, 3, 3, 3, 3, 3, 3])
    indx_S3x7_start = M * pos_3x7 + imaxPR
    indx_S3x7_end = M * (pos_3x7 + 6) + imaxPR
    pos_0x6 = _strfind_int(syms, [0, 0, 0, 0, 0, 0])
    indx_S0x6_start = M * (pos_0x6 + 1) + imaxPR - 1
    indx_S0x6_end = M * (pos_0x6 + 5) + imaxPR
    unit_pulse = np.zeros(M)
    unit_pulse[0] = 1
    shifting_vector = np.kron(symbols, unit_pulse)
    Bit_stream_response = lfilter(PR, [1.0], shifting_vector)
    icent3 = int(np.floor((indx_S3x7_end - indx_S3x7_start) / 2 + indx_S3x7_start)[0])
    icent0 = int(np.floor((indx_S0x6_end - indx_S0x6_start) / 2 + indx_S0x6_start)[0])
    P_3 = float(np.mean(Bit_stream_response[icent3 - M:icent3 + M + 1]))
    P_0 = float(np.mean(Bit_stream_response[icent0 - M:icent0 + M + 1]))
    VMA = P_3 - P_0
    return SimpleNamespace(P_3=P_3, P_0=P_0, VMA=VMA)


# --- inline str2csv ---
def _str2csv(c):
    return ','.join(c)


# --- inline pdf2sgm ---
def _pdf2sgm(pdf):
    x = np.asarray(pdf.x, dtype=float)
    y = np.asarray(pdf.y, dtype=float)
    avg = np.sum(x * y)
    return float(np.sqrt(np.sum((x - avg) ** 2 * y)))


# --- inline Burst_Probability_Calc helpers ---
def _conv_fct_b(p1, p2):
    if p1.BinSize != p2.BinSize:
        raise ValueError('bin size must be equal')
    p = SimpleNamespace(**vars(p1))
    p.Min = _mround(p1.Min + p2.Min)         # MATLAB round: half away from zero
    p.y = _conv1d(p1.y, p2.y)
    pMax = p.Min + len(p.y) - 1
    p.x = _colon_x(p.Min, pMax, p.BinSize)   # MATLAB colon, not (Min:pMax)*BinSize
    return p


def _d_cpdf_b(binsize, values, probs):
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if np.all(values == 0):
        return SimpleNamespace(BinSize=binsize, Min=0, y=np.array([1.0]), x=np.array([0.0]))
    if np.size(probs) < np.size(values):
        # MATLAB reads probs(k) for k = 1..length(values); a short probs is an
        # out-of-bound error, not a shorter answer. zip() below stopped at the
        # shorter of the two and silently normalised what it had.
        raise IndexError('d_cpdf: probs is shorter than values')
    # MATLAB issorted() needs every element <= the next, which is FALSE across
    # a NaN. np.diff(values) < 0 is also false across a NaN, so [-1 NaN 1] was
    # called sorted and answered instead of being rejected.
    if not np.all(values[:-1] <= values[1:]):
        si = np.argsort(values, kind='stable')
        values, probs = values[si], probs[si]
    values = binsize * _mround_arr(values / binsize)
    t_start = int(round(values[0] / binsize))
    t_end = int(round(values[-1] / binsize))
    t = np.arange(t_start, t_end + 1) * binsize
    pdf_y = np.zeros(len(t))
    for k, (v, prob) in enumerate(zip(values, probs)):
        if k == 0:
            bin_idx = 0
        elif k == len(values) - 1:
            bin_idx = len(t) - 1
        else:
            bin_idx = int(np.argmin(np.abs(t - v)))
        pdf_y[bin_idx] += prob
    pdf_y = pdf_y / np.sum(pdf_y)

    if np.any(pdf_y < 0):
        raise ValueError('PDF must be real and nonnegative')
    # find(pdf.y) selects NONZERO, and NaN counts as nonzero. `> 0` drops
    # NaN, so an all-zero or NaN-bearing probs vector left the support
    # empty and raised instead of answering.
    support = np.where(pdf_y != 0)[0]
    pdf_y = pdf_y[support[0]:support[-1] + 1]
    pdf_min = t_start + int(support[0])
    return SimpleNamespace(BinSize=binsize, Min=pdf_min, y=pdf_y,
                           x=np.arange(pdf_min, -pdf_min + 1) * binsize)


def _Init_PDF_Fast_b(EmptyPDF, values, probs):
    pdf = SimpleNamespace(**vars(EmptyPDF))
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    rvd = _mround_arr(values / pdf.BinSize).astype(int)
    pdf.x = np.arange(rvd[0], rvd[-1] + 1) * pdf.BinSize
    pdf.Min = int(rvd[0])
    pdf.y = np.zeros(len(pdf.x))
    bp = rvd - rvd[0]
    # pdf.x spans rvd[0]..rvd[-1]; a value outside that span (i.e.
    # `values` is not ascending) makes bin_placement fall off the array
    # and MATLAB stops. A negative index is legal in numpy, so Python
    # wrapped round and added the probability to the wrong bin.
    if bp.size and (bp.min() < 0 or bp.max() >= pdf.y.size):
        raise IndexError('Init_PDF_Fast: values must be ascending')
    pdf.y[bp[0]] = probs[0]
    for k in range(1, len(values)):
        pdf.y[bp[k]] += probs[k]
    return pdf


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
