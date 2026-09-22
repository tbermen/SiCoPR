# ============================================================
# MATLAB→Python translation notes for Create_Noise_PDF
# MATLAB lines: 1552–1680
# ============================================================
# Inputs: A_s, param, fom_result, chdata, OP, sigma_bn, PSD_results.
# RX_CALIBRATION==1: compute sigma_ne via get_sigma_noise (inlined).
# NS.sigma_N = fom_result.sigma_N.
# Non-MMSE path: sigma_TX uses SNR_TX param; sigma_G = norm([sigma_RJ*sigma_X*norm(h_J), sigma_N, sigma_TX]).
# MMSE path: sigma_TX/sigma_G/sigma_rjit/sigma_N from PSD_results.
# NS.ber_q: erfcinv-based from specBER, or Noise_Crest_Factor if nonzero.
# normal_dist inlined as _normal_dist.
# conv_fct inlined as _conv_fct.
# d_cpdf inlined as _d_cpdf.
# NS.p_DD = get_pdf_from_sampled_signal(A_DD*h_J, levels, delta_y) [inlined].
# NS.sci_pdf = chdata[0].pdfr.
# sci_mxi, sci_msi: find(cumsum(sci_pdf.y) >= specBER, 1, 'first') 1-based → argmax on cumsum.
# CCI loop: k=2..number_of_s4p_files (1-based) → Python k=1..N-1 (0-based).
# combined_interference_and_noise_pdf = conv(isi_and_xtalk_pdf, noise_pdf).
# N_qb != 0 path (Eq 93A-37, MATLAB L1674-1675): calls the top-level
#   adjust_Rx_noise_for_quantization. That fn references fom_result.RxFFE, so the
#   quantization path is only valid together with RxFFE (true in MATLAB too).
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


def _mround(x):
    """MATLAB round(): half away from zero, where Python's round() is banker's."""
    x = float(x)
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    # Off a tie round() is exact, and unlike floor(x + 0.5) it does not
    # send 0.49999999999999994 to 1: that sum is exactly 1.0 in binary.
    return int(round(x))


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

from scipy.signal import fftconvolve
from scipy.special import erfcinv
from types import SimpleNamespace



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


def _d_cpdf(binsize, values, probs):
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if np.all(values == 0):
        return SimpleNamespace(BinSize=binsize, Min=0, y=np.array([1.0]), x=np.array([0.0]))
    if np.size(probs) < np.size(values):
        # MATLAB reads probs(k) for k = 1..length(values); a short probs is an
        # out-of-bound error, not a shorter answer.  zip() below would stop at
        # the shorter of the two and silently normalise whatever it collected.
        # COM Octave 4p16p0: d_cpdf(1,[-1 0 1],[0.5 0.5]) errors
        # "probs(3): out of bound 2 (dimensions are 1x2)".
        raise IndexError('d_cpdf: probs is shorter than values')
    # ~issorted(values): MATLAB requires every element <= the next, which is
    # false as soon as a NaN is present.  np.diff(values) < 0 is False across a
    # NaN, so that form calls [-1 NaN 1] sorted where MATLAB does not.
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
    # find(pdf.y) selects *nonzero*, and NaN counts as nonzero.  `> 0` drops
    # NaN, so an all-zero or NaN-bearing probs vector (pdf.y = 0/0) left the
    # support empty and raised instead of answering.
    # COM Octave 4p16p0: d_cpdf(1,[-1 0 1],[0.5 NaN 0.5]) returns
    # Min=-1, x=[-1 0 1], y=[NaN NaN NaN]; likewise probs=[0 0 0].
    support = np.where(pdf_y != 0)[0]
    pdf_y = pdf_y[support[0]:support[-1] + 1]
    pdf_min = t_start + int(support[0])
    return SimpleNamespace(BinSize=binsize, Min=pdf_min, y=pdf_y,
                           x=np.arange(pdf_min, -pdf_min + 1) * binsize)


def _Init_PDF_Fast(EmptyPDF, values, probs):
    pdf = SimpleNamespace(**vars(EmptyPDF))
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    rvd = _mround_arr(values / pdf.BinSize).astype(int)
    pdf.x = np.arange(rvd[0], rvd[-1] + 1) * pdf.BinSize
    pdf.Min = int(rvd[0])
    pdf.y = np.zeros(len(pdf.x))
    bp = rvd - rvd[0]
    if np.any(bp < 0) or np.any(bp >= len(pdf.y)):
        # pdf.x only spans rvd(1)..rvd(end), so any value that rounds outside
        # that span (i.e. `values` is not ascending) makes bin_placement fall
        # off the array and MATLAB stops.  A negative index is legal in numpy,
        # so Python wrapped round and added the probability to the wrong bin.
        # COM Octave 4p16p0: Init_PDF_Fast(E,[0 -0.2 0.3],[0.2 0.3 0.5]) with
        # BinSize=0.1 errors "pdf(-1): subscripts must be either integers
        # 1 to (2^63)-1 or logicals"; Python answered y=[0.2 0 0.3 0.5].
        raise IndexError('Init_PDF_Fast: values must be ascending')
    pdf.y[bp[0]] = probs[0]
    for k in range(1, len(values)):
        pdf.y[bp[k]] += probs[k]
    return pdf


def _conv_fct(p1, p2):
    if p1.BinSize != p2.BinSize:
        raise ValueError('bin size must be equal')
    p = SimpleNamespace(**vars(p1))
    p.Min = _mround(p1.Min + p2.Min)         # MATLAB round: half away from zero
    p.y = _conv1d(p1.y, p2.y)
    pMax = p.Min + len(p.y) - 1
    p.x = _colon_x(p.Min, pMax, p.BinSize)   # (p.Min*BinSize:BinSize:pMax*BinSize)
    return p


def _normal_dist(sigma, nsigma, binsize):
    pdf = SimpleNamespace()
    pdf.BinSize = binsize
    pdf.Min = -_mround(2 * nsigma * sigma / binsize)
    pdf.x = np.arange(pdf.Min, -pdf.Min + 1) * binsize
    eps = np.finfo(float).eps
    pdf.y = np.exp(-pdf.x ** 2 / (2 * sigma ** 2 + eps))
    pdf.y = pdf.y / np.sum(pdf.y)
    return pdf


def _get_pdf_from_sampled_signal(input_vector, L, BinSize):
    iv = np.asarray(input_vector, dtype=float).ravel()
    if len(iv) == 0:
        return _d_cpdf(BinSize, 0, 1)
    if _mmax(np.abs(iv)) > BinSize:
        iv = iv[np.abs(iv) > BinSize]
    else:
        return _d_cpdf(BinSize, 0, 1)
    iv[np.abs(iv) < BinSize] = 0.0
    b = np.sign(iv)
    sort_idx = np.argsort(np.abs(iv), kind='stable')[::-1]
    iv = np.abs(iv[sort_idx]) * b[sort_idx]
    values = 2.0 * np.arange(L) / (L - 1) - 1.0
    prob = np.ones(L) / L
    pdf = _d_cpdf(BinSize, 0, 1)
    empty_pdf = pdf
    for v in iv:
        pdfn = _Init_PDF_Fast(empty_pdf, np.abs(v) * values, prob)
        pdf = _conv_fct(pdf, pdfn)
    return pdf


def _get_sigma_noise(H_ctf, param, chdata, sigma_bn):
    """Inlined get_sigma_noise for RX_CALIBRATION path."""
    H_ctf = np.asarray(H_ctf, dtype=complex)
    faxis = np.asarray(chdata[1].faxis, dtype=float).ravel()
    H_r = 1.0 / np.polyval(
        [1, 2.613126, 3.414214, 2.613126, 1],
        1j * faxis / (float(param.f_r) * float(param.fb))
    )
    idx0 = int(np.argmax(faxis >= float(param.fb) / 2))
    idxfbby2 = idx0 + 1

    if len(chdata) >= 2:
        Hnoise_channel = np.asarray(chdata[1].sdd21, dtype=complex).ravel()
    else:
        Hnoise_channel = np.ones(len(faxis), dtype=complex)

    f = faxis
    f_hp = float(param.f_hp)
    if f_hp != 0:
        H_hp = (-1j * f / f_hp) / (1 + 1j * f / f_hp)
    else:
        H_hp = np.ones(len(f), dtype=complex)

    H_np = Hnoise_channel * H_ctf * H_r * H_hp
    sigma_NE = float(sigma_bn) * np.sqrt(np.mean(np.abs(H_np[:idxfbby2]) ** 2))
    sigma_HP = float(sigma_bn) * np.mean(np.abs(H_hp[:idxfbby2]) ** 2)
    return sigma_NE, sigma_HP


def Create_Noise_PDF(A_s, param, fom_result, chdata, OP, sigma_bn, PSD_results=None):
    """Compute combined interference and noise PDF/CDF (MATLAB lines 1552-1680).

    Returns (PDF, CDF, NS) where NS is a SimpleNamespace of noise parameters.
    """
    NS = SimpleNamespace()
    delta_y = float(param.delta_y)

    if OP.RX_CALIBRATION:
        ctle_i = int(fom_result.ctle) - 1  # 0-based
        faxis2 = np.asarray(chdata[1].faxis, dtype=float).ravel()
        ctle_gain2 = ((10.0 ** (float(param.ctle_gdc_values[ctle_i]) / 20.0)
                       + 1j * faxis2 / float(param.CTLE_fz[ctle_i]))
                      / ((1 + 1j * faxis2 / float(param.CTLE_fp1[ctle_i]))
                         * (1 + 1j * faxis2 / float(param.CTLE_fp2[ctle_i]))))
        ctype = str(param.CTLE_type)
        if ctype == 'CL93':
            H_low2 = 1.0
        elif ctype == 'CL120d':
            ghp_i = int(fom_result.best_G_high_pass) - 1
            H_low2 = ((10.0 ** (float(param.g_DC_HP_values[ghp_i]) / 20.0)
                       + 1j * faxis2 / float(param.f_HP[ghp_i]))
                      / (1 + 1j * faxis2 / float(param.f_HP[ghp_i])))
        elif ctype == 'CL120e':
            H_low2 = ((1 + 1j * faxis2 / float(param.f_HP_P[ctle_i]))
                      / (1 + 1j * faxis2 / float(param.f_HP_Z[ctle_i])))
        else:
            H_low2 = 1.0
        H_ctf2 = H_low2 * ctle_gain2
        sigma_ne, NS.sigma_hp = _get_sigma_noise(H_ctf2, param, chdata, sigma_bn)
    else:
        sigma_ne = 0.0

    NS.sigma_N = float(fom_result.sigma_N)

    use_mmse = (str(OP.FFE_OPT_METHOD).upper() == 'MMSE' and OP.RxFFE)
    if not use_mmse:
        if not OP.SNR_TXwC0:
            NS.sigma_TX = float((param.levels - 1) * A_s / param.R_LM
                                * 10.0 ** (-param.SNR_TX / 20.0))
        else:
            cur = int(fom_result.cur) - 1  # 0-based
            NS.sigma_TX = float((param.levels - 1) * A_s
                                / float(fom_result.txffe[cur])
                                / param.R_LM
                                * 10.0 ** (-param.SNR_TX / 20.0))
        h_J = np.asarray(fom_result.h_J, dtype=float).ravel()
        NS.sigma_rjit = float(param.sigma_RJ * param.sigma_X * np.linalg.norm(h_J))
        NS.sigma_G = float(np.linalg.norm([NS.sigma_rjit, NS.sigma_N, NS.sigma_TX]))
    else:
        if OP.PSDRXCAL:
            NS.sigma_hp = float(sigma_bn)
        NS.sigma_TX = float(PSD_results.S_tn_rms)
        NS.sigma_G = float(PSD_results.S_G_rms)
        NS.sigma_rjit = float(PSD_results.S_rj_rms)
        NS.sigma_N = float(PSD_results.S_rn_rms)

    if param.Noise_Crest_Factor == 0:
        NS.ber_q = float(np.sqrt(2) * erfcinv(2 * param.specBER))
    else:
        NS.ber_q = float(param.Noise_Crest_Factor)

    NS.gaussian_noise_pdf = _normal_dist(NS.sigma_G, NS.ber_q, delta_y)

    if OP.force_BBN_Q_factor:
        NS.ne_noise_pdf = _normal_dist(sigma_ne, OP.BBN_Q_factor, delta_y)
    else:
        NS.ne_noise_pdf = _normal_dist(sigma_ne, NS.ber_q, delta_y)
    NS.gaussian_noise_pdf = _conv_fct(NS.gaussian_noise_pdf, NS.ne_noise_pdf)

    h_J = np.asarray(fom_result.h_J, dtype=float).ravel()
    NS.p_DD = _get_pdf_from_sampled_signal(param.A_DD * h_J, int(param.levels), delta_y)

    NS.noise_pdf = _conv_fct(NS.gaussian_noise_pdf, NS.p_DD)
    gaussian_rjitt_pdf = _normal_dist(NS.sigma_rjit, NS.ber_q, delta_y)
    NS.jitt_pdf = _conv_fct(gaussian_rjitt_pdf, NS.p_DD)

    NS.sci_pdf = chdata[0].pdfr

    def _first_ber_idx(pdf_obj):
        cdf_y = np.cumsum(np.asarray(pdf_obj.y, dtype=float))
        idxs = np.where(cdf_y >= float(param.specBER))[0]
        return int(idxs[0]) if len(idxs) > 0 else len(cdf_y) - 1

    sci_mxi = _first_ber_idx(NS.sci_pdf)
    NS.thru_peak_interference_at_BER = float(abs(NS.sci_pdf.x[sci_mxi]))
    NS.sci_sigma = float(abs(NS.sci_pdf.x[sci_mxi]
                             / (erfcinv(2 * param.specBER) * np.sqrt(2))))

    if not OP.RX_CALIBRATION:
        MDNEXT_cci_pdf = _d_cpdf(delta_y, 0, 1)
        MDFEXT_cci_pdf = _d_cpdf(delta_y, 0, 1)
        # MATLAB k=2..number_of_s4p_files (1-based) → Python k=1..len(chdata)-1 (0-based)
        n_files = getattr(param, 'number_of_s4p_files', len(chdata))
        for k in range(1, n_files):
            if k >= len(chdata):
                break
            if chdata[k].type == 'NEXT':
                MDNEXT_cci_pdf = _conv_fct(MDNEXT_cci_pdf, chdata[k].pdfr)
            elif chdata[k].type == 'FEXT':
                MDFEXT_cci_pdf = _conv_fct(MDFEXT_cci_pdf, chdata[k].pdfr)
            else:
                raise ValueError(f'Crosstalk PDF unexpected channel type: {chdata[k].type!r}')

        mdnxi = _first_ber_idx(MDNEXT_cci_pdf)
        NS.MDNEXT_peak_interference = float(abs(MDNEXT_cci_pdf.x[mdnxi]))
        mdfxi = _first_ber_idx(MDFEXT_cci_pdf)
        NS.MDFEXT_peak_interference = float(abs(MDFEXT_cci_pdf.x[mdfxi]))

        NS.cci_pdf = _conv_fct(MDFEXT_cci_pdf, MDNEXT_cci_pdf)
        cci_mxi = _first_ber_idx(NS.cci_pdf)
        NS.crosstalk_peak_interference_at_BER = float(abs(NS.cci_pdf.x[cci_mxi]))
        NS.cci_sigma = float(abs(NS.cci_pdf.x[cci_mxi]
                                 / (erfcinv(2 * param.specBER) * np.sqrt(2))))
        NS.isi_and_xtalk_pdf = _conv_fct(NS.sci_pdf, NS.cci_pdf)
    else:
        NS.isi_and_xtalk_pdf = NS.sci_pdf

    mxi = _first_ber_idx(NS.isi_and_xtalk_pdf)
    NS.peak_interference_at_BER = float(abs(NS.isi_and_xtalk_pdf.x[mxi]))

    combined_interference_and_noise_pdf = _conv_fct(NS.isi_and_xtalk_pdf, NS.noise_pdf)

    if int(param.N_qb) != 0:
        # Equation 93A-37 (MATLAB L1674-1675): apply Rx ADC quantization noise.
        # adjust_Rx_noise_for_quantization is a top-level fn in the assembled module.
        chdata, NS, combined_interference_and_noise_pdf = adjust_Rx_noise_for_quantization(
            combined_interference_and_noise_pdf, NS, chdata, fom_result, param, OP)

    CDF = np.cumsum(np.asarray(combined_interference_and_noise_pdf.y, dtype=float))
    return combined_interference_and_noise_pdf, CDF, NS
