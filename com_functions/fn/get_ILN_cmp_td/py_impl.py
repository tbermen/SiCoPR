# ============================================================
# MATLAB→Python translation notes for get_ILN_cmp_td
# MATLAB lines: 6270–6378
# ============================================================
# Computes complex IL fitting and TD ILN.
# Polynomial fit: fmbg = [ones sdd21, sqrt(f)*sdd21, f*sdd21, f^2*sdd21] (column matrix).
# LGw = (sdd21 * log(abs(sdd21)) + 1j*unwrap(angle(sdd21))) transposed.
# alpha = pinv(fmbg) @ LGw → least-squares.
# efit_C = sum(alpha_i * basis_i); FIT = exp(efit_C).
# efit = dB(FIT); ILN = dB(sdd21) - efit.
# TD: calls the real s21_to_impulse_DC for non-zero sdd21 (eps-valued path for all-zero).
# ipeak: 0-based argmax of TD_ILN.REF.PR.
# Loop im=0..M-1: norm, pdf, cdf → FOM_PDF.
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

from scipy.signal import lfilter, fftconvolve
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

    The MATLAB comment calls this "equivalent to (p.Min:p.Min+length(p.y)-1)*
    p.BinSize", and it is not: the colon accumulates from the first element as
    a+k*d and pins the last element to the stated limit, while (a:b)*d forms
    each element as one product.  Swept over 7920 (Min, length, BinSize)
    combinations against Octave: the product form got 24.6% of the elements
    wrong, all by 1 ulp; this form got none.  The last element is pinned only
    when accumulation overshoots the limit -- pinning unconditionally is wrong,
    e.g. Min=-3, BinSize=0.1, 2 bins -> [-0.30000000000000004,
    -0.20000000000000004], not [..., -0.2].
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


def _s21_to_impulse_DC_zero(freq_array, time_step, OP, param):
    """Zero-input path of s21_to_impulse_DC (eps-valued output)."""
    freq_array = np.asarray(freq_array, dtype=float)
    fmax = 1.0 / time_step / 2.0
    freq_step = float(freq_array[2] - freq_array[1]) if len(freq_array) > 2 else float(freq_array[1] - freq_array[0])
    n_steps = _mround(fmax / freq_step)
    fout = np.arange(0, n_steps + 1) * (fmax / n_steps)
    IL_interp = np.full(len(fout), np.finfo(float).eps, dtype=complex)
    IL_symmetric = np.concatenate([
        [np.real(IL_interp[0])],
        IL_interp[1:-1],
        [np.real(IL_interp[-1])],
        np.conj(IL_interp[1:-1])[::-1]
    ])
    impulse_response = np.real(np.fft.ifft(IL_symmetric))
    L = len(impulse_response)
    t_base = np.arange(L) / (freq_step * L)
    ir_peak = _mmax(np.abs(impulse_response))
    thresh = getattr(OP, 'impulse_response_truncation_threshold', 1e-7)
    last_arr = np.where(np.abs(impulse_response) > ir_peak * thresh)[0]
    ir_last = int(last_arr[-1]) if len(last_arr) > 0 else L - 1
    voltage = impulse_response[:ir_last + 1]
    t_out = t_base[:ir_last + 1]
    return voltage, t_out, 0.0, -np.inf


def _d_cpdf(binsize, values, probs):
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if np.all(values == 0):
        return SimpleNamespace(BinSize=binsize, Min=0, y=np.array([1.0]), x=np.array([0.0]))
    if np.size(probs) < np.size(values):
        # MATLAB reads probs(k) for k = 1..length(values); a short probs is an
        # out-of-bound error, not a shorter answer.  zip() below stops at the
        # shorter of the two and silently normalised whatever it collected.
        # COM Octave 4p16p0: d_cpdf(1,[-1 0 1],[0.5 0.5]) errors
        # "probs(3): out of bound 2 (dimensions are 1x2)".
        raise IndexError('d_cpdf: probs is shorter than values')
    # ~issorted(values): MATLAB requires every element <= the next, which is
    # false as soon as a NaN is present.  np.diff(values) < 0 is False across a
    # NaN, so that form called [-1 NaN 1] sorted where MATLAB does not.
    if not np.all(values[:-1] <= values[1:]):
        si = np.argsort(values, kind='stable')
        values, probs = values[si], probs[si]
    values = binsize * _mround_arr(values / binsize)
    t_start = int(round(values[0] / binsize))
    t_end = int(round(values[-1] / binsize))
    t = np.arange(t_start, t_end + 1) * binsize
    pdf_y = np.zeros(len(t))
    for k, (v, prob) in enumerate(zip(values, probs)):
        bin_idx = (0 if k == 0 else len(t) - 1 if k == len(values) - 1
                   else int(np.argmin(np.abs(t - v))))
        pdf_y[bin_idx] += prob
    pdf_y = pdf_y / np.sum(pdf_y)

    if np.any(pdf_y < 0):
        raise ValueError('PDF must be real and nonnegative')
    # find(pdf.y) selects *nonzero*, and NaN counts as nonzero.  `> 0` drops
    # NaN, so an all-zero or NaN-bearing probs vector (pdf.y = 0/0) left the
    # support empty and raised instead of answering.  COM Octave 4p16p0:
    # d_cpdf(1,[-1 0 1],[0.5 NaN 0.5]) returns Min=-1, y=[NaN NaN NaN].
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
    p.Min = _mround(p1.Min + p2.Min)   # MATLAB round: half AWAY FROM ZERO
    p.y = _conv1d(p1.y, p2.y)
    pMax = p.Min + len(p.y) - 1
    p.x = _colon_x(p.Min, pMax, p.BinSize)   # (p.Min*BinSize:BinSize:pMax*BinSize)
    return p


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


def get_ILN_cmp_td(sdd21, faxis_f2, OP, param, A_T=None):
    """Complex IL fitting and TD ILN (MATLAB lines 6270-6378).

    sdd21: complex S21 (row or column vector).
    Returns (ILN, efit, TD_ILN).
    ILN and efit are in dB. TD_ILN contains time-domain struct.
    For non-zero sdd21: uses the real s21_to_impulse_DC for the TD section.
    """
    if A_T is None:
        A_T = 1.0

    sdd21 = np.asarray(sdd21, dtype=complex).ravel()
    faxis_f2 = np.asarray(faxis_f2, dtype=float).ravel()

    # Override OP settings for this function (per MATLAB L6279-6280, 6307)
    OP_copy = SimpleNamespace(**vars(OP))
    OP_copy.impulse_response_truncation_threshold = 1e-7
    OP_copy.interp_sparam_mag = 'trend_to_DC'
    OP_copy.interp_sparam_phase = 'interp_to_DC'

    # Polynomial basis for complex log fitting
    f = faxis_f2
    fmbg = np.column_stack([
        sdd21,
        np.sqrt(f) * sdd21,
        f * sdd21,
        f ** 2 * sdd21,
    ])  # shape (N, 4)

    unwraplog = np.log(np.abs(sdd21) + np.finfo(float).eps) + 1j * np.unwrap(np.angle(sdd21))
    LGw = (sdd21 * unwraplog).reshape(-1, 1)

    # Least-squares: alpha = pinv(fmbg) @ LGw
    alpha, _, _, _ = np.linalg.lstsq(fmbg, LGw, rcond=None)
    alpha = alpha.ravel()

    efit_C = (alpha[0] + alpha[1] * np.sqrt(f) + alpha[2] * f + alpha[3] * f ** 2)
    FIT = np.exp(efit_C)

    dB = lambda x: 20.0 * np.log10(np.abs(x) + np.finfo(float).eps)
    efit = dB(np.abs(FIT))
    ILN = dB(sdd21) - efit

    print('computing TD_ILN (dB) ...', end='')

    M = int(param.samples_per_ui)
    BinSize = float(getattr(OP, 'BinSize', 1e-4))
    specBER = float(param.specBER)

    # TX/BT/BW filters (MATLAB L6309-6313). H_bt forced use_BT=1; H_bw computed
    # but NOT used in the s21_to_impulse_DC call (matches MATLAB). H_tw = ones.
    H_bt = Bessel_Thomson_Filter(param, f, 1)
    H_bw = Butterworth_Filter(param, f, 1)
    H_t = np.exp(-(np.pi * f / 1e9 * float(OP.transmitter_transition_time) / 1.6832) ** 2)  # Eq 93A-46
    H_tw = np.ones(len(f))

    TD_ILN = SimpleNamespace()

    # REF (from sdd21) — MATLAB L6315-6318: s21_to_impulse_DC(sdd21.*H_bt.*H_t.*H_tw)
    if np.all(sdd21 == 0):
        fir_r, t_r, caus_r, trunc_r = _s21_to_impulse_DC_zero(f, float(param.sample_dt), OP_copy, param)
    else:
        fir_r, t_r, caus_r, trunc_r = s21_to_impulse_DC(
            sdd21 * H_bt * H_t * H_tw, f, float(param.sample_dt), OP_copy, param)

    TD_ILN.REF = SimpleNamespace(FIR=fir_r, t=t_r,
                                  causality_correction_dB=caus_r, truncation_dB=trunc_r)
    TD_ILN.REF.PR = lfilter(np.ones(M), [1.0], fir_r)

    # FIT (from complex fitted response) — MATLAB L6321-6324: s21_to_impulse_DC(FIT.*H_bt.*H_t.*H_tw)
    if np.all(sdd21 == 0):
        fir_f, t_f, caus_f, trunc_f = _s21_to_impulse_DC_zero(f, float(param.sample_dt), OP_copy, param)
    else:
        fir_f, t_f, caus_f, trunc_f = s21_to_impulse_DC(
            FIT * H_bt * H_t * H_tw, f, float(param.sample_dt), OP_copy, param)

    TD_ILN.FIT = SimpleNamespace(FIR=fir_f, t=t_f,
                                  causality_correction_dB=caus_f, truncation_dB=trunc_f)
    TD_ILN.FIT.PR = lfilter(np.ones(M), [1.0], fir_f)

    ipeak = int(np.where(TD_ILN.REF.PR == _mmax(TD_ILN.REF.PR))[0][0])
    range_end = min(len(TD_ILN.REF.PR), len(TD_ILN.FIT.PR))
    irange = slice(ipeak, range_end)

    TD_ILN.ILN = TD_ILN.FIT.PR[irange] - TD_ILN.REF.PR[irange]
    TD_ILN.t = TD_ILN.FIT.t[irange] if len(TD_ILN.FIT.t) > ipeak else TD_ILN.FIT.t
    TD_ILN.FOM = -np.inf
    TD_ILN.FOM_PDF = -np.inf

    rms_fom = -np.inf
    for im in range(M):
        samples = TD_ILN.ILN[im::M]
        TD_ILN.FOM = max(float(TD_ILN.FOM), float(np.linalg.norm(samples)))
        pdf = _get_pdf_from_sampled_signal(samples, int(param.levels), BinSize)
        pdf_y = np.asarray(pdf.y, dtype=float)
        pdf_x = np.asarray(pdf.x, dtype=float)
        rms = float(np.sqrt(np.dot(pdf_y, pdf_x ** 2)) * np.sqrt(2))
        cdf_y = np.cumsum(pdf_y)
        if rms > rms_fom:
            rms_fom = rms
            idx_ber = np.where(cdf_y >= specBER)[0]
            if len(idx_ber) > 0:
                TD_ILN.FOM_PDF = float(-pdf_x[idx_ber[0]])
            TD_ILN.PDF = pdf

    # MATLAB db = @(x) 20*log10(abs(x)); the abs() handles a negative fit_peak
    # (FIT.PR at the REF peak index can be < 0 -> bare log10 gives NaN).
    fit_peak = float(TD_ILN.FIT.PR[ipeak])
    TD_ILN.SNR_ISI_FOM = (float(20.0 * np.log10(abs(fit_peak / TD_ILN.FOM)))
                           if TD_ILN.FOM != 0 else np.inf)
    TD_ILN.SNR_ISI_FOM_PDF = (float(20.0 * np.log10(abs(fit_peak / TD_ILN.FOM_PDF)))
                               if TD_ILN.FOM_PDF != 0 else np.inf)
    print(f'{TD_ILN.SNR_ISI_FOM_PDF:.6g} dB')

    return ILN, efit, TD_ILN
