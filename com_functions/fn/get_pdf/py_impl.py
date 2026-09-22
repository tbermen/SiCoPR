# ============================================================
# MATLAB→Python translation notes for get_pdf
# MATLAB lines: 7377–7472
# ============================================================
# chdata.eq_pulse_response: row vector in MATLAB; use .ravel() in Python.
# t_s: 1-based sample index in MATLAB → 0-based in Python (called cursor_i or t_s-1).
# SBR(t_s + M*(0:ndfe)) → Python: SBR[t_s + M*np.arange(ndfe+1)] (0-based t_s).
# start_cancel = t_s - M//2  (MATLAB t_s - samp_UI/2).
# end_cancel = t_s + (0.5+ndfe)*M - 1  → Python slice end = t_s + (ndfe+1)*M - M//2.
# effective_cancellation_samples = kron(effective_cancelled_cursors, ones(1,M)):
#   Python: np.repeat(effective_cancelled_cursors, M).
# residual_response[start_cancel:end_cancel] -= effective_cancellation_samples.
# nui = round(len(residual_response)/M).
# vs: (nui-2) × M matrix; MATLAB vs(i,:) = residual_response(M*(1:nui-2)+i) 1-based:
#   Python: for i in range(M): vs[:,i] = residual_response[M*(1+np.arange(nui-2))+i-1] (0-based).
# phases for THRU: mod(t_s_0based+1, M) → if ==0, use M-1 (0-based index into vs cols).
#   MATLAB: phases = mod(t_s, M) where t_s is 1-based; if ==0, phases=M → col M (1-based)=col M-1 (0-based).
#   Python: t_s_matlab = t_s_python + 1; phase_matlab = (t_s_matlab) % M; if ==0: M → col M-1; else: col phase_matlab-1.
# phases for non-THRU: all columns 0..M-1.
# MMSE path (OP.FFE_OPT_METHOD=='MMSE' and OP.RxFFE): use ixphase column directly.
# else: loop over phases, compute sigma of each pdf, pick max.
# dfe_delta: quantization step. If ==0, ideal_cancelled_cursors_q = ideal_cancelled_cursors.
# bmax_vec/bmin_vec: [cursor, bmax*cursor] for non-floating; [cursor, use_bmax*cursor] for floating.
# dfe_clipper inlined as _dfe_clipper.
# get_pdf_from_sampled_signal inlined as _get_pdf_from_sampled_signal.
# ============================================================

import math

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

from scipy.signal import fftconvolve
from types import SimpleNamespace



# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.
_CONV_FFT_MIN = 128



def _mround(x):
    """MATLAB round(): half away from zero, where Python's round() is banker's.

    Not a theoretical concern here. `nui = round(len(residual_response)/M)` is a
    ratio of two INTEGERS, so it lands exactly on .5 whenever the response length
    is an odd multiple of M/2 -- a discrete quantity that really does hit the tie,
    not a continuous one where ties are measure-zero. When it does, MATLAB rounds
    up and Python rounds down to even, the vs matrix loses a row, and one ISI
    sample is dropped from the PDF: sgm_isi comes out LOW.

    Found on 4 of 208 reference case-instances (T1_R23 and T4_R10, both crosstalk
    conditions), where exactly one of ~98,000 round() calls in a run sits on the
    tie. docs/AUDIT_FINDINGS.md listed this site and dismissed it as "measure-zero
    for continuous data", which is true of continuous inputs and false of this one.
    """
    x = float(x)
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    # Off a tie round() is exact, and unlike floor(x + 0.5) it does not
    # send 0.49999999999999994 to 1: that sum is exactly 1.0 in binary.
    return int(round(x))

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


def _dfe_clipper(input_arr, max_threshold, min_threshold):
    inp = np.asarray(input_arr, dtype=float)
    hi = np.asarray(max_threshold, dtype=float)
    lo = np.asarray(min_threshold, dtype=float)

    # MATLAB isrow(input): true for 1-D or 2-D with shape[0]==1
    is_row = inp.ndim <= 1 or (inp.ndim == 2 and inp.shape[0] == 1)
    if is_row:
        hi = hi.ravel()                 # (:).' in MATLAB
        lo = lo.ravel()
    else:
        hi = hi.ravel().reshape(-1, 1)  # (:) in MATLAB — column vector
        lo = lo.ravel().reshape(-1, 1)

    out = inp.copy()
    # Both masks are taken from the ORIGINAL input: MATLAB computes
    # input<min_threshold, not clip_output<min_threshold, so with crossed
    # bounds (min>max) an element can be raised after being lowered.
    # Octave: dfe_clipper([0 1.5 3],[1 1 1],[2 2 2]) -> [2 2 1].
    # NaN compares false both ways and passes through unclipped.
    mask_hi = inp > hi
    mask_lo = inp < lo

    # MATLAB writes max_threshold(input>max_threshold): a logical index into
    # the THRESHOLD array. It errors as soon as a true position falls past the
    # end of that array, so a scalar threshold works only while nothing beyond
    # the first element is clipped. Octave:
    #     dfe_clipper([3 1 1], 2, -9) -> [2 1 1]      (only position 1 true)
    #     dfe_clipper([1 3 1], 2, -9) -> error: max_threshold(2): out of bound 1
    # numpy would instead broadcast the scalar and return a plausible answer
    # for a call MATLAB refuses.  Assigning POSITIONALLY rather than with a
    # boolean mask is what makes that emulation possible: numpy requires a
    # boolean index to match the array's shape exactly, while MATLAB only
    # requires every TRUE position to be in range.
    out_f = out.ravel(order='F')
    for mask, thr, nm in ((mask_hi, hi, 'max_threshold'),
                          (mask_lo, lo, 'min_threshold')):
        where = np.nonzero(np.asarray(mask).ravel(order='F'))[0]
        if where.size == 0:
            continue
        if where.max() >= thr.size:
            raise IndexError(
                'dfe_clipper: %s(%d): out of bound %d -- MATLAB indexes the '
                'threshold array with the input-shaped logical mask, so it '
                'errors here rather than broadcasting.'
                % (nm, where.max() + 1, thr.size))
        out_f[where] = thr.ravel(order='F')[where]
    return out_f.reshape(inp.shape, order='F')


def get_pdf(chdata, delta_y, t_s, param, OP, ixphase=None):
    """Compute ISI PDF from equalised pulse response (MATLAB lines 7377-7472).

    t_s: 0-based Python cursor index.
    ixphase: pre-determined phase column (0-based), used for MMSE+RxFFE path.
    Returns pdf SimpleNamespace.
    """
    SBR = np.asarray(chdata.eq_pulse_response, dtype=float).ravel()
    ch_type = chdata.type
    M = int(param.samples_per_ui)
    residual_response = SBR.copy()

    if ch_type == 'THRU':
        ndfe = int(param.N_bmax) if param.Floating_DFE else int(param.ndfe)
        idx = t_s + M * np.arange(ndfe + 1)
        idx = idx[idx < len(SBR)]
        ideal_cancelled_cursors = SBR[idx]

        if param.dfe_delta != 0:
            cursor_val = residual_response[t_s]
            dq = abs(ideal_cancelled_cursors / (cursor_val * param.dfe_delta))
            ideal_cancelled_cursors_q = (np.floor(dq) * cursor_val * param.dfe_delta
                                         * np.sign(ideal_cancelled_cursors))
        else:
            ideal_cancelled_cursors_q = ideal_cancelled_cursors

        cursor = SBR[t_s]
        if param.Floating_DFE:
            bmax_v = cursor * np.concatenate([[1.0], np.asarray(param.use_bmax, dtype=float).ravel()])
            bmin_v = cursor * np.concatenate([[1.0], np.asarray(param.use_bmin, dtype=float).ravel()])
        else:
            bmax_v = cursor * np.concatenate([[1.0], np.asarray(param.bmax, dtype=float).ravel()])
            bmin_v = cursor * np.concatenate([[1.0], np.asarray(param.bmin, dtype=float).ravel()])

        n_iq = len(ideal_cancelled_cursors_q)
        effective_cancelled_cursors = _dfe_clipper(
            ideal_cancelled_cursors_q,
            bmax_v[:n_iq],
            bmin_v[:n_iq]
        )

        effective_cancellation_samples = np.repeat(effective_cancelled_cursors, M)

        start_cancel = t_s - M // 2
        end_cancel = t_s + (1 + ndfe) * M - M // 2  # exclusive end for Python slice
        # trim to valid range
        if start_cancel < 0:
            effective_cancellation_samples = effective_cancellation_samples[-start_cancel:]
            start_cancel = 0
        ec_len = min(len(effective_cancellation_samples), end_cancel - start_cancel, len(residual_response) - start_cancel)
        residual_response[start_cancel:start_cancel + ec_len] -= effective_cancellation_samples[:ec_len]

    nui = _mround(len(residual_response) / M)

    # Build vs matrix: (nui-2) × M; MATLAB vs(i,:) = residual_response(M*(1:nui-2)+i) 1-based
    # Python (0-based): vs[:,i] = residual_response[M*np.arange(1, nui-1) + i]  (i=0..M-1)
    vs = np.zeros((nui - 2, M))
    for i in range(M):
        row_indices = M * np.arange(1, nui - 1) + i  # 0-based
        valid = row_indices[row_indices < len(residual_response)]
        vs[:len(valid), i] = residual_response[valid]

    # Determine phases
    use_mmse = (str(OP.FFE_OPT_METHOD).upper() == 'MMSE' and OP.RxFFE)
    if use_mmse:
        if ch_type == 'THRU':
            # MATLAB L7451-7452: THRU uses the CURSOR phase mod(t_s,M), NOT ixphase
            # (ixphase is only set for crosstalk channels; the THRU entry stays at its
            # init value 1, which is off-cursor and inflates the ISI PDF).
            t_s_matlab = t_s + 1  # 0-based -> 1-based
            phase_matlab = t_s_matlab % M
            ph_col = (M - 1) if phase_matlab == 0 else (phase_matlab - 1)
            pdf = _get_pdf_from_sampled_signal(vs[:, ph_col], int(param.levels), delta_y)
        else:
            # crosstalk: use the pre-determined ixphase (0-based column)
            ph_col = ixphase if ixphase is not None else 0
            pdf = _get_pdf_from_sampled_signal(vs[:, ph_col], int(param.levels), delta_y)
        return pdf

    # Non-MMSE: loop over phases, pick max sigma
    if ch_type == 'THRU':
        t_s_matlab = t_s + 1  # convert to 1-based
        phase_matlab = t_s_matlab % M
        if phase_matlab == 0:
            phases = [M - 1]  # MATLAB index M → 0-based M-1
        else:
            phases = [phase_matlab - 1]  # MATLAB phase_matlab (1-based col) → 0-based
    else:
        phases = list(range(M))

    mxV = np.zeros(M)
    pdf_samples = {}
    for k in phases:
        p = _get_pdf_from_sampled_signal(vs[:, k], int(param.levels), delta_y)
        pdf_samples[k] = p
        mxV[k] = float(np.sqrt(np.sum(p.x ** 2 * p.y)))

    best_k = phases[int(np.argmax([mxV[k] for k in phases]))]
    return pdf_samples[best_k]
