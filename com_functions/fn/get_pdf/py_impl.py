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
from com_functions.fn.dfe_clipper.py_impl import dfe_clipper as _dfe_clipper
from com_functions.fn.Init_PDF_Fast.py_impl import Init_PDF_Fast as _Init_PDF_Fast
from com_functions.fn.conv_fct.py_impl import conv_fct as _conv_fct
from com_functions.fn.d_cpdf.py_impl import d_cpdf as _d_cpdf

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


from scipy.signal import fftconvolve
from types import SimpleNamespace



# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.



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
    # NOTE: MATLAB sort(x,'descend') is STABLE and this reverses each tied run
    # (COM Octave: sort(abs([0.2 -0.2 0.5 0.2 -0.5 0.1]),'descend') indexes
    # [3 5 1 2 4 6]; this gives [5 3 4 2 1 6]).  Left as is because the loop
    # below uses only abs(v), so a tied run is a run of identical factors and
    # the PDF is bit-identical either way -- verified over 4 tie-rich vectors.
    # The canonical get_pdf_from_sampled_signal has the same line.
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
        # No clipping: MATLAB refuses a postcursor past the end of the pulse
        # response rather than quietly using fewer taps, and numpy's fancy
        # indexing raises the same way.
        # COM Octave: SBR 1x60, t_s=41, ndfe=3 -> "error: SBR(65): out of
        #   bound 60 (dimensions are 1x60)".  The port silently used 2 taps.
        idx = t_s + M * np.arange(ndfe + 1)
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
        # No trimming: MATLAB indexes residual_response(start_cancel:end_cancel)
        # and refuses either end of that range when it leaves the array, where a
        # python slice would silently shorten and cancel the wrong samples.
        # COM Octave: t_s=3 -> "error: residual_response(-1): subscripts must be
        #   either integers 1 to (2^63)-1 or logicals";
        #   rr=1x10; rr(5:12)=rr(5:12)-ones(1,8) -> "error: rr(12): out of
        #   bound 10 (dimensions are 1x10)".
        if start_cancel < 0:
            raise IndexError(
                'get_pdf: residual_response(%d) - the DFE cancellation window '
                'starts before the pulse response' % (start_cancel + 1))
        if end_cancel > len(residual_response):
            raise IndexError(
                'get_pdf: residual_response(%d): out of bound %d - the DFE '
                'cancellation window runs past the pulse response'
                % (end_cancel, len(residual_response)))
        residual_response[start_cancel:end_cancel] -= effective_cancellation_samples

    nui = _mround(len(residual_response) / M)

    # Build vs matrix: (nui-2) × M; MATLAB vs(i,:) = residual_response(M*(1:nui-2)+i) 1-based
    # Python (0-based): vs[:,i] = residual_response[M*np.arange(1, nui-1) + i]  (i=0..M-1)
    vs = np.zeros((nui - 2, M))
    for i in range(M):
        row_indices = M * np.arange(1, nui - 1) + i  # 0-based
        valid = row_indices[row_indices < len(residual_response)]
        vs[:len(valid), i] = residual_response[valid]

    # Determine phases.  MATLAB strcmp(OP.FFE_OPT_METHOD,'MMSE') is CASE
    # SENSITIVE (strcmpi is the insensitive one), so 'mmse' takes the phase-loop
    # path, not the MMSE one.
    # COM Octave: OP.FFE_OPT_METHOD='mmse', OP.RxFFE=1, type='NEXT' returns a
    #   407-bin pdf (the max-sigma phase); .upper() gave the 295-bin ixphase pdf.
    use_mmse = (str(OP.FFE_OPT_METHOD) == 'MMSE' and OP.RxFFE)
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

    # MATLAB sizes mxV from `phases`, which for THRU is a SCALAR, then writes
    # mxV(k) at the 1-based phase.  That write auto-grows mxV and leaves zeros
    # in 1..k-1, so when the phase's sigma is 0 max() picks index 1 -- a phase
    # pdf_samples never got, i.e. an unset struct whose every field is [].
    # COM Octave: THRU, t_s=40, M=8 (phase 8) with no residual at that phase
    #   returns numel(pdf.x)==0, numel(pdf.y)==0, pdf.BinSize==[].
    #   The port returned the delta pdf instead.
    if ch_type == 'THRU':
        k1 = phases[0] + 1                       # MATLAB's 1-based phase
        if k1 > 1 and mxV[phases[0]] == 0.0:
            return SimpleNamespace(BinSize=np.array([]), Min=np.array([]),
                                   y=np.array([]), x=np.array([]))
        return pdf_samples[phases[0]]

    best_k = phases[int(np.argmax([mxV[k] for k in phases]))]
    return pdf_samples[best_k]
