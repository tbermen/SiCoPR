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

import numpy as np
from types import SimpleNamespace


def _d_cpdf(binsize, values, probs):
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if np.all(values == 0):
        return SimpleNamespace(BinSize=binsize, Min=0, y=np.array([1.0]), x=np.array([0.0]))
    if np.any(np.diff(values) < 0):
        si = np.argsort(values)
        values, probs = values[si], probs[si]
    values = binsize * np.round(values / binsize)
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
    support = np.where(pdf_y > 0)[0]
    pdf_y = pdf_y[support[0]:support[-1] + 1]
    pdf_min = t_start + int(support[0])
    return SimpleNamespace(BinSize=binsize, Min=pdf_min, y=pdf_y,
                           x=np.arange(pdf_min, -pdf_min + 1) * binsize)


def _Init_PDF_Fast(EmptyPDF, values, probs):
    pdf = SimpleNamespace(**vars(EmptyPDF))
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    rvd = np.round(values / pdf.BinSize).astype(int)
    pdf.x = np.arange(rvd[0], rvd[-1] + 1) * pdf.BinSize
    pdf.Min = int(rvd[0])
    pdf.y = np.zeros(len(pdf.x))
    bp = rvd - rvd[0]
    pdf.y[bp[0]] = probs[0]
    for k in range(1, len(values)):
        pdf.y[bp[k]] += probs[k]
    return pdf


def _conv_fct(p1, p2):
    p = SimpleNamespace(**vars(p1))
    p.Min = int(round(p1.Min + p2.Min))
    p.y = np.convolve(np.asarray(p1.y, dtype=float), np.asarray(p2.y, dtype=float))
    pMax = p.Min + len(p.y) - 1
    p.x = np.arange(p.Min, pMax + 1) * p.BinSize
    return p


def _get_pdf_from_sampled_signal(input_vector, L, BinSize):
    iv = np.asarray(input_vector, dtype=float).ravel()
    if len(iv) == 0:
        return _d_cpdf(BinSize, 0, 1)
    if np.max(np.abs(iv)) > BinSize:
        iv = iv[np.abs(iv) > BinSize]
    else:
        return _d_cpdf(BinSize, 0, 1)
    iv[np.abs(iv) < BinSize] = 0.0
    b = np.sign(iv)
    sort_idx = np.argsort(np.abs(iv))[::-1]
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
    hi = np.asarray(max_threshold, dtype=float).ravel()
    lo = np.asarray(min_threshold, dtype=float).ravel()
    out = inp.copy()
    out = np.where(out > hi, hi, out)
    out = np.where(out < lo, lo, out)
    return out


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

    nui = round(len(residual_response) / M)

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
