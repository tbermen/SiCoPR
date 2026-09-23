# ============================================================
# MATLAB→Python translation notes for get_pdf_full
# MATLAB lines: 7520–7663
# ============================================================
# Same as get_pdf but first upsamples pulse response to param.samples_for_C2M.
# old_time = [0..N-1]/samples_per_ui; shifted so t_s is at 0.
# new_time: built so that time=0 is always present; step=1/samp_UI.
# SBR = interp1(old_time, pulse_orig, new_time) (np.interp).
# t_s = argmin(abs(new_time)) after resampling (0-based).
# THRU path: bmax_vec/bmin_vec scaled by residual_response[t_s] (cursor amplitude).
# Floating_DFE: use N_bmax postcursors; otherwise ndfe.
# kron(effective_cancelled_cursors, ones(samp_UI)) → np.repeat.
# effective_cancellation_samples = np.repeat(..., samp_UI) for each postcursor.
# start_cancel = t_s - half_UI + 1 + samp_UI (0-based: t_s - half_UI + samp_UI).
# uiv_start = start_cancel - samp_UI (0-based).
# residual_response[uiv_start:uiv_end+1] = 0.
# vs: reshape residual_response[samp_UI:samp_UI*(nui-1)] → (nui-2, samp_UI).
# phases (MATLAB 1-based): mod(t_s+1, samp_UI); if 0→samp_UI → in Python: (t_s+1)%samp_UI if 0→samp_UI.
# Actually in Python t_s is 0-based. MATLAB t_s (1-based). MATLAB phases = mod(t_s,samp_UI); if 0→samp_UI.
# Python: t_s_matlab = t_s+1; phases_matlab = t_s_matlab % samp_UI; if 0→samp_UI. Then 0-based: phases_matlab-1.
# shift_amount = half_UI - phases (MATLAB); circshift with that amount.
# h_j_full: (L, samp_UI) matrix where L = vs_raw.shape[0].
# ============================================================

import math

import numpy as np
from com_functions.fn.dfe_clipper.py_impl import dfe_clipper as _dfe_clipper
from com_functions.fn.Init_PDF_Fast.py_impl import Init_PDF_Fast as _Init_PDF_Fast
from com_functions.fn.conv_fct.py_impl import conv_fct as _conv_fct
from com_functions.fn.d_cpdf.py_impl import d_cpdf as _d_cpdf

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

_EPS = np.finfo(float).eps


def _colon(step, limit):
    """MATLAB `0:step:limit`, for a positive step.

    Verbatim from get_StepR/get_PulseR, where it was verified against COM
    Octave. A colon stops at or BEFORE the limit: `arange(0, limit+step, step)`
    is one element too long when limit/step is not an integer, and
    `floor(limit/step)+1` is one too short when the quotient lands a fraction
    of an eps below an integer.
    """
    n = int(round(limit / step + 1.0))
    if n > 0 and (n - 1) * step > limit + 3.0 * _EPS * abs(limit):
        n -= 1
    out = np.arange(max(n, 0)) * step
    if out.size:
        out[0] = 0.0
    if out.size > 1 and out[-1] > limit:
        out[-1] = limit
    return out


def _get_center_of_UI(samp_UI):
    M = int(samp_UI)
    return M // 2 + 1  # 1-based MATLAB half_UI


def _get_pdf_from_sampled_signal(input_vector, L, BinSize):
    iv = np.asarray(input_vector, dtype=float).ravel()
    if len(iv) == 0:
        return _d_cpdf(BinSize, 0.0, 1.0)
    mask = np.abs(iv) > BinSize
    if not np.any(mask):
        return _d_cpdf(BinSize, 0.0, 1.0)
    iv = iv[mask]
    iv[np.abs(iv) < BinSize] = 0.0
    b = np.sign(iv)
    sort_idx = np.argsort(np.abs(iv), kind='stable')[::-1]
    iv = np.abs(iv[sort_idx]) * b[sort_idx]
    values = 2.0 * np.arange(L) / (L - 1) - 1.0
    prob = np.ones(L) / L
    empty_pdf = _d_cpdf(BinSize, 0.0, 1.0)
    pdf = empty_pdf
    for v in iv:
        pdfn = _Init_PDF_Fast(empty_pdf, np.abs(v) * values, prob)
        pdf = _conv_fct(pdf, pdfn)
    return pdf


def get_pdf_full(chdata, delta_y, t_s, param, OP, pdf_range=None):
    """Upsampled-pulse PDF computation (MATLAB lines 7520-7663).

    Resamples chdata.eq_pulse_response to param.samples_for_C2M then calls
    get_pdf_from_sampled_signal on each phase.

    Returns (pdf_list, h_j_full, A_s_vec).
    pdf_list: list of SimpleNamespace per sample position (0-based indexed by samp_UI position).
    """
    t_s_orig = int(t_s)
    pulse_type = str(chdata.type)
    pulse_orig = np.asarray(chdata.eq_pulse_response, dtype=float).ravel()
    M_orig = int(param.samples_per_ui)
    samp_UI = int(param.samples_for_C2M)

    # Time axes (0-based)
    old_time = np.arange(len(pulse_orig)) / M_orig
    original_sample_time = old_time[t_s_orig]
    old_time = old_time - original_sample_time

    # New time axis forcing 0 in axis. ML 7990-7992:
    #     new_timea=[0:-1/samp_UI:min(old_time)];
    #     new_timeb=[0:1/samp_UI:max(old_time)];
    #     new_time =[fliplr(new_timea) new_timeb(2:end)];
    #
    # D12. The port used arange(..., floor(x*samp_UI) + 2), which is one point
    # LONGER than the colon on each side, and that extra point lies OUTSIDE
    # [min(old_time), max(old_time)]: a MATLAB colon stops at or before its
    # limit and never steps past it. Two extra samples shift the cursor
    # t_s = argmin(|new_time|) by one, which moves the sampling phase and the
    # centring circshift, and the port disagreed with the MATLAB-faithful
    # oracle in 1 of 32 phase columns.
    #
    # _colon is the helper verified against COM Octave for get_StepR and
    # get_PulseR, which are the same reference construct. It takes a POSITIVE
    # step, and the two colons here are symmetric about zero, so the negative
    # side is built on the magnitude and negated.
    new_timea = -_colon(1.0 / samp_UI, abs(min(old_time)))[::-1]
    new_timeb = _colon(1.0 / samp_UI, max(old_time))
    new_time = np.concatenate([new_timea, new_timeb[1:]])

    SBR = np.interp(new_time, old_time, pulse_orig)

    # New cursor: argmin(abs(new_time))
    t_s = int(np.argmin(np.abs(new_time)))

    residual_response = SBR.copy()

    half_UI = _get_center_of_UI(samp_UI)  # 1-based MATLAB

    A_s_vec = None

    if pulse_type == 'THRU':
        if not getattr(param, 'Floating_DFE', False):
            ndfe = int(param.ndfe)
            post_indices = t_s + samp_UI * np.arange(1, ndfe + 1)
            bmax_vec = residual_response[t_s] * np.asarray(param.bmax, dtype=float).ravel()[:ndfe]
            bmin_vec = residual_response[t_s] * np.asarray(param.bmin, dtype=float).ravel()[:ndfe]
        else:
            ndfe = int(param.N_bmax)
            post_indices = t_s + samp_UI * np.arange(1, ndfe + 1)
            bmax_vec = residual_response[t_s] * np.asarray(param.use_bmax, dtype=float).ravel()[:ndfe]
            bmin_vec = residual_response[t_s] * np.asarray(param.use_bmin, dtype=float).ravel()[:ndfe]

        # Clip to valid range
        valid_mask = (post_indices >= 0) & (post_indices < len(SBR))
        post_indices = post_indices[valid_mask].astype(int)
        bmax_vec = bmax_vec[:len(post_indices)]
        bmin_vec = bmin_vec[:len(post_indices)]

        ideal_cancelled_cursors = SBR[post_indices]

        dfe_delta = float(getattr(param, 'dfe_delta', 0))
        if dfe_delta != 0:
            cursor_amp = residual_response[t_s]
            ideal_cancelled_cursors_q = (
                np.floor(np.abs(ideal_cancelled_cursors / (cursor_amp * dfe_delta)))
                * cursor_amp * dfe_delta * np.sign(ideal_cancelled_cursors)
            )
        else:
            ideal_cancelled_cursors_q = ideal_cancelled_cursors

        effective_cancelled_cursors = _dfe_clipper(ideal_cancelled_cursors_q, bmax_vec, bmin_vec)
        effective_cancellation_samples = np.repeat(effective_cancelled_cursors, samp_UI)

        # MATLAB L8035: start_cancel = t_s - half_UI + 1 + samp_UI, with t_s
        # and half_UI both 1-based. Here t_s is the 0-based argmin of the
        # resampled time axis and half_UI is still MATLAB's 1-based centre,
        # so substituting t_s_ml = t_s + 1 and taking one off for 0-based
        # leaves exactly this expression. Subtracting a further 1 -- as this
        # did until 2026-09-22 -- put the cancellation window and A_s_vec one
        # sample early.
        start_cancel = t_s - half_UI + 1 + samp_UI
        n_cancel = len(post_indices) * samp_UI
        end_cancel = start_cancel + n_cancel  # exclusive

        if end_cancel <= len(residual_response):
            residual_response[start_cancel:end_cancel] -= effective_cancellation_samples[:n_cancel]

        uiv_start = start_cancel - samp_UI
        uiv_end = uiv_start + samp_UI  # exclusive
        uiv_end = min(uiv_end, len(residual_response))
        R_LM = float(param.R_LM)
        L_lvl = int(param.levels)
        A_s_vec = R_LM * SBR[uiv_start:uiv_end] / (L_lvl - 1)
        residual_response[uiv_start:uiv_end] = 0.0

    nui = _mround(len(residual_response) / samp_UI)
    block_start = samp_UI  # 1-based → 0-based: samp_UI (same as MATLAB index samp_UI+1 → 0-based samp_UI)
    block_end = samp_UI * (nui - 1)

    if block_end > len(residual_response) or block_start >= block_end:
        block_end = len(residual_response)
        n_rows = (block_end - block_start) // samp_UI
        block_end = block_start + n_rows * samp_UI
    n_rows = (block_end - block_start) // samp_UI

    vs = residual_response[block_start:block_end].reshape(n_rows, samp_UI)
    vs_raw = SBR[block_start:block_end].reshape(n_rows, samp_UI)

    L_r = vs_raw.shape[0]

    # Phases (1-based MATLAB convention)
    if pulse_type == 'THRU':
        t_s_matlab = t_s + 1
        phase_matlab = t_s_matlab % samp_UI
        if phase_matlab == 0:
            phase_matlab = samp_UI
        phases = [phase_matlab]  # 1-based, single phase
    else:
        phases = list(range(1, samp_UI + 1))

    # pdf_range: 1-based MATLAB indexing
    if pdf_range is None or len(pdf_range) == 0:
        pdf_range_set = set(range(1, samp_UI + 1))
    else:
        pdf_range_arr = np.asarray(pdf_range, dtype=int).ravel()
        pdf_range_set = set(range(int(min(pdf_range_arr)), int(max(pdf_range_arr)) + 1))

    shift_amount = half_UI - phases[0]  # only used for single phase (THRU)

    # circshift of vs by shift_amount along columns (axis=1)
    vs_shift = np.roll(vs, shift_amount, axis=1)

    h_j_full = np.zeros((L_r, samp_UI))
    pdf_list = [None] * samp_UI  # 0-based

    for k_matlab in range(1, samp_UI + 1):
        if k_matlab not in pdf_range_set:
            continue
        k0 = k_matlab - 1  # 0-based
        samples = vs_shift[:, k0]
        pdf_list[k0] = _get_pdf_from_sampled_signal(samples, int(param.levels), float(delta_y))

        # Compute h_j_full column
        hk_matlab = k_matlab - shift_amount  # 1-based
        if hk_matlab < 1:
            hk_matlab += samp_UI
        elif hk_matlab > samp_UI:
            hk_matlab -= samp_UI
        hk0 = hk_matlab - 1  # 0-based

        if hk_matlab == 1:
            # early UI is last column (index samp_UI-1=0-based)
            h_j_full[:L_r - 1, k0] = (vs_raw[1:, hk0 + 1] - vs_raw[:L_r - 1, samp_UI - 1]) / 2 * samp_UI
        elif hk_matlab == samp_UI:
            # late UI is first column (0-based: 0)
            h_j_full[:L_r - 1, k0] = (vs_raw[1:, 0] - vs_raw[:L_r - 1, hk0 - 1]) / 2 * samp_UI
        else:
            h_j_full[:L_r, k0] = (vs_raw[:, hk0 + 1] - vs_raw[:, hk0 - 1]) / 2 * samp_UI

    return pdf_list, h_j_full, A_s_vec
