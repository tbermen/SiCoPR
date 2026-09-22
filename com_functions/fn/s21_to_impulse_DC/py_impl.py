# ============================================================
# MATLAB→Python translation notes for s21_to_impulse_DC
# MATLAB lines: 11076–11146
# ============================================================
# Converts freq-domain IL to time-domain impulse response.
# Inline: interp_Sparam (MATLAB lines 7950-8165) as _interp_Sparam.
# All-zero IL: returns eps-valued response without calling interp_Sparam.
# Alternating projections for causality enforcement (Quatieri & Oppenheim 1981).
# IL_symmetric: real(IL[0]), IL[1:-1], real(IL[-1]), conj(IL[1:-1]) reversed.
# t_base = k / (freq_step * L).
# Truncation threshold: OP.impulse_response_truncation_threshold.
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



def _length(x):
    """MATLAB length(): the longest dimension, 0 when empty, 1 for a scalar."""
    if x.size == 0:
        return 0
    return max(x.shape) if x.ndim else 1


def _Tukey_Window(f, param, fr=None, fb=None):
    f = np.asarray(f, dtype=float)
    if fr is None and fb is None:
        fb = float(param.fb)
        fr = float(param.f_r) * float(param.fb)
    fperiod = 2.0 * (float(fb) - float(fr))
    # MATLAB CONCATENATES three counted pieces — ones(1,n<fr), the raised
    # cosine of the in-band values, zeros(1,n>fb) — so the answer is grouped by
    # category and only lines up with f when f ascends.  Element-wise np.where
    # silently returned a different vector for any other order.  COM Octave,
    # fr=1e9 fb=3e9: f=[1e9 1e9 3e9 3e9 0 9e9] -> [1 1 1 0 0 0]
    # (element-wise gave [1 1 0 0 1 0]).
    flat = np.atleast_1d(f).ravel(order='F')   # MATLAB linear-index order
    n_lo = int(np.count_nonzero(flat < fr))
    n_hi = int(np.count_nonzero(flat > fb))
    band = flat[(flat >= fr) & (flat <= fb)]
    mid = 0.5 * np.cos(2 * np.pi * (band - fb) / fperiod - np.pi) + 0.5
    # Only the middle piece keeps the orientation of f, so for a column or a
    # matrix MATLAB's horizontal concatenation fails unless that piece has at
    # most one element or is the only non-empty one.  COM Octave, column f:
    # "horizontal dimensions mismatch (1x2 vs 5x1)".
    if f.ndim > 1 and f.shape[0] != 1 and mid.size > 1 and (n_lo or n_hi):
        raise ValueError('Tukey_Window: horizontal dimensions mismatch '
                         '(1x%d vs %dx1)' % (n_lo or n_hi, mid.size))
    H_tw = np.concatenate([np.ones(n_lo), mid, np.zeros(n_hi)])
    n = _length(f)
    # The pieces cover every element of f only while each one lands in exactly
    # one category.  A NaN lands in none, so H_tw comes up short and MATLAB's
    # H_tw(1:length(f)) is an out-of-bound read.  COM Octave, f=[0 1.5e9 NaN
    # 3.5e9]: "H_tw(4): out of bound 3".  np.where answered 0 for the NaN.
    if H_tw.size < n:
        raise IndexError('Tukey_Window: H_tw(%d): out of bound %d' % (n, H_tw.size))
    return H_tw[:n]


def _interp_extrap(fout, fin, y):
    """interp1(fin, y, fout, 'linear', 'extrap') — extrapolates on the end-segment
    slope rather than clamping the way np.interp does."""
    fout = np.asarray(fout, dtype=float); fin = np.asarray(fin, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.interp(fout, fin, y)
    if len(fin) >= 2:
        lo = fout < fin[0]
        if np.any(lo):
            out[lo] = y[0] + (y[1] - y[0]) / (fin[1] - fin[0]) * (fout[lo] - fin[0])
        hi = fout > fin[-1]
        if np.any(hi):
            out[hi] = y[-1] + (y[-1] - y[-2]) / (fin[-1] - fin[-2]) * (fout[hi] - fin[-1])
    return out


def _interp_Sparam(Sin, fin, fout, opt_mag, opt_phase, OP, param):
    """Inlined interp_Sparam (MATLAB lines 7950-8165)."""
    Sin = np.asarray(Sin, dtype=complex).ravel()
    fin = np.asarray(fin, dtype=float).ravel()
    fout = np.asarray(fout, dtype=float).ravel()
    eps_val = np.finfo(float).tiny        # tiny/realmin: log-domain guards below
    eps_mag = np.finfo(float).eps         # fix B02-D4 (MATLAB rev 4p15p0 line 8091): machine eps for |S| floor / HF threshold

    H_mag = np.abs(Sin)
    H_mag[H_mag < eps_mag] = eps_mag
    H_ph = np.unwrap(np.angle(Sin))

    if len(H_ph) > 1 and np.mean(np.diff(H_ph)) > 0:
        if getattr(OP, 'DEBUG', False):
            import warnings
            warnings.warn('Anti-causal response found. Finer frequency step is required')
        else:
            raise ValueError('Anti-causal response found. Finer frequency step is required for this channel')

    mag_method = str(opt_mag)
    if mag_method in ('linear_trend_to_DC', 'linear_trend_to_DC_log_trend_to_inf'):
        fin_x = fin.copy(); H_mag_x = H_mag.copy()
        if fin[0] > 0:
            n = min(10, len(fin))
            p = np.polyfit(fin[:n], H_mag[:n], 1)
            fin_x = np.concatenate([[0.0], fin_x])
            H_mag_x = np.concatenate([[float(np.polyval(p, 0))], H_mag_x])
        hf_log = H_mag[-1]
        if fin[-1] < fout[-1]:
            # MATLAB mid_freq_ind = round(length(fin)/2) is a 1-BASED index;
            # floor division as 0-based starts one point late for even len(fin).
            mid = max(0, int(np.floor(len(fin) / 2.0 + 0.5)) - 1)
            with np.errstate(all='ignore'):
                p2 = np.polyfit(fin[mid:], H_mag[mid:], 1)
            hf = float(np.polyval(p2, fout[-1]))
            if hf > H_mag[-1]: hf = H_mag[-1]; hf_log = H_mag[-1]
            elif hf < eps_mag: hf = eps_mag; hf_log = np.finfo(float).tiny  # fix B02-D4 (MATLAB line 8125)
            fin_x = np.concatenate([fin_x, [fout[-1]]])
            H_mag_x = np.concatenate([H_mag_x, [hf]])
        H_mag_i = np.interp(fout, fin_x, H_mag_x)
        if mag_method == 'linear_trend_to_DC_log_trend_to_inf' and fin[-1] < fout[-1]:
            log_x = np.concatenate([np.log(H_mag_x[:-1]), [np.log(max(hf_log, eps_val))]])
            H_lmi = np.exp(np.interp(fout, fin_x, log_x))
            idx = np.searchsorted(fout, fin[-1], side='right')
            H_mag_i[idx:] = H_lmi[idx:]

    elif mag_method == 'trend_to_DC':
        fin_x = fin.copy(); H_mag_x = H_mag.copy()
        if fin[0] > 0:
            n = min(10, len(fin))
            with np.errstate(divide='ignore', invalid='ignore'):
                p = np.polyfit(fin[:n], np.log10(H_mag[:n] + eps_val), 1)
            fin_x = np.concatenate([[0.0], fin_x])
            H_mag_x = np.concatenate([[10.0 ** float(np.polyval(p, 0))], H_mag_x])
        if fin[-1] < fout[-1]:
            # MATLAB mid_freq_ind = round(length(fin)/2) is a 1-BASED index;
            # floor division as 0-based starts one point late for even len(fin).
            mid = max(0, int(np.floor(len(fin) / 2.0 + 0.5)) - 1)
            with np.errstate(divide='ignore', invalid='ignore'):
                p2 = np.polyfit(fin[mid:], np.log10(H_mag[mid:] + eps_val), 1)
            hf = 10.0 ** float(np.polyval(p2, fout[-1]))
            if hf > H_mag[-1]: hf = H_mag[-1]
            fin_x = np.concatenate([fin_x, [fout[-1]]])
            H_mag_x = np.concatenate([H_mag_x, [hf]])
        with np.errstate(divide='ignore', invalid='ignore'):
            H_mag_i = 10.0 ** np.interp(fout, fin_x, np.log10(H_mag_x + eps_val))

    elif mag_method == 'extrap_to_DC_or_zero':
        if fin[0] > 0 and 20 * np.log10(H_mag[0] + eps_val) < -20:
            fin_x2 = np.concatenate([[0.0], fin])
            H_log_x = np.concatenate([[-100.0], np.log10(H_mag)])
            mask = fout <= fin[-1]
            H_mag_i = np.zeros(len(fout))
            H_mag_i[mask] = 10.0 ** np.interp(fout[mask], fin_x2, H_log_x)
        else:
            mask = fout <= fin[-1]; H_mag_i = np.zeros(len(fout))
            with np.errstate(divide='ignore', invalid='ignore'):
                H_mag_i[mask] = 10.0 ** np.interp(fout[mask], fin, np.log10(H_mag))
        H_mag_i[fout > fin[-1]] = H_mag[-1]

    elif mag_method == 'extrap_to_DC':
        mask = fout <= fin[-1]; H_mag_i = np.zeros(len(fout))
        with np.errstate(divide='ignore', invalid='ignore'):
            H_mag_i[mask] = 10.0 ** np.interp(fout[mask], fin, np.log10(H_mag))
        H_mag_i[fout > fin[-1]] = H_mag[-1]

    elif mag_method in ('old', 'pchip'):  # treat 'pchip' as linear fallback
        H_mag_i = np.interp(fout, fin, H_mag)

    else:
        raise ValueError(f'interp_Sparam: invalid opt_interp_Sparam_mag = {mag_method!r}')

    # fix B02-D5 (MATLAB rev 4p15p0 line 8185): interp1(fin,H_ph,fout,'linear','extrap')
    # extends the end-segment slope beyond [fin[0], fin[-1]] instead of clamping.
    H_ph_i = np.interp(fout, fin, H_ph)
    if len(fin) >= 2:
        lo = fout < fin[0]
        H_ph_i[lo] = H_ph[0] + (H_ph[1] - H_ph[0]) / (fin[1] - fin[0]) * (fout[lo] - fin[0])
        hi = fout > fin[-1]
        H_ph_i[hi] = H_ph[-1] + (H_ph[-1] - H_ph[-2]) / (fin[-1] - fin[-2]) * (fout[hi] - fin[-1])
    ph_method = str(opt_phase)

    if ph_method in ('old', 'pchip'):
        H_ph_i = H_ph_i - H_ph_i[0]
    elif ph_method == 'zero_DC':
        H_ph_i[0] = 0.0
    elif ph_method == 'interp_to_DC':
        if fin[0] != 0:
            H_ph_i = np.interp(fout, np.concatenate([[0.0], fin]), np.concatenate([[0.0], H_ph]))
    elif ph_method == 'interp_and_shift_to_DC':
        if fin[0] != 0:
            dc_t = H_ph[0] - (H_ph[1] - H_ph[0]) / (fin[1] - fin[0]) * fin[0] if len(fin) > 1 else 0.0
            H_ph_i = np.interp(fout, np.concatenate([[0.0], fin]),
                                np.concatenate([[0.0], H_ph - dc_t]))
    elif ph_method == 'trend_and_shift_to_DC':
        gd = -np.diff(H_ph) / np.diff(fin)
        n_lf = min(50, len(gd))
        lf = gd[:n_lf]; m = np.median(lf); s = np.std(lf, ddof=1)     # MATLAB/Octave std normalises by N-1; numpy defaults to N
        mask_g = np.abs(lf - m) < s
        lf_t = np.mean(lf[mask_g]) if np.any(mask_g) else m
        H_ph_c = H_ph.copy()
        for k in range(min(9, len(fin) - 1), -1, -1):
            H_ph_c[k] = H_ph_c[k + 1] + lf_t * (fin[k + 1] - fin[k])
        if fin[0] != 0:
            dc_t = H_ph_c[0] + lf_t * (fin[0] - 0)
            fin_x = np.concatenate([[0.0], fin]); ph_x = np.concatenate([[0.0], H_ph_c - dc_t])
        else:
            fin_x = fin.copy(); ph_x = H_ph_c.copy()
        if fout[-1] > fin[-1]:
            all_gd = -np.diff(ph_x) / np.diff(fin_x)
            hf_pt = ph_x[-1] - np.median(all_gd) * (fout[-1] - fin_x[-1])
            fin_x = np.concatenate([fin_x, [fout[-1]]]); ph_x = np.concatenate([ph_x, [hf_pt]])
        H_ph_i = np.interp(fout, fin_x, ph_x)
    elif ph_method == 'extrap_cubic_to_dc_linear_to_inf':
        # Full port of MATLAB L8198-8228. This was previously a stub
        # (`np.interp(fout, fin, H_ph)  # simplified: use linear`) that skipped the
        # low-frequency group-delay outlier correction, used the raw phase, and
        # clamped instead of extrapolating. Since this is the CONFIGURED DEFAULT
        # phase method, the LF group delay was wrong on every run, shifting and
        # reshaping the impulse response (peak time off by 53 samples, pulse peak
        # 0.94% high, steady-state 0.46% low).
        # Note MATLAB computes a pchip variant and a blend, then discards both with
        # a final `H_ph_i = H_ph_linear;` — so only the linear branch matters.
        if fin[0] != 0:
            group_delay = -np.diff(H_ph) / np.diff(fin)
            n_lf = min(50, len(group_delay))
            lf_gd = group_delay[:n_lf]
            m_lf = np.median(lf_gd)
            sd_lf = np.std(lf_gd, ddof=1)     # MATLAB/Octave std normalises by N-1; numpy defaults to N
            mask_lf = np.abs(lf_gd - m_lf) < sd_lf
            lf_trend = float(np.mean(lf_gd[mask_lf])) if np.any(mask_lf) else float(m_lf)

            # MATLAB: for k=10:-1:1, H_ph(k) = H_ph(k+1) + lf_trend*(fin(k+1)-fin(k))
            H_ph_corr = H_ph.copy()
            for k in range(min(9, len(fin) - 2), -1, -1):
                H_ph_corr[k] = H_ph_corr[k + 1] + lf_trend * (fin[k + 1] - fin[k])

            H_ph_lin = _interp_extrap(fout, fin, H_ph_corr)

            if fout[-1] > fin[-1]:
                n_hf = min(51, len(group_delay))       # MATLAB group_delay(end-50:end)
                hf_gd = group_delay[-n_hf:]
                m_hf = np.median(hf_gd)
                sd_hf = np.std(hf_gd, ddof=1)     # MATLAB/Octave std normalises by N-1; numpy defaults to N
                mask_hf = np.abs(hf_gd - m_hf) < sd_hf
                hf_trend = -float(np.mean(hf_gd[mask_hf])) if np.any(mask_hf) else -float(m_hf)
                hf_ext = np.where(fout > fin[-1])[0]
                if len(hf_ext) > 0:
                    last_idx = int(hf_ext[0]) - 1
                    H_ph_lin[hf_ext] = (H_ph_lin[last_idx] +
                                        (fout[hf_ext] - fout[last_idx]) * hf_trend)
            H_ph_i = H_ph_lin
    else:
        raise ValueError(f'interp_Sparam: invalid opt_interp_Sparam_phase = {ph_method!r}')

    H_i = H_mag_i * np.exp(1j * H_ph_i)

    if getattr(OP, 'ZERO_PAD', False):
        zpf = float(getattr(param, 'zero_pad_tukey_window_in_fb', 0))
        fb = float(getattr(param, 'fb', 1.0))
        tl = fin[-1] + zpf * fb; zp = fin[-1]
        if zpf == 0:
            H_tw = np.ones(len(fout))
        elif zpf > 0:
            H_tw = _Tukey_Window(fout, param, fin[-1], tl); zp = tl
        else:
            H_tw = _Tukey_Window(fout, param, tl, fin[-1])
        H_i = H_tw * H_i; H_i[fout > zp] = np.finfo(float).tiny
    return H_i


def _mround(x):
    """MATLAB round(): half away from zero (fix B03-D7, MATLAB rev 4p15p0 line 11232)."""
    x = float(x)
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    # Off a tie round() is exact, and unlike floor(x + 0.5) it does not
    # send 0.49999999999999994 to 1: that sum is exactly 1.0 in binary.
    return int(round(x))


def s21_to_impulse_DC(IL, freq_array, time_step, OP, param):
    """Frequency-domain IL → time-domain impulse response (MATLAB lines 11076-11146).

    IL: complex 1D array
    freq_array: frequency axis in Hz (must have at least 3 points)
    time_step: sample_dt
    OP: options with interp_sparam_mag, interp_sparam_phase, EC_PULSE_TOL, EC_REL_TOL,
        EC_DIFF_TOL, ENFORCE_CAUSALITY, impulse_response_truncation_threshold
    param: parameter namespace

    Returns (voltage, t_base, causality_correction_dB, truncation_dB).
    """
    freq_array = np.asarray(freq_array, dtype=float)
    IL = np.asarray(IL, dtype=complex)

    fmax = 1.0 / time_step / 2.0
    freq_step = (freq_array[2] - freq_array[1])
    # fix B03-D7 (MATLAB rev 4p15p0 line 11232): fout=0:1/round(fmax/freq_step)*fmax:fmax
    # uses MATLAB round (half-away), not Python builtin round (banker's).
    n_steps = max(1, _mround(fmax / freq_step))
    fout = np.arange(0, n_steps + 1) * (fmax / n_steps)

    if np.all(IL == 0):
        IL_interp = np.full(len(fout), np.finfo(float).eps, dtype=complex)
    else:
        IL_interp = _interp_Sparam(
            IL, freq_array, fout,
            getattr(OP, 'interp_sparam_mag', 'extrap_to_DC'),
            getattr(OP, 'interp_sparam_phase', 'interp_to_DC'),
            OP, param)
        # Replace NaN with previous value
        for idx in np.where(np.isnan(IL_interp))[0]:
            IL_interp[idx] = IL_interp[idx - 1] if idx > 0 else np.finfo(float).eps

    IL_col = IL_interp.ravel()
    IL_symmetric = np.concatenate([
        [np.real(IL_col[0])],
        IL_col[1:-1],
        [np.real(IL_col[-1])],
        np.conj(IL_col[1:-1])[::-1]
    ])
    impulse_response = np.real(np.fft.ifft(IL_symmetric))
    L = len(impulse_response)
    t_base = np.arange(L) / (freq_step * L)

    original_impulse_response = impulse_response.copy()

    abs_ir = np.abs(impulse_response)
    half = L // 2
    candidates = np.where(abs_ir[:half] > _mmax(abs_ir[:half]) * OP.EC_PULSE_TOL)[0]
    start_ind = int(candidates[0]) if len(candidates) > 0 else 0

    err = np.inf
    while not np.all(impulse_response == 0):
        # fix B03-D6 (MATLAB rev 4p15p0 lines 11260-11261): impulse_response(1:start_ind)=0
        # is 1-based INCLUSIVE, so 0-based candidates[0] must be zeroed too -> [:start_ind+1].
        impulse_response[:start_ind + 1] = 0
        # fix B03-D6 (MATLAB line 11261): impulse_response(floor(L/2):end)=0 -> 0-based [half-1:].
        impulse_response[half - 1:] = 0
        IL_modified = np.abs(IL_symmetric) * np.exp(1j * np.angle(np.fft.fft(impulse_response)))
        ir_modified = np.real(np.fft.ifft(IL_modified))
        delta = np.abs(impulse_response - ir_modified)
        err_prev = err
        # fix B03-D6 (MATLAB line 11267): err = max(delta)/max(impulse_response) uses the
        # SIGNED max, not max(abs(.)).
        peak = _mmax(impulse_response)
        err = _mmax(delta) / peak if peak != 0 else 0.0
        if err < OP.EC_REL_TOL or abs(err_prev - err) < OP.EC_DIFF_TOL:
            break
        impulse_response = ir_modified

    ir_norm = np.linalg.norm(impulse_response)
    causality_correction_dB = (20 * np.log10(
        np.linalg.norm(impulse_response - original_impulse_response) / ir_norm)
        if ir_norm > 0 else 0.0)

    if not OP.ENFORCE_CAUSALITY:
        impulse_response = original_impulse_response

    ir_peak = _mmax(np.abs(impulse_response))
    last_arr = np.where(np.abs(impulse_response) > ir_peak * OP.impulse_response_truncation_threshold)[0]
    ir_last = int(last_arr[-1]) if len(last_arr) > 0 else L - 1

    voltage = impulse_response[:ir_last + 1]
    t_base_out = t_base[:ir_last + 1]

    tail_norm = np.linalg.norm(impulse_response[ir_last + 1:])
    v_norm = np.linalg.norm(voltage)
    truncation_dB = 20 * np.log10(tail_norm / v_norm) if v_norm > 0 and tail_norm > 0 else -np.inf

    return voltage, t_base_out, causality_correction_dB, truncation_dB
