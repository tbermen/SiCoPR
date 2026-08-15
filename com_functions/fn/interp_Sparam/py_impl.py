# ============================================================
# MATLAB→Python translation notes for interp_Sparam
# MATLAB lines: 7950–8165
# ============================================================
# Interpolates S-parameters Sin(fin) → Sout(fout).
# Steps:
#   1. Compute H_mag = abs(Sin); H_ph = unwrap(angle(Sin))
#   2. Anti-causal check: mean(diff(H_ph)) > 0 → warn/error
#   3. Magnitude interpolation based on opt_interp_Sparam_mag:
#      'linear_trend_to_DC': linear DC extrapolation + linear magnitude
#      'linear_trend_to_DC_log_trend_to_inf': same + log HF extrapolation
#      'trend_to_DC': log DC extrapolation (polyfit on log10 of mag)
#      'extrap_to_DC_or_zero': log interp, AC-coupled if |S(f1)| < -20 dB
#      'extrap_to_DC': log interp + hold HF constant
#      'old': linear interp
#   4. Phase interpolation based on opt_interp_Sparam_phase:
#      'old': shift phase so H_ph_i[0]=0
#      'zero_DC': set H_ph_i[0]=0
#      'interp_to_DC': add (0,0) to fin/fin and interpolate
#      'trend_and_shift_to_DC': correct LF group delay trend, shift to DC=0
#      'interp_and_shift_to_DC': same but no LF correction
#      'extrap_cubic_to_dc_linear_to_inf': pchip for LF, linear for HF
#   5. If ZERO_PAD: apply Tukey window + zero-pad above fin[-1]
#
# Returns Sout = H_mag_i * exp(1j*H_ph_i)
# ============================================================

import numpy as np


def _interp_extrap(fout, fin, y):
    """interp1(fin, y, fout, 'linear', 'extrap') — linear interp that EXTRAPOLATES
    on the end-segment slope instead of clamping (np.interp clamps)."""
    fout = np.asarray(fout, dtype=float)
    fin = np.asarray(fin, dtype=float)
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


def _Tukey_Window(f, param, fr=None, fb=None):
    """Inlined Tukey_Window (MATLAB lines 4677-4696)."""
    f = np.asarray(f, dtype=float)
    if fr is None and fb is None:
        fb = float(param.fb)
        fr = float(param.f_r) * float(param.fb)
    fperiod = 2.0 * (float(fb) - float(fr))
    return np.where(
        f < fr, 1.0,
        np.where((f >= fr) & (f <= fb),
                 0.5 * np.cos(2 * np.pi * (f - fb) / fperiod - np.pi) + 0.5,
                 0.0))[:len(f)]


def interp_Sparam(Sin, fin, fout, opt_interp_Sparam_mag, opt_interp_Sparam_phase, OP, param):
    """Interpolate S-parameters from fin to fout (MATLAB lines 7950-8165).

    Sin: complex 1D array, length = len(fin)
    fin: input frequency axis
    fout: output frequency axis
    opt_interp_Sparam_mag: magnitude interpolation method
    opt_interp_Sparam_phase: phase interpolation method
    OP: options namespace
    param: parameter namespace

    Returns Sout: complex 1D array, length = len(fout)
    """
    Sin = np.asarray(Sin, dtype=complex).ravel()
    fin = np.asarray(fin, dtype=float).ravel()
    fout = np.asarray(fout, dtype=float).ravel()

    eps = np.finfo(float).tiny        # tiny/realmin: log-domain guards below
    eps_mag = np.finfo(float).eps     # fix B02-D4 (MATLAB rev 4p15p0 line 8091): machine eps for |S| floor / HF threshold
    H_mag = np.abs(Sin)
    H_mag[H_mag < eps_mag] = eps_mag
    H_ph = np.unwrap(np.angle(Sin))

    # Anti-causal check
    if len(H_ph) > 1 and np.mean(np.diff(H_ph)) > 0:
        if getattr(OP, 'DEBUG', False):
            import warnings
            warnings.warn('Anti-causal response found. Finer frequency step is required for this channel')
        else:
            raise ValueError('Anti-causal response found. Finer frequency step is required for this channel')

    # ---- Magnitude interpolation ----
    mag_method = str(opt_interp_Sparam_mag)

    if mag_method in ('linear_trend_to_DC', 'linear_trend_to_DC_log_trend_to_inf'):
        fin_x = fin.copy()
        H_mag_x = H_mag.copy()
        if fin[0] > 0:
            n_pts = min(10, len(fin))
            p = np.polyfit(fin[:n_pts], H_mag[:n_pts], 1)
            dc_val = float(np.polyval(p, 0))
            fin_x = np.concatenate([[0.0], fin_x])
            H_mag_x = np.concatenate([[dc_val], H_mag_x])

        hf_logtrend_val = H_mag[-1]
        if fin[-1] < fout[-1]:
            # MATLAB: mid_freq_ind = round(length(fin)/2), used as a 1-BASED index
            # into fin. Python needs the 0-based equivalent, and MATLAB's round is
            # half-away-from-zero. For even len(fin) (48004 points in the 802.3dj
            # channels) floor division starts the HF trend fit one point late.
            _n = len(fin)
            mid = max(0, int(np.floor(_n / 2.0 + 0.5)) - 1)
            with np.errstate(all='ignore'):
                p2 = np.polyfit(fin[mid:], H_mag[mid:], 1)
            hf_val = float(np.polyval(p2, fout[-1]))
            if hf_val > H_mag[-1]:
                hf_val = H_mag[-1]
                hf_logtrend_val = H_mag[-1]
            elif hf_val < eps_mag:                  # fix B02-D4 (MATLAB line 8125): machine eps
                hf_val = eps_mag
                hf_logtrend_val = np.finfo(float).tiny
            fin_x = np.concatenate([fin_x, [fout[-1]]])
            H_mag_x = np.concatenate([H_mag_x, [hf_val]])

        H_mag_i = np.interp(fout, fin_x, H_mag_x)

        if mag_method == 'linear_trend_to_DC_log_trend_to_inf' and fin[-1] < fout[-1]:
            logmag_x = np.concatenate([np.log(H_mag_x[:-1]), [np.log(max(hf_logtrend_val, eps))]])
            H_logmag_i = np.exp(np.interp(fout, fin_x, logmag_x))
            idx = np.searchsorted(fout, fin[-1], side='right')
            H_mag_i[idx:] = H_logmag_i[idx:]

    elif mag_method == 'trend_to_DC':
        fin_x = fin.copy()
        H_mag_x = H_mag.copy()
        if fin[0] > 0:
            n_pts = min(10, len(fin))
            with np.errstate(divide='ignore', invalid='ignore'):
                p = np.polyfit(fin[:n_pts], np.log10(H_mag[:n_pts] + eps), 1)
            dc_val = 10.0 ** float(np.polyval(p, 0))
            fin_x = np.concatenate([[0.0], fin_x])
            H_mag_x = np.concatenate([[dc_val], H_mag_x])
        if fin[-1] < fout[-1]:
            mid = max(0, len(fin) // 2)
            with np.errstate(divide='ignore', invalid='ignore'):
                p2 = np.polyfit(fin[mid:], np.log10(H_mag[mid:] + eps), 1)
            hf_val = 10.0 ** float(np.polyval(p2, fout[-1]))
            if hf_val > H_mag[-1]:
                hf_val = H_mag[-1]
            fin_x = np.concatenate([fin_x, [fout[-1]]])
            H_mag_x = np.concatenate([H_mag_x, [hf_val]])
        with np.errstate(divide='ignore', invalid='ignore'):
            H_mag_i = 10.0 ** np.interp(fout, fin_x, np.log10(H_mag_x + eps))

    elif mag_method == 'extrap_to_DC_or_zero':
        if fin[0] > 0 and 20 * np.log10(H_mag[0] + eps) < -20:
            fin_x2 = np.concatenate([[0.0], fin])
            H_log_x = np.concatenate([[-100.0], np.log10(H_mag)])
            mask = fout <= fin[-1]
            H_mag_i = np.zeros(len(fout))
            H_mag_i[mask] = 10.0 ** np.interp(fout[mask], fin_x2, H_log_x)
        else:
            mask = fout <= fin[-1]
            H_mag_i = np.zeros(len(fout))
            with np.errstate(divide='ignore', invalid='ignore'):
                H_mag_i[mask] = 10.0 ** np.interp(fout[mask], fin, np.log10(H_mag))
        H_mag_i[fout > fin[-1]] = H_mag[-1]

    elif mag_method == 'extrap_to_DC':
        mask = fout <= fin[-1]
        H_mag_i = np.zeros(len(fout))
        with np.errstate(divide='ignore', invalid='ignore'):
            H_mag_i[mask] = 10.0 ** np.interp(fout[mask], fin, np.log10(H_mag))
        H_mag_i[fout > fin[-1]] = H_mag[-1]

    elif mag_method == 'old':
        H_mag_i = np.interp(fout, fin, H_mag)

    else:
        raise ValueError(f'interp_Sparam: invalid opt_interp_Sparam_mag = {mag_method!r}')

    # ---- Phase interpolation ----
    # fix B02-D5 (MATLAB rev 4p15p0 line 8185): interp1(fin,H_ph,fout,'linear','extrap')
    # extends the end-segment slope beyond [fin[0], fin[-1]] instead of clamping.
    H_ph_i = np.interp(fout, fin, H_ph)
    if len(fin) >= 2:
        lo = fout < fin[0]
        H_ph_i[lo] = H_ph[0] + (H_ph[1] - H_ph[0]) / (fin[1] - fin[0]) * (fout[lo] - fin[0])
        hi = fout > fin[-1]
        H_ph_i[hi] = H_ph[-1] + (H_ph[-1] - H_ph[-2]) / (fin[-1] - fin[-2]) * (fout[hi] - fin[-1])
    ph_method = str(opt_interp_Sparam_phase)

    if ph_method == 'old':
        H_ph_i = H_ph_i - H_ph_i[0]

    elif ph_method == 'zero_DC':
        H_ph_i[0] = 0.0

    elif ph_method == 'interp_to_DC':
        if fin[0] != 0:
            fin_ext = np.concatenate([[0.0], fin])
            ph_ext = np.concatenate([[0.0], H_ph])
            H_ph_i = np.interp(fout, fin_ext, ph_ext)

    elif ph_method == 'interp_and_shift_to_DC':
        if fin[0] != 0:
            dc_trend = H_ph[0] - (H_ph[1] - H_ph[0]) / (fin[1] - fin[0]) * fin[0] if len(fin) > 1 else 0.0
            fin_ext = np.concatenate([[0.0], fin])
            ph_ext = np.concatenate([[0.0], H_ph - dc_trend])
            H_ph_i = np.interp(fout, fin_ext, ph_ext)

    elif ph_method == 'trend_and_shift_to_DC':
        group_delay = -np.diff(H_ph) / np.diff(fin)
        n_lf = min(50, len(group_delay))
        lf_gd = group_delay[:n_lf]
        m = np.median(lf_gd)
        sigma = np.std(lf_gd)
        mask = np.abs(lf_gd - m) < sigma
        lf_trend = np.mean(lf_gd[mask]) if np.any(mask) else m

        H_ph_corr = H_ph.copy()
        for k in range(min(9, len(fin) - 1), -1, -1):
            H_ph_corr[k] = H_ph_corr[k + 1] + lf_trend * (fin[k + 1] - fin[k])

        if fin[0] != 0:
            dc_trend = H_ph_corr[0] + lf_trend * (fin[0] - 0)
            fin_x = np.concatenate([[0.0], fin])
            ph_x = np.concatenate([[0.0], H_ph_corr - dc_trend])
        else:
            fin_x = fin.copy()
            ph_x = H_ph_corr.copy()

        if fout[-1] > fin[-1]:
            all_gd = -np.diff(ph_x) / np.diff(fin_x)
            hf_phase_trend = ph_x[-1] - np.median(all_gd) * (fout[-1] - fin_x[-1])
            fin_x = np.concatenate([fin_x, [fout[-1]]])
            ph_x = np.concatenate([ph_x, [hf_phase_trend]])

        H_ph_i = np.interp(fout, fin_x, ph_x)

    elif ph_method == 'extrap_cubic_to_dc_linear_to_inf':
        if fin[0] != 0:
            group_delay = -np.diff(H_ph) / np.diff(fin)
            n_lf = min(50, len(group_delay))
            lf_gd = group_delay[:n_lf]
            m = np.median(lf_gd)
            sigma = np.std(lf_gd)
            mask = np.abs(lf_gd - m) < sigma
            lf_trend = np.mean(lf_gd[mask]) if np.any(mask) else m

            H_ph_corr = H_ph.copy()
            for k in range(min(9, len(fin) - 1), -1, -1):
                H_ph_corr[k] = H_ph_corr[k + 1] + lf_trend * (fin[k + 1] - fin[k])

            # Linear extrapolation
            if fout[-1] > fin[-1]:
                # MATLAB: group_delay(end-50:end) -> 51 samples, not 50
                n_hf = min(51, len(group_delay))
                hf_gd = group_delay[-n_hf:]
                m_hf = np.median(hf_gd)
                sigma_hf = np.std(hf_gd)
                mask_hf = np.abs(hf_gd - m_hf) < sigma_hf
                hf_trend_val = -np.mean(hf_gd[mask_hf]) if np.any(mask_hf) else -m_hf

                hf_ext_idx = np.where(fout > fin[-1])[0]
                if len(hf_ext_idx) > 0:
                    last_idx = hf_ext_idx[0] - 1
                    H_ph_linear_hf = (H_ph_corr[-1] +
                                      (fout[hf_ext_idx] - fin[-1]) * hf_trend_val)

            # MATLAB uses interp1(...,'linear','extrap') here; np.interp would clamp
            # below fin[0], freezing the phase across the whole DC region and shifting
            # the impulse response in time.
            H_ph_cubic = _interp_extrap(fout, fin, H_ph_corr)
            H_ph_lin = _interp_extrap(fout, fin, H_ph_corr)
            if fout[-1] > fin[-1] and len(hf_ext_idx) > 0:
                # MATLAB anchors the HF trend at fout(last_data_sample) using the
                # interpolated phase there, not at fin[-1]/H_ph_corr[-1].
                last_idx = int(hf_ext_idx[0]) - 1
                H_ph_lin[hf_ext_idx] = (H_ph_lin[last_idx] +
                                        (fout[hf_ext_idx] - fout[last_idx]) * hf_trend_val)

            diff = np.abs(H_ph_cubic - H_ph_lin)
            indx = int(np.argmin(diff))
            H_ph_i = H_ph_cubic.copy()
            H_ph_i[indx:] = H_ph_lin[indx:]
            H_ph_i = H_ph_lin  # MATLAB uses linear in final version
        # else H_ph_i stays as computed above

    else:
        raise ValueError(f'interp_Sparam: invalid opt_interp_Sparam_phase = {ph_method!r}')

    H_i = H_mag_i * np.exp(1j * H_ph_i)

    # ---- Zero padding ----
    if getattr(OP, 'ZERO_PAD', False):
        zero_pad_tukey = float(getattr(param, 'zero_pad_tukey_window_in_fb', 0))
        fb = float(getattr(param, 'fb', 1.0))
        tukey_limit = fin[-1] + zero_pad_tukey * fb
        zp_freq = fin[-1]

        if zero_pad_tukey == 0:
            H_tw = np.ones(len(fout))
        else:
            if zero_pad_tukey > 0:
                H_tw = _Tukey_Window(fout, param, fin[-1], tukey_limit)
                zp_freq = tukey_limit
            else:
                H_tw = _Tukey_Window(fout, param, tukey_limit, fin[-1])

        H_i = H_tw * H_i
        H_i[fout > zp_freq] = np.finfo(float).tiny

    return H_i
