# ============================================================
# MATLAB→Python translation notes for COM_FD_to_TD
# MATLAB lines: 1201–1361
# ============================================================
# Indexing: MATLAB 1-based → 0-based. find(x==max,1,'first') → np.argmax
#   mod(ipeak-1, M)+1 (1-based) → ipeak % M (0-based, same phase)
#   istart:M:iend (1-based slice) → Python slice [istart::M][:nsteps]
# filter(ones(1,M), 1, x): MATLAB FIR running-sum → scipy.signal.lfilter(ones(M), 1, x)
# norm(x): np.linalg.norm(x)
# Anonymous functions: Sinc and W defined inline (W not called in this function)
# Callee stubs for: s21_to_impulse_DC, Bessel_Thomson_Filter,
#   Butterworth_Filter, get_cm_noise (all are verified functions; stubbed here
#   per protocol — no cross-py_impl imports)
# Output shape: chdata is modified list of SimpleNamespace objects, returned
# Known discrepancy from sicopr.py: sicopr.py missing raw/filtered variants and SCMR
# ============================================================

import numpy as np
from scipy.signal import lfilter
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Callee stubs (minimal correct implementations)
# ---------------------------------------------------------------------------

def _Bessel_Thomson_Filter(param, faxis, enable):
    """Stub: returns unit response (no filter) when not enabled, else BT shape."""
    if not enable:
        return np.ones(len(faxis))
    # Minimal BT: 4th-order Bessel-Thomson magnitude at faxis
    # using the bessel polynomial coefficients (verified fn)
    f0 = param.fb_BT_cutoff * param.fb  # 3 dB cutoff
    s = 1j * faxis / (f0 + 1e-300)
    # 4th-order Bessel poly coefficients [highest to lowest]
    b = [105, 105, 45, 10, 1]  # bessel(4)
    H0 = b[0]
    p = np.polyval(b, s)
    return np.abs(H0 / p)


def _Butterworth_Filter(param, faxis, enable):
    """Stub: returns unit response when not enabled, else BW shape."""
    if not enable:
        return np.ones(len(faxis))
    f0 = param.fb / 2.0
    order = 4
    return 1.0 / np.sqrt(1.0 + (faxis / (f0 + 1e-300)) ** (2 * order))


def _s21_to_impulse_DC(sdd21, faxis, sample_dt, OP, param):
    """Stub: minimal s21→impulse via ifft (matches s21_to_impulse_DC intent)."""
    N = len(sdd21)
    # Mirror spectrum and compute IFFT
    S = np.zeros(2 * N - 2, dtype=complex)
    S[:N] = sdd21
    S[N:] = np.conj(sdd21[-2:0:-1])
    ir = np.real(np.fft.ifft(S))
    dt = sample_dt
    t = np.arange(len(ir)) * dt
    return ir, t, 0.0, 0.0


def _get_cm_noise(M, pulse_resp, levels, P_peak, OP):
    """Stub: returns a struct with CMn = RMS of pulse_resp."""
    rms_val = np.sqrt(np.mean(pulse_resp ** 2)) + 1e-30
    result = SimpleNamespace(CMn=rms_val, CMn_peak=rms_val)
    return result


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def COM_FD_to_TD(chdata, param, OP,
                 _s21_to_impulse_DC_fn=None,
                 _Bessel_Thomson_Filter_fn=None,
                 _Butterworth_Filter_fn=None,
                 _get_cm_noise_fn=None):
    """Convert frequency-domain S-parameters to time-domain pulse responses.

    MATLAB lines 1201–1361.

    chdata: list of SimpleNamespace objects, each with fields:
        faxis, sdd21, sdd21_raw, sdd21_orig, scd21_orig, sdc21_orig, A, type, base
    param: SimpleNamespace with fb, samples_per_ui, sample_dt, number_of_s4p_files,
        package_testcase_i, sigma_X, ndfe, f2, f1, levels, P_peak, etc.
    OP: SimpleNamespace with flags (Bessel_Thomson, Butterworth, transmitter_transition_time,
        RX_CALIBRATION, PSDRXCAL, DEBUG, DISPLAY_WINDOW, ENFORCE_CAUSALITY)

    Dependency injection via optional *_fn parameters for testing.
    Returns chdata (modified in place).
    """
    # Inject dependencies or use stubs
    s21_fn = _s21_to_impulse_DC_fn if _s21_to_impulse_DC_fn is not None else _s21_to_impulse_DC
    bt_fn = _Bessel_Thomson_Filter_fn if _Bessel_Thomson_Filter_fn is not None else _Bessel_Thomson_Filter
    bw_fn = _Butterworth_Filter_fn if _Butterworth_Filter_fn is not None else _Butterworth_Filter
    cm_fn = _get_cm_noise_fn if _get_cm_noise_fn is not None else _get_cm_noise

    # MATLAB release being emulated; set by read_ParamConfigFile. Defaults to the
    # 4p15p0 baseline when absent so a hand-built param still behaves as before.
    # 4p16p0 or later (4p17p0 left this function unchanged); string order is
    # release order.
    _v416 = str(getattr(param, 'matlab_version', '4p15p0')) >= '4p16p0'

    M = int(param.samples_per_ui)

    # ---- Sinc helper (MATLAB sinc = sin(pi*x)/(pi*x)) ----
    def _sinc(x):
        px = np.pi * x + np.finfo(float).tiny  # eps(0) ≈ 5e-324; tiny is close enough
        return np.sin(px) / px

    # ---- Build Rx and Tx filters ----
    faxis = chdata[0].faxis
    H_bt = bt_fn(param, faxis, OP.Bessel_Thomson)
    H_bw = bw_fn(param, faxis, OP.Butterworth)
    H_r = H_bw * H_bt
    # Gaussian transmitter filter (eq uses transition time in ns; faxis in Hz)
    ttr = OP.transmitter_transition_time  # in ns (9e-3 typical)
    H_t = np.exp(-(np.pi * faxis / 1e9 * ttr / 1.6832) ** 2)
    H_filters = H_r * H_t

    for i in range(int(param.number_of_s4p_files)):
        ch = chdata[i]

        # ---- Impulse responses ----
        ch.uneq_imp_response, ch.t, ch.causality_correction_dB, ch.truncation_dB = \
            s21_fn(ch.sdd21, ch.faxis, param.sample_dt, OP, param)

        ch.uneq_pulse_response = lfilter(np.ones(M), 1, ch.uneq_imp_response)
        if _v416:
            # ML 4p16p0 L1236. cumsum of the pulse response decimated at the
            # symbol rate: MATLAB (1:samples_per_ui:end) is 1-based and starts
            # at the first sample, so the Python slice is [::M], not [M-1::M].
            ch.uneq_step_response = np.cumsum(ch.uneq_pulse_response[::M])

        ch.uneq_imp_response_raw, ch.t_raw, ch.causality_correction_dB, ch.truncation_dB = \
            s21_fn(ch.sdd21_raw, ch.faxis, param.sample_dt, OP, param)

        ch.uneq_pulse_response_raw = lfilter(np.ones(M), 1, ch.uneq_imp_response_raw)
        if _v416:
            ch.uneq_step_response_raw = np.cumsum(ch.uneq_pulse_response_raw[::M])

        ch.uneq_imp_response_raw_filtered, ch.t_raw_fltr, ch.causality_correction_dB, ch.truncation_dB = \
            s21_fn(ch.sdd21_raw * H_filters, ch.faxis, param.sample_dt, OP, param)
        if _v416:
            # ML 4p16p0 L1249-1250. Both fields are new: 4p15p0 computed the
            # filtered raw IMPULSE response but never a pulse or step from it.
            ch.uneq_pulse_response_raw_filtered = \
                lfilter(np.ones(M), 1, ch.uneq_imp_response_raw_filtered)
            ch.uneq_step_response_raw_filtered = \
                np.cumsum(ch.uneq_pulse_response_raw_filtered[::M])

        ch.uneq_imp_response_orig, ch.t_orig, ch.causality_correction_dB, ch.truncation_dB = \
            s21_fn(ch.sdd21_orig, ch.faxis, param.sample_dt, OP, param)

        ch.uneq_pulse_response_orig = lfilter(np.ones(M), 1, ch.uneq_imp_response_orig)
        if _v416:
            ch.uneq_step_response_orig = np.cumsum(ch.uneq_pulse_response_orig[::M])
        # Note: MATLAB recomputes this after the next filtered version
        ch.uneq_pulse_response_orig_filtered = lfilter(np.ones(M), 1, ch.uneq_pulse_response_orig)

        ch.uneq_imp_response_orig_filtered, ch.t_orig_fltr, ch.causality_correction_dB, ch.truncation_dB = \
            s21_fn(ch.sdd21_orig * H_filters, ch.faxis, param.sample_dt, OP, param)

        ch.uneq_pulse_response_orig_filtered = lfilter(np.ones(M), 1, ch.uneq_imp_response_orig_filtered)
        if _v416:
            # ML 4p16p0 L1265 -- after the SECOND assignment of
            # uneq_pulse_response_orig_filtered, so it uses the filtered-impulse
            # version, not the double-filtered one computed above.
            ch.uneq_step_response_orig_filtered = \
                np.cumsum(ch.uneq_pulse_response_orig_filtered[::M])

        # ---- Amplitude scaling ----
        USE_channel_amplitude = True
        if OP.RX_CALIBRATION and i > 0:
            USE_channel_amplitude = False
        if OP.PSDRXCAL and i == len(chdata) - 1:
            USE_channel_amplitude = False
        if USE_channel_amplitude:
            ch.uneq_imp_response = ch.uneq_imp_response * ch.A
            if _v416:
                # ML 4p16p0 L1277-1278. In 4p15p0 only the IMPULSE response was
                # scaled; the pulse response was built earlier from the unscaled
                # impulse and never corrected, so the two were inconsistent by a
                # factor of A. This is the one change in 4p16p0 that moves
                # already-reported numbers (peak_uneq_pulse_mV,
                # steady_state_voltage_mV), which is why it is version-gated.
                ch.uneq_pulse_response = ch.uneq_pulse_response * ch.A
                ch.uneq_step_response = ch.uneq_step_response * ch.A

        # ---- CD/DC common-mode impulse responses ----
        # 4p16p0 L1280/L1295 guard both conversions: "some test fixtures have
        # almost zero CM and will cause TD conversion to fail". 4p15p0 called
        # them unconditionally.
        if (not _v416) or float(np.mean(np.abs(ch.scd21_orig))) > 1e-6:
            (ch.uneq_CD_imp_response_filtered, ch.t_CD_fltr,
             ch.causality_correction_CD_dB, _trunc_cd) = \
                s21_fn(ch.scd21_orig * H_filters, ch.faxis, param.sample_dt, OP, param)
            ch.uneq_pulse_CD_response_filtered = \
                lfilter(np.ones(M), 1, ch.uneq_CD_imp_response_filtered)
        else:
            ch.t_CD_fltr = ch.t_raw
            ch.uneq_pulse_CD_response_filtered = np.zeros(len(ch.t_raw))
            _trunc_cd = 0.0
        # 4p16p0 L1284 fixed the field-name typo truncation__CD_dB ->
        # truncation_CD_dB. Both names are set so neither spelling breaks a
        # reader, whichever version is selected.
        ch.truncation_CD_dB = _trunc_cd
        ch.truncation__CD_dB = _trunc_cd

        if (not _v416) or float(np.mean(np.abs(ch.sdc21_orig))) > 1e-6:
            (ch.uneq_DC_imp_response_filtered, ch.t_DC_fltr,
             ch.causality_correction_DC_dB, _trunc_dc) = \
                s21_fn(ch.sdc21_orig * H_filters, ch.faxis, param.sample_dt, OP, param)
            ch.uneq_pulse_DC_response_filtered = \
                lfilter(np.ones(M), 1, ch.uneq_DC_imp_response_filtered)
        else:
            ch.t_DC_fltr = ch.t_raw
            ch.uneq_pulse_DC_response_filtered = np.zeros(len(ch.t_raw))
            _trunc_dc = 0.0
        ch.truncation_DC_dB = _trunc_dc
        ch.truncation__DC_dB = _trunc_dc

        # ---- Frequency band indices ----
        fax = ch.faxis
        a_idx = np.searchsorted(fax, param.f2, side='left')
        index_f2 = len(fax) - 1 if a_idx >= len(fax) else a_idx
        b_idx = np.searchsorted(fax, param.f1, side='right') - 1
        index_f1 = 0 if b_idx < 0 else b_idx

        # ---- Common-mode RMS (CD, DC) ----
        rss = -np.inf
        for im in range(M):
            rss = max(rss, np.linalg.norm(ch.uneq_pulse_CD_response_filtered[im::M]))
        ch.CD_CM_RMS = rss * np.sqrt(param.sigma_X)

        ch.VCM_CD_HF_struct = cm_fn(M, ch.uneq_pulse_CD_response_filtered, param.levels, param.P_peak, OP)
        ch.VCM_DC_HF_struct = cm_fn(M, ch.uneq_pulse_DC_response_filtered, param.levels, param.P_peak, OP)

        # ---- SCMR metrics (peak and RMS-based) ----
        # Use the first channel's filtered orig PR for signal power (i==0 only, but applied to all)
        PR_ORIG_fltr = chdata[0].uneq_pulse_response_orig_filtered
        ipeak = int(np.argmax(PR_ORIG_fltr))
        V_peak = PR_ORIG_fltr[ipeak]
        istart = ipeak % M  # 0-based phase offset
        nsteps = len(PR_ORIG_fltr) // M
        iend = istart + nsteps * M
        PR_sampled = PR_ORIG_fltr[istart:iend:M]
        P_signal = float(np.linalg.norm(PR_sampled) ** 2)
        ch.P_signal = P_signal

        CMn_cd = ch.VCM_CD_HF_struct.CMn
        CMn_dc = ch.VCM_DC_HF_struct.CMn
        # ML 1342-1345: 10*log10(V_peak^2/CMn^2) with NO 1e-300 floor on the
        # denominator. On a perfectly balanced channel CMn is exactly 0, where
        # the reference gives +Inf and the floored form gives a finite ~2993 dB
        # that reads like a measurement. Same defect FD_Processing carried on
        # its SCMR_FD_CD_ch_dB, and its test fixture used scd21_orig = zeros,
        # so that one was live. errstate only silences numpy; MATLAB divides by
        # zero quietly.
        # np.float64, not Python float: a Python scalar divided by zero RAISES
        # ZeroDivisionError, where MATLAB and numpy both give Inf. errstate
        # governs numpy's warning only, so the operands have to be numpy types
        # for the reference's own answer to come out at all.
        _cd = np.float64(CMn_cd) ** 2
        _dc = np.float64(CMn_dc) ** 2
        _vp = np.float64(V_peak) ** 2
        _ps = np.float64(P_signal)
        with np.errstate(divide='ignore', invalid='ignore'):
            ch.SCMR_CD_ch_pk = 10.0 * np.log10(_vp / _cd)
            ch.SCMR_CD_ch = 10.0 * np.log10(_ps / _cd)
            ch.SCMR_DC_ch_pk = 10.0 * np.log10(_vp / _dc)
            ch.SCMR_DC_ch = 10.0 * np.log10(_ps / _dc)

        # ---- Console output (matches MATLAB fprintf) ----
        if not OP.DISPLAY_WINDOW and i == 0:
            print('processing COM PDF ', end='')
        print(f'{ch.base}\tCausality correction = {ch.causality_correction_dB:.1f} dB', end='')
        if OP.ENFORCE_CAUSALITY:
            print()
        else:
            print(' (not applied)')
        print(f'{ch.base}\tTruncation ratio = {ch.truncation_dB:.1f} dB')

    return chdata


if __name__ == '__main__':
    import numpy as np
    from types import SimpleNamespace

    # Smoke test: single channel with unit impulse at sample 0
    N = 64
    M = 4
    faxis = np.linspace(0, 25e9, N)
    sdd21 = np.ones(N, dtype=complex)
    sdd21_raw = sdd21.copy()
    sdd21_orig = sdd21.copy()
    scd21_orig = np.zeros(N, dtype=complex)
    sdc21_orig = np.zeros(N, dtype=complex)

    ch = SimpleNamespace(
        faxis=faxis, sdd21=sdd21, sdd21_raw=sdd21_raw, sdd21_orig=sdd21_orig,
        scd21_orig=scd21_orig, sdc21_orig=sdc21_orig,
        A=1.0, type='THRU', base='test_ch'
    )

    param = SimpleNamespace(
        samples_per_ui=M, sample_dt=1 / (2 * 25e9), number_of_s4p_files=1,
        package_testcase_i=1, sigma_X=1.0, ndfe=4, f2=10e9, f1=0.5e9,
        levels=4, P_peak=1e-4, fb=53.125e9, fb_BT_cutoff=0.473, Z0=50.0,
    )
    OP = SimpleNamespace(
        Bessel_Thomson=False, Butterworth=False, transmitter_transition_time=8e-3,
        RX_CALIBRATION=False, PSDRXCAL=False, DEBUG=False, DISPLAY_WINDOW=False,
        ENFORCE_CAUSALITY=False,
    )

    result = COM_FD_to_TD([ch], param, OP)
    print(f'uneq_pulse_response max = {max(result[0].uneq_pulse_response):.4f}')
    print(f'P_signal = {result[0].P_signal:.4f}')
    print('Smoke test PASSED')
