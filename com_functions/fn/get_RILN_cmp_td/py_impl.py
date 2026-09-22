# ============================================================
# MATLAB→Python translation notes for get_RILN_cmp_td
# MATLAB lines: 6694–6875
# ============================================================
# Row/col: MATLAB uses row vectors; Python uses 1D arrays (ravel).
# 1000-echo sum: geometric series in FD; looped faithfully.
# fmin_idx: MATLAB 1-based find(f>=fmin,1,'first') → np.searchsorted(f,fmin)
# fmbg least squares: np.linalg.lstsq(fmbg, LGw) replaces normal equations
# filter(ones(1,M),1,x) → lfilter(ones(M),1,x)
# ipeak: MATLAB 1-based argmax → 0-based np.argmax; range ipeak:range_end
#   MATLAB inclusive → Python slice [ipeak:range_end] (exclusive upper OK since
#   MATLAB range_end = min(len,len) and slicing is safe)
# calculate_delay_CausalityEnforcement: may raise; caught with try/except
# get_pdf_from_sampled_signal: returns SimpleNamespace with .x and .y
# All callee functions are stubbed; no cross-py_impl imports.
# ============================================================

import numpy as np
from scipy.signal import lfilter
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Callee stubs
# ---------------------------------------------------------------------------

def _Bessel_Thomson_Filter(param, faxis, enable):
    if not enable:
        return np.ones(len(faxis))
    f0 = param.fb_BT_cutoff * param.fb
    s = 1j * faxis / (f0 + 1e-300)
    b = [105, 105, 45, 10, 1]
    return np.abs(b[0] / np.polyval(b, s))


def _length(x):
    """MATLAB length(): the longest dimension, 0 when empty, 1 for a scalar."""
    if x.size == 0:
        return 0
    return max(x.shape) if x.ndim else 1


def _Butterworth_Filter(param, faxis, enable):
    faxis = np.asarray(faxis, dtype=float)
    # MATLAB `if enable` is true only for a non-empty value whose elements are
    # ALL non-zero; `not enable` raised on any numpy array of more than one
    # element.  ones(1,length(f)) uses the LONGEST dimension, and len(f) raised
    # TypeError on a scalar f where MATLAB gives 1.
    use = np.asarray(enable)
    if not (use.size and np.all(use)):
        return np.ones(_length(faxis))
    f0 = param.fb / 2.0
    return 1.0 / np.sqrt(1.0 + (faxis / (f0 + 1e-300)) ** 8)


def _s21_to_impulse_DC(sdd21, faxis, sample_dt, OP, param):
    N = len(sdd21)
    S = np.zeros(2 * N - 2, dtype=complex)
    S[:N] = sdd21
    S[N:] = np.conj(sdd21[-2:0:-1])
    ir = np.real(np.fft.ifft(S))
    t = np.arange(len(ir)) * sample_dt
    return ir, t, 0.0, 0.0


def _calculate_delay_CausalityEnforcement(faxis, sdd21, param, OP):
    """Stub: returns delay = 0."""
    return 0.0, 0


def _get_pdf_from_sampled_signal(samples, levels, bin_size, flag):
    """Stub: Gaussian PDF approximation from sample RMS."""
    rms = float(np.sqrt(np.mean(np.asarray(samples, dtype=float)**2))) + 1e-30
    n_bins = max(64, int(8 * rms / (bin_size + 1e-30)))
    x = np.linspace(-4 * rms, 4 * rms, n_bins)
    dx = x[1] - x[0]
    y = np.exp(-0.5 * (x / rms)**2) / (rms * np.sqrt(2 * np.pi)) * dx
    y = y / y.sum()
    return SimpleNamespace(x=x, y=y)


def _normal_dist(sigma, n_sigma, bin_size):
    """Stub: Gaussian PDF with given sigma."""
    n_bins = max(64, int(n_sigma * 2 * sigma / (bin_size + 1e-300)))
    x = np.linspace(-n_sigma * sigma, n_sigma * sigma, n_bins)
    dx = x[1] - x[0]
    y = np.exp(-0.5 * (x / (sigma + 1e-300))**2) / ((sigma + 1e-300) * np.sqrt(2 * np.pi)) * dx
    y = y / y.sum()
    return SimpleNamespace(x=x, y=y)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_RILN_cmp_td(sdd21, RIL_struct, faxis_f2, OP, param, A_T,
                    _Bessel_Thomson_Filter_fn=None,
                    _Butterworth_Filter_fn=None,
                    _s21_to_impulse_DC_fn=None,
                    _calculate_delay_fn=None,
                    _get_pdf_fn=None,
                    _normal_dist_fn=None):
    """Compute TD reflection and re-reflection noise (RILN) figure of merit.

    MATLAB lines 6694–6875.

    sdd21: complex insertion loss array (1D, length matching faxis_f2)
    RIL_struct: SimpleNamespace with fields RIL, rho_port1, rho_port2, freq
    faxis_f2: frequency axis (Hz, at least up to fb)
    OP: SimpleNamespace with Bessel_Thomson, Butterworth, transmitter_transition_time,
        BinSize, impulse_response_truncation_threshold, etc.
    param: SimpleNamespace with samples_per_ui, sample_dt, fb, fb_BT_cutoff,
           levels, specBER, etc.
    A_T: amplitude threshold (not used in this function body; kept for signature compat)

    Returns RILN_TD_struct SimpleNamespace with fields:
        REF, FIT, RIL, REF_noise — each with FIR, PR, t, causality/truncation dBs
        ILN, t, FOM, FOM_PDF, SNR_ISI_FOM, SNR_ISI_FOM_PDF, PDF
    """
    bt_fn = _Bessel_Thomson_Filter_fn or _Bessel_Thomson_Filter
    bw_fn = _Butterworth_Filter_fn or _Butterworth_Filter
    s21_fn = _s21_to_impulse_DC_fn or _s21_to_impulse_DC
    delay_fn = _calculate_delay_fn or _calculate_delay_CausalityEnforcement
    pdf_fn = _get_pdf_fn or _get_pdf_from_sampled_signal
    norm_fn = _normal_dist_fn or _normal_dist

    M = int(param.samples_per_ui)

    # ---- Flatten inputs to 1D row-like arrays ----
    sdd21 = np.asarray(sdd21, dtype=complex).ravel()
    RIL = np.asarray(RIL_struct.RIL, dtype=complex).ravel()
    rho_port1 = np.asarray(RIL_struct.rho_port1, dtype=complex).ravel()
    rho_port2 = np.asarray(RIL_struct.rho_port2, dtype=complex).ravel()
    RIL_f = np.asarray(RIL_struct.freq, dtype=float).ravel()
    faxis_f2 = np.asarray(faxis_f2, dtype=float).ravel()

    # ---- Override OP fields (matching MATLAB L6836-6838) ----
    # MATLAB passes OP BY VALUE, so these assignments are local to this function.
    # Python passes it by reference: mutating OP here leaked 'trend_to_DC' /
    # 'interp_to_DC' into every S-parameter interpolation performed afterwards,
    # replacing the configured 'linear_trend_to_DC' /
    # 'extrap_cubic_to_dc_linear_to_inf'. That changes the DC/low-frequency
    # extrapolation and hence the impulse and pulse responses — visible as a
    # low steady-state voltage, a high pulse peak, and a large ISI error, while
    # leaving Nyquist-band magnitude metrics (IL, ICN) untouched.
    # get_ILN_cmp_td already uses this OP_copy pattern; match it here.
    OP = SimpleNamespace(**vars(OP))
    OP.interp_sparam_mag = 'trend_to_DC'
    OP.interp_sparam_phase = 'interp_to_DC'
    OP.impulse_response_truncation_threshold = 1e-7

    # ---- Echo sum: port1 and port2 reflection/re-reflection noise ----
    number_of_echos = 1000
    fmin = 1e9
    port2_rn = np.zeros(len(RIL), dtype=complex)
    port1_rn = np.zeros(len(RIL), dtype=complex)
    abs_RIL = np.abs(RIL)
    for m in range(1, number_of_echos + 1):
        port2_rn += (abs_RIL * (RIL ** (2 * m)) *
                     (rho_port1 ** m) * (rho_port2 ** m) *
                     (1 + rho_port1) * (1 + rho_port2))
        port1_rn += (abs_RIL * (RIL ** (2 * m - 1)) *
                     (rho_port1 ** (m - 1)) * (rho_port2 ** m) *
                     (1 + rho_port1) * (1 + rho_port1))

    # ---- Remove data below fmin (bad TD conversion near DC) ----
    fmin_idx = int(np.searchsorted(RIL_f, fmin))
    port2_rn = port2_rn[fmin_idx:]
    port1_rn = port1_rn[fmin_idx:]
    f_rn = RIL_f[fmin_idx:]

    # ---- Log-domain polynomial fit to sdd21 (trend removal for ILN) ----
    f = faxis_f2
    N = len(f)
    sdd21_col = sdd21.reshape(-1, 1)  # N×1
    fmbg = np.column_stack([
        sdd21,
        np.sqrt(f) * sdd21,
        f * sdd21,
        f**2 * sdd21,
    ])  # N×4
    unwraplog = np.log(np.abs(sdd21) + 1e-300) + 1j * np.unwrap(np.angle(sdd21))
    LGw = sdd21 * unwraplog  # N, complex
    alpha, _, _, _ = np.linalg.lstsq(fmbg, LGw, rcond=None)
    efit_C = alpha[0] + alpha[1] * np.sqrt(f) + alpha[2] * f + alpha[3] * f**2
    FIT = np.exp(efit_C)

    # ---- TD responses for REF, FIT ----
    H_bw = bw_fn(param, f, 1)
    H_t = np.exp(-(np.pi * f / 1e9 * OP.transmitter_transition_time / 1.6832) ** 2)
    H_tw = np.ones(len(f))  # Tukey window overridden to ones in MATLAB

    RILN_TD_struct = SimpleNamespace()

    RILN_TD_struct.REF = SimpleNamespace()
    (RILN_TD_struct.REF.FIR,
     RILN_TD_struct.REF.t,
     RILN_TD_struct.REF.causality_correction_dB,
     RILN_TD_struct.REF.truncation_dB) = s21_fn(sdd21 * H_bw * H_t * H_tw, f, param.sample_dt, OP, param)
    RILN_TD_struct.REF.PR = lfilter(np.ones(M), 1, RILN_TD_struct.REF.FIR)

    RILN_TD_struct.FIT = SimpleNamespace()
    (RILN_TD_struct.FIT.FIR,
     RILN_TD_struct.FIT.t,
     RILN_TD_struct.FIT.causality_correction_dB,
     RILN_TD_struct.FIT.truncation_dB) = s21_fn(FIT * H_bw * H_t * H_tw, f, param.sample_dt, OP, param)
    RILN_TD_struct.FIT.PR = lfilter(np.ones(M), 1, RILN_TD_struct.FIT.FIR)

    # ---- TD response for RIL ----
    H_bw_r = bw_fn(param, RIL_f, 1)
    H_t_r = np.exp(-(np.pi * RIL_f / 1e9 * OP.transmitter_transition_time / 1.6832) ** 2)
    H_tw_r = np.ones(len(RIL_f))

    RILN_TD_struct.RIL = SimpleNamespace()
    (RILN_TD_struct.RIL.FIR,
     RILN_TD_struct.RIL.t,
     RILN_TD_struct.RIL.causality_correction_dB,
     RILN_TD_struct.RIL.truncation_dB) = s21_fn(RIL * H_bw_r * H_t_r * H_tw_r, RIL_f, param.sample_dt, OP, param)
    RILN_TD_struct.RIL.PR = lfilter(np.ones(M), 1, RILN_TD_struct.RIL.FIR)

    # ---- Channel delay → apply to port2 noise ----
    try:
        delay_sec, _ = delay_fn(f, sdd21, param, OP)
    except Exception:
        delay_sec = 0.0
    port2_rn = port2_rn * np.exp(-1j * 2 * np.pi * f_rn * delay_sec)

    # ---- TD response for REF_noise (port2 reflection/re-reflection) ----
    H_bw_n = bw_fn(param, f_rn, 1)
    H_t_n = np.exp(-(np.pi * f_rn / 1e9 * OP.transmitter_transition_time / 1.6832) ** 2)
    H_tw_n = np.ones(len(f_rn))

    RILN_TD_struct.REF_noise = SimpleNamespace()
    (RILN_TD_struct.REF_noise.FIR,
     RILN_TD_struct.REF_noise.t,
     RILN_TD_struct.REF_noise.causality_correction_dB,
     RILN_TD_struct.REF_noise.truncation_dB) = s21_fn(port2_rn * H_bw_n * H_t_n * H_tw_n,
                                                       f_rn, param.sample_dt, OP, param)
    RILN_TD_struct.REF_noise.PR = lfilter(np.ones(M), 1, RILN_TD_struct.REF_noise.FIR)

    # ---- Extract ILN from cursor of REF.PR ----
    REF_PR = RILN_TD_struct.REF.PR
    RN_PR = RILN_TD_struct.REF_noise.PR
    ipeak = int(np.argmax(REF_PR))  # 0-based
    range_end = min(len(REF_PR), len(RN_PR))
    RILN_TD_struct.ILN = RN_PR[ipeak:range_end]
    RILN_TD_struct.t = RILN_TD_struct.REF_noise.t[ipeak:range_end]
    RILN_TD_struct.FOM = -np.inf
    RILN_TD_struct.FOM_PDF = -np.inf
    rms_fom = -np.inf

    # ---- FOM: max norm over M sample phases ----
    bin_size = float(getattr(OP, 'BinSize', 1e-3))
    specBER = float(getattr(param, 'specBER', 1e-4))
    ILN = RILN_TD_struct.ILN

    best_pdf = None
    for im in range(M):
        phase_samples = ILN[im::M]
        if len(phase_samples) == 0:
            continue
        RILN_TD_struct.FOM = max(RILN_TD_struct.FOM, float(np.linalg.norm(phase_samples)))
        pdf = pdf_fn(phase_samples, param.levels, bin_size, 0)
        # rms from PDF: sqrt(sum(y * x^2))
        rms = float(np.sqrt(np.sum(pdf.y * pdf.x**2))) * np.sqrt(2)
        if rms > rms_fom:
            rms_fom = rms
            # CDF threshold
            cdf_y = np.cumsum(pdf.y)
            idx_ber = int(np.searchsorted(cdf_y, specBER))
            if idx_ber < len(pdf.x):
                RILN_TD_struct.FOM_PDF = -pdf.x[idx_ber]
            else:
                RILN_TD_struct.FOM_PDF = -np.inf
            best_pdf = pdf

    RILN_TD_struct.PDF = best_pdf

    # ---- SNR metrics ----
    FIT_peak = float(RILN_TD_struct.FIT.PR[ipeak]) if ipeak < len(RILN_TD_struct.FIT.PR) else 1.0
    RILN_TD_struct.SNR_ISI_FOM = 20.0 * np.log10(
        abs(FIT_peak) / (abs(RILN_TD_struct.FOM) + 1e-300))
    RILN_TD_struct.SNR_ISI_FOM_PDF = 20.0 * np.log10(
        abs(FIT_peak) / (abs(RILN_TD_struct.FOM_PDF) + 1e-300))

    return RILN_TD_struct


if __name__ == '__main__':
    from types import SimpleNamespace
    import numpy as np

    # Smoke test
    N = 64
    f = np.linspace(1e9, 26.5625e9, N)
    sdd21 = np.exp(-0.1 * f / 1e9) * np.exp(-1j * 2 * np.pi * f * 1e-10)
    rho = 0.1 * np.ones(N)
    RIL_struct = SimpleNamespace(
        RIL=0.05 * np.ones(N, dtype=complex),
        rho_port1=rho.astype(complex),
        rho_port2=rho.astype(complex),
        freq=f,
    )
    param = SimpleNamespace(
        samples_per_ui=4, sample_dt=1 / (2 * 26.5625e9),
        fb=53.125e9, fb_BT_cutoff=0.473, levels=4, specBER=1e-4,
    )
    OP = SimpleNamespace(
        transmitter_transition_time=8e-3, BinSize=1e-3,
        impulse_response_truncation_threshold=1e-7,
    )
    result = get_RILN_cmp_td(sdd21, RIL_struct, f, OP, param, 1.0)
    print(f'SNR_ISI_FOM = {result.SNR_ISI_FOM:.2f} dB')
    print(f'FOM = {result.FOM:.4e}')
    print('Smoke test PASSED')
