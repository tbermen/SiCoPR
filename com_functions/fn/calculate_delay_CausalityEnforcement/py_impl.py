# ============================================================
# MATLAB→Python translation notes for calculate_delay_CausalityEnforcement
# MATLAB lines: 4933–5086
# ============================================================
# Channel delay via causality enforcement (IEEE Std 370-2020), 8 steps:
#   1. sdd21_conj = [sdd21, conj(flip(sdd21[1:]))]   (cascade negative freqs)
#   2. log|.|   3. IFFT   4. *sign(t) (+1j first half, -1j second)
#   5. FFT->real causal phase   6. |orig|*exp(-1j*phase)
#   7. interp_Sparam both orig & causal to uniform grid -> pulse responses (ifft)
#   8. delay = peak-index diff refined by circshift norm-minimisation
# interp_Sparam is now implemented (top-level in the assembled module).
# NOTE: MATLAB lines 5048-5049 SWAP the variable names (..._enforced_PR_reduced
# is assigned from sdd21_PR and vice-versa) — kept faithfully.
# ============================================================

import numpy as np


def calculate_delay_CausalityEnforcement(freq, sdd21, param, OP):
    sdd21 = np.asarray(sdd21, dtype=complex).ravel()
    freq = np.asarray(freq, dtype=float).ravel()
    n = len(freq)

    # Step 1: cascade negative frequencies  [sdd21, conj(sdd21(end:-1:2))]
    sdd21_conj = np.concatenate([sdd21, np.conj(sdd21[-1:0:-1])])

    # Step 2: log-magnitude
    sdd21_mag_conj = np.real(np.log(np.abs(sdd21_conj)))

    # Step 3: IFFT of the magnitude
    sdd21_mag_time = np.fft.ifft(sdd21_mag_conj)

    # Step 4: multiply by sign(t): first n by +1j, remainder by -1j
    sdd21_mag_time = sdd21_mag_time.astype(complex)
    sdd21_mag_time[:n] = 1j * sdd21_mag_time[:n]
    sdd21_mag_time[n:] = -1j * sdd21_mag_time[n:]

    # Step 5: causal phase = real(fft(.))
    sdd21_phase = np.real(np.fft.fft(sdd21_mag_time))

    # Step 6: causal function = |original| * exp(-1j*causal_phase)
    sdd21_causality_enforced = np.abs(sdd21_conj) * np.exp(-1j * sdd21_phase)
    sdd21_causality_enforced = sdd21_causality_enforced[:n]

    # Step 7: f-domain -> t-domain pulse response (interp, not s21_to_impulse)
    time_step = float(param.sample_dt)
    fmax = 1.0 / time_step / 2.0
    freq_step = (freq[2] - freq[1]) / 1.0
    nstep = int(round(fmax / freq_step))
    step = fmax / nstep                       # MATLAB 1/round(fmax/freq_step)*fmax
    fout = np.arange(0, fmax + step * 0.5, step)
    M = int(param.samples_per_ui)

    def _pulse_response(ILin):
        IL = np.asarray(interp_Sparam(
            ILin, freq, fout, OP.interp_sparam_mag, OP.interp_sparam_phase, OP, param),
            dtype=complex).ravel()
        nan_idx = np.where(np.isnan(IL))[0]
        for ii in nan_idx:
            if ii > 0:
                IL[ii] = IL[ii - 1]
        # conjugate-symmetric padding for a real ifft
        IL_sym = np.concatenate([[np.real(IL[0])], IL[1:-1],
                                 [np.real(IL[-1])], np.conj(IL[1:-1])[::-1]])
        return lfilter(np.ones(M), 1, np.real(np.fft.ifft(IL_sym)))

    sdd21_PR = _pulse_response(sdd21)
    sdd21_causality_enforced_PR = _pulse_response(sdd21_causality_enforced)

    freq_step2 = (fout[2] - fout[1]) / 1.0
    L = len(sdd21_PR)
    t_base = np.arange(L) / (freq_step2 * L)

    # Step 8: delay (MATLAB var-name swap at L5048-5049 preserved)
    end95 = int(np.floor(len(sdd21_PR) * 95 / 100))
    sdd21_causality_enforced_PR_reduced = sdd21_PR[:end95]
    sdd21_PR_reduced = sdd21_causality_enforced_PR[:end95]

    peak_x_idx = int(np.argmax(sdd21_causality_enforced_PR_reduced))
    peak_y_idx = int(np.argmax(sdd21_PR_reduced))
    peak_idx_difference = peak_x_idx - peak_y_idx

    if peak_idx_difference != 0:
        search_bounds = min(peak_x_idx, peak_y_idx)
        error_value = float(len(sdd21_causality_enforced_PR_reduced))
        error_idx = 0
        for shift_value in range(peak_idx_difference - search_bounds,
                                 peak_idx_difference + search_bounds + 1):
            shifted = np.roll(sdd21_PR_reduced, shift_value)
            current_error = float(np.linalg.norm(shifted - sdd21_PR_reduced))
            if error_value > current_error:
                error_idx = shift_value
                error_value = current_error
        if error_idx == 0:
            raise RuntimeError(
                'calculate_delay_CausalityEnforcement: odd case computing channel delay')
        delay_idx = error_idx
    else:
        delay_idx = 1

    # MATLAB t_base(abs(delay_idx)) is 1-based -> Python abs(delay_idx)-1
    delay_sec = float(t_base[abs(delay_idx) - 1])
    return delay_sec, delay_idx
