import numpy as np


def OptFom_Calc_Noise_XC(H_low_xc, ctle_gain_xc, SETTINGS, param, OP):
    """Compute cross-correlation noise for WIENER-HOPF FFE method (MATLAB lines 3010-3033).

    Returns Noise_XC array, or empty list if FFE_OPT_METHOD is not WIENER-HOPF.
    """
    H_low_xc = np.asarray(H_low_xc, dtype=complex).ravel()
    ctle_gain_xc = np.asarray(ctle_gain_xc, dtype=complex).ravel()

    if str(OP.FFE_OPT_METHOD).upper() != 'WIENER-HOPF':
        return []

    H_r_xc = np.asarray(SETTINGS.H_r_xc, dtype=complex).ravel()
    f_xc = np.asarray(SETTINGS.f_xc, dtype=float).ravel()
    N_fft_by2 = int(SETTINGS.N_fft_by2)

    H_ctf_xc = H_low_xc * ctle_gain_xc
    H_rx_ctle_xc = H_r_xc * H_ctf_xc
    P = H_rx_ctle_xc * np.conj(H_rx_ctle_xc)  # power spectrum (real)
    Var_eta0 = float(param.eta_0) * float(f_xc[-1]) / 1e9

    # MATLAB: ifft(P, 2*N, 'symmetric') = real(ifft([P; zeros(N,1)]))
    N = len(P)
    XC = np.real(np.fft.ifft(np.concatenate([P, np.zeros(N)])))

    M = int(param.samples_per_ui)
    Noise_XC = Var_eta0 * XC[0:N_fft_by2:M]

    if OP.Do_White_Noise:
        Noise_XC = Noise_XC[:1]

    return Noise_XC
