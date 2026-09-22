import numpy as np


def OptFom_Calc_Noise_XC(H_low_xc, ctle_gain_xc, SETTINGS, param, OP):
    """Compute cross-correlation noise for WIENER-HOPF FFE method (MATLAB lines 3010-3033).

    Returns Noise_XC array, or empty list if FFE_OPT_METHOD is not WIENER-HOPF.
    """
    H_low_xc = np.asarray(H_low_xc, dtype=complex).ravel()
    ctle_gain_xc = np.asarray(ctle_gain_xc, dtype=complex).ravel()

    # The three SETTINGS reads are above the switch in MATLAB, so a SETTINGS
    # without them is an error for EVERY method, not just WIENER-HOPF.
    # COM Octave, OP.FFE_OPT_METHOD='MMSE' and SETTINGS without H_r_xc:
    #   error: structure has no member 'H_r_xc'
    #   OptFom_Calc_Noise_XC at line 4 column 1
    H_r_xc = np.asarray(SETTINGS.H_r_xc, dtype=complex).ravel()
    f_xc = np.asarray(SETTINGS.f_xc, dtype=float).ravel()
    N_fft_by2 = int(SETTINGS.N_fft_by2)

    # COM Octave: upper() makes the case label case-insensitive; 'wiener-hopf'
    # and 'Wiener-Hopf' both reach the ifft on line 16.
    if str(OP.FFE_OPT_METHOD).upper() != 'WIENER-HOPF':
        return []

    H_ctf_xc = H_low_xc * ctle_gain_xc
    H_rx_ctle_xc = H_r_xc * H_ctf_xc
    P = H_rx_ctle_xc * np.conj(H_rx_ctle_xc)  # power spectrum (real)
    Var_eta0 = float(param.eta_0) * float(f_xc[-1]) / 1e9

    # OPEN, NOT ORACLE-ABLE: MATLAB's ifft(X,n,'symmetric') has no Octave
    # equivalent -- Octave rejects the flag ("invalid conversion from string to
    # real scalar", COM Octave at line 16), so the reference cannot be run on
    # this branch and the identity below is a reading, not a measurement.
    # If MATLAB's flag means "use only the half spectrum and mirror it", which
    # is what a real-output ifft does and what np.fft.irfft does, the right
    # line is  np.fft.irfft(np.concatenate([P, [0.0]]), 2*N)  and this one is
    # about 2x too small (exactly 2*this - P[0]/(2*N)). If instead it means
    # "Hermitian-symmetrise then transform", this line is right, because
    # real(ifft(X)) IS the ifft of the Hermitian part of X. Needs MATLAB to
    # settle. Left as-is: the branch is 'obsolete not used' in the reference
    # and no shipped config selects WIENER-HOPF.
    # MATLAB: XC_rx_ctle = ifft(H.*conj(H), 2*length(H), 'symmetric').
    #
    # ifft(X,n,'symmetric') PADS X to n first, then uses only the first n/2+1
    # entries and infers the rest by conjugate symmetry. For X of length N and
    # n = 2N that is the half spectrum [P, 0] mirrored, i.e. exactly
    # np.fft.irfft([P, 0], 2N).
    #
    # The port previously used real(ifft([P; zeros(N,1)])), which transforms
    # the zero-padded vector as given and keeps the real part. That is the ifft
    # of the HERMITIAN PART of X, not of its symmetric extension, and it is
    # wrong here -- not by a constant factor either.
    #
    # Octave has no 'symmetric' flag, but the construction is reproducible
    # there by building the mirrored spectrum explicitly, which is what
    # octave/patches/OptFom_Calc_Noise_XC.m now does. Measured, N=4:
    #   symmetric : 0.281975  0.187496569158  0.065625  0.0250034308417
    #   old form  : 0.1941125 0.146873284579  0.0859375 0.0656267154208
    N = len(P)
    XC = np.fft.irfft(np.concatenate([P, [0.0]]), 2 * N)

    M = int(param.samples_per_ui)
    Noise_XC = Var_eta0 * XC[0:N_fft_by2:M]

    if OP.Do_White_Noise:
        Noise_XC = Noise_XC[:1]

    return Noise_XC
