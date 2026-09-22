import numpy as np


def OptFom_FD_or_TD_Fields(TDMODE):
    """Return field name strings for FD or TD mode (MATLAB lines 3505-3514)."""
    # MATLAB `if TDMODE` is true only for a NON-EMPTY value whose elements are
    # ALL non-zero -- the same trap the Butterworth/Bessel/Raised-Cosine family
    # carries.  `if TDMODE:` raised on any numpy array of other than one
    # element, and answered TD for the list [1, 0].
    # COM Octave:  []   -> uneq_imp_response      [1 0] -> uneq_imp_response
    #              [1 1] -> uneq_pulse_response   ''    -> uneq_imp_response
    #              'x'  -> uneq_pulse_response    -1    -> uneq_pulse_response
    td = np.asarray(TDMODE)
    if td.size and np.all(td):
        return 'uneq_pulse_response', 'ctle_pulse_response'
    return 'uneq_imp_response', 'ctle_imp_response'
