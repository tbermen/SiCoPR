def OptFom_FD_or_TD_Fields(TDMODE):
    """Return field name strings for FD or TD mode (MATLAB lines 3505-3514)."""
    if TDMODE:
        return 'uneq_pulse_response', 'ctle_pulse_response'
    return 'uneq_imp_response', 'ctle_imp_response'
