from types import SimpleNamespace


def OptFom_Initialize_Loop_Struct():
    """Create and return the THIS loop struct with all fields initialized to empty (MATLAB lines 3538-3579)."""
    THIS = SimpleNamespace(
        FOM=0,
        tx_index_vector=[],
        ctle_index=[],
        g_LP_index=[],
        itick=[],
        g_dc=[],
        g_DC_low=[],
        H_ctf=[],
        txffe=[],
        cursor_i=[],
        A_s=[],
        A_p=[],
        far_cursors=[],
        precursors=[],
        dfetaps=[],
        tail_RSS=[],
        floating_tap_coef=[],
        excess_dfe_cursors=[],
        C=[],
        MMSE_results=[],
        PSD_results=[],
        floating_tap_locations=[],
        sigma_N=[],
        sigma_TX=[],
        total_noise_rms=[],
        ISI_N=[],
        h_J=[],
        sigma_ne=[],
    )
    return THIS
