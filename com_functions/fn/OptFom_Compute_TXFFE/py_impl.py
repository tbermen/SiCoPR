import numpy as np
from com_functions.fn.FFE_Fast.py_impl import FFE_Fast as _FFE_Fast
from com_functions.fn.OptFom_FD_or_TD_Fields.py_impl import OptFom_FD_or_TD_Fields as _OptFom_FD_or_TD_Fields
from types import SimpleNamespace


# The inlined copy that used to live here returned the wrong field names:
# ('td_ctle_imp_response','td_ctle_imp_response') for TDMODE and
# ('ctle_resp','ctle_imp_response') otherwise, so the TD path read a field the
# reference never names.  It also used a bare `if TDMODE:`, which the canonical
# has already been corrected for.  Collapsed onto the canonical import.
# COM Octave: OptFom_FD_or_TD_Fields(1) -> 'uneq_pulse_response',
#   'ctle_pulse_response';  OptFom_FD_or_TD_Fields(0) -> 'uneq_imp_response',
#   'ctle_imp_response'.


# --- inline from FFE_Fast (MATLAB 2049-2062) ---


def OptFom_Compute_TXFFE(chdata, pulse_struc, txffe, ctle_response_updated, param, OP):
    """Compute TX FFE-equalized pulse response (MATLAB lines 3370-3417).

    Returns (sbr, chdata, pulse_struc).
    """
    _, ctle_field = _OptFom_FD_or_TD_Fields(OP.TDMODE)
    txffe = np.asarray(txffe, dtype=float).ravel()
    n_taps = len(txffe)
    M = int(param.samples_per_ui)
    num_pre = int(param.cursor_index) - 1  # 0-based

    ich = int(param.num_s4p_files) if OP.RxFFE_with_MMSE else 1

    # MATLAB struct arrays auto-grow when pulse_struc(ii).field is assigned; a
    # Python list does not, so extend it to ich entries before indexing (fixes
    # IndexError on multi-channel/crosstalk runs with RxFFE_with_MMSE).
    while len(pulse_struc) < ich:
        pulse_struc.append(SimpleNamespace(pulse_ctle_circshift=None))

    if ctle_response_updated:
        for ii in range(ich):
            cd_field = getattr(chdata[ii], ctle_field)
            signal = np.asarray(cd_field).ravel()
            if OP.TDMODE:
                pulse = signal.copy()
            else:
                # conv2(signal, ones(M,1)) then take first len(signal) elements
                pulse_full = np.convolve(signal, np.ones(M))
                pulse = pulse_full[:len(signal)]
            pulse_struc[ii].pulse_ctle = pulse

            # Build circshift matrix: shape (len(pulse), n_taps)
            mat = np.zeros((len(pulse), n_taps))
            for k in range(n_taps):
                shift = (k - num_pre) * M  # MATLAB: (k-1-num_pre)*M with 1-based k
                mat[:, k] = np.roll(pulse, shift)
            pulse_struc[ii].pulse_ctle_circshift = mat

    sbr = _FFE_Fast(txffe, pulse_struc[0].pulse_ctle_circshift)
    chdata[0].pulse_response_w_CFT_TXFFE_noRxFFE = sbr

    if ich > 1:
        for ii in range(1, ich):
            if chdata[ii].type in ('FEXT', 'THRU'):
                chdata[ii].pulse_response_w_CFT_TXFFE_noRxFFE = _FFE_Fast(
                    txffe, pulse_struc[ii].pulse_ctle_circshift)
            else:
                chdata[ii].pulse_response_w_CFT_TXFFE_noRxFFE = pulse_struc[ii].pulse_ctle

    return sbr, chdata, pulse_struc
