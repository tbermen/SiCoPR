import numpy as np
from com_functions.fn.combines4p.py_impl import combines4p as _combines4p
from com_functions.fn.synth_tline.py_impl import synth_tline as _synth_tline


def add_brdorig(chdata, param, OP):
    """Add board trace S-parameters to channel data (MATLAB lines 4821-4841).

    Synthesizes TX and RX board traces via synth_tline then cascades with
    channel S-parameters using combines4p. OP.include_pcb selects cascade order:
      1 = tx_trace → channel → rx_trace
      2 = channel → rx_trace
    """
    chtype = chdata.type
    if chtype == 'THRU':
        z_bp_tx = param.z_bp_tx
    elif chtype == 'NEXT':
        z_bp_tx = param.z_bp_next
    elif chtype == 'FEXT':
        z_bp_tx = param.z_bp_fext
    else:
        raise ValueError(f'Unknown chdata.type: {chtype!r}')

    f = np.asarray(chdata.faxis, dtype=float).ravel()
    gc = param.brd_gamma0_a1_a2
    tau = param.brd_tau

    s11tx, s12tx, s21tx, s22tx = _synth_tline(f, param.brd_Z_c[0], param.Z0, gc, tau, z_bp_tx)
    s11rx, s12rx, s21rx, s22rx = _synth_tline(f, param.brd_Z_c[1], param.Z0, gc, tau, param.z_bp_rx)

    sdd11 = np.asarray(chdata.sdd11_raw, dtype=complex).ravel()
    sdd12 = np.asarray(chdata.sdd12_raw, dtype=complex).ravel()
    sdd21 = np.asarray(chdata.sdd21_raw, dtype=complex).ravel()
    sdd22 = np.asarray(chdata.sdd22_raw, dtype=complex).ravel()

    # Compared, not truncated. MATLAB's `switch` tests equality, so 1.5 falls
    # through both cases and leaves the outputs unassigned. COM Octave,
    # add_brdorig with OP.include_pcb=1.5:
    #     error: element number 1 undefined in return list
    # int() rounded 1.5 down to 1 and returned the include_pcb=1 cascade,
    # which the reference will not produce.
    include_pcb = OP.include_pcb
    if include_pcb == 1:
        s11o, s12o, s21o, s22o = _combines4p(s11tx, s12tx, s21tx, s22tx, sdd11, sdd12, sdd21, sdd22)
        return _combines4p(s11o, s12o, s21o, s22o, s11rx, s12rx, s21rx, s22rx)
    if include_pcb == 2:
        return _combines4p(sdd11, sdd12, sdd21, sdd22, s11rx, s12rx, s21rx, s22rx)
    raise ValueError(f'OP.include_pcb must be 1 or 2, got {include_pcb}')
