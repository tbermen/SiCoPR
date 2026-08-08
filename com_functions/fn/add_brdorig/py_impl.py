import numpy as np


def _synth_tline(f, Z_c, Z_0, gamma_coeff, tau, d):
    f = np.asarray(f, dtype=float).ravel()
    gamma_coeff = np.asarray(gamma_coeff, dtype=float)
    f_GHz = f / 1e9
    gamma_1 = gamma_coeff[1] * (1.0 + 1j)
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma_2 = gamma_coeff[2] * (1.0 - 2j / np.pi * np.log(f_GHz)) + 2j * np.pi * tau
    gamma = gamma_coeff[0] + gamma_1 * np.sqrt(f_GHz) + gamma_2 * f_GHz
    gamma[f_GHz == 0] = gamma_coeff[0]
    rho_rl = 0.0 if d == 0 else (Z_c - 2.0 * Z_0) / (Z_c + 2.0 * Z_0)
    exp_gd = np.exp(-d * gamma)
    exp_gd2 = exp_gd ** 2
    denom = 1.0 - rho_rl ** 2 * exp_gd2
    s11 = rho_rl * (1.0 - exp_gd2) / denom
    s21 = (1.0 - rho_rl ** 2) * exp_gd / denom
    return s11, s21.copy(), s21.copy(), s11.copy()


def _combines4p(a11, a12, a21, a22, b11, b12, b21, b22):
    def sq(x): return np.asarray(x, dtype=complex).ravel()
    a11, a12, a21, a22 = sq(a11), sq(a12), sq(a21), sq(a22)
    b11, b12, b21, b22 = sq(b11), sq(b12), sq(b21), sq(b22)
    Nv = 1 - a22 * b11
    return (a11 + a12 * a21 * b11 / Nv, a12 * b12 / Nv,
            b21 * a21 / Nv, b22 + b12 * b21 * a22 / Nv)


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

    include_pcb = int(OP.include_pcb)
    if include_pcb == 1:
        s11o, s12o, s21o, s22o = _combines4p(s11tx, s12tx, s21tx, s22tx, sdd11, sdd12, sdd21, sdd22)
        return _combines4p(s11o, s12o, s21o, s22o, s11rx, s12rx, s21rx, s22rx)
    if include_pcb == 2:
        return _combines4p(sdd11, sdd12, sdd21, sdd22, s11rx, s12rx, s21rx, s22rx)
    raise ValueError(f'OP.include_pcb must be 1 or 2, got {include_pcb}')
