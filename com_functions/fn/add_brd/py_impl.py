# ============================================================
# MATLAB→Python translation notes for add_brd
# MATLAB lines: 4768–4820
# ============================================================
# Inline helpers: synth_tline (11292–11316), combines4p (5327–5370).
# gamma_coeff: MATLAB 1-based (1)(2)(3) → Python 0-based [0][1][2].
# combines4p: vectorized S-param cascade via chain matrix formula.
# f(f<eps)=eps: avoid log(0) in gamma computation.
# brd_Z_c, C_0, C_1: indexed with 1-based MATLAB → Python [0] and [1].
# ============================================================

import numpy as np
from com_functions.fn.combines4p.py_impl import combines4p as _combines4p
from com_functions.fn.synth_tline.py_impl import synth_tline as _synth_tline


def add_brd(chdata, param, OP):
    ctype = chdata.type
    if ctype in ('THRU', 'NOISE'):
        z_bp_tx = param.z_bp_tx
        z_bp_rx = param.z_bp_rx
    elif ctype == 'NEXT':
        z_bp_tx = param.z_bp_rx
        z_bp_rx = param.z_bp_next
    elif ctype == 'FEXT':
        z_bp_tx = param.z_bp_fext
        z_bp_rx = param.z_bp_rx
    else:
        raise ValueError(f'Unknown chdata.type: {ctype}')

    zref = param.Z0
    c1 = np.asarray(param.C_0, dtype=float)
    c2 = np.asarray(param.C_1, dtype=float)
    f = np.asarray(chdata.faxis, dtype=float)
    f = np.where(f < np.finfo(float).eps, np.finfo(float).eps, f)

    # TX side pads
    s11pad1t = -1j * 2 * np.pi * f * c1[0] * zref / (2 + 1j * 2 * np.pi * f * c1[0] * zref)
    s21pad1t = 2 / (2 + 1j * 2 * np.pi * f * c1[0] * zref)
    s11pad2t = -1j * 2 * np.pi * f * c2[0] * zref / (2 + 1j * 2 * np.pi * f * c2[0] * zref)
    s21pad2t = 2 / (2 + 1j * 2 * np.pi * f * c2[0] * zref)

    s11tx, s12tx, s21tx, s22tx = _synth_tline(
        chdata.faxis, param.brd_Z_c[0], param.Z0, param.brd_gamma0_a1_a2, param.brd_tau, z_bp_tx)
    s11tx, s12tx, s21tx, s22tx = _combines4p(
        s11pad1t, s21pad1t, s21pad1t, s11pad1t, s11tx, s12tx, s21tx, s22tx)
    s11tx, s12tx, s21tx, s22tx = _combines4p(
        s11tx, s12tx, s21tx, s22tx, s11pad2t, s21pad2t, s21pad2t, s11pad2t)

    # RX side pads
    s11pad1r = -1j * 2 * np.pi * f * c1[1] * zref / (2 + 1j * 2 * np.pi * f * c1[1] * zref)
    s21pad1r = 2 / (2 + 1j * 2 * np.pi * f * c1[1] * zref)
    s11pad2r = -1j * 2 * np.pi * f * c2[1] * zref / (2 + 1j * 2 * np.pi * f * c2[1] * zref)
    s21pad2r = 2 / (2 + 1j * 2 * np.pi * f * c2[1] * zref)

    s11rx, s12rx, s21rx, s22rx = _synth_tline(
        chdata.faxis, param.brd_Z_c[1], param.Z0, param.brd_gamma0_a1_a2, param.brd_tau, z_bp_rx)
    s11rx, s12rx, s21rx, s22rx = _combines4p(
        s11pad2r, s21pad2r, s21pad2r, s11pad2r, s11rx, s12rx, s21rx, s22rx)
    s11rx, s12rx, s21rx, s22rx = _combines4p(
        s11rx, s12rx, s21rx, s22rx, s11pad1r, s21pad1r, s21pad1r, s11pad1r)

    if OP.include_pcb == 1:
        s11o1, s12o1, s21o1, s22o1 = _combines4p(
            s11tx, s12tx, s21tx, s22tx,
            chdata.sdd11_raw, chdata.sdd12_raw, chdata.sdd21_raw, chdata.sdd22_raw)
        s11out, s12out, s21out, s22out = _combines4p(
            s11o1, s12o1, s21o1, s22o1, s11rx, s12rx, s21rx, s22rx)
    elif OP.include_pcb == 2:
        s11out, s12out, s21out, s22out = _combines4p(
            chdata.sdd11_raw, chdata.sdd12_raw, chdata.sdd21_raw, chdata.sdd22_raw,
            s11rx, s12rx, s21rx, s22rx)
    else:
        raise ValueError(f'Unexpected OP.include_pcb={OP.include_pcb}')

    return s11out, s12out, s21out, s22out
