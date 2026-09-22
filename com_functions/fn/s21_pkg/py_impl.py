# ============================================================
# MATLAB→Python translation notes for s21_pkg
# MATLAB lines: 10900–11075
# ============================================================
# Concatenates TX pkg + channel + RX pkg, computes VTF (Eq 93A-18).
# Inlines: make_full_pkg (→ make_pkg, synth_tline, combines4p), Bessel_Thomson.
#
# Key paths:
#   INC_PACKAGE=0: return s21 directly (warning)
#   RX_CALIBRATION=1 and channel_number=2: only RX pkg (channel+RX)
#   Normal: (TX pkg not skipped): TX+channel; then +RX pkg
#   PSDRXCAL and channel==num_s4p_files: TX pkg bypassed (identity)
#   Mode 'dc'/'cd': RTX/RRX/Z0gamma adjustments
#
# VTF (Eq 93A-18):
#   s21p = H_t * s21 * (1-gamma_tx)*(1+gamma_rx) /
#          (1 - s11*gamma_tx - s22*gamma_rx
#             - s21*s12*gamma_tx*gamma_rx
#             + s11*s22*gamma_tx*gamma_rx)
#
# SCH returned as SimpleNamespace with fields:
#   .Frequencies, .Parameters (shape [2,2,nfreq]), .NumPorts, .Impedance
# ============================================================

import math

import numpy as np
from com_functions.fn.Bessel_Thomson_Filter.py_impl import Bessel_Thomson_Filter as _Bessel_Thomson_Filter
from com_functions.fn.bessel.py_impl import bessel as _bessel
from com_functions.fn.combines4p.py_impl import combines4p as _combines4p
from com_functions.fn.synth_tline.py_impl import synth_tline as _synth_tline
import copy
from types import SimpleNamespace
from math import factorial


# ---- inlined helpers --------------------------------------------------------

def _make_pkg(f, pkg_len, cpad, cball, pkg_z, pkg_param, lcomp=0.0, cbump=0.0):
    f = np.asarray(f, dtype=float)
    # MATLAB `f(f<eps)=eps` is eps(1) = 2.220446049250313e-16, not the
    # smallest positive double. np.finfo(float).tiny (2.2250738585072014e-308)
    # is 292 orders out, and it is the DC point that gets it, where
    # synth_tline then takes sqrt and log of it. The canonical
    # com_functions/fn/make_pkg/py_impl.py already uses .eps; this inlined
    # copy had not been brought across.
    eps_val = np.finfo(float).eps
    f = np.where(f < eps_val, eps_val, f)
    zref = float(pkg_param.Z0)
    tau = float(pkg_param.pkg_tau)
    gamma_coeff = np.asarray(pkg_param.pkg_gamma0_a1_a2, dtype=float).ravel()

    jw_cpad_z = 1j * 2 * np.pi * f * float(cpad) * zref
    s11pad = -jw_cpad_z / (2.0 + jw_cpad_z)
    s21pad = 2.0 / (2.0 + jw_cpad_z)
    s12pad, s22pad = s21pad.copy(), s11pad.copy()

    if float(lcomp) > 0.0:
        jw_lcomp_z = 1j * 2 * np.pi * f * float(lcomp) / zref
        s11comp = jw_lcomp_z / (2.0 + jw_lcomp_z)
        s21comp = 2.0 / (2.0 + jw_lcomp_z)
        s11pad, s12pad, s21pad, s22pad = _combines4p(
            s11pad, s12pad, s21pad, s22pad,
            s11comp, s21comp, s21comp, s11comp)

    if float(cbump) > 0.0:
        jw_cbump_z = 1j * 2 * np.pi * f * float(cbump) * zref
        s11bump = -jw_cbump_z / (2.0 + jw_cbump_z)
        s21bump = 2.0 / (2.0 + jw_cbump_z)
        s11pad, s12pad, s21pad, s22pad = _combines4p(
            s11pad, s12pad, s21pad, s22pad,
            s11bump, s21bump, s21bump, s11bump)

    S11, S12, S21, S22 = _synth_tline(f, float(pkg_z), zref, gamma_coeff, tau, float(pkg_len))
    s11out1, s12out1, s21out1, s22out1 = _combines4p(
        s11pad, s12pad, s21pad, s22pad, S11, S12, S21, S22)

    jw_cball_z = 1j * 2 * np.pi * f * float(cball) * zref
    s11ball = -jw_cball_z / (2.0 + jw_cball_z)
    s21ball = 2.0 / (2.0 + jw_cball_z)
    s11out, s12out, s21out, s22out = _combines4p(
        s11out1, s12out1, s21out1, s22out1,
        s11ball, s21ball, s21ball, s11ball)
    return s11out, s12out, s21out, s22out


def _make_full_pkg(type_, faxis, param, channel_type, mode='dd', include_die=1):
    faxis = np.asarray(faxis, dtype=float)
    pkg_param = copy.copy(param)

    pkg_name_list = getattr(param, 'PKG_NAME', None)
    if pkg_name_list is not None and len(pkg_name_list) > 0:
        type_upper = str(type_).upper()
        pkg_name = pkg_name_list[0] if type_upper == 'TX' else pkg_name_list[1]
        pkg_struct = param.PKG[pkg_name]
        for field in ['pkg_gamma0_a1_a2', 'pkg_tau']:
            setattr(pkg_param, field, getattr(pkg_struct, field))

    C_diepad = np.asarray(param.C_diepad, dtype=float)
    C_pkg_board = np.asarray(param.C_pkg_board, dtype=float).ravel()
    L_comp = np.asarray(param.L_comp, dtype=float)
    C_bump = np.asarray(param.C_bump, dtype=float).ravel()

    if not include_die:
        C_diepad = C_diepad * 0
        L_comp = L_comp * 0
        C_bump = C_bump * 0

    z_p = np.atleast_2d(np.asarray(param.z_p_next_cases))
    ncases, mele = z_p.shape
    is_vector = C_diepad.ndim == 1 or (C_diepad.ndim == 2 and min(C_diepad.shape) == 1)
    C_flat = C_diepad.ravel()
    L_flat = L_comp.ravel()

    if is_vector:
        Cd_Tx = float(C_flat[0]) if len(C_flat) > 0 else 0.0
        Cd_Rx = float(C_flat[1]) if len(C_flat) > 1 else 0.0
        Lcomp_Tx = float(L_flat[0]) if len(L_flat) > 0 else 0.0
        Lcomp_Rx = float(L_flat[1]) if len(L_flat) > 1 else 0.0
        extra_LC = 0
        num_blocks = mele
    else:
        C2d = C_diepad.reshape(2, -1)
        L2d = L_comp.reshape(2, -1)
        Cd_Tx_arr = C2d[0, :]
        Cd_Rx_arr = C2d[1, :]
        Lcomp_Tx = L2d[0, :]
        Lcomp_Rx = L2d[1, :]
        extra_LC = len(Cd_Tx_arr) - 1
        num_blocks = mele + extra_LC
        Cd_Tx, Cd_Rx = Cd_Tx_arr, Cd_Rx_arr

    insert_zeros = np.zeros(extra_LC)

    def _to_scalar(v):
        return float(np.asarray(v).ravel()[0])

    pkg_Z_c = np.asarray(param.pkg_Z_c, dtype=float)
    type_upper = str(type_).upper()

    if type_upper == 'TX':
        if mele == 1:
            Cpad = np.array([_to_scalar(Cd_Tx)])
            Lcomp = np.array([_to_scalar(Lcomp_Tx)])
            Cbump = np.array([float(C_bump[0])])
            Cball = np.array([float(C_pkg_board[0])])
            Zpkg = np.array([float(pkg_Z_c.ravel()[0])])
        elif mele == 4:
            # MATLAB: Cpad=[Cd_Tx 0 0 0]; Lcomp=[Lcomp_Tx 0 0 0] (L8390-8391).
            # Cd_Tx/Lcomp_Tx are ROW VECTORS when C_d/L_comp are a 2xN matrix
            # (N die LC sections per side), so MATLAB yields len(Cd_Tx)+3 entries,
            # matching num_blocks = mele + extra_LC. _to_scalar() kept only the
            # first section, dropping the rest of the die LC network.
            Cpad = np.concatenate([np.atleast_1d(np.asarray(Cd_Tx, dtype=float)).ravel(),
                                   np.zeros(3)])
            Lcomp = np.concatenate([np.atleast_1d(np.asarray(Lcomp_Tx, dtype=float)).ravel(),
                                    np.zeros(3)])
            Cbump = np.array([float(C_bump[0]), 0.0, 0.0, 0.0])
            C_v = np.asarray(param.C_v, dtype=float).ravel()
            Cball = np.array([0.0, 0.0, float(C_v[0]), float(C_pkg_board[0])])
            if pkg_Z_c.ndim == 1:
                Zpkg = pkg_Z_c[:4]
            else:
                Zpkg = pkg_Z_c[0, :4]
        else:
            raise ValueError(f'make_full_pkg: unsupported mele={mele}')
        ch_upper = str(channel_type).upper()
        if ch_upper == 'THRU':
            Len = np.asarray(param.Pkg_len_TX, dtype=float).ravel()
        elif ch_upper == 'NEXT':
            Len = np.asarray(param.Pkg_len_NEXT, dtype=float).ravel()
        elif ch_upper == 'FEXT':
            Len = np.asarray(param.Pkg_len_FEXT, dtype=float).ravel()
        else:
            Len = np.asarray(param.Pkg_len_TX, dtype=float).ravel()
    else:  # RX
        cb1 = float(C_bump[1]) if len(C_bump) > 1 else float(C_bump[0])
        cp1 = float(C_pkg_board[1]) if len(C_pkg_board) > 1 else float(C_pkg_board[0])
        if mele == 1:
            Cpad = np.array([_to_scalar(Cd_Rx)])
            Lcomp = np.array([_to_scalar(Lcomp_Rx)])
            Cbump = np.array([cb1])
            Cball = np.array([cp1])
            zc_vals = pkg_Z_c.ravel()
            Zpkg = np.array([float(zc_vals[1]) if len(zc_vals) > 1 else float(zc_vals[0])])
        elif mele == 4:
            # MATLAB: Cpad=[Cd_Rx 0 0 0]; Lcomp=[Lcomp_Rx 0 0 0] (L8390-8391).
            # Cd_Rx/Lcomp_Rx are ROW VECTORS when C_d/L_comp are a 2xN matrix
            # (N die LC sections per side), so MATLAB yields len(Cd_Rx)+3 entries,
            # matching num_blocks = mele + extra_LC. _to_scalar() kept only the
            # first section, dropping the rest of the die LC network.
            Cpad = np.concatenate([np.atleast_1d(np.asarray(Cd_Rx, dtype=float)).ravel(),
                                   np.zeros(3)])
            Lcomp = np.concatenate([np.atleast_1d(np.asarray(Lcomp_Rx, dtype=float)).ravel(),
                                    np.zeros(3)])
            Cbump = np.array([cb1, 0.0, 0.0, 0.0])
            C_v = np.asarray(param.C_v, dtype=float).ravel()
            cv1 = float(C_v[1]) if len(C_v) > 1 else float(C_v[0])
            Cball = np.array([0.0, 0.0, cv1, cp1])
            if pkg_Z_c.ndim == 1:
                Zpkg = pkg_Z_c[:4]
            else:
                Zpkg = pkg_Z_c[1, :4] if pkg_Z_c.shape[0] > 1 else pkg_Z_c[0, :4]
        else:
            raise ValueError(f'make_full_pkg: unsupported mele={mele}')
        Len = np.asarray(param.Pkg_len_RX, dtype=float).ravel()

    Cball = np.concatenate([insert_zeros, Cball.ravel()])
    Cbump = np.concatenate([insert_zeros, Cbump.ravel()])
    Len = np.concatenate([insert_zeros, Len.ravel()])
    Zpkg = np.concatenate([insert_zeros, Zpkg.ravel()])

    if mode.lower() in ('dc', 'cd'):
        pkg_param.Z0 = float(param.Z0) / 2.0
        Cpad = Cpad * 2.0
        Cball = Cball * 2.0
        Zpkg = Zpkg * 2.0
        Lcomp = Lcomp / 2.0
        Cbump = Cbump * 2.0

    def _el(arr, j):
        a = np.asarray(arr, dtype=float).ravel()
        return float(a[j]) if j < len(a) else 0.0

    n_blocks = int(num_blocks)
    if n_blocks == 1:
        s11o, s12o, s21o, s22o = _make_pkg(
            faxis, _el(Len, 0), _el(Cpad, 0), _el(Cball, 0), _el(Zpkg, 0),
            pkg_param, _el(Lcomp, 0), _el(Cbump, 0))
    else:
        for j in range(n_blocks):
            sp11, sp12, sp21, sp22 = _make_pkg(
                faxis, _el(Len, j), _el(Cpad, j), _el(Cball, j), _el(Zpkg, j),
                pkg_param, _el(Lcomp, j), _el(Cbump, j))
            if j == 0:
                s11o, s12o, s21o, s22o = sp11, sp12, sp21, sp22
            else:
                s11o, s12o, s21o, s22o = _combines4p(s11o, s12o, s21o, s22o, sp11, sp12, sp21, sp22)
    return s11o, s12o, s21o, s22o


# ---- main function ----------------------------------------------------------

def s21_pkg(chdata, param, OP, channel_number, mode='dd', include_die=1):
    """Compute package-inclusive S21 and VTF (MATLAB lines 10900-11075).

    Returns (s21p, SCH, sigma_ACCM_at_tp0).
    SCH: SimpleNamespace with .Frequencies, .Parameters (2×2×nfreq), .NumPorts, .Impedance.
    sigma_ACCM_at_tp0: scalar (non-zero only in DC mode for channel_number==1).
    """
    if mode is None:
        mode = 'dd'
    mode = str(mode)

    s21 = np.asarray(getattr(chdata, f's{mode}21_raw'), dtype=complex)
    s12 = np.asarray(getattr(chdata, f's{mode}12_raw'), dtype=complex)
    s11 = np.asarray(getattr(chdata, f's{mode}11_raw'), dtype=complex)
    s22 = np.asarray(getattr(chdata, f's{mode}22_raw'), dtype=complex)
    faxis = np.asarray(chdata.faxis, dtype=float)
    channel_type = str(chdata.type)

    if mode.lower() == 'dd':
        s11 = s11 * float(param.kappa1)
        s22 = s22 * float(param.kappa2)

    Z0 = float(param.Z0)
    sigma_ACCM_at_tp0 = 0.0

    R_diepad = np.asarray(param.R_diepad, dtype=float).ravel()

    # TX package
    tx_mode = 'dc' if mode.lower() == 'dc' else 'dd'

    psdrxcal = bool(getattr(OP, 'PSDRXCAL', False))
    num_s4p = int(getattr(param, 'num_s4p_files', 0))

    if psdrxcal and channel_number == num_s4p:
        nf = len(faxis)
        s11out = np.zeros(nf)
        s12out = np.ones(nf)
        s21out = np.ones(nf)
        s22out = np.zeros(nf)
    else:
        s11out, s12out, s21out, s22out = _make_full_pkg(
            'TX', faxis, param, channel_type, tx_mode, include_die)

    # RX package (always 'dd' unless 'cd' mode)
    rx_mode = 'cd' if mode.lower() == 'cd' else 'dd'
    s11in, s12in, s21in, s22in = _make_full_pkg(
        'RX', faxis, param, channel_type, rx_mode, include_die)

    # Reflection coefficients
    tx_rd_sel = int(getattr(param, 'Tx_rd_sel', 1)) - 1  # 1-based → 0-based
    rx_rd_sel = int(getattr(param, 'Rx_rd_sel', 2)) - 1

    if mode.lower() == 'dc':
        RTX = float(R_diepad[tx_rd_sel]) / 2.0
        RRX = float(R_diepad[rx_rd_sel])
        Z0gamma_TX = Z0 / 2.0
        Z0gamma_RX = Z0
    elif mode.lower() == 'cd':
        RTX = float(R_diepad[tx_rd_sel])
        RRX = float(R_diepad[rx_rd_sel]) / 2.0
        Z0gamma_TX = Z0
        Z0gamma_RX = Z0 / 2.0
    else:
        RTX = float(R_diepad[tx_rd_sel])
        RRX = float(R_diepad[rx_rd_sel])
        Z0gamma_TX = Z0
        Z0gamma_RX = Z0

    ideal_tx = bool(getattr(OP, 'IDEAL_TX_TERM', False))
    ideal_rx = bool(getattr(OP, 'IDEAL_RX_TERM', False))
    rx_cal = int(getattr(OP, 'RX_CALIBRATION', 0))
    include_pcb = int(getattr(OP, 'include_pcb', 0))

    if ideal_tx or (rx_cal == 1 and channel_number == 2) or include_pcb == 2:
        gamma_tx = 0.0
    else:
        gamma_tx = (RTX - Z0gamma_TX) / (RTX + Z0gamma_TX)

    if ideal_rx:
        gamma_rx = 0.0
    else:
        gamma_rx = (RRX - Z0gamma_RX) / (RRX + Z0gamma_RX)

    inc_pkg = int(getattr(OP, 'INC_PACKAGE', 1))
    nf = len(faxis)

    if inc_pkg == 0:
        import warnings
        warnings.warn('do not use INC_PACKAGE = 0. Instead use package parameters)')
        # SCH and sigma_ACCM_at_tp0 are assigned inside the INC_PACKAGE~=0
        # branch only, so with INC_PACKAGE=0 the reference has no second or
        # third output to give -- and every call site here asks for all three.
        # COM Octave, INC_PACKAGE=0 with [s21p,SCH,sigma]=s21_pkg(...):
        #   error: element number 2 undefined in return list
        # The port used to invent an SCH, which is a two-port the reference
        # never builds.
        raise ValueError('element number 2 undefined in return list: '
                         's21_pkg leaves SCH unassigned when INC_PACKAGE = 0')

    if rx_cal == 1 and channel_number == 2:
        # Only RX pkg: channel + RX pkg
        s11rx, s12rx, s21rx, s22rx = _combines4p(
            s11, s12, s21, s22,
            s22out, s12out, s21out, s11out)
        SCH = SimpleNamespace(
            Frequencies=faxis,
            Parameters=np.zeros((2, 2, nf), dtype=complex),
            NumPorts=2,
            Impedance=Z0 * 2)
        SCH.Parameters[0, 0, :] = s11rx
        SCH.Parameters[1, 1, :] = s22rx
        SCH.Parameters[0, 1, :] = s12rx
        SCH.Parameters[1, 0, :] = s21rx
        if include_die:
            s21p = (s21rx * (1 - gamma_tx) * (1 + gamma_rx) /
                    (1.0 - s11rx * gamma_tx - s22rx * gamma_rx
                     - s21rx ** 2 * gamma_tx * gamma_rx
                     + s11rx * s22rx * gamma_tx * gamma_rx))
        else:
            s21p = s21rx
        return s21p, SCH, sigma_ACCM_at_tp0

    # Normal path
    if not ideal_tx:
        s11, s12, s21, s22 = _combines4p(
            s11out, s12out, s21out, s22out,
            s11, s12, s21, s22)

    # TX filter H_t
    T_r_filter_type = int(getattr(OP, 'T_r_filter_type', 0))
    H_t = np.ones(nf)
    if ideal_tx or T_r_filter_type == 1:
        tr = float(getattr(OP, 'transmitter_transition_time', 0.0))
        if T_r_filter_type == 0:
            H_t = np.exp(-(np.pi * faxis / 1e9 * tr / 1.6832) ** 2)
        else:
            f9 = faxis / 1e9
            meas_point = int(getattr(OP, 'T_r_meas_point', 0))
            if meas_point == 1:
                k = 1.9466 + 7.12 * np.sqrt(1 - 6.51e-3 / tr)
                denom = (f9 ** 4 * (k * tr) ** 4
                         - f9 ** 3 * (k * tr) ** 3 * 10j
                         - 45 * f9 ** 2 * (k * tr) ** 2
                         + f9 * (k * tr) * 105j + 105)
                H_t = 105.0 / denom
            else:
                H_t = np.exp(-2 * (np.pi * f9 * tr / 1.6832) ** 2) * np.exp(-1j * 2 * np.pi * f9 * tr * 3)

    if not ideal_rx:
        s11, s12, s21, s22 = _combines4p(
            s11, s12, s21, s22,
            s22in, s21in, s12in, s11in)
    else:
        import warnings
        warnings.warn('do not use IDEAL_RX_TERM. Instead hard code package '
                      'and TR parameters')

    if include_die:
        s21p = (H_t * s21 * (1 - gamma_tx) * (1 + gamma_rx) /
                (1.0 - s11 * gamma_tx - s22 * gamma_rx
                 - s21 * s12 * gamma_tx * gamma_rx
                 + s11 * s22 * gamma_tx * gamma_rx))
    else:
        s21p = s21.copy()

    # DC mode: compute AC CM RMS at test point 0
    if mode.lower() == 'dc':
        OP_bt = copy.copy(OP)
        OP_bt.TX_BesselThomson = 1
        H_bt = _Bessel_Thomson_Filter(param, faxis, 1)
        if channel_number == 1:
            ACCM_max_freq = float(getattr(param, 'ACCM_MAX_Freq', faxis[-1]))
            f_int = faxis[faxis <= ACCM_max_freq]
            nf_int = len(f_int)
            H_cc = s21out * (1 - gamma_tx) / (1.0 - s11out * gamma_tx) * H_bt
            ac_cm_rms = float(getattr(param, 'AC_CM_RMS_TX', 0.0))
            # No guard on nf_int or f_int(end): MATLAB divides by f_int(end)
            # whatever it is. COM Octave, dc mode, channel 1:
            #   ACCM_MAX_Freq=0   -> f_int=[0], empty sum, 0/0 -> sigma = NaN
            #   ACCM_MAX_Freq=-1  -> f_int empty ->
            #        error: f_int(0): subscripts must be either integers
            #               1 to (2^63)-1 or logicals
            # The port answered 0.0 for both.
            sigma_ACCM_at_tp0 = np.sqrt(
                2 * ac_cm_rms ** 2
                * np.sum(np.abs(H_cc[1:nf_int]) ** 2 * np.diff(f_int))
                / f_int[-1])

    SCH = SimpleNamespace(
        Frequencies=faxis,
        Parameters=np.zeros((2, 2, nf), dtype=complex),
        NumPorts=2,
        Impedance=(Z0 / 2.0 if mode.lower() in ('dc', 'cd') else Z0 * 2.0))
    SCH.Parameters[0, 0, :] = s11
    SCH.Parameters[1, 1, :] = s22
    SCH.Parameters[0, 1, :] = s12
    SCH.Parameters[1, 0, :] = s21

    return s21p, SCH, sigma_ACCM_at_tp0
