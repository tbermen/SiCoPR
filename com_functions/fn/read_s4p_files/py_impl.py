# ============================================================
# MATLAB→Python translation notes for read_s4p_files
# MATLAB lines: 10594–10767
# ============================================================
# Reads all s4p (or s2p) files from chdata list.
# For each channel, calls read_p4_s4params or read_p2_s2params.
# Sets chdata[i].faxis, sdd11_raw, sdd21_raw, sdd12_raw, sdd22_raw,
#   sdcXX_raw, scdXX_raw, sccXX_raw, and copies to _orig fields.
# If OP.include_pcb: calls _add_brd to add board reflections.
# Then calls _s21_pkg for sdd21p (with package), _s21_pkg(include_die=0) for sdd21p_nodie,
#   and _s21_pkg(mode='cd') for sdc21p + sigma_ACCM_at_tp0.
# Skips reading if chdata[i].faxis is already set (multiple test cases).
# Validates: all channels must share same frequency axis as THRU.
# No imports from sibling py_impl.py files: all helpers are inlined.
# ============================================================

import numpy as np

def _mextreme_complex(a, take):
    """MATLAB orders complex values by magnitude, then by angle; numpy orders
    them lexicographically by real part, so max([3+4i, 5]) is 3+4i in MATLAB
    and 5 in numpy. take is -1 for max, 0 for min."""
    f = np.asarray(a).ravel()
    good = ~np.isnan(np.abs(f))
    if not good.any():
        return f[0]
    g = f[good]
    return g[np.lexsort((np.angle(g), np.abs(g)))[take]]


def _mmax(a):
    """MATLAB max(): a NaN is skipped unless every element is NaN, and complex
    values are ordered by magnitude then angle.

    np.max propagates a NaN, so one bad sample swallows the result where MATLAB
    ignores it. np.nanmax matches MATLAB but warns on an all-NaN input, where
    MATLAB quietly returns NaN. The isnan test also keeps the ordinary no-NaN
    case on np.max's faster path.
    """
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, -1)
    if a.dtype.kind != 'f':
        return np.max(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.max(a)
    return np.nanmax(a)


def _mmin(a):
    """MATLAB min(): the mirror of _mmax."""
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, 0)
    if a.dtype.kind != 'f':
        return np.min(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.min(a)
    return np.nanmin(a)

import warnings
import copy
from types import SimpleNamespace
from math import factorial


def _mround(x):
    """MATLAB round(): half away from zero."""
    x = float(x)
    t = int(x)                      # int() truncates toward zero
    if abs(x - t) == 0.5:           # exact tie: MATLAB goes away from zero
        return t + (1 if x > 0 else -1)
    # Off a tie round() is exact, and unlike floor(x + 0.5) it does not
    # send 0.49999999999999994 to 1: that sum is exactly 1.0 in binary.
    return int(round(x))


def _auto_port_order(sch, F, flip_victim=0):
    """Inlined auto_port_order (MATLAB lines 4932-5057, r4p15p0). Returns 1-based list."""
    sch = np.asarray(sch)
    F = np.asarray(F, dtype=float).ravel()
    MinThruEnergy = 0.1
    if sch.shape[2] != 4:
        raise ValueError('Auto Port Order routine only works for 4 port S-parameters')
    LowFreq_Matrix = np.abs(sch[0, :, :])
    Raw_LowFreq_Matrix = LowFreq_Matrix.copy()
    LowFreq_Matrix = LowFreq_Matrix - np.diag(np.diag(LowFreq_Matrix))
    max_matrix = np.maximum(np.triu(LowFreq_Matrix), np.tril(LowFreq_Matrix).T)
    LowFreq_Matrix = max_matrix + np.triu(max_matrix).T
    ConnectedPorts = np.zeros(4, dtype=int)
    for k in range(4):
        col = LowFreq_Matrix[:, k]
        idx = int(np.argmax(col))
        if col[idx] < MinThruEnergy:
            raise ValueError('Unable to determine port connections:  Low Energy')
        ConnectedPorts[k] = idx + 1
    for k in range(1, 5):
        if ConnectedPorts[ConnectedPorts[k - 1] - 1] != k:
            raise ValueError('Unable to determine port connections:  Ambiguous connections')
    port_order = [1, 0, int(ConnectedPorts[0]), 0]
    others = sorted(set(range(1, 5)) - {1, int(ConnectedPorts[0])})
    port_order[1], port_order[3] = others[0], others[-1]
    if ConnectedPorts[port_order[1] - 1] != port_order[3]:
        raise ValueError('Unable to determine port connections:  Ambiguous connections')
    try:
        TxN, RxN = port_order[1], port_order[3]
        vector1 = sch[:, TxN - 1, 0] if Raw_LowFreq_Matrix[TxN - 1, 0] > Raw_LowFreq_Matrix[0, TxN - 1] else sch[:, 0, TxN - 1]
        vector2 = sch[:, RxN - 1, 0] if Raw_LowFreq_Matrix[RxN - 1, 0] > Raw_LowFreq_Matrix[0, RxN - 1] else sch[:, 0, RxN - 1]
        Floc = F.copy()
        if Floc[0] == 0:
            vector1, vector2, Floc = vector1[1:], vector2[1:], Floc[1:]
        pd1 = -1.0 * np.unwrap(np.angle(vector1)) / (Floc * 2 * np.pi)
        pd2 = -1.0 * np.unwrap(np.angle(vector2)) / (Floc * 2 * np.pi)
        qs, tqs = _mround(len(Floc) / 4.0), _mround(len(Floc) * 3.0 / 4.0)
        m1, m2 = float(np.mean(pd1[qs - 1:tqs])), float(np.mean(pd2[qs - 1:tqs]))
        if max(m1, m2) > min(m1, m2) * 2:
            if int(np.argmax([m1, m2])) + 1 == 1:
                port_order[1], port_order[3] = port_order[3], port_order[1]
        else:
            print('Did not use phase delay in auto-port discovery since the phase delay '
                  'of Near End and Far End are similar')
    except Exception as ME_msg:
        print(str(ME_msg))
        print('Unable to use phase delay to determine port order')
    if flip_victim:
        port_order = [port_order[2], port_order[3], port_order[0], port_order[1]]
    print(f'Auto Port Order: [{" ".join(str(p) for p in port_order)}]')
    return port_order


# ---- inlined helpers (from synth_tline, combines4p, make_pkg, make_full_pkg, s21_pkg, add_brd) ---

def _synth_tline(f, Z_c, Z_0, gamma_coeff, tau, d):
    f = np.asarray(f, dtype=float)
    f_GHz = f / 1e9
    eps_val = np.finfo(float).tiny
    f_GHz_s = np.where(f_GHz == 0, eps_val, f_GHz)
    gamma_coeff = np.asarray(gamma_coeff, dtype=float).ravel()
    gamma0, a1, a2 = gamma_coeff[0], gamma_coeff[1], gamma_coeff[2]
    gamma_1 = a1 * (1.0 + 1j)
    gamma_2 = a2 * (1.0 - 2j / np.pi * np.log(f_GHz_s)) + 2j * np.pi * float(tau)
    gamma = gamma0 + gamma_1 * np.sqrt(f_GHz_s) + gamma_2 * f_GHz_s
    gamma = np.where(f_GHz == 0, gamma0, gamma)
    if float(d) == 0.0:
        rho_rl = 0.0
    else:
        rho_rl = (float(Z_c) - 2.0 * float(Z_0)) / (float(Z_c) + 2.0 * float(Z_0))
    exp_gd = np.exp(-float(d) * gamma)
    denom = 1.0 - rho_rl ** 2 * exp_gd ** 2
    s11 = rho_rl * (1.0 - exp_gd ** 2) / denom
    s21 = (1.0 - rho_rl ** 2) * exp_gd / denom
    return s11, s21, s21, s11


def _combines4p(s11_1, s12_1, s21_1, s22_1, s11_2, s12_2, s21_2, s22_2):
    N = 1.0 - s22_1 * s11_2
    s11 = s11_1 + s12_1 * s21_1 * s11_2 / N
    s12 = s12_1 * s12_2 / N
    s21 = s21_2 * s21_1 / N
    s22 = s22_2 + s12_2 * s21_2 * s22_1 / N
    return s11, s12, s21, s22


def _make_pkg(f, pkg_len, cpad, cball, pkg_z, pkg_param, lcomp=0.0, cbump=0.0):
    f = np.asarray(f, dtype=float)
    eps_val = np.finfo(float).tiny
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
        # param.PKG is a SimpleNamespace keyed by package name (see read_ParamConfigFile)
        pkg_struct = param.PKG[pkg_name] if isinstance(param.PKG, dict) else getattr(param.PKG, pkg_name)
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


def _bessel(n):
    a = np.zeros(n + 1)
    for ii in range(n + 1):
        a[ii] = factorial(2 * n - ii) / (2 ** (n - ii) * factorial(ii) * factorial(n - ii))
    return a


def _Bessel_Thomson_Filter(param, f, use_BT):
    f = np.asarray(f, dtype=float)
    if use_BT:
        a = _bessel(int(param.BTorder))
        acoef = a[::-1]
        H_bt = a[0] / np.polyval(acoef, 1j * f / (float(param.fb_BT_cutoff) * float(param.fb)))
    else:
        H_bt = np.ones(len(f))
    return H_bt


def _s21_pkg(chdata, param, OP, channel_number, mode='dd', include_die=1):
    """Inlined s21_pkg (MATLAB lines 10900-11075)."""
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

    rx_mode = 'cd' if mode.lower() == 'cd' else 'dd'
    s11in, s12in, s21in, s22in = _make_full_pkg(
        'RX', faxis, param, channel_type, rx_mode, include_die)

    tx_rd_sel = int(getattr(param, 'Tx_rd_sel', 1)) - 1
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
        warnings.warn('do not use INC_PACKAGE = 0. Instead use package parameters)')
        s21p = s21.copy()
        SCH = SimpleNamespace(
            Frequencies=faxis,
            Parameters=np.zeros((2, 2, nf), dtype=complex),
            NumPorts=2,
            Impedance=Z0 * 2)
        SCH.Parameters[0, 0, :] = s11
        SCH.Parameters[1, 1, :] = s22
        SCH.Parameters[0, 1, :] = s12
        SCH.Parameters[1, 0, :] = s21
        return s21p, SCH, sigma_ACCM_at_tp0

    if rx_cal == 1 and channel_number == 2:
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

    if not ideal_tx:
        s11, s12, s21, s22 = _combines4p(
            s11out, s12out, s21out, s22out,
            s11, s12, s21, s22)

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

    if include_die:
        s21p = (H_t * s21 * (1 - gamma_tx) * (1 + gamma_rx) /
                (1.0 - s11 * gamma_tx - s22 * gamma_rx
                 - s21 * s12 * gamma_tx * gamma_rx
                 + s11 * s22 * gamma_tx * gamma_rx))
    else:
        s21p = s21.copy()

    if mode.lower() == 'dc':
        H_bt = _Bessel_Thomson_Filter(param, faxis, 1)
        if channel_number == 1:
            ACCM_max_freq = float(getattr(param, 'ACCM_MAX_Freq', faxis[-1]))
            f_int = faxis[faxis <= ACCM_max_freq]
            nf_int = len(f_int)
            H_cc = s21out[:nf_int] * (1 - gamma_tx) / (1.0 - s11out[:nf_int] * gamma_tx) * H_bt[:nf_int]
            ac_cm_rms = float(getattr(param, 'AC_CM_RMS_TX', 0.0))
            if nf_int > 1 and f_int[-1] > 0:
                sigma_ACCM_at_tp0 = np.sqrt(
                    2 * ac_cm_rms ** 2
                    * np.sum(np.abs(H_cc[1:]) ** 2 * np.diff(f_int))
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


def _add_brd(chdata, param, OP):
    """Inlined add_brd (MATLAB lines 4768-4820)."""
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


# ---- inlined Touchstone readers (read_p4_s4params, read_p2_s2params) --------

def _read_p4_s4params_inline(infile, ports, param, OP):
    """Inlined read_p4_s4params.

    r4p15p0: empty ports triggers auto port-order detection; the resolved
    1-based order is returned as the 6th output.
    """
    import re, os

    # r4p15p0: ports may be empty -> auto-detect after read
    if ports is None:
        port_order = []
    else:
        port_order = [int(p) for p in np.asarray(ports).ravel()]
    Z_renorm = float(param.Z0)
    ext = os.path.splitext(infile)[1].lower()
    m = re.search(r'\d+', ext)
    nport = int(m.group()) if m else 4

    with open(infile, 'r', errors='replace') as fid:
        raw = fid.read()
    lines = raw.splitlines()
    option_line = None
    data_lines = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith('!'):
            continue
        if s.startswith('#') and option_line is None:
            option_line = s
        else:
            data_lines.append(s.split('!')[0].strip())

    if option_line is None:
        raise ValueError(f'No # option line in {infile}')
    opt_tokens = option_line[1:].upper().split()
    freq_scale_map = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}
    freq_mult = freq_scale_map.get(opt_tokens[0], 1e9)
    try:
        s_idx = opt_tokens.index('S'); fmt = opt_tokens[s_idx + 1]
    except (ValueError, IndexError):
        fmt = 'MA'
    try:
        r_idx = opt_tokens.index('R'); file_Z0 = float(opt_tokens[r_idx + 1])
    except (ValueError, IndexError):
        file_Z0 = 50.0

    all_tokens = []
    for line in data_lines:
        all_tokens.extend(line.split())
    vals = []
    for t in all_tokens:
        try:
            vals.append(float(t))
        except ValueError:
            pass
    vals = np.array(vals, dtype=float)
    n_per_row = 1 + nport * nport * 2
    nfreq = len(vals) // n_per_row
    data = vals[:nfreq * n_per_row].reshape(nfreq, n_per_row)
    freq = data[:, 0] * freq_mult
    ri = data[:, 1:]
    re_d, im_d = ri[:, 0::2], ri[:, 1::2]
    if fmt == 'RI':
        cdata = re_d + 1j * im_d
    elif fmt == 'MA':
        cdata = re_d * np.exp(1j * im_d * np.pi / 180)
    elif fmt == 'DB':
        cdata = (10.0 ** (re_d / 20.0)) * np.exp(1j * im_d * np.pi / 180)
    else:
        raise ValueError(f'Unsupported format {fmt}')

    sp = np.zeros((nport, nport, nfreq), dtype=complex)
    for j in range(nport):
        sp[j, :, :] = cdata[:, j * nport:(j + 1) * nport].T
    if nport == 2:
        sp[0, 1, :], sp[1, 0, :] = sp[1, 0, :].copy(), sp[0, 1, :].copy()
    if abs(file_Z0 - Z_renorm) > 1e-9:
        rho = (Z_renorm - file_Z0) / (Z_renorm + file_Z0)
        I = np.eye(nport)
        for k in range(nfreq):
            s_old = sp[:, :, k]
            sp[:, :, k] = np.linalg.solve(I - rho * s_old, s_old - rho * I)
    sch = np.transpose(sp, (2, 0, 1))
    # r4p15p0: auto-detect port order when none supplied (before range-limiting)
    if len(port_order) == 0:
        port_order = _auto_port_order(sch, freq)
    po = [p - 1 for p in port_order]
    if len(po) == nport:
        sch = sch[:, po, :][:, :, po]

    flim = float(getattr(param, 'flim', float('inf')))
    idx_lim = np.where(freq >= flim)[0]
    if len(idx_lim) > 0:
        iend = int(idx_lim[0]) + 1
        sch = sch[:iend]; freq = freq[:iend]
        limited = 1
    else:
        param.flim = float(freq[-1])
        limited = 0

    nf = len(freq)
    T = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, -1.0]])
    T_inv = np.linalg.inv(T)
    Txpskew = float(getattr(param, 'Txpskew', 0.0))
    Txnskew = float(getattr(param, 'Txnskew', 0.0))
    Rxpskew = float(getattr(param, 'Rxpskew', 0.0))
    Rxnskew = float(getattr(param, 'Rxnskew', 0.0))
    D = np.zeros((nf, 4, 4), dtype=complex)
    for i in range(nf):
        f = freq[i]
        s1 = np.exp(2j * np.pi * f * Txpskew * 1e-12)
        s2 = np.exp(2j * np.pi * f * Txnskew * 1e-12)
        s3 = np.exp(2j * np.pi * f * Rxpskew * 1e-12)
        s4 = np.exp(2j * np.pi * f * Rxnskew * 1e-12)
        sigma = np.array([[s1**2, s1*s2, s1*s3, s1*s4], [s1*s2, s2**2, s2*s3, s2*s4],
                           [s1*s3, s2*s3, s3**2, s3*s4], [s1*s4, s2*s4, s3*s4, s4**2]])
        D[i] = T @ (sigma * sch[i]) @ T_inv

    SDD = np.zeros((nf, 2, 2), dtype=complex)
    SDD[:, 0, 0] = D[:, 1, 1]; SDD[:, 1, 1] = D[:, 3, 3]
    SDD[:, 0, 1] = D[:, 1, 3]; SDD[:, 1, 0] = D[:, 3, 1]
    SDC = np.zeros((nf, 2, 2), dtype=complex)
    SDC[:, 0, 0] = D[:, 1, 0]; SDC[:, 1, 1] = D[:, 3, 2]
    SDC[:, 0, 1] = D[:, 1, 2]; SDC[:, 1, 0] = D[:, 3, 0]
    SCC = np.zeros((nf, 2, 2), dtype=complex)
    SCC[:, 0, 0] = D[:, 0, 0]; SCC[:, 1, 1] = D[:, 2, 2]
    SCC[:, 0, 1] = D[:, 0, 2]; SCC[:, 1, 0] = D[:, 2, 0]
    SCD = np.zeros((nf, 2, 2), dtype=complex)
    SCD[:, 0, 0] = D[:, 0, 1]; SCD[:, 1, 1] = D[:, 2, 3]
    SCD[:, 0, 1] = D[:, 0, 3]; SCD[:, 1, 0] = D[:, 2, 1]

    Sch = SimpleNamespace(freq=freq, m=sch, flim=getattr(param, 'flim', freq[-1]), limited=limited)
    return Sch, SDD, SDC, SCC, SCD, port_order


def _read_p2_s2params_inline(infile, ports, param, OP):
    """Inlined read_p2_s2params."""
    import re, os
    Z_renorm = float(param.Z0)
    with open(infile, 'r', errors='replace') as fid:
        raw = fid.read()
    lines = raw.splitlines()
    option_line = None
    data_lines = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith('!'):
            continue
        if s.startswith('#') and option_line is None:
            option_line = s
        else:
            data_lines.append(s.split('!')[0].strip())
    if option_line is None:
        raise ValueError(f'No # option line in {infile}')
    opt_tokens = option_line[1:].upper().split()
    freq_scale_map = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}
    freq_mult = freq_scale_map.get(opt_tokens[0], 1e9)
    try:
        s_idx = opt_tokens.index('S'); fmt = opt_tokens[s_idx + 1]
    except (ValueError, IndexError):
        fmt = 'MA'
    try:
        r_idx = opt_tokens.index('R'); file_Z0 = float(opt_tokens[r_idx + 1])
    except (ValueError, IndexError):
        file_Z0 = 50.0

    all_tokens = []
    for line in data_lines:
        all_tokens.extend(line.split())
    vals = []
    for t in all_tokens:
        try:
            vals.append(float(t))
        except ValueError:
            pass
    vals = np.array(vals, dtype=float)
    n_per_row = 1 + 2 * 2 * 2
    nfreq = len(vals) // n_per_row
    data = vals[:nfreq * n_per_row].reshape(nfreq, n_per_row)
    freq = data[:, 0] * freq_mult
    ri = data[:, 1:]
    re_d, im_d = ri[:, 0::2], ri[:, 1::2]
    if fmt == 'RI':
        cdata = re_d + 1j * im_d
    elif fmt == 'MA':
        cdata = re_d * np.exp(1j * im_d * np.pi / 180)
    elif fmt == 'DB':
        cdata = (10.0 ** (re_d / 20.0)) * np.exp(1j * im_d * np.pi / 180)
    else:
        raise ValueError(f'Unsupported format {fmt}')
    sp = np.zeros((2, 2, nfreq), dtype=complex)
    for j in range(2):
        sp[j, :, :] = cdata[:, j * 2:(j + 1) * 2].T
    sp[0, 1, :], sp[1, 0, :] = sp[1, 0, :].copy(), sp[0, 1, :].copy()
    if abs(file_Z0 - Z_renorm) > 1e-9:
        rho = (Z_renorm - file_Z0) / (Z_renorm + file_Z0)
        I2 = np.eye(2)
        for k in range(nfreq):
            s_old = sp[:, :, k]
            sp[:, :, k] = np.linalg.solve(I2 - rho * s_old, s_old - rho * I2)
    sch = np.transpose(sp, (2, 0, 1))
    flim = float(getattr(param, 'flim', float('inf')))
    idx_lim = np.where(freq >= flim)[0]
    if len(idx_lim) > 0:
        iend = int(idx_lim[0]) + 1
        sch = sch[:iend]; freq = freq[:iend]; limited = 1
    else:
        param.flim = float(freq[-1]); limited = 0
    nf = len(freq)
    T = np.array([[1.0, 1.0], [1.0, -1.0]])
    T_inv = np.linalg.inv(T)
    D = np.zeros((nf, 2, 2), dtype=complex)
    for i in range(nf):
        D[i] = T @ sch[i] @ T_inv
    SDD = np.zeros((nf, 1, 1), dtype=complex); SDD[:, 0, 0] = D[:, 1, 1]
    SDC = np.zeros((nf, 1, 1), dtype=complex); SDC[:, 0, 0] = D[:, 1, 0]
    SCC = np.zeros((nf, 1, 1), dtype=complex); SCC[:, 0, 0] = D[:, 0, 0]
    SCD = np.zeros((nf, 1, 1), dtype=complex); SCD[:, 0, 0] = D[:, 0, 1]
    Sch = SimpleNamespace(freq=freq, m=sch, flim=getattr(param, 'flim', freq[-1]), limited=limited)
    return Sch, SDD, SDC, SCC, SCD


# ---- main function ----------------------------------------------------------

def read_s4p_files(param, OP, chdata):
    """Read S4P/S2P files into chdata (MATLAB lines 10594-10767).

    Returns (chdata, SDDch, SDDp2p, param).
    SDDch: (nfreq, 2, 2) complex differential S-matrix for last file read.
    SDDp2p: list of NaN for s2p files, SDDch-shaped array for s4p.
    param: returned so an auto-detected snpPortsOrder propagates to the caller (r4p15p0).
    """
    num_files = len(chdata)
    SDDch = None
    SDDp2p = [float('nan')] * num_files

    for i in range(num_files):
        ch = chdata[i]
        if not (hasattr(OP, 'DISPLAY_WINDOW') and OP.DISPLAY_WINDOW):
            print(f'{i + 1} ', end='', flush=True)

        faxis = getattr(ch, 'faxis', None)
        if faxis is None or len(np.asarray(faxis)) == 0:
            ext = str(ch.ext).lower()
            ports = list(getattr(param, 'snpPortsOrder', [1, 3, 2, 4]))

            if ext == '.s2p':
                SDDp2p[i] = float('nan')
                Sch, SDDch, SDCch, SCCch, SCDch = _read_p2_s2params_inline(
                    ch.filename, [1, 2], param, OP)
                nf = len(Sch.freq)
                ch.fmaxi = nf
                ch.faxis = Sch.freq
                ch.sdd11_raw = SDDch[:nf, 0, 0]
                ch.sdc11_raw = SDCch[:nf, 0, 0]
                ch.scd11_raw = SCDch[:nf, 0, 0]
                ch.scc11_raw = SCCch[:nf, 0, 0]
                ch.sdd11_orig = ch.sdd11_raw.copy()
                ch.sdc11_orig = ch.sdc11_raw.copy()
                ch.scd11_orig = ch.scd11_raw.copy()
                ch.scc11_orig = ch.scc11_raw.copy()
                ch.sdd11 = ch.sdd11_raw.copy()
                SDDch[:nf, 0, 0] = ch.sdd11_raw

            elif ext == '.s4p':
                # r4p15p0: port_order must be empty (auto) or length 4
                if len(ports) != 0 and len(ports) != 4:
                    raise ValueError(
                        f'The number of ports defined ({len(ports)}) does not match '
                        f'the sNp file type (.s4p)')
                if int(getattr(param, 'package_testcase_i', 1)) == 1:
                    Sch, SDDch, SDCch, SCCch, SCDch, resolved_ports = _read_p4_s4params_inline(
                        ch.filename, ports, param, OP)
                    # r4p15p0: if order was auto-detected (empty input), persist it to param
                    # so crosstalk files reuse the THRU-determined order.
                    if len(ports) == 0:
                        param.snpPortsOrder = resolved_ports
                else:
                    raise RuntimeError('read_s4p_files: logic error for package_testcase_i != 1')

                nf = len(Sch.freq)
                ch.fmaxi = nf
                ch.faxis = Sch.freq

                for p, q in [(0, 1), (1, 0), (0, 0), (1, 1)]:
                    tag = f'sdd{p+1}{q+1}'
                    setattr(ch, f'{tag}_raw', SDDch[:nf, p, q])
                    setattr(ch, f'{tag}_orig', SDDch[:nf, p, q].copy())
                for p, q in [(0, 1), (1, 0), (0, 0), (1, 1)]:
                    tag = f'sdc{p+1}{q+1}'
                    setattr(ch, f'{tag}_raw', SDCch[:nf, p, q])
                    setattr(ch, f'{tag}_orig', SDCch[:nf, p, q].copy())
                for p, q in [(0, 1), (1, 0), (0, 0), (1, 1)]:
                    tag = f'scd{p+1}{q+1}'
                    setattr(ch, f'{tag}_raw', SCDch[:nf, p, q])
                    setattr(ch, f'{tag}_orig', SCDch[:nf, p, q].copy())
                for p, q in [(0, 1), (1, 0), (0, 0), (1, 1)]:
                    tag = f'scc{p+1}{q+1}'
                    setattr(ch, f'{tag}_raw', SCCch[:nf, p, q])
                    setattr(ch, f'{tag}_orig', SCCch[:nf, p, q].copy())

                if getattr(OP, 'include_pcb', 0):
                    (ch.sdd11_raw, ch.sdd12_raw,
                     ch.sdd21_raw, ch.sdd22_raw) = _add_brd(ch, param, OP)

                ch.sdd11 = ch.sdd11_raw.copy()
                ch.sdd22 = ch.sdd22_raw.copy()
            else:
                raise ValueError(f'read_s4p_files: unsupported extension {ext!r} in {ch.filename!r}')

            freq = ch.faxis
            fb = float(param.fb)
            if freq[-1] < fb and not getattr(OP, 'ZERO_PAD', False):
                print(f' INFO: In {ch.filename}: max freq {freq[-1]/1e9:.3g} GHz < fb {fb/1e9:.3g} GHz')
            max_start_freq = float(getattr(param, 'max_start_freq', 0.1e9))
            if freq[0] > max_start_freq:
                print(f'INFO: In {ch.filename}: min freq {freq[0]/1e9:.2g} GHz > recommended {max_start_freq/1e9:.2g} GHz')
            if len(freq) > 1:
                freqstep = np.diff(freq)
                if max(freqstep) - min(freqstep) > 1:
                    warnings.warn(f'In {ch.filename}: non-uniform frequency steps', stacklevel=2)

            if i > 0 and chdata[0].faxis is not None:
                fax0 = np.asarray(chdata[0].faxis)
                if len(ch.faxis) != len(fax0):
                    raise ValueError(f'Crosstalk file {ch.filename!r} has different number of frequency points')
                if _mmax(np.abs(ch.faxis - fax0)) > 1:
                    raise ValueError(f'Crosstalk file {ch.filename!r} has a different frequency axis')
        else:
            SDDch = np.zeros((len(ch.faxis), 2, 2), dtype=complex)
            SDDch[:, 0, 1] = ch.sdd12_raw
            SDDch[:, 1, 0] = ch.sdd21_raw
            SDDch[:, 0, 0] = ch.sdd11_raw
            SDDch[:, 1, 1] = ch.sdd22_raw

        ch.sigma_ACCM_at_tp0 = 0.0

        flag_s2p = getattr(param, 'FLAG', SimpleNamespace()).S2P if hasattr(param, 'FLAG') else False
        if not flag_s2p:
            inc_pkg = int(getattr(OP, 'INC_PACKAGE', 1))
            rx_cal = int(getattr(OP, 'RX_CALIBRATION', 0))

            if inc_pkg != 0 or (rx_cal == 1 and i == 0):
                if rx_cal == 1 and i == 1:
                    ch.sdd21 = ch.sdd21_raw.copy()
                else:
                    s21p_val, SDDp2p_i, _ = _s21_pkg(ch, param, OP, i + 1)
                    ch.sdd21p = s21p_val
                    SDDp2p[i] = SDDp2p_i
                    s21p_nodie, _, _ = _s21_pkg(ch, param, OP, i + 1, 'dd', 0)
                    ch.sdd21p_nodie = s21p_nodie
                    ch.sdd21 = ch.sdd21p.copy()
                    try:
                        sdc21p, _, sigma_ac = _s21_pkg(ch, param, OP, i + 1, 'cd')
                        ch.sdc21p = sdc21p
                        ch.sdc21 = sdc21p.copy()
                        ch.sigma_ACCM_at_tp0 = sigma_ac
                    except Exception:
                        pass
            else:
                ch.sdd21 = ch.sdd21_raw.copy()

            ch.sdd21f = ch.sdd21_orig.copy()

    if not (hasattr(OP, 'DISPLAY_WINDOW') and OP.DISPLAY_WINDOW):
        print()

    return chdata, SDDch, SDDp2p, param
