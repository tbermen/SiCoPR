# ============================================================
# MATLAB→Python translation notes for make_full_pkg
# MATLAB lines: 8166–8358
# ============================================================
# Builds 2-port S-param model for TX or RX package.
# Inlines: synth_tline (11292), make_pkg (8359), combines4p (5327).
#
# Layout (per block):
#   Cpad → [Lcomp] → [Cbump] → Tline → Cball
#
# num_blocks determined by mele (columns of z_p_next_cases):
#   mele=1 → 1 block; mele=4 → 4 blocks
#   If C_diepad is 2D matrix: num_blocks = mele + len(Cd_Tx)-1
#
# Mode 'dc'/'cd': Z0/=2, Cpad*=2, Cball*=2, Zpkg*=2, Lcomp/=2, Cbump*=2
#
# PKG_NAME: if set, swaps pkg_gamma0_a1_a2 and pkg_tau from named struct.
# ============================================================

import numpy as np
import copy
from types import SimpleNamespace


def _synth_tline(f, Z_c, Z_0, gamma_coeff, tau, d):
    """Inlined synth_tline (MATLAB lines 11292-11316)."""
    f = np.asarray(f, dtype=complex)
    f_r = f.real
    with np.errstate(divide='ignore', invalid='ignore'):
        f_GHz = f_r / 1e9
    eps = np.finfo(float).tiny
    f_GHz_safe = np.where(f_GHz == 0, eps, f_GHz)

    gamma_coeff = np.asarray(gamma_coeff, dtype=float).ravel()
    gamma0, a1, a2 = gamma_coeff[0], gamma_coeff[1], gamma_coeff[2]

    gamma_1 = a1 * (1.0 + 1j)
    gamma_2 = a2 * (1.0 - 2j / np.pi * np.log(f_GHz_safe)) + 2j * np.pi * float(tau)
    gamma = gamma0 + gamma_1 * np.sqrt(f_GHz_safe) + gamma_2 * f_GHz_safe
    gamma = np.where(f_GHz == 0, gamma0, gamma)

    if float(d) == 0.0:
        rho_rl = 0.0
    else:
        rho_rl = (Z_c - 2.0 * Z_0) / (Z_c + 2.0 * Z_0)

    exp_gd = np.exp(-float(d) * gamma)
    denom = 1.0 - rho_rl ** 2 * exp_gd ** 2
    s11 = rho_rl * (1.0 - exp_gd ** 2) / denom
    s21 = (1.0 - rho_rl ** 2) * exp_gd / denom
    return s11, s21, s21, s11  # s11, s12, s21, s22


def _combines4p(s11_1, s12_1, s21_1, s22_1, s11_2, s12_2, s21_2, s22_2):
    """Inlined combines4p (MATLAB lines 5327-5370)."""
    N = 1.0 - s22_1 * s11_2
    s11 = s11_1 + s12_1 * s21_1 * s11_2 / N
    s12 = s12_1 * s12_2 / N
    s21 = s21_2 * s21_1 / N
    s22 = s22_2 + s12_2 * s21_2 * s22_1 / N
    return s11, s12, s21, s22


def _make_pkg(f, pkg_len, cpad, cball, pkg_z, pkg_param, lcomp=0.0, cbump=0.0):
    """Inlined make_pkg (MATLAB lines 8359-8405)."""
    f = np.asarray(f, dtype=float)
    eps_val = np.finfo(float).tiny
    f = np.where(f < eps_val, eps_val, f)
    zref = float(pkg_param.Z0)
    tau = float(pkg_param.pkg_tau)
    gamma_coeff = np.asarray(pkg_param.pkg_gamma0_a1_a2, dtype=float).ravel()

    # Cpad shunt: Eq 93A-8
    jw_cpad_z = 1j * 2 * np.pi * f * float(cpad) * zref
    s11pad = -jw_cpad_z / (2.0 + jw_cpad_z)
    s21pad = 2.0 / (2.0 + jw_cpad_z)
    s12pad = s21pad.copy()
    s22pad = s11pad.copy()

    # Optional Lcomp series
    if float(lcomp) > 0.0:
        jw_lcomp_z = 1j * 2 * np.pi * f * float(lcomp) / zref
        s11comp = jw_lcomp_z / (2.0 + jw_lcomp_z)
        s21comp = 2.0 / (2.0 + jw_lcomp_z)
        s11pad, s12pad, s21pad, s22pad = _combines4p(
            s11pad, s12pad, s21pad, s22pad,
            s11comp, s21comp, s21comp, s11comp)

    # Optional Cbump shunt
    if float(cbump) > 0.0:
        jw_cbump_z = 1j * 2 * np.pi * f * float(cbump) * zref
        s11bump = -jw_cbump_z / (2.0 + jw_cbump_z)
        s21bump = 2.0 / (2.0 + jw_cbump_z)
        s11pad, s12pad, s21pad, s22pad = _combines4p(
            s11pad, s12pad, s21pad, s22pad,
            s11bump, s21bump, s21bump, s11bump)

    # Transmission line: Eqs 93A-9 to 93A-14
    S11, S12, S21, S22 = _synth_tline(f, float(pkg_z), zref, gamma_coeff, tau, float(pkg_len))
    s11out1, s12out1, s21out1, s22out1 = _combines4p(
        s11pad, s12pad, s21pad, s22pad,
        S11, S12, S21, S22)

    # Cball shunt: Eq 93A-8
    jw_cball_z = 1j * 2 * np.pi * f * float(cball) * zref
    s11ball = -jw_cball_z / (2.0 + jw_cball_z)
    s21ball = 2.0 / (2.0 + jw_cball_z)
    s11out, s12out, s21out, s22out = _combines4p(
        s11out1, s12out1, s21out1, s22out1,
        s11ball, s21ball, s21ball, s11ball)

    return s11out, s12out, s21out, s22out


def make_full_pkg(type_, faxis, param, channel_type, mode='dd', include_die=1):
    """Build TX or RX package S-params (MATLAB lines 8166-8358).

    type_: 'TX' or 'RX'
    faxis: frequency array (Hz)
    param: parameter namespace
    channel_type: 'THRU', 'FEXT', 'NEXT', or 'NOISE'
    mode: 'dd' (default), 'dc', or 'cd'
    include_die: if 0, zero out Cpad/Lcomp/Cbump

    Returns (s11out, s12out, s21out, s22out), each shape (nfreq,).
    """
    faxis = np.asarray(faxis, dtype=float)

    # Make a local copy of param so modifications don't persist
    pkg_param = copy.copy(param)

    # PKG_NAME swap: if named packages defined, swap gamma/tau for TX vs RX
    pkg_name_list = getattr(param, 'PKG_NAME', None)
    if pkg_name_list is not None and len(pkg_name_list) > 0:
        swap_fields = ['pkg_gamma0_a1_a2', 'pkg_tau']
        type_upper = str(type_).upper()
        if type_upper == 'TX':
            pkg_name = pkg_name_list[0]
        elif type_upper == 'RX':
            pkg_name = pkg_name_list[1]
        else:
            raise ValueError(f'make_full_pkg: type must be TX or RX, got {type_}')
        # param.PKG is a SimpleNamespace keyed by package name (see read_ParamConfigFile)
        pkg_struct = param.PKG[pkg_name] if isinstance(param.PKG, dict) else getattr(param.PKG, pkg_name)
        for field in swap_fields:
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

    # Determine vector vs matrix C_diepad/L_comp
    is_vector_cd = (C_diepad.ndim == 1) or (C_diepad.ndim == 2 and min(C_diepad.shape) == 1)
    C_diepad_flat = C_diepad.ravel()
    L_comp_flat = L_comp.ravel()

    if is_vector_cd:
        Cd_Tx = C_diepad_flat[0] if len(C_diepad_flat) > 0 else 0.0
        Cd_Rx = C_diepad_flat[1] if len(C_diepad_flat) > 1 else 0.0
        Lcomp_Tx = L_comp_flat[0] if len(L_comp_flat) > 0 else 0.0
        Lcomp_Rx = L_comp_flat[1] if len(L_comp_flat) > 1 else 0.0
        num_blocks = mele
        extra_LC = 0
    else:
        # 2D matrix: row 0 = TX, row 1 = RX
        C_diepad_2d = C_diepad.reshape(2, -1)
        L_comp_2d = L_comp.reshape(2, -1)
        Cd_Tx = C_diepad_2d[0, :]
        Cd_Rx = C_diepad_2d[1, :]
        Lcomp_Tx = L_comp_2d[0, :]
        Lcomp_Rx = L_comp_2d[1, :]
        extra_LC = len(Cd_Tx) - 1
        num_blocks = mele + extra_LC

    insert_zeros = np.zeros(extra_LC)

    type_upper = str(type_).upper()
    if type_upper == 'TX':
        if mele == 1:
            Cpad = np.array([float(Cd_Tx) if np.isscalar(Cd_Tx) else float(Cd_Tx.ravel()[0])])
            Lcomp = np.array([float(Lcomp_Tx) if np.isscalar(Lcomp_Tx) else float(Lcomp_Tx.ravel()[0])])
            Cbump = np.array([float(C_bump[0])])
            Cball = np.array([float(C_pkg_board[0])])
            pkg_Z_c = np.asarray(param.pkg_Z_c, dtype=float).ravel()
            Zpkg = np.array([float(pkg_Z_c[0])])
        elif mele == 4:
            cd_val = float(Cd_Tx) if np.isscalar(Cd_Tx) else float(np.asarray(Cd_Tx).ravel()[0])
            lc_val = float(Lcomp_Tx) if np.isscalar(Lcomp_Tx) else float(np.asarray(Lcomp_Tx).ravel()[0])
            # MATLAB: Cpad=[Cd_Tx 0 0 0]; Lcomp=[L_comp_Tx 0 0 0]  (L8390-8391).
            # Cd_Tx/L_comp_Tx are ROW VECTORS when C_d/L_comp are given as a
            # 2xN matrix (N die LC sections per side), so MATLAB's horizontal
            # concatenation yields len(Cd_Tx)+3 entries — matching
            # num_blocks = mele + extra_LC. Taking only Cd_Tx[0] dropped every
            # die section after the first and produced a 4-entry array, losing
            # the die LC delay (~15 ps here) and reshaping the pulse.
            Cpad = np.concatenate([np.atleast_1d(np.asarray(Cd_Tx, dtype=float)).ravel(),
                                   np.zeros(3)])
            Lcomp = np.concatenate([np.atleast_1d(np.asarray(Lcomp_Tx, dtype=float)).ravel(),
                                    np.zeros(3)])
            Cbump = np.array([float(C_bump[0]), 0.0, 0.0, 0.0])
            C_v = np.asarray(param.C_v, dtype=float).ravel()
            Cball = np.array([0.0, 0.0, float(C_v[0]), float(C_pkg_board[0])])
            pkg_Z_c = np.asarray(param.pkg_Z_c, dtype=float)
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
        elif ch_upper == 'NOISE':
            raise ValueError('make_full_pkg: TX pkg should not be used for NOISE channels')
        else:
            Len = np.asarray(param.Pkg_len_TX, dtype=float).ravel()

    elif type_upper == 'RX':
        if mele == 1:
            Cpad = np.array([float(Cd_Rx) if np.isscalar(Cd_Rx) else float(Cd_Rx.ravel()[0])])
            Lcomp = np.array([float(Lcomp_Rx) if np.isscalar(Lcomp_Rx) else float(Lcomp_Rx.ravel()[0])])
            Cbump = np.array([float(C_bump[1]) if len(C_bump) > 1 else float(C_bump[0])])
            Cball = np.array([float(C_pkg_board[1]) if len(C_pkg_board) > 1 else float(C_pkg_board[0])])
            pkg_Z_c = np.asarray(param.pkg_Z_c, dtype=float).ravel()
            Zpkg = np.array([float(pkg_Z_c[1]) if len(pkg_Z_c) > 1 else float(pkg_Z_c[0])])
        elif mele == 4:
            cd_val = float(Cd_Rx) if np.isscalar(Cd_Rx) else float(np.asarray(Cd_Rx).ravel()[0])
            lc_val = float(Lcomp_Rx) if np.isscalar(Lcomp_Rx) else float(np.asarray(Lcomp_Rx).ravel()[0])
            # MATLAB: Cpad=[Cd_Rx 0 0 0]; Lcomp=[L_comp_Rx 0 0 0]  (L8390-8391).
            # Cd_Rx/L_comp_Rx are ROW VECTORS when C_d/L_comp are given as a
            # 2xN matrix (N die LC sections per side), so MATLAB's horizontal
            # concatenation yields len(Cd_Rx)+3 entries — matching
            # num_blocks = mele + extra_LC. Taking only Cd_Rx[0] dropped every
            # die section after the first and produced a 4-entry array, losing
            # the die LC delay (~15 ps here) and reshaping the pulse.
            Cpad = np.concatenate([np.atleast_1d(np.asarray(Cd_Rx, dtype=float)).ravel(),
                                   np.zeros(3)])
            Lcomp = np.concatenate([np.atleast_1d(np.asarray(Lcomp_Rx, dtype=float)).ravel(),
                                    np.zeros(3)])
            cb_val = float(C_bump[1]) if len(C_bump) > 1 else float(C_bump[0])
            Cbump = np.array([cb_val, 0.0, 0.0, 0.0])
            C_v = np.asarray(param.C_v, dtype=float).ravel()
            cb_pkg = float(C_pkg_board[1]) if len(C_pkg_board) > 1 else float(C_pkg_board[0])
            Cball = np.array([0.0, 0.0, float(C_v[1]) if len(C_v) > 1 else float(C_v[0]), cb_pkg])
            pkg_Z_c = np.asarray(param.pkg_Z_c, dtype=float)
            if pkg_Z_c.ndim == 1:
                Zpkg = pkg_Z_c[:4] if len(pkg_Z_c) >= 4 else np.pad(pkg_Z_c, (0, 4 - len(pkg_Z_c)))
            else:
                Zpkg = pkg_Z_c[1, :4] if pkg_Z_c.shape[0] > 1 else pkg_Z_c[0, :4]
        else:
            raise ValueError(f'make_full_pkg: unsupported mele={mele}')

        Len = np.asarray(param.Pkg_len_RX, dtype=float).ravel()
    else:
        raise ValueError(f'make_full_pkg: type must be TX or RX, got {type_}')

    # Prepend insert_zeros
    Cball = np.concatenate([insert_zeros, Cball.ravel()])
    Cbump = np.concatenate([insert_zeros, Cbump.ravel()])
    Len = np.concatenate([insert_zeros, Len.ravel()])
    Zpkg = np.concatenate([insert_zeros, Zpkg.ravel()])

    # Mode 'dc' or 'cd': scale parameters for common-mode
    if mode.lower() in ('dc', 'cd'):
        pkg_param.Z0 = float(param.Z0) / 2.0
        Cpad = Cpad * 2.0
        Cball = Cball * 2.0
        Zpkg = Zpkg * 2.0
        Lcomp = Lcomp / 2.0
        Cbump = Cbump * 2.0

    # Build and cascade blocks
    n_blocks = int(num_blocks)
    # Ensure arrays are long enough
    def _el(arr, j):
        arr = np.asarray(arr, dtype=float).ravel()
        return float(arr[j]) if j < len(arr) else 0.0

    if n_blocks == 1:
        s11out, s12out, s21out, s22out = _make_pkg(
            faxis, _el(Len, 0), _el(Cpad, 0), _el(Cball, 0), _el(Zpkg, 0),
            pkg_param, _el(Lcomp, 0), _el(Cbump, 0))
    else:
        for j in range(n_blocks):
            sp11, sp12, sp21, sp22 = _make_pkg(
                faxis, _el(Len, j), _el(Cpad, j), _el(Cball, j), _el(Zpkg, j),
                pkg_param, _el(Lcomp, j), _el(Cbump, j))
            if j == 0:
                s11out, s12out, s21out, s22out = sp11, sp12, sp21, sp22
            else:
                s11out, s12out, s21out, s22out = _combines4p(
                    s11out, s12out, s21out, s22out,
                    sp11, sp12, sp21, sp22)

    return s11out, s12out, s21out, s22out
