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
from com_functions.fn.combines4p.py_impl import combines4p as _combines4p
from com_functions.fn.synth_tline.py_impl import synth_tline as _synth_tline
import copy
from types import SimpleNamespace


def _lin(arr, k, name):
    """MATLAB `A(k)`: linear indexing is COLUMN-major, and out of range errors.

    COM Octave, make_full_pkg('RX', ...) with mele=1 and the shipped 2x4
    param.pkg_Z_c = [87.5 92.5 90 95; 88 93 91 96] (the workbook stores
    package_Z_c as cases-by-[Tx Rx] and transposes it, so it is 2-by-mele):
    pkg_Z_c(2) is element (2,1) = 88, not the row-major (1,2) = 92.5 that
    numpy's .ravel()[1] returns.  s21 at 1 GHz was
    0.78218388281501161-0.57074890734687478j from the reference against
    0.78497930629192492-0.57106813157164604j from the port.
    COM Octave, C_bump=[1.5e-13] for 'RX': "error: C_bump(2): out of bound 1
    (dimensions are 1x1)" — the reference refuses, it does not fall back.
    """
    a = np.asarray(arr, dtype=float).ravel(order='F')
    if k >= a.size:
        raise IndexError('make_full_pkg: %s(%d): out of bound %d'
                         % (name, k + 1, a.size))
    return float(a[k])


def _row(arr, i, name):
    """MATLAB `A(i,:)`: a whole row, and a row that is not there errors.

    COM Octave, make_full_pkg('RX', ...) with mele=4 and a 1-D pkg_Z_c:
    "error: param(2,_): out of bound 1 (dimensions are 1x4)".
    """
    a = np.atleast_2d(np.asarray(arr, dtype=float))
    if i >= a.shape[0]:
        raise IndexError('make_full_pkg: %s(%d,:): out of bound %d'
                         % (name, i + 1, a.shape[0]))
    return a[i, :]


def _make_pkg(f, pkg_len, cpad, cball, pkg_z, pkg_param, lcomp=0.0, cbump=0.0):
    """Inlined make_pkg (MATLAB lines 8359-8405)."""
    f = np.asarray(f, dtype=float)
    # MATLAB `f(f<eps)=eps` is eps(1) = 2.220446049250313e-16, not the
    # smallest positive double. np.finfo(float).tiny is 292 orders out,
    # and it is the DC point that gets it, which then goes into
    # synth_tline's sqrt and log. Matches com_functions/fn/make_pkg.
    eps_val = np.finfo(float).eps
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
    C_pkg_board = np.asarray(param.C_pkg_board, dtype=float)
    L_comp = np.asarray(param.L_comp, dtype=float)
    C_bump = np.asarray(param.C_bump, dtype=float)

    if not include_die:
        C_diepad = C_diepad * 0
        L_comp = L_comp * 0
        C_bump = C_bump * 0

    z_p = np.atleast_2d(np.asarray(param.z_p_next_cases))
    ncases, mele = z_p.shape

    # Determine vector vs matrix C_diepad/L_comp
    is_vector_cd = (C_diepad.ndim == 1) or (C_diepad.ndim == 2 and min(C_diepad.shape) == 1)

    if is_vector_cd:
        # MATLAB C_diepad(1)/(2), L_comp(1)/(2): a 1-element parameter is
        # "error: C_diepad(2): out of bound 1", not a silent zero.
        Cd_Tx = _lin(C_diepad, 0, 'C_diepad')
        Cd_Rx = _lin(C_diepad, 1, 'C_diepad')
        Lcomp_Tx = _lin(L_comp, 0, 'L_comp')
        Lcomp_Rx = _lin(L_comp, 1, 'L_comp')
        num_blocks = mele
        extra_LC = 0
    else:
        # 2D matrix: row 0 = TX, row 1 = RX
        Cd_Tx = _row(C_diepad, 0, 'C_diepad')
        Cd_Rx = _row(C_diepad, 1, 'C_diepad')
        Lcomp_Tx = _row(L_comp, 0, 'L_comp')
        Lcomp_Rx = _row(L_comp, 1, 'L_comp')
        extra_LC = len(Cd_Tx) - 1
        num_blocks = mele + extra_LC

    insert_zeros = np.zeros(extra_LC)

    # MATLAB `switch type / case 'TX'` is case-SENSITIVE, unlike the strcmpi
    # above.  COM Octave, make_full_pkg('Tx', ...): the switch matches nothing,
    # Cball is never assigned and it fails with "error: 'Cball' undefined".
    type_str = str(type_)
    if type_str == 'TX':
        if mele == 1:
            # MATLAB L8389-8390: Cpad=Cd_Tx; Lcomp=L_comp_Tx -- the WHOLE row
            # when C_diepad/L_comp are given as a 2xN matrix of die LC
            # sections. Taking only the first entry dropped every section
            # after it, which is the same truncation already fixed in the
            # mele==4 branch below. Unreachable on the shipped workbooks,
            # which set mele=4 (z_p_next_cases is 4x4) and zero die values.
            Cpad = np.atleast_1d(np.asarray(Cd_Tx, dtype=float)).ravel()
            Lcomp = np.atleast_1d(np.asarray(Lcomp_Tx, dtype=float)).ravel()
            Cbump = np.array([_lin(C_bump, 0, 'C_bump')])
            Cball = np.array([_lin(C_pkg_board, 0, 'C_pkg_board')])
            Zpkg = np.array([_lin(param.pkg_Z_c, 0, 'pkg_Z_c')])
        elif mele == 4:
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
            Cbump = np.array([_lin(C_bump, 0, 'C_bump'), 0.0, 0.0, 0.0])
            Cball = np.array([0.0, 0.0, _lin(param.C_v, 0, 'C_v'),
                              _lin(C_pkg_board, 0, 'C_pkg_board')])
            Zpkg = _row(param.pkg_Z_c, 0, 'pkg_Z_c')
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
            # MATLAB's channel_type switch has no otherwise, so Len is never
            # assigned.  COM Octave, channel_type='BOGUS': "error: 'Len'
            # undefined near line 145" — it does not fall back to Pkg_len_TX.
            raise ValueError(
                f'make_full_pkg: unsupported channel_type={channel_type}')

    elif type_str == 'RX':
        if mele == 1:
            # MATLAB L8416-8417: the whole row, as for TX above.
            Cpad = np.atleast_1d(np.asarray(Cd_Rx, dtype=float)).ravel()
            Lcomp = np.atleast_1d(np.asarray(Lcomp_Rx, dtype=float)).ravel()
            Cbump = np.array([_lin(C_bump, 1, 'C_bump')])
            Cball = np.array([_lin(C_pkg_board, 1, 'C_pkg_board')])
            Zpkg = np.array([_lin(param.pkg_Z_c, 1, 'pkg_Z_c')])
        elif mele == 4:
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
            Cbump = np.array([_lin(C_bump, 1, 'C_bump'), 0.0, 0.0, 0.0])
            Cball = np.array([0.0, 0.0, _lin(param.C_v, 1, 'C_v'),
                              _lin(C_pkg_board, 1, 'C_pkg_board')])
            Zpkg = _row(param.pkg_Z_c, 1, 'pkg_Z_c')
        else:
            raise ValueError(f'make_full_pkg: unsupported mele={mele}')

        # MATLAB's RX switch covers THRU/NEXT/FEXT/NOISE, all of them
        # Pkg_len_RX, and has no otherwise.  COM Octave, channel_type='BOGUS'
        # on 'RX': "error: 'Len' undefined near line 145".
        if str(channel_type).upper() not in ('THRU', 'NEXT', 'FEXT', 'NOISE'):
            raise ValueError(
                f'make_full_pkg: unsupported channel_type={channel_type}')
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

    def _el(arr, j, name):
        # MATLAB indexes Len(j), Cpad(j), ... straight; a vector shorter than
        # num_blocks is an error, not an implicit zero.  COM Octave, mele=4
        # with a scalar param.Pkg_len_TX: "error: Len(2): out of bound 1
        # (dimensions are 1x1)".
        arr = np.asarray(arr, dtype=float).ravel()
        if j >= len(arr):
            raise IndexError('make_full_pkg: %s(%d): out of bound %d'
                             % (name, j + 1, len(arr)))
        return float(arr[j])

    if n_blocks == 1:
        s11out, s12out, s21out, s22out = _make_pkg(
            faxis, _el(Len, 0, 'Len'), _el(Cpad, 0, 'Cpad'),
            _el(Cball, 0, 'Cball'), _el(Zpkg, 0, 'Zpkg'),
            pkg_param, _el(Lcomp, 0, 'Lcomp'), _el(Cbump, 0, 'Cbump'))
    else:
        for j in range(n_blocks):
            sp11, sp12, sp21, sp22 = _make_pkg(
                faxis, _el(Len, j, 'Len'), _el(Cpad, j, 'Cpad'),
                _el(Cball, j, 'Cball'), _el(Zpkg, j, 'Zpkg'),
                pkg_param, _el(Lcomp, j, 'Lcomp'), _el(Cbump, j, 'Cbump'))
            if j == 0:
                s11out, s12out, s21out, s22out = sp11, sp12, sp21, sp22
            else:
                s11out, s12out, s21out, s22out = _combines4p(
                    s11out, s12out, s21out, s22out,
                    sp11, sp12, sp21, sp22)

    return s11out, s12out, s21out, s22out
