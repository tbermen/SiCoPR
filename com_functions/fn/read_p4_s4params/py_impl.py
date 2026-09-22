# ============================================================
# MATLAB→Python translation notes for read_p4_s4params
# MATLAB lines: 10391–10524
# ============================================================
# Reads 4-port Touchstone, applies TX/RX skew correction, converts to mixed-mode.
# T = [[1,1,0,0],[1,-1,0,0],[0,0,1,1],[0,0,1,-1]]
# sigma_matrix for each frequency point based on TX/RX p/n skew.
# Snew = sigma_matrix .* S (elementwise, not matrix multiply)
# W = T * Snew * inv(T)
# D matrix indexing (MATLAB 1-based):
#   SDD(1,1) = D(2,2); SDD(2,2) = D(4,4); SDD(1,2) = D(2,4); SDD(2,1) = D(4,2)
#   SDC(1,1) = D(2,1); SDC(2,2) = D(4,3); SDC(1,2) = D(2,3); SDC(2,1) = D(4,1)
#   SCC(1,1) = D(1,1); SCC(2,2) = D(3,3); SCC(1,2) = D(1,3); SCC(2,1) = D(3,1)
#   SCD(1,1) = D(1,2); SCD(2,2) = D(3,4); SCD(1,2) = D(1,4); SCD(2,1) = D(3,2)
# ============================================================

import numpy as np
from com_functions.fn.auto_port_order.py_impl import auto_port_order as _auto_port_order
from types import SimpleNamespace


def _read_Nport_touchstone(touchstone_file, port_order, Z_renorm):
    """Inlined read_Nport_touchstone."""
    import re, os
    # r4p15p0: empty port_order -> auto-detect after read
    if port_order is None:
        port_order = []
    else:
        port_order = [int(p) for p in np.asarray(port_order).ravel()]
    Z_renorm = float(Z_renorm)
    ext = os.path.splitext(touchstone_file)[1].lower()
    m = re.search(r'\d+', ext)
    nport = int(m.group()) if m else 4

    with open(touchstone_file, 'r', errors='replace') as fid:
        raw = fid.read()

    lines = raw.splitlines()
    option_line = None
    data_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('!'):
            continue
        if stripped.startswith('#') and option_line is None:
            option_line = stripped
        else:
            data_lines.append(stripped.split('!')[0].strip())

    if option_line is None:
        raise ValueError(f'No # option line in {touchstone_file}')

    opt_tokens = option_line[1:].upper().split()
    freq_scale_map = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}
    freq_mult = freq_scale_map.get(opt_tokens[0], 1e9)
    try:
        s_idx = opt_tokens.index('S')
        fmt = opt_tokens[s_idx + 1]
    except (ValueError, IndexError):
        fmt = 'MA'
    try:
        r_idx = opt_tokens.index('R')
        file_Z0 = float(opt_tokens[r_idx + 1])
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
    ri_flat = data[:, 1:]
    re_data = ri_flat[:, 0::2]
    im_data = ri_flat[:, 1::2]
    if fmt == 'RI':
        cdata = re_data + 1j * im_data
    elif fmt == 'MA':
        cdata = re_data * np.exp(1j * im_data * np.pi / 180.0)
    elif fmt == 'DB':
        mag = 10.0 ** (re_data / 20.0)
        cdata = mag * np.exp(1j * im_data * np.pi / 180.0)
    else:
        raise ValueError(f'Unsupported format {fmt}')

    sp = np.zeros((nport, nport, nfreq), dtype=complex)
    for j in range(nport):
        sp[j, :, :] = cdata[:, j * nport:(j + 1) * nport].T
    if nport == 2:
        temp = sp[0, 1, :].copy()
        sp[0, 1, :] = sp[1, 0, :]
        sp[1, 0, :] = temp

    if abs(file_Z0 - Z_renorm) > 1e-9:
        rho = (Z_renorm - file_Z0) / (Z_renorm + file_Z0)
        I = np.eye(nport)
        for k in range(nfreq):
            s_old = sp[:, :, k]
            sp[:, :, k] = np.linalg.solve(I - rho * s_old, s_old - rho * I)

    sch = np.transpose(sp, (2, 0, 1))
    # r4p15p0: auto-detect port order when none supplied
    if len(port_order) == 0:
        port_order = _auto_port_order(sch, freq)
    po = [p - 1 for p in port_order]
    if len(po) == nport:
        sch = sch[:, po, :][:, :, po]
    return sch, freq, port_order


def _rangelimit(sch, freq, param, OP):
    flim = float(getattr(param, 'flim', float('inf')))
    idx = np.where(freq >= flim)[0]
    if len(idx) > 0:
        iend = int(idx[0]) + 1
        return sch[:iend], freq[:iend], 1, param
    else:
        param.flim = float(freq[-1])
        return sch, freq, 0, param


def read_p4_s4params(infile, plot_ini_s_params, plot_dif_s_params, ports, OP, param):
    """Read 4-port Touchstone and convert to mixed-mode (MATLAB lines 10391-10524).

    Returns (data, SDD, SDC, SCC, SCD, ports).
    All SXX arrays have shape (nfreq, 2, 2).
    r4p15p0: no [1 3 2 4] default; empty ports triggers auto-detection and the
    resolved order is returned as the 6th output.
    """
    sch, freq, ports = _read_Nport_touchstone(infile, ports, float(param.Z0))
    sch, freq, limited, param = _rangelimit(sch, freq, param, OP)

    nfreq = len(freq)

    # Skew parameters (default 0 if not set)
    Txpskew = float(getattr(param, 'Txpskew', 0.0))
    Txnskew = float(getattr(param, 'Txnskew', 0.0))
    Rxpskew = float(getattr(param, 'Rxpskew', 0.0))
    Rxnskew = float(getattr(param, 'Rxnskew', 0.0))

    T = np.array([[1.0, 1.0, 0.0, 0.0],
                  [1.0, -1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0, -1.0]])
    T_inv = np.linalg.inv(T)

    D = np.zeros((nfreq, 4, 4), dtype=complex)
    for i in range(nfreq):
        f = freq[i]
        s1 = np.exp(2j * np.pi * f * Txpskew * 1e-12)
        s2 = np.exp(2j * np.pi * f * Txnskew * 1e-12)
        s3 = np.exp(2j * np.pi * f * Rxpskew * 1e-12)
        s4 = np.exp(2j * np.pi * f * Rxnskew * 1e-12)
        sigma_matrix = np.array([
            [s1 ** 2,   s1 * s2, s1 * s3, s1 * s4],
            [s1 * s2,   s2 ** 2, s2 * s3, s2 * s4],
            [s1 * s3,   s2 * s3, s3 ** 2, s3 * s4],
            [s1 * s4,   s2 * s4, s3 * s4, s4 ** 2],
        ])
        S = sch[i, :, :]
        Snew = sigma_matrix * S  # elementwise (Sigfct .* S)
        D[i] = T @ Snew @ T_inv

    # Extract mixed-mode S-params (0-based Python, MATLAB 1-based → subtract 1)
    SDD = np.zeros((nfreq, 2, 2), dtype=complex)
    SDD[:, 0, 0] = D[:, 1, 1]   # D(2,2)
    SDD[:, 1, 1] = D[:, 3, 3]   # D(4,4)
    SDD[:, 0, 1] = D[:, 1, 3]   # D(2,4)
    SDD[:, 1, 0] = D[:, 3, 1]   # D(4,2)

    SDC = np.zeros((nfreq, 2, 2), dtype=complex)
    SDC[:, 0, 0] = D[:, 1, 0]   # D(2,1)
    SDC[:, 1, 1] = D[:, 3, 2]   # D(4,3)
    SDC[:, 0, 1] = D[:, 1, 2]   # D(2,3)
    SDC[:, 1, 0] = D[:, 3, 0]   # D(4,1)

    SCC = np.zeros((nfreq, 2, 2), dtype=complex)
    SCC[:, 0, 0] = D[:, 0, 0]   # D(1,1)
    SCC[:, 1, 1] = D[:, 2, 2]   # D(3,3)
    SCC[:, 0, 1] = D[:, 0, 2]   # D(1,3)
    SCC[:, 1, 0] = D[:, 2, 0]   # D(3,1)

    SCD = np.zeros((nfreq, 2, 2), dtype=complex)
    SCD[:, 0, 0] = D[:, 0, 1]   # D(1,2)
    SCD[:, 1, 1] = D[:, 2, 3]   # D(3,4)
    SCD[:, 0, 1] = D[:, 0, 3]   # D(1,4)
    SCD[:, 1, 0] = D[:, 2, 1]   # D(3,2)

    data = SimpleNamespace()
    data.m = sch
    data.freq = freq
    data.flim = getattr(param, 'flim', freq[-1])
    data.limited = limited

    return data, SDD, SDC, SCC, SCD, ports
