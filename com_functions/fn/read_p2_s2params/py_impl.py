# ============================================================
# MATLAB→Python translation notes for read_p2_s2params
# MATLAB lines: 10259–10390
# ============================================================
# Reads 2-port Touchstone, converts single-ended → differential mode.
# T = [[1,1],[1,-1]]; W = T * (S / T) = T * S * inv(T).
# D[i] = W for each frequency point i (0-based in Python).
# SDD(:,1,1) = D(:,2,2) → Python D[:,1,1]
# SDC(:,1,1) = D(:,2,1) → Python D[:,1,0]
# SCC(:,1,1) = D(:,1,1) → Python D[:,0,0]
# SCD(:,1,1) = D(:,1,2) → Python D[:,0,1]
# rangelimit: truncates freq axis at param.flim.
# Plotting skipped (DISPLAY_WINDOW handled silently).
# ============================================================

import numpy as np
from types import SimpleNamespace


def _read_Nport_touchstone(touchstone_file, port_order, Z_renorm):
    """Inlined read_Nport_touchstone."""
    import re, os
    port_order = [int(p) for p in port_order]
    Z_renorm = float(Z_renorm)
    ext = os.path.splitext(touchstone_file)[1].lower()
    m = re.search(r'\d+', ext)
    nport = int(m.group()) if m else 2

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
    po = [p - 1 for p in port_order]
    if len(po) == nport:
        sch = sch[:, po, :][:, :, po]
    return sch, freq


def _rangelimit(sch, freq, param, OP):
    flim = float(getattr(param, 'flim', float('inf')))
    idx = np.where(freq >= flim)[0]
    if len(idx) > 0:
        iend = int(idx[0]) + 1
        return sch[:iend], freq[:iend], 1, param
    else:
        param.flim = float(freq[-1])
        return sch, freq, 0, param


def read_p2_s2params(infile, plot_ini_s_params, plot_dif_s_params, ports, OP, param):
    """Read 2-port Touchstone and convert to differential mode (MATLAB lines 10259-10390).

    Returns (data, SDD, SDC, SCC, SCD).
    All SXX arrays have shape (nfreq, 1, 1).
    """
    if not ports or len(ports) == 0:
        ports = [1, 2]
    ports = [1, 2]  # MATLAB overrides ports to [1,2]

    sch, freq = _read_Nport_touchstone(infile, ports, float(param.Z0))
    sch, freq, limited, param = _rangelimit(sch, freq, param, OP)

    nfreq = len(freq)
    T = np.array([[1.0, 1.0], [1.0, -1.0]])
    T_inv = np.linalg.inv(T)

    D = np.zeros((nfreq, 2, 2), dtype=complex)
    for i in range(nfreq):
        S = sch[i, :, :]
        D[i] = T @ S @ T_inv

    # Extract mixed-mode S-params (MATLAB 1-based → Python 0-based)
    SDD = np.zeros((nfreq, 1, 1), dtype=complex)
    SDC = np.zeros((nfreq, 1, 1), dtype=complex)
    SCC = np.zeros((nfreq, 1, 1), dtype=complex)
    SCD = np.zeros((nfreq, 1, 1), dtype=complex)

    SDD[:, 0, 0] = D[:, 1, 1]  # MATLAB D(:,2,2)
    SDC[:, 0, 0] = D[:, 1, 0]  # MATLAB D(:,2,1)
    SCC[:, 0, 0] = D[:, 0, 0]  # MATLAB D(:,1,1)
    SCD[:, 0, 0] = D[:, 0, 1]  # MATLAB D(:,1,2)

    data = SimpleNamespace()
    data.m = sch
    data.freq = freq
    data.flim = getattr(param, 'flim', freq[-1])
    data.limited = limited

    return data, SDD, SDC, SCC, SCD
