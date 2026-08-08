# ============================================================
# MATLAB→Python translation notes for read_Nport_touchstone
# MATLAB lines: 9414–9568
# ============================================================
# Reads any N-port Touchstone file; handles RI/MA/DB formats.
# Port reordering via port_order (1-based MATLAB → 0-based Python).
# Impedance renormalization: if file Z0 != Z_renorm, apply rho-based renorm.
# 2-port swap: always swap S12/S21 per Touchstone 1.x spec.
# Returns (sch, schFreqAxis):
#   sch: shape (nfreq, nport, nport) complex
#   schFreqAxis: shape (nfreq,) float, Hz
# ============================================================

import re
import numpy as np


def _mround(x):
    """MATLAB round(): half away from zero."""
    return int(np.floor(float(x) + 0.5)) if x >= 0 else int(np.ceil(float(x) - 0.5))


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
    port_order = [0, 0, 0, 0]
    port_order[0] = 1
    port_order[2] = int(ConnectedPorts[0])
    others = sorted(set(range(1, 5)) - {1, int(ConnectedPorts[0])})
    port_order[1] = others[0]
    port_order[3] = others[-1]
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
        qs = _mround(len(Floc) / 4.0)
        tqs = _mround(len(Floc) * 3.0 / 4.0)
        m1 = float(np.mean(pd1[qs - 1:tqs]))
        m2 = float(np.mean(pd2[qs - 1:tqs]))
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


def read_Nport_touchstone(touchstone_file, port_order, Z_renorm):
    """Read N-port Touchstone file (MATLAB lines 9414-9568).

    port_order: 1-based port ordering vector (e.g. [1,3,2,4] for 4-port swap).
    Z_renorm: target reference impedance in ohms.

    Returns (sch, schFreqAxis):
      sch: (nfreq, nport, nport) complex S-parameter array.
      schFreqAxis: (nfreq,) Hz frequency array.
    """
    # r4p15p0: port_order may be empty ([], None, 0-size array) -> auto-detect below
    if port_order is None:
        port_order = []
    else:
        port_order = [int(p) for p in np.asarray(port_order).ravel()]
    Z_renorm = float(Z_renorm)

    # Extract nport from file extension
    import os
    ext = os.path.splitext(touchstone_file)[1].lower()
    m = re.search(r'\d+', ext)
    nport = int(m.group()) if m else 2

    with open(touchstone_file, 'r', errors='replace') as fid:
        raw = fid.read()

    # Remove comment lines, find option line
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
            # Remove inline comments
            data_lines.append(stripped.split('!')[0].strip())

    if option_line is None:
        raise ValueError(f'read_Nport_touchstone: no # option line found in {touchstone_file}')

    # Parse option line: # [Hz/kHz/MHz/GHz] S [RI/MA/DB] R <Z0>
    opt_tokens = option_line[1:].upper().split()
    freq_scale_map = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}
    freq_mult = freq_scale_map.get(opt_tokens[0], 1e9) if opt_tokens else 1e9

    try:
        s_idx = opt_tokens.index('S')
        fmt = opt_tokens[s_idx + 1] if s_idx + 1 < len(opt_tokens) else 'MA'
    except ValueError:
        fmt = 'MA'

    # Get port impedance
    try:
        r_idx = opt_tokens.index('R')
        file_Z0 = float(opt_tokens[r_idx + 1]) if r_idx + 1 < len(opt_tokens) else 50.0
    except (ValueError, IndexError):
        file_Z0 = 50.0

    # Tokenize all data
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
    if nfreq == 0:
        raise ValueError(f'read_Nport_touchstone: no data parsed from {touchstone_file}')

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
        raise ValueError(f'read_Nport_touchstone: unsupported format {fmt}')

    # Build 3D S-parameter array: (nfreq, nport, nport) — row-major
    sp = np.zeros((nport, nport, nfreq), dtype=complex)
    for j in range(nport):
        sp[j, :, :] = cdata[:, j * nport:(j + 1) * nport].T

    # 2-port swap (Touchstone 1.x spec)
    if nport == 2:
        temp = sp[0, 1, :].copy()
        sp[0, 1, :] = sp[1, 0, :]
        sp[1, 0, :] = temp

    Spar_S = sp
    Spar_Z0 = file_Z0

    # Renormalize if needed
    if abs(Spar_Z0 - Z_renorm) > 1e-9:
        print(f'INFO: S-parameter reference impedance of {Spar_Z0:.6g} ohms renormalized to {Z_renorm:.6g} ohms')
        rho = (Z_renorm - Spar_Z0) / (Z_renorm + Spar_Z0)
        I = np.eye(nport)
        for k in range(nfreq):
            s_old = Spar_S[:, :, k]
            Spar_S[:, :, k] = np.linalg.solve(I - rho * s_old, s_old - rho * I)

    # Shift: put frequency as first dimension → (nfreq, nport, nport)
    sch = np.transpose(Spar_S, (2, 0, 1))

    # r4p15p0: auto-detect port order when none supplied (empty port_order)
    if len(port_order) == 0:
        port_order = _auto_port_order(sch, freq)

    # Port reordering (port_order is 1-based)
    po = [p - 1 for p in port_order]  # 0-based
    if len(po) == nport:
        sch = sch[:, po, :][:, :, po]

    return sch, freq, port_order
