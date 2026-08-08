# ============================================================
# MATLAB→Python translation notes for readdataSnPx
# MATLAB lines: 10768–10884
# ============================================================
# Reads Touchstone SnP file.
# Header loop: skip lines until '#' found.
# Config line parsing: extract units (between '#' and 'S') and format (after 'S').
# Data loop: read one float (freq); skip non-numeric lines; read nport^2 pairs.
# Formats: RI (re+im), MA (mag+angle_deg), DB (db+angle_deg → M=10^(db/20)).
# 2-port: swap S21/S12 per Touchstone spec.
# Freq scaling: Hz, kHz, MHz, GHz.
# Returns result.freq (1D array) and result.cs (3D array: nport x nport x nfreq).
# Python: use result.cs[i,j,k] = S(i+1,j+1) at freq k.
# ============================================================

import re
import numpy as np
from types import SimpleNamespace


def readdataSnPx(filename, nport):
    """Read Touchstone SnP file (MATLAB lines 10768-10884).

    filename: path to Touchstone file.
    nport: number of ports.
    Returns SimpleNamespace with .freq (array, Hz) and .cs (nport x nport x nfreq array).
    """
    nport = int(nport)

    with open(filename, 'r', errors='replace') as fid:
        lines = fid.readlines()

    # Find # option line
    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#'):
            header_line = stripped
            data_start = i + 1
            break

    if header_line is None:
        raise ValueError(f'readdataSnPx: could not find # config line in {filename}')

    # Parse config line: # [units] S [format] R [resistance]
    tokens = header_line[1:].split()
    tokens_upper = [t.upper() for t in tokens]
    try:
        s_idx = tokens_upper.index('S')
    except ValueError:
        raise ValueError(f'readdataSnPx: could not find S in config line: {header_line}')

    units = tokens_upper[0] if s_idx > 0 else 'GHz'
    fmt = tokens_upper[s_idx + 1] if s_idx + 1 < len(tokens_upper) else 'MA'

    # Collect all data tokens from remaining lines (skip comments)
    all_tokens = []
    for line in lines[data_start:]:
        s = line.strip()
        if not s or s.startswith('!') or s.startswith('#'):
            continue
        # Remove inline comments
        s = s.split('!')[0].strip()
        all_tokens.extend(s.split())

    freq_list = []
    cs_list = []
    n_vals_per_freq = 1 + nport * nport * 2  # 1 freq + nport^2 pairs

    idx = 0
    while idx < len(all_tokens):
        try:
            freq_val = float(all_tokens[idx])
        except ValueError:
            idx += 1
            continue
        idx += 1

        row = np.zeros((nport, nport), dtype=complex)
        for ni in range(nport):
            for nj in range(nport):
                if idx + 1 >= len(all_tokens):
                    break
                try:
                    a = float(all_tokens[idx])
                    b = float(all_tokens[idx + 1])
                except (ValueError, IndexError):
                    break
                idx += 2
                if fmt == 'MA':
                    row[ni, nj] = a * np.exp(1j * b * np.pi / 180.0)
                elif fmt == 'RI':
                    row[ni, nj] = complex(a, b)
                elif fmt == 'DB':
                    M = 10.0 ** (a / 20.0)
                    row[ni, nj] = complex(M * np.cos(b * np.pi / 180.0),
                                          M * np.sin(b * np.pi / 180.0))
                else:
                    raise ValueError(f'readdataSnPx: unknown format {fmt}')

        freq_list.append(freq_val)
        cs_list.append(row)

    freq = np.array(freq_list, dtype=float)
    nfreq = len(freq)
    cs = np.zeros((nport, nport, nfreq), dtype=complex)
    for k, row in enumerate(cs_list):
        cs[:, :, k] = row

    # 2-port: swap S21 and S12 per Touchstone spec
    if nport == 2:
        temp = cs[1, 0, :].copy()
        cs[1, 0, :] = cs[0, 1, :]
        cs[0, 1, :] = temp

    # Scale freq to Hz
    scale_map = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}
    scale = scale_map.get(units, 1e9)
    freq = freq * scale

    result = SimpleNamespace()
    result.freq = freq
    result.cs = cs
    return result
