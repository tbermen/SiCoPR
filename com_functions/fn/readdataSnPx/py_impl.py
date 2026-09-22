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
        # MATLAB never gets here: `while ~strcmp(str(1),'#')` calls fgetl,
        # which returns the NUMBER -1 at end of file, and -1 is neither '#'
        # nor empty, so the n>1000 escape inside `if isempty(str)` is never
        # reached and the loop spins forever.  COM Octave on a file with no
        # option line did not terminate (killed at 120 s).  There is no
        # reference answer to match, so refusing is the only honest option.
        raise ValueError(f'readdataSnPx: could not find # config line in {filename}')

    # Parse the option line the way MATLAB does:
    #   A = sscanf(str,'%1s %2s %1s %2s %1s %2s',[1,inf])
    # which is just the line with every space removed, then
    #   p = find(A=='S'); units = lower(A(2:p-1)); format = A(p+1:p+2).
    # So the S is matched CASE SENSITIVELY and the format is exactly the two
    # characters after it, not the whole token.  COM Octave 4p16p0:
    #   '# ghz s ri r 50' -> error: readdataSnP: Unknown data format
    #   '# GHz S'         -> error: A(7): out of bound 5 (dimensions are 1x5)
    # Upper-casing the tokens accepted a lowercase option line the reference
    # refuses, and defaulting the format to MA answered where it errors.
    A = ''.join(header_line.split())
    p = A.find('S')
    if p >= 0 and len(A) < p + 3:
        raise ValueError('readdataSnPx: option line %r has no two-character '
                         'format after the S' % header_line)
    units = A[1:p].upper() if p >= 0 else ''
    fmt = A[p + 1:p + 3].upper() if p >= 0 else ''

    # MATLAB reads the data with fscanf, so a record may span lines, and a
    # token that is not a number means two different things:
    #   between records, fscanf('%f') returns nothing, fscanf('%s') eats the
    #     token and fgetl() then discards THE REST OF THAT LINE;
    #   inside a record, fscanf('%f') returns [] and cs(ni,nj,nk)=[] errors.
    # Flattening the file into one token stream lost the line boundary: COM
    # Octave, a line reading 'JUNK 2.0 0.11 ...', dropped that whole record,
    # where this read the numbers that followed the stray token.
    # A '!' is not special to the reference either -- it is simply the first
    # token fscanf cannot read as a number, which is why it ends the line.
    toks = []                       # (text, line number)
    for ln, line in enumerate(lines[data_start:]):
        head, bang, _rest = line.partition('!')
        toks.extend((t, ln) for t in head.split())
        if bang:
            toks.append(('!', ln))  # fscanf stops here; fgetl eats the rest

    freq_list = []
    cs_list = []

    idx = 0
    while idx < len(toks):
        try:
            freq_val = float(toks[idx][0])
        except ValueError:
            ln = toks[idx][1]
            idx += 1
            while idx < len(toks) and toks[idx][1] == ln:
                idx += 1          # fgetl: discard the rest of the line
            continue
        idx += 1

        row = np.zeros((nport, nport), dtype=complex)
        for ni in range(nport):
            for nj in range(nport):
                # MATLAB assigns cs(ni,nj,nk) = [] when the pair is missing.
                # COM Octave, a record cut short: "error: =: nonconformant
                # arguments (op1 is 1x1, op2 is 0x1)".  Breaking out left
                # zeros in the unread entries and reported them as data.
                if idx + 2 > len(toks):
                    raise ValueError(
                        'readdataSnPx: %s ends part-way through the record at '
                        'frequency %g' % (filename, freq_val))
                try:
                    a = float(toks[idx][0])
                    b = float(toks[idx + 1][0])
                except ValueError:
                    raise ValueError(
                        'readdataSnPx: %s has a non-numeric entry in the '
                        'record at frequency %g' % (filename, freq_val))
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

    if not freq_list:
        # MATLAB never assigns freq or cs, so `result.cs = cs` errors.
        # COM Octave, option line and nothing else: "error: 'cs' undefined".
        raise ValueError('readdataSnPx: %s holds no S-parameter data' % filename)

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

    # Scale freq to Hz.  MATLAB's `switch lower(units)` has no otherwise, so
    # a unit it does not recognise -- including an option line with no unit at
    # all -- leaves the frequencies exactly as the file gave them.  COM
    # Octave, '# S RI R 50' and '# THz S RI R 50': freq = [1 2], not 1e9 times
    # that.  Defaulting to GHz scaled those files by a billion.
    scale_map = {'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9}
    scale = scale_map.get(units, 1.0)
    freq = freq * scale

    result = SimpleNamespace()
    result.freq = freq
    result.cs = cs
    return result
