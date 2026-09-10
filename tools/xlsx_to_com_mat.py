"""xlsx_to_com_mat.py -- a COM configuration workbook as the .mat Octave reads.

    python tools/xlsx_to_com_mat.py CONFIG.xlsx                    # -> ./CONFIG.mat
    python tools/xlsx_to_com_mat.py CONFIG.xlsx -o out/c1.mat
    python tools/xlsx_to_com_mat.py CONFIG.xlsx --set RESULT_DIR=out/ --set SAVE_FIGURES=0
    python tools/xlsx_to_com_mat.py CONFIG.xlsx --verify-octave

GNU Octave has no xlsread, so the Octave-capable release files under octave/
take their configuration as a .mat holding one variable, `parameter`: a 2-D
cell array identical to the raw output of `xlsread(file, 'COM_Settings')`.
`xls_parameter` then finds each keyword by strcmpi and reads the cell to its
right, exactly as it does for the workbook. The engine's own SAVE_CONFIG2MAT
path needs MATLAB to have read the workbook first, so it is no help here.

xlsread raw semantics, replicated: numeric cell -> double, text -> char, empty
-> NaN, bool -> 1/0, fully empty trailing rows and columns trimmed. Formula
cells are read as their cached values; a workbook with no cached values is
refused rather than converted with '=...' strings in it.

`--set KEY=VALUE` replaces the value cell to the right of an existing keyword
(a number if it parses as one, else text) and refuses a keyword that is absent
or appears twice. It never appends a row: a new keyword changes the grid the
engine sees, and the point of this file is to hand Octave the workbook as is.

Copyright 2025 802-COM Authors (workbook layout)
Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import argparse
import glob
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import openpyxl
import scipy.io

SHEET = 'COM_Settings'


def read_grid(path):
    """COM_Settings as a list of rows of xlsread-raw Python values."""
    wb = openpyxl.load_workbook(path, data_only=True)
    try:
        ws = wb[SHEET]
        grid = [[_cell(ws.cell(r, c).value) for c in range(1, ws.max_column + 1)]
                for r in range(1, ws.max_row + 1)]
    finally:
        wb.close()
    return grid


def _cell(v):
    if v is None:
        return float('nan')
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v)
    return float('nan') if s.strip() == '' else s


def _is_blank(v):
    return isinstance(v, float) and math.isnan(v)


def trim(grid):
    """Drop fully empty trailing rows and columns, as xlsread does."""
    while grid and all(_is_blank(v) for v in grid[-1]):
        grid.pop()
    if not grid:
        return grid
    ncol = max(len(r) for r in grid)
    grid = [r + [float('nan')] * (ncol - len(r)) for r in grid]
    while ncol > 0 and all(_is_blank(r[ncol - 1]) for r in grid):
        ncol -= 1
        grid = [r[:ncol] for r in grid]
    return grid


def find_keyword(grid, key):
    """(row, col) of every cell whose text equals key, case-insensitively."""
    return [(ri, ci) for ri, row in enumerate(grid) for ci, v in enumerate(row)
            if isinstance(v, str) and v.strip().lower() == key.lower()]


def set_value(grid, key, value):
    """Replace the value cell right of `key`. -> (old, new)."""
    hits = find_keyword(grid, key)
    if not hits:
        raise KeyError('keyword %r not found; a row is never appended' % key)
    if len(hits) > 1:
        raise KeyError('keyword %r appears %d times at %r; which one xls_parameter '
                       'reads is not something to guess' % (key, len(hits), hits))
    ri, ci = hits[0]
    if ci + 1 >= len(grid[ri]):
        grid[ri].append(float('nan'))
    old = grid[ri][ci + 1]
    try:
        new = float(value)
    except ValueError:
        new = value
    grid[ri][ci + 1] = new
    return old, new


def to_object_array(grid):
    """2-D object ndarray, filled cell by cell, so scipy writes a cell array."""
    arr = np.empty((len(grid), len(grid[0]) if grid else 0), dtype=object)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            arr[r, c] = v
    return arr


def convert(src, dst, overrides=()):
    """Workbook -> .mat. -> (grid, log lines)."""
    grid = trim(read_grid(src))
    bad = [(r + 1, c + 1, v) for r, row in enumerate(grid) for c, v in enumerate(row)
           if isinstance(v, str) and v.startswith('=')]
    if bad:
        raise ValueError('formula strings in the grid at %r: the workbook has no '
                         'cached values; open and save it in Excel first' % bad[:5])
    log = []
    for key, value in overrides:
        old, new = set_value(grid, key, value)
        log.append('%s: %r -> %r' % (key, old, new))
    d = os.path.dirname(os.path.abspath(dst))
    os.makedirs(d, exist_ok=True)
    scipy.io.savemat(dst, {'parameter': to_object_array(grid)}, format='5',
                     do_compression=False, oned_as='row')
    return grid, log


def _unwrap(v):
    a = np.asarray(v)
    if a.dtype.kind in 'US':
        return str(a.ravel()[0]) if a.size else ''
    return float('nan') if a.size == 0 else float(a.ravel()[0])


def verify_roundtrip(dst, grid):
    """Read the .mat back with scipy and compare every cell to the grid."""
    p = scipy.io.loadmat(dst)['parameter']
    if p.dtype != object or p.shape != (len(grid), len(grid[0])):
        raise AssertionError('parameter is %s %r, expected object %r'
                             % (p.dtype, p.shape, (len(grid), len(grid[0]))))
    for r in range(p.shape[0]):
        for c in range(p.shape[1]):
            want, got = grid[r][c], _unwrap(p[r, c])
            if _is_blank(want):
                ok = isinstance(got, float) and math.isnan(got)
            else:
                ok = got == want
            if not ok:
                raise AssertionError('r%dc%d: wrote %r, read back %r' % (r + 1, c + 1, want, got))
    return p.shape


def find_octave():
    """octave-cli on PATH, else the stock Windows install location, else None."""
    for name in ('octave-cli', 'octave-cli.exe', 'octave'):
        p = shutil.which(name)
        if p:
            return p
    hits = sorted(glob.glob(r'C:\Program Files\GNU Octave\Octave-*\mingw64\bin\octave-cli.exe'))
    return hits[-1] if hits else None


def verify_octave(dst, octave=None):
    """Load the .mat in Octave and report the cell array's shape and two classes."""
    octave = octave or find_octave()
    if not octave:
        raise RuntimeError('octave-cli not found')
    ev = ("load('%s'); assert(iscell(parameter)); "
          "printf('OCT,%%d,%%d,%%s,%%s\\n', rows(parameter), columns(parameter), "
          "class(parameter{3,1}), class(parameter{3,2}));" % dst.replace('\\', '/'))
    p = subprocess.run([octave, '--no-gui', '--no-window-system', '--eval', ev],
                       capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith('OCT,'):
            _, nr, nc, k, v = line.strip().split(',')
            return int(nr), int(nc), k, v
    raise RuntimeError('octave load failed:\n%s\n%s' % (p.stdout, p.stderr))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    ap.add_argument('config', help='COM configuration workbook (.xlsx)')
    ap.add_argument('-o', '--out', help='output .mat (default: ./<stem>.mat)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='replace the value of an existing keyword; repeatable')
    ap.add_argument('--verify-octave', action='store_true',
                    help='also load the result in octave-cli and report its shape')
    a = ap.parse_args(argv)

    overrides = []
    for s in a.set:
        if '=' not in s:
            ap.error('--set wants KEY=VALUE, got %r' % s)
        k, v = s.split('=', 1)
        overrides.append((k.strip(), v.strip()))
    dst = a.out or os.path.splitext(os.path.basename(a.config))[0] + '.mat'
    grid, log = convert(a.config, dst, overrides)
    shape = verify_roundtrip(dst, grid)
    print('%s -> %s  (%d x %d cells, scipy round trip exact)'
          % (os.path.basename(a.config), dst, shape[0], shape[1]))
    for line in log:
        print('  ' + line)
    if a.verify_octave:
        nr, nc, k, v = verify_octave(dst)
        print('  octave: %d x %d, parameter{3,1} %s, parameter{3,2} %s' % (nr, nc, k, v))
    return 0


if __name__ == '__main__':
    sys.exit(main())
