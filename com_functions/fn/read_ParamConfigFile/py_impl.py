# ============================================================
# MATLAB→Python translation notes for read_ParamConfigFile
# MATLAB lines: 9601–10258
# ============================================================
# Struct fields: MATLAB struct → SimpleNamespace
# parameter cell array: MATLAB 2D cell → Python list-of-lists
# xlsread → _load_parameter_sheet (CSV/Excel reader)
# load(mat) → scipy.io.loadmat (optional, not needed for CSV)
# isnan(x) → _isnan(x) handles strings and None
# isempty(x) → _isempty(x) handles None, '', [], ()
# strsplit(s) → s.split() for whitespace split
# isfield(s, f) → hasattr(s, f)
# filesep → os.sep
# date → datetime.date.today().strftime('%d-%b-%Y')
# regexprep → re.sub
# strrep → str.replace
# MATLAB matrix string '[a b; c d]' → _parse_matlab_matrix(s)
# Column-major vs row-major: no significant arrays (all scalar/1D)
# 1-based indexing: no significant index arithmetic
# Output: (param, OP) tuple matching MATLAB [param, OP]
# Known discrepancy from sicopr.py: sicopr.py missing pkg block parsing + many params
# ============================================================

import csv
import datetime
import math
import os
import re
import numpy as np
from types import SimpleNamespace

_SENTINEL = object()

# Which MATLAB release to emulate. '4p17p0', the current IEEE release, is the
# default (VERSION.json, the owner's call, 2026-10-03). The 208-case MATLAB
# reference results are 4p15p0 output, so anything reproducing them must ask
# for '4p15p0' explicitly (docs/VERSIONS.md says what differs between releases).
# A config's 'COM Version' keyword, if present, wins over this default.
COM_MATLAB_VERSION = '4p17p0'


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def _parse_cell(cell):
    """Convert a CSV cell to float if possible, else keep as string.

    An empty cell is NaN, not '': that is what xlsread's raw output gives for
    a blank, and the reference depends on it -- ML 9660 reads PKG_NAME with a
    '' default and then asks `if isnan(param.PKG_NAME)`, which only makes
    sense if a blank cell arrives as NaN. COM Octave, f_b with a blank value
    cell: param.fb = NaN, where the port raised
    "TypeError: can't multiply sequence by non-int of type float" on '' * 1e9.
    """
    if cell is None:
        return float('nan')
    if isinstance(cell, bool):
        return cell
    if isinstance(cell, (int, float)):
        return _whole_to_int(cell)
    s = str(cell).strip()
    if not s:
        return float('nan')
    try:
        return _whole_to_int(float(s))
    except ValueError:
        return s


def _whole_to_int(v):
    """A whole number as int, as openpyxl returns one written '32'.

    MATLAB holds every number as a double and uses 32.0 as a count without
    comment; the port counts with Python ints (np.ones(M), range(N)). A cell
    written '32.0' (a workbook resaved by another tool, 2026-09-28) or read from
    CSV arrived as a float and np.ones(32.0) raised. Magnitudes past 2**53 are
    left alone: they are not counts, and an int that large would not fit int64.
    """
    if isinstance(v, float) and v.is_integer() and abs(v) < 2 ** 53:
        return int(v)
    return v


def _load_csv(path):
    """Load a CSV config file → list-of-lists (parameter sheet)."""
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([_parse_cell(c) for c in row])
    return rows


def _load_excel(path):
    """Load an Excel config file → list-of-lists (parameter sheet)."""
    try:
        import openpyxl
    except ImportError:
        raise ImportError('openpyxl required for Excel files: pip install openpyxl')
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    # read_only=True keeps the underlying zip handle open until the workbook is
    # closed. On Windows that leaves the config file LOCKED for the rest of the
    # process, so anything that reads a config and then tries to rewrite or
    # delete it fails with WinError 32. Every row is materialised here, so
    # there is nothing to lose by closing immediately.
    try:
        ws = wb['COM_Settings'] if 'COM_Settings' in wb.sheetnames else wb.active
        rows = []
        for row in ws.iter_rows(values_only=True):
            if all(c is None for c in row):
                continue
            rows.append([_parse_cell(c) for c in row])
    finally:
        wb.close()
    return rows


def _load_parameter_sheet(param_file):
    """Load CSV or Excel config file → list-of-lists parameter sheet."""
    _, ext = os.path.splitext(param_file)
    ext = ext.lower()
    if ext == '.csv':
        return _load_csv(param_file)
    elif ext in ('.xls', '.xlsx'):
        return _load_excel(param_file)
    elif ext == '.mat':
        import scipy.io
        mat = scipy.io.loadmat(param_file)
        parameter = mat.get('parameter')
        if parameter is None:
            raise KeyError("'parameter' variable not found in .mat file")
        return parameter.tolist()
    else:
        # Try Excel for unknown extensions (matches MATLAB xlsread fallback)
        return _load_excel(param_file)


# ---------------------------------------------------------------------------
# MATLAB helper function inlines
# ---------------------------------------------------------------------------

def _isnan(x):
    """MATLAB isnan — returns True if x is NaN; False for strings/None."""
    if x is None:
        return False
    if isinstance(x, str):
        return False
    try:
        v = float(x)
        return math.isnan(v)
    except (TypeError, ValueError):
        return False


def _isempty(x):
    """MATLAB isempty — True if None, empty string, or empty sequence."""
    if x is None:
        return True
    if isinstance(x, str):
        return x == ''
    if isinstance(x, np.ndarray):
        return x.size == 0
    try:
        return len(x) == 0
    except TypeError:
        return False


def _parse_matlab_scalar_or_range(token):
    """Parse a single token that may be a scalar or a MATLAB range start:step:end."""
    token = token.rstrip(';').strip()
    if ':' in token:
        parts = token.split(':')
        if len(parts) == 2:
            start, stop = float(parts[0]), float(parts[1])
            step = 1.0
        elif len(parts) == 3:
            start, step, stop = float(parts[0]), float(parts[1]), float(parts[2])
        else:
            return [float(token)]
        n = int(round((stop - start) / step)) + 1
        return list(np.linspace(start, stop, max(n, 0)))
    return [float(token)]


def _parse_matlab_matrix(s):
    """Parse MATLAB matrix literal '[a b c; d e f]' or range '[a:b:c]' → numpy array."""
    import re as _re
    s = s.strip()
    if s.startswith('['):
        s = s[1:]
    if s.endswith(']'):
        s = s[:-1]
    s = s.strip()
    if not s:
        return np.array([])
    # Normalise spaced colon ranges: '-0.2 : 0.02 : 0' → '-0.2:0.02:0'
    s = _re.sub(r'\s*:\s*', ':', s)
    # Normalise unary minus separated from digit: '- 0.2' → '-0.2'
    s = _re.sub(r'-\s+(?=[\d.])', '-', s)
    row_strs = [r.strip() for r in s.split(';') if r.strip()]
    rows = []
    for r in row_strs:
        vals = r.split()
        row = []
        for v in vals:
            if v and v != ':':
                row.extend(_parse_matlab_scalar_or_range(v))
        rows.append(row)
    if len(rows) == 1:
        return np.array(rows[0])
    return np.array(rows)


def _eval_matlab_value(v, eval_if_string):
    """Evaluate a spreadsheet value, handling MATLAB matrix strings."""
    if not isinstance(v, str) or not eval_if_string:
        return v
    v_str = v.strip()
    if v_str.startswith('['):
        return _parse_matlab_matrix(v_str)
    try:
        return float(v_str)
    except ValueError:
        pass
    try:
        return eval(v_str)  # noqa: S307
    except Exception:
        return v_str


def _xls_param(parameter, param_name, eval_if_string=False, default_value=_SENTINEL):
    """Inline xls_parameter — case-insensitive lookup in 2D parameter sheet."""
    name_lower = param_name.lower()
    matches = [
        (r, c)
        for r, row in enumerate(parameter)
        for c, cell in enumerate(row)
        if isinstance(cell, str) and cell.strip().lower() == name_lower
    ]
    if len(matches) == 0:
        if default_value is _SENTINEL:
            raise KeyError(f'Mandatory parameter "{param_name}" not found in config file')
        return default_value
    if len(matches) > 1:
        raise ValueError(f'{len(matches)} occurrences of "{param_name}" found in spreadsheet')
    r, c = matches[0]
    # A short row: xlsread pads the sheet to a rectangle, so the cell past the
    # end of a row is a blank one, i.e. NaN, not ''. Not measured directly --
    # Octave's csvread4com (SiCoPR's own, not the reference) leaves '' there --
    # but '' guaranteed a TypeError on the first arithmetic, which is neither
    # reference's answer.
    p = parameter[r][c + 1] if c + 1 < len(parameter[r]) else float('nan')
    v = _eval_matlab_value(p, eval_if_string)
    if name_lower in _COUNT_KEYWORDS:
        v = _as_count(v, param_name)
    return v


# Keywords the engine uses as counts or sizes (np.ones(M), range(N_b), slice
# ends). MATLAB holds them as doubles and accepts 32.0 anywhere; Python needs an
# int. Conditioned here, where every source (xlsx, csv, .mat, a string cell)
# passes, so a type artefact in the file cannot reach the engine.
_COUNT_KEYWORDS = {'m', 'l', 'n_b', 'n_v', 'n_bx', 'n_bg', 'n_bf', 'n_bmax', 'n_f',
                   'ffe_pre_tap_len', 'ffe_post_tap_len', 'samples_for_c2m',
                   'num_ui_rxff_noise', 'n_qb', 'n', 'n_tail_start', 'ts_anchor',
                   'local search'}


def _as_count(v, name):
    """A count keyword's value as int: whole numbers of any numeric type pass,
    a blank (NaN) or a string is left for the caller, anything else is an
    error naming the keyword -- as MATLAB's ones(1, 32.5) refuses a fraction."""
    a = np.asarray(v) if not isinstance(v, str) else None
    if a is None or a.dtype.kind not in 'biuf' or a.size != 1:
        return v
    x = a.ravel()[0]
    if a.dtype.kind == 'b' or not np.isfinite(x):
        return v
    if float(x) != int(x):
        raise ValueError('config keyword "%s" must be a whole number, got %r'
                         % (name, float(x)))
    return int(x)


def _xls_param_txffe(parameter, param_name):
    """Inline xls_parameter_txffe — returns (value, found_flag)."""
    name_lower = param_name.lower()
    matches = [
        (r, c)
        for r, row in enumerate(parameter)
        for c, cell in enumerate(row)
        if isinstance(cell, str) and cell.strip().lower() == name_lower
    ]
    if len(matches) == 0:
        return 0, 0
    if len(matches) > 1:
        raise ValueError(f'{len(matches)} occurrences of "{param_name}" found in spreadsheet')
    r, c = matches[0]
    p = parameter[r][c + 1] if c + 1 < len(parameter[r]) else 0
    if isinstance(p, str):
        p = _eval_matlab_value(p, True)
    return p, 1


def _read_pkg_params(block):
    """Inline read_package_parameters for a .START block (list-of-lists)."""
    pkg = SimpleNamespace()

    def xp(key, default=None):
        return _xls_param(block, key, eval_if_string=True, default_value=default)

    pkg.C_pkg_board = np.atleast_1d(np.asarray(xp('C_p', 0.0), dtype=float)) * 1e-9
    pkg.R_diepad = np.atleast_1d(np.asarray(xp('R_d', np.array([50.0, 50.0])), dtype=float))
    pkg.a_thru = float(xp('A_v', 0.5))
    pkg.a_fext = float(xp('A_fe', 0.5))
    pkg.a_next = float(xp('A_ne', 0.5))

    raw_tx = xp('z_p (TX)', _parse_matlab_matrix('[ 8 24 30 45 ; 1 1 1 1 ; 1 1 1 1 ; 0.5 0.5 0.5 0.5 ]'))
    if isinstance(raw_tx, str):
        raw_tx = _parse_matlab_matrix(raw_tx)
    z_p_tx = np.atleast_2d(np.asarray(raw_tx, dtype=float)).T
    ncases, mele = z_p_tx.shape
    pkg.flex = {2: 2, 4: 4, 1: 1}.get(mele, mele)
    pkg.z_p_tx_cases = z_p_tx

    def _load_zp(key):
        r = xp(key, np.zeros_like(z_p_tx))
        if isinstance(r, str):
            r = _parse_matlab_matrix(r)
        # Transpose, exactly as z_p (TX) above and as MATLAB does for all four
        # (com_ieee8023_4p15p0.m L10019/10035/10041/10047 each end in .').
        # The spreadsheet stores rows = package segments, columns = cases; the
        # engine indexes [case, :]. Omitting the transpose here fed the RX/NEXT/FEXT
        # package a row of the matrix (segment across cases) instead of a case
        # column, which for a square z_p matrix passes the shape check below
        # silently while producing a grossly over-long package.
        arr = np.atleast_2d(np.asarray(r, dtype=float)).T
        if arr.shape != (ncases, mele):
            raise ValueError('All TX, NEXT, FEXT, Rx cases must agree')
        return arr

    pkg.z_p_next_cases = _load_zp('z_p (NEXT)')
    pkg.z_p_fext_cases = _load_zp('z_p (FEXT)')
    pkg.z_p_rx_cases = _load_zp('z_p (RX)')

    pkg.pkg_gamma0_a1_a2 = np.atleast_1d(np.asarray(
        xp('package_tl_gamma0_a1_a2', np.array([0.0, 1.734e-3, 1.455e-4])), dtype=float))
    pkg.pkg_tau = float(xp('package_tl_tau', 6.141e-3))

    raw_zc = xp('package_Z_c', _parse_matlab_matrix('[92 92 ; 70 70; 80 80; 100 100]'))
    if isinstance(raw_zc, str):
        raw_zc = _parse_matlab_matrix(raw_zc)
    pkg_Z_c = np.atleast_2d(np.asarray(raw_zc, dtype=float)).T
    pkg.pkg_Z_c = pkg_Z_c

    if mele == 2:
        zeros4 = np.zeros((ncases, 2))
        for attr in ['z_p_fext_cases', 'z_p_next_cases', 'z_p_tx_cases', 'z_p_rx_cases']:
            setattr(pkg, attr, np.hstack([getattr(pkg, attr), zeros4]))
        pkg.pkg_Z_c = np.vstack([pkg.pkg_Z_c, np.full((2, pkg.pkg_Z_c.shape[1]), 100.0)])

    return pkg


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def read_ParamConfigFile(paramFile, OP):
    """Read COM parameter config file (MATLAB lines 9601–10258).

    paramFile: path to .csv, .xls/.xlsx, or .mat file
    OP: SimpleNamespace with pre-initialized fields (GET_FD, TDMODE,
        CONFIG2MAT_ONLY, SAVE_CONFIG2MAT set by caller)
    Returns: (param, OP) where param and OP are SimpleNamespace objects
    """
    param = SimpleNamespace()

    # ---- File path handling (MATLAB: fileparts) ----
    filepath = os.path.dirname(paramFile)
    name, ext = os.path.splitext(os.path.basename(paramFile))
    matcongfile = os.path.join(filepath, name + '.mat') if filepath else name + '.mat'

    # ---- Load parameter sheet ----
    parameter = _load_parameter_sheet(paramFile)

    OP.SAVE_KEYWORD_FILE = _xls_param(parameter, 'SAVE_KEYWORD_FILE', False, 0)

    # ---- Parse .START / .END package blocks ----
    first_column_data = [row[0] if row else '' for row in parameter]
    start_data_rows = [i for i, v in enumerate(first_column_data) if v == '.START']
    if start_data_rows:
        end_data_rows = [i for i, v in enumerate(first_column_data) if v == '.END']
        if len(start_data_rows) != len(end_data_rows):
            raise ValueError('Number of .START and .END must be the same')
        first_start_row = start_data_rows[0]
        special_parameter = parameter
        parameter = parameter[:first_start_row]
        param.PKG = SimpleNamespace()
        for j in range(len(start_data_rows)):
            this_block = special_parameter[start_data_rows[j] + 1: end_data_rows[j]]
            pkg_name = str(special_parameter[start_data_rows[j]][1])
            param.PKG.__dict__[pkg_name] = _read_pkg_params(this_block)

    # ---- PKG_NAME ----
    param.PKG_NAME = _xls_param(parameter, 'PKG_NAME', False, '')
    if _isnan(param.PKG_NAME):
        param.PKG_NAME = ''
    if _isempty(param.PKG_NAME):
        param.PKG_NAME = []
    else:
        param.PKG_NAME = str(param.PKG_NAME).split()
    if param.PKG_NAME and not hasattr(param, 'PKG'):
        raise ValueError('PKG_NAME can only be used if .START blocks for package parameters are used')
    for pkg_n in param.PKG_NAME:
        if not hasattr(param.PKG, pkg_n):
            raise ValueError(f'Package Block "{pkg_n}" not found')

    # ---- Legacy layout parameters ----
    # ML 10161-10163 makes these double row vectors, not lists; a Python list
    # multiplies by repeating, which is not what a MATLAB vector does.
    param.c = np.array([0.4e-12, 0.4e-12])
    param.alen = np.array([20.0, 30.0, 550.0])
    param.az = np.array([100.0, 120.0, 100.0])

    param.kappa1 = _xls_param(parameter, 'kappa1', True, 1)
    param.kappa2 = _xls_param(parameter, 'kappa2', True, 1)

    OP.dynamic_txffe = _xls_param(parameter, 'Dynamic TXFFE', False, 1)
    OP.FloatingDFE_Development = _xls_param(parameter, 'FloatingDFE_Development', False, 1)

    # ---- Frequency parameters ----
    param.fb = _xls_param(parameter, 'f_b') * 1e9
    param.f2 = _xls_param(parameter, 'f_2', True, param.fb / 1e9) * 1e9
    param.f2_ild = _xls_param(parameter, 'f_2_ILD', True, param.f2 / 1e9) * 1e9
    param.max_start_freq = _xls_param(parameter, 'f_min') * 1e9
    param.f1 = _xls_param(parameter, 'f_1', True, param.max_start_freq / 1e9) * 1e9
    param.max_freq_step = _xls_param(parameter, 'Delta_f') * 1e9
    param.tx_ffe_c0_min = _xls_param(parameter, 'c(0)', False)

    # ---- TX FFE taps ----
    if OP.dynamic_txffe:
        pre_count = 1
        while True:
            p, found = _xls_param_txffe(parameter, f'c(-{pre_count})')
            if not found:
                break
            param.__dict__[f'tx_ffe_cm{pre_count}_values'] = p
            pre_count += 1
        post_count = 1
        while True:
            p, found = _xls_param_txffe(parameter, f'c({post_count})')
            if not found:
                break
            param.__dict__[f'tx_ffe_cp{post_count}_values'] = p
            post_count += 1
    else:
        param.tx_ffe_cm4_values = _xls_param(parameter, 'c(-4)', True, 0)
        param.tx_ffe_cm3_values = _xls_param(parameter, 'c(-3)', True, 0)
        param.tx_ffe_cm2_values = _xls_param(parameter, 'c(-2)', True, 0)
        param.tx_ffe_cm1_values = _xls_param(parameter, 'c(-1)', True, 0)
        param.tx_ffe_cp1_values = _xls_param(parameter, 'c(1)', True, 0)
        param.tx_ffe_cp2_values = _xls_param(parameter, 'c(2)', True, 0)
        param.tx_ffe_cp3_values = _xls_param(parameter, 'c(3)', True, 0)

    param.ndfe = _xls_param(parameter, 'N_b')
    param.N_v = _xls_param(parameter, 'N_v', True, param.ndfe)
    param.D_p = _xls_param(parameter, 'D_p', True, 4)
    param.N_p = _xls_param(parameter, 'N_p', True, 400)
    param.N_bx = _xls_param(parameter, 'N_bx', True, param.ndfe)

    # ---- Floating tap parameters ----
    param.N_bg = _xls_param(parameter, 'N_bg', True, 0)
    param.N_bf = _xls_param(parameter, 'N_bf', True, 6)
    param.N_bmax = _xls_param(parameter, 'N_bmax', True, param.ndfe)
    param.N_bmax = _xls_param(parameter, 'N_f', True, param.ndfe)  # N_f supersedes N_bmax
    param.N_f = _xls_param(parameter, 'N_f', True, param.ndfe)
    if param.N_bg == 0:
        param.N_bmax = param.ndfe
    param.bmaxg = _xls_param(parameter, 'bmaxg', True, 0.2)

    param.B_float_RSS_MAX = _xls_param(parameter, 'B_float_RSS_MAX', True, 0)
    param.N_tail_start = _xls_param(parameter, 'N_tail_start', True, 0)

    param.dfe_delta = _xls_param(parameter, 'N_b_step', True, 0)
    param.ffe_pre_tap_len = _xls_param(parameter, 'ffe_pre_tap_len', True, 0)
    param.RxFFE_cmx = param.ffe_pre_tap_len
    param.ffe_post_tap_len = _xls_param(parameter, 'ffe_post_tap_len', True, 0)
    param.RxFFE_cpx = param.ffe_post_tap_len
    param.ffe_tap_step_size = _xls_param(parameter, 'ffe_tap_step_size', True, 0)
    param.RxFFE_stepz = param.ffe_tap_step_size
    param.ffe_main_cursor_min = _xls_param(parameter, 'ffe_main_cursor_min', True, 1)
    param.ffe_pre_tap1_max = _xls_param(parameter, 'ffe_pre_tap1_max', True, 0.7)
    param.ffe_post_tap1_max = _xls_param(parameter, 'ffe_post_tap1_max', True, 0.7)
    param.ffe_tapn_max = _xls_param(parameter, 'ffe_tapn_max', True, 0.7)
    param.ffe_backoff = _xls_param(parameter, 'ffe_backoff', True, 0)

    OP.RxFFE = bool(param.RxFFE_cmx != 0 or param.RxFFE_cpx != 0)

    param.num_ui_RXFF_noise = _xls_param(parameter, 'num_ui_RXFF_noise', True, 2048)
    param.flim = _xls_param(parameter, 'flim', True, 1000e9)
    if param.flim < 1000:
        param.flim *= 1e9

    # ---- CTLE parameters ----
    param.g_DC_HP_values = _xls_param(parameter, 'g_DC_HP', True, [])
    param.f_HP = 1e9 * np.atleast_1d(np.asarray(_xls_param(parameter, 'f_HP_PZ', True, []), dtype=float))
    param.f_HP_Z = 1e9 * np.atleast_1d(np.asarray(_xls_param(parameter, 'f_HP_Z', True, []), dtype=float))
    param.f_HP_P = 1e9 * np.atleast_1d(np.asarray(_xls_param(parameter, 'f_HP_P', True, []), dtype=float))

    param.Min_VEO = _xls_param(parameter, 'EH_min', True, 0)
    param.Max_VEO = _xls_param(parameter, 'EH_max', True, np.inf)
    param.Min_VEO_Test = _xls_param(parameter, 'EH_min_test', True, 0)
    param.Min_VEO_Test = _xls_param(parameter, 'Min_VEO_Test', True, param.Min_VEO_Test)

    param.CTLE_type = _xls_param(parameter, 'CTLE_type', False, 'CL93')
    if not _isempty(param.g_DC_HP_values):
        param.CTLE_type = 'CL120d'
    if not _isempty(param.f_HP_Z):
        param.CTLE_type = 'CL120e'

    param.ctle_gdc_values = _xls_param(parameter, 'g_DC', True)
    # ML 10268-10270. The 1e9 converts a GHz config entry to Hz, but the
    # DEFAULT is param.fb/4, and param.fb is already in Hz (ML 10175), so an
    # omitted key gives 1e9 * 1.328e10 = 1.328e19 Hz. The trailing comments on
    # those lines read "fp1 is in GHz", so the defaults look like they were
    # meant to be param.fb/4/1e9.
    #
    # UPSTREAM DEFECT, reproduced deliberately (owner ruling 2026-09-23). The
    # port used to divide by 1e9 and get the physically sensible value, which
    # is a silent correction of the reference and exactly what
    # docs/VERIFICATION.md says not to do. A 1.328e19 Hz pole makes the CTLE
    # transfer function identically 1, i.e. no filtering. COM Octave on a
    # config omitting all three:
    #     param.CTLE_fp1 = 1.328125e+19   (fb/4 = 1.328125e+10)
    #     param.CTLE_fp2 = 5.3125e+19
    #     param.CTLE_fz  = 1.328125e+19
    # Every shipped workbook supplies f_p1, f_p2 and f_z, so a normal run does
    # not reach the defaults. Reported for the COM ad hoc as item A12.
    param.CTLE_fp1 = 1e9 * _xls_param(parameter, 'f_p1', True, param.fb / 4)
    param.CTLE_fp2 = 1e9 * _xls_param(parameter, 'f_p2', True, param.fb)
    param.CTLE_fz = 1e9 * _xls_param(parameter, 'f_z', True, param.fb / 4)

    if param.CTLE_type == 'CL93':
        param.ctle_gdc_values = _xls_param(parameter, 'g_DC', True)
        param.CTLE_fp1 = 1e9 * _xls_param(parameter, 'f_p1', True, param.fb / 4)
        param.CTLE_fp2 = 1e9 * _xls_param(parameter, 'f_p2', True, param.fb)
        param.CTLE_fz = 1e9 * _xls_param(parameter, 'f_z', True, param.fb / 4)
    elif param.CTLE_type == 'CL120d':
        param.g_DC_HP_values = _xls_param(parameter, 'g_DC_HP', True, [])
        param.f_HP = 1e9 * np.atleast_1d(np.asarray(_xls_param(parameter, 'f_HP_PZ', True, []), dtype=float))
    elif param.CTLE_type == 'CL120e':
        gdc = np.atleast_1d(np.asarray(param.ctle_gdc_values, dtype=float))
        param.CTLE_fz = np.atleast_1d(np.asarray(param.CTLE_fz, dtype=float)) / (10.0 ** (gdc / 20.0))

    param.GDC_MIN = _xls_param(parameter, 'GDC_MIN', True, 0)
    param.cursor_gain = _xls_param(parameter, 'crusor_gain', True, 0)

    # ---- Signal amplitude parameters ----
    param.a_thru = _xls_param(parameter, 'A_v', True, 0.5)
    param.a_fext = _xls_param(parameter, 'A_fe', True, 0.5)
    param.a_next = _xls_param(parameter, 'A_ne', True, 0.5)
    param.a_icn_fext = _xls_param(parameter, 'A_ft', True, param.a_fext)
    param.a_icn_next = _xls_param(parameter, 'A_nt', True, param.a_next)
    param.levels = _xls_param(parameter, 'L')
    param.specBER = _xls_param(parameter, 'DER_0')
    param.DER_CDR = _xls_param(parameter, 'DER_CDR', True, 1e-2)
    param.N_qb = _xls_param(parameter, 'N_qb', True, 0)
    param.P_qc = _xls_param(parameter, 'P_qc', True, 2 * param.specBER)
    # ---- MATLAB version switch --------------------------------------------
    # 4p15p0 is the baseline: it is what the 208-case MATLAB reference corpus
    # was produced with, and what the correlation
    # (FOM bit-exact 198/208) is evidence for. 4p16p0 behaviour is opt-in so
    # that evidence is not silently invalidated.
    #
    #   python sicopr.py ... --matlab-version 4p16p0
    #   import sicopr; sicopr.COM_MATLAB_VERSION = '4p16p0'
    #   or the config keyword 'COM Version'
    #
    # Read before anything that branches on it -- the Clip Method default is
    # the first such consumer, immediately below.
    param.matlab_version = str(_xls_param(parameter, 'COM Version', False,
                                          COM_MATLAB_VERSION)).strip()
    if param.matlab_version not in ('4p15p0', '4p16p0', '4p17p0'):
        raise ValueError("unknown COM Version %r (expected '4p15p0', '4p16p0' "
                         "or '4p17p0')" % param.matlab_version)
    # Release names share one fixed shape, so string order is release order;
    # a 4p16p0 change carries into 4p17p0.
    _v416 = param.matlab_version >= '4p16p0'
    _v417 = param.matlab_version >= '4p17p0'

    # 4p16p0 L10262 flipped this default from 'Fast' to 'Slow'. Configs that
    # name the keyword are unaffected either way; configs that omit it change
    # behaviour, which is why it follows the version.
    param.clip_method = _xls_param(parameter, 'Clip Method', False,
                                   'Slow' if _v416 else 'Fast')
    param.P_peak = _xls_param(parameter, 'P_peak', True, param.specBER)
    param.pass_threshold = _xls_param(parameter, 'COM Pass threshold', False, 0)
    param.add_rx_noise = _xls_param(parameter, 'add_rx_noise', True, param.pass_threshold)
    param.ERL_pass_threshold = _xls_param(parameter, 'ERL Pass threshold', False, 0)
    param.VEC_pass_threshold = _xls_param(parameter, 'VEC Pass threshold', False, 0)

    param.sigma_RJ = _xls_param(parameter, 'sigma_RJ')
    param.A_DD = _xls_param(parameter, 'A_DD')
    param.eta_0 = _xls_param(parameter, 'eta_0')

    param.trunc = _xls_param(parameter, 'trunc', True, 128)
    param.trunc = _xls_param(parameter, 'N_tc', True, param.trunc)
    param.Q_budget_adj = _xls_param(parameter, 'Q_budget_adj', True, 0)
    param.SNDR = _xls_param(parameter, 'SNR_TX', True)
    param.S_tn_w_AM = _xls_param(parameter, 'S_tn_w_AM', False, 0)
    param.R_LM = _xls_param(parameter, 'R_LM')
    # ML 10235 is `xls_parameter(parameter, 'M', 32)`: the THIRD argument of
    # xls_parameter is eval_if_string, not default_value, so the 32 is a
    # truthy eval flag and M has NO default -- it is mandatory. COM Octave,
    # a config without M: error (the reference's own missing-parameter path,
    # which calls the undefined missingParameter). The port quietly used 32.
    param.samples_per_ui = _xls_param(parameter, 'M', True)
    param.ts_sample_adj_range = _xls_param(parameter, 'sample_adjustment', True, [0, 0])
    param.ts_anchor = _xls_param(parameter, 'ts_anchor', True, 0)

    # ---- DFE limits ----
    param.bmax = np.full(int(param.ndfe), float(_xls_param(parameter, 'b_max(1)'))) if param.ndfe > 0 else np.array([])
    if _isempty(param.bmax):
        param.bmin = param.bmax.copy()
    else:
        bmin1 = _xls_param(parameter, 'b_min(1)', True, -param.bmax[0])
        param.bmin = np.full(int(param.ndfe), float(bmin1))

    if param.ndfe >= 2:
        bmax2 = _xls_param(parameter, 'b_max(2..N_b)', True, 0.2)
        bmax2_arr = np.atleast_1d(np.asarray(bmax2, dtype=float)).ravel()
        if len(bmax2_arr) == 1:
            param.bmax[1:] = float(bmax2_arr[0])
        else:
            param.bmax[1:len(bmax2_arr) + 1] = bmax2_arr[:len(param.bmax) - 1]
        bmin2 = _xls_param(parameter, 'b_min(2..N_b)', True, -float(bmax2_arr[0]))
        bmin2_arr = np.atleast_1d(np.asarray(bmin2, dtype=float)).ravel()
        if len(bmin2_arr) == 1:
            param.bmin[1:] = float(bmin2_arr[0])
        else:
            param.bmin[1:len(bmin2_arr) + 1] = bmin2_arr[:len(param.bmin) - 1]

    # ---- CTLE qual ----
    gqual_raw = _xls_param(parameter, 'G_Qual', True, [])
    param.gqual = np.atleast_2d(np.asarray(gqual_raw, dtype=float)) if not _isempty(gqual_raw) else np.array([])
    g2qual_raw = _xls_param(parameter, 'G2_Qual', True, [])
    param.g2qual = np.atleast_1d(np.asarray(g2qual_raw, dtype=float)) if not _isempty(g2qual_raw) else np.array([])
    if not _isempty(param.gqual) or not _isempty(param.g2qual):
        if param.gqual.shape[0] != len(param.g2qual):
            raise ValueError('gqual and g2qual size mismatch')
        if param.gqual.ndim < 2 or param.gqual.shape[1] != 2:
            raise ValueError('gqual must be Nx2 matrix')

    # ---- Package electrical parameters ----
    param.C_pkg_board = _xls_param(parameter, 'C_p', True, 0) * 1e-9
    param.C_diepad = _xls_param(parameter, 'C_d', True, 0) * 1e-9
    param.L_comp = _xls_param(parameter, 'L_s', True, 0) * 1e-9
    param.C_bump = _xls_param(parameter, 'C_b', True, 0) * 1e-9
    # ML 10273 is `xls_parameter(parameter, 'R_0', 50)`, the same shape as M:
    # mandatory, and eval_if_string is the truthy 50, so a string value IS
    # evaluated. COM Octave, R_0 = '[50 50]': param.Z0 = [50 50]; without
    # R_0 at all: error. The port used 50 as a default and did not eval.
    param.Z0 = _xls_param(parameter, 'R_0', True)
    param.C_v = _xls_param(parameter, 'C_v', True, 0) * 1e-9
    param.R_diepad = _xls_param(parameter, 'R_d', True, [50, 50])
    param.Z_t = _xls_param(parameter, 'Z_t', True, param.Z0)
    param.TR_TDR = _xls_param(parameter, 'TR_TDR', True, 8e-3)

    # ---- Package trace lengths ----
    _default_zp_str = '[ 8 24 30 45 ; 1 1 1 1 ; 1 1 1 1 ; 0.5 0.5 0.5 0.5 ]'
    _default_zp = _parse_matlab_matrix(_default_zp_str).T

    def _load_zp_cases(key):
        raw = _xls_param(parameter, key, True, _default_zp_str)
        if isinstance(raw, str):
            raw = _parse_matlab_matrix(raw)
        arr = np.atleast_2d(np.asarray(raw, dtype=float))
        return arr.T  # MATLAB transposes: .'

    param.z_p_tx_cases = _load_zp_cases('z_p (TX)')
    ncases, mele = param.z_p_tx_cases.shape
    if mele == 2:
        param.flex = 2
    elif mele == 4:
        param.flex = 4
    elif mele == 1:
        param.flex = 1
    else:
        raise ValueError('config file syntax error: z_p (TX) must have 1, 2, or 4 columns')

    # board parameters. MATLAB L10379-10380 sets these on EVERY read, whatever
    # Include PCB says, and the port set neither. COM Octave, this config:
    # param.C_0 and param.C_1 are present and 0, and were the only two fields
    # of 210 that the reference had and the port did not.
    # Their absence is what made parameter_size_adjustment's make_length2 loop
    # skip a missing field instead of erroring, and it would have crashed
    # add_brd (sicopr.py L7549) on any config with Include PCB set.
    param.C_0 = _xls_param(parameter, 'C_0', True, 0) * 1e-9
    param.C_1 = _xls_param(parameter, 'C_1', True, 0) * 1e-9

    def _load_zp_check(key):
        arr = _load_zp_cases(key)
        if arr.shape != (ncases, mele):
            raise ValueError('All TX, NEXT, FEXT, Rx cases must agree')
        return arr

    param.z_p_next_cases = _load_zp_check('z_p (NEXT)')
    param.z_p_fext_cases = _load_zp_check('z_p (FEXT)')
    param.z_p_rx_cases = _load_zp_check('z_p (RX)')

    # ---- Package model parameters ----
    param.pkg_gamma0_a1_a2 = _xls_param(parameter, 'package_tl_gamma0_a1_a2', True,
                                         np.array([0.0, 1.734e-3, 1.455e-4]))
    param.pkg_tau = _xls_param(parameter, 'package_tl_tau', True, 6.141e-3)

    _default_Zc_str = '[92 92 ; 70 70; 80 80; 100 100]'
    raw_zc = _xls_param(parameter, 'package_Z_c', True, _default_Zc_str)
    if isinstance(raw_zc, str):
        raw_zc = _parse_matlab_matrix(raw_zc)
    param.pkg_Z_c = np.atleast_2d(np.asarray(raw_zc, dtype=float)).T  # MATLAB .'
    ncases1, mele1 = param.pkg_Z_c.shape
    if mele1 != mele:
        raise ValueError('tx rx pairs must have the same number element entries as TX, NEXT, FEXT, Rx')

    if mele1 == 2:  # expand 2-element flex to 4
        for attr in ['z_p_fext_cases', 'z_p_next_cases', 'z_p_tx_cases', 'z_p_rx_cases']:
            setattr(param, attr + 'x', np.hstack([getattr(param, attr), np.zeros((ncases, 2))]))
            setattr(param, attr, getattr(param, attr + 'x'))
        param.pkg_Z_c = np.vstack([param.pkg_Z_c, np.full((2, param.pkg_Z_c.shape[1]), 100.0)])

    param.PKG_Tx_FFE_preset = _xls_param(parameter, 'PKG_Tx_FFE_preset', True, 0)

    # ---- Board parameters ----
    param.brd_gamma0_a1_a2 = _xls_param(parameter, 'board_tl_gamma0_a1_a2', True,
                                          np.array([0.0, 4.114e-4, 2.547e-4]))
    param.brd_tau = _xls_param(parameter, 'board_tl_tau', True, 6.191e-3)
    param.brd_Z_c = _xls_param(parameter, 'board_Z_c', True, 109.8)
    param.z_bp_tx = _xls_param(parameter, 'z_bp (TX)', True, 151)
    param.z_bp_next = _xls_param(parameter, 'z_bp (NEXT)', True, 72)
    param.z_bp_fext = _xls_param(parameter, 'z_bp (FEXT)', True, 72)
    param.z_bp_rx = _xls_param(parameter, 'z_bp (RX)', True, 151)

    # ---- Misc signal parameters ----
    param.snpPortsOrder = _xls_param(parameter, 'Port Order', True, [1, 3, 2, 4])
    param.delta_IL = _xls_param(parameter, 'delta_IL', False, 1)
    param.f_v = _xls_param(parameter, 'f_v', True, 4)
    param.f_f = _xls_param(parameter, 'f_f', True, 4)
    param.f_n = _xls_param(parameter, 'f_n', True, 4)
    param.f_r = _xls_param(parameter, 'f_r', True, 4)
    param.fb_BT_cutoff = _xls_param(parameter, 'TDR_f_BT_3db', True, 0.4730)
    param.BTorder = _xls_param(parameter, 'BTorder', False, 4)
    param.RC_Start = _xls_param(parameter, 'RC_Start', False, param.fb / 2)
    param.RC_end = _xls_param(parameter, 'RC_end', False, param.fb * param.f_r)
    param.beta_x = _xls_param(parameter, 'beta_x', False, 0)
    param.rho_x = _xls_param(parameter, 'rho_x', False, 0.618)
    param.tfx = _xls_param(parameter, 'fixture delay time', True, -1)
    param.Grr_limit = _xls_param(parameter, 'Grr_limit', False, 1)
    param.Grr = _xls_param(parameter, 'Grr', False, param.Grr_limit)
    param.Gx = _xls_param(parameter, 'Gx', False, 0)
    if param.Gx == 0:
        pass  # keep existing Grr
    elif param.Gx == 1:
        param.Grr = 2

    # Hansel adaptive-local-search branch: 0 = legacy local search, 1 = adaptive.
    # Mainline in 4p16p0 (L10390); previously only in his branch file.
    param.NonZeroLSMethod = _xls_param(parameter, 'Non-zero Local Search Method', True, 0)
    # 4p16p0 L10391. Empty means "use the built-in min_radius rule"; a positive
    # value overrides it. Read in both modes -- an unused parameter is harmless,
    # and reading it keeps the config surface identical across versions.
    param.Overwrite_Min_Radius = _xls_param(parameter, 'Overwrite Minimum Radius', True, None)
    param.LOCAL_SEARCH = _xls_param(parameter, 'Local Search', True, 0)
    param.Tukey_Window = _xls_param(parameter, 'Tukey_Window', True, 0)
    param.zero_pad_tukey_window_in_fb = _xls_param(parameter, 'zero_pad_tukey_window_in_fb', True, 0)
    param.Noise_Crest_Factor = _xls_param(parameter, 'Noise_Crest_Factor', True, 0)
    param.AC_CM_RMS = _xls_param(parameter, 'AC_CM_RMS', True, 0)
    param.ACCM_MAX_Freq = _xls_param(parameter, 'ACCM_MAX_Freq', True, param.fb)
    param.T_O = _xls_param(parameter, 'T_O', True, 0)
    param.T_O = _xls_param(parameter, 'T_h', True, param.T_O)
    param.samples_for_C2M = _xls_param(parameter, 'samples_for_C2M', True, 100)

    OP.Histogram_Window_Weight = _xls_param(parameter, 'Histogram_Window_Weight', False, 'rectangle')
    param.sigma_r = _xls_param(parameter, 'sigma_r', True, 0.020)
    param.Qr = _xls_param(parameter, 'Qr', True, param.sigma_r)
    _ql_default = param.T_O / param.Qr / 1000 if param.Qr != 0 else 0.0
    param.QL = _xls_param(parameter, 'QL', True, _ql_default)

    # ---- Experimental parameters ----
    param.skew_ps = _xls_param(parameter, 'skew_ps', True, 0)
    param.imb_Z_fctr = _xls_param(parameter, 'imb_Z_fctr', True, 1)
    param.imb_C_fctr = _xls_param(parameter, 'imb_C_fctr', True, 1)
    param.awgn_mv = param.AC_CM_RMS
    param.flip = _xls_param(parameter, 'flip', True, 0)
    param.f_hp = _xls_param(parameter, 'f_hp', True, 0)
    param.Q = _xls_param(parameter, 'Q', True, 0)

    # ---- Floating DFE / RXFFE flags ----
    param.Floating_RXFFE = False
    param.Floating_DFE = False
    if param.N_bg > 0:
        param.Floating_DFE = True
    if OP.RxFFE:
        param.Floating_DFE = False
        if param.N_bg > 0:
            param.Floating_RXFFE = True

    # ---- Skew parameters ----
    param.Txpskew = _xls_param(parameter, 'Txpskew', True, 0)
    param.Txnskew = _xls_param(parameter, 'Txnskew', True, 0)
    param.Rxpskew = _xls_param(parameter, 'Rxpskew', True, 0)
    param.Rxnskew = _xls_param(parameter, 'Rxnskew', True, 0)
    # 4p17p0 L11034: dB/GHz threshold on d(CICP residual)/df for the apparent
    # channel bandwidth (get_ACBW). Earlier releases do not read it.
    if _v417:
        param.T_dev = _xls_param(parameter, 'T_dev', False, 1)

    # ---- OP flags ----
    OP.TIMESTAMP = _xls_param(parameter, 'TIMESTAMP', False, 0)
    OP.WRITE_CSV_TRANSPOSED = _xls_param(parameter, 'WRITE_CSV_TRANSPOSED', False, 0)
    OP.DO_NOT_COMPUTE_COM = _xls_param(parameter, 'DO_NOT_COMPUTE_COM', False, 0)
    OP.DO_NOT_COMPUTE_COM = _xls_param(parameter, 'NO_COM', False, OP.DO_NOT_COMPUTE_COM)
    OP.include_pcb = _xls_param(parameter, 'Include PCB', False)
    OP.exit_if_deployed = _xls_param(parameter, 'exit if deployed', False, 0)
    OP.INCLUDE_CTLE = _xls_param(parameter, 'INCLUDE_CTLE', False, 1)
    OP.EXE_MODE = _xls_param(parameter, 'EXE_MODE', False, 1)
    OP.INCLUDE_FILTER = _xls_param(parameter, 'INCLUDE_TX_RX_FILTER', False, 1)
    OP.force_pdf_bin_size = _xls_param(parameter, 'Force PDF bin size', False, 0)
    OP.BinSize = _xls_param(parameter, 'PDF bin size', False, 1e-5)
    OP.DEBUG = _xls_param(parameter, 'DIAGNOSTICS', False, False)
    OP.DISPLAY_WINDOW = _xls_param(parameter, 'DISPLAY_WINDOW', False, True)
    OP.CSV_REPORT = _xls_param(parameter, 'CSV_REPORT', False, True)
    OP.SAVE_TD = _xls_param(parameter, 'SAVE_TD', False, False)
    OP.SAVE_FIGURES = _xls_param(parameter, 'SAVE_FIGURES', False, False)
    OP.SAVE_FIGURE_to_CSV = _xls_param(parameter, 'SAVE_FIGURE_to_CSV', False, False)
    OP.GET_FD = _xls_param(parameter, 'Display frequency domain', False, OP.GET_FD)
    OP.INC_PACKAGE = _xls_param(parameter, 'INC_PACKAGE', False, True)
    if not OP.INC_PACKAGE:
        print(' Warning!!! INC_PACKAGE=0 not fully supported')
    OP.EW = _xls_param(parameter, 'EW', False, False)
    OP.IDEAL_TX_TERM = _xls_param(parameter, 'IDEAL_TX_TERM', False, False)
    if OP.IDEAL_TX_TERM:
        print(' Warning!!! IDEAL_TX_TERM not supported')
    OP.IDEAL_RX_TERM = _xls_param(parameter, 'IDEAL_RX_TERM', False, False)
    if OP.IDEAL_RX_TERM:
        print(' Warning!!! IDEAL_RX_TERM not supported')
    OP.TDMODE = _xls_param(parameter, 'TDMODE', False, OP.TDMODE)
    OP.FT_COOP = _xls_param(parameter, 'FT_COOP', False, False)

    # ML 10459 passes no default, so RESULT_DIR is mandatory. COM Octave, a
    # config without it: error. The port defaulted to ''.
    result_dir = _xls_param(parameter, 'RESULT_DIR')
    if isinstance(result_dir, (int, float)):
        result_dir = ''
    OP.RESULT_DIR = str(result_dir).replace('\\', os.sep)
    today = datetime.date.today().strftime('%d-%b-%Y')
    OP.RESULT_DIR = OP.RESULT_DIR.replace('{date}', today)

    OP.BREAD_CRUMBS = _xls_param(parameter, 'BREAD_CRUMBS', False, False)
    OP.BREAD_CRUMBS_FIELDS = _xls_param(parameter, 'BREAD_CRUMBS_FIELDS', False, '')
    OP.COM_CONTRIBUTION_CURVES = _xls_param(parameter, 'COM_CONTRIBUTION', False, 0)
    OP.ENFORCE_CAUSALITY = _xls_param(parameter, 'Enforce Causality', False, 0)
    OP.EC_REL_TOL = _xls_param(parameter, 'Enforce Causality REL_TOL', False, 1e-2)
    OP.EC_DIFF_TOL = _xls_param(parameter, 'Enforce Causality DIFF_TOL', False, 1e-3)
    OP.EC_PULSE_TOL = _xls_param(parameter, 'Enforce Causality pulse start tolerance', False, 0.01)
    OP.pkg_len_select = _xls_param(parameter, 'z_p select', True, 1)
    OP.RX_CALIBRATION = _xls_param(parameter, 'RX_CALIBRATION', False, False)
    OP.PSDRXCAL = _xls_param(parameter, 'PSDRXCAL', False, False)
    OP.RIT_REF_PTR = _xls_param(parameter, 'RIT_REF_PTR', False, 'clause_179')
    OP.sigma_bn_STEP = _xls_param(parameter, 'Sigma BBN step', False, 5e-3)
    OP.BBN_Q_factor = _xls_param(parameter, 'BBN Q factor', False, 5)
    OP.force_BBN_Q_factor = _xls_param(parameter, 'Force BBN Q factor', False, False)
    OP.transmitter_transition_time = _xls_param(parameter, 'T_r', True, 8e-3)
    OP.RL_norm_test = _xls_param(parameter, 'ERL_FOM', False, 1)
    OP.T_r_meas_point = _xls_param(parameter, 'T_r_meas_point', False, 0)
    OP.T_r_filter_type = _xls_param(parameter, 'T_r_filter_type', False, 0)
    OP.FORCE_TR = _xls_param(parameter, 'FORCE_TR', False, False)
    if OP.FORCE_TR:
        OP.T_r_meas_point = 0
        OP.T_r_filter_type = 1
    OP.TDR = _xls_param(parameter, 'TDR', False, False)
    OP.TDR_duration = _xls_param(parameter, 'TDR_duration', False, 5)
    OP.N = _xls_param(parameter, 'N', False, 0)
    OP.WC_PORTZ = _xls_param(parameter, 'WC_PORTZ', False, False)
    OP.T_k = _xls_param(parameter, 'T_k', False, 0.6) * 1e-9
    OP.ERL_ONLY = _xls_param(parameter, 'ERL_ONLY', False, 0)
    OP.ERL = _xls_param(parameter, 'ERL', False, False)
    OP.PTDR = 1 if OP.ERL else 0
    OP.Report_Modal_ERL = _xls_param(parameter, 'Report_Modal_ERL', False, '0')
    OP.SHOW_BRD = _xls_param(parameter, 'SHOW_BRD', False, 0)
    if OP.WC_PORTZ:
        OP.TDR = 1
    OP.TDR_W_TXPKG = _xls_param(parameter, 'TDR_W_TXPKG', False, 0)
    OP.Bessel_Thomson = _xls_param(parameter, 'Bessel_Thomson', False, False)
    OP.TDR_Butterworth = _xls_param(parameter, 'TDR_Butterworth', False, True)
    OP.Butterworth = _xls_param(parameter, 'Butterworth', False, 1)
    OP.Raised_Cosine = _xls_param(parameter, 'Raised_Cosine', False, 0)
    OP.inc_reflect_board = _xls_param(parameter, 'inc_reflect_board', False, 0)
    OP.AUTO_TFX = _xls_param(parameter, 'AUTO_TFX', False, 0)
    OP.LIMIT_JITTER_CONTRIB_TO_DFE_SPAN = _xls_param(
        parameter, 'LIMIT_JITTER_CONTRIB_TO_DFE_SPAN', False, False)
    OP.impulse_response_truncation_threshold = _xls_param(
        parameter, 'Impulse response truncation threshold', False, 1e-3)
    OP.interp_sparam_mag = _xls_param(
        parameter, 'S-parameter magnitude extrapolation policy', False, 'linear_trend_to_DC')
    OP.interp_sparam_phase = _xls_param(
        parameter, 'S-parameter phase extrapolation policy', False, 'extrap_cubic_to_dc_linear_to_inf')
    OP.ZERO_PAD = _xls_param(parameter, 'ZERO_PAD', False, 0)
    OP.PMD_type = _xls_param(parameter, 'PMD_type', False, 'C2C')
    OP.PHY = _xls_param(parameter, 'PHY', False, OP.PMD_type)
    if str(OP.PHY).upper() == 'C2M':
        OP.EW = True
    else:
        param.T_O = 0
    if param.Min_VEO != 0 and str(OP.PHY).upper() == 'C2C':
        OP.PHY = 'C2Mcom'

    OP.TDECQ = _xls_param(parameter, 'TDECQ', False, 0)
    if isinstance(OP.TDECQ, str) and OP.TDECQ.lower() not in ('none', 'vma'):
        raise ValueError(f'{OP.TDECQ} unrecognized TDECQ keyword')

    OP.RUNTAG = _xls_param(parameter, 'RUNTAG', False, '')
    if _isnan(OP.RUNTAG):
        OP.RUNTAG = ''
    if not isinstance(OP.RUNTAG, str):
        OP.RUNTAG = str(OP.RUNTAG)

    OP.CDR = _xls_param(parameter, 'CDR', False, 'MM')
    OP.Optimize_loop_speed_up = _xls_param(parameter, 'Optimize_loop_speed_up', True, 0)
    OP.use_simple_EP_model = _xls_param(parameter, 'Use simple error propagation model', False, False)
    OP.nburst = _xls_param(parameter, 'Max burst length calculated', False, 0)
    OP.COM_EP_margin = _xls_param(parameter, 'Error propagation COM margin', False, 0)
    OP.USE_ETA0_PSD = _xls_param(parameter, 'USE_ETA0_PSD', False, 0)
    OP.SAVE_CONFIG2MAT = _xls_param(parameter, 'SAVE_CONFIG2MAT', False, 0)
    OP.PLOT_CM = _xls_param(parameter, 'PLOT_CM', False, 0)
    OP.CM_MASK_REPORT = _xls_param(parameter, 'CM_MASK_REPORT', False, 0)
    OP.fraction_of_F_range_start_extrap_from = _xls_param(
        parameter, 'fraction_of_F_range_start_extrap_from', True, 0.75)
    OP.COMPUTE_RILN = _xls_param(parameter, 'COMPUTE_RILN', False, 0)
    OP.COMPUTE_TDILN = _xls_param(parameter, 'COMPUTE_TDILN', False, OP.COMPUTE_RILN)
    OP.SAVE_KEYWORD_FILE = _xls_param(parameter, 'SAVE_KEYWORD_FILE', False, 0)
    OP.SNR_TXwC0 = _xls_param(parameter, 'SNR_TXwC0', False, 0)
    OP.MLSD = _xls_param(parameter, 'MLSD', False, 0)
    OP.MLSE = _xls_param(parameter, 'MLSE', False, OP.MLSD)
    if OP.MLSE != 0:
        if param.T_O != 0:
            raise ValueError('MLSD not presently supported for VEC')
        if OP.COM_CONTRIBUTION_CURVES != 0:
            print('Warning: COM_CONTRIBUTION_CURVES not functional yet with MLSE')
            OP.COM_CONTRIBUTION_CURVES = 0
    OP.RXFFE_FLOAT_CTL = _xls_param(parameter, 'RXFFE FLOAT CTL', False, 'FOM')
    OP.RXFFE_TAP_CONSTRAINT = _xls_param(parameter, 'RXFFE TAP CONSTRAINT', False, 'Unity Cursor')
    if OP.MLSE and param.ndfe == 0:
        raise ValueError('At least DFE 1 must be set to use MLSE')
    OP.TIME_AXIS = _xls_param(parameter, 'TIME_AXIS', False, 'UI')
    OP.Do_XT_Noise = _xls_param(parameter, 'Do_XT_Noise', False, 1)
    OP.FFE_SNR = _xls_param(parameter, 'FFE_SNR', False, 1)
    OP.Do_Colored_Noise = _xls_param(parameter, 'Do_Colored_Noise', False, 1)
    OP.Do_White_Noise = _xls_param(parameter, 'Do_White_Noise', False, 0)
    OP.FFE_OPT_METHOD = _xls_param(parameter, 'FFE_OPT_METHOD', False, 'MMSE')
    OP.TS_SRCH_MODE = _xls_param(parameter, 'TS_SRCH_MODE', False, 'full-sweep')

    # ---- TDMODE disables FD operations ----
    if OP.TDMODE:
        OP.GET_FD = False
        OP.ERL_ONLY = 0
        OP.ERL = 0
        OP.PTDR = 0
        OP.TDR = 0
        OP.RX_CALIBRATION = 0
        OP.PSDRXCAL = 0

    # ---- RX Calibration + MMSE swap ----
    if OP.RX_CALIBRATION:
        if OP.FFE_OPT_METHOD == 'MMSE' and OP.RxFFE:
            OP.PSDRXCAL = 1
            OP.RX_CALIBRATION = 0
    # 4p17p0 L11207: report the apparent channel bandwidth (FD_Processing ->
    # get_ACBW). Off by default; earlier releases have no such keyword, so a
    # workbook that sets it changes nothing under them.
    OP.ACBW = _xls_param(parameter, 'ACBW', False, 0) if _v417 else 0

    # ---- Validate PSDRXCAL ----
    if OP.PSDRXCAL:
        if OP.RX_CALIBRATION:
            raise ValueError('RX Calibration must be turned off when PSDRXCAL is enabled')
        if OP.FFE_OPT_METHOD != 'MMSE':
            raise ValueError('FFE Method must be "MMSE" when PSDRXCAL is enabled')
        if not OP.RxFFE:
            raise ValueError('RxFFE must be enabled when PSDRXCAL is enabled')

    # ---- Optional mat file save ----
    if getattr(OP, 'SAVE_CONFIG2MAT', False) or getattr(OP, 'CONFIG2MAT_ONLY', False):
        try:
            import scipy.io
            scipy.io.savemat(matcongfile, {'parameter': parameter})
        except Exception:
            pass

    OP.SNDR_REF = _xls_param(parameter, 'SNDR_REF', False, 0)
    preset_txffe = _xls_param(parameter, 'preset_txffe', True, [])
    if not _isempty(preset_txffe):
        arr = np.atleast_2d(np.asarray(preset_txffe, dtype=float))
        param.preset = []
        for ii in range(arr.shape[0]):
            ps = SimpleNamespace()
            ps.txffe = arr[ii, :]
            param.preset.append(ps)

    # ---- Package name substitution ----
    if param.PKG_NAME:
        if len(param.PKG_NAME) == 1:
            param.PKG_NAME = [param.PKG_NAME[0], param.PKG_NAME[0]]
        tx_rx_fields = ['C_pkg_board', 'R_diepad']
        tx_rx_fields_matrix = ['pkg_Z_c']
        tx_fields = ['z_p_tx_cases', 'z_p_fext_cases', 'pkg_gamma0_a1_a2', 'pkg_tau', 'a_thru', 'a_fext']
        rx_fields = ['z_p_rx_cases', 'a_next', 'z_p_next_cases']
        tx_pkg = param.PKG.__dict__[param.PKG_NAME[0]]
        rx_pkg = param.PKG.__dict__[param.PKG_NAME[1]]

        for fld in tx_rx_fields:
            tx_v = np.atleast_1d(np.asarray(getattr(tx_pkg, fld), dtype=float))
            rx_v = np.atleast_1d(np.asarray(getattr(rx_pkg, fld), dtype=float))
            setattr(param, fld, np.array([tx_v[0], rx_v[1] if len(rx_v) > 1 else rx_v[0]]))

        for fld in tx_rx_fields_matrix:
            tx_v = np.atleast_2d(getattr(tx_pkg, fld))[0, :]
            rx_v = np.atleast_2d(getattr(rx_pkg, fld))[1, :] if np.atleast_2d(getattr(rx_pkg, fld)).shape[0] > 1 else np.atleast_2d(getattr(rx_pkg, fld))[0, :]
            setattr(param, fld, np.vstack([tx_v, rx_v]))

        for fld in tx_fields:
            setattr(param, fld, getattr(tx_pkg, fld))

        for fld in rx_fields:
            setattr(param, fld, getattr(rx_pkg, fld))

    return param, OP


if __name__ == '__main__':
    import tempfile
    # Smoke test with a minimal CSV (all mandatory parameters)
    minimal_csv = '''\
f_b,53.125
f_min,0.05
Delta_f,0.01
c(0),0.68
N_b,4
b_max(1),0.7
L,4
DER_0,1e-4
sigma_RJ,0.01
A_DD,0.05
eta_0,1.7e-4
R_LM,1
Include PCB,0
g_DC,-1
SNR_TX,30
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(minimal_csv)
        fname = f.name

    OP = SimpleNamespace(GET_FD=False, TDMODE=0, CONFIG2MAT_ONLY=False, SAVE_CONFIG2MAT=False)
    try:
        param, OP = read_ParamConfigFile(fname, OP)
        print(f'fb={param.fb:.3f} Hz, ndfe={param.ndfe}, levels={param.levels}')
        print('Smoke test PASSED')
    finally:
        os.unlink(fname)
