import math
import re

import numpy as np

_NUMBER = re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?')
_TOKEN = re.compile(r'^%s(?::%s){0,2}$' % (_NUMBER.pattern, _NUMBER.pattern))
# Octave's colon operator counts intervals with a tolerant floor; 3*eps is the
# tolerance it uses.  Without it '[-0.2:0.05:0.05]' comes out one point short,
# because (0.05+0.2)/0.05 is 4.999999999999999 in binary.
_COLON_TOL = 3.0 * np.finfo(float).eps


def _colon(token):
    """MATLAB a:b or a:s:b.  COM Octave: '[ -0.34:.02:0]' -> 18 points ending
    at exactly 0; '[ 0.14:.02:0]' -> empty; '[0.14:-.02:0]' -> 8 points.
    The last element is set to the limit when start + k*step overshoots it:
    COM Octave '[0:0.1:0.3]' ends at 0.3, not 0.30000000000000004 (finding F12)."""
    parts = token.split(':')
    if len(parts) == 2:
        start, step, stop = float(parts[0]), 1.0, float(parts[1])
    else:
        start, step, stop = (float(p) for p in parts)
    n = int(math.floor((stop - start) / step + _COLON_TOL)) + 1
    x = start + np.arange(max(n, 0)) * step
    if n > 0 and ((step > 0 and x[-1] > stop) or (step < 0 and x[-1] < stop)):
        x[-1] = stop                                    # colon end clamp
    return x


def _matlab_matrix(body):
    """The inside of a MATLAB '[...]': rows on ';', elements on commas and
    whitespace, each element a number or a colon range.  Raises ValueError on
    anything else, so the caller can fall through to Python's eval."""
    rows = []
    for row in body.split(';'):
        vals = []
        for token in row.replace(',', ' ').split():
            if not _TOKEN.match(token):
                raise ValueError(token)
            vals.extend(_colon(token) if ':' in token else [float(token)])
        rows.append(vals)
    if len(rows) == 1:
        return np.array(rows[0], dtype=float)
    return np.array(rows, dtype=float)


def _matlab_eval(s):
    """MATLAB eval() of a spreadsheet value string.

    Python's eval cannot read MATLAB's array syntax at all, and every TxFFE
    range in a COM config is written in it: the value cell for c(-1) in
    examples/akinwale_CR_22dB_VendorX is the string '[ -0.34:.02:0]', on which
    eval() raises SyntaxError.  COM Octave, same string through
    xls_parameter_txffe, returns the 18 points -0.34 .. 0 in steps of 0.02.
    Also pinned: '[1 0 0]' -> [1 0 0], '[1, 2, 3]' -> [1 2 3],
    '[1 2 3; 4 5 6]' -> 2x3, '[]' -> empty, '0.4' -> 0.4, '2*3' -> 6,
    'not code' -> "syntax error" (so junk must still raise).
    """
    t = s.strip()
    if t.startswith('[') and t.endswith(']'):
        return _matlab_matrix(t[1:-1])
    try:
        return float(t)
    except ValueError:
        pass
    if _TOKEN.match(t):
        return _colon(t)
    return eval(t)  # noqa: S307 — matches MATLAB eval() for parameter loading


def xls_parameter_txffe(param_sheet, param_name):
    """Case-insensitive search in 2D param_sheet; return (value, found) (MATLAB lines 11494-11515).

    param_sheet is a list-of-lists (rows × cols).
    Returns (p, 1) on match, (0, 0) if not found.
    Raises ValueError if multiple matches found.
    """
    name_lower = param_name.lower()
    matches = [
        (r, c)
        for r, row in enumerate(param_sheet)
        for c, cell in enumerate(row)
        if isinstance(cell, str) and cell.lower() == name_lower
    ]
    if len(matches) == 0:
        return 0, 0
    if len(matches) > 1:
        raise ValueError(
            f'{len(matches)} occurrences of "{param_name}" found. Please recheck spreadsheet'
        )
    r, c = matches[0]
    p = param_sheet[r][c + 1]
    if isinstance(p, str):
        p = _matlab_eval(p)
    return p, 1
