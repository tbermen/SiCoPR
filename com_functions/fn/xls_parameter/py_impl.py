import math
import re

import numpy as np

_SENTINEL = object()


def xls_parameter(param_sheet, param_name, eval_if_string=False, default_value=_SENTINEL):
    """Case-insensitive lookup in 2D param_sheet (MATLAB lines 11408-11493).

    param_sheet: list-of-lists.
    Returns p (the value in the cell one column to the right of the match).
    Raises KeyError if not found and no default_value was provided.
    Raises ValueError if multiple matches found.
    The SAVE_KEYWORD_FILE block is dead code (always false) and is omitted.
    """
    name_lower = param_name.lower()
    matches = [
        (r, c)
        for r, row in enumerate(param_sheet)
        for c, cell in enumerate(row)
        if isinstance(cell, str) and cell.lower() == name_lower
    ]

    if len(matches) == 0:
        if default_value is _SENTINEL:
            raise KeyError(
                f'Mandatory parameter "{param_name}" is missing or incorrect'
            )
        # MATLAB assigns `p = default_value` and then falls through to the
        # `ischar(p) && eval_if_string` line below, so a *default* that is a
        # string is evaluated too. COM Octave:
        #     xls_parameter(sheet, 'ZZ', true, '1+1')             -> 2
        #     xls_parameter(sheet, 'ZZ', true, '[92 92 ; 70 70]') -> 2x2 matrix
        #     xls_parameter(sheet, 'ZZ', 0,    '1+1')             -> '1+1'
        # Returning here instead handed back the unevaluated string. That is
        # reachable: line 10403 of the reference reads
        #     param.pkg_Z_c = xls_parameter(parameter, 'package_Z_c', true,
        #                                   '[92 92 ; 70 70; 80 80; 100 100]').'
        # so a config without package_Z_c transposed a string.
        p = default_value
    elif len(matches) > 1:
        raise ValueError(
            f'{len(matches)} occurrences of "{param_name}" found. Please recheck spreadsheet'
        )
    else:
        r, c = matches[0]
        p = param_sheet[r][c + 1]

    # MATLAB's eval(p), through the MATLAB subset a value cell is written in
    # (xls_eval below), never Python's eval: that read '[92 92 ; 70 70]' as a
    # SyntaxError and '0.15*ones(1,3)' as a NameError, and ran any Python text.
    if isinstance(p, str) and eval_if_string:
        p = xls_eval(p)
    return p


# ---------------------------------------------------------------------------
# xls_eval: MATLAB's eval() of a value cell, for the MATLAB a cell is written in
#
# The reference reads every value cell it asks to evaluate with eval(p)
# (ML 12452 in 4p17p0; xls_parameter_txffe, ML 12536, the same), for a .xlsx,
# a .csv (xlsread, ML 10623) and a .mat (load, ML 10621) alike. So a cell may
# hold any MATLAB expression, and configs in circulation use vector ones:
# 802.3ck writes b_max(2..N_b) as [0.3 0.2*ones(1,4) 0.1*ones(1,6)]. The port
# read brackets of plain numbers only and refused those (finding F19).
#
# This evaluates the subset configs use, with MATLAB's semantics, and nothing
# else: numbers, Inf and NaN; + - * / .* ./ with MATLAB's implicit expansion
# (* is a matrix product unless one side is a scalar); unary + and -;
# parentheses; the colon a:b and a:s:b; ones and zeros with 0, 1 or 2 integer
# sizes; and bracket concatenation with MATLAB's whitespace rules ([1 -2] is
# two elements, [1 - 2] one). Text outside the subset raises MatlabValueError,
# where MATLAB either errors too (an unknown name, a syntax error, a fraction
# as a size) or evaluates something no COM config uses; it is never handed to
# Python's eval.
# ---------------------------------------------------------------------------

class MatlabValueError(ValueError):
    """A value cell the MATLAB subset cannot read (see xls_eval)."""


_XE_TOKEN = re.compile(r'(?P<ws>[ \t]+)|(?P<nl>\r?\n)'
                       r'|(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'
                       r'|(?P<id>[A-Za-z_]\w*)'
                       r'|(?P<op>\.\*|\./|[-+*/:,;()\[\]])')

# Colon: start + k*step, the count taken with a tolerant floor and the last
# element clamped to the limit when the sum overshoots it, as COM Octave does
# (findings F06 and F12): '[0:0.3:1]' stops at 0.8999999999999999, '[0:0.1:0.3]'
# ends at exactly 0.3.
_XE_COLON_TOL = 3.0 * np.finfo(float).eps

_XE_CONSTANTS = {'Inf': math.inf, 'inf': math.inf, 'NaN': math.nan, 'nan': math.nan}


def _xe_size(v, fname):
    """One size argument of ones/zeros: a whole number, negative read as 0.
    COM Octave: ones(1,2.5) -> error, 'conversion of 2.5 to int64_t value
    failed' (MATLAB: 'Size inputs must be integers')."""
    if v.size != 1:
        raise MatlabValueError('%s: a size must be a scalar' % fname)
    x = float(v.ravel()[0])
    if not math.isfinite(x) or x != math.floor(x):
        raise MatlabValueError('%s: size %r is not a whole number' % (fname, x))
    return max(int(x), 0)


def _xe_filled(fname, args, fill):
    if len(args) > 2:
        raise MatlabValueError('%s with %d sizes: only 0, 1 or 2 are read'
                               % (fname, len(args)))
    sizes = [_xe_size(a, fname) for a in args]
    if not sizes:
        sizes = [1, 1]
    elif len(sizes) == 1:
        sizes = sizes * 2                              # ones(n) is n-by-n
    return np.full(sizes, fill, dtype=float)


_XE_FUNCTIONS = {'ones': 1.0, 'zeros': 0.0}
_XE_SPACE = ('ws', 'nl')


def _xe_colon(parts):
    for p in parts:
        if p.size != 1:
            raise MatlabValueError('a colon operand must be a scalar')
    vals = [float(p.ravel()[0]) for p in parts]
    if len(vals) == 2:
        start, step, stop = vals[0], 1.0, vals[1]
    else:
        start, step, stop = vals
    if step == 0 or any(math.isnan(v) for v in (start, step, stop)):
        return np.zeros((1, 0))
    n = int(math.floor((stop - start) / step + _XE_COLON_TOL)) + 1
    if n <= 0:
        return np.zeros((1, 0))
    x = start + np.arange(n) * step
    if (step > 0 and x[-1] > stop) or (step < 0 and x[-1] < stop):
        x[-1] = stop                                    # colon end clamp
    return x.reshape(1, -1)


def _xe_binary(op, a, b):
    if op == '*' and a.size != 1 and b.size != 1:
        if a.shape[1] != b.shape[0]:
            raise MatlabValueError('inner matrix dimensions must agree for *')
        return a @ b
    if op == '/' and b.size != 1:
        raise MatlabValueError('/ by a matrix is not read; only by a scalar')
    for da, db in zip(a.shape, b.shape):
        if da != db and da != 1 and db != 1:
            raise MatlabValueError('operands of %s have sizes %s and %s'
                                   % (op, a.shape, b.shape))
    if op == '+':
        return a + b
    if op == '-':
        return a - b
    if op in ('*', '.*'):
        return a * b
    return a / b                                        # '/' and './'


def _xe_horzcat(parts):
    parts = [p for p in parts if p.size]               # [1 ones(1,0) 2] is [1 2]
    if not parts:
        return np.zeros((0, 0))
    if len({p.shape[0] for p in parts}) > 1:
        raise MatlabValueError('dimensions of arrays being concatenated are not consistent')
    return np.hstack(parts)


def _xe_vertcat(rows):
    rows = [r for r in rows if r.size]
    if not rows:
        return np.zeros((0, 0))
    if len({r.shape[1] for r in rows}) > 1:
        raise MatlabValueError('vertical dimensions mismatch')
    return np.vstack(rows)


def _xe_tokens(text):
    """Tokens as (kind, text, space_before), with the element separators that
    whitespace makes inside brackets written in as ',' (MATLAB's rules: a
    space between two operands separates them; before a sign it separates only
    when the sign is not followed by a space, so [1 -2] is two elements and
    [1 - 2], [1- 2] one). Inside parentheses whitespace never separates."""
    raw, pos = [], 0
    while pos < len(text):
        m = _XE_TOKEN.match(text, pos)
        if not m:
            raise MatlabValueError('unexpected character %r' % text[pos])
        kind = m.lastgroup
        raw.append((kind, m.group(0)))
        pos = m.end()
    toks, ws, stack = [], False, []
    for i, (kind, t) in enumerate(raw):
        if kind == 'ws':
            ws = True
            continue
        if kind == 'nl':
            if stack and stack[-1] == '[':
                toks.append(('op', ';', ws))           # a new line in brackets is a row
            ws = True
            continue
        if ws and stack and stack[-1] == '[' and toks:
            prev = toks[-1]
            ends = prev[0] in ('num', 'id') or prev[1] in (')', ']')
            if ends:
                starts = kind in ('num', 'id') or t in ('(', '[')
                if t in ('+', '-'):
                    nxt = raw[i + 1][0] if i + 1 < len(raw) else 'ws'
                    starts = nxt not in _XE_SPACE
                if starts:
                    toks.append(('op', ',', True))
        toks.append((kind, t, ws))
        ws = False
        if t in ('(', '['):
            stack.append(t)
        elif t in (')', ']') and stack:
            stack.pop()
    return toks


class _XEParser:
    """Recursive descent over MATLAB's precedence: colon, then binary + -,
    then * / .* ./, then unary + -. Every value is a 2-D float array."""

    def __init__(self, toks):
        self.toks, self.i = toks, 0

    def peek(self):
        return self.toks[self.i][1] if self.i < len(self.toks) else None

    def take(self, want=None):
        if self.i >= len(self.toks):
            raise MatlabValueError('unexpected end of text')
        t = self.toks[self.i]
        if want is not None and t[1] != want:
            raise MatlabValueError('expected %r, found %r' % (want, t[1]))
        self.i += 1
        return t

    def range_(self):
        parts = [self.additive()]
        while self.peek() == ':' and len(parts) < 3:
            self.take()
            parts.append(self.additive())
        return parts[0] if len(parts) == 1 else _xe_colon(parts)

    def additive(self):
        v = self.multiplicative()
        while self.peek() in ('+', '-'):
            op = self.take()[1]
            v = _xe_binary(op, v, self.multiplicative())
        return v

    def multiplicative(self):
        v = self.unary()
        while self.peek() in ('*', '/', '.*', './'):
            op = self.take()[1]
            v = _xe_binary(op, v, self.unary())
        return v

    def unary(self):
        if self.peek() in ('+', '-'):
            op = self.take()[1]
            v = self.unary()
            return -v if op == '-' else v
        return self.primary()

    def primary(self):
        kind, t, _ = self.take()
        if kind == 'num':
            return np.array([[float(t)]])
        if kind == 'id':
            if t in _XE_CONSTANTS:
                return np.array([[_XE_CONSTANTS[t]]])
            if t not in _XE_FUNCTIONS:
                raise MatlabValueError('%r is not read in a value cell' % t)
            args = []
            if self.peek() == '(':
                self.take()
                if self.peek() != ')':
                    args.append(self.range_())
                    while self.peek() == ',':
                        self.take()
                        args.append(self.range_())
                self.take(')')
            return _xe_filled(t, args, _XE_FUNCTIONS[t])
        if t == '(':
            v = self.range_()
            self.take(')')
            return v
        if t == '[':
            return self.matrix()
        raise MatlabValueError('unexpected %r' % t)

    def matrix(self):
        rows, row = [], []
        while True:
            t = self.peek()
            if t == ']':
                self.take()
                rows.append(_xe_horzcat(row))
                return _xe_vertcat(rows)
            if t == ',':
                self.take()
            elif t == ';':
                self.take()
                rows.append(_xe_horzcat(row))
                row = []
            else:
                row.append(self.range_())


def xls_eval(text, as_matrix=False):
    """MATLAB's eval() of a value cell, in the shapes the port uses.

    A 1-by-N result is a 1-D array, anything taller 2-D, an empty one
    np.array([]). A scalar is a float, unless the text is a bracket literal
    ('[46.25]') or as_matrix is set: then it is a 1-element array, as the
    config reader has always returned a bracket. Raises MatlabValueError for
    text outside the subset described above.
    """
    toks = _xe_tokens(text.strip())
    if not toks:
        raise MatlabValueError('empty expression')
    parser = _XEParser(toks)
    v = parser.range_()
    if parser.i != len(toks):
        raise MatlabValueError('unexpected %r' % toks[parser.i][1])
    if v.size == 0:
        return np.array([])
    if v.size == 1 and not as_matrix and toks[0][1] != '[':
        return float(v[0, 0])
    if v.shape[0] == 1:
        return v[0]
    return v
