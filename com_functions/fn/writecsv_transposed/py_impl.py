import csv
import math

import numpy as np


def _g(v, prec):
    """%.*g with MATLAB's spelling of the non-finite values."""
    if math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    return '%.*g' % (prec, v)


def _complex_num2str(z):
    """MATLAB num2str for one complex value: both parts under the precision
    the pair selects, joined by the sign of the imaginary part."""
    re, im = float(z.real), float(z.imag)
    parts = [v for v in (re, im) if not (math.isnan(v) or math.isinf(v))]
    biggest = max((abs(v) for v in parts), default=0.0)
    dgt = 0 if biggest == 0 else int(math.floor(math.log10(biggest)))
    if all(v == math.trunc(v) for v in parts):
        prec = min(dgt + 1, 16)
    else:
        prec = max(min(dgt + 5, 16), 5)
    return '%s%s%si' % (_g(re, prec), '-' if im < 0 else '+',
                        _g(abs(im), prec))


def _num2str(x):
    """MATLAB num2str for a real numeric value or array.

    Scalars: five significant digits for a non-integer, the exact digits for
    an integer. Arrays: every element right-aligned in a fixed field, the
    whole result then trimmed. The widths below reproduce COM Octave:
        [1 2 3]      -> '1  2  3'        [100 2]     -> '100    2'
        [1.5 2.5]    -> '1.5         2.5'
        [pi 1/3]     -> '3.1416     0.33333'
        [1e-13 1]    -> '1e-13           1'
        linspace(0,1,5) ->
            '0        0.25         0.5        0.75           1'
        []           -> ''               [1 2;3 4]   -> '1  2' / '3  4'
    NOT VERIFIED against MATLAB: MATLAB's num2str widens the field by one for
    an array containing a negative, and Octave does not. Every pinned test
    uses non-negative arrays, where the two rules agree, and scalars, which
    were checked directly.
    """
    a = np.asarray(x)
    if a.size == 0:
        return ''
    if a.dtype.kind == 'c':
        # Complex scalars are checked: COM Octave num2str gives 1+2i -> '1+2i',
        # pi+1i -> '3.1416+1i', 1/3+1i/7 -> '0.33333+0.14286i'. The field
        # widths num2str uses for a complex ARRAY were not checked, so those
        # are simply joined.
        return '  '.join(_complex_num2str(v) for v in a.ravel())
    a = a.astype(float)
    flat = a.ravel()
    finite = flat[np.isfinite(flat)]
    biggest = float(np.max(np.abs(finite))) if finite.size else 0.0
    dgt = 0 if biggest == 0 else int(math.floor(math.log10(biggest)))
    if finite.size == flat.size and np.all(finite == np.trunc(finite)):
        prec = min(dgt + 1, 16)
        width = prec + 2
    else:
        prec = max(min(dgt + 5, 16), 5)
        width = prec + 7

    rows = a.reshape(1, -1) if a.ndim < 2 else a
    text = [''.join('%*s' % (width, _g(float(v), prec)) for v in row)
            for row in rows]
    lead = min(len(r) - len(r.lstrip(' ')) for r in text)
    trail = min(len(r) - len(r.rstrip(' ')) for r in text)
    # strtrim on a char matrix drops the blank columns every row shares.
    # A matrix input therefore leaves several rows; MATLAB puts that
    # multi-row char array in one cell, and what writecell does with it could
    # not be checked here, so the rows are simply joined.
    return '\n'.join(r[lead:len(r) - trail] for r in text)


def _is_logical(v):
    return isinstance(v, (bool, np.bool_)) or (
        isinstance(v, np.ndarray) and v.dtype == bool)


def _is_numeric(v):
    if isinstance(v, (int, float, complex, np.number)):
        return True
    if isinstance(v, np.ndarray):
        return v.dtype.kind in 'iufc'
    # A list or tuple of numbers is how a MATLAB numeric row vector arrives
    # when the caller did not build an ndarray; a list of anything else is a
    # cell array, which MATLAB reports as '[unsupported]'.
    return (isinstance(v, (list, tuple)) and len(v) > 0
            and all(isinstance(e, (int, float, complex, np.number))
                    and not isinstance(e, bool) for e in v))


def _val_to_str(v):
    # MATLAB's order: ischar, isnumeric, islogical, isstruct, isempty, else.
    # islogical is lifted above isnumeric here only because a Python bool IS
    # an int; in MATLAB the two predicates are disjoint.
    if isinstance(v, str):
        return v
    if _is_logical(v):
        # ML 11399 is `string(val)`, which is "true"/"false", not "True".
        # A logical ARRAY gives a string array, and what writecell writes for
        # that could not be checked without MATLAB.
        flat = np.asarray(v).ravel()
        return ' '.join('true' if bool(b) else 'false' for b in flat)
    if _is_numeric(v):
        return _num2str(v)
    if hasattr(v, '__dict__') or isinstance(v, dict):
        return '[struct]'
    if v is None:
        return ''
    try:
        if len(v) == 0:
            return ''
    except TypeError:
        pass
    return '[unsupported]'


def writecsv_transposed(output_args, filename):
    """Write struct fields as two-column CSV (Field, Value) (MATLAB lines 11384-11407)."""
    d = output_args if isinstance(output_args, dict) else vars(output_args)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Field', 'Value'])
        for field, value in d.items():
            writer.writerow([field, _val_to_str(value)])
