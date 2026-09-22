import math

import numpy as np


def _num2str(x):
    """MATLAB num2str for one numeric value.

    num2str is NOT str(): it prints five significant digits for a non-integer
    and the exact digits for an integer, which is what lands in results.csv.
    The precision rule is MATLAB's own --
        integer-valued : min(floor(log10(|x|)) + 1, 16) significant digits
        otherwise      : max(min(floor(log10(|x|)) + 5, 16), 5)
    -- and a complex value formats both parts under whichever rule the pair
    selects. COM Octave, num2str of each:
        pi -> 3.1416          1/3 -> 0.33333       1.23456789 -> 1.2346
        -0.000123456 -> -0.00012346               0.1+0.2 -> 0.3
        42.5 -> 42.5          123456.7 -> 123456.7  1e10+0.5 -> 10000000000.5
        1e-13 -> 1e-13        1e-5 -> 1e-05        eps -> 2.2204e-16
        123456789 -> 123456789                    1e15 -> 1000000000000000
        1e16 -> 1e+16         1e20 -> 1e+20        2^53 -> 9007199254740992
        realmax -> 1.797693134862316e+308         true -> 1     -0 -> -0
        NaN -> NaN            Inf -> Inf           -Inf -> -Inf
        1+2i -> 1+2i          pi+1i -> 3.1416+1i   1/3+1i/7 -> 0.33333+0.14286i
    """
    if np.iscomplexobj(x):
        re, im = float(x.real), float(x.imag)
        prec = _num2str_precision([re, im])
        return '%s%s%si' % (_g(re, prec), '-' if im < 0 else '+',
                            _g(abs(im), prec))
    v = float(x)
    return _g(v, _num2str_precision([v]))


def _num2str_precision(values):
    """The %g precision MATLAB's num2str picks for a set of values."""
    finite = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    biggest = max((abs(v) for v in finite), default=0.0)
    dgt = 0 if biggest == 0 else int(math.floor(math.log10(biggest)))
    if all(v == math.trunc(v) for v in finite):
        return min(dgt + 1, 16)
    return max(min(dgt + 5, 16), 5)


def _g(v, prec):
    """%.*g with MATLAB's spelling of the non-finite values."""
    if math.isnan(v):
        return 'NaN'
    if math.isinf(v):
        return 'Inf' if v > 0 else '-Inf'
    return '%.*g' % (prec, v)


def _mat2str(a):
    """MATLAB mat2str: 15 significant digits, space between columns, ';'
    between rows, no brackets around a scalar, and NO line wrapping.

    np.array2string is none of those -- it prints 8 digits and breaks the line
    at 75 characters, which would split a CSV field across two lines.
    COM Octave, mat2str of each:
        [1 2 3] -> [1 2 3]                 [1 2;3 4] -> [1 2;3 4]
        [1;2;3] -> [1;2;3]                 [] -> []
        pi -> 3.14159265358979             [1.5] -> 1.5
        [pi 1/3] -> [3.14159265358979 0.333333333333333]
        [0.5 -0.25 1/3] -> [0.5 -0.25 0.333333333333333]
        [1e-13 1] -> [1e-13 1]             logical([1 0 1]) -> [true false true]
    """
    a = np.asarray(a)
    if a.size == 0:
        return '[]'
    if a.dtype == bool:
        fmt = lambda v: 'true' if v else 'false'      # noqa: E731
    else:
        fmt = lambda v: _g(float(v), 15)              # noqa: E731
    # A 1-D Python sequence is a MATLAB row; a column can only arrive as 2-D.
    rows = a if a.ndim == 2 else a.reshape(1, -1)
    body = ';'.join(' '.join(fmt(v) for v in row) for row in rows)
    return body if a.size == 1 else '[' + body + ']'


def _item_to_str(v):
    if hasattr(v, '__dict__') or isinstance(v, dict):
        return 'struct'
    if isinstance(v, str):
        return v
    if v is None:
        return ''
    try:
        if len(v) == 0:
            return ''
    except TypeError:
        pass
    arr = np.asarray(v)
    if arr.size == 1:
        return _num2str(arr.flat[0])
    return '"%s"' % _mat2str(arr)


def Write_CSV(output_args, csv_file):
    """Write struct fields as single-row CSV header + data (MATLAB lines 4742-4767)."""
    d = output_args if isinstance(output_args, dict) else vars(output_args)
    fields = list(d.keys())
    if not fields:
        # COM Octave, Write_CSV(struct()): str2csv indexes cell_tmp{2,end} on
        # an empty cell and stops --
        #   error: cell_tmp(_,0): subscripts must be either integers
        #          1 to (2^63)-1 or logicals
        #   str2csv at line 6 column 1 / Write_CSV at line 21 column 1
        raise IndexError('cell_tmp(_,0): subscripts must be either integers '
                         '1 to (2^63)-1 or logicals')
    values = [_item_to_str(d[k]) for k in fields]
    # newline='' so the '\n' below is written as one byte. MATLAB's fopen 'w'
    # is binary, and COM Octave leaves LF alone; Python's text mode would turn
    # every line ending into CRLF on Windows.
    with open(csv_file, 'w', newline='') as fid:
        fid.write(','.join(fields) + '\n')
        fid.write(','.join(values) + '\n')
