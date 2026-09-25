"""_sicopr_case.py -- one case through the engine, as `python -m sicopr` runs it,
plus every result field at full precision.

    python tools/_sicopr_case.py OUT.json VERSION CONFIG THRU [--fext F ...] [--next N ...]

The engine's own CSV report is written exactly as the command line writes it:
MATLAB's format, so five significant digits for most values (num2str, matched
to the reference in 0ce577c). A comparison against another engine needs the
doubles, so this also writes OUT.json, one object per package case, every
attribute of the result at full precision (JSON floats round-trip). Used by
octave_compare.run_sicopr; the COM calculation is the command line's, unchanged.

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import json
import sys

import numpy as np


def jsonable(v):
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, np.ndarray):
        if v.dtype.kind in 'fiub':
            return [jsonable(x) for x in v.ravel().tolist()]
        return str(v)
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, str) or v is None:
        return v
    return str(v)


def main(argv):
    out_json, version, config, thru = argv[:4]
    rest, fext, nxt, cur = argv[4:], [], [], None
    for a in rest:
        if a in ('--fext', '--next'):
            cur = fext if a == '--fext' else nxt
        else:
            cur.append(a)
    import sicopr
    sicopr.COM_MATLAB_VERSION = version            # as the --matlab-version flag does
    results = sicopr._run_com(config, len(fext), len(nxt), [thru] + fext + nxt,
                              export_mat=False)
    if results is None:
        print('ERROR: COM returned None.')
        return 1
    if not isinstance(results, list):
        results = [results]
    cases = [{k: jsonable(v) for k, v in sorted(vars(r).items())}
             for r in results if r is not None]
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(cases, fh)
    for i, c in enumerate(cases, 1):
        print('--- Case %d --- COM_dB = %r' % (i, c.get('COM_dB')))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
