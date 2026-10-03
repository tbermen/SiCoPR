"""Run a function from the COM reference under Octave, to pin as a test oracle.

A unit test that checks a Python port against a *reading* of the MATLAB cannot
catch a divergence that lives in a library default: `sigma = std(x)` and
`sigma = np.std(x)` read alike and are not alike. Catching that needs the
reference executed, not paraphrased.

This extracts a named function verbatim from
`octave/com_ieee8023_<ver>_octave_compat.m`, together with any subfunctions it
needs, runs it under Octave on inputs given from Python, and returns the
outputs. Nothing is rewritten, so what runs is the reference code.

Use it to *generate* values, then paste them into the unit test. Tests pin the
numbers rather than calling Octave, so the suite stays fast and runs where
Octave is not installed. Re-run this when the reference version changes.

    from octave_oracle import call
    out = call('interp_Sparam',
               args=["Sin(:).'", "fin(:).'", "fout(:).'",
                     "'linear_trend_to_DC'", "'extrap_cubic_to_dc_linear_to_inf'",
                     'OP', 'param'],
               inputs={'Sin': Sin, 'fin': fin, 'fout': fout},
               outputs=['Sout'],
               setup="OP.DEBUG=0; OP.ZERO_PAD=0; param=struct()",
               needs=['interp_Sparam'])

CLI, to print literals ready to paste:

    python tools/octave_oracle.py --list            # functions available
    python tools/octave_oracle.py --show std        # how Octave evaluates a snippet

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import argparse
import io
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import scipy.io

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

DEFAULT_VERSION = '4p17p0'


def compat_path(version=DEFAULT_VERSION):
    return os.path.join(_ROOT, 'octave',
                        'com_ieee8023_%s_octave_compat.m' % version)


def _find_octave():
    from xlsx_to_com_mat import find_octave
    return find_octave()


def _source(version):
    return io.open(compat_path(version), encoding='latin-1').read()


def function_names(version=DEFAULT_VERSION):
    """Every function defined in the compat file."""
    return sorted(set(re.findall(r'\nfunction\s+(?:\[[^\]]*\]|\w+)?\s*=?\s*'
                                 r'\.{0,3}\s*\n?\s*(\w+)\s*\(', _source(version))))


def extract(name, version=DEFAULT_VERSION):
    """The verbatim text of one function, or None."""
    src = _source(version)
    # `function [a,b] = name(...)`, `function name(...)`, and the
    # continuation form `function [a] = ...\n    name(...)`.
    pat = re.compile(r'\nfunction\b[^\n]*?[\[\s=]' + re.escape(name) + r'\s*\(')
    m = pat.search(src)
    if not m:
        pat = re.compile(r'\nfunction\b[^\n]*\.\.\.\s*\n\s*' + re.escape(name) + r'\s*\(')
        m = pat.search(src)
        if not m:
            return None
    start = m.start() + 1
    nxt = src.find('\nfunction ', start + 10)
    return src[start:nxt if nxt != -1 else len(src)]


def call(name, args, inputs, outputs, setup='', needs=(),
         version=DEFAULT_VERSION, workdir=None):
    """Run `name(*args)` under Octave and return {output: ndarray}.

    args     Octave expressions, as text, in call order.
    inputs   {name: array} placed in the workspace first.
    outputs  names to capture, in the order the function returns them.
    setup    Octave statements run before the call (build OP, param, ...).
    needs    other functions from the compat file to make available.
    """
    d = workdir or tempfile.mkdtemp(prefix='oct_oracle_')
    os.makedirs(d, exist_ok=True)
    for fn in set(list(needs) + [name]):
        body = extract(fn, version)
        if body is None:
            raise KeyError('%s is not defined in %s'
                           % (fn, os.path.basename(compat_path(version))))
        io.open(os.path.join(d, fn + '.m'), 'w', encoding='latin-1').write(body)

    fwd = lambda p: os.path.abspath(p).replace('\\', '/')
    scipy.io.savemat(os.path.join(d, 'in.mat'), inputs or {'dummy_': 0})
    out_mat = os.path.join(d, 'out.mat')
    lhs = '[%s]' % ', '.join(outputs) if len(outputs) > 1 else outputs[0]
    stmts = ["addpath('%s')" % fwd(d), "load('%s')" % fwd(os.path.join(d, 'in.mat'))]
    if setup:
        stmts.append(setup)
    stmts.append('%s = %s(%s)' % (lhs, name, ', '.join(args)))
    stmts.append("save('-v7','%s',%s)"
                 % (fwd(out_mat), ', '.join("'%s'" % o for o in outputs)))
    q = subprocess.run([_find_octave(), '--no-gui', '--no-window-system',
                        '--eval', ';'.join(stmts)],
                       cwd=d, capture_output=True, text=True, errors='replace')
    if not os.path.isfile(out_mat):
        raise RuntimeError('Octave produced no output for %s:\n%s'
                           % (name, (q.stdout + q.stderr)[-2000:]))
    got = scipy.io.loadmat(out_mat)
    return {o: got[o] for o in outputs}


def literal(x, places=17):
    """A Python literal for a scalar, at full double precision."""
    a = np.asarray(x).ravel()
    if a.size == 1:
        v = a[0]
        if np.iscomplexobj(a):
            return '%.*g%+.*gj' % (places, v.real, places, v.imag)
        return '%.*g' % (places, float(v))
    return '[' + ', '.join(literal(v, places) for v in a) + ']'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--version', default=DEFAULT_VERSION)
    ap.add_argument('--list', action='store_true',
                    help='list functions defined in the compat file')
    ap.add_argument('--has', metavar='NAME', help='is NAME extractable?')
    ap.add_argument('--show', metavar='EXPR',
                    help='evaluate an Octave expression and print it')
    a = ap.parse_args(argv)

    if a.list:
        names = function_names(a.version)
        print('%d functions in %s' % (len(names), os.path.basename(compat_path(a.version))))
        for n in names:
            print('   ', n)
        return 0
    if a.has:
        body = extract(a.has, a.version)
        print('%s: %s' % (a.has, 'found, %d chars' % len(body) if body else 'NOT FOUND'))
        return 0 if body else 1
    if a.show:
        d = tempfile.mkdtemp(prefix='oct_show_')
        fwd = lambda p: os.path.abspath(p).replace('\\', '/')
        out = os.path.join(d, 'o.mat')
        q = subprocess.run(
            [_find_octave(), '--no-gui', '--no-window-system', '--eval',
             "ans_ = %s; save('-v7','%s','ans_')" % (a.show, fwd(out))],
            capture_output=True, text=True, errors='replace')
        if os.path.isfile(out):
            print(literal(scipy.io.loadmat(out)['ans_']))
            return 0
        print((q.stdout + q.stderr)[-1500:])
        return 1
    ap.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
