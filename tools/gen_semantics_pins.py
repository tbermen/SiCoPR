"""Run COM Octave's own builtins on fixed inputs; write the literals.

The expected values in tests/test_matlab_semantics.py (LIVE SEMANTIC PARITY)
come from here: every Octave result is written as a Python literal to
pins_block.py in the output directory, to replace the _OCT block in the test.
The inputs are defined here and nowhere else derived, so the pins may ship.
Regenerate, never edit.

    python tools/gen_semantics_pins.py [OUTDIR]

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import json
import os
import subprocess
import sys

import numpy as np
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import octave_oracle as oo                                    # noqa: E402

HERE = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(
    __import__('tempfile').gettempdir(), 'sicopr_semantics_pins')

# ------------------------------------------------------------------ inputs
# Short decimals, so the literals stay readable; generic enough that no
# identity hides a difference.
IN = dict(
    v=np.array([0.8147, -0.9058, 0.127, 0.9134, -0.6324, 0.0975, 0.2785]),
    vn=np.array([0.8147, np.nan, 0.127, 0.9134, -0.6324, np.inf, 0.2785]),
    vc=np.array([0.5 + 0.2j, -0.3 + 0.9j, 0.7 - 0.4j, -0.1 - 0.6j, 0.25 + 0.0j]),
    M=np.array([[0.9575, 0.1576, 0.9572, 0.4218],
                [0.9649, 0.9706, 0.4854, 0.9157],
                [-0.7922, 0.9595, 0.8003, 0.0357]]),
    Mn=np.array([[0.9575, np.nan, 0.9572, 0.4218],
                 [0.9649, 0.9706, np.inf, 0.9157],
                 [-0.7922, 0.9595, 0.8003, 0.0357]]),
    Mc=np.array([[1 + 2j, 3 - 1j, 0.5 + 0.5j, -2 + 0j],
                 [0 - 1j, 2 + 2j, -1 + 0.25j, 1 - 3j],
                 [4 + 0j, -0.5 - 1j, 2 - 2j, 0.75 + 1j]]),
    t=np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 0.49999999999999994,
                -3.7, 3.7, np.nan, np.inf, -np.inf]),
    mx=np.array([-7.5, -3.0, -0.5, 0.0, 0.5, 3.0, 7.5, 5.2, -5.2]),
    xg=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    yg=np.array([10.0, 20.0, 35.0, 55.0, 60.0]),
    xq=np.array([0.5, 1.0, 1.7, 2.5, 3.3, 4.9, 5.0, 6.0]),
    fb=np.array([0.2, 0.3]),
    fa=np.array([1.0, -0.5, 0.25]),
    A=np.array([[4.0, -2.0, 1.0], [3.0, 6.0, -4.0], [2.0, 1.0, 8.0]]),
    bA=np.array([12.0, -25.0, 32.0]),
    Ao=np.array([[1.0, 2.0, 0.5], [2.0, -1.0, 3.0], [0.5, 1.5, -2.0],
                 [3.0, 0.25, 1.0], [-1.0, 2.0, 2.0]]),
    bo=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    Au=np.array([[1.0, 2.0, 0.5, 3.0, -1.0], [2.0, -1.0, 3.0, 0.25, 2.0],
                 [0.5, 1.5, -2.0, 1.0, 2.0]]),
    bu=np.array([1.0, 2.0, 3.0]),
    As=np.array([[1.0, 2.0], [2.0, 4.0]]),
    bs=np.array([1.0, 3.0]),
    ey=np.array([1e-12, 1e-9, 2e-9, 1e-5, 1e-3, 0.5, 1.5, 1.999]),
)

EXPR = [
    # name, octave expression
    ('std_v', 'std(v)'), ('std_vn', 'std(vn)'), ('std_vc', 'std(vc)'),
    ('std_M', 'std(M)'), ('std_Mn', 'std(Mn)'),
    ('var_v', 'var(v)'), ('var_M', 'var(M)'), ('var_vc', 'var(vc)'),
    ('mean_M', 'mean(M)'), ('mean_Mn', 'mean(Mn)'), ('mean_Mc', 'mean(Mc)'),
    ('sum_M', 'sum(M)'), ('sum_Mn', 'sum(Mn)'), ('sum_Mc', 'sum(Mc)'),
    ('sum_v', 'sum(v)'),
    ('round_t', 'round(t)'),
    ('mod_3', 'mod(mx, 3)'), ('mod_m3', 'mod(mx, -3)'), ('mod_0', 'mod(mx, 0)'),
    ('mod_07', 'mod(mx, 0.7)'),
    ('rem_3', 'rem(mx, 3)'), ('rem_m3', 'rem(mx, -3)'), ('rem_0', 'rem(mx, 0)'),
    ('rem_07', 'rem(mx, 0.7)'),
    ('interp_lin', "interp1(xg, yg, xq)"),
    ('interp_lin_x', "interp1(xg, yg, xq, 'linear', 'extrap')"),
    ('interp_pchip_x', "interp1(xg, yg, xq, 'pchip', 'extrap')"),
    ('interp_pchip_oct', "interp1(xg, yg, xq, 'pchip')"),
    ('filter_v', 'filter(fb, fa, v)'),
    ('filter_M', 'filter(fb, fa, M)'),
    ('fft_v', 'fft(v)'), ('fft_vc', 'fft(vc)'), ('fft_M', 'fft(M)'),
    ('fft_vn', 'fft(vn)'),
    ('norm_v', 'norm(v)'), ('norm_v1', 'norm(v, 1)'), ('norm_vinf', 'norm(v, Inf)'),
    ('norm_vc', 'norm(vc)'), ('norm_M', 'norm(M)'), ('norm_Mfro', "norm(M, 'fro')"),
    ('norm_Mc', 'norm(Mc)'), ('norm_M1', 'norm(M, 1)'),
    ('tr_Mc', "Mc.'"), ('ctr_Mc', "Mc'"), ('ctr_vc', "vc'"),
    ('ml_sq', 'A \\ bA(:)'), ('ml_over', 'Ao \\ bo(:)'), ('ml_under', 'Au \\ bu(:)'),
    ('ml_sing_oct', 'As \\ bs(:)'),
    ('erfcinv_oct', 'erfcinv(ey)'),
    ('erfc_rt_oct', 'erfc(erfcinv(ey))'),
]


def main():
    d = os.path.join(HERE, 'oct')
    os.makedirs(HERE, exist_ok=True)
    os.makedirs(d, exist_ok=True)
    scipy.io.savemat(os.path.join(d, 'in.mat'), IN)
    lines = ["load('in.mat');", "warning('off', 'all');"]
    for i, (name, ex) in enumerate(EXPR):
        # row vectors in, as MATLAB callers hold them
        lines.append('%s = %s;' % (name, ex))
    lines.append("save('-v7', 'out.mat', %s);" % ', '.join("'%s'" % n for n, _ in EXPR))
    with open(os.path.join(d, 'run.m'), 'w', encoding='utf-8') as fh:
        # loadmat vectors come back as 1xN rows, as MATLAB holds them
        fh.write('\n'.join(lines) + '\n')
    q = subprocess.run([oo._find_octave(), '--no-gui', '--no-window-system', '-q',
                        'run.m'], cwd=d, capture_output=True, text=True)
    if not os.path.isfile(os.path.join(d, 'out.mat')):
        print(q.stdout[-2000:], q.stderr[-2000:])
        raise SystemExit(1)
    ver = subprocess.run([oo._find_octave(), '--no-gui', '-q', '--eval',
                          'disp(version())'], capture_output=True, text=True).stdout.strip()
    out = scipy.io.loadmat(os.path.join(d, 'out.mat'))
    pins = {}
    for name, _ in EXPR:
        a = np.asarray(out[name])
        pins[name] = a
    np.savez(os.path.join(HERE, 'pins.npz'), **pins)
    with open(os.path.join(HERE, 'pins_block.py'), 'w', encoding='utf-8') as fh:
        fh.write('# COM Octave %s builtins, generated by tools/gen_semantics_pins.py\n' % ver)
        for name, ex in EXPR:
            fh.write('_OCT[%r] = %s  # %s\n' % (name, lit(pins[name]), ex))
    print('Octave', ver, '-', len(EXPR), 'results')


def lit1(x):
    if isinstance(x, complex) or np.iscomplexobj(x):
        x = complex(x)
        return 'complex(%s, %s)' % (lit1(x.real), lit1(x.imag))
    x = float(x)
    if np.isnan(x):
        return 'np.nan'
    if np.isinf(x):
        return 'np.inf' if x > 0 else '-np.inf'
    return repr(x)


def lit(a):
    a = np.asarray(a)
    if a.ndim == 2 and 1 in a.shape and a.size >= 1 and not (a.shape[0] > 1 and a.shape[1] > 1):
        shape = a.shape
        body = '[' + ', '.join(lit1(x) for x in a.ravel(order='F')) + ']'
        return 'np.array(%s).reshape(%s)' % (body, shape) if shape[0] > 1 else \
            'np.array(%s)' % body if shape[1] > 1 or shape == (1, 1) else body
    rows = ['[' + ', '.join(lit1(x) for x in r) + ']' for r in a]
    return 'np.array([%s])' % ', '.join(rows)


if __name__ == '__main__':
    main()
