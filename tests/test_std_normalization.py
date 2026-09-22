"""MATLAB's `std` normalises by N-1; numpy's `np.std` defaults to N.

Where the reference MATLAB uses `std` only to form a threshold -- as
`interp_Sparam`'s phase branch does, in
`abs(low_freq_gd - m) < sigma` -- the 1% difference between the two
normalisations is not a rounding difference. It decides membership. A single
group-delay sample crossing that threshold changes `lf_trend`, which rewrites
the first ten phase samples, which moves the extrapolated phase at DC.

That path runs only when the channel does not already start at DC
(`if fin[0] != 0`). Every channel in the 208-case MATLAB reference corpus
starts at DC, so the defect was invisible there while the 1368-case corpus,
where 648 of 1368 channels start at 10 MHz, showed FOM differences against
COM Octave of up to 3.34e-4 dB on 84 cases.

The oracle below is COM Octave's own `interp_Sparam`, extracted verbatim from
`octave/com_ieee8023_4p16p0_octave_compat.m` and run on the synthetic input
this test rebuilds. With `np.std` left at its default the DC value is 12%
away from it.

Run: python tests/test_std_normalization.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import sicopr  # noqa: E402

# COM Octave's interp_Sparam on the input built by _fixture(), options
# linear_trend_to_DC / extrap_cubic_to_dc_linear_to_inf, OP.ZERO_PAD = 0.
_OCTAVE_ORACLE = {
    0: 0.99551467368075997 + 0.00032860821987716881j,     # DC: the extrapolated point
    1: 0.94673980940840263 - 0.30724845157815045j,
    2: 0.80388826043639316 - 0.58360559209752749j,
    3: 0.58332679250451458 - 0.80222427311898858j,
    10: -0.98497540133478545 - 0.00051804913883706015j,
    100: 0.94951056414620771 - 0.00090894483179480043j,
}
_OCTAVE_SUM_ABS = 4016.1085110174708
_OCTAVE_N = 6001


def _fixture():
    """A synthetic channel that does not start at DC, whose 50th low-frequency
    group-delay sample sits between the N and N-1 thresholds."""
    fin = np.arange(1, 4001) * 10e6                     # 10 MHz .. 40 GHz
    rng = np.random.default_rng(5)
    f_ghz = fin / 1e9
    mag = 10 ** (-(0.4 * np.sqrt(f_ghz) + 0.05 * f_ghz) / 20)
    ph = -2 * np.pi * fin * 5e-9 + 2e-3 * rng.standard_normal(fin.size)
    Sin = mag * np.exp(1j * ph)
    fmax, fstep = 60e9, fin[2] - fin[1]
    fout = np.arange(0, round(fmax / fstep) + 1) * (fmax / round(fmax / fstep))
    return Sin, fin, fout


def test_fixture_is_sensitive_to_the_normalisation():
    """Guard the guard: if the crafted input stopped straddling the threshold,
    the oracle comparison below would pass no matter which std was used."""
    Sin, fin, _ = _fixture()
    gd = -np.diff(np.unwrap(np.angle(Sin))) / np.diff(fin)
    lf = gd[:50]
    m = np.median(lf)
    kept = [np.abs(lf - m) < np.std(lf, ddof=d) for d in (0, 1)]
    assert not np.array_equal(kept[0], kept[1]), (
        'fixture no longer distinguishes N from N-1 normalisation; '
        're-craft it before trusting this file')


def test_interp_Sparam_matches_octave():
    """Both copies of interp_Sparam must reproduce COM Octave at DC."""
    Sin, fin, fout = _fixture()
    OP = types.SimpleNamespace(DEBUG=0, ZERO_PAD=0)
    param = types.SimpleNamespace()
    for name in ('interp_Sparam', '_s21_to_impulse_DC__interp_Sparam'):
        fn = getattr(sicopr, name)
        got = np.asarray(fn(Sin, fin, fout, 'linear_trend_to_DC',
                            'extrap_cubic_to_dc_linear_to_inf', OP, param)).ravel()
        assert got.size == _OCTAVE_N, '%s: %d points, expected %d' % (
            name, got.size, _OCTAVE_N)
        for idx, want in _OCTAVE_ORACLE.items():
            rel = abs(got[idx] - want) / abs(want)
            assert rel < 1e-13, '%s: Sout[%d] is %r, COM Octave gives %r (rel %.2e)' % (
                name, idx, got[idx], want, rel)
        rel = abs(float(np.sum(np.abs(got))) - _OCTAVE_SUM_ABS) / _OCTAVE_SUM_ABS
        assert rel < 1e-13, '%s: sum|Sout| differs from COM Octave by %.2e' % (name, rel)


def test_every_std_declares_its_normalisation():
    """No `np.std` may rely on the numpy default. The reference MATLAB uses
    `std` in three places, each of which the assembler duplicates into an
    inlined copy, so the engine carries six."""
    src = open(os.path.join(_ROOT, 'sicopr.py'), encoding='utf-8').read()
    calls = re.findall(r'np\.std\([^)]*\)', src)
    assert calls, 'no np.std found in sicopr.py -- has the engine changed?'
    bare = [c for c in calls if 'ddof' not in c]
    assert not bare, (
        'np.std without an explicit ddof normalises by N, MATLAB std by N-1: %s' % bare)
    assert len(calls) == 6, (
        'expected 6 np.std sites (3 MATLAB sites x 2 copies), found %d: %s'
        % (len(calls), calls))


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS %s' % name)
            except AssertionError as exc:
                fails += 1
                print('FAIL %s\n     %s' % (name, exc))
    print('%d failed' % fails)
    sys.exit(1 if fails else 0)
