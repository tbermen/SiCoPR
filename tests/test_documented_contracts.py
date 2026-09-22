"""Three-way check: the documented contract, the reference, and the port.

The Octave oracle tests pin what the reference code *does*. This file pins what
the two languages are *documented to do*, and holds all three against each
other:

    spec   -- MATLAB's documented rule, written out from first principles
    octave -- what COM Octave actually returns for the same probe
    port   -- what SiCoPR does

Each leg catches something different:

  spec vs octave   our reading of the documentation. If these disagree, either
                   the doc says something other than we think, or the reference
                   does something other than its doc. Both are worth knowing
                   before a port is built on the assumption.
  port vs octave   an ordinary porting bug.

`std` would have been caught here the moment the contract was written down,
because writing "MATLAB normalises by N-1, numpy by N" in one place forces the
comparison that reading either source alone does not.

Probes are chosen so the two conventions give *different* answers. A probe on
which they agree proves nothing, so each contract carries `separates`, the
value the numpy default would give, and the test asserts the probe really does
tell them apart.

Octave values were produced by tools/octave_oracle.py against
octave/com_ieee8023_4p16p0_octave_compat.m; re-run it if the reference moves.

Run: python tests/test_documented_contracts.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from audit_check import check, xcheck, finish   # noqa: E402
import sicopr                                    # noqa: E402


def _helper(stem):
    """Any surviving definition of a helper: the canonical if there is one,
    otherwise an inlined copy. Named copies come and go as the assembler's
    inlining changes, and this test is about the helper's SEMANTICS, not about
    which caller happens to carry it."""
    fn = getattr(sicopr, stem, None)
    if fn is not None:
        return fn
    for a in sorted(dir(sicopr)):
        if a.endswith('__' + stem):
            return getattr(sicopr, a)
    raise AttributeError('no definition of %s survives in sicopr.py' % stem)


X = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
R = np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5])
N = np.array([1.0, np.nan, 3.0])


def _same(a, b, tol=1e-15):
    a, b = np.atleast_1d(np.asarray(a, dtype=float)), np.atleast_1d(np.asarray(b, dtype=float))
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=tol, atol=tol, equal_nan=False)
    return bool(np.all(both_nan | close))


# ---------------------------------------------------------------------------
# name, MATLAB's documented rule, numpy's documented default, the probe,
# spec (the MATLAB rule written out), octave (measured), port (what we do),
# separates (what the numpy default would give -- must differ from octave)
# ---------------------------------------------------------------------------
CONTRACTS = [
    dict(
        name='std_normalises_by_N_minus_1',
        matlab='std(X) normalises by N-1 (w=0, the default weight)',
        numpy='np.std defaults to ddof=0, normalising by N',
        spec=lambda: math.sqrt(sum((v - X.mean()) ** 2 for v in X) / (len(X) - 1)),
        octave=2.1380899352993952,
        port=lambda: np.std(X, ddof=1),
        separates=lambda: np.std(X),
    ),
    dict(
        name='var_normalises_by_N_minus_1',
        matlab='var(X) normalises by N-1',
        numpy='np.var defaults to ddof=0',
        spec=lambda: sum((v - X.mean()) ** 2 for v in X) / (len(X) - 1),
        octave=4.5714285714285712,
        port=lambda: np.var(X, ddof=1),
        separates=lambda: np.var(X),
    ),
    dict(
        name='round_goes_half_away_from_zero',
        matlab='round(X) rounds halves away from zero',
        numpy='np.round rounds halves to the nearest even value',
        spec=lambda: np.sign(R) * np.floor(np.abs(R) + 0.5),
        octave=[1, 2, 3, 4, -1, -2, -3],
        port=lambda: np.array([sicopr._OptFom_Adaptive_Local_Search__mround(v)
                               for v in R], dtype=float),
        separates=lambda: np.round(R),
    ),
    dict(
        name='max_omits_nan',
        matlab='max(X) ignores NaN unless every element is NaN',
        numpy='np.max propagates NaN; np.nanmax ignores it',
        spec=lambda: max(v for v in N if not math.isnan(v)),
        octave=3.0,
        port=lambda: _helper('mmax')(N),
        separates=lambda: np.max(N),
    ),
    dict(
        name='min_omits_nan',
        matlab='min(X) ignores NaN unless every element is NaN',
        numpy='np.min propagates NaN; np.nanmin ignores it',
        spec=lambda: min(v for v in N if not math.isnan(v)),
        octave=1.0,
        port=lambda: _helper('mmin')(N),
        separates=lambda: np.min(N),
    ),
    dict(
        name='mod_takes_the_sign_of_the_divisor',
        matlab='mod(-7,3) is 2; mod follows the divisor',
        numpy='np.mod follows the divisor, matching MATLAB mod',
        spec=lambda: -7 - 3 * math.floor(-7 / 3),
        octave=2.0,
        port=lambda: np.mod(-7.0, 3.0),
        separates=lambda: np.fmod(-7.0, 3.0),   # rem's answer, -1
    ),
    dict(
        name='rem_takes_the_sign_of_the_dividend',
        matlab='rem(-7,3) is -1; rem follows the dividend',
        numpy='np.fmod follows the dividend, matching MATLAB rem',
        spec=lambda: -7 - 3 * math.trunc(-7 / 3),
        octave=-1.0,
        port=lambda: np.fmod(-7.0, 3.0),
        separates=lambda: np.mod(-7.0, 3.0),
    ),
    dict(
        name='fix_truncates_towards_zero',
        matlab='fix(-2.5) is -2; floor(-2.5) is -3',
        numpy='np.trunc matches fix, np.floor matches floor',
        spec=lambda: math.trunc(-2.5),
        octave=-2.0,
        port=lambda: np.trunc(-2.5),
        separates=lambda: np.floor(-2.5),
    ),
    dict(
        name='interp1_extrap_continues_the_end_slope',
        matlab="interp1(x,y,xi,'linear','extrap') continues the end segment",
        numpy='np.interp clamps to the end value instead',
        spec=lambda: 10.0 + (20.0 - 10.0) / (2.0 - 1.0) * (0.0 - 1.0),
        octave=0.0,
        port=lambda: sicopr._interp_Sparam__interp_extrap(
            np.array([0.0]), np.array([1.0, 2.0, 3.0]),
            np.array([10.0, 20.0, 30.0]))[0],
        separates=lambda: np.interp(0.0, [1.0, 2.0, 3.0], [10.0, 20.0, 30.0]),
    ),
]

# Contracts where both languages agree; pinned so a future default change is
# caught rather than assumed away.
AGREEING = [
    ('mean_propagates_nan', 'mean([1 NaN 3]) is NaN in both', np.nan,
     lambda: np.mean(N)),
    ('sum_propagates_nan', 'sum([1 NaN 3]) is NaN in both', np.nan,
     lambda: np.sum(N)),
    ('median_of_even_count_averages_the_middle_two',
     'median([1 2 3 4]) is 2.5 in both', 2.5, lambda: np.median([1.0, 2.0, 3.0, 4.0])),
    ('median_propagates_nan', 'median([1 NaN 3]) is NaN in both', np.nan,
     lambda: np.median(N)),
    ('unwrap_breaks_at_pi', 'unwrap([0 3.5]) is [0 -2.7831853071795862] in both',
     -2.7831853071795862, lambda: np.unwrap([0.0, 3.5])[1]),
    ('floor_goes_towards_minus_infinity', 'floor(-2.5) is -3 in both', -3.0,
     lambda: np.floor(-2.5)),
]

for c in CONTRACTS:
    spec, oct_v, sep = c['spec'](), c['octave'], c['separates']()

    # the probe must actually tell the two conventions apart
    check('%s__probe_separates_the_conventions' % c['name'],
          not _same(sep, oct_v),
          'the probe gives %r under the numpy default and %r under MATLAB; if '
          'these are equal the contract below proves nothing' % (sep, oct_v))

    # leg 1: does our reading of the documentation match the reference?
    check('%s__documented_rule_matches_octave' % c['name'],
          _same(spec, oct_v),
          'MATLAB doc says: %s. Written out, that gives %r. COM Octave returns '
          '%r. One of the two readings is wrong -- check the documentation '
          'before trusting either.' % (c['matlab'], spec, oct_v))

    # leg 2: does the port match the reference?
    if c['port'] is not None:
        check('%s__port_matches_octave' % c['name'],
              _same(c['port'](), oct_v),
              'numpy doc says: %s. SiCoPR returns %r where COM Octave returns '
              '%r.' % (c['numpy'], c['port'](), oct_v))

for name, doc, want, got in AGREEING:
    check(name, _same(got(), want),
          '%s -- SiCoPR/numpy returns %r, expected %r. A default changed.'
          % (doc, got(), want))

# ---- max/min and NaN: closed 2026-09-22 ---------------------------------
# The contracts above prove MATLAB omits NaN and numpy propagates it. Every
# max/min in the engine now goes through _mmax/_mmin, which skip NaN, return
# NaN quietly when every element is one, and order complex by magnitude then
# angle the way MATLAB does. test_matlab_semantics.py lints that no bare
# np.max/np.min remains.
_cx = np.array([3 + 4j, 5 + 0j])
check('maxmin_port_orders_complex_like_matlab',
      _helper('mmax')(_cx) == (3 + 4j)
      and _helper('mmin')(_cx) == (5 + 0j),
      'both elements have magnitude 5, so MATLAB breaks the tie on angle: '
      'max is 3+4i and min is 5 (checked against COM Octave). numpy compares '
      'the real part first and would give 5 for max. Got max %s min %s'
      % (_helper('mmax')(_cx), _helper('mmin')(_cx)))

check('maxmin_port_returns_nan_only_when_all_nan',
      np.isnan(_helper('mmax')(np.array([np.nan, np.nan])))
      and _helper('mmax')(np.array([1.0, np.nan])) == 1.0,
      'MATLAB max of an all-NaN vector is NaN, and of [1 NaN] is 1')

# ---- a trap with no site yet, pinned before one appears -------------------
# MATLAB and Python's ** both give a complex cube root of a negative number;
# np.power returns NaN. Worth pinning because the failure is silent.
with np.errstate(invalid='ignore'):
    _np_pow = np.power(-8.0, 1.0 / 3.0)
check('negative_base_fractional_power_is_a_known_numpy_trap',
      np.isnan(_np_pow) and abs(complex((-8) ** (1 / 3)).imag - 1.7320508075688772) < 1e-12,
      'np.power(-8, 1/3) returned %r. MATLAB gives 1+1.7320508075688772i and '
      "Python's ** gives the same complex value; np.power gives NaN. If this "
      'check fails numpy changed behaviour -- revisit any fractional power in '
      'the engine.' % _np_pow)

finish()
