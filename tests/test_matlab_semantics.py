"""Where a numpy call stands in for a MATLAB builtin, the defaults must match.

A port is a translation of behaviour, not of tokens. `sigma = std(x)` becoming
`sigma = np.std(x)` reads as a faithful line and is not one: MATLAB normalises
by N-1, numpy by N. Nothing in either source shows it. Only the library
contract behind the call does, and that has to be looked up once per builtin
and then held in place by a test.

This file is that list. Each entry names a MATLAB builtin whose default differs
from the numpy call that replaces it, the rule the port follows, and a check
that the rule still holds. Two kinds of check:

  * **structural** -- no call site may rely on the differing default. This is
    what the argsort lint does, and what caught nothing for `std` because
    `std` was never on the list.
  * **behavioural** -- drive the engine's own helper with an input on which the
    two conventions disagree, and assert it gives MATLAB's answer. This needs
    no COM run: a tie, a NaN or an out-of-range point is enough. A divergence
    that no corpus happens to reach is still a divergence.

The list was built reactively -- each entry added after something bit. It is
written down here so the next one can be added by reading a contract instead.

Run: python tests/test_matlab_semantics.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from audit_check import check, finish   # noqa: E402
import sicopr                                   # noqa: E402


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


SRC = open(os.path.join(_ROOT, 'sicopr.py'), encoding='utf-8').read()


def sites(pattern):
    return re.findall(pattern, SRC)


def calls(name):
    """Every `name(...)` in the engine, with the argument list intact.

    A plain regex stops at the first ')', so `np.argsort(np.abs(iv),
    kind='stable')` comes back truncated before the argument that matters and
    the check fails on code that is correct. Match the parentheses instead.
    """
    out = []
    for m in re.finditer(re.escape(name) + r'\(', SRC):
        i = m.end()
        depth = 1
        while i < len(SRC) and depth:
            depth += {'(': 1, ')': -1}.get(SRC[i], 0)
            i += 1
        out.append(SRC[m.start():i])
    return out


# ---------------------------------------------------------------- the list
#
# builtin      MATLAB                     numpy default              rule here
# -----------------------------------------------------------------------------
# std, var     normalise by N-1           N (ddof=0)                 pass ddof=1
# sort         stable                     quicksort                  kind='stable'
# round        half away from zero        half to even               _mround
# max, min     skip NaN; complex by       propagate NaN; complex     _mmax/_mmin
#              magnitude then angle       lexicographic by real
# interp1      'extrap' extrapolates      np.interp clamps           _interp_extrap
# unwrap       tol = pi                   discont = pi               same, no action
# mean, sum    propagate NaN              propagate NaN              same, no action


# ---- std / var ------------------------------------------------------------
std_calls = calls('np.std') + calls('np.var')
check('std_every_site_passes_ddof',
      std_calls and all('ddof' in c for c in std_calls),
      'np.std/np.var without ddof normalises by N; MATLAB uses N-1: %s'
      % [c for c in std_calls if 'ddof' not in c])

# The behavioural side of this one lives in test_std_normalization.py, which
# drives COM Octave's own interp_Sparam as the oracle rather than a reading
# of it. Kept there because it needs the extracted reference.


# ---- sort / argsort -------------------------------------------------------
argsorts = calls('np.argsort')
check('argsort_every_site_is_stable',
      argsorts and all('stable' in c for c in argsorts),
      'MATLAB sort is stable; np.argsort defaults to quicksort: %s'
      % [c for c in argsorts if 'stable' not in c])


# ---- interp1 'extrap' -----------------------------------------------------
_ie = sicopr._interp_Sparam__interp_extrap
_fin, _y = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])
check('interp_extrap_extrapolates_below_range',
      abs(float(_ie(np.array([0.0]), _fin, _y)[0]) - 0.0) < 1e-12,
      "MATLAB interp1(...,'extrap') continues the end slope to 0.0 at f=0; "
      'np.interp would clamp to 10.0. Got %r'
      % float(_ie(np.array([0.0]), _fin, _y)[0]))


# ---- round ----------------------------------------------------------------
# Every _mround helper the assembler emitted must implement MATLAB's rule.
_mrounds = [n for n in dir(sicopr) if n.endswith('__mround')]
_want = {0.5: 1, 1.5: 2, 2.5: 3, -0.5: -1, -2.5: -3}
_bad = []
for _n in _mrounds:
    _f = getattr(sicopr, _n)
    for _x, _w in _want.items():
        if _f(_x) != _w:
            _bad.append('%s(%s)=%s want %s' % (_n, _x, _f(_x), _w))
check('mround_helpers_round_half_away_from_zero',
      _mrounds and not _bad,
      'MATLAB round() goes half away from zero: %s' % _bad[:4])

# The helpers must also be right at the boundary. floor(x + 0.5) is the obvious
# implementation and is wrong: 0.49999999999999994 + 0.5 is exactly 1.0 in
# double precision, so it rounds the largest double below a half up to 1 where
# MATLAB gives 0. Every _mround carried that bug until 2026-09-22.
_edge = {0.49999999999999994: 0, -0.49999999999999994: 0,
         1.4999999999999998: 1, -1.4999999999999998: -1}
_edge_bad = ['%s(%.17g)=%s want %s' % (n, x, getattr(sicopr, n)(x), w)
             for n in _mrounds for x, w in _edge.items()
             if getattr(sicopr, n)(x) != w]
_marrs = [n for n in dir(sicopr) if n.endswith('__mround_arr')]
_arr_probe = np.array([0.5, 1.5, 2.5, -0.5, -2.5, 0.49999999999999994, 1e16])
_arr_want = np.array([1.0, 2.0, 3.0, -1.0, -3.0, 0.0, 1e16])
_arr_bad = [n for n in _marrs
            if not np.array_equal(getattr(sicopr, n)(_arr_probe), _arr_want)]
check('every_inlined_mround_arr_copy_rounds_half_away',
      _marrs and not _arr_bad,
      'inlined _mround_arr copies that disagree with MATLAB: %s' % _arr_bad[:4])

check('mround_helpers_are_exact_just_below_a_half',
      not _edge_bad,
      'the largest double below a half must round down, as MATLAB does: %s'
      % _edge_bad[:4])

# No bare rounding may remain where the MATLAB it translates uses round().
# Reviewed exceptions, each tie-free by construction:
#   int(round(p1.Min + p2.Min))          a sum of two integer bin indices
#   int(round(values[0|-1] / binsize))   values were just snapped to the grid
#   int(round((stop - start) / step))    a colon-range count, not a MATLAB
#                                        round(); MATLAB's ':' has its own rule
#   return int(round(x))                 inside _mround itself, off a tie
#   np.round(x) / np.round(values)       inside _mround_arr itself, off a tie
_ALLOWED = (r'int\(round\(p\d\.Min \+ p\d\.Min\)\)',
            r'int\(round\(values\[-?\d\] / binsize\)\)',
            r'int\(round\(\(stop - start\) / step\)\)',
            r'return int\(round\(x\)\)',
            r'np\.round\(x\)',
            r'np\.round\(values\)')
def _code_lines(src):
    """Source lines with comments and docstrings dropped, so prose about
    rounding is not mistaken for a rounding call."""
    out, in_doc, delim = [], False, ''
    for line in src.split('\n'):
        s = line.strip()
        if in_doc:
            if delim in s:
                in_doc = False
            continue
        if s.startswith('#'):
            continue
        for d in ('"""', "'''"):
            if s.startswith(d) or s.startswith('r' + d):
                body = s.split(d, 1)[1]
                if d not in body:
                    in_doc, delim = True, d
                s = ''
                break
        out.append(s.split('  #')[0])
    return out


_bare = [s for s in _code_lines(SRC)
         if re.search(r'(?<![\w.])(?:np\.)?round\(', s)
         and not any(re.search(p, s) for p in _ALLOWED)]
check('round_every_site_uses_matlab_semantics',
      not _bare,
      'these round half to even, where the MATLAB they translate goes half '
      'away from zero -- use _mround (scalar) or _mround_arr (array): %s'
      % _bare[:6])

# The bin snap, driven at unit level: no COM run needed to see it.
#
# Expected values are COM Octave's own d_cpdf, via tools/octave_oracle.py. An
# earlier version of this check guessed them by hand and guessed the *shape*
# wrong -- it assumed bins 0..4 where the reference gives Min=1 and four bins.
# That is the argument for the oracle in one line.
_d_cpdf = _helper('d_cpdf')
_r = _d_cpdf(1.0, np.array([0.5, 1.5, 2.5, 3.5]), np.full(4, 0.25))
_y_got, _min_got = np.asarray(_r.y), _r.Min
_OCT_Y, _OCT_MIN = np.array([0.25, 0.25, 0.25, 0.25]), 1
check('round_pdf_bin_snap_matches_matlab',
      _min_got == _OCT_MIN and _y_got.shape == _OCT_Y.shape
      and np.allclose(_y_got, _OCT_Y),
      'snapping [0.5,1.5,2.5,3.5] at binsize 1 gives Min=%s mass %s; COM '
      'Octave gives Min=%s mass %s. Under np.round 0.5 goes to 0 and 2.5 to 2, '
      'merging two values into one bin.'
      % (_min_got, list(_y_got), _OCT_MIN, list(_OCT_Y)))


# ---- max / min and NaN ----------------------------------------------------
maxmin = sites(r'np\.(?:max|min)\(')
nanaware = sites(r'np\.nan(?:max|min)\(')
# Every max/min must take MATLAB's view of NaN. Inside the two helpers the bare
# numpy calls are the implementation, so those lines are the only exception.
_MAXMIN_ALLOWED = (
    r'return np\.n?(?:max|min)\(a\)',
    # A variable named `finite` is NaN-filtered at the point of definition
    # (`finite = flat[np.isfinite(flat)]`), so there is no NaN left for np.max
    # to propagate and _mmax would only add an indirection. Narrow on purpose:
    # it matches that one name, so a new bare site cannot hide behind it.
    r'np\.n?(?:max|min)\(np\.abs\(finite\)\)',
)
_bare_maxmin = [s for s in _code_lines(SRC)
                if re.search(r'(?<![\w.])np\.(?:max|min)\(', s)
                and not any(re.search(p, s) for p in _MAXMIN_ALLOWED)]
check('maxmin_every_site_uses_matlab_nan_semantics',
      not _bare_maxmin,
      'MATLAB max/min skip NaN; np.max/np.min propagate it, so one bad sample '
      'swallows the result. Use _mmax/_mmin: %s' % _bare_maxmin[:6])

# ...and every inlined copy of the helpers must behave, on the cases that
# separate the two conventions. The assembler emits one copy per calling
# function, so checking a single copy leaves the rest unverified -- and an
# inlined copy that drifts is exactly the defect class test_inlined_copies.py
# exists for.
_mmaxes = [n for n in dir(sicopr) if n.endswith('__mmax')]
_mmins = [n for n in dir(sicopr) if n.endswith('__mmin')]
# The counts do NOT have to match. This used to require one _mmax per _mmin,
# which held only because the callers that needed one happened to need both;
# collapsing the duplicate helper copies onto imports (2026-09-22) left 23 and
# 12. What matters is that at least one of each survives to be driven, and that
# EVERY surviving copy behaves -- which the probes below check one by one.
check('inlined_maxmin_copies_exist_to_check',
      _mmaxes and _mmins,
      'no inlined _mmax/_mmin copies found (%d and %d), so the per-copy checks '
      'below are vacuous' % (len(_mmaxes), len(_mmins)))
_mmax = getattr(sicopr, _mmaxes[0])
_mmin = getattr(sicopr, _mmins[0])
_nan_probes = [([1.0, np.nan, 3.0], 3.0, 1.0),
               ([1.0, 2.0, np.nan], 2.0, 1.0),
               ([-5.0, np.nan, -1.0], -1.0, -5.0)]
_nan_bad = ['%s %s -> max %s want %s, min %s want %s'
            % (nx, p, getattr(sicopr, nx)(np.array(p)), wx,
               getattr(sicopr, nn)(np.array(p)), wn)
            for nx, nn in zip(_mmaxes, _mmins)
            for p, wx, wn in _nan_probes
            if getattr(sicopr, nx)(np.array(p)) != wx
            or getattr(sicopr, nn)(np.array(p)) != wn]
check('mmax_mmin_skip_nan_like_matlab',
      not _nan_bad,
      'checked against COM Octave: max([1 NaN 3]) is 3, min is 1. %s' % _nan_bad)
# MATLAB orders complex by magnitude then angle; numpy by real part first.
# Verified against COM Octave: max([3+4i, 5]) is 3+4i, min is 5.
_cx = np.array([3 + 4j, 5 + 0j])
_cx_bad = [n for n in _mmaxes if getattr(sicopr, n)(_cx) != (3 + 4j)]     + [n for n in _mmins if getattr(sicopr, n)(_cx) != (5 + 0j)]
check('mmax_mmin_order_complex_by_magnitude_then_angle',
      not _cx_bad,
      'both elements have magnitude 5, so MATLAB breaks the tie on angle and '
      'returns 3+4i for max and 5 for min; numpy compares the real part first '
      'and returns 5 for max. Copies that disagree: %s' % _cx_bad[:4])

_allnan = np.array([np.nan, np.nan])
_allnan_bad = [n for n in _mmaxes + _mmins
               if not np.isnan(getattr(sicopr, n)(_allnan))]
check('mmax_mmin_return_nan_when_all_nan',
      not _allnan_bad,
      'MATLAB max of an all-NaN vector is NaN; the helpers must return it '
      'quietly rather than warn or raise')


# ---- conventions that happen to agree -------------------------------------
check('unwrap_default_matches_matlab',
      abs(float(np.unwrap(np.array([0.0, 3.5]))[1]) - (3.5 - 2 * np.pi)) < 1e-12,
      'MATLAB unwrap and np.unwrap both break at pi; if this fails one of them '
      'changed its default')

finish()
