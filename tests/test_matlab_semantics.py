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
            r'np\.round\(values\)',
            # `idx != np.round(idx)` is an INTEGRALITY TEST, not a rounding.
            # Both conventions agree that x == round(x) exactly when x is an
            # integer, so the half-away-from-zero rule cannot change the
            # answer. Used to reject a non-integer MATLAB subscript.
            r'idx\[idx != np\.round\(idx\)\]',
            # _colon's `n = int(round(limit / step + 1.0))` is a first guess at
            # the element count, immediately corrected by an overshoot test on
            # the next line, so the tie rule cannot decide the answer. Verified
            # at exactly-half quotients (3620.5, 10.5, 0.5): the result matches
            # MATLAB's floor(q)+1 in every case.
            r'n = int\(round\(limit / step \+ 1\.0\)\)')
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


# =============================================================================
# LIVE SEMANTIC PARITY, against COM Octave's own builtins (Phase 3, 2026-09-24)
#
# Everything above checks the ENGINE for a rule someone already knew. This
# section asks the builtins themselves: the same fixed inputs -- real and
# complex, vector and matrix, a NaN and an Inf in the mix -- through Octave's
# builtin and through numpy, and the answers compared at rtol 1e-12.
#
# Each row names the Octave expression, the numpy call a translator would
# write by default, and the call that matches MATLAB. The row asserts the
# MATCHING call against the pinned Octave value, and records whether the
# DEFAULT call also agrees; a default that disagrees is a row of the report.
# Expected values are COM Octave 11.3 builtins, run by
# tools/gen_semantics_pins.py and pinned here as literals, so this needs no
# Octave. Regenerate them with it; never edit them.
#
# Two classes of row do not assert Octave's value:
#   OCTAVE_NOT_MATLAB  Octave is known to differ from MATLAB here. The row
#                      asserts the MATLAB behaviour documented in builtins.md
#                      and the octave/patches header, and pins Octave's own
#                      answer only to show the difference.
#   ACCUMULATION       the answer depends on summation order alone (sum, fft).
#                      The observed difference is recorded, not asserted; only
#                      the shape and the NaN/Inf pattern must agree.
# =============================================================================

import scipy.interpolate as _si          # noqa: E402
import scipy.linalg as _sla              # noqa: E402
import scipy.signal as _ss               # noqa: E402
import scipy.special as _sp              # noqa: E402

_v = np.array([0.8147, -0.9058, 0.127, 0.9134, -0.6324, 0.0975, 0.2785])
_vn = np.array([0.8147, np.nan, 0.127, 0.9134, -0.6324, np.inf, 0.2785])
_vc = np.array([0.5 + 0.2j, -0.3 + 0.9j, 0.7 - 0.4j, -0.1 - 0.6j, 0.25 + 0.0j])
_M = np.array([[0.9575, 0.1576, 0.9572, 0.4218],
               [0.9649, 0.9706, 0.4854, 0.9157],
               [-0.7922, 0.9595, 0.8003, 0.0357]])
_Mn = np.array([[0.9575, np.nan, 0.9572, 0.4218],
                [0.9649, 0.9706, np.inf, 0.9157],
                [-0.7922, 0.9595, 0.8003, 0.0357]])
_Mc = np.array([[1 + 2j, 3 - 1j, 0.5 + 0.5j, -2 + 0j],
                [0 - 1j, 2 + 2j, -1 + 0.25j, 1 - 3j],
                [4 + 0j, -0.5 - 1j, 2 - 2j, 0.75 + 1j]])
_t = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 0.49999999999999994,
               -3.7, 3.7, np.nan, np.inf, -np.inf])
_mx = np.array([-7.5, -3.0, -0.5, 0.0, 0.5, 3.0, 7.5, 5.2, -5.2])
_xg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
_yg = np.array([10.0, 20.0, 35.0, 55.0, 60.0])
_xq = np.array([0.5, 1.0, 1.7, 2.5, 3.3, 4.9, 5.0, 6.0])
_fb, _fa = np.array([0.2, 0.3]), np.array([1.0, -0.5, 0.25])
_A = np.array([[4.0, -2.0, 1.0], [3.0, 6.0, -4.0], [2.0, 1.0, 8.0]])
_bA = np.array([12.0, -25.0, 32.0])
_Ao = np.array([[1.0, 2.0, 0.5], [2.0, -1.0, 3.0], [0.5, 1.5, -2.0],
                [3.0, 0.25, 1.0], [-1.0, 2.0, 2.0]])
_bo = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
_Au = np.array([[1.0, 2.0, 0.5, 3.0, -1.0], [2.0, -1.0, 3.0, 0.25, 2.0],
                [0.5, 1.5, -2.0, 1.0, 2.0]])
_bu = np.array([1.0, 2.0, 3.0])
_As, _bs = np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 3.0])
_ey = np.array([1e-12, 1e-9, 2e-9, 1e-5, 1e-3, 0.5, 1.5, 1.999])

_OCT = {}
# COM Octave 11.3.0 builtins, generated by tools/gen_semantics_pins.py
_OCT['std_v'] = np.array([0.6773784452543777])  # std(v)
_OCT['std_vn'] = np.array([np.nan])  # std(vn)
_OCT['std_vc'] = np.array([0.7158910531638177])  # std(vc)
_OCT['std_M'] = np.array([1.0123327236305926, 0.4662145107136843, 0.24026889797336098, 0.4410990856183373])  # std(M)
_OCT['std_Mn'] = np.array([1.0123327236305926, np.nan, np.nan, 0.4410990856183373])  # std(Mn)
_OCT['var_v'] = np.array([0.458841558095238])  # var(v)
_OCT['var_M'] = np.array([1.0248175433333335, 0.21735597000000004, 0.057729143333333344, 0.1945684033333333])  # var(M)
_OCT['var_vc'] = np.array([0.5125000000000001])  # var(vc)
_OCT['mean_M'] = np.array([0.3767333333333333, 0.6959, 0.7476333333333334, 0.4577333333333333])  # mean(M)
_OCT['mean_Mn'] = np.array([0.3767333333333333, np.nan, np.inf, 0.4577333333333333])  # mean(Mn)
_OCT['mean_Mc'] = np.array([complex(1.6666666666666667, 0.3333333333333333), complex(1.5, 0.0), complex(0.5, -0.4166666666666667), complex(-0.08333333333333333, -0.6666666666666666)])  # mean(Mc)
_OCT['sum_M'] = np.array([1.1302, 2.0877, 2.2429, 1.3732])  # sum(M)
_OCT['sum_Mn'] = np.array([1.1302, np.nan, np.inf, 1.3732])  # sum(Mn)
_OCT['sum_Mc'] = np.array([complex(5.0, 1.0), complex(4.5, 0.0), complex(1.5, -1.25), complex(-0.25, -2.0)])  # sum(Mc)
_OCT['sum_v'] = np.array([0.6929000000000001])  # sum(v)
_OCT['round_t'] = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 0.0, -4.0, 4.0, np.nan, np.inf, -np.inf])  # round(t)
_OCT['mod_3'] = np.array([1.5, 0.0, 2.5, 0.0, 0.5, 0.0, 1.5, 2.2, 0.7999999999999998])  # mod(mx, 3)
_OCT['mod_m3'] = np.array([-1.5, 0.0, -0.5, -0.0, -2.5, -0.0, -1.5, -0.7999999999999998, -2.2])  # mod(mx, -3)
_OCT['mod_0'] = np.array([-7.5, -3.0, -0.5, 0.0, 0.5, 3.0, 7.5, 5.2, -5.2])  # mod(mx, 0)
_OCT['mod_07'] = np.array([0.1999999999999993, 0.5, 0.19999999999999996, 0.0, 0.5, 0.20000000000000018, 0.5, 0.3000000000000007, 0.39999999999999947])  # mod(mx, 0.7)
_OCT['rem_3'] = np.array([-1.5, -0.0, -0.5, 0.0, 0.5, 0.0, 1.5, 2.2, -2.2])  # rem(mx, 3)
_OCT['rem_m3'] = np.array([-1.5, 0.0, -0.5, 0.0, 0.5, 0.0, 1.5, 2.2, -2.2])  # rem(mx, -3)
_OCT['rem_0'] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])  # rem(mx, 0)
_OCT['rem_07'] = np.array([-0.5, -0.20000000000000018, -0.5, 0.0, 0.5, 0.20000000000000018, 0.5, 0.3000000000000007, -0.3000000000000007])  # rem(mx, 0.7)
_OCT['interp_lin'] = np.array([np.nan, 10.0, 17.0, 27.5, 41.0, 59.5, 60.0, np.nan])  # interp1(xg, yg, xq)
_OCT['interp_lin_x'] = np.array([5.0, 10.0, 17.0, 27.5, 41.0, 59.5, 60.0, 65.0])  # interp1(xg, yg, xq, 'linear', 'extrap')
_OCT['interp_pchip_x'] = np.array([7.062499999999999, 10.0, 16.5485, 26.857142857142858, 41.336, 59.932, 60.0, 51.0])  # interp1(xg, yg, xq, 'pchip', 'extrap')
_OCT['interp_pchip_oct'] = np.array([np.nan, 10.0, 16.5485, 26.857142857142858, 41.336, 59.932, 60.0, np.nan])  # interp1(xg, yg, xq, 'pchip')
_OCT['filter_v'] = np.array([0.16294, 0.14472, -0.21471500000000002, 0.07724249999999999, 0.23983999999999997, -0.069610625, -0.009815312499999984])  # filter(fb, fa, v)
_OCT['filter_M'] = np.array([[0.1915, 0.03152, 0.19144000000000003, 0.08436], [0.57598, 0.25716, 0.47996, 0.35186], [0.37114499999999995, 0.60378, 0.4978, 0.43668999999999997]])  # filter(fb, fa, M)
_OCT['fft_v'] = np.array([complex(0.6929000000000001, 0.0), complex(0.12045664574024406, 0.22646516734960276), complex(0.927220505349007, 2.375961802299982), complex(1.4573228489107488, -0.9701310256809317), complex(1.4573228489107488, 0.9701310256809317), complex(0.927220505349007, -2.375961802299982), complex(0.12045664574024406, -0.22646516734960276)])  # fft(v)
_OCT['fft_vc'] = np.array([complex(1.05, 0.10000000000000003), complex(0.9726468687804171, 1.3399851714407562), complex(1.064656470147911, 0.24699481248458277), complex(0.3870656225395207, -1.9212593911093832), complex(-0.9743689614678486, 1.2342794071840442)])  # fft(vc)
_OCT['fft_M'] = np.array([[complex(1.1301999999999999, 0.0), complex(2.0877, 0.0), complex(2.2429, 0.0), complex(1.3732, 0.0)], [complex(0.8711500000000001, -1.521693236989637), complex(-0.80745, -0.009612881982007267), complex(0.31435, 0.2727113996517197), complex(-0.05389999999999995, -0.762102355330306)], [complex(0.8711500000000001, 1.521693236989637), complex(-0.80745, 0.009612881982007267), complex(0.31435, -0.2727113996517197), complex(-0.05389999999999995, 0.762102355330306)]])  # fft(M)
_OCT['fft_vn'] = np.array([complex(np.nan, 0.0), complex(np.nan, np.nan), complex(np.nan, np.nan), complex(np.nan, np.nan), complex(np.nan, np.nan), complex(np.nan, np.nan), complex(np.nan, np.nan)])  # fft(vn)
_OCT['norm_v'] = np.array([1.6797727673706346])  # norm(v)
_OCT['norm_v1'] = np.array([3.7693000000000003])  # norm(v, 1)
_OCT['norm_vinf'] = np.array([0.9134])  # norm(v, Inf)
_OCT['norm_vc'] = np.array([1.5074813431681335])  # norm(vc)
_OCT['norm_M'] = np.array([2.1463977125976097])  # norm(M)
_OCT['norm_Mfro'] = np.array([2.6782424796870057])  # norm(M, 'fro')
_OCT['norm_Mc'] = np.array([6.941394681533221])  # norm(Mc)
_OCT['norm_M1'] = np.array([2.7146])  # norm(M, 1)
_OCT['tr_Mc'] = np.array([[complex(1.0, 2.0), complex(0.0, -1.0), complex(4.0, 0.0)], [complex(3.0, -1.0), complex(2.0, 2.0), complex(-0.5, -1.0)], [complex(0.5, 0.5), complex(-1.0, 0.25), complex(2.0, -2.0)], [complex(-2.0, 0.0), complex(1.0, -3.0), complex(0.75, 1.0)]])  # Mc.'
_OCT['ctr_Mc'] = np.array([[complex(1.0, -2.0), complex(0.0, 1.0), complex(4.0, -0.0)], [complex(3.0, 1.0), complex(2.0, -2.0), complex(-0.5, 1.0)], [complex(0.5, -0.5), complex(-1.0, -0.25), complex(2.0, 2.0)], [complex(-2.0, -0.0), complex(1.0, 3.0), complex(0.75, -1.0)]])  # Mc'
_OCT['ctr_vc'] = np.array([complex(0.5, -0.2), complex(-0.3, -0.9), complex(0.7, 0.4), complex(-0.1, 0.6), complex(0.25, -0.0)]).reshape((5, 1))  # vc'
_OCT['ml_sq'] = np.array([1.0, -2.0, 4.0]).reshape((3, 1))  # A \ bA(:)
_OCT['ml_over'] = np.array([0.6703623219240497, 1.4405660280460917, 0.6149627084125068]).reshape((3, 1))  # Ao \ bo(:)
_OCT['ml_under'] = np.array([0.4357768765664756, 0.2819642891233735, -0.14223259797036564, 0.3167888029192698, 0.8789555645858504]).reshape((5, 1))  # Au \ bu(:)
_OCT['ml_sing_oct'] = np.array([0.2799999999999999, 0.5599999999999998]).reshape((2, 1))  # As \ bs(:)
_OCT['erfcinv_oct'] = np.array([5.042029740174564, 4.320005384416531, 4.241090006387659, 3.12341327434193, 2.3267537655135335, 0.4769362762044699, -0.4769362762044699, -2.326753765513555])  # erfcinv(ey)
_OCT['erfc_rt_oct'] = np.array([1.0000000561491337e-12, 1.0000000044029044e-09, 2.000000107480665e-09, 9.999999999931012e-06, 0.0009999999999999556, 0.5, 1.5, 1.999])  # erfc(erfcinv(ey))


def _mmod(x, y):
    """MATLAB mod: x - floor(x./y).*y, and mod(x,0) is x (numpy: NaN)."""
    x = np.asarray(x, dtype=float)
    if y == 0:
        return x.copy()
    return x - np.floor(x / y) * y


def _mrem(x, y):
    """MATLAB rem: x - fix(x./y).*y; rem(x,0) is NaN."""
    x = np.asarray(x, dtype=float)
    return x - np.fix(x / y) * y


def _basic_solution(A, b):
    """MATLAB's A\\b for an underdetermined full-row-rank A: a BASIC solution,
    at most rank(A) nonzeros, from QR with column pivoting."""
    Q, R, piv = _sla.qr(A, pivoting=True, mode='economic')
    r = A.shape[0]
    z = _sla.solve_triangular(R[:, :r], Q.T @ b)
    x = np.zeros(A.shape[1])
    x[piv[:r]] = z
    return x


def _same(got, want, rtol=1e-12):
    """shape (after squeezing MATLAB's 2-D), NaN/Inf pattern, then values."""
    g = np.asarray(got)
    w = np.asarray(want)
    if g.size != w.size:
        return False, 'size %s vs %s' % (g.shape, w.shape)
    if w.ndim < 2 or 1 in w.shape or g.ndim < 2:
        # a MATLAB 1xN / Nx1 / 1x1 against numpy's 1-D or scalar
        w = w.ravel()
        g = g.ravel()
    if g.shape != w.shape:
        return False, 'shape %s vs %s' % (g.shape, w.shape)
    g = g.astype(complex)
    w = w.astype(complex)
    for part in ('real', 'imag'):
        a, b = getattr(g, part), getattr(w, part)
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            return False, 'NaN pattern differs (%s)' % part
        fa = np.isfinite(a) & np.isfinite(b)
        if not np.array_equal(a[~np.isnan(a) & ~fa], b[~np.isnan(b) & ~fa]):
            return False, 'Inf pattern differs (%s)' % part
        if fa.any():
            d = np.abs(a[fa] - b[fa])
            tol = rtol * np.maximum(np.abs(b[fa]), 1.0)
            if np.any(d > tol):
                i = int(np.argmax(d - tol))
                return False, 'max |d| %.3g (%s)' % (float(np.max(d)), part)
    return True, ''


# name, pinned Octave key, class, numpy default, MATLAB-matching call
_ROWS = [
    ('std vector', 'std_v', 'assert', lambda: np.std(_v), lambda: np.std(_v, ddof=1)),
    ('std with NaN/Inf', 'std_vn', 'assert', lambda: np.std(_vn), lambda: np.std(_vn, ddof=1)),
    ('std complex', 'std_vc', 'assert', lambda: np.std(_vc), lambda: np.std(_vc, ddof=1)),
    ('std matrix', 'std_M', 'assert', lambda: np.std(_M), lambda: np.std(_M, axis=0, ddof=1)),
    ('std matrix NaN/Inf', 'std_Mn', 'assert', lambda: np.std(_Mn), lambda: np.std(_Mn, axis=0, ddof=1)),
    ('var vector', 'var_v', 'assert', lambda: np.var(_v), lambda: np.var(_v, ddof=1)),
    ('var matrix', 'var_M', 'assert', lambda: np.var(_M), lambda: np.var(_M, axis=0, ddof=1)),
    ('var complex', 'var_vc', 'assert', lambda: np.var(_vc), lambda: np.var(_vc, ddof=1)),
    ('mean matrix', 'mean_M', 'assert', lambda: np.mean(_M), lambda: np.mean(_M, axis=0)),
    ('mean matrix NaN/Inf', 'mean_Mn', 'assert', lambda: np.mean(_Mn), lambda: np.mean(_Mn, axis=0)),
    ('mean complex matrix', 'mean_Mc', 'assert', lambda: np.mean(_Mc), lambda: np.mean(_Mc, axis=0)),
    ('sum matrix', 'sum_M', 'accumulation', lambda: np.sum(_M), lambda: np.sum(_M, axis=0)),
    ('sum matrix NaN/Inf', 'sum_Mn', 'accumulation', lambda: np.sum(_Mn), lambda: np.sum(_Mn, axis=0)),
    ('sum complex matrix', 'sum_Mc', 'accumulation', lambda: np.sum(_Mc), lambda: np.sum(_Mc, axis=0)),
    ('sum vector', 'sum_v', 'accumulation', lambda: np.sum(_v), lambda: np.sum(_v)),
    ('round ties/NaN/Inf', 'round_t', 'assert', lambda: np.round(_t), lambda: _helper('mround_arr')(_t)),
    ('mod(x, 3)', 'mod_3', 'assert', lambda: np.mod(_mx, 3), lambda: _mmod(_mx, 3)),
    ('mod(x, -3)', 'mod_m3', 'assert', lambda: np.mod(_mx, -3), lambda: _mmod(_mx, -3)),
    ('mod(x, 0)', 'mod_0', 'assert', lambda: np.mod(_mx, 0), lambda: _mmod(_mx, 0)),
    ('mod(x, 0.7)', 'mod_07', 'assert', lambda: np.mod(_mx, 0.7), lambda: _mmod(_mx, 0.7)),
    ('rem(x, 3)', 'rem_3', 'assert', lambda: np.fmod(_mx, 3), lambda: _mrem(_mx, 3)),
    ('rem(x, -3)', 'rem_m3', 'assert', lambda: np.fmod(_mx, -3), lambda: _mrem(_mx, -3)),
    ('rem(x, 0)', 'rem_0', 'assert', lambda: np.fmod(_mx, 0), lambda: _mrem(_mx, 0)),
    ('rem(x, 0.7)', 'rem_07', 'assert', lambda: np.fmod(_mx, 0.7), lambda: _mrem(_mx, 0.7)),
    ('interp1 linear', 'interp_lin', 'assert', lambda: np.interp(_xq, _xg, _yg),
     lambda: np.interp(_xq, _xg, _yg, left=np.nan, right=np.nan)),
    ("interp1 linear 'extrap'", 'interp_lin_x', 'assert', lambda: np.interp(_xq, _xg, _yg),
     lambda: _si.interp1d(_xg, _yg, fill_value='extrapolate')(_xq)),
    ("interp1 pchip 'extrap'", 'interp_pchip_x', 'assert',
     lambda: _si.PchipInterpolator(_xg, _yg, extrapolate=False)(_xq),
     lambda: _si.PchipInterpolator(_xg, _yg)(_xq)),
    ('interp1 pchip, no extrap', 'interp_pchip_x', 'octave_not_matlab',
     lambda: _si.PchipInterpolator(_xg, _yg, extrapolate=False)(_xq),
     lambda: _si.PchipInterpolator(_xg, _yg)(_xq)),
    ('filter vector', 'filter_v', 'assert', lambda: _ss.lfilter(_fb, _fa, _v),
     lambda: _ss.lfilter(_fb, _fa, _v)),
    ('filter matrix', 'filter_M', 'assert', lambda: _ss.lfilter(_fb, _fa, _M),
     lambda: _ss.lfilter(_fb, _fa, _M, axis=0)),
    ('fft real vector', 'fft_v', 'accumulation', lambda: np.fft.fft(_v), lambda: np.fft.fft(_v)),
    ('fft complex vector', 'fft_vc', 'accumulation', lambda: np.fft.fft(_vc), lambda: np.fft.fft(_vc)),
    ('fft matrix', 'fft_M', 'accumulation', lambda: np.fft.fft(_M), lambda: np.fft.fft(_M, axis=0)),
    ('fft with NaN/Inf', 'fft_vn', 'accumulation', lambda: np.fft.fft(_vn), lambda: np.fft.fft(_vn)),
    ('norm vector', 'norm_v', 'assert', lambda: np.linalg.norm(_v), lambda: np.linalg.norm(_v)),
    ('norm(v, 1)', 'norm_v1', 'assert', lambda: np.linalg.norm(_v, 1), lambda: np.linalg.norm(_v, 1)),
    ('norm(v, Inf)', 'norm_vinf', 'assert', lambda: np.linalg.norm(_v, np.inf), lambda: np.linalg.norm(_v, np.inf)),
    ('norm complex vector', 'norm_vc', 'assert', lambda: np.linalg.norm(_vc), lambda: np.linalg.norm(_vc)),
    ('norm matrix', 'norm_M', 'assert', lambda: np.linalg.norm(_M), lambda: np.linalg.norm(_M, 2)),
    ("norm(M, 'fro')", 'norm_Mfro', 'assert', lambda: np.linalg.norm(_M), lambda: np.linalg.norm(_M, 'fro')),
    ('norm complex matrix', 'norm_Mc', 'assert', lambda: np.linalg.norm(_Mc), lambda: np.linalg.norm(_Mc, 2)),
    ('norm(M, 1)', 'norm_M1', 'assert', lambda: np.linalg.norm(_M, 1), lambda: np.linalg.norm(_M, 1)),
    (".' complex matrix", 'tr_Mc', 'assert', lambda: _Mc.T, lambda: _Mc.T),
    ("' complex matrix", 'ctr_Mc', 'assert', lambda: _Mc.T, lambda: _Mc.conj().T),
    ("' complex vector", 'ctr_vc', 'assert', lambda: _vc.T, lambda: _vc.conj()),
    ('A\\b square', 'ml_sq', 'assert', lambda: np.linalg.solve(_A, _bA), lambda: np.linalg.solve(_A, _bA)),
    ('A\\b overdetermined', 'ml_over', 'assert', lambda: np.linalg.lstsq(_Ao, _bo, rcond=None)[0],
     lambda: np.linalg.lstsq(_Ao, _bo, rcond=None)[0]),
    ('A\\b underdetermined', 'ml_under', 'octave_not_matlab',
     lambda: np.linalg.lstsq(_Au, _bu, rcond=None)[0], lambda: _basic_solution(_Au, _bu)),
    ('A\\b exactly singular', 'ml_sing_oct', 'octave_not_matlab',
     lambda: np.linalg.lstsq(_As, _bs, rcond=None)[0], lambda: np.full(2, np.inf)),
    ('erfcinv moderate', 'erfcinv_oct', 'assert', lambda: _sp.erfcinv(_ey), lambda: _sp.erfcinv(_ey)),
    ('erfcinv tail', 'erfcinv_oct', 'octave_not_matlab', lambda: _sp.erfcinv(_ey), lambda: _sp.erfcinv(_ey)),
]

# Which elements of a pinned vector a row covers, where one Octave result
# serves two rows: erfcinv's first three points are the tail, where Octave's
# builtin is the inaccurate one.
_SLICE = {'erfcinv moderate': slice(3, None), 'erfcinv tail': slice(0, 3)}


def _documented_matlab(name, match):
    """OCTAVE_NOT_MATLAB rows: the MATLAB behaviour, checked on the match."""
    if name == 'interp1 pchip, no extrap':
        # MATLAB interp1 extrapolates pchip outside the data without being
        # asked (octave/patches/H_interp.m); Octave returns NA there. MATLAB's
        # answer is Octave's WITH 'extrap', which is what the key pins.
        return _same(match, _OCT['interp_pchip_x'])
    if name == 'A\\b underdetermined':
        # MATLAB: a basic solution, at most rank(A) nonzeros, that solves the
        # system. Octave: the minimum-norm solution, every entry nonzero. The
        # VALUE of MATLAB's basic solution is not confirmed without MATLAB;
        # its defining properties are.
        x = np.asarray(match)
        ok = (np.count_nonzero(np.abs(x) > 1e-14) <= np.linalg.matrix_rank(_Au)
              and np.allclose(_Au @ x, _bu, rtol=0, atol=1e-12))
        return ok, '' if ok else 'not a basic solution: %s' % x
    if name == 'A\\b exactly singular':
        # MATLAB warns and returns Inf (octave/patches/mldivide_matlab.m:
        # [1 2; 2 4] \ [1; 3] is [Inf; Inf]). Octave: minimum-norm [0.28 0.56].
        # The engine's only square solve (force) raises instead, by ruling.
        return bool(np.all(np.isinf(match))), ''
    if name == 'erfcinv tail':
        # MATLAB's erfcinv is accurate to ~1e-15; Octave's builtin is off by
        # ~1e-9 relative here (octave/patches/erfcinv.m). Accuracy is checked
        # by the round trip through erfc, well conditioned in this direction.
        y = _ey[_SLICE[name]]
        rt = _sp.erfc(np.asarray(match))
        ok = bool(np.all(np.abs(rt - y) <= 1e-14 * y))
        return ok, '' if ok else 'erfc(erfcinv(y)) off by %.3g rel' % float(
            np.max(np.abs(rt - y) / y))
    raise KeyError(name)


_report = []
# Inf - Inf inside std/var is a NaN numpy announces on stderr, and run_all.ps1
# treats any stderr as a failed step. The NaN is the expected answer here.
_quiet = np.errstate(all='ignore')
_quiet.__enter__()
for _name, _key, _cls, _default, _match in _ROWS:
    _want = np.asarray(_OCT[_key])
    _sl = _SLICE.get(_name)
    if _sl is not None:
        _want = _want.ravel()[_sl]
    _m = np.asarray(_match())
    _d = np.asarray(_default())
    if _sl is not None:
        _m, _d = _m.ravel()[_sl], _d.ravel()[_sl]
    _dok, _dwhy = _same(_d, _want)
    if _cls == 'assert':
        _ok, _why = _same(_m, _want)
        check('parity__%s' % _name, _ok,
              '%s: the MATLAB-matching call disagrees with COM Octave: %s'
              % (_name, _why))
    elif _cls == 'accumulation':
        # summation order alone: recorded, not asserted, beyond shape and the
        # non-finite pattern
        _ok, _why = _same(_m, _want, rtol=np.inf)
        check('parity__%s' % _name, _ok,
              '%s: shape or NaN/Inf pattern differs from COM Octave: %s'
              % (_name, _why))
        _mm, _ww = _m.astype(complex).ravel(), _want.astype(complex).ravel()
        _f = np.isfinite(_mm) & np.isfinite(_ww)
        _dmax = float(np.max(np.abs(_mm[_f] - _ww[_f]))) if _f.any() else 0.0
        _why = 'accumulation order: max |d| %.2g, recorded' % _dmax
    else:
        _ok, _why = _documented_matlab(_name, _m)
        check('parity__%s' % _name, _ok,
              '%s: not the documented MATLAB behaviour: %s' % (_name, _why))
        _dok = False if _key.endswith('_oct') or _name in (
            'interp1 pchip, no extrap', 'A\\b underdetermined') else _dok
        _why = 'OCTAVE != MATLAB; asserted the MATLAB behaviour'
    _report.append((_name, _cls, _dok, _dwhy, _why))

_quiet.__exit__(None, None, None)

print('\nlive parity against COM Octave builtins (numpy DEFAULT vs Octave):')
for _name, _cls, _dok, _dwhy, _why in _report:
    print('  %-26s %-18s default %s%s%s' % (
        _name, _cls, 'agrees' if _dok else 'DIFFERS',
        '' if _dok else ' (%s)' % (_dwhy or 'see row'),
        ('; matching call: ' + _why) if _cls == 'accumulation' else ''))

finish()
