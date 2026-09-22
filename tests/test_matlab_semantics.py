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

from audit_check import check, xcheck, finish   # noqa: E402
import sicopr                                   # noqa: E402

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
# max, min     ignore NaN                 propagate NaN              OPEN
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

# ...but most sites still call np.round, which goes half to even.
np_rounds = sites(r'np\.round\(')
int_rounds = sites(r'int\(round\(')
xcheck('round_every_site_uses_matlab_semantics',
       not np_rounds and not int_rounds,
       'OPEN: %d np.round and %d int(round( sites round half to even, where '
       'the MATLAB they translate uses half away from zero. All three np.round '
       'patterns map to a MATLAB round(): the PDF bin snap '
       'binsize*round(values/binsize) (ML 5949), round(values/pdf.BinSize) '
       '(ML 2240) and round(2*(seq+1)) (ML 4462-4465). The seq form is tie-free '
       'for the PAM level sets COM uses; the bin snaps are not. Fixing them '
       'changes engine output at ties, so it is sequenced against the 1368-case '
       're-run rather than done here.' % (len(np_rounds), len(int_rounds)))

# The bin snap, driven at unit level: no COM run needed to see it.
_d_cpdf = sicopr._Bathtub_Contribution_Wrapper__d_cpdf
_r = _d_cpdf(1.0, np.array([0.5, 1.5, 2.5, 3.5]), np.full(4, 0.25))
_y_got = np.asarray(_r.y)
# MATLAB round gives bins 1,2,3,4 -> one quarter of the mass in each.
_y_matlab = np.array([0.0, 0.25, 0.25, 0.25, 0.25])
xcheck('round_pdf_bin_snap_matches_matlab',
       _y_got.shape == _y_matlab.shape and np.allclose(_y_got, _y_matlab),
       'OPEN, same cause as above: snapping [0.5,1.5,2.5,3.5] at binsize 1 '
       'gives mass %s, because np.round sends 0.5->0 and 2.5->2 and so merges '
       'two values into one bin. MATLAB round gives 1,2,3,4 and mass %s.'
       % (list(_y_got), list(_y_matlab)))


# ---- max / min and NaN ----------------------------------------------------
maxmin = sites(r'np\.(?:max|min)\(')
nanaware = sites(r'np\.nan(?:max|min)\(')
xcheck('maxmin_nan_semantics_reviewed',
       bool(nanaware) or not maxmin,
       'OPEN: %d np.max/np.min sites and %d nan-aware ones. MATLAB max/min '
       'ignore NaN, numpy propagates it, and NaN does occur in the pipeline '
       '(interp_Sparam repairs NaNs in IL before use). No site has been shown '
       'to receive a NaN, and none has been shown not to. Needs a per-site '
       'review, not a blanket swap: np.nanmax where MATLAB would skip, plain '
       'np.max where a NaN should still poison the result.'
       % (len(maxmin), len(nanaware)))


# ---- conventions that happen to agree -------------------------------------
check('unwrap_default_matches_matlab',
      abs(float(np.unwrap(np.array([0.0, 3.5]))[1]) - (3.5 - 2 * np.pi)) < 1e-12,
      'MATLAB unwrap and np.unwrap both break at pi; if this fails one of them '
      'changed its default')

finish()
