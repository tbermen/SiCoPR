"""Differential test: inlined copies of a function must behave like the original.

The assembler inlines helper functions into their callers, so one MATLAB
function can exist as many Python copies. There are currently 178 such copies
of 70 functions. A fix applied to `com_functions/fn/<name>/py_impl.py` reaches
the CANONICAL copy only -- every inlined one keeps the old behaviour, silently.

That is not hypothetical. Engine defect #6 of the 208-case MATLAB correlation
was the package die network truncated to 1 of 3 LC sections, and it had to be
fixed in THREE inlined copies. The per-function unit tests never saw it: they
import `py_impl.py` and exercise the canonical version, which is the one copy
that was already right.

Comparing source text does not work -- 139 of the 178 copies differ textually
for legitimate reasons (deliberate stubs, cosmetic rewrites, renamed
parameters). So this compares BEHAVIOUR: drive both copies with identical
inputs and diff the numeric output.

Two layers:
  A. arity parity for all 178 copies, pinned to a reviewed baseline
  B. behavioural differential for every copy that can be driven synthetically

Run: python tests/test_inlined_copies.py
"""
import ast
import copy as _copy
import io
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from audit_check import check, xcheck, finish  # noqa: E402
import com  # noqa: E402

_src = io.open(os.path.join(_ROOT, 'com.py'), encoding='utf-8').read()
_TOPS = {n.name: n for n in ast.parse(_src).body
         if isinstance(n, ast.FunctionDef)}
with io.open(os.path.join(_ROOT, 'com_functions', 'registry.json'),
             encoding='utf-8-sig') as _f:
    _REG = {x['name'] if isinstance(x, dict) else x
            for x in json.load(_f)['functions']}


def inlined_copies():
    """-> [(copy_name, canonical_name, parent), ...]"""
    out = []
    for name in sorted(_TOPS):
        if not (name.startswith('_') and '__' in name[1:]):
            continue
        parent, _, child = name[1:].partition('__')
        if child in _REG and child in _TOPS:
            out.append((name, child, parent))
    return out


COPIES = inlined_copies()

# ---------------------------------------------------------------- layer A
def _arity(fn):
    a = fn.args
    return (len(a.args), len(a.defaults), len(a.kwonlyargs))


# Copies whose signature genuinely differs from the canonical one, reviewed
# 2026-08-18. Most are deliberate: a narrower stub, or a caller that always
# supplies what the canonical version defaults.
KNOWN_ARITY = {
    ('get_pdf_from_sampled_signal', 'Burst_Probability_Calc'),
    ('get_pdf_from_sampled_signal', 'COM_eye_width'),
    ('get_pdf_from_sampled_signal', 'Create_Noise_PDF'),
    ('get_pdf_from_sampled_signal', 'RILN_TD'),
    ('get_pdf_from_sampled_signal', 'adjust_Rx_noise_for_quantization'),
    ('get_pdf_from_sampled_signal', 'get_ILN_cmp_td'),
    ('get_pdf_from_sampled_signal', 'get_RILN_cmp_td'),
    ('get_pdf_from_sampled_signal', 'get_TDR'),
    ('get_pdf_from_sampled_signal', 'get_cm_noise'),
    ('get_pdf_from_sampled_signal', 'get_pdf'),
    ('get_pdf_from_sampled_signal', 'get_pdf_full'),
    ('get_cm_noise', 'COM_FD_to_TD'),
    ('get_pdf_full', 'COM_eye_width'),
    ('SL', 'add_pkg_with_die'),
    ('Tukey_Window', 'get_TDR'),
    ('make_pkg', 'make_full_pkg'),
    ('make_pkg', 'read_s4p_files'),
    ('make_pkg', 's21_pkg'),
    ('get_TDR', 'process_sxp'),
    ('rangelimit', 'read_p2_s2params'),
    ('rangelimit', 'read_p4_s4params'),
}

_arity_bad = {(child, parent) for name, child, parent in COPIES
              if _arity(_TOPS[name]) != _arity(_TOPS[child])}

check("no_new_inlined_arity_mismatch",
      not (_arity_bad - KNOWN_ARITY),
      "inlined copies whose signature newly differs from the canonical "
      "function: %s -- an inlined copy that cannot accept the same arguments "
      "cannot have tracked a change to the original"
      % sorted(_arity_bad - KNOWN_ARITY))

check("known_arity_mismatch_list_is_current",
      not (KNOWN_ARITY - _arity_bad),
      "these copies no longer differ in arity: %s -- remove them from "
      "KNOWN_ARITY" % sorted(KNOWN_ARITY - _arity_bad))

# ---------------------------------------------------------------- layer B
_rng = np.random.default_rng(12345)
_F = np.linspace(0.0, 50e9, 256)
_Z = _rng.standard_normal(256) + 1j * _rng.standard_normal(256)
_PDF_A = com.normal_dist(0.01, 5, 1e-4)
_PDF_B = com.normal_dist(0.02, 5, 1e-4)

# Synthetic inputs, built once per comparison and deep-copied for each side so
# neither copy can perturb the other's arguments.
FACTORY = {
    'bessel':            lambda: (5,),
    'pam':               lambda: (np.tile([-1.0, -1.0, -1.0, 1.0,
                                           1.0, 1.0, 1.0, -1.0], 8),),
    'normal_dist':       lambda: (0.01, 5, 1e-4),
    'combines4p':        lambda: tuple(_Z.copy() for _ in range(8)),
    'pdf_to_cdf':        lambda: (_PDF_A,),
    'conv_fct':          lambda: (_PDF_A, _PDF_B),
    'get_center_of_UI':  lambda: (32,),
    'pdf2sgm':           lambda: (_PDF_A,),
    'scalePDF':          lambda: (_PDF_A, 0.5),
    'Init_PDF_Fast':     lambda: (com.normal_dist(0.01, 5, 1e-4),
                                  np.array([-2e-3, 0.0, 2e-3]),
                                  np.array([0.25, 0.5, 0.25])),
}

# Copies known to differ behaviourally, reviewed 2026-08-18. Each is a FALLBACK
# STUB reached only when dependency injection is skipped; com.py wires the real
# function in production (see the _wired_* partials near the top of com.py).
# They are recorded rather than fixed because they are unreachable today -- but
# they are a live trap if a new call path ever forgets to inject.
KNOWN_BEHAVIOUR = {
    ('normal_dist', 'COM_eye_width'):
        'fallback stub spans +/-nsigma; ML 8542 uses -round(2*nsigma*sigma/'
        'binsize) to "capture more of the tails" -> 1000 bins vs 2001. '
        'Dead in production: com.py injects _normal_dist_fn=normal_dist.',
    ('normal_dist', 'get_RILN_cmp_td'):
        'same +/-nsigma truncation, and the stub omits the BinSize and Min '
        'fields entirely. get_RILN_cmp_td has no wired caller.',
    ('pdf_to_cdf', 'COM_eye_width'):
        'canonical returns a CDF with yB/yT tails; this stub returns a PDF-'
        'shaped x/y/BinSize/Min. Dead in production (injected).',
    ('get_center_of_UI', 'get_pdf_full'):
        'INTENTIONAL: the get_pdf_full copy is M//2+1 (1-based, MATLAB-'
        'faithful) where the canonical is M//2. See audit finding D12.',
}


def _flat(v):
    """Flatten any return value to one numeric vector for comparison."""
    if isinstance(v, tuple):
        parts = [_flat(x) for x in v]
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.array([])
    if hasattr(v, '__dict__'):
        parts = []
        for k in sorted(vars(v)):
            try:
                parts.append(np.atleast_1d(
                    np.asarray(getattr(v, k), dtype=complex)).ravel())
            except (TypeError, ValueError):
                pass
        return np.concatenate(parts) if parts else np.array([])
    try:
        a = np.atleast_1d(np.asarray(v))
    except (TypeError, ValueError):
        return np.array([])
    return a.astype(complex).ravel() if a.dtype.kind in 'ifcb' else np.array([])


_compared = _skipped = 0
for _name, _child, _parent in COPIES:
    if _child not in FACTORY:
        _skipped += 1
        continue
    _key = (_child, _parent)
    try:
        _args = FACTORY[_child]()
        _a = getattr(com, _name)(*_copy.deepcopy(_args))
        _b = getattr(com, _child)(*_copy.deepcopy(_args))
    except Exception as _e:                                  # noqa: BLE001
        _skipped += 1
        if _key not in KNOWN_BEHAVIOUR:
            check("inlined_copy_callable__%s__in__%s" % (_child, _parent),
                  False,
                  "could not drive the copy with the canonical arguments: "
                  "%s: %s" % (type(_e).__name__, _e))
        continue

    _compared += 1
    _fa, _fb = _flat(_a), _flat(_b)
    _same = (_fa.shape == _fb.shape and _fa.size > 0
             and np.max(np.abs(_fa - _fb)) <= 1e-12)
    _detail = ('shape %s vs %s' % (_fa.shape, _fb.shape)
               if _fa.shape != _fb.shape
               else 'max|delta| = %.3g' % (np.max(np.abs(_fa - _fb))
                                           if _fa.size else float('nan')))

    if _key in KNOWN_BEHAVIOUR:
        xcheck("inlined_copy_matches__%s__in__%s" % (_child, _parent),
               _same,
               "%s (%s)" % (KNOWN_BEHAVIOUR[_key], _detail))
    else:
        check("inlined_copy_matches__%s__in__%s" % (_child, _parent),
              _same,
              "the inlined copy of %s inside %s no longer behaves like the "
              "canonical function (%s). A fix applied to "
              "com_functions/fn/%s/py_impl.py does NOT reach this copy -- "
              "engine defect #6 was exactly this, across three copies."
              % (_child, _parent, _detail, _child))

print("\n%d inlined copies of %d functions; %d compared behaviourally, "
      "%d not drivable synthetically"
      % (len(COPIES), len({c for _, c, _ in COPIES}), _compared, _skipped))
finish()
