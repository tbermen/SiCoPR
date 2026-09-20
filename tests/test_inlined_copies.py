"""Differential test: inlined copies of a function must behave like the original.

The assembler inlines helper functions into their callers, so one MATLAB
function can exist as many Python copies. There are currently 177 such copies
of 70 functions, and this test prints the count it actually found, so the two
cannot drift apart unnoticed. A fix applied to `com_functions/fn/<name>/py_impl.py` reaches
the CANONICAL copy only -- every inlined one keeps the old behaviour, silently.

That is not hypothetical. Engine defect #6 of the 208-case MATLAB correlation
was the package die network truncated to 1 of 3 LC sections, and it had to be
fixed in THREE inlined copies. The per-function unit tests never saw it: they
import `py_impl.py` and exercise the canonical version, which is the one copy
that was already right.

Comparing source text does not work -- most of the copies differ textually
for legitimate reasons (deliberate stubs, cosmetic rewrites, renamed
parameters). So this compares BEHAVIOUR: drive both copies with identical
inputs and diff the numeric output.

Two layers:
  A. arity parity for every copy, pinned to a reviewed baseline
  B. behavioural differential for every copy that can be driven synthetically

Run: python tests/test_inlined_copies.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

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
import sicopr  # noqa: E402

_src = io.open(os.path.join(_ROOT, 'sicopr.py'), encoding='utf-8').read()
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

# ------------------------------------------------- the provenance manifest
#
# assemble_sicopr.py writes com_functions/inlined_copies.json as it mints the
# mangled names, so it is the record of which upstream helper each copy came
# from. It is generated, so it can only go stale by someone editing sicopr.py
# by hand -- which is exactly the thing CONTRIBUTING.md forbids. Checking it
# against what is actually in the engine catches that.
_MANIFEST = os.path.join(_ROOT, 'com_functions', 'inlined_copies.json')
if os.path.exists(_MANIFEST):
    with io.open(_MANIFEST, encoding='utf-8') as _f:
        _man = json.load(_f)
    _man_pairs = {(c['helper'], c['inlined_into']) for c in _man['copies']
                  if 'canonical' in c}
    _found_pairs = {(child, parent) for _, child, parent in COPIES}
    check("manifest_covers_every_inlined_copy",
          not (_found_pairs - _man_pairs),
          "sicopr.py contains inlined copies that com_functions/"
          "inlined_copies.json does not record: %s -- re-run "
          "assemble_sicopr.py"
          % sorted(_found_pairs - _man_pairs)[:8])
    check("manifest_records_no_phantom_copies",
          not (_man_pairs - _found_pairs),
          "the manifest records copies that are not in sicopr.py: %s -- the "
          "manifest is ahead of the engine, so one of them was not "
          "regenerated" % sorted(_man_pairs - _found_pairs)[:8])
else:
    check("manifest_covers_every_inlined_copy", False,
          "com_functions/inlined_copies.json is missing -- run "
          "assemble_sicopr.py")

# ---------------------------------------------------------------- layer B
_rng = np.random.default_rng(12345)
_F = np.linspace(0.0, 50e9, 256)
_Z = _rng.standard_normal(256) + 1j * _rng.standard_normal(256)
_PDF_A = sicopr.normal_dist(0.01, 5, 1e-4)
# A decaying ISI tail with sign changes -- the shape findbankloc and the PDF
# builders actually see, and one where ties in `ndiff` are reachable.
_ISI_TAIL = (np.exp(-np.arange(48) / 9.0)
             * np.cos(np.arange(48) / 2.3) * 0.03)


def _FILT_PARAM():
    from types import SimpleNamespace
    return SimpleNamespace(fb=106.25e9, fb_BT_cutoff=0.75, fb_BW_cutoff=0.75,
                           BTorder=4, f_r=0.75, RC_Start=20e9, RC_end=40e9)

_PDF_B = sicopr.normal_dist(0.02, 5, 1e-4)

# A short frequency axis for the lumped-element S-parameter builders, which
# allocate per-frequency arrays and choke on the 256-point _F.
_FR = np.linspace(0.0, 40e9, 16)
_F4 = np.linspace(0.0, 50e9, 64)
# An eye contour: rows are sampling phase, columns voltage.
_EYE = np.abs(np.outer(np.linspace(0, 1, 33), np.linspace(-0.3, 0.3, 65)))


def _PORT_CUBE():
    """A 4-port cube with unambiguous through paths on 1-2 and 3-4."""
    s = np.full((_F4.size, 4, 4), 0.005 + 0j)
    att = 0.9 * np.exp(-_F4 / 120e9) * np.exp(-2j * np.pi * _F4 * 3e-10)
    for a, b in ((0, 1), (2, 3)):
        s[:, a, b] = att
        s[:, b, a] = att
    for i in range(4):
        s[:, i, i] = 0.02
    return s

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
    'Init_PDF_Fast':     lambda: (sicopr.normal_dist(0.01, 5, 1e-4),
                                  np.array([-2e-3, 0.0, 2e-3]),
                                  np.array([0.25, 0.5, 0.25])),

    # --- added 2026-08-22 to raise behavioural coverage -------------------
    # Coverage was 50 of 178 copies (28%). The gap is what let defect #10c
    # through: MMSE and force each carried a "simplified" _findbankloc that
    # picked the highest-power non-overlapping banks instead of running the
    # real badV/goodV admissibility loop. Arity matched, so layer A passed;
    # no factory existed, so layer B never ran. These ten factories cover the
    # ten most-copied undrivable functions.
    'findbankloc':       lambda: (_ISI_TAIL.copy(), 3, 20, 2, 1.0, 0.2, 2),
    'd_cpdf':            lambda: (1e-4, np.array([-2e-3, 0.0, 2e-3]),
                                  np.array([0.25, 0.5, 0.25])),
    'get_pdf_from_sampled_signal':
                         lambda: (_ISI_TAIL.copy(), 4, 1e-4, 0),
    'dfe_clipper':       lambda: (_ISI_TAIL.copy(),
                                  np.full(_ISI_TAIL.size, 0.05),
                                  np.full(_ISI_TAIL.size, -0.05)),
    'CDF_inv_ev':        lambda: (1e-5, _PDF_A, sicopr.pdf_to_cdf(_PDF_A).y),
    'FFE':               lambda: (np.array([0.0, 1.0, -0.1]), 1, 32,
                                  _rng.standard_normal(512)),
    'Bessel_Thomson_Filter':
                         lambda: (_FILT_PARAM(), _F.copy(), 1),
    'Butterworth_Filter':
                         lambda: (_FILT_PARAM(), _F.copy(), 1),
    'Tukey_Window':      lambda: (_F.copy(), _FILT_PARAM(), 20e9, 40e9),
    'synth_tline':       lambda: (_F.copy(), 100.0, 100.0,
                                  np.array([0.0, 1.1e-9, 1.0e-4, 0.0]),
                                  6.5e-12, 0.15),

    # --- added 2026-08-31 -------------------------------------------------
    # Second sweep over the undrivable set: 64 copies had no factory, so the
    # only thing checked about them was arity. These are the ones drivable
    # from pure synthetic inputs, without a param/OP struct or a Touchstone
    # file. Four of them turned out to differ -- see KNOWN_BEHAVIOUR; the
    # rest agree, which is what the count in the summary line is worth.
    'PRBS13Q':           lambda: (),
    'conv_fct_MeanNotZero': lambda: (_PDF_A, _PDF_B),
    'CDF_ev':            lambda: (1e-5, _PDF_A, sicopr.pdf_to_cdf(_PDF_A).y),
    'scaleCDF':          lambda: (_PDF_A, 0.02, 1e-5, 0.5),
    'combine_pdf_same_voltage_axis': lambda: (_PDF_A, _PDF_B),
    'cdf_to_ber_contour': lambda: (sicopr.pdf_to_cdf(_PDF_A), 1e-5),
    'Full_Grid_Matrix':  lambda: ([[1.0, 2.0], [3.0, 4.0], [5.0]],),
    'compute_hard_cap':  lambda: (1, 2.0, 0.5, 3),
    'vma':               lambda: (_ISI_TAIL.copy(), 4),
    'hrem':              lambda: (_ISI_TAIL.copy(), 5, 4, 0.2),
    'H_interp':          lambda: (_Z.copy(), _F.copy(),
                                  np.linspace(0.0, 40e9, 128), 106.25e9),
    'TD_CTLE':           lambda: (_ISI_TAIL.copy(), 106.25e9, 20e9, 25e9,
                                  40e9, -6.0, 32),
    'FD_CTLE':           lambda: (_F.copy(), 20e9, 25e9, 40e9, -6.0),
    'FFE_Fast':          lambda: (np.array([0.0, 1.0, -0.1]),
                                  _rng.standard_normal((3, 128))),
    'find_eye_width':    lambda: (_EYE.copy(), 16, 32, 0.05),
    'floatingDFE':       lambda: (_ISI_TAIL.copy(), 3, 2, 20, 24, 0.2,
                                  1.0, 0.02),
    # NOT R_series2 / r_parrelell2 / SL. Their add_pkg_with_die copies are
    # reimplementations with a DIFFERENT SIGNATURE, not translations of the
    # same one: the copies take nfreq (an int) where the canonical takes the
    # frequency axis, return a bare (s11, s12, s21, s22) tuple instead of an
    # sparameters object, and _add_pkg_with_die__SL is (S, R, zref) against
    # the canonical (S, f, R, R_0). No single argument tuple drives both
    # sides, so a behavioural comparison here would be meaningless rather
    # than reassuring. The formulas themselves are transcribed identically
    # and are checked directly in tests/test_tline_tdr_erl.py.
    #
    # A random cube is rejected as "Ambiguous connections" by design, so this
    # is a plausible channel: low-loss through paths on 1-2 and 3-4, weak
    # crosstalk elsewhere. Answer is [1, 3, 2, 4], not the identity.
    'auto_port_order':   lambda: (_PORT_CUBE(), _F4.copy(), 0),
}

# Copies known to differ behaviourally, reviewed 2026-08-18. Each is a FALLBACK
# STUB reached only when dependency injection is skipped; sicopr.py wires the real
# function in production (see the _wired_* partials near the top of sicopr.py).
# They are recorded rather than fixed because they are unreachable today -- but
# they are a live trap if a new call path ever forgets to inject.
KNOWN_BEHAVIOUR = {
    ('normal_dist', 'COM_eye_width'):
        'fallback stub spans +/-nsigma; ML 8542 uses -round(2*nsigma*sigma/'
        'binsize) to "capture more of the tails" -> 1000 bins vs 2001. '
        'Dead in production: sicopr.py injects _normal_dist_fn=normal_dist.',
    ('normal_dist', 'get_RILN_cmp_td'):
        'same +/-nsigma truncation, and the stub omits the BinSize and Min '
        'fields entirely. get_RILN_cmp_td has no wired caller.',
    ('pdf_to_cdf', 'COM_eye_width'):
        'canonical returns a CDF with yB/yT tails; this stub returns a PDF-'
        'shaped x/y/BinSize/Min. Dead in production (injected).',
    ('get_center_of_UI', 'get_pdf_full'):
        'INTENTIONAL: the get_pdf_full copy is M//2+1 (1-based, MATLAB-'
        'faithful) where the canonical is M//2. See audit finding D12.',

    # --- surfaced 2026-08-22 when behavioural coverage rose 50 -> 113 copies.
    # Every one was checked against sicopr.py's _wired_* partials before being
    # recorded; none is a live divergence. The filter stubs matter because they
    # are BADLY wrong, not subtly so -- the get_TDR Bessel stub hardcodes the
    # 4th-order coefficients without reversing them, giving DC gain 105 instead
    # of 1 and returning a magnitude where MATLAB returns a complex response
    # (ML 1033-1040 uses fliplr). Harmless only for as long as the injection
    # holds.
    ('Bessel_Thomson_Filter', 'get_TDR'):
        'fallback stub: hardcoded coefficients, not reversed (DC gain 105 vs 1) '
        'and magnitude-only. Dead in production: sicopr.py:132 injects the real '
        'function into _wired_get_TDR.',
    ('Butterworth_Filter', 'get_TDR'):
        'fallback stub alongside the Bessel one. Injected at sicopr.py:133.',
    ('Tukey_Window', 'get_TDR'):
        'fallback stub returning ones, matching the MATLAB override inside this '
        'function (H_tw=ones). Injected at sicopr.py:134.',
    ('get_pdf_from_sampled_signal', 'get_TDR'):
        'fallback stub: a Gaussian fitted to the sample RMS, not the successive '
        'delta convolution. Injected at sicopr.py:138 (_get_pdf_fn).',

    # --- surfaced 2026-08-31 by the second factory sweep (113 -> 127 copies).
    # Three more injection stubs and one deliberate index-base split. Each was
    # read against its caller before being recorded here.
    ('H_interp', 'get_PSDs'):
        'fallback stub: np.interp on the MAGNITUDE only, so it drops the phase '
        'the canonical carries through unwrap/pchip, and never applies the '
        'fb/2 cutoff. Dead in production: sicopr.py:5211 passes '
        '_H_interp_fn=H_interp into get_PSDs.',
    ('find_eye_width', 'COM_eye_width'):
        'fallback stub returning samp_UI//4 on both sides regardless of the '
        'contour. Dead in production: sicopr.py:177 injects '
        '_find_eye_width_fn=find_eye_width.',
    ('combine_pdf_same_voltage_axis', 'COM_eye_width'):
        'fallback stub adds y elementwise and keeps pdf_a.x, where the '
        'canonical resamples onto a common axis -- so the two disagree in '
        'LENGTH (4002 vs 8002) whenever the inputs differ in span. Dead in '
        'production: injected alongside the other COM_eye_width helpers.',
    ('cdf_to_ber_contour', 'COM_eye_width'):
        'fallback stub, and badly wrong rather than subtly so: it returns '
        '(top, bottom) where the canonical returns (bottom, top), and finds '
        'the top edge with searchsorted(1 - specBER) instead of the flipped '
        'argmax(y > specBER) of ML 5254-5257. Dead in production: sicopr.py:176 '
        'injects _cdf_to_ber_fn=cdf_to_ber_contour, and sicopr.py:1810 falls '
        'back to this only if that injection is ever dropped.',
    ('hrem', 'floating_taps_1sttest'):
        'INTENTIONAL index-base split, the same shape as the get_center_of_UI '
        'entry above: the canonical takes a 1-BASED index and subtracts 1 '
        '(sicopr.py "convert 1-based -> 0-based"), the copy takes the index '
        '0-based and slices directly. Correct as wired -- floating_taps_1sttest '
        'passes ig1/ig2/ig3 from range(N_b, ...), which the comment at '
        'sicopr.py:9605 documents as the 0-based form of MATLAB ig1 = N_b+1:end1. '
        'Driving both with the same integer therefore shifts the window by one '
        'on purpose.',
    ('Bessel_Thomson_Filter', 'COM_FD_to_TD'):
        'fallback stub; injected at sicopr.py:148.',
    ('Butterworth_Filter', 'COM_FD_to_TD'):
        'fallback stub; injected at sicopr.py:149.',
    ('get_pdf_from_sampled_signal', 'COM_eye_width'):
        'fallback stub (Gaussian); injected at sicopr.py:167 (_get_pdf_ss_fn).',
    ('Bessel_Thomson_Filter', 'get_RILN_cmp_td'):
        'fallback stub. get_RILN_cmp_td has no wired caller, so dead by '
        'unreachability rather than by injection.',
    ('Butterworth_Filter', 'get_RILN_cmp_td'):
        'fallback stub; get_RILN_cmp_td has no wired caller.',
    ('get_pdf_from_sampled_signal', 'get_RILN_cmp_td'):
        'fallback stub (Gaussian); get_RILN_cmp_td has no wired caller.',

    # These two are LIVE -- called directly, not injected -- and are recorded
    # because they are correct where they stand, not because they are dead.
    # The canonical findbankloc returns `idx + (idx_st - 1)`, i.e. 0-BASED, which
    # is what floatingDFE and OptFom_Compute_DFE want. The MMSE and force copies
    # return `idx + idx_st`, 1-BASED, because their callers mirror MATLAB
    # arithmetic directly: MMSE does idx + RxFFE_cmx + 1 exactly as ML 2576 does,
    # and MATLAB's findbankloc is 1-based. Each is locally right; the hazard is
    # that one NAME now means two bases, which is the same shape as
    # FIX_SUMMARY #15. Changing it needs an oracle for the floating-tap paths,
    # and the 208 configs do not exercise them.
    ('findbankloc', 'MMSE'):
        'returns 1-BASED (idx + idx_st) where the canonical returns 0-based, '
        'because MMSE then applies ML 2576 arithmetic verbatim. Locally correct; '
        'see FIX_SUMMARY #15 for the producer-dependent-base hazard.',
    ('findbankloc', 'force'):
        'returns 1-BASED, same reason as the MMSE copy -- force indexes with '
        '`pos = cmx + 1 + k - 1  # idx is 1-based`.',
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
        # Copies legitimately take fewer positional args than the canonical --
        # get_pdf_from_sampled_signal has 3- and 4-arg forms, Tukey_Window 2-
        # and 4-arg. Drive each side with the arguments IT accepts, taken from
        # the same factory tuple, so the comparison stays like-for-like.
        _na = len(_TOPS[_name].args.args)
        _nb = len(_TOPS[_child].args.args)
        _a = getattr(sicopr, _name)(*_copy.deepcopy(_args[:_na]))
        _b = getattr(sicopr, _child)(*_copy.deepcopy(_args[:_nb]))
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
