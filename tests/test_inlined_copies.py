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
import re
import sys

import numpy as np
from types import SimpleNamespace

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
    """-> [(copy_name, canonical_name, parent), ...]

    A caller carrying TWO copies of the same helper distinguishes them with a
    suffix -- Output_Arg_Fill has both `_conv_fct` and `_conv_fct_b`. Matching
    the text after `__` against the registry verbatim misses every suffixed
    one, so they were invisible to both layers of this test and sat on stale
    code that nothing compared against anything. Strip a trailing _b/_c/_d or
    digits before looking the canonical up.
    """
    out = []
    for name in sorted(_TOPS):
        if not (name.startswith('_') and '__' in name[1:]):
            continue
        parent, _, child = name[1:].partition('__')
        if child in _REG and child in _TOPS:
            out.append((name, child, parent))
            continue
        base = re.sub(r'_(?:b|c|d|\d+)$', '', child)
        if base != child and base in _REG and base in _TOPS:
            out.append((name, base, parent))
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
    # Output_Arg_Fill's second get_pdf copy, found 2026-09-22 once
    # inlined_copies() started matching suffixed names: it is a
    # narrower form than the canonical.
    ('get_pdf', 'Output_Arg_Fill'),
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
    # Key off `canonical`, not `helper`: a second copy is recorded under a
    # suffixed helper name (`conv_fct_b`) but the assembler resolves it to the
    # reference function it duplicates, which is what this compares against.
    _man_pairs = {(c['canonical'], c['inlined_into'])
                  for c in _man['copies'] if 'canonical' in c}
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

# ---------------------------------------------------------------------------
# Edge cases, added 2026-09-22.
#
# The factory inputs above are all well-behaved: ascending real vectors,
# scalars in range, nothing empty or NaN. That is exactly why copy drift kept
# hiding. The Octave-oracle pass of 2026-09-22 corrected twenty-odd canonical
# functions and NOT ONE nominal comparison moved, because every fix lived at an
# edge -- a column where MATLAB wants a row, a NaN, a scalar threshold, an
# index past the end, a value nothing matches.
#
# So each entry here is an input where a canonical function and a copy that
# missed the fix give DIFFERENT answers, one of them usually by raising. Keep
# them cheap: they run once per copy of the function.
# ---------------------------------------------------------------------------
EDGE = {
    'pam': [
        # unmatched first pair back-fills 0; unmatched last pair SHORTENS
        lambda: (np.array([0.0, 0.0, 1.0, 1.0]),),
        lambda: (np.array([1.0, 1.0, 0.0, 0.0]),),
        lambda: (np.array([1.0]),),          # MATLAB never assigns dataout
    ],
    'hrem': [
        lambda: (_ISI_TAIL.reshape(-1, 1).copy(), 5, 4, 0.2),   # column
        lambda: (_ISI_TAIL.copy(), 45, 10, 0.2),                # window past end
    ],
    'dfe_clipper': [
        lambda: (_ISI_TAIL.copy(), np.array(0.001), np.array(-0.001)),
    ],
    'Tukey_Window': [
        # non-ascending f: MATLAB concatenates three COUNTED pieces
        lambda: (np.array([1e9, 1e9, 3e9, 3e9, 0.0, 9e9]), _FILT_PARAM(),
                 1e9, 3e9),
        lambda: (np.array([0.0, 1e9, np.nan, 9e9]), _FILT_PARAM(), 1e9, 3e9),
    ],
    'bessel': [lambda: (-1,), lambda: (2.5,), lambda: (0,)],
    'TD_CTLE': [
        lambda: (_ISI_TAIL.astype(complex) + 1j * 0.01, 106.25e9, 20e9,
                 25e9, 40e9, -6.0, 32),                     # complex input
        lambda: (np.array([1.0]), 106.25e9, 20e9, 25e9, 40e9, -6.0, 32),
    ],
    'Butterworth_Filter': [
        lambda: (_FILT_PARAM(), np.asarray(1e9), 0),        # scalar f
    ],
    # Structurally identical to Butterworth_Filter in the reference, and it has
    # the same two traps: MATLAB `if` is true only when every element is
    # non-zero, and length() is the longest dimension.
    'Bessel_Thomson_Filter': [
        lambda: (_FILT_PARAM(), np.asarray(1e9), 0),        # scalar f
        lambda: (_FILT_PARAM(), _F.copy(), [1, 0]),         # falsy in MATLAB
        lambda: (_FILT_PARAM(), np.ones((2, 3)) * 1e9, 0),  # length() = 3
    ],
    'H_interp': [
        lambda: (_Z.copy(), _F[::-1].copy(),
                 np.linspace(0.0, 40e9, 128), 106.25e9),    # descending f_old
        lambda: (_Z.copy(), _F.copy(),
                 np.linspace(0.0, 40e9, 128), 1e6),         # inq empty
    ],
    'd_cpdf': [
        lambda: (1e-4, np.array([-2e-3, 0.0, 2e-3]),
                 np.array([0.5, np.nan, 0.5])),             # NaN counts as support
        lambda: (1e-4, np.array([-2e-3, 0.0, 2e-3]), np.array([0.5, 0.5])),
    ],
    'CDF_ev': [
        lambda: (-1.0, _PDF_A, sicopr.pdf_to_cdf(_PDF_A).y),   # nothing matches
    ],
    'cdf_to_ber_contour': [
        lambda: (sicopr.pdf_to_cdf(_PDF_A), 0.9),              # no crossing
    ],
    'Init_PDF_Fast': [
        lambda: (sicopr.normal_dist(0.01, 5, 1e-4),
                 np.array([0.0, -2e-3, 3e-3]),                 # NOT ascending
                 np.array([0.2, 0.3, 0.5])),
    ],
    # conv_fct's own drift is a ONE-ULP difference in p.x (the colon form vs
    # arange*BinSize), which the 1e-12 absolute tolerance below cannot see on
    # a voltage axis of order 1e-5. These two cases make the same fix show up
    # categorically instead: an empty operand, and a half-integer Min where
    # MATLAB's round goes away from zero.
    'conv_fct': [
        lambda: (SimpleNamespace(BinSize=1e-4, Min=-1,
                                 y=np.array([0.25, 0.5, 0.25]),
                                 x=np.array([-1e-4, 0.0, 1e-4])),
                 SimpleNamespace(BinSize=1e-4, Min=0,
                                 y=np.array([]), x=np.array([]))),
        lambda: (SimpleNamespace(BinSize=1e-4, Min=0.5,
                                 y=np.array([0.25, 0.5, 0.25]),
                                 x=np.array([5e-5, 1.5e-4, 2.5e-4])),
                 SimpleNamespace(BinSize=1e-4, Min=0,
                                 y=np.array([0.5, 0.5]),
                                 x=np.array([0.0, 1e-4]))),
    ],
    'conv_fct_MeanNotZero': [
        lambda: (SimpleNamespace(BinSize=1e-4, Min=0.5,
                                 y=np.array([0.25, 0.5, 0.25]),
                                 x=np.array([5e-5, 1.5e-4, 2.5e-4])),
                 SimpleNamespace(BinSize=1e-4, Min=0,
                                 y=np.array([0.5, 0.5]),
                                 x=np.array([0.0, 1e-4]))),
    ],
}

# Guards the canonical has and many copies do not -- older drift than the
# 2026-09-22 pass, and invisible to everything above because every input there
# uses matching bin sizes and non-negative probabilities. Kept separate only so
# the provenance stays readable; they are driven exactly like the rest.
EDGE['conv_fct'].append(
    lambda: (SimpleNamespace(BinSize=1e-4, Min=-1,
                             y=np.array([0.25, 0.5, 0.25]),
                             x=np.array([-1e-4, 0.0, 1e-4])),
             SimpleNamespace(BinSize=2e-4, Min=0,      # DIFFERENT bin size
                             y=np.array([0.5, 0.5]),
                             x=np.array([0.0, 2e-4]))))
EDGE['d_cpdf'].append(
    lambda: (1e-4, np.array([-2e-3, 0.0, 2e-3]),
             np.array([0.5, -0.25, 0.75])))            # negative probability

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


# Copies that are deliberate FALLBACK STUBS rather than translations. The
# codebase already marks them: the stub's docstring starts with "Stub:".
# Two carry no docstring to mark, so they are named here with the reason.
EXTRA_STUBS = {
    ('_COM_eye_width__conv_fct_MeanNotZero',
     'a one-line delegate to the _conv_fct stub above it'),
    ('_get_RILN_cmp_td__Butterworth_Filter',
     'sits under the "Callee stubs" header and computes 1/sqrt(1+(f/(fb/2))^8), '
     'not the reference polynomial'),
}
_EXTRA_STUB_NAMES = {n for n, _ in EXTRA_STUBS}


def _is_stub(copy_name):
    """True for a deliberate stand-in, by the codebase's own marker."""
    if copy_name in _EXTRA_STUB_NAMES:
        return True
    doc = (ast.get_docstring(_TOPS[copy_name]) or '').strip().lower()
    return doc.startswith('stub')


def _outcome(fn, args, n_pos):
    """Run fn and return a comparable outcome, treating a raise as a result.

    Raising IS behaviour. The 2026-09-22 oracle pass fixed twenty-odd
    functions and most of those fixes turned a silent wrong answer INTO a
    raise, matching a call the MATLAB reference refuses. A differential that
    skips whenever either side raises cannot see any of that: the canonical
    would raise, the stale copy would answer, and the test would call it a
    skip rather than a difference.
    """
    try:
        return ('value', _flat(fn(*_copy.deepcopy(args[:n_pos]))))
    except Exception as e:                                   # noqa: BLE001
        return ('raise', type(e).__name__)


def _differ(a, b):
    """(same, detail) for two outcomes."""
    ka, va = a
    kb, vb = b
    if ka != kb:
        return False, ('canonical %s, copy %s'
                       % (va if ka == 'raise' else 'returned a value',
                          vb if kb == 'raise' else 'returned a value'))
    if ka == 'raise':
        return va == vb, 'canonical raised %s, copy raised %s' % (va, vb)
    if va.shape != vb.shape:
        return False, 'shape %s vs %s' % (va.shape, vb.shape)
    if va.size == 0:
        return True, 'both empty'
    # NaN and Inf have to be compared as PATTERNS, not by subtraction.
    # nan - nan and inf - inf are both nan, and `nan <= 1e-12` is False, so a
    # plain max|delta| reports two bit-identical arrays as different -- it fails
    # a function compared against ITSELF. That matters here because several of
    # the 2026-09-22 fixes are "propagate NaN the way MATLAB does", so the very
    # cases this test exists to check are the ones it would have got wrong.
    nan_a, nan_b = np.isnan(va), np.isnan(vb)
    if not np.array_equal(nan_a, nan_b):
        return False, ('NaN pattern differs (%d vs %d)'
                       % (int(nan_a.sum()), int(nan_b.sum())))
    keep = ~nan_a
    if not keep.any():
        return True, 'all NaN, identical pattern'
    ka, kb = va[keep], vb[keep]
    fin_a, fin_b = np.isfinite(ka), np.isfinite(kb)
    if not np.array_equal(fin_a, fin_b):
        return False, 'infinity pattern differs'
    if not np.array_equal(ka[~fin_a], kb[~fin_b]):
        return False, 'infinities differ in sign'
    ka, kb = ka[fin_a], kb[fin_a]
    if ka.size == 0:
        return True, 'all non-finite, identical pattern'
    d = np.max(np.abs(ka - kb))
    return d <= 1e-12, 'max|delta| = %.3g' % d


_compared = _skipped = 0
for _name, _child, _parent in COPIES:
    if _child not in FACTORY:
        _skipped += 1
        continue
    _key = (_child, _parent)
    # Copies legitimately take fewer positional args than the canonical --
    # get_pdf_from_sampled_signal has 3- and 4-arg forms, Tukey_Window 2- and
    # 4-arg. Drive each side with the arguments IT accepts, taken from the
    # same case tuple, so the comparison stays like-for-like.
    _na = len(_TOPS[_name].args.args)
    _nb = len(_TOPS[_child].args.args)
    _canon, _copyfn = getattr(sicopr, _name), getattr(sicopr, _child)

    try:
        _cases = [('nominal', FACTORY[_child]())]
    except Exception as _e:                                  # noqa: BLE001
        _skipped += 1
        check("inlined_copy_callable__%s__in__%s" % (_child, _parent),
              False, "could not build the canonical arguments: %s: %s"
              % (type(_e).__name__, _e))
        continue
    # A FALLBACK STUB is not a translation of the canonical -- it is a crude
    # stand-in reached only when dependency injection is skipped, and it is
    # expected to differ. Driving one with the edge inputs produces noise, and
    # worse, it pressures whoever is fixing copies into making the stub
    # faithful just to quiet the test: the test steering the code rather than
    # describing it. The nominal comparison still runs, so a stub that drifts
    # from the canonical on ordinary input is still reported.
    _edges = (() if (_key in KNOWN_BEHAVIOUR or _is_stub(_name))
              else EDGE.get(_child, ()))
    for _i, _mk in enumerate(_edges):
        try:
            _cases.append(('edge%d' % _i, _mk()))
        except Exception as _e:                              # noqa: BLE001
            # Never swallow this. Silently dropping an edge case that fails to
            # BUILD makes a test that exercises nothing look like a test that
            # passes -- which is how the conv_fct cases sat inert behind a
            # NameError until the failure count refused to move.
            check("edge_case_builds__%s__edge%d" % (_child, _i), False,
                  "the edge case could not be constructed, so it tested "
                  "nothing: %s: %s" % (type(_e).__name__, _e))

    for _label, _args in _cases:
        _oa = _outcome(_canon, _args, _na)
        _ob = _outcome(_copyfn, _args, _nb)
        # A case that BOTH sides refuse to be driven with says nothing about
        # drift -- it usually means the synthetic inputs do not fit.
        if (_oa[0] == 'raise' and _ob[0] == 'raise'
                and _oa[1] == 'TypeError' and _ob[1] == 'TypeError'):
            _skipped += 1
            continue
        _same, _detail = _differ(_oa, _ob)
        _compared += 1
        _tag = "%s__in__%s" % (_child, _parent)
        if _label != 'nominal':
            _tag += "__" + _label
        if _key in KNOWN_BEHAVIOUR and _label == 'nominal':
            xcheck("inlined_copy_matches__%s" % _tag, _same,
                   "%s (%s)" % (KNOWN_BEHAVIOUR[_key], _detail))
        else:
            check("inlined_copy_matches__%s" % _tag, _same,
                  "the inlined copy of %s inside %s no longer behaves like the "
                  "canonical function on the %s input (%s). A fix applied to "
                  "com_functions/fn/%s/py_impl.py does NOT reach this copy -- "
                  "engine defect #6 was exactly this, across three copies."
                  % (_child, _parent, _label, _detail, _child))

print("\n%d inlined copies of %d functions; %d comparison(s) made, "
      "%d skipped as not drivable synthetically"
      % (len(COPIES), len({c for _, c, _ in COPIES}), _compared, _skipped))

# A copy with no FACTORY entry is only ARITY-checked: nothing ever compares what
# it computes. That is a silent blind spot, not a neutral gap -- it is how
# find_eye_width's vref_intersect copy kept its negative-index wrap through the
# whole 2026-09-22 propagation pass while this test reported success. The set is
# pinned so a NEW undrivable copy has to be looked at, and so the count can only
# go down.
BASELINE_UNDRIVABLE = 36
_undrivable = sorted({(c, p) for _, c, p in COPIES if c not in FACTORY})

check("undrivable_copy_set_does_not_grow",
      len(_undrivable) <= BASELINE_UNDRIVABLE,
      "inlined copies that nothing drives behaviourally rose from %d to %d. "
      "Each is arity-checked only, so a defect in one is invisible here. Add a "
      "FACTORY entry, or record why no single argument tuple can drive both "
      "sides. Current set: %s"
      % (BASELINE_UNDRIVABLE, len(_undrivable), _undrivable))

check("undrivable_baseline_is_current",
      len(_undrivable) >= BASELINE_UNDRIVABLE,
      "behavioural coverage of the copies improved: %d undrivable, down from "
      "%d -- lower BASELINE_UNDRIVABLE to %d so the gain is held"
      % (len(_undrivable), BASELINE_UNDRIVABLE, len(_undrivable)))

finish()
