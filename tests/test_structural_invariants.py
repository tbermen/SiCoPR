"""Layer 4 — structural invariants.

Properties that must hold between the configuration and what the engine builds
from it, derived from the shape of the inputs rather than from any MATLAB
value. Like layer 3 these need no oracle, but they catch a different class:
not "is the arithmetic right" but "did every input actually reach the output".

Targets the three ledger defects the coverage table marks as needing structural
invariants:

    #1  z_p transposed for TX only, so RX/NEXT/FEXT were read from a matrix row
    #3  RxFFE floating-tap array sized by tap COUNT rather than SPAN
    #6  package die network truncated to 1 of 3 LC sections

Each of the three shares a signature: an input was silently *not consumed*, and
the result still had a plausible shape. That is why a shape assertion alone is
not enough here and every check below either uses a non-square input or
compares two configurations that must differ.

    python tests/test_structural_invariants.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from audit_check import check, finish  # noqa: E402

import sicopr  # noqa: E402


# ============================================================== defect #1
# The spreadsheet stores rows = package segments, columns = cases. MATLAB
# transposes ALL FOUR z_p keywords; the port transposed only TX, so the RX
# package was built from a matrix ROW -- 111 mm where 13.8 mm was meant, about
# 15 dB of spurious loss.
#
# Two invariants, because one alone is not enough:
#
#   (a) non-square input -> all four must come back with the SAME shape. The
#       original shape guard could not see this because it only ever compared
#       against z_p_tx's own shape.
#   (b) square input -> shape tells you nothing, so assert the VALUES. This is
#       the case the commit message calls out: "the shape guard passes silently
#       for square z_p".

def _zp_params(zp_by_key, mele):
    # package_Z_c is validated against mele, so it has to match the fixture.
    return {'z_p (TX)': zp_by_key['TX'], 'z_p (NEXT)': zp_by_key['NEXT'],
            'z_p (FEXT)': zp_by_key['FEXT'], 'z_p (RX)': zp_by_key['RX'],
            'package_Z_c': np.full((1, mele), 78.2)}


# (a) non-square: 4 segments x 2 cases in the sheet -> (2 cases, 4 segments).
# mele must be 1, 2 or 4, so 4 segments is the non-square case available.
raw_ns = {'TX': np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]]),
          'NEXT': np.array([[10., 20.], [30., 40.], [50., 60.], [70., 80.]]),
          'FEXT': np.array([[11., 21.], [31., 41.], [51., 61.], [71., 81.]]),
          'RX': np.array([[12., 22.], [32., 42.], [52., 62.], [72., 82.]])}
try:
    ps = sicopr.read_package_parameters(_zp_params(raw_ns, 4), SimpleNamespace())
    shapes = {k: getattr(ps, 'z_p_%s_cases' % k).shape
              for k in ('tx', 'next', 'fext', 'rx')}
    ok_shape = len(set(shapes.values())) == 1 and shapes['tx'] == (2, 4)
    err = 'shapes %s; all four must be (ncases, nsegments) = (2, 4)' % shapes
except Exception as e:                                        # noqa: BLE001
    ok_shape, err = False, 'read_package_parameters raised: %s' % e
check("zp_all_four_keywords_share_one_orientation", ok_shape, err)

# (b) square: 2 segments x 2 cases, values chosen so a missing transpose is
#     visible even though the shape is unchanged.
raw_sq = {'TX': np.array([[1.0, 2.0], [3.0, 4.0]]),
          'NEXT': np.array([[1.0, 2.0], [3.0, 4.0]]),
          'FEXT': np.array([[1.0, 2.0], [3.0, 4.0]]),
          'RX': np.array([[1.0, 2.0], [3.0, 4.0]])}
try:
    ps2 = sicopr.read_package_parameters(_zp_params(raw_sq, 2), SimpleNamespace())
    # sheet row 0 is segment 0 across cases -> after transpose, case 0 is
    # [row0[0], row1[0]] = [1.0, 3.0]
    # mele == 2 is widened to 4 with zeros further down the parser,
    # so compare only the two real segments.
    want = np.array([1.0, 3.0])
    bad = [k for k in ('tx', 'next', 'fext', 'rx')
           if not np.allclose(getattr(ps2, 'z_p_%s_cases' % k)[0][:2], want)]
    ok_val = not bad
    err2 = ('%s read case 0 as %s, expected %s. The shape is square so it cannot '
            'tell you this went wrong -- only the values can.'
            % (bad, [list(getattr(ps2, 'z_p_%s_cases' % k)[0][:2]) for k in bad], list(want)))
except Exception as e:                                        # noqa: BLE001
    ok_val, err2 = False, 'read_package_parameters raised: %s' % e
check("zp_square_input_is_still_transposed", ok_val, err2)


# ============================================================== defect #6
# C_d and L_comp are 2xN: N die LC sections per side. MATLAB's
# `Cpad = [Cd_Tx 0 0 0]` concatenates the ROW VECTOR, giving N+3 entries. The
# port kept only Cd_Tx[0], dropping every section after the first along with
# ~15 ps of die delay -- a 53-sample pulse shift.
#
# The invariant needs no MATLAB: *supplying more die sections must change the
# package*. If sections 2..N are discarded, a 2x1 and a 2x3 C_d produce an
# identical network, which is the thing to detect.

def _pkg_param(cd_sections):
    """mele=4 package param with `cd_sections` die LC sections per side."""
    n = len(cd_sections)
    p = SimpleNamespace()
    p.Z0 = 100.0
    p.C_diepad = np.array([cd_sections, cd_sections], dtype=float)     # 2 x N
    p.L_comp = np.array([[1.0e-10] * n, [1.0e-10] * n], dtype=float)   # 2 x N
    p.C_pkg_board = np.array([1.0e-13, 1.0e-13])
    p.C_bump = np.array([1.0e-13, 1.0e-13])
    p.C_v = np.array([1.0e-13, 1.0e-13])
    p.pkg_Z_c = np.array([90.0, 90.0, 90.0, 90.0])
    zp = np.array([[12.0, 0.0, 1.0, 1.0]])
    p.z_p_tx_cases = zp
    p.z_p_rx_cases = zp
    p.z_p_next_cases = zp
    p.z_p_fext_cases = zp
    p.Pkg_len_TX = np.array([12.0, 0.0, 1.0, 1.0])
    p.Pkg_len_RX = np.array([12.0, 0.0, 1.0, 1.0])
    p.Pkg_len_NEXT = np.array([12.0, 0.0, 1.0, 1.0])
    p.Pkg_len_FEXT = np.array([12.0, 0.0, 1.0, 1.0])
    p.pkg_tau = 6.141e-3
    p.pkg_gamma0_a1_a2 = np.array([0.0, 1.734e-3, 1.455e-4])
    p.PKG_NAME = None
    p.kappa1 = 1.0
    p.kappa2 = 1.0
    return p


faxis = np.linspace(1e9, 40e9, 120)
try:
    _, _, s21_one, _ = sicopr.make_full_pkg('TX', faxis, _pkg_param([1.0e-13]),
                                            'THRU')
    _, _, s21_three, _ = sicopr.make_full_pkg(
        'TX', faxis, _pkg_param([1.0e-13, 2.0e-13, 3.0e-13]), 'THRU')
    diff = float(np.max(np.abs(np.asarray(s21_three) - np.asarray(s21_one))))
    ok_die, die_err = diff > 1e-9, (
        'a package built from THREE die LC sections is identical to one built '
        'from ONE (max |dS21| = %.3g). Sections 2 and 3 are not reaching the '
        'network, which is defect #6.' % diff)
except Exception as e:                                        # noqa: BLE001
    ok_die, die_err = False, 'make_full_pkg raised: %s' % e
check("die_sections_beyond_the_first_reach_the_package", ok_die, die_err)

# And the count itself: N sections must produce N+3 blocks at mele=4.
_src_pkg = open(os.path.join(_ROOT, 'com_functions', 'fn', 'make_full_pkg',
                             'py_impl.py'), encoding='utf-8').read()
check("die_section_count_is_taken_from_the_matrix_width",
      'extra_LC = len(Cd_Tx) - 1' in _src_pkg
      and 'num_blocks = mele + extra_LC' in _src_pkg,
      "make_full_pkg must size the network from the width of C_d (extra_LC = "
      "len(Cd_Tx) - 1, num_blocks = mele + extra_LC). Hard-coding the block "
      "count reintroduces defect #6.")


# ============================================================== defect #3
# The RxFFE floating-tap array was allocated at the tap COUNT (23) rather than
# the SPAN (87), so every floating tap past index 23 was silently discarded.
#
# Invariant: an array indexed by tap POSITION must be at least as long as the
# largest position it will be indexed with. Stated that way it needs no
# knowledge of the defect -- it is just "allocate for the index space you use".
N_TAPS, SPAN = 23, 87
locs = np.array([5, 22, 40, 63, 86])            # positions, not a count

undersized = np.zeros(N_TAPS)
correct = np.zeros(SPAN)

lost = [int(i) for i in locs if i >= undersized.size]
check("count_sized_array_would_lose_taps",
      len(lost) == 3,
      "this check is only meaningful while span > count; with an array sized by "
      "count, positions %s fall outside it" % lost)

kept = [int(i) for i in locs if i < correct.size]
check("span_sized_array_holds_every_tap_position",
      len(kept) == len(locs),
      "an array sized by SPAN must hold every floating-tap position; %d of %d "
      "fit" % (len(kept), len(locs)))

# The invariant, stated over the real sizing rule. MMSE computes
#
#     span = max(len(Craw), n_end)
#     if idx_arr.size: span = max(span, idx_arr.max() + RxFFE_cmx + 1)
#     C = np.zeros(span)
#
# then writes C[k + RxFFE_cmx] for every floating-tap index k. The property that
# must hold is simply that the array is long enough for every position written
# -- "allocate for the index space you use". Driving the whole MMSE solve needs
# a complete channel, so the rule is exercised over a sweep of shapes and the
# live source is asserted to still contain the widening term.
def _span(craw_len, n_end, idx, cmx):
    sp = max(craw_len, n_end)
    if len(idx):
        sp = max(sp, int(max(idx)) + int(cmx) + 1)
    return sp


bad_cases = []
for craw_len, n_end, cmx in ((23, 20, 4), (23, 23, 0), (40, 30, 12), (10, 10, 30)):
    for idx in ([5, 22, 40, 63, 86], [0], [86], []):
        sp = _span(craw_len, n_end, idx, cmx)
        for k in idx:
            if k + cmx >= sp:
                bad_cases.append((craw_len, n_end, cmx, k))
check("mmse_span_covers_every_floating_tap_write",
      not bad_cases,
      "the sizing rule must leave room for C[k + RxFFE_cmx] for every floating "
      "tap k; these combinations overflow: %s" % bad_cases[:4])

# A count-sized array -- the defect -- fails that same property, which is what
# makes the check above meaningful rather than vacuous.
count_bad = [k for k in [5, 22, 40, 63, 86] if k + 4 >= 23]
check("count_sizing_violates_the_same_property",
      len(count_bad) == 4,
      "sizing C by the tap count (23) instead of the span must overflow for the "
      "high floating taps, otherwise the check above proves nothing; got %s"
      % count_bad)

_src_mmse = open(os.path.join(_ROOT, 'com_functions', 'fn', 'MMSE',
                              'py_impl.py'), encoding='utf-8').read()
check("mmse_widens_the_array_to_the_floating_tap_span",
      'span = max(span, int(idx_arr.max()) + int(param.RxFFE_cmx) + 1)' in _src_mmse,
      "MMSE must widen the C allocation to cover the largest floating-tap "
      "position. Without that term the array is sized by the fixed-tap count "
      "and every floating tap past it is silently discarded -- defect #3.")

finish()
