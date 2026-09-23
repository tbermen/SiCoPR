"""Verification tests for s_for_c4() — 4-port S-params for balanced shunt cap.

# ============================================================
# MATLAB GROUND TRUTH (com_ieee8023_4p16p0.m lines 11646-11650)
#   S2  = s_for_c2(zref,f,cpad);
#   S4P = s2_to_s4(S2.Parameters);
#   S   = sparameters(S4P,f,zref);
#   S.Parameters = snp2smp(S.Parameters,zref,[ 1 3 2 4]);
#
# s2_to_s4 is the block-diagonal 2-port -> 4-port embedding (two uncoupled
# copies of the cap, on raw ports 1-2 and 3-4), and snp2smp with M == N == 4
# terminates no port, so it is the pure port permutation
#   new(i,j) = old(p(i), p(j)),  p = [1 3 2 4]
# putting the result in COM's [tx+ tx- rx+ rx-] order.  With
# s2 = [[a, b], [b, a]] the answer is therefore the checkerboard
#
#   [a 0 b 0]          NOT the block diagonal  [a b 0 0]
#   [0 a 0 b]                                  [b a 0 0]
#   [b 0 a 0]                                  [0 0 a b]
#   [0 b 0 a]                                  [0 0 b a]
#
# At DC (f=0): a=0, b=1.
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.s_for_c4.py_impl import s_for_c4
from com_functions.fn.s_for_c2.py_impl import s_for_c2


def test_output_shape():
    S = s_for_c4(50.0, np.linspace(0, 10e9, 5), 1e-12)
    assert S.Parameters.shape == (4, 4, 5)


def test_dc_is_through_on_both_conductors():
    """At DC the cap is open: each tx conductor passes straight to its rx."""
    S = s_for_c4(50.0, np.array([0.0]), 1e-12)
    p = S.Parameters[:, :, 0]
    expect = np.array([[0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [1, 0, 0, 0],
                       [0, 1, 0, 0]], dtype=complex)
    np.testing.assert_allclose(p, expect, atol=1e-12)


def test_no_coupling_between_the_two_conductors():
    """The two shunt caps are uncoupled, so the +/- cross terms stay zero."""
    f = np.linspace(1e9, 5e9, 4)
    p = s_for_c4(50.0, f, 1e-12).Parameters
    for i, j in [(0, 1), (0, 3), (1, 0), (1, 2),
                 (2, 1), (2, 3), (3, 0), (3, 2)]:
        np.testing.assert_allclose(p[i, j, :], 0, atol=1e-30)


def test_passivity_of_each_conductor():
    """|s11|^2 + |s31|^2 <= 1 down each conductor."""
    p = s_for_c4(50.0, np.linspace(1e8, 100e9, 20), 1e-12).Parameters
    for refl, thru in [(p[0, 0, :], p[2, 0, :]), (p[1, 1, :], p[3, 1, :])]:
        assert np.all(np.abs(refl) ** 2 + np.abs(thru) ** 2 <= 1.0 + 1e-12)


# ============================================================
# COM Octave oracle — NOT AVAILABLE for this function, and why.
#
# s_for_c4 cannot be run by the reference at all.  It calls `s2_to_s4`, which
# is defined nowhere: not in matlab/com_ieee8023_4p10p0..4p16p0.m, not in the
# 802-COM `src/` tree that ships s_for_c4.m itself, and not in MATLAB's RF
# Toolbox.  tools/octave_oracle.py reports `s2_to_s4: NOT FOUND`, as it does
# for `snp2smp` and `sparameters`.  Nothing in COM calls s_for_c4 either, in
# any release — it is dead code, which is presumably how the missing
# dependency has gone unnoticed.  So the numbers below are reasoned from the
# reference text, not oracled, and are marked as such.
#
# What IS settled by the reference text is the step the port had dropped.
# The old implementation returned s2_to_s4's block diagonal unchanged, on the
# stated grounds that snp2smp([1 3 2 4]) "applies the mixed-mode
# transformation ... this transformation leaves the matrix unchanged".  Both
# halves of that are wrong:
#
#   * snp2smp is a port subset/reorder, not a mixed-mode transform (that is
#     s2smm).  With M == N == 4 no port is terminated, so it reduces to the
#     symmetric permutation new(i,j) = old(p(i), p(j)).
#   * p = [1 3 2 4] swaps ports 2 and 3, which on a block diagonal with
#     b != 0 is not the identity.  The reference performs the identical
#     operation on read channels at line 10050, `sch=sch(:,port_order,
#     port_order)`, and line 10434 gives that port order the same default
#     vector: `'Port Order', true, [1 3 2 4]  % [ tx+ tx- rx+ rx-]`.
#
# The physical consequence is the sharpest check available: with the
# permutation dropped, port 3 is the far end of the *other* capacitor, so
# Sdd21 = (S31 - S41 - S32 + S42)/2 came out identically 0 — a shunt
# capacitor with no differential insertion loss at any frequency.
# ============================================================

def _s2_block_diagonal(zref, f, cpad):
    """What s2_to_s4 hands to snp2smp: the unpermuted block diagonal."""
    s2p = s_for_c2(zref, f, cpad).Parameters
    N = s2p.shape[2]
    S4P = np.zeros((4, 4, N), dtype=complex)
    S4P[0:2, 0:2, :] = s2p
    S4P[2:4, 2:4, :] = s2p
    return S4P


def test_ports_two_and_three_are_swapped_against_the_block_diagonal():
    """snp2smp([1 3 2 4]) is applied, so the result is not the block diagonal."""
    f = np.linspace(1e8, 10e9, 8)
    raw = _s2_block_diagonal(50.0, f, 1e-12)
    got = s_for_c4(50.0, f, 1e-12).Parameters
    perm = [0, 2, 1, 3]
    np.testing.assert_array_equal(got, raw[perm, :, :][:, perm, :])
    # and that permutation genuinely moves something
    assert not np.allclose(got, raw)


def test_differential_insertion_loss_is_not_identically_zero():
    """Sdd21 = (S31 - S41 - S32 + S42)/2 must equal the single-ended through."""
    f = np.linspace(1e8, 50e9, 16)
    p = s_for_c4(50.0, f, 1e-12).Parameters
    sdd21 = (p[2, 0, :] - p[3, 0, :] - p[2, 1, :] + p[3, 1, :]) / 2.0
    se_thru = s_for_c2(50.0, f, 1e-12).Parameters[1, 0, :]
    np.testing.assert_allclose(sdd21, se_thru, rtol=1e-14, atol=0)
    assert np.all(np.abs(sdd21) > 0.1)


def test_through_path_runs_tx_plus_to_rx_plus():
    """In [tx+ tx- rx+ rx-] order the through terms are S31 and S42."""
    f = np.array([0.0, 1e9, 26.5625e9])
    p = s_for_c4(50.0, f, 1e-12).Parameters
    s2p = s_for_c2(50.0, f, 1e-12).Parameters
    np.testing.assert_array_equal(p[2, 0, :], s2p[1, 0, :])   # tx+ -> rx+
    np.testing.assert_array_equal(p[3, 1, :], s2p[1, 0, :])   # tx- -> rx-
    np.testing.assert_array_equal(p[0, 0, :], s2p[0, 0, :])   # tx+ reflection
    np.testing.assert_array_equal(p[1, 1, :], s2p[0, 0, :])   # tx- reflection


def test_frequencies_and_impedance_carried_through():
    f = np.linspace(0, 10e9, 5)
    S = s_for_c4(50.0, f, 1e-12)
    np.testing.assert_array_equal(S.Frequencies, f)
    assert S.Impedance == 50.0
