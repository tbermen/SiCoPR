"""Audit batch B05: G5 S-parameter interpolation and cascade.

Functions under audit (MATLAB com_ieee8023_4p15p0.m -> sicopr.py):
  combines4p     ML 5459-5502 -> py 8052-8064
  stot           ML 11418-11424 -> py 17398-17428
  ttos           ML 11460-11466 -> py 17543-17573
  make_full_pkg  ML 8296-8488 -> py 12088-12264
  s_for_c2       ML 11289-11295 -> py 17098-17122
  s_for_c4       ML 11297-11301 -> py 17149-17173

Traps checked (audit prompt section 3 item 5, section 4 items 8/9/13):
  - S<->T conversion convention and eps guard (delta computed before the eps
    replacement of the divisor).
  - Cascade identity (through yields the original) and T-matrix cross-path
    (combines4p == ttos(stot(S1)@stot(S2))).
  - Reciprocity (S12==S21) and passivity (|S11|^2+|S21|^2 <= 1) preservation.
  - make_full_pkg block dispatch: TX vs RX index selection, dc/cd mode scaling,
    multi-block cascade count.
  - s_for_c4 port reorder snp2smp([1 3 2 4]).

Oracles transcribed from the cited MATLAB lines. Cascade identity, S<->T round
trip, and passivity are analytic invariants (not transcription echoes). FAIL
rows document divergences. Run: python tests/test_sparam_cascade.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import copy
from types import SimpleNamespace

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, xcheck, finish  # noqa: E402
import sicopr  # noqa: E402


def rel_err(a, b):
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    denom = max(np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(a - b)) / denom)


rng = np.random.default_rng(50505)


def rand_passive_2port(nf):
    """A random reciprocal, passive-ish 2-port (S12==S21, |S11|^2+|S21|^2<1)."""
    s11 = 0.3 * (rng.standard_normal(nf) + 1j * rng.standard_normal(nf))
    s21 = 0.5 * (rng.standard_normal(nf) + 1j * rng.standard_normal(nf))
    # normalise so it does not exceed passivity
    scale = np.sqrt(np.abs(s11) ** 2 + np.abs(s21) ** 2) + 1e-9
    s11, s21 = s11 / scale * 0.9, s21 / scale * 0.9
    s22 = 0.3 * (rng.standard_normal(nf) + 1j * rng.standard_normal(nf))
    return s11, s21.copy(), s21.copy(), s22  # (s11, s12, s21, s22) reciprocal


# ===========================================================================
# 1. stot / ttos round-trip (ML 11418-11424, 11460-11466)
# ===========================================================================
nf = 32
s11, s12, s21, s22 = rand_passive_2port(nf)
S = np.zeros((2, 2, nf), dtype=complex)
S[0, 0], S[0, 1], S[1, 0], S[1, 1] = s11, s12, s21, s22

T = sicopr.stot(S)
S_rt = sicopr.ttos(T)
check("stot_ttos_round_trip",
      rel_err(S_rt, S) <= 1e-11,
      "ttos(stot(S)) != S")
T_rt = sicopr.stot(sicopr.ttos(T))
check("ttos_stot_round_trip",
      rel_err(T_rt, T) <= 1e-11,
      "stot(ttos(T)) != T")

# Transcription of ML 11421-11424 (delta uses ORIGINAL s21, before eps guard).
s21o = S[1, 0, :].copy()
delta = S[0, 0] * S[1, 1] - S[0, 1] * s21o
s21g = s21o.copy(); s21g[s21g == 0] = np.finfo(float).eps
T_ml = np.empty((2, 2, nf), dtype=complex)
T_ml[0, 0], T_ml[0, 1] = 1.0 / s21g, -S[1, 1] / s21g
T_ml[1, 0], T_ml[1, 1] = S[0, 0] / s21g, -delta / s21g
check("stot_matches_matlab_transcription",
      rel_err(T, T_ml) <= 1e-13,
      "stot != Mavaddat T transcription")

# ===========================================================================
# 2. combines4p cascade (ML 5482-5502)
# ===========================================================================
a = rand_passive_2port(nf)   # (s11,s12,s21,s22)
b = rand_passive_2port(nf)

o11, o12, o21, o22 = sicopr.combines4p(*a, *b)

# Cross-path: direct cascade formula == ttos(stot(A) @ stot(B)) elementwise.
A = np.zeros((2, 2, nf), dtype=complex)
A[0, 0], A[0, 1], A[1, 0], A[1, 1] = a
B = np.zeros((2, 2, nf), dtype=complex)
B[0, 0], B[0, 1], B[1, 0], B[1, 1] = b
TA, TB = sicopr.stot(A), sicopr.stot(B)
Tcasc = np.einsum('ijf,jkf->ikf', TA, TB)   # T_A @ T_B per frequency
Scasc = sicopr.ttos(Tcasc)
check("combines4p_matches_T_matrix_cascade",
      rel_err(o11, Scasc[0, 0]) <= 1e-9 and rel_err(o12, Scasc[0, 1]) <= 1e-9
      and rel_err(o21, Scasc[1, 0]) <= 1e-9 and rel_err(o22, Scasc[1, 1]) <= 1e-9,
      "combines4p != ttos(stot(A)@stot(B)); T convention/order mismatch")

# Cascade identity: cascading with a matched through returns the original.
thru = (np.zeros(nf, dtype=complex), np.ones(nf, dtype=complex),
        np.ones(nf, dtype=complex), np.zeros(nf, dtype=complex))
i11, i12, i21, i22 = sicopr.combines4p(*a, *thru)
check("combines4p_through_is_identity",
      rel_err(i11, a[0]) <= 1e-12 and rel_err(i12, a[1]) <= 1e-12
      and rel_err(i21, a[2]) <= 1e-12 and rel_err(i22, a[3]) <= 1e-12,
      "cascading with a matched through changed the network")

# Reciprocity preserved: reciprocal inputs -> reciprocal output.
check("combines4p_preserves_reciprocity",
      rel_err(o12, o21) <= 1e-12,
      "cascade of reciprocal 2-ports is not reciprocal (S12 != S21)")

# ===========================================================================
# 3. s_for_c2 (ML 11289-11295) — shunt capacitor
# ===========================================================================
zref = 50.0
f = np.linspace(0, 40e9, 64)
cpad = 0.2e-12
S2 = sicopr.s_for_c2(zref, f, cpad)
p = S2.Parameters

# Transcription (ML 11291-11294).
jwCz = 1j * 2 * np.pi * f * cpad * zref
den = 2 + jwCz
check("s_for_c2_matches_matlab",
      rel_err(p[0, 0], -jwCz / den) <= 1e-13 and rel_err(p[1, 0], 2 / den) <= 1e-13
      and rel_err(p[0, 1], 2 / den) <= 1e-13 and rel_err(p[1, 1], -jwCz / den) <= 1e-13,
      "s_for_c2 != shunt-cap S-parameter transcription")
# DC: cap open -> full through (S11=0, S21=1).
check("s_for_c2_dc_is_through",
      abs(p[0, 0, 0]) <= 1e-12 and abs(p[1, 0, 0] - 1.0) <= 1e-12,
      "at DC the shunt cap is not an open (through)")
# Lossless: |S11|^2 + |S21|^2 == 1 at every frequency.
passivity = np.abs(p[0, 0]) ** 2 + np.abs(p[1, 0]) ** 2
check("s_for_c2_lossless",
      np.max(np.abs(passivity - 1.0)) <= 1e-12,
      "shunt cap is not lossless (|S11|^2+|S21|^2 != 1)")

# ===========================================================================
# 4. s_for_c4 (ML 11297-11301) — port reorder snp2smp([1 3 2 4])
# ===========================================================================
S4 = sicopr.s_for_c4(zref, f, cpad).Parameters   # (4,4,N)

# MATLAB-faithful: block-diagonal embedding then reorder to [1 3 2 4] (0-based [0,2,1,3]).
s2p = sicopr.s_for_c2(zref, f, cpad).Parameters
blockdiag = np.zeros((4, 4, len(f)), dtype=complex)
blockdiag[0:2, 0:2, :] = s2p
blockdiag[2:4, 2:4, :] = s2p
perm = [0, 2, 1, 3]                      # MATLAB [1 3 2 4] -> 0-based
reordered = blockdiag[np.ix_(perm, perm)]

# The reorder is NOT the identity for the block-diagonal (pure permutation fact).
check("s_for_c4_reorder_is_nontrivial",
      rel_err(reordered, blockdiag) > 1e-3,
      "snp2smp([1 3 2 4]) unexpectedly equals the block-diagonal")
# EXPECTED FAIL: sicopr.py returns the block-diagonal, skipping the reorder.
# Was DIVERGENT (unused fn): the port returned the block-diagonal and skipped
# the snp2smp([1 3 2 4]) reorder that ML 11301 applies, on a comment claiming
# the step "leaves the matrix unchanged". It does not: with M==N==4 no port is
# terminated, so it reduces to new(i,j)=old(p(i),p(j)), and [1 3 2 4] swaps
# ports 2 and 3. The consequence was Sdd21 identically ZERO -- a shunt
# capacitor with no differential insertion loss at any frequency. RESOLVED
# 2026-09-22; promoted to check().
#
# Note the function remains unreachable: its dependency s2_to_s4 is defined
# nowhere in any release, the 802-COM src tree, or RF Toolbox, so the reference
# cannot execute it either, and nothing calls s_for_c4. The block-diagonal
# reading of s2_to_s4 is therefore still an assumption; what is now pinned is
# the reorder, which the reference plainly does apply.
check("s_for_c4_applies_port_reorder",
      rel_err(S4, reordered) <= 1e-12,
      "s_for_c4 must apply the snp2smp([1 3 2 4]) reorder of ML 11301; without "
      "it Sdd21 is identically zero")

# ===========================================================================
# 5. make_full_pkg dispatch (ML 8296-8488)
# ===========================================================================
def mk_param(mele):
    if mele == 1:
        z_p = [[85.0]]
        pkg_Z_c = [90.0, 92.0]
        Pkg_len_TX = 7e-3
        Pkg_len_RX = 6e-3
        C_v = [0.0, 0.0]
    else:  # mele == 4
        z_p = [[85.0, 85.0, 85.0, 85.0]]
        pkg_Z_c = [[90.0, 88.0, 91.0, 89.0], [92.0, 90.0, 93.0, 91.0]]
        Pkg_len_TX = [7e-3, 1e-3, 1e-3, 1e-3]
        Pkg_len_RX = [6e-3, 1e-3, 1e-3, 1e-3]
        C_v = [1e-14, 1.1e-14]
    return SimpleNamespace(
        PKG_NAME=None,
        C_diepad=[1.0e-13, 1.2e-13],     # [Cd_Tx, Cd_Rx]
        C_pkg_board=[0.5e-13, 0.6e-13],
        L_comp=[0.1e-9, 0.12e-9],
        C_bump=[0.3e-13, 0.4e-13],
        z_p_next_cases=z_p,
        pkg_Z_c=pkg_Z_c, C_v=C_v,
        Pkg_len_TX=Pkg_len_TX, Pkg_len_RX=Pkg_len_RX,
        Pkg_len_NEXT=Pkg_len_TX, Pkg_len_FEXT=Pkg_len_TX,
        Z0=50.0, pkg_tau=6.141e-3, pkg_gamma0_a1_a2=[0.0, 1.734e-3, 1.455e-4])


fpk = np.linspace(1e7, 40e9, 128)
par1 = mk_param(1)

# mele=1 TX -> single make_pkg with Cd_Tx (index 0), C_pkg_board[0], pkg_Z_c[0].
o_tx = sicopr.make_full_pkg('TX', fpk, par1, 'THRU', 'dd', 1)
o_tx_oracle = sicopr.make_pkg(fpk, 7e-3, 1.0e-13, 0.5e-13, 90.0, par1, 0.1e-9, 0.3e-13)
check("make_full_pkg_TX_mele1_dispatch",
      all(rel_err(o_tx[k], o_tx_oracle[k]) <= 1e-12 for k in range(4)),
      "TX mele=1 did not dispatch make_pkg with the TX (index-0) parameters")

# mele=1 RX -> uses Cd_Rx (index 1), C_pkg_board[1], pkg_Z_c[1], Pkg_len_RX.
o_rx = sicopr.make_full_pkg('RX', fpk, par1, 'THRU', 'dd', 1)
o_rx_oracle = sicopr.make_pkg(fpk, 6e-3, 1.2e-13, 0.6e-13, 92.0, par1, 0.12e-9, 0.4e-13)
check("make_full_pkg_RX_mele1_index_selection",
      all(rel_err(o_rx[k], o_rx_oracle[k]) <= 1e-12 for k in range(4)),
      "RX mele=1 did not use the RX (index-1) parameters")
check("make_full_pkg_TX_RX_differ",
      rel_err(o_tx[2], o_rx[2]) > 1e-6,
      "TX and RX packages are identical (index selection not applied)")

# dc/cd mode scaling: Z0/2, Cpad*2, Cball*2, Zpkg*2, Lcomp/2, Cbump*2 (ML 8467-8475).
par1_dc = copy.copy(par1); par1_dc.Z0 = 25.0
o_tx_dc = sicopr.make_full_pkg('TX', fpk, par1, 'THRU', 'dc', 1)
o_tx_dc_oracle = sicopr.make_pkg(fpk, 7e-3, 1.0e-13 * 2, 0.5e-13 * 2, 90.0 * 2,
                              par1_dc, 0.1e-9 / 2, 0.3e-13 * 2)
check("make_full_pkg_dc_mode_scaling",
      all(rel_err(o_tx_dc[k], o_tx_dc_oracle[k]) <= 1e-12 for k in range(4)),
      "dc mode did not scale Z0/2, Cpad*2, Cball*2, Zpkg*2, Lcomp/2, Cbump*2")

# Physics: package is reciprocal and passive.
recip = rel_err(o_tx[1], o_tx[2])
psv = np.max(np.abs(o_tx[0]) ** 2 + np.abs(o_tx[2]) ** 2)
check("make_full_pkg_reciprocal", recip <= 1e-12,
      "package S12 != S21 (not reciprocal)")
check("make_full_pkg_passive", psv <= 1.0 + 1e-9,
      "package |S11|^2+|S21|^2 = %.6f > 1 (not passive)" % psv)

# mele=4 multi-block cascade == manual 4-block cascade via combines4p.
par4 = mk_param(4)
o4 = sicopr.make_full_pkg('TX', fpk, par4, 'THRU', 'dd', 1)
Cpad4 = [1.0e-13, 0, 0, 0]
Lcomp4 = [0.1e-9, 0, 0, 0]
Cbump4 = [0.3e-13, 0, 0, 0]
Cball4 = [0, 0, 1e-14, 0.5e-13]
Zpkg4 = [90.0, 88.0, 91.0, 89.0]
Len4 = [7e-3, 1e-3, 1e-3, 1e-3]
acc = None
for j in range(4):
    seg = sicopr.make_pkg(fpk, Len4[j], Cpad4[j], Cball4[j], Zpkg4[j], par4, Lcomp4[j], Cbump4[j])
    acc = seg if acc is None else sicopr.combines4p(*acc, *seg)
check("make_full_pkg_mele4_multiblock_cascade",
      all(rel_err(o4[k], acc[k]) <= 1e-11 for k in range(4)),
      "mele=4 cascade != manual 4-block combines4p chain")

finish()
