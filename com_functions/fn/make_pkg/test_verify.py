"""Verification tests for make_pkg().

# ============================================================
# MATLAB GROUND TRUTH (lines 8359-8405)
# Returns 4 S-parameter arrays (s11,s12,s21,s22) for package model:
#   pad capacitor → (optional L/C comps) → transmission line → ball capacitor
# f(f<eps)=eps to prevent division by zero
# For matched tline (pkg_z=Z0, len=0): s11≈0, s21≈1
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.make_pkg.py_impl import make_pkg


def _param():
    return SimpleNamespace(
        pkg_tau=0.0,
        pkg_gamma0_a1_a2=np.array([0.0, 0.01, 0.001]),
        Z0=50.0,
    )


def test_output_length():
    """Output arrays have length N == len(f)."""
    N = 20
    f = np.linspace(1e9, 50e9, N)
    s11, s12, s21, s22 = make_pkg(f, 0.0, 1e-12, 1e-12, 50.0, _param())
    for arr in (s11, s12, s21, s22):
        assert len(arr) == N


def test_output_is_complex():
    """S-parameters are complex arrays."""
    f = np.linspace(1e9, 50e9, 10)
    s11, s12, s21, s22 = make_pkg(f, 0.01, 1e-12, 1e-12, 50.0, _param())
    assert np.iscomplexobj(s11)


def test_symmetric_tline_s21_near_one():
    """Matched tline (pkg_z=Z0) with tiny length → |s21| close to 1."""
    f = np.linspace(1e8, 10e9, 50)
    s11, _, s21, _ = make_pkg(f, 0.001, 1e-15, 1e-15, 50.0, _param())
    # For very small caps and very short tline, s21 ≈ 1
    np.testing.assert_allclose(np.abs(s21), 1.0, atol=0.1)


def test_with_lcomp():
    """Optional lcomp argument accepted without error."""
    f = np.linspace(1e9, 20e9, 10)
    s11, s12, s21, s22 = make_pkg(f, 0.01, 1e-12, 1e-12, 50.0, _param(), 0.1e-9)
    assert len(s21) == 10


def test_with_cbump():
    """Optional cbump argument accepted without error."""
    f = np.linspace(1e9, 20e9, 10)
    s11, s12, s21, s22 = make_pkg(f, 0.01, 1e-12, 1e-12, 50.0, _param(), 0.0, 0.1e-12)
    assert len(s21) == 10


# ============================================================
# COM Octave oracle — make_pkg extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m) with combines4p and synth_tline, run by
# tools/octave_oracle.py.
#
# param.pkg_tau=6.141e-3, pkg_gamma0_a1_a2=[0 1.734e-3 1.455e-4], Z0=50,
# pkg_len=12, cpad=0.18e-12, cball=0.11e-12, pkg_z=87.5, on
# f = [0 1e6 1e9 13.28125e9 26.5625e9 53e9].  Findings:
#
#   * `f(f<eps)=eps` is eps(1)=2.220446049250313e-16.  The DC point therefore
#     comes back as s11=-1.3084771361653632e-15-1.3131782722674007e-15j, not
#     as the ~1e-323 a np.finfo(float).tiny floor would give.  The canonical
#     had the right constant; three INLINED copies did not.
#   * f<eps is clamped whatever its sign, so -1e9, -1, 0 and eps/2 all return
#     the identical row (case D below).
#   * lcomp<=0 and cbump<=0 skip their branches entirely, so passing 0,0 is
#     bit-identical to passing no varargin at all (cases B and C below).
#   * pkg_len=0 forces rho_rl=0 in synth_tline, so the line is transparent and
#     only the three capacitors remain (case E).
#
# Tolerance: Octave's complex `*` and `/` are not correctly rounded and neither
# is numpy's -- measured over 300 random pairs, Octave was nearer the exact
# result 43 times and numpy 25 (for 2./z), which puts a ~1 ULP floor under
# every element.  The four-stage cascade here, plus the catastrophic
# cancellation of (1-exp_gamma_d^2) in synth_tline at f=1e6, amplifies that to
# 1.1e-13 relative at worst, so these are pinned at rtol=1e-11.  That is still
# 11 orders tighter than the eps/tiny defect above, which moves s11 at DC by
# a relative 1.0.
# ============================================================

def _oct_param():
    return SimpleNamespace(
        pkg_tau=6.141e-3,
        pkg_gamma0_a1_a2=np.array([0.0, 1.734e-3, 1.455e-4]),
        Z0=50.0,
    )


OCT_F = np.array([0.0, 1e6, 1e9, 13.28125e9, 26.5625e9, 53e9])
OCT_RTOL = 1e-11

# case A: the call shape COM actually uses (8 args, lcomp and cbump both > 0)
A_S11 = np.array([
    -1.3084771361653632e-15 - 1.3131782722677496e-15j,
    -8.8529973588096026e-05 - 0.0001980569027350556j,
    -0.051206339700247294 - 0.086600783728771613j,
    -0.29825587809508924 - 0.35217160968790728j,
    -0.53613500709953921 - 0.26755205212841371j,
    -0.21237741345598438 - 0.46806862794363557j])
A_S21 = np.array([
    0.99999999999999012 - 9.8926096510196324e-15j,
    0.99933383716989233 - 0.0012039371361369192j,
    0.82844507207382789 - 0.50942327653199915j,
    0.60940038866345669 - 0.51356800882747733j,
    0.25584827817503947 - 0.60693021907410993j,
    -0.50129140003350958 - 0.24091163175083646j])
A_S22 = np.array([
    -1.3084771361653632e-15 - 1.3131782722677496e-15j,
    -8.854907010112229e-05 - 0.00019804570952829462j,
    -0.057561663377428129 - 0.082306764757327913j,
    -0.27567786860389054 - 0.34996964690838461j,
    -0.4891538849132348 - 0.30299010859620529j,
    -0.13236975155256075 - 0.39524098635922744j])

# case B: six arguments, no varargin
B_S11 = np.array([
    -1.3084771361653632e-15 - 1.3131782722674007e-15j,
    -8.8522775517507489e-05 - 0.00019648972191781385j,
    -0.048533318905434483 - 0.087050813456371889j,
    -0.22723994712355861 - 0.43000924608448526j,
    -0.50814127657326547 - 0.54010651605704363j,
    -0.77667186387923737 - 0.48461904471646156j])
B_S21 = np.array([
    0.99999999999999012 - 9.892609651015098e-15j,
    0.9993338618557851 - 0.0011835305263990198j,
    0.83868806782456884 - 0.49256036109665879j,
    0.70344495152158026 - 0.34823791672407134j,
    0.41392086662885741 - 0.36508817246900577j,
    0.15436631004008944 - 0.21034245525113654j])
B_S22 = np.array([
    -1.3084771361653632e-15 - 1.3131782722674007e-15j,
    -8.8544464863105012e-05 - 0.00019647700954640091j,
    -0.055853038085671373 - 0.082318576552781589j,
    -0.20112953995867044 - 0.42083736372621483j,
    -0.42590115676879264 - 0.55653684055080777j,
    -0.63407100296143959 - 0.59240299310572508j])

# case E: pkg_len=0, so synth_tline is transparent (rho_rl forced to 0)
E_S11 = np.array([
    -1.023095562055623e-52 - 1.0114818644225032e-26j,
    -2.0750843210230624e-09 - 4.555309338252549e-05j,
    -0.0020707872671299251 - 0.045458762711101344j,
    -0.2679502876651576 - 0.44289155671034963j,
    -0.59417392681831727 - 0.49105119031280053j,
    -0.85356378599419924 - 0.35354299488668711j])
E_S21 = np.array([
    1 - 1.0114818644225032e-26j,
    0.99999999792491567 - 4.5553093382525483e-05j,
    0.99792921273287016 - 0.045458762711101337j,
    0.73204971233484217 - 0.44289155671034958j,
    0.40582607318168273 - 0.49105119031280048j,
    0.14643621400580081 - 0.35354299488668717j])


def test_oracle_full_call_with_lcomp_and_cbump():
    """Pin the COM Octave answer for the 8-argument call COM itself makes."""
    s11, s12, s21, s22 = make_pkg(OCT_F, 12.0, 0.18e-12, 0.11e-12, 87.5,
                                  _oct_param(), 0.15e-9, 0.07e-12)
    np.testing.assert_allclose(s11, A_S11, rtol=OCT_RTOL, atol=0)
    np.testing.assert_allclose(s21, A_S21, rtol=OCT_RTOL, atol=0)
    np.testing.assert_allclose(s12, A_S21, rtol=OCT_RTOL, atol=0)
    np.testing.assert_allclose(s22, A_S22, rtol=OCT_RTOL, atol=0)


def test_oracle_six_argument_call():
    """Pin the COM Octave answer with no varargin at all."""
    s11, s12, s21, s22 = make_pkg(OCT_F, 12.0, 0.18e-12, 0.11e-12, 87.5,
                                  _oct_param())
    np.testing.assert_allclose(s11, B_S11, rtol=OCT_RTOL, atol=0)
    np.testing.assert_allclose(s21, B_S21, rtol=OCT_RTOL, atol=0)
    np.testing.assert_allclose(s22, B_S22, rtol=OCT_RTOL, atol=0)


def test_dc_floor_is_eps_one_not_realmin():
    """`f(f<eps)=eps` is eps(1)=2.22e-16.

    A np.finfo(float).tiny floor is 292 orders smaller and drives the DC point
    of synth_tline's sqrt/log to ~0, collapsing s11 to ~1e-323.  Pinned with
    pkg_len=0 so only the capacitors contribute and the value is clean.
    """
    s11, _, s21, _ = make_pkg(np.array([0.0]), 0.0, 0.18e-12, 0.11e-12, 87.5,
                              _oct_param())
    np.testing.assert_allclose(s11[0], E_S11[0], rtol=OCT_RTOL, atol=0)
    np.testing.assert_allclose(s21[0], E_S21[0], rtol=OCT_RTOL, atol=0)
    assert abs(s11[0]) > 1e-27          # a `tiny` floor gives ~1e-318


def test_oracle_pkg_len_zero_transparent_line():
    """pkg_len=0 forces rho_rl=0, leaving only the three capacitors."""
    s11, _, s21, _ = make_pkg(OCT_F, 0.0, 0.18e-12, 0.11e-12, 87.5, _oct_param())
    np.testing.assert_allclose(s11, E_S11, rtol=OCT_RTOL, atol=0)
    np.testing.assert_allclose(s21, E_S21, rtol=OCT_RTOL, atol=0)


def test_negative_and_subeps_frequencies_all_clamp_to_eps():
    """f<eps is clamped whatever the sign, so the whole row is one value."""
    f = np.array([-1e9, -1.0, 0.0, np.finfo(float).eps / 2])
    s11, _, s21, _ = make_pkg(f, 12.0, 0.18e-12, 0.11e-12, 87.5, _oct_param())
    np.testing.assert_array_equal(s11, np.full(4, s11[0]))
    np.testing.assert_allclose(s11[0], B_S11[0], rtol=OCT_RTOL, atol=0)
    np.testing.assert_allclose(s21[0], B_S21[0], rtol=OCT_RTOL, atol=0)


def test_nonpositive_lcomp_and_cbump_are_exactly_skipped():
    """`if lcomp>0` / `if cbump>0` skip, so 0,0 is bit-identical to no varargin."""
    base = make_pkg(OCT_F, 12.0, 0.18e-12, 0.11e-12, 87.5, _oct_param())
    for extra in ((0.0, 0.0), (-1e-9, -1e-12), (0.0,)):
        got = make_pkg(OCT_F, 12.0, 0.18e-12, 0.11e-12, 87.5, _oct_param(), *extra)
        for a, b in zip(base, got):
            np.testing.assert_array_equal(a, b)
