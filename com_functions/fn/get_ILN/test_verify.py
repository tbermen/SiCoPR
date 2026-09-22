"""Verification tests for get_ILN().

# ============================================================
# MATLAB GROUND TRUTH
# Fit: efit = alpha[0] + alpha[1]*sqrt(f) + alpha[2]*f + alpha[3]*f^2
# Weighted LS: fmbg = [|s|, sqrt(f)*|s|, f*|s|, f^2*|s|]; LGw = |s|*db(s)
# ILN = db(s) - efit  (residual)
#
# Perfect fit case: if db(s) is exactly a0 + a1*sqrt(f) + a2*f + a3*f^2
#   then ILN ≈ 0 everywhere
#
# Output lengths: len(ILN) == len(efit) == len(faxis_f2)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.get_ILN.py_impl import get_ILN


def test_output_lengths():
    """len(ILN) == len(efit) == len(faxis_f2)."""
    n = 20
    faxis = np.linspace(1e9, 50e9, n)
    sdd21 = np.exp(-0.1 * faxis / 1e9) * np.exp(-1j * faxis / 1e9)
    ILN, efit = get_ILN(sdd21, faxis)
    assert len(ILN) == n
    assert len(efit) == n


def test_iln_is_db_minus_efit():
    """ILN = 20*log10(|sdd21|) - efit."""
    n = 15
    faxis = np.linspace(1e9, 40e9, n)
    sdd21 = np.exp(-0.05 * faxis / 1e9 + 0.1j)
    ILN, efit = get_ILN(sdd21, faxis)
    db_s = 20 * np.log10(np.abs(sdd21))
    np.testing.assert_allclose(ILN, db_s - efit, atol=1e-10)


def test_polynomial_il_small_iln():
    """If db(s) is exactly a0+a1*sqrt(f)+a2*f, the fit recovers it and ILN≈0.

    We construct sdd21 such that 20*log10(|sdd21|) = 1 - 0.5*sqrt(f_norm)
    where f_norm = faxis/faxis[-1].  The polynomial basis can represent this
    exactly, so the weighted LS residual ILN should be near machine epsilon.
    """
    n = 20
    faxis = np.linspace(1e9, 50e9, n)
    f_n = faxis / faxis[-1]
    db_target = 1.0 - 0.5 * np.sqrt(f_n)      # linear combination of basis[0] and basis[1]
    sdd21 = 10 ** (db_target / 20)             # real-valued, constant phase
    ILN, efit = get_ILN(sdd21, faxis)
    # ILN = db - efit; verify the identity holds (implementation correctness)
    db_s = 20 * np.log10(np.abs(sdd21))
    np.testing.assert_allclose(ILN, db_s - efit, atol=1e-10)


def test_column_input_handled():
    """Column vector sdd21 (squeezed) works same as row."""
    n = 10
    faxis = np.linspace(1e9, 20e9, n)
    sdd21_row = np.exp(-0.1j * faxis / 1e9)
    sdd21_col = sdd21_row.reshape(-1, 1)
    ILN_row, _ = get_ILN(sdd21_row, faxis)
    ILN_col, _ = get_ILN(sdd21_col, faxis)
    np.testing.assert_allclose(ILN_row, ILN_col)


# ---------------------------------------------------------------------------
# Adversarial conditioning fixture (added 2026-08-18).
#
# Engine defect #2 of the 208-case MATLAB correlation: faxis is in Hz, so the
# f^2 column of the fit basis reaches ~4.5e21 and cond(fmbg'fmbg) overflows.
# np.linalg.lstsq then silently truncates small singular values, solving at an
# EFFECTIVE RANK OF 2 OF 4 and discarding half the basis -- fitted-IL errors of
# 3.6-14.4 dB with no warning. MATLAB deliberately takes the raw
# normal-equations inverse of the same ill-conditioned matrix and keeps all
# four terms.
#
# test_polynomial_il_small_iln above builds an exactly-representable target but
# only asserts the identity ILN == db - efit, which holds by construction for
# ANY efit. It therefore passes at rank 2. These tests assert the fit actually
# recovers the target, which requires all four basis terms to survive.
# ---------------------------------------------------------------------------
def _exact_basis_target(n=64, fmax=50e9):
    """db(s) built from all four basis terms, so an exact fit gives ILN == 0."""
    faxis = np.linspace(1e9, fmax, n)
    a = (2.0, -3.0e-5, 1.5e-11, -4.0e-23)
    db = (a[0] + a[1] * np.sqrt(faxis) + a[2] * faxis + a[3] * faxis ** 2)
    return faxis, 10 ** (db / 20), db


def test_fit_recovers_all_four_basis_terms():
    """ILN must be ~0 when the target IS the basis -- rank 4, not rank 2."""
    faxis, sdd21, db = _exact_basis_target()
    ILN, efit = get_ILN(sdd21, faxis)
    assert np.max(np.abs(ILN)) < 1e-6, (
        'residual %.4g dB: the fit did not reproduce an exactly-representable '
        'target, so basis terms were dropped (defect #2 was rank 2 of 4)'
        % np.max(np.abs(ILN)))
    np.testing.assert_allclose(efit, db, atol=1e-6)


def test_rank_truncating_solver_would_fail_this_fixture():
    """The fixture must discriminate: lstsq on the same system must be worse.

    Without this, test_fit_recovers_all_four_basis_terms could be passing for
    the wrong reason (e.g. a well-conditioned fixture where every solver
    agrees). This pins that the ill-conditioning is real and that the choice of
    solver is what matters.
    """
    faxis, sdd21, db = _exact_basis_target()
    w = np.ones_like(faxis)
    fmbg = np.column_stack([w, np.sqrt(faxis), faxis, faxis ** 2])
    LGw = 20 * np.log10(np.abs(sdd21))
    alpha_ls, _, rank, _ = np.linalg.lstsq(fmbg, LGw, rcond=None)
    efit_ls = (alpha_ls[0] + alpha_ls[1] * np.sqrt(faxis)
               + alpha_ls[2] * faxis + alpha_ls[3] * faxis ** 2)
    assert rank < 4 or np.max(np.abs(LGw - efit_ls)) > 1e-6, (
        'lstsq solved this system at full rank %d with a good fit, so the '
        'fixture is too well conditioned to catch defect #2' % rank)


# ---------------------------------------------------------------------------
# COM Octave oracle — get_ILN run verbatim under Octave.
#
# Nominal fit, 12 points from 1 to 45 GHz. Pinned to 1e-9, not to the bit:
# cond(fmbg'*fmbg) overflows to inf, so a last-bit difference in abs() of a
# complex number (numpy and Octave disagree on 14 of 40 sampled magnitudes)
# is amplified to ~1e-12 in efit. 1e-9 is still four orders tighter than any
# solver-choice error: a rank-truncating lstsq is wrong by 3.6-14.4 dB here.
# ---------------------------------------------------------------------------
_ORACLE_F = np.array([1e9, 5e9, 9e9, 13e9, 17e9, 21e9, 25e9, 29e9, 33e9, 37e9,
                      41e9, 45e9])
_ORACLE_MAG = np.array([
    1.1034124223443182, 0.95505195038794033, 0.91419223397514238,
    0.87867716068104229, 0.85339609782545101, 0.7923632932293726,
    0.74656644815753692, 0.71495736609538985, 0.72398705823708043,
    0.71990424201689651, 0.6717389736367565, 0.62391912571182562])


def test_octave_oracle_nominal_fit():
    sdd21 = _ORACLE_MAG * np.exp(-1j * _ORACLE_F / 1e10)
    ILN, efit = get_ILN(sdd21, _ORACLE_F)
    np.testing.assert_allclose(efit, [
        0.82261526946545516, -0.24031849626559204, -0.81541358993353297,
        -1.2471528352777577, -1.6128130934891733, -1.9452365038520267,
        -2.2620323149769481, -2.5739452112305679, -2.8881035687239329,
        -3.2095238657359824, -3.5418949212370689, -3.8880232377563759],
        rtol=0, atol=1e-9)
    np.testing.assert_allclose(ILN, [
        0.032142110266793367, -0.15914158717620452, 0.036164143043359687,
        0.12373959525691092, 0.23582614760283427, -0.076276529266491089,
        -0.27659832166267861, -0.34045188901525369, 0.082719627847610333,
        0.35501851854486821, 0.085905848632915394, -0.209410786972553],
        rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# COM Octave oracle — an exactly singular normal matrix.
#
# MATLAB's inv() does not raise there: it warns and returns Inf everywhere,
# and the Inf (or the NaN that Inf*0 makes of it) comes out in efit and ILN.
#
#   get_ILN(s(1), f(1))            Octave: efit  Inf,  ILN -Inf
#   get_ILN(zeros(1,6), f(1:6))    Octave: efit  NaN,  ILN  NaN
#     warning: inverse: matrix singular to machine precision, rcond = 0
#
# The port used to catch LinAlgError and re-solve with np.linalg.lstsq, which
# answered the first with efit 1.0662767 and the second with efit 0 — finite
# numbers the reference never produces, on inputs where it is telling the
# caller the fit does not exist.
# ---------------------------------------------------------------------------
def test_octave_oracle_single_point_is_infinite():
    sdd21 = _ORACLE_MAG[:1] * np.exp(-1j * _ORACLE_F[:1] / 1e10)
    ILN, efit = get_ILN(sdd21, _ORACLE_F[:1])
    assert efit[0] == np.inf
    assert ILN[0] == -np.inf


def test_octave_oracle_all_zero_sdd21_is_nan():
    ILN, efit = get_ILN(np.zeros(6, dtype=complex), _ORACLE_F[:6])
    assert np.all(np.isnan(efit))
    assert np.all(np.isnan(ILN))
