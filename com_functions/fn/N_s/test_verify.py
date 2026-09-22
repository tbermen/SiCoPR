"""Verification tests for N_s().

# ============================================================
# MATLAB GROUND TRUTH
# clause_178: Ns_of_f(1:inq) = 2*sigma_ns^2 / f_b  (flat spectrum)
#   f_b=100e9, sigma_ns=0.01, f=[0,25e9,50e9,75e9,100e9]
#   inq = find(f <= 50e9, 1,'last') = 3 (1-based, elements 0..50e9)
#   Ns[:3] = 2*0.01^2/100e9 = 2e-12
#   Ns[3:] = 0
#
# clause_179:
#   beta = 1 - (2*f_hp/f_b)*atan(f_b/(2*f_hp))
#   Ns[:inq] = (2*sigma_ns^2)/(beta*f_b) * (f[:inq]/f_hp)^2 / (1+(f[:inq]/f_hp)^2)
#   At f=0: Ns=0 (DC is zero)
#
# unsupported RIT → ValueError
# f_hp<=0 with clause_179 → ValueError
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.N_s.py_impl import N_s


def make_op(rit):
    return SimpleNamespace(RIT_REF_PTR=rit)


def make_param(fb=100e9, f_hp=10e9):
    return SimpleNamespace(fb=fb, f_hp=f_hp)


def test_clause_178_flat():
    """clause_178: Ns is flat (constant) up to fb/2."""
    f = np.array([0.0, 25e9, 50e9, 75e9, 100e9])
    p = make_param(fb=100e9)
    op = make_op('clause_178')
    out = N_s(f, p, 0.01, op)
    expected_val = 2 * 0.01**2 / 100e9
    np.testing.assert_allclose(out[:3], expected_val)
    assert out[3] == 0.0
    assert out[4] == 0.0


def test_clause_178_output_length():
    f = np.linspace(0, 100e9, 20)
    out = N_s(f, make_param(), 0.005, make_op('clause_178'))
    assert len(out) == 20


def test_clause_179_dc_is_zero():
    """clause_179: Ns(0) = 0 because f[0]=0."""
    f = np.array([0.0, 10e9, 50e9])
    out = N_s(f, make_param(fb=100e9, f_hp=10e9), 0.01, make_op('clause_179'))
    assert out[0] == pytest.approx(0.0)


def test_clause_179_positive():
    """clause_179: all in-band values ≥ 0."""
    f = np.linspace(0, 50e9, 10)
    out = N_s(f, make_param(fb=100e9, f_hp=5e9), 0.01, make_op('clause_179'))
    assert np.all(out >= 0)


def test_annex_176d_equals_clause_179():
    """annex_176d uses same formula as clause_179."""
    f = np.linspace(1e9, 50e9, 5)
    p = make_param(fb=100e9, f_hp=10e9)
    out1 = N_s(f, p, 0.01, make_op('clause_179'))
    out2 = N_s(f, p, 0.01, make_op('annex_176d'))
    np.testing.assert_allclose(out1, out2)


def test_unsupported_rit_raises():
    with pytest.raises(ValueError):
        N_s(np.array([1e9]), make_param(), 0.01, make_op('bad_clause'))


def test_clause_179_fhp_zero_raises():
    with pytest.raises((ValueError, ZeroDivisionError)):
        N_s(np.array([1e9]), make_param(f_hp=0), 0.01, make_op('clause_179'))


# ============================================================
# COM Octave oracle — N_s extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m and run by tools/octave_oracle.py.
# param.fb=100e9, param.f_hp=10e9, sigma_ns=0.01, f=[0 25 50 75 100]e9:
#
#   clause_178 -> [2.0000000000000002e-15 x3, 0, 0]
#   clause_179 -> [0, 2.3770725964419546e-15, 2.6513502037237188e-15, 0, 0]
#   annex_176d -> the same as clause_179, and 'Clause_179' (mixed case) too
#   COLUMN f   -> the SAME 1x5 rows: Ns_of_f is always zeros(1,length(f)) and is
#                 written by linear indexing, so a column f is answered, not
#                 refused ("could not broadcast (3,1) into (3,)")
#   f=[60 70]e9 (nothing <= fb/2, inq is []) -> [0 0], no error
#   f=[] -> []
#   f=[0 60 10 70]e9 -> [2e-15 2e-15 2e-15 0]: find(...,1,'last') carries the
#                 out-of-band 60e9 along with it
#   f_hp=0 with clause_178 -> fine (that branch never looks at f_hp)
#   f_hp=0 with clause_179 -> error "Parameter f_hp must be greater than 0..."
#   RIT_REF_PTR='clause_162' -> error 'unsuported RIT Reference Pointer'
#   sigma_ns=-0.01 -> same as +0.01 (it is squared)
#
#   f as a 2x3 matrix [0 40e9 60e9; 10e9 45e9 70e9] -> a 1x4 result: length()
#   counts only the longest dimension (3) while find() walks all six elements
#   in column-major order and inq=4 GROWS the array:
#       clause_178 -> [2.0000000000000002e-15] * 4
#       clause_179 -> [0, 1.3787021059363338e-15, 2.595203964115452e-15,
#                      2.6276440136668952e-15]
# ============================================================

OCT_F = np.array([0.0, 25e9, 50e9, 75e9, 100e9])
OCT_178 = [2.0000000000000002e-15, 2.0000000000000002e-15,
           2.0000000000000002e-15, 0.0, 0.0]
OCT_179 = [0.0, 2.3770725964419546e-15, 2.6513502037237188e-15, 0.0, 0.0]


def test_oracle_clause_178():
    out = N_s(OCT_F, make_param(), 0.01, make_op('clause_178'))
    np.testing.assert_allclose(out, OCT_178, rtol=1e-14, atol=0)


def test_oracle_clause_179():
    out = N_s(OCT_F, make_param(), 0.01, make_op('clause_179'))
    np.testing.assert_allclose(out, OCT_179, rtol=1e-14, atol=0)


def test_oracle_annex_176d_and_mixed_case():
    p = make_param()
    np.testing.assert_allclose(N_s(OCT_F, p, 0.01, make_op('annex_176d')),
                               OCT_179, rtol=1e-14, atol=0)
    np.testing.assert_allclose(N_s(OCT_F, p, 0.01, make_op('Clause_179')),
                               OCT_179, rtol=1e-14, atol=0)


def test_column_f_is_answered():
    """MATLAB writes into a row by linear indexing, so a column f is legal."""
    col = OCT_F.reshape(-1, 1)
    np.testing.assert_allclose(N_s(col, make_param(), 0.01, make_op('clause_179')),
                               OCT_179, rtol=1e-14, atol=0)
    np.testing.assert_allclose(N_s(col, make_param(), 0.01, make_op('clause_178')),
                               OCT_178, rtol=1e-14, atol=0)


def test_matrix_f_grows_to_the_last_in_band_index():
    """length() counts the longest dimension; find() walks every element."""
    f = np.array([[0.0, 40e9, 60e9], [10e9, 45e9, 70e9]])
    out = N_s(f, make_param(), 0.01, make_op('clause_178'))
    assert out.size == 4
    np.testing.assert_allclose(out, [2.0000000000000002e-15] * 4, rtol=1e-14, atol=0)
    out = N_s(f, make_param(), 0.01, make_op('clause_179'))
    np.testing.assert_allclose(out, [0.0, 1.3787021059363338e-15,
                                     2.595203964115452e-15,
                                     2.6276440136668952e-15], rtol=1e-14, atol=0)


def test_nothing_in_band_is_all_zeros():
    f = np.array([60e9, 70e9])
    for rit in ('clause_178', 'clause_179'):
        np.testing.assert_array_equal(N_s(f, make_param(), 0.01, make_op(rit)),
                                      np.zeros(2))


def test_unsorted_f_carries_out_of_band_entries():
    """find(...,1,'last') stretches the fill over the 60e9 entry."""
    out = N_s(np.array([0.0, 60e9, 10e9, 70e9]), make_param(), 0.01,
              make_op('clause_178'))
    np.testing.assert_allclose(out, [2.0000000000000002e-15] * 3 + [0.0],
                               rtol=1e-14, atol=0)


def test_fhp_zero_is_only_checked_for_the_highpass_clauses():
    out = N_s(OCT_F, make_param(f_hp=0), 0.01, make_op('clause_178'))
    np.testing.assert_allclose(out, OCT_178, rtol=1e-14, atol=0)


def test_negative_sigma_is_squared():
    np.testing.assert_allclose(N_s(OCT_F, make_param(), -0.01, make_op('clause_178')),
                               OCT_178, rtol=1e-14, atol=0)
