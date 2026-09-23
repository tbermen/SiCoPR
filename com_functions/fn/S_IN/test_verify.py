"""Verification tests for S_IN().

# ============================================================
# MATLAB GROUND TRUTH
# S_IN_of_f = Ns_of_f / 2 * |Hn21 * H_CTF * H_R|^2
# With Hn21 = 0 → S_IN = 0
# With sigma_ns=0 (→ tiny) and clause_178: Ns ≈ 0 → S_IN ≈ 0
# At DC (f=0): H_CTF(0) = 1 (G_DC=0,G_DC2=0); H_R(0)=1; Ns_178(0)=flat value
#   S_IN(0) = Ns(0)/2 * |Hn21(0)|^2
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.S_IN.py_impl import S_IN


def make_param(fb=100e9, f_r=0.75, sigma_ns=0.01, f_hp=10e9):
    return SimpleNamespace(
        fb=fb, f_r=f_r, sigma_ns=sigma_ns, f_hp=f_hp,
        CTLE_fp1=np.array([15e9]),
        CTLE_fz=np.array([6e9]),
        CTLE_fp2=np.array([30e9]),
        f_HP=np.array([0.01]),
    )


def make_op(rit='clause_178'):
    return SimpleNamespace(RIT_REF_PTR=rit)


def test_zero_noise_path():
    """Hn21 = 0 → S_IN = 0."""
    f = np.linspace(0, 50e9, 10)
    out = S_IN(f, np.zeros(10), 0.0, 0.0, make_param(), make_op())
    np.testing.assert_allclose(out, 0.0)


def test_output_length():
    f = np.linspace(1e9, 50e9, 12)
    out = S_IN(f, np.ones(12), 0.0, 0.0, make_param(), make_op())
    assert len(out) == 12


def test_non_negative():
    """S_IN is real and non-negative."""
    f = np.linspace(1e9, 50e9, 15)
    out = S_IN(f, np.ones(15), 0.0, 0.0, make_param(), make_op())
    assert np.all(out >= 0)


def test_scales_quadratically_with_noise_path():
    """Doubling |Hn21| quadruples S_IN."""
    f = np.array([10e9, 20e9])
    p = make_param()
    op = make_op()
    out1 = S_IN(f, np.ones(2), 0.0, 0.0, p, op)
    out2 = S_IN(f, 2.0 * np.ones(2), 0.0, 0.0, p, op)
    np.testing.assert_allclose(out2, 4 * out1, rtol=1e-10)


# ============================================================
# COM Octave oracle — S_IN and N_s extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m (both byte-identical to
# matlab/com_ieee8023_4p16p0.m) and run by tools/octave_oracle.py.
#
# param.fb=106.25e9, f_r=0.75, sigma_ns=0.01, f_hp=0.6875e9,
# CTLE_fp1=21.25e9, CTLE_fz=10.625e9, CTLE_fp2=42.5e9, f_HP=0.6875e9,
# G_DC=-12, G_DC2=0, Hn21=1, on
# f = [0 1e9 10e9 26.5625e9 53.125e9 60e9 106.25e9].
#
# DIVERGENCE FIXED — the inlined _N_s had not picked up the corrections made
# in com_functions/fn/N_s/py_impl.py, so it still did `np.zeros(len(f))` on an
# un-raveled f.  With a COLUMN f and clause_179 the port raised
#   ValueError: could not broadcast input array from shape (5,1) into shape (5,)
# on a call the reference answers.  MATLAB's Ns_of_f is always the row
# zeros(1,length(f)) written through linear indexing, so a column f is
# answered: S_IN then broadcasts the 1xN Ns against the Nx1 |H|^2 and returns
# an N x N matrix (OCT_COL_179 below).  clause_178 already worked only because
# its right-hand side is a scalar.
#
# Matrix f is refused by both, and now at the same place: N_s grows to 1x5 for
# a 2x3 f (length() is 3 but find() walks all six elements in column-major
# order and the last f<=fb/2 is the fifth), and the final `.*` then fails --
#   Octave: "product: nonconformant arguments (op1 is 1x5, op2 is 2x3)"
#   Python: "operands could not be broadcast together with shapes (5,) (2,3)"
# Before the fix Python failed earlier, inside _N_s.
#
# Also confirmed:
#   * `sigma_ns = param.sigma_ns + eps(0)` is the smallest denormal,
#     5e-324, so sigma_ns=0 still gives an all-zero S_IN, not a NaN.
#   * f <= fb/2 is inclusive: f=53.125e9 is kept, 60e9 and 106.25e9 give 0.
#   * S_IN reads param.f_HP while N_s reads param.f_hp.  MATLAB struct fields
#     are case sensitive and these are two different parameters.
#   * OP.RIT_REF_PTR is lowercased, so 'Clause_179' is accepted.
#   * annex_176d and clause_179 give identical results.
#   * f<0 is not rejected; clause_179 squares f/f_hp so it is answered.
#
# Tolerance: Octave's and numpy's complex `*` and `/` are each not correctly
# rounded, which puts a ~1 ULP floor under every element; measured worst
# disagreement over these probes was 9.7e-16 relative, so these are pinned at
# rtol=1e-13.
# ============================================================

def oct_param(f_HP=0.6875e9, f_hp=0.6875e9, sigma_ns=0.01):
    return SimpleNamespace(
        fb=106.25e9, f_r=0.75, sigma_ns=sigma_ns, f_hp=f_hp,
        CTLE_fp1=np.array([21.25e9]),
        CTLE_fz=np.array([10.625e9]),
        CTLE_fp2=np.array([42.5e9]),
        f_HP=np.array([f_HP]),
    )


OCT_F = np.array([0.0, 1e9, 10e9, 26.5625e9, 53.125e9, 60e9, 106.25e9])
OCT_RTOL = 1e-13

OCT_178 = np.array([5.9384220656959376e-17, 6.7534256753067836e-17,
                    6.9281405713445731e-16, 1.6671449230222401e-15,
                    1.222026332655106e-15, 0, 0])
OCT_179 = np.array([0, 4.6802364136148359e-17, 7.0374262775454243e-16,
                    1.7003078931068348e-15, 1.2469610522635901e-15, 0, 0])
OCT_CPLX = np.array([1.4846055164239844e-17, 1.6883564188266959e-17,
                     1.7320351428361445e-16, 4.1678623075556003e-16,
                     3.0550658316377659e-16, 0, 0])
OCT_GDC2 = np.array([2.976261327217299e-17, 5.6722265024296521e-17,
                     6.9118831458685235e-16, 1.6665882148589016e-15,
                     1.2219242638768606e-15, 0, 0])
# G_DC2=-6, clause_179, f_HP=3.3e9 (distinct from f_hp=0.6875e9) and f_HP=f_hp
OCT_FHP_DISTINCT = np.array([0, 1.4703753192441413e-17, 6.5199126658888719e-16,
                             1.6809553473554456e-15, 1.2433719753922502e-15, 0, 0])
OCT_FHP_SAME = np.array([0, 3.5554132950746796e-17, 7.0126358622787859e-16,
                         1.6994555456517299e-15, 1.2468047014651829e-15, 0, 0])
# column f = [0 25e9 53.125e9]', so the 1xN Ns broadcasts against the Nx1 |H|^2
OCT_COL_F = np.array([0.0, 25e9, 53.125e9]).reshape(-1, 1)
OCT_COL_179 = np.array([[0, 6.0560269856533348e-17, 6.0595920316517813e-17],
                        [0, 1.6746250417066064e-15, 1.6756108555608009e-15],
                        [0, 1.2462274263880576e-15, 1.2469610522635901e-15]])
OCT_COL_178 = np.array([[5.9384220656959376e-17] * 3,
                        [1.6421046872803268e-15] * 3,
                        [1.222026332655106e-15] * 3])


def test_oracle_clause_178():
    out = S_IN(OCT_F, np.ones(7), -12.0, 0.0, oct_param(), make_op('clause_178'))
    np.testing.assert_allclose(out, OCT_178, rtol=OCT_RTOL, atol=0)


def test_oracle_clause_179():
    out = S_IN(OCT_F, np.ones(7), -12.0, 0.0, oct_param(), make_op('clause_179'))
    np.testing.assert_allclose(out, OCT_179, rtol=OCT_RTOL, atol=0)


def test_annex_176d_equals_clause_179_and_case_is_folded():
    """N_s shares one branch, and lower(OP.RIT_REF_PTR) accepts mixed case."""
    for rit in ('annex_176d', 'Annex_176D', 'Clause_179', 'CLAUSE_179'):
        out = S_IN(OCT_F, np.ones(7), -12.0, 0.0, oct_param(), make_op(rit))
        np.testing.assert_allclose(out, OCT_179, rtol=OCT_RTOL, atol=0)


def test_oracle_complex_noise_path_and_second_stage_gain():
    out = S_IN(OCT_F, (0.3 - 0.4j) * np.ones(7), -12.0, 0.0,
               oct_param(), make_op('clause_178'))
    np.testing.assert_allclose(out, OCT_CPLX, rtol=OCT_RTOL, atol=0)
    out = S_IN(OCT_F, np.ones(7), -12.0, -3.0, oct_param(), make_op('clause_178'))
    np.testing.assert_allclose(out, OCT_GDC2, rtol=OCT_RTOL, atol=0)


def test_column_f_returns_the_outer_product_not_a_raise():
    """MATLAB's 1xN Ns against an Nx1 |H|^2 broadcasts to N x N.

    clause_179 raised here before the inlined _N_s was brought back in step
    with com_functions/fn/N_s/py_impl.py.
    """
    out = S_IN(OCT_COL_F, np.ones((3, 1)), -12.0, 0.0,
               oct_param(), make_op('clause_179'))
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out, OCT_COL_179, rtol=OCT_RTOL, atol=0)
    out = S_IN(OCT_COL_F, np.ones((3, 1)), -12.0, 0.0,
               oct_param(), make_op('clause_178'))
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out, OCT_COL_178, rtol=OCT_RTOL, atol=0)


def test_matrix_f_is_refused_at_the_final_product():
    """N_s grows to 1x5 for a 2x3 f, so the last `.*` is nonconformant."""
    f = np.array([[0.0, 1e9, 10e9], [20e9, 60e9, 106.25e9]])
    for rit in ('clause_178', 'clause_179'):
        with pytest.raises(ValueError) as exc:
            S_IN(f, np.ones((2, 3)), -12.0, 0.0, oct_param(), make_op(rit))
        assert 'broadcast' in str(exc.value)
        assert '(5,)' in str(exc.value)       # the grown Ns, as in Octave's 1x5


def test_reads_f_HP_not_f_hp():
    """S_IN's CTLE LF pole/zero come from param.f_HP; N_s uses param.f_hp."""
    out = S_IN(OCT_F, np.ones(7), -12.0, -6.0,
               oct_param(f_HP=3.3e9, f_hp=0.6875e9), make_op('clause_179'))
    np.testing.assert_allclose(out, OCT_FHP_DISTINCT, rtol=OCT_RTOL, atol=0)
    same = S_IN(OCT_F, np.ones(7), -12.0, -6.0,
                oct_param(f_HP=0.6875e9, f_hp=0.6875e9), make_op('clause_179'))
    np.testing.assert_allclose(same, OCT_FHP_SAME, rtol=OCT_RTOL, atol=0)
    # the two fields are distinct (atol=0: these values are all ~1e-16)
    assert not np.allclose(out, same, rtol=1e-6, atol=0)


def test_band_edge_is_inclusive():
    """f <= fb/2 keeps f = fb/2 exactly and zeroes everything above it."""
    out = S_IN(OCT_F, np.ones(7), -12.0, 0.0, oct_param(), make_op('clause_178'))
    assert out[4] > 0        # f == fb/2
    assert out[5] == 0.0 and out[6] == 0.0


def test_sigma_ns_zero_is_zero_not_nan():
    """`param.sigma_ns + eps(0)` is 5e-324, so the whole PSD is exactly 0."""
    out = S_IN(OCT_F, np.ones(7), -12.0, 0.0,
               oct_param(sigma_ns=0.0), make_op('clause_178'))
    np.testing.assert_array_equal(out, np.zeros(7))


def test_f_hp_non_positive_only_rejected_by_the_1_f_branches():
    """clause_179/annex_176d error; clause_178 never looks at f_hp."""
    for rit in ('clause_179', 'annex_176d'):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match='f_hp'):
                S_IN(OCT_F, np.ones(7), -12.0, 0.0,
                     oct_param(f_hp=bad), make_op(rit))
    out = S_IN(OCT_F, np.ones(7), -12.0, 0.0,
               oct_param(f_hp=0.0), make_op('clause_178'))
    np.testing.assert_allclose(out, OCT_178, rtol=OCT_RTOL, atol=0)


def test_unknown_rit_pointer_raises():
    with pytest.raises(ValueError, match='RIT Reference Pointer'):
        S_IN(OCT_F, np.ones(7), -12.0, 0.0, oct_param(), make_op('clause_999'))
