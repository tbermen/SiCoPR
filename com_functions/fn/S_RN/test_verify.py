"""Verification tests for S_RN().

# ============================================================
# MATLAB GROUND TRUTH
# S_RN_of_f = eta_0/2 * |H_CTF * H_R|^2
# H_CTF = (10^(G_DC/20) + j*f/z1) * (10^(G_DC2/20) + j*f/zlf)
#          / ((1+j*f/p1) * (1+j*f/p2) * (1+j*f/plf))
# H_R = 1 / polyval([1,2.613126,3.414214,2.613126,1], j*f/(f_r*fb))
#
# G_DC=0, G_DC2=0: H_CTF numerator at DC = (1 + 0) * (1 + 0) = 1
#                  H_CTF denominator at DC = 1 * 1 * 1 = 1  → H_CTF(0)=1
# H_R(0) = 1/polyval(...,0) = 1/1 = 1
# S_RN(0) = eta_0/2 * 1 = eta_0/2
#
# f → ∞: H_CTF and H_R fall off → S_RN decreases
# Output length == len(f)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.S_RN.py_impl import S_RN


def make_param(fb=106.25e9, f_r=0.75, eta_0=1e-14,
               fp1=15e9, fz=6e9, fp2=30e9, fHP=0.01):
    return SimpleNamespace(
        fb=fb, f_r=f_r, eta_0=eta_0,
        CTLE_fp1=np.array([fp1]),
        CTLE_fz=np.array([fz]),
        CTLE_fp2=np.array([fp2]),
        f_HP=np.array([fHP]),
    )


def test_dc_value():
    """At f=0, G_DC=0, G_DC2=0: S_RN(0) = eta_0/2."""
    f = np.array([0.0])
    p = make_param(eta_0=2.0, fHP=1.0)
    out = S_RN(f, 0.0, 0.0, p)
    assert float(out[0]) == pytest.approx(1.0, rel=1e-6)


def test_output_length():
    f = np.linspace(0, 50e9, 15)
    out = S_RN(f, 0.0, 0.0, make_param())
    assert len(out) == 15


def test_non_negative():
    """S_RN is real and non-negative everywhere."""
    f = np.linspace(0, 100e9, 20)
    out = S_RN(f, 0.0, 0.0, make_param())
    assert np.all(np.isreal(out))
    assert np.all(out >= 0)


def test_positive_gdc_increases_output():
    """Positive G_DC amplifies numerator → S_RN increases at DC."""
    f = np.array([0.0])
    p = make_param(eta_0=1.0, fHP=1.0)
    out0 = float(S_RN(f, 0.0, 0.0, p)[0])
    out10 = float(S_RN(f, 10.0, 0.0, p)[0])
    assert out10 > out0


# ============================================================
# COM Octave oracle — S_RN extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m and run by tools/octave_oracle.py.
# param.CTLE_fp1=[15e9 20e9], CTLE_fz=[6e9 7e9], CTLE_fp2=[30e9], f_HP=[0.66e9],
# fb=106.25e9, f_r=0.75, eta_0=8.2e-9  (only the first element of each list is
# read), f=[0 1e9 20e9 53.125e9 120e9]:
#
#   S_RN(f, -5, 0, param) -> [1.2965338406690357e-09, 1.4026234506316369e-09,
#                             1.1676798710179013e-08, 5.5451278033857772e-09,
#                             5.4122286147607244e-11]
#   f=[-20e9 -1e9 0], G_DC=-5, G_DC2=-3 -> [1.1670462690110094e-08,
#                             1.1903317201696051e-09, 6.4980620890905654e-10]
#   f=0, G_DC=G_DC2=0 -> eta_0/2 = 4.1000000000000003e-09
#   f 2x2 -> a 2x2 result;  f=[] -> []
#   f_HP=0 -> all NaN
#   G_DC=[-5 -6] (or G_DC2=[0 1]), whatever length(f) is -> error "for x^y, only
#       square matrix arguments are permitted and one argument must be scalar.
#       Use .^ for elementwise power."  MATLAB writes 10^(G_DC/20) with the
#       MATRIX power, so a vector gain is refused; numpy's ** broadcast it.
#   G_DC=[-5] (1x1) is still a scalar and is accepted.
# ============================================================

OCT_F = np.array([0.0, 1e9, 20e9, 53.125e9, 120e9])


def oct_param():
    return make_param(fb=106.25e9, f_r=0.75, eta_0=8.2e-9,
                      fp1=15e9, fz=6e9, fp2=30e9, fHP=0.66e9)


def test_oracle_values():
    out = S_RN(OCT_F, -5.0, 0.0, oct_param())
    np.testing.assert_allclose(
        out, [1.2965338406690357e-09, 1.4026234506316369e-09,
              1.1676798710179013e-08, 5.5451278033857772e-09,
              5.4122286147607244e-11], rtol=1e-13, atol=0)


def test_oracle_negative_frequencies():
    out = S_RN(np.array([-20e9, -1e9, 0.0]), -5.0, -3.0, oct_param())
    np.testing.assert_allclose(
        out, [1.1670462690110094e-08, 1.1903317201696051e-09,
              6.4980620890905654e-10], rtol=1e-13, atol=0)


def test_oracle_dc_is_eta0_over_2():
    out = S_RN(np.array([0.0]), 0.0, 0.0, oct_param())
    assert float(out[0]) == pytest.approx(4.1000000000000003e-09, rel=1e-13)


def test_vector_gain_is_a_matrix_power_error():
    """10^(G_DC/20) is MATLAB's ^, so a non-scalar gain is refused there."""
    p = oct_param()
    with pytest.raises(ValueError):
        S_RN(OCT_F, np.array([-5.0, -6.0]), 0.0, p)
    with pytest.raises(ValueError):
        S_RN(OCT_F, -5.0, np.array([0.0, 1.0]), p)
    # length(f) == length(G_DC) is refused just the same
    with pytest.raises(ValueError):
        S_RN(np.array([0.0, 20e9]), np.array([-5.0, -6.0]), 0.0, p)


def test_one_by_one_gain_is_still_a_scalar():
    out = S_RN(np.array([0.0, 20e9]), np.array([-5.0]), 0.0, oct_param())
    np.testing.assert_allclose(out, [1.2965338406690357e-09,
                                     1.1676798710179013e-08], rtol=1e-13, atol=0)


def test_zero_f_HP_is_all_nan():
    out = S_RN(OCT_F, -5.0, 0.0, make_param(fb=106.25e9, f_r=0.75, eta_0=8.2e-9,
                                            fp1=15e9, fz=6e9, fp2=30e9, fHP=0.0))
    assert np.all(np.isnan(out))


def test_matrix_f_keeps_its_shape():
    out = S_RN(np.array([[0.0, 1e9], [20e9, 53.125e9]]), -5.0, 0.0, oct_param())
    assert out.shape == (2, 2)
    np.testing.assert_allclose(
        out, [[1.2965338406690357e-09, 1.4026234506316369e-09],
              [1.1676798710179013e-08, 5.5451278033857772e-09]],
        rtol=1e-13, atol=0)


def test_empty_f():
    assert S_RN(np.array([]), -5.0, 0.0, oct_param()).size == 0
