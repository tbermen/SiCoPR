"""Verification tests for get_CICP_fit_residual (new in 4p17p0).

Its values against COM Octave are pinned through get_ACBW's test, which runs it
on every candidate window of the sweep. Here: exact recovery of the basis on a
well-conditioned fit, the mask cleaning, and the reference's error.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)

from com_functions.fn.get_CICP_fit_residual.py_impl import get_CICP_fit_residual  # noqa: E402


def test_recovers_the_com_basis_and_extrapolates():
    f = np.linspace(0.5, 4.0, 60)
    a = np.array([1.5, -0.7, 0.3, 0.02])
    y = a[0] + a[1] * np.sqrt(f) + a[2] * f + a[3] * f ** 2
    res, efit, alpha, used = get_CICP_fit_residual(y, f, f <= 2.0)
    np.testing.assert_allclose(alpha, a, rtol=1e-9)
    np.testing.assert_allclose(res, 0.0, atol=1e-10)      # extrapolated past the window too
    assert used.sum() == np.count_nonzero(f <= 2.0)


def test_mask_drops_nonfinite_and_negative_frequency():
    f = np.linspace(-1.0, 4.0, 40)
    y = 1 + f.copy()
    y[20] = np.nan
    with np.errstate(invalid='ignore'):          # sqrt(f<0) in the extrapolation, as in MATLAB
        _r, _e, _a, used = get_CICP_fit_residual(y, f, np.ones(40, bool))
    assert not used[f < 0].any() and not used[20]


def test_fewer_than_four_samples_is_an_error():
    f = np.linspace(1, 2, 10)
    with pytest.raises(ValueError, match='at least 4'):
        get_CICP_fit_residual(f, f, f < 1.25)


def test_backslash_falls_back_to_lu_and_stops_when_singular():
    """MATLAB's A\\b on a symmetric matrix with a positive diagonal tries Cholesky
    first and falls back to LU. Here Cholesky fails (the leading 2x2 block is
    indefinite) and LU meets an exactly singular matrix (row 3 = row 1 + row 2),
    where MATLAB returns Inf. A least-squares substitute would return a finite
    answer the reference never gives, so the port stops and names the solve.

    (A PSD singular matrix, e.g. a fit window at one frequency, does NOT reach
    here: Cholesky completes on a rounding-sized pivot and returns finite
    numbers, as MATLAB's Cholesky does, with only a warning.)"""
    from com_functions.fn.get_CICP_fit_residual.py_impl import _backslash_sym
    A = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [3.0, 3.0, 6.0]])
    with pytest.raises(ValueError, match='singular'):
        _backslash_sym(A, np.array([1.0, 0.0, 1.0]))
    # and a non-singular indefinite one is solved by the LU fallback
    B = np.array([[1.0, 2.0], [2.0, 1.0]])
    np.testing.assert_allclose(_backslash_sym(B, np.array([1.0, 0.0])), [-1 / 3, 2 / 3], rtol=1e-15)
