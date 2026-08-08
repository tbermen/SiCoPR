"""Verification tests for Burst_Probability_Calc().

# ============================================================
# MATLAB GROUND TRUTH (lines 1083-1132)
# Computes probability of error bursts of lengths 1..min(ndfe,nburst).
# p_burst = cumprod(p_error_propagation).
# p_error_propagation[k] = P(noise > error_threshold) under burst k.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.Burst_Probability_Calc.py_impl import Burst_Probability_Calc


def _gaussian_pdf(BinSize=0.01, sigma=0.05, N=200):
    x = np.arange(-N, N + 1) * BinSize
    y = np.exp(-0.5 * (x / sigma) ** 2)
    y /= y.sum()
    return SimpleNamespace(BinSize=BinSize, Min=-N, x=x, y=y)


def _snr_struct():
    return SimpleNamespace(
        A_s=1.0,
        combined_interference_and_noise_pdf=_gaussian_pdf(),
    )


def _param(ndfe=3):
    return SimpleNamespace(
        pass_threshold=3.0,
        ndfe=ndfe,
        levels=4,
        delta_y=0.01,
    )


def _op(nburst=3, simple=True):
    return SimpleNamespace(
        COM_EP_margin=0.0,
        nburst=nburst,
        use_simple_EP_model=simple,
    )


def test_returns_two_outputs():
    """Returns (p_burst, p_error_propagation)."""
    result = Burst_Probability_Calc(_snr_struct(), [0.3, 0.2, 0.1], _param(), _op())
    assert len(result) == 2


def test_p_burst_length():
    """p_burst has length min(ndfe, nburst)."""
    p_burst, p_ep = Burst_Probability_Calc(_snr_struct(), [0.3, 0.2, 0.1], _param(3), _op(3))
    assert len(p_burst) == len(p_ep)
    assert len(p_burst) == min(3, 3)


def test_p_burst_is_cumprod():
    """p_burst = cumprod(p_error_propagation)."""
    p_burst, p_ep = Burst_Probability_Calc(_snr_struct(), [0.3, 0.2, 0.1], _param(), _op())
    np.testing.assert_allclose(p_burst, np.cumprod(p_ep))


def test_p_values_in_zero_one():
    """All probabilities are in [0, 1]."""
    p_burst, _ = Burst_Probability_Calc(_snr_struct(), [0.3, 0.2, 0.1], _param(), _op())
    assert np.all(p_burst >= 0) and np.all(p_burst <= 1)


def test_no_dfe_taps_returns_length_one():
    """With ndfe=1 and nburst=1, result has length 1."""
    p_burst, p_ep = Burst_Probability_Calc(_snr_struct(), [], _param(ndfe=1), _op(nburst=1))
    assert len(p_burst) == 1
    assert len(p_ep) == 1
