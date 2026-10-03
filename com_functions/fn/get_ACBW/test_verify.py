"""Verification tests for get_ACBW (new in 4p17p0).

COM Octave: get_ACBW, get_CICP_fit_sweep, get_CICP_fit_residual and
get_BW_from_CICP_residual extracted verbatim from
octave/com_ieee8023_4p17p0_octave_compat.m by tools/octave_oracle.py and run on
_channel() below, called as FD_Processing calls it (Hch = sdd21f, fGHz =
faxis/1e9, the reference's own defaults, T_dev from the keyword). Generator:
runs/4p17p0_oracles/gen_acbw_oracle.py (local).

Residual, named and bounded: the CICP fit solves 4x4 normal equations in GHz
powers up to f^2 (cond ~1e10), so last-bit differences in forming fmbg'*fmbg,
which depend on the BLAS, come out ~1e-11 relative in alpha and ~1e-9 dB in the
fit. MATLAB and Octave would differ from each other by as much; the reference
turns the near-singular warning off. Every decision is exact: the fit window
chosen, CICP_db, and the bandwidth in both branches.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)

from com_functions.fn.get_ACBW.py_impl import get_ACBW  # noqa: E402


def _channel(n=2001):
    f = np.linspace(10e6, 100e9, n)
    g = f / 1e9
    loss_db = 0.5 + 1.1 * np.sqrt(g) + 0.25 * g + 0.004 * g ** 2 + 6.0 / (1 + np.exp(-(g - 55) / 4))
    ripple = 0.3 * np.sin(2 * np.pi * g / 7.3)
    mag = 10 ** (-(loss_db + ripple) / 20)
    return f, mag * np.exp(1j * (-2 * np.pi * f * 1.2e-9))


# COM Octave, get_ACBW on _channel() (identical for T_dev 1 and 0.05 except Bch)
_IDX = [0, 1, 100, 500, 1000, 1500, 2000]
_OCT_ALPHA = [-206.53225435117176, 26.83320986213381, -2.978394956247432, 0.027137671266326113]
_OCT_CICP_DB = [-201.99942931165026, -195.79201464910574, -157.1831136495148, -129.77908389229222,
                -98.4542998220295, -49.19789507125925, 0.0]
_OCT_RESIDUAL = [1.8792852891034784, 4.346337466814049, 3.5305328315918416, 0.07798692364056592,
                 -0.5940866439255359, -4.324437748388775, -35.33706130868424]
_OCT_DIFF = [0.0, 0.0, -0.0, 0.07854393350874707, -0.06994793554696055, -0.6852148182765426,
             -0.8668211987758241]
_OCT_FIT_EVERY_400 = [-203.87871460075374, -135.2344657721587, -112.53177405850322,
                      -79.68314419846556, -31.113502122724697, 35.33706130868424]
_OCT_BCH = {1.0: 100.0, 0.05: 21.80782}


def _run(T_dev, **extra):
    f, H = _channel()
    return get_ACBW(H, f / 1e9, SimpleNamespace(DISPLAY_WINDOW=0), SimpleNamespace(T_dev=T_dev, **extra))


def test_bandwidth_matches_com_octave_in_both_branches():
    # T_dev 1: no slope reaches it, so the last frequency is reported;
    # T_dev 0.05: the first crossing inside the valid region.
    for T, want in _OCT_BCH.items():
        assert _run(T)[0] == want, 'T_dev %g: Bch %r, COM Octave %r' % (T, _run(T)[0], want)


def test_cicp_and_fit_match_com_octave():
    B, CICP_db, fit, res, alpha, dres = _run(1.0)
    np.testing.assert_allclose(CICP_db[_IDX], _OCT_CICP_DB, rtol=0, atol=1e-12)
    np.testing.assert_allclose(alpha, _OCT_ALPHA, rtol=1e-9, atol=0)
    np.testing.assert_allclose(fit[::400], _OCT_FIT_EVERY_400, rtol=0, atol=1e-7)
    np.testing.assert_allclose(res[_IDX], _OCT_RESIDUAL, rtol=0, atol=1e-7)
    np.testing.assert_allclose(dres[_IDX], _OCT_DIFF, rtol=0, atol=1e-9)


def test_smoothing_window_is_always_2_GHz():
    """The reference tests isfield(param,'smooth_window_ghz') (lower case) and sets
    smooth_window_GHz, so a caller's smooth_window_GHz is overwritten with 2."""
    a = _run(0.05)
    b = _run(0.05, smooth_window_GHz=9.0)
    assert a[0] == b[0] and np.array_equal(a[5], b[5])


def test_caller_param_is_not_modified():
    f, H = _channel()
    p = SimpleNamespace(T_dev=1.0)
    get_ACBW(H, f / 1e9, SimpleNamespace(DISPLAY_WINDOW=0), p)
    assert vars(p) == {'T_dev': 1.0}, 'MATLAB passes param by value'
