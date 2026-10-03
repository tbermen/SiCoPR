"""Verification tests for get_CICP_fit_sweep (new in 4p17p0).

Its values against COM Octave are pinned through get_ACBW's test. Here: MATLAB's
colon and polyfit, the first-minimum choice, and the reference's defaults,
including the two it gets wrong (f_test_min_GHz assigned twice; no
f_test_max_GHz default).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)

from com_functions.fn.get_CICP_fit_sweep.py_impl import (  # noqa: E402
    get_CICP_fit_sweep, _colon, _polyfit1)

OP = SimpleNamespace(DISPLAY_WINDOW=0)


def test_colon_is_matlabs():
    assert _colon(20, 5, 130) == list(range(20, 131, 5))
    assert _colon(10, 10, 5) == []


def test_polyfit_order_one():
    x = np.linspace(20, 60, 81)
    np.testing.assert_allclose(_polyfit1(x, 3.0 * x - 2.0), [3.0, -2.0], rtol=1e-12)


def test_f_test_max_has_no_default():
    f = np.linspace(1, 100, 200)
    with pytest.raises(ValueError, match='f_test_max_GHz'):
        get_CICP_fit_sweep(np.log(f), f, OP)


def test_f_test_min_default_is_the_second_assignment():
    import inspect
    assert inspect.signature(get_CICP_fit_sweep).parameters['f_test_min_GHz'].default == 60


def test_axis_that_never_reaches_f_test_max_is_an_error():
    f = np.linspace(1, 50, 200)
    with pytest.raises(ValueError, match='f_test_max'):
        get_CICP_fit_sweep(np.log(f), f, OP, f_fit_min_GHz=2, f_upper_min_GHz=20,
                           f_upper_max_GHz=40, f_test_min_GHz=20, f_test_max_GHz=60, step_GHz=5)


def test_returns_the_window_it_chose():
    f = np.linspace(0.5, 100, 400)
    y = -40 + 8 * np.log10(f) + 0.002 * f ** 1.8
    out = get_CICP_fit_sweep(y, f, OP, f_fit_min_GHz=10, f_upper_min_GHz=20, f_upper_max_GHz=60,
                             f_test_min_GHz=20, f_test_max_GHz=60, step_GHz=10)
    res, fit, alpha, idx_fit, idx_valid, f_upper = out
    assert f_upper in (20, 30, 40, 50, 60)
    assert np.array_equal(idx_fit, (f >= 10) & (f <= f_upper))
    assert np.array_equal(idx_valid, f >= 10)
    np.testing.assert_allclose(res, y - fit, atol=1e-12)
