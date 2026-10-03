"""Verification tests for get_BW_from_CICP_residual (new in 4p17p0).

COM Octave: get_BW_from_CICP_residual extracted verbatim from
octave/com_ieee8023_4p17p0_octave_compat.m by tools/octave_oracle.py, run on the
inputs _case() builds: an uneven frequency axis (MATLAB gradient's centred
difference, not numpy's non-uniform formula), NaNs in the residual (movmean
'omitnan'), an odd smoothing window, and a crossing inside idx_fit. Generator:
runs/4p17p0_oracles/gen_bw_oracle.py (local). Agreement ~1e-15; signed zeros
included. Get_ACBW's test covers an even window (40 samples).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _ROOT)

from com_functions.fn.get_BW_from_CICP_residual.py_impl import (  # noqa: E402
    get_BW_from_CICP_residual, _movmean_omitnan, _gradient)

# COM Octave: Bch_GHz 8.0, idx_BW 15 (1-based)
_OCT_BCH, _OCT_IDX1 = 8.0, 15
_OCT_DIFF = [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.10093834005527436, 0.11129281979667602, 0.11817499235657634, 0.12144955546571048,
             0.12108118647235504, 0.11713552125793893, 0.10977733523462332, 0.09926597683584049,
             0.08594818005392074, 0.07024845521481055, 0.05265732429550091, 0.03371772682513058,
             0.01400997311032437, -0.00586433821588761, -0.025298992619969596, -0.04369996001670934,
             -0.06050185744388903, -0.07518361782385646, -0.08728293057519579, -0.0964090538740727,
             -0.10225364349217725, -0.1045992981058655, -0.10332558425575879, -0.10353588668499686,
             -0.0653403626553246, -0.06087154836842503, -0.0, -0.01370144537100404,
             0.029542786860906792, 0.062444458607409326, 0.09888114842539925, 0.13190458787715748,
             0.15939641104145613, 0.1796312715807001, 0.19139942805125346, 0.19409269564950946,
             0.1877476575493844, 0.17304306567940594, 0.15125161561336656, 0.12414952195326458,
             0.09389031884491442, 0.06285185205958695, 0.033467333805764665, 0.008052463773088369,
             -0.0113591005964831, -0.023160234825231323, -0.02628453308434148, -0.02028207598997412,
             -0.005351397232857453, 0.017674622177501268, 0.04738792984450675, 0.08190534366088507,
             0.11900235130040974, 0.15627116111348655, 0.19129177496055091, 0.22180392831271833,
             0.24586768133002893, 0.26200125097208504, 0.2692862912133413, 0.2674331412370729,
             0.2568014057445971, 0.2383744051552291, 0.21368931100222346, 0.18472793038263666,
             0.1537758991913351, 0.12326028841525154, 0.09557716150915996, 0.07292133484680982,
             0.057130436713349046, 0.04955434439821719, 0.05095927584165932, 0.06147335009297111,
             0.08057748449266332, 0.07110196956150176, 0.05003176595506087]


def _case():
    f = np.concatenate([np.linspace(1, 20, 39), np.linspace(20.7, 60, 50)])
    res = 0.002 * (f - 12) ** 2 - 0.4 * np.sin(f / 3)
    res[[5, 6, 40]] = np.nan
    return f, res, f >= 8


def test_matches_com_octave():
    f, res, mask = _case()
    B, idx0, d = get_BW_from_CICP_residual(f, res, mask, 0.09, SimpleNamespace(DISPLAY_WINDOW=0),
                                           smooth_window_GHz=1.5)
    assert B == _OCT_BCH and idx0 == _OCT_IDX1 - 1
    np.testing.assert_allclose(d, _OCT_DIFF, rtol=0, atol=1e-13)
    assert [np.signbit(x) for x in d if x == 0] == [np.signbit(x) for x in _OCT_DIFF if x == 0]


# COM Octave (Octave 11.3 built-ins, 2026-10-03):
#   movmean(1:7,3,'omitnan')   1.5 2 3 4 5 6 6.5
#   movmean(1:7,4,'omitnan')   1.5 2 2.5 3.5 4.5 5.5 6
#   movmean([1 NaN 3],3,'omitnan')   1 2 3
#   x=[0 1 3 6]; gradient(x.^2,x)    1 3 7 9
def test_movmean_window_shape_is_matlabs():
    x = np.arange(1.0, 8.0)
    np.testing.assert_array_equal(_movmean_omitnan(x, 3), [1.5, 2, 3, 4, 5, 6, 6.5])
    np.testing.assert_array_equal(_movmean_omitnan(x, 4), [1.5, 2, 2.5, 3.5, 4.5, 5.5, 6])
    np.testing.assert_array_equal(_movmean_omitnan(np.array([1, np.nan, 3.0]), 3), [1, 2, 3])


def test_gradient_is_the_centred_difference_not_numpys():
    x = np.array([0.0, 1.0, 3.0, 6.0])
    y = x ** 2
    np.testing.assert_array_equal(_gradient(y, x), [1.0, 3.0, 7.0, 9.0])
    assert not np.array_equal(_gradient(y, x), np.gradient(y, x))


def test_errors_where_the_reference_errors():
    f, res, mask = _case()
    op = SimpleNamespace(DISPLAY_WINDOW=0)
    with pytest.raises(ValueError, match='same length'):
        get_BW_from_CICP_residual(f[:-1], res, mask, 1, op, smooth_window_GHz=1)
    with pytest.raises(ValueError, match='enough valid'):
        get_BW_from_CICP_residual(f, res, np.zeros_like(mask), 1, op, smooth_window_GHz=1)
    with pytest.raises(ValueError, match='smooth_window'):
        get_BW_from_CICP_residual(f, res, mask, 1, op)


def test_window_length_rounds_a_tie_away_from_zero():
    """Nwin = max(3, round(smooth/df)) with smooth/df = 4.5 exactly: MATLAB 5,
    round-half-to-even 4. A 4-wide and a 5-wide moving mean differ, so the slope
    does too; 4.5 must give what 5.0 gives."""
    f = np.arange(0.0, 60.0, 1.0)                         # df = 1 GHz exactly
    res = 0.002 * (f - 12) ** 2 - 0.4 * np.sin(f / 3)
    op = SimpleNamespace(DISPLAY_WINDOW=0)
    d45 = get_BW_from_CICP_residual(f, res, f >= 8, 0.09, op, smooth_window_GHz=4.5)[2]
    d50 = get_BW_from_CICP_residual(f, res, f >= 8, 0.09, op, smooth_window_GHz=5.0)[2]
    d40 = get_BW_from_CICP_residual(f, res, f >= 8, 0.09, op, smooth_window_GHz=4.0)[2]
    assert np.array_equal(d45, d50)
    assert not np.array_equal(d45, d40)
