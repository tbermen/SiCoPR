"""Verification tests for get_PulseR().

# ============================================================
# MATLAB GROUND TRUTH (lines 6655-6682)
# cb_step=False: pulse = filter(ones(1,M), 1, ir)  [running sum of M samples]
# cb_step=True:  pulse = filter(drive_pulse, 1, ir) [shaped edge]
# PDR = (1+pulse)./(1-pulse) * ZT * 2
# result.PDR, result.pulse
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.get_PulseR.py_impl import get_PulseR


def _param(M=4, fb=50e9, TR_TDR=10.0):
    return SimpleNamespace(samples_per_ui=M, fb=fb, TR_TDR=TR_TDR)


def test_no_cbstep_pulse_is_running_sum():
    """cb_step=False: pulse[k] = sum of ir[k-M+1..k] (rectangular filter)."""
    ir = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    M = 4
    result = get_PulseR(ir, _param(M=M), False, 1.0)
    # pulse[3] should be 1.0 (only ir[0] in window), pulse[7] should be 1.0
    assert result.pulse[3] == pytest.approx(1.0)
    assert result.pulse[7] == pytest.approx(1.0)


def test_output_length_matches_input():
    """Output length == len(ir)."""
    ir = np.zeros(20)
    result = get_PulseR(ir, _param(), False, 50.0)
    assert len(result.PDR) == 20
    assert len(result.pulse) == 20


def test_pdr_formula():
    """PDR = (1+pulse)/(1-pulse)*ZT*2."""
    ir = np.zeros(8)
    result = get_PulseR(ir, _param(), False, 50.0)
    expected_pdr = (1 + result.pulse) / (1 - result.pulse) * 50.0 * 2
    np.testing.assert_allclose(result.PDR, expected_pdr)


def test_returns_namespace_with_fields():
    """Result has PDR and pulse fields."""
    ir = np.ones(8) * 0.01
    result = get_PulseR(ir, _param(), False, 25.0)
    assert hasattr(result, 'PDR')
    assert hasattr(result, 'pulse')


def test_cbstep_true_output_length():
    """cb_step=True: output length still matches ir length."""
    ir = np.zeros(40)
    result = get_PulseR(ir, _param(), True, 50.0)
    assert len(result.PDR) == 40


# ============================================================
# COM Octave oracle — get_PulseR extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m) and run by tools/octave_oracle.py, with
# param.fb=106.25e9, samples_per_ui=32, TR_TDR=8e-3 (the shipped default,
# com_ieee8023_4p16p0.m line 10361) and ZT=50.
#
# DIVERGENCE 1, the important one: `tedge = 0:dt:edge_time*2` was ported as
# np.arange(0, edge_time*2 + dt, dt), which is ONE ELEMENT TOO LONG whenever
# edge_time*2/dt is not an integer -- and at the default settings it is 54.4.
# drive_pulse = [edge ones(1,samples_per_ui)] was therefore 88 samples where
# the reference builds 87, so every TDR pulse response carried a spurious
# trailing drive sample.  A delta impulse response makes drive_pulse visible
# directly, and Octave's runs out at index 87 where the port still had a 1.
# Plain floor() does not fix it either: the quotient can land a fraction of
# an eps BELOW an integer, e.g. TR_TDR=0.5325 at these settings gives
# 3620.9999999999995 where Octave returns 3622 points (drive_pulse 3654).
# The spelling now pinned matched Octave bit-exactly on 672 combinations of
# fb in {26.5625, 42.5, 53.125, 100, 106.25, 112.5, 120} GHz,
# samples_per_ui in {4,8,16,32,64,128} and TR_TDR in {0, +-1e-3, 8e-3, ...,
# 0.5325, 0.5575} ns, including the last-element clamp to the limit.
#
# DIVERGENCE 2: an empty ir.  filter(b,1,[]) returns [] in the reference;
# scipy's lfilter raised "v cannot be empty".
#
# DIVERGENCE 3: TR_TDR=0 makes fedge=1/0.  The reference carries Inf into
# 0*Inf = NaN and returns an all-NaN pulse and PDR; Python's scalar division
# raised ZeroDivisionError.
#
# DIVERGENCE 4: a fractional samples_per_ui.  ones(1,2.5) is refused --
# "conversion of 2.5 to int64_t value failed" -- while int() silently
# truncated it to 2 and answered.  ones(1,4.0) is accepted, and ones(1,0) and
# ones(1,-3) both give the empty row, so only a non-integral value is refused.
#
# Tolerance: MATLAB's filter and scipy's lfilter accumulate in different
# orders, which leaves a ~1 ULP floor (worst measured 4.6e-15 relative over
# these probes), so the filtered arrays are pinned at rtol=1e-12.  The
# drive-pulse probes below are bit-exact.
# ============================================================

def _oct_param(M=32, fb=106.25e9, TR_TDR=8e-3):
    return SimpleNamespace(samples_per_ui=M, fb=fb, TR_TDR=TR_TDR)


OCT_RTOL = 1e-12
OCT_DRIVE_LEN = 87                 # 55 tedge points + 32 ones; arange gave 88
OCT_TR5325_DRIVE_LEN = 3654        # 3622 + 32; floor() gives 3653, arange 3655
OCT_DRIVE_HEAD = np.array([
    2.2204460492503131e-16, 0.028870920176194748, 0.057717770556606096,
    0.086516501412531532, 0.1152431031326977, 0.14387362624016675,
    0.17238420135910437, 0.20075105911476943, 0.22895054995013431,
    0.25695916384260498])

OCT_IR = np.array([
    6.8385534506368341e-05, 0.0027194950806199237, 0.002449442157171865,
    -0.0010206141535753349, -0.00059593902221289415, -0.0010547683860668503,
    0.0011394527151439202, -0.00011212887809123519, 0.0014937712325130879,
    -0.0036946495979482191, 0.0031330975493990411, -0.0001928643203112411,
    0.0013607569065482922, -0.00027313266795365548, -0.00075819713414970659,
    0.00092622031719517351, 0.0016490270550602261, -0.00040505974138690303,
    -0.00030557235714039418, 0.001371397221618516, -0.0017406812838943425,
    -0.0030287670074627912, 0.00078996372549906, -0.0013411316473757589,
    -0.0038406811802360573, -0.0016281073278907191, -0.00093519511778549403,
    -0.0023864049548645225, -0.0029849277681260676, 7.3275653889610175e-05,
    0.0017944985134554952, -0.00046626415592091369, -0.0014871920590176896,
    0.0007699876174958166, 0.0014344716143887678, -0.00060002119697695477,
    0.0010893356158417859, 0.0020857509531659077, -0.00041391287241664793,
    -0.0016270310839631445, 0.00069530119703101905, 0.00049509148192569512,
    0.002197625536828817, -0.0025691615576106898, -0.0013232258607110954,
    -0.0016763339214313491, -0.0034680296924657029, 0.00025286911039399242])

OCT_CB_PULSE = np.array([
    1.5184638992053654e-20, 1.9743533079403758e-06, 8.2461385982162094e-05,
    0.00023359731928167165, 0.00035507243185048403, 0.00045904621124434895,
    0.00053218514819024188, 0.00063777744874067685, 0.00073960076792142847,
    0.00088393402914884074, 0.00092086241759819511, 0.0010474784888061131,
    0.0011676531028237559, 0.0013261405440437755, 0.0014756367854024365,
    0.0016020129338317778, 0.001753794310838136, 0.0019517224713428414,
    0.0021363290259517002, 0.0023103323598082258, 0.0025220030605185013,
    0.0026813160873008449, 0.0027509504001746281, 0.0028410982149145022,
    0.0028901576905468736, 0.0028259236309609572, 0.0027123286314111464,
    0.002569472409586983, 0.0023555763018978521, 0.0020535387299756669,
    0.0017519046505706502, 0.0015006188249441698, 0.0012346204524783845,
    0.00092465616903054568, 0.00063615124726474107, 0.00038853047926245513,
    0.00012326262784228051, -0.00011065786648873079, -0.0002842685556599799,
    -0.00046959229455635426, -0.00070149841712284901, -0.00091274571243727179,
    -0.0011089383015306542, -0.0012407588931001036, -0.0014457191175183315,
    -0.0016876767886180876, -0.0019766247400048002, -0.0023640499802597388])

OCT_NOCB_PULSE = np.array([
    6.8385534506368341e-05, 0.0027878806151262919, 0.0052373227722981568,
    0.0042167086187228217, 0.0036207695965099276, 0.0025660012104430771,
    0.0037054539255869973, 0.0035933250474957622, 0.0050870962800088505,
    0.0013924466820606314, 0.004525544231459672, 0.0043326799111484309,
    0.0056934368176967233, 0.0054203041497430677, 0.0046621070155933612,
    0.0055883273327885351, 0.0072373543878487615, 0.006832294646461858,
    0.0065267222893214639, 0.0078981195109399799, 0.006157438227045637,
    0.0031286712195828458, 0.0039186349450819058, 0.0025775032977061471,
    -0.0012631778825299101, -0.0028912852104206292, -0.003826480328206123,
    -0.0062128852830706455, -0.0091978130511967127, -0.0091245373973071024,
    -0.007330038883851607, -0.0077963030397725204, -0.00935188063329658,
    -0.011301388096420688, -0.012316358639203787, -0.011895765682605405,
    -0.010210491044550725, -0.0070699717053179668, -0.0086233372928785374,
    -0.010138239498750444, -0.010936709534232515, -0.0067469684543586001,
    -0.0076824404669288242, -0.010058737704228271, -0.012742720471487659,
    -0.014145921724965354, -0.01685575428328135, -0.017529105490082533])
OCT_NOCB_PDR_HEAD = np.array([
    100.0136780422815, 100.55913492444235, 101.05297934717335,
    100.846912908706, 100.72678544193334])


def _drive_pulse_from_delta(param, n=4000):
    """filter(drive_pulse, 1, delta) IS drive_pulse, zero padded."""
    ir = np.zeros(n)
    ir[0] = 1.0
    return get_PulseR(ir, param, True, 50.0).pulse


def test_oracle_drive_pulse_length_at_default_TR_TDR():
    """`0:dt:edge_time*2` is 55 points at the shipped defaults, not 56."""
    pulse = _drive_pulse_from_delta(_oct_param(), 200)
    assert int(np.nonzero(pulse)[0][-1]) + 1 == OCT_DRIVE_LEN
    np.testing.assert_array_equal(pulse[:10], OCT_DRIVE_HEAD)
    np.testing.assert_array_equal(pulse[OCT_DRIVE_LEN - 4:OCT_DRIVE_LEN],
                                  np.ones(4))
    np.testing.assert_array_equal(pulse[OCT_DRIVE_LEN:], 0.0)


def test_oracle_drive_pulse_length_when_quotient_is_just_below_an_integer():
    """TR_TDR=0.5325 gives 3620.9999999999995 steps; the reference keeps 3622."""
    pulse = _drive_pulse_from_delta(_oct_param(TR_TDR=0.5325), 4000)
    assert int(np.nonzero(pulse)[0][-1]) + 1 == OCT_TR5325_DRIVE_LEN


def test_oracle_drive_pulse_length_when_quotient_is_exact():
    """TR_TDR=10e-3 is 68 steps exactly -> 69 points + 32 ones."""
    pulse = _drive_pulse_from_delta(_oct_param(TR_TDR=10e-3), 300)
    assert int(np.nonzero(pulse)[0][-1]) + 1 == 69 + 32


# OCT_IR tiled three times (144 samples), so the response runs past sample 87
# and the spurious 88th drive sample actually reaches the output.
OCT_CB_LONG_80_100 = np.array([
    -0.0097879608158936079, -0.010179069357613037, -0.010542947228798482,
    -0.010862089379873482, -0.011196957089543253, -0.011498382474520738,
    -0.011736304813572442, -0.012051470026183307, -0.015062253260899862,
    -0.017779610221596638, -0.017008248118035443, -0.016593647876581248,
    -0.01578883031068554, -0.017209368885965352, -0.017419781529647606,
    -0.019329935973634787, -0.016037343635190986, -0.019562481887377553,
    -0.019673771653571419, -0.021259976124679759])
OCT_CB_LONG_TAIL = np.array([
    -0.01578883031068554, -0.017209368885965352, -0.017419781529647606,
    -0.019329935973634787])


def test_oracle_cb_step_pulse_past_the_drive_pulse_length():
    """An ir longer than drive_pulse exposes the extra 88th sample."""
    r = get_PulseR(np.tile(OCT_IR, 3), _oct_param(), True, 50.0)
    assert r.pulse.size == 144
    np.testing.assert_allclose(r.pulse[80:100], OCT_CB_LONG_80_100,
                               rtol=OCT_RTOL, atol=0)
    np.testing.assert_allclose(r.pulse[-4:], OCT_CB_LONG_TAIL,
                               rtol=OCT_RTOL, atol=0)


def test_oracle_cb_step_pulse_and_pdr():
    r = get_PulseR(OCT_IR, _oct_param(), True, 50.0)
    np.testing.assert_allclose(r.pulse, OCT_CB_PULSE, rtol=OCT_RTOL, atol=1e-19)
    np.testing.assert_allclose(
        r.PDR, (1 + OCT_CB_PULSE) / (1 - OCT_CB_PULSE) * 50.0 * 2,
        rtol=OCT_RTOL, atol=0)


def test_oracle_rectangular_pulse_and_pdr():
    r = get_PulseR(OCT_IR, _oct_param(), False, 50.0)
    np.testing.assert_allclose(r.pulse, OCT_NOCB_PULSE, rtol=OCT_RTOL, atol=0)
    np.testing.assert_allclose(r.PDR[:5], OCT_NOCB_PDR_HEAD, rtol=OCT_RTOL, atol=0)


def test_empty_ir_returns_empty_not_a_raise():
    """filter(b,1,[]) is []; scipy's lfilter raised 'v cannot be empty'."""
    for cb in (False, True):
        r = get_PulseR(np.zeros(0), _oct_param(), cb, 50.0)
        assert r.pulse.size == 0 and r.PDR.size == 0


def test_zero_TR_TDR_gives_nan_not_a_raise():
    """fedge = 1/0 is Inf, and 0*Inf = NaN carries through the whole result."""
    r = get_PulseR(np.array([1.0, 0.5, 0.25, 0.0, 0.0]),
                   _oct_param(TR_TDR=0.0), True, 50.0)
    assert np.all(np.isnan(r.pulse))
    assert np.all(np.isnan(r.PDR))


def test_fractional_samples_per_ui_is_refused():
    """ones(1,2.5) is an error in the reference; int() had truncated it to 2."""
    for cb in (False, True):
        with pytest.raises(ValueError, match='samples_per_ui'):
            get_PulseR(OCT_IR, _oct_param(M=2.5), cb, 50.0)


def test_integral_float_samples_per_ui_is_accepted():
    """ones(1,4.0) is fine -- only a non-integral value is refused."""
    a = get_PulseR(OCT_IR, _oct_param(M=32.0), False, 50.0)
    b = get_PulseR(OCT_IR, _oct_param(M=32), False, 50.0)
    np.testing.assert_array_equal(a.pulse, b.pulse)


def test_non_positive_samples_per_ui_gives_an_empty_filter():
    """ones(1,0) and ones(1,-3) are both the empty row, so pulse is all zero."""
    for M in (0, -3):
        r = get_PulseR(OCT_IR, _oct_param(M=M), False, 50.0)
        np.testing.assert_array_equal(r.pulse, np.zeros(OCT_IR.size))
        np.testing.assert_array_equal(r.PDR, np.full(OCT_IR.size, 100.0))


def test_zero_samples_per_ui_with_cb_step_makes_dt_infinite():
    """dt = 1/fb/0 is Inf, so `0:Inf:L` collapses to the single base point.

    drive_pulse is then just that one edge sample, 2*cos(-pi/4)^2-1, which
    rounds to 2.2204460492503131e-16 rather than to 0.  COM Octave, ir =
    [1 0.5 0.25 0 0]:  pulse = [2.220446049250313e-16 1.1102230246251565e-16
    5.5511151231257827e-17 0 0].  The scalar 1/0 had raised ZeroDivisionError,
    and 0*Inf would have made the base point NaN.
    """
    r = get_PulseR(np.array([1.0, 0.5, 0.25, 0.0, 0.0]),
                   _oct_param(M=0), True, 50.0)
    np.testing.assert_array_equal(r.pulse, np.array(
        [2.220446049250313e-16, 1.1102230246251565e-16,
         5.5511151231257827e-17, 0.0, 0.0]))
    # ones(1,-3) is empty and -3 does not make dt infinite, so the single
    # edge sample is the ordinary one and the response is identically zero.
    r = get_PulseR(np.array([1.0, 0.5, 0.25, 0.0, 0.0]),
                   _oct_param(M=-3), True, 50.0)
    np.testing.assert_array_equal(r.pulse, np.zeros(5))
