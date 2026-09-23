"""Verification tests for Apply_EQ().

# ============================================================
# MATLAB GROUND TRUTH (lines 905-976)
# Applies CTLE, TX-FFE to each channel's impulse/pulse response.
# INCLUDE_CTLE=0: eq_ir = uneq_ir (passthrough).
# CTLE_type='CL93': single TD_CTLE call.
# FFE applied for THRU/FEXT types.
# OP.RxFFE=False: eq_pulse_response set without force().
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import com_functions.fn.Apply_EQ.py_impl as _mod
from com_functions.fn.Apply_EQ.py_impl import Apply_EQ


def _ir(N=200, peak=10):
    ir = np.zeros(N)
    ir[peak] = 1.0
    return ir


def _param(M=8, n_files=1):
    return SimpleNamespace(
        fb=25e9,
        CTLE_fz=np.array([5e9]),
        CTLE_fp1=np.array([10e9]),
        CTLE_fp2=np.array([20e9]),
        ctle_gdc_values=np.array([0.0]),
        CTLE_type='CL93',
        samples_per_ui=M,
        number_of_s4p_files=n_files,
        f_HP=None,
        g_DC_HP_values=None,
        f_HP_Z=None,
        f_HP_P=None,
    )


def _fom(M=8, t_s=1, cur=5):
    ir = _ir()
    return SimpleNamespace(
        ctle=1,
        best_G_high_pass=1,
        sbr=ir,
        txffe=np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
        cur=cur,
        t_s=t_s,
        RxFFE=np.array([0.0, 1.0, 0.0]),
    )


def _op():
    return SimpleNamespace(INCLUDE_CTLE=0, RxFFE=False, FFE_OPT_METHOD='OTHER')


def _chdata(N=200, ctype='THRU', M=8):
    ir = _ir(N)
    t = np.arange(N) / (25e9 * M)
    return [SimpleNamespace(
        uneq_imp_response=ir,
        uneq_pulse_response=np.cumsum(ir),
        t=t,
        type=ctype,
    )]


def test_passthrough_no_ctle():
    """INCLUDE_CTLE=0: eq_imp_response == uneq_imp_response."""
    p = _param()
    fom = _fom()
    cd = _chdata()
    result = Apply_EQ(p, fom, cd, _op())
    np.testing.assert_array_equal(result[0].eq_imp_response, cd[0].uneq_imp_response)


def test_eq_pulse_response_set():
    """eq_pulse_response is set when RxFFE=False."""
    p = _param()
    fom = _fom()
    cd = _chdata()
    result = Apply_EQ(p, fom, cd, _op())
    assert hasattr(result[0], 'eq_pulse_response')


def test_pulse_sampled_set():
    """pulse_sampled_w_tx_ffe_ctle is a downsampled version of eq_pulse."""
    p = _param(M=8)
    fom = _fom(M=8, t_s=1)
    cd = _chdata(M=8)
    result = Apply_EQ(p, fom, cd, _op())
    assert hasattr(result[0], 'pulse_sampled_w_tx_ffe_ctle')
    assert len(result[0].pulse_sampled_w_tx_ffe_ctle) > 0


def test_ctle_applied_cl93():
    """INCLUDE_CTLE=1, CL93: eq_imp_response != uneq_imp_response."""
    p = _param()
    fom = _fom()
    cd = _chdata()
    op = SimpleNamespace(INCLUDE_CTLE=1, RxFFE=False, FFE_OPT_METHOD='OTHER')
    result = Apply_EQ(p, fom, cd, op)
    assert not np.allclose(result[0].eq_imp_response, cd[0].uneq_imp_response)


def test_rxffe_invokes_force(monkeypatch):
    """RxFFE=True routes the pulse through force() (no longer a stub).

    force() is a top-level fn in the assembled module; inject a spy to confirm
    the RxFFE branch reaches it and eq_pulse_response is still set. (force() is
    unit-tested in its own directory.)"""
    p = _param()
    fom = _fom()
    cd = _chdata()
    calls = {'n': 0}

    def spy_force(eq_pulse, param, OP, t_s, RxFFE):
        calls['n'] += 1
        return np.asarray(eq_pulse), None, None

    monkeypatch.setattr(_mod, 'force', spy_force, raising=False)
    op = SimpleNamespace(INCLUDE_CTLE=0, RxFFE=True, FFE_OPT_METHOD='OTHER')
    result = Apply_EQ(p, fom, cd, op)
    assert calls['n'] == 1
    assert hasattr(result[0], 'eq_pulse_response')


# ---------------------------------------------------------------------------
# COM Octave oracle (2026-09-22): Apply_EQ run verbatim under Octave from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m for this function).
#
# Two divergences it found:
#
# 1. The pulse-response extension built the new time stamps as
#    t(end) + k*(1/(fb*M)).  MATLAB L927 is (1:samples_added)/fb/M + t(end) --
#    two successive divisions, not one multiply by a pre-formed dt.  They are
#    different doubles (2 of 7, and 3 of 13, sample offsets differ at
#    fb=25e9 M=8), and the last extended time stamp came out
#    2.2999999999999998e-10 where Octave gives 2.3000000000000001e-10.
#
# 2. param.CTLE_type outside {CL93, CL120d, CL120e} fell through to the
#    unequalised response.  MATLAB's switch has no `otherwise`, so eq_ir is
#    never assigned:  COM Octave, CTLE_type='NOSUCH', INCLUDE_CTLE=1 ->
#    "error: 'eq_ir' undefined near line 46, column 31".
#
# Convention: fom_result.t_s is 0-BASED here (it is BEST.cursor_i) and 1-based
# in MATLAB, so the oracle run below used fom_result.t_s = 4 for t_s = 3.
# ---------------------------------------------------------------------------
_OCT_M, _OCT_FB, _OCT_N, _OCT_SBR = 8, 25e9, 40, 47


def _oct_ir():
    ir = np.zeros(_OCT_N)
    ir[5] = 1.0
    ir[6] = -0.3
    ir[9] = 0.12
    return ir


def _oct_inputs(ctle_type='CL93', include_ctle=0):
    ir = _oct_ir()
    p = SimpleNamespace(
        fb=_OCT_FB, CTLE_fz=np.array([5e9, 6e9]), CTLE_fp1=np.array([10e9, 11e9]),
        CTLE_fp2=np.array([20e9, 21e9]), ctle_gdc_values=np.array([0.0, -3.0]),
        CTLE_type=ctle_type, samples_per_ui=_OCT_M, number_of_s4p_files=1,
        f_HP=None, g_DC_HP_values=None, f_HP_Z=None, f_HP_P=None)
    fom = SimpleNamespace(ctle=1, best_G_high_pass=1, sbr=np.zeros(_OCT_SBR),
                          txffe=np.array([0.0, -0.2, 0.9, -0.1, 0.0]), cur=3,
                          t_s=3, RxFFE=np.array([0.0, 1.0, 0.0]))
    cd = [SimpleNamespace(uneq_imp_response=ir.copy(),
                          uneq_pulse_response=np.cumsum(ir),
                          t=np.arange(_OCT_N) / (_OCT_FB * _OCT_M), type='THRU')]
    op = SimpleNamespace(INCLUDE_CTLE=include_ctle, RxFFE=False,
                         FFE_OPT_METHOD='OTHER')
    return p, fom, cd, op


def test_octave_extended_time_axis_is_two_divisions():
    """COM Octave: the 7 samples appended to chdata(1).t.

    Bit-exact.  t(end)+k*(1/(fb*M)) puts the last stamp at
    2.2999999999999998e-10; MATLAB's (1:k)/fb/M + t(end) gives
    2.3000000000000001e-10."""
    p, fom, cd, op = _oct_inputs()
    out = Apply_EQ(p, fom, cd, op)
    t = np.asarray(out[0].t)
    assert t.size == _OCT_SBR
    expect_tail = np.array([2.0000000000000001e-10, 2.0499999999999999e-10,
                            2.1e-10, 2.1500000000000001e-10,
                            2.1999999999999999e-10, 2.25e-10,
                            2.3000000000000001e-10])
    np.testing.assert_array_equal(t[_OCT_N:], expect_tail)


def test_octave_eq_pulse_response_with_txffe():
    """COM Octave, INCLUDE_CTLE=0, txffe=[0 -0.2 0.9 -0.1 0], cur=3, M=8."""
    p, fom, cd, op = _oct_inputs()
    out = Apply_EQ(p, fom, cd, op)
    expect = np.array([
        -0.13999999999999999, -0.16400000000000001, -0.16400000000000001,
        -0.16400000000000001, -0.16400000000000001, 0.93600000000000005,
        0.60599999999999998, 0.60599999999999998, 0.60599999999999998,
        0.73799999999999999, 0.73799999999999999, 0.73799999999999999,
        0.73799999999999999, -0.26200000000000001, 0.038000000000000006,
        0.038000000000000006, 0.038000000000000006, -0.082000000000000003,
        -0.082000000000000003, -0.082000000000000003, -0.082000000000000003,
        0.017999999999999999, -0.012, -0.012, -0.012, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.20000000000000001,
        -0.13999999999999999, -0.13999999999999999])
    np.testing.assert_array_equal(np.asarray(out[0].eq_pulse_response), expect)


def test_octave_sampled_pulse_and_time():
    """COM Octave, sample_start = mod(t_s-1, M)+1 with MATLAB t_s = 4."""
    p, fom, cd, op = _oct_inputs()
    out = Apply_EQ(p, fom, cd, op)
    np.testing.assert_array_equal(
        np.asarray(out[0].pulse_sampled_w_tx_ffe_ctle),
        np.array([-0.16400000000000001, 0.73799999999999999,
                  -0.082000000000000003, 0, 0, 0]))
    np.testing.assert_array_equal(
        np.asarray(out[0].t_sampled_w_tx_ffe_ctle),
        np.array([1.5e-11, 5.4999999999999997e-11, 9.4999999999999995e-11,
                  1.35e-10, 1.7499999999999999e-10, 2.1500000000000001e-10]))


def test_octave_unknown_ctle_type_raises():
    """COM Octave, CTLE_type='NOSUCH' with INCLUDE_CTLE=1:
    "error: 'eq_ir' undefined".  The port returned the unequalised channel."""
    p, fom, cd, op = _oct_inputs(ctle_type='NOSUCH', include_ctle=1)
    with pytest.raises(ValueError, match='CTLE_type'):
        Apply_EQ(p, fom, cd, op)


# ---------------------------------------------------------------------------
# COM Octave oracle (2026-09-23): the two high-pass CTLE_type variants,
# CL120d and CL120e (MATLAB L933-942).  Both cascade a second TD_CTLE onto the
# CL93 output, and they differ in which parameter vector supplies its corners
# and in which index selects them:
#
#   CL120d  TD_CTLE(eq_ir, FB, FHP,  FHP,  100e100, GDCHP, M)
#           FHP   = param.f_HP(fom_result.best_G_high_pass)
#           GDCHP = param.g_DC_HP_values(fom_result.best_G_high_pass)
#   CL120e  TD_CTLE(eq_ir, FB, FHPZ, FHPP, 1e99,    0,     M)
#           FHPZ  = param.f_HP_Z(fom_result.ctle)
#           FHPP  = param.f_HP_P(fom_result.ctle)
#
# The two indices are deliberately different here (ctle = 2, best_G_high_pass
# = 1, MATLAB 1-based) and every vector holds two distinct values, so reading
# one index where the reference reads the other changes the answer.
#
# Not bit-exact: the residual is the IIR accumulation order of filter(), the
# same ~1e-16 seen in TD_CTLE's own oracle, which pins at rtol=1e-13.  The
# CTLE parameters themselves would have to be wrong by far more than that to
# land inside the tolerance.
# ---------------------------------------------------------------------------
_HP_OCT_TOL = dict(rtol=1e-13, atol=1e-16)


def _hp_inputs(ctle_type):
    """Same channel as _oct_inputs, with the high-pass vectors populated."""
    ir = _oct_ir()
    p = SimpleNamespace(
        fb=_OCT_FB, CTLE_fz=np.array([5e9, 6e9]), CTLE_fp1=np.array([10e9, 11e9]),
        CTLE_fp2=np.array([20e9, 21e9]), ctle_gdc_values=np.array([0.0, -3.0]),
        CTLE_type=ctle_type, samples_per_ui=_OCT_M, number_of_s4p_files=1,
        f_HP=np.array([0.667e9, 1.5e9]), g_DC_HP_values=np.array([-4.0, -6.0]),
        f_HP_Z=np.array([1e9, 2e9]), f_HP_P=np.array([2e9, 3e9]))
    fom = SimpleNamespace(ctle=2, best_G_high_pass=1, sbr=np.zeros(_OCT_SBR),
                          txffe=np.array([0.0, -0.2, 0.9, -0.1, 0.0]), cur=3,
                          t_s=3, RxFFE=np.array([0.0, 1.0, 0.0]))
    cd = [SimpleNamespace(uneq_imp_response=ir.copy(),
                          uneq_pulse_response=np.cumsum(ir),
                          t=np.arange(_OCT_N) / (_OCT_FB * _OCT_M), type='THRU')]
    op = SimpleNamespace(INCLUDE_CTLE=1, RxFFE=False, FFE_OPT_METHOD='OTHER')
    return p, fom, cd, op


# COM Octave, CTLE_type='CL120d', INCLUDE_CTLE=1.
_OCT_CL120D_EQ_IR = [
    0, 0, 0,
    0, 0, 0.41203997069250137,
    0.42305930034488204, -0.013344381352700413, -0.05796411370185766,
    -0.016344773343393806, 0.0062162325722530552, -0.0307696452217496,
    -0.03990794306973415, -0.037802469238939214, -0.031998473932904788,
    -0.025718593299020178, -0.02017800531147778, -0.015700746063477662,
    -0.012246744606718141, -0.0096523700672696174, -0.0077339904639734504,
    -0.006327704111648463, -0.0053005771601128656, -0.0045501109580482401,
    -0.0039996038388792308, -0.003592731680446045, -0.003288643001050967,
    -0.0030579539264808781, -0.0028796484443443788, -0.0027387535371467004,
    -0.0026246295726624204, -0.0025297292803772757, -0.0024487046858368,
    -0.0023777682644060415, -0.0023142378358301354, -0.0022562132906387864,
    -0.002202347427579715, -0.0021516837380863543, -0.00210354169676476,
    -0.0020574357036953102, -0.0020130178352417852, -0.0019700374245168154,
    -0.0019283125309216985, -0.0018877098048360483, -0.0018481302784559368,
    -0.0018094993389739119, -0.0017717596529603205,
]
_OCT_CL120D_EQ_PULSE = [
    -0.15096696193228132, -0.14774157594866205, -0.14902559554710157,
    -0.1429102590332425, -0.13496552076787593, 0.32580351912675098,
    0.7975342258527911, 0.78796594682024423, 0.72671243255353613,
    0.7118733310884986, 0.72116053583932049, 0.6892444001088498,
    0.64689246082493701, 0.19453531479193215, -0.26266219144712261,
    -0.27669824053173475, -0.24013001190472683, -0.24033751289506924,
    -0.25936743593450207, -0.23860380700145289, -0.20662732375322063,
    -0.13403358126041523, -0.065034886254613899, -0.045149907288641371,
    -0.034678136632949177, -0.024088319099420337, -0.014374610969466909,
    -0.010711712085373467, -0.0096956597316363025, -0.0097304946871258572,
    -0.010096149110879185, -0.010489112550408903, -0.010798280830049575,
    -0.010997161361894579, -0.011093191934744232, -0.011104767673725775,
    -0.011051474390425243, -0.010950443508543592, -0.010815415587741633,
    -0.013649570050415324, -0.013412541800184051, -0.01316708038588775,
    -0.012917188127992238, -0.012665709054353984, -0.094822637860319331,
    -0.17918523895000057, -0.17626986762762628,
]

# COM Octave, CTLE_type='CL120e', INCLUDE_CTLE=1.
_OCT_CL120E_EQ_IR = [
    0, 0, 0,
    0, 0, 0.61112684930100936,
    0.61433997845882016, -0.045000255004915828, -0.10803788212535938,
    -0.0420418787373633, -0.0060599916355356239, -0.05935626406109374,
    -0.070313221858483363, -0.064557813093698124, -0.053647130475146403,
    -0.042447557869359429, -0.032735601594545256, -0.02492943809748752,
    -0.018902948930833257, -0.014354649122990226, -0.010964480161975362,
    -0.0084522817736251022, -0.0065927678283345167, -0.0052128511716014589,
    -0.0041830837326219629, -0.0034082544000123365, -0.0028190601898892436,
    -0.0023653748000798443, -0.0020110674666285601, -0.0017301288966909253,
    -0.0015038268804736259, -0.0013186425783629919, -0.0011647849605481043,
    -0.0010351269927806246, -0.00092444640814081985, -0.00082888497285352224,
    -0.00074556377481737679, -0.00067230958484412718, -0.00060746013337939974,
    -0.00054972538806652568, -0.00049808854764770913, -0.00045173520052604703,
    -0.00041000246331549824, -0.00037234230442175116, -0.00033829495103453319,
    -0.0003074694764972357, -0.00027952951279758932,
]
_OCT_CL120E_EQ_PULSE = [
    -0.21383290790436782, -0.2054912017981852, -0.20433754265030352,
    -0.19251773423256732, -0.17850074412771383, 0.50660962583975788,
    1.1930765442286546, 1.1520329822350612, 1.039417713431644,
    0.99815753444004185, 0.99527213342711929, 0.93285117278451424,
    0.8576995247725776, 0.17724953742322536, -0.49478373099258904,
    -0.49543321940964824, -0.42256788229176429, -0.40726673458163482,
    -0.42143617473203793, -0.37739694974621746, -0.31874244257258127,
    -0.20202342872043161, -0.093893579634476448, -0.061416455038697582,
    -0.043853076770462809, -0.026669880988169081, -0.011289008148139365,
    -0.0053061207167756433, -0.0034360241989765707, -0.0032082036041124681,
    -0.0035128663651376978, -0.0038853327390581671, -0.0041574549129641068,
    -0.0042904369746539998, -0.0042965622341398857, -0.0042039573556135867,
    -0.0040417990672747713, -0.0038349449959754495, -0.0036026951424930967,
    -0.003941528099411503, -0.0036433312050085324, -0.0033555913327025831,
    -0.0030820531605346363, -0.0028248137416686742, -0.12481019202964712,
    -0.24745561355508364, -0.23825006167028626,
]


def test_octave_ctle_type_CL120d():
    """COM Octave, CTLE_type='CL120d': CL93 stage then the f_HP high-pass
    stage, indexed by fom_result.best_G_high_pass (= 1, not ctle = 2)."""
    p, fom, cd, op = _hp_inputs('CL120d')
    out = Apply_EQ(p, fom, cd, op)[0]
    np.testing.assert_allclose(np.asarray(out.eq_imp_response),
                               _OCT_CL120D_EQ_IR, **_HP_OCT_TOL)
    np.testing.assert_allclose(np.asarray(out.eq_pulse_response),
                               _OCT_CL120D_EQ_PULSE, **_HP_OCT_TOL)
    np.testing.assert_allclose(
        np.asarray(out.pulse_sampled_w_tx_ffe_ctle),
        np.asarray(_OCT_CL120D_EQ_PULSE)[3::_OCT_M], **_HP_OCT_TOL)


def test_octave_ctle_type_CL120e():
    """COM Octave, CTLE_type='CL120e': CL93 stage then the f_HP_Z/f_HP_P
    stage, both indexed by fom_result.ctle (= 2)."""
    p, fom, cd, op = _hp_inputs('CL120e')
    out = Apply_EQ(p, fom, cd, op)[0]
    np.testing.assert_allclose(np.asarray(out.eq_imp_response),
                               _OCT_CL120E_EQ_IR, **_HP_OCT_TOL)
    np.testing.assert_allclose(np.asarray(out.eq_pulse_response),
                               _OCT_CL120E_EQ_PULSE, **_HP_OCT_TOL)
    np.testing.assert_allclose(
        np.asarray(out.pulse_sampled_w_tx_ffe_ctle),
        np.asarray(_OCT_CL120E_EQ_PULSE)[3::_OCT_M], **_HP_OCT_TOL)
