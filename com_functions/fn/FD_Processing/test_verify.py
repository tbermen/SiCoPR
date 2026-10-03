# ============================================================
# MATLAB GROUND TRUTH
# FD_Processing: freq-domain metrics — IL fit, FOM_ILD, ICN.
# MATLAB lines 1684–2025.
#
# Key invariants:
# 1. A/Aicn assigned from param arrays based on channel type
# 2. DO_ONCE=False → returns immediately after A assignment
# 3. GET_FD=False → initializes output_args fields, returns early
# 4. FOM_ILD = sqrt(delta_f/fb * sum(PWF[f1:f2_ild] * ILD_magft^2))
# 5. MDFEXT_ICN = sqrt(2*df/fb * sum(Aicn^2 * PWF * |MDFEXT|^2))
# 6. ICN = sqrt(2*df/fb * sum(PWF * |PSXT|^2))  (power-sum of all XT)
# 7. NOISE channels are skipped in the ICN accumulation
# ============================================================

import os
import numpy as np
import pytest
from types import SimpleNamespace

from com_functions.fn.FD_Processing.py_impl import FD_Processing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_faxis(N=64, fb=53.125e9):
    return np.linspace(0.0, fb / 2, N)


def _make_ch(ch_type='THRU', N=64, fb=53.125e9, ftr=None):
    faxis = _make_faxis(N, fb)
    if ftr is None:
        ftr = 0.75 * fb / 2
    mag = 0.5 * np.ones(N)
    z = np.zeros(N, dtype=complex)
    return SimpleNamespace(
        type=ch_type,
        faxis=faxis,
        ftr=ftr,
        sdd21f=mag.astype(complex),
        sdd21=mag.astype(complex),
        sdd21_orig=mag.astype(complex),
        sdd21_raw=mag.astype(complex),
        sdd21p=mag.astype(complex),
        sdd21p_nodie=mag.astype(complex),
        scd21_orig=z.copy(),
        sdc21_orig=z.copy(),
    )


def _make_param(fb=53.125e9, N=64, n_chan=1):
    # n_chan: MATLAB loops `for i=1:param.number_of_s4p_files`, so a fixture
    # that hands FD_Processing more channels than it declares describes a call
    # the reference would not make -- and, before 2026-09-23, one the port
    # answered differently (it iterated the whole list).
    faxis = _make_faxis(N, fb)
    return SimpleNamespace(
        package_testcase_i=1,
        number_of_s4p_files=n_chan,
        a_thru=np.array([1.0]),
        a_fext=np.array([0.5]),
        a_next=np.array([0.3]),
        a_icn_fext=0.5,
        a_icn_next=0.3,
        f1=float(faxis[5]),
        f2=float(faxis[-5]),
        f2_ild=float(faxis[-8]),
        fb=fb,
        f_r=0.75,
        ui=1.0 / fb,
        samples_per_ui=4,
        sample_dt=1.0 / (2 * fb),
        sigma_X=1.0,
        P_peak=1e-4,
    )


def _make_op():
    return SimpleNamespace(
        WC_PORTZ=False,
        TDMODE=False,
        GET_FD=True,
        INCLUDE_FILTER=False,
        COMPUTE_TDILN=False,
        COMPUTE_RILN=False,
        include_pcb=False,
        pkg_len_select=np.array([1]),
        DEBUG=False,
        DISPLAY_WINDOW=False,
    )


def _stub_get_ILN(sdd21, faxis):
    """Returns (ILD_magft, fit) both same length as input."""
    n = len(sdd21)
    ILD = np.abs(sdd21)
    fit = -20.0 * np.log10(np.abs(sdd21) + 1e-300)
    return ILD, fit


def _stub_get_ILN_cmp_td(sdd21, faxis, OP, param, A):
    n = len(sdd21)
    ILD = np.abs(sdd21)
    fit = -20.0 * np.log10(np.abs(sdd21) + 1e-300)
    TD_ILN = SimpleNamespace(SNR_ISI_FOM_PDF=3.5, FOM=0.1 + 0j)
    return ILD, fit, TD_ILN


def _stub_capture_RIL_RILN(chdata):
    N = len(np.asarray(chdata[0].faxis).ravel())
    return SimpleNamespace(RILN_dB=np.zeros(N))


# ---------------------------------------------------------------------------
# Test 1 – A/Aicn assignment by channel type
# ---------------------------------------------------------------------------

def test_amplitude_assignment():
    """A and Aicn are set from param arrays based on channel type."""
    param = _make_param(n_chan=4)
    OP = _make_op()
    chdata = [
        _make_ch('THRU'),
        _make_ch('FEXT'),
        _make_ch('NEXT'),
        _make_ch('NOISE'),
    ]

    chdata_out, _ = FD_Processing(chdata, SimpleNamespace(), param, OP,
                                   DO_ONCE=False, _get_ILN_fn=_stub_get_ILN)

    assert chdata_out[0].A == pytest.approx(float(param.a_thru[0]))
    assert chdata_out[1].A == pytest.approx(float(param.a_fext[0]))
    assert chdata_out[2].A == pytest.approx(float(param.a_next[0]))
    assert chdata_out[3].A == pytest.approx(1.0)
    assert chdata_out[1].Aicn == pytest.approx(float(param.a_icn_fext))
    assert chdata_out[2].Aicn == pytest.approx(float(param.a_icn_next))


# ---------------------------------------------------------------------------
# Test 2 – DO_ONCE=False returns early without computing FOM_ILD
# ---------------------------------------------------------------------------

def test_do_once_false_early_return():
    """DO_ONCE=False → only A/Aicn set, output_args fields not initialized."""
    param = _make_param()
    OP = _make_op()
    output_args = SimpleNamespace()
    chdata, out = FD_Processing([_make_ch('THRU')], output_args, param, OP,
                                 DO_ONCE=False, _get_ILN_fn=_stub_get_ILN)

    assert not hasattr(out, 'FOM_ILD'), "FOM_ILD should not be set with DO_ONCE=False"
    assert not hasattr(out, 'fitted_IL_dB_at_Fnq'), "output_args should not be initialized"


# ---------------------------------------------------------------------------
# Test 3 – GET_FD=False initializes fields but returns before computing FOM
# ---------------------------------------------------------------------------

def test_get_fd_false_fields_initialized():
    """GET_FD=False → output_args fields are initialized to [] but no FOM computed."""
    param = _make_param()
    OP = _make_op()
    OP.GET_FD = False
    output_args = SimpleNamespace()
    chdata, out = FD_Processing([_make_ch('THRU')], output_args, param, OP,
                                 DO_ONCE=True, _get_ILN_fn=_stub_get_ILN)

    assert hasattr(out, 'FOM_ILD')
    assert out.FOM_ILD == []
    assert out.FOM_RILN == []
    assert not hasattr(out, 'P_signal_FD'), "P_signal_FD should not be set"


# ---------------------------------------------------------------------------
# Test 4 – FOM_ILD is positive for a THRU channel
# ---------------------------------------------------------------------------

def test_fom_ild_positive_thru():
    """FOM_ILD = sqrt(df/fb * sum(PWF * ILD^2)) > 0 for non-trivial channel."""
    param = _make_param()
    OP = _make_op()
    output_args = SimpleNamespace()
    _, out = FD_Processing([_make_ch('THRU')], output_args, param, OP,
                            DO_ONCE=True, _get_ILN_fn=_stub_get_ILN)

    assert hasattr(out, 'FOM_ILD')
    assert float(out.FOM_ILD) > 0.0, f"FOM_ILD={out.FOM_ILD} should be > 0"


# ---------------------------------------------------------------------------
# Test 5 – MDFEXT_ICN computed for FEXT channel, ICN for FEXT+NEXT
# ---------------------------------------------------------------------------

def test_fext_next_icn():
    """FEXT → MDFEXT_ICN set; NEXT → MDNEXT_ICN set; ICN_mV > 0."""
    param = _make_param(n_chan=3)
    OP = _make_op()
    output_args = SimpleNamespace()
    chdata = [_make_ch('THRU'), _make_ch('FEXT'), _make_ch('NEXT')]
    _, out = FD_Processing(chdata, output_args, param, OP,
                            DO_ONCE=True, _get_ILN_fn=_stub_get_ILN)

    assert hasattr(out, 'MDFEXT_ICN_92_47_mV')
    assert float(out.MDFEXT_ICN_92_47_mV) > 0.0
    assert hasattr(out, 'MDNEXT_ICN_92_46_mV')
    assert float(out.MDNEXT_ICN_92_46_mV) > 0.0
    assert hasattr(out, 'ICN_mV')
    assert float(out.ICN_mV) > 0.0


# ---------------------------------------------------------------------------
# Test 6 – NOISE channel is skipped; ICN_mV unchanged vs FEXT-only case
# ---------------------------------------------------------------------------

def test_noise_channel_skipped():
    """NOISE channel type is skipped — ICN_mV same whether or not NOISE is present."""
    OP = _make_op()

    out1 = SimpleNamespace()
    FD_Processing([_make_ch('THRU'), _make_ch('FEXT')], out1,
                   _make_param(n_chan=2), OP,
                   DO_ONCE=True, _get_ILN_fn=_stub_get_ILN)

    out2 = SimpleNamespace()
    FD_Processing([_make_ch('THRU'), _make_ch('FEXT'), _make_ch('NOISE')], out2,
                   _make_param(n_chan=3), OP,
                   DO_ONCE=True, _get_ILN_fn=_stub_get_ILN)

    assert hasattr(out1, 'ICN_mV')
    assert hasattr(out2, 'ICN_mV')
    assert float(out1.ICN_mV) == pytest.approx(float(out2.ICN_mV), rel=1e-9), \
        "NOISE channel should not change ICN_mV"


# ===========================================================================
# Oracle-backed cases.
#
# The tests above were written from a READING of the MATLAB and check shapes
# and signs; they cannot see an epsilon floor, an interp1 that returns NaN
# where np.interp clamps, or a loop that runs one channel too far.  Everything
# below was produced by EXECUTING FD_Processing under Octave
# (tools/octave_oracle.py) on these exact inputs and pinning what came back.
#
# get_ILN is injected as a deterministic stub on BOTH sides -- the same
# `efit = 0.001*k - 0.05`, `ILN = 0.002*cos(k/3) + 0.0007*k` file was dropped
# into the Octave workdir.  The real get_ILN forms ((fmbg'*fmbg)^-1)*fmbg'*LGw,
# an explicit inverse of a badly conditioned normal-equations matrix, and
# disagreed between Octave and numpy at 1e-10 relative -- a get_ILN question,
# not an FD_Processing one, and one that would have forced a tolerance wide
# enough to hide everything below.  The S-parameters are REAL-valued so no
# abs() of a complex number is involved.
# ===========================================================================

from com_functions.fn.FD_Processing.py_impl import _W

_ONF = 32
_OFB = 53.125e9
_OUI = 1.0 / _OFB
_OFTR = 0.75 * _OFB / 2

# COM Octave: W(linspace(0,60e9,32), 0.75*fb/2, 0.75*fb, 53.125e9) with
# Sinc = @(x) sin(pi*x+eps(0))./(pi*x+eps(0)).
_O_W_REF = [1.8823529411764707e-11, 1.8739805380915194e-11, 1.8470696874931067e-11, 1.7965629028353138e-11, 1.715331345112932e-11, 1.5966965447524548e-11, 1.4382822339225506e-11, 1.2457677684578559e-11, 1.0337362532306354e-11, 8.2181934208305187e-12, 6.2804674208248324e-12, 4.6375046979259491e-12, 3.325344582252627e-12, 2.3234833771681862e-12, 1.5835326765112696e-12, 1.0508965934238107e-12, 6.7623754614425606e-13, 4.1931804918308166e-13, 2.4872270770294095e-13, 1.4009028586117752e-13, 7.441793337430574e-14, 3.7041419191360442e-14, 1.7132971947097691e-14, 7.2582942626526725e-15, 2.734459661090145e-15, 8.5681599604868903e-16, 1.855228449370476e-16, 1.0972636979600937e-17, 1.0308866302989359e-17, 5.0482489520230776e-17, 8.4784415168654998e-17, 1.0230649927995059e-16]


def _ofx(faxis=None):
    """The exact vectors handed to Octave."""
    f = _make_oracle_faxis() if faxis is None else np.asarray(faxis, float)
    thru = 0.82 * np.exp(-f / 7.3e10) - 0.004 * np.cos(f / 9e9)
    return dict(
        faxis=f,
        thru=thru,
        fext=0.021 * (1.0 + f / 1.1e11) + 0.0013 * np.sin(f / 1.3e10),
        nxt=0.034 * np.exp(-f / 2.9e11) - 0.0009 * np.cos(f / 8e9),
        scd=0.011 * (1.0 + f / 2.4e11),
        sdc=0.007 * (1.0 + f / 1.7e11),
        nodie=thru * 0.93 + 0.002,
        p2p=thru * 0.88 + 0.003,
        raw=thru * 1.05 - 0.001,
        orig=thru * 0.97,
    )


def _make_oracle_faxis():
    return np.linspace(0.0, 60e9, _ONF)


def _oracle_param(n_chan=3, faxis=None, **kw):
    fax = _make_oracle_faxis() if faxis is None else np.asarray(faxis, float)
    p = SimpleNamespace(
        package_testcase_i=1, number_of_s4p_files=n_chan,
        a_thru=np.array([0.4, 0.6]), a_fext=np.array([0.22, 0.31]),
        a_next=np.array([0.17, 0.27]), a_icn_fext=0.19, a_icn_next=0.13,
        f1=float(fax[3]), f2=float(fax[-4]), f2_ild=float(fax[-7]),
        fb=_OFB, f_r=0.75, ui=_OUI, samples_per_ui=8,
        sample_dt=1.0 / (8 * _OFB),
        sigma_X=0.4082482904638631, P_peak=1e-4, Tx_rd_sel=1, base='x')
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _oracle_op(**kw):
    o = SimpleNamespace(WC_PORTZ=False, TDMODE=False, GET_FD=True,
                        INCLUDE_FILTER=False, COMPUTE_TDILN=False,
                        COMPUTE_RILN=False, include_pcb=False,
                        pkg_len_select=np.array([1, 2]),
                        DEBUG=False, DISPLAY_WINDOW=False)
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def _oracle_chdata(types=('THRU', 'FEXT', 'NEXT'), f=None):
    f = f or _ofx()
    sig = {'THRU': 'thru', 'FEXT': 'fext', 'NEXT': 'nxt', 'NOISE': 'nxt'}
    return [SimpleNamespace(
        type=t, faxis=f['faxis'].copy(), ftr=_OFTR,
        sdd21f=f[sig[t]].astype(complex), sdd21=f[sig[t]].astype(complex),
        sdd21_orig=f['orig'].astype(complex), sdd21_raw=f['raw'].astype(complex),
        sdd21p=f['p2p'].astype(complex), sdd21p_nodie=f['nodie'].astype(complex),
        scd21_orig=f['scd'].astype(complex), sdc21_orig=f['sdc'].astype(complex))
        for t in types]


def _oracle_ILN(sdd21, faxis):
    """The stub dropped into the Octave workdir, transcribed."""
    k = np.arange(1, len(np.asarray(sdd21).ravel()) + 1)
    return 0.002 * np.cos(k / 3.0) + 0.0007 * k, 0.001 * k - 0.05


def _oracle_ILN_td(sdd21, faxis, OP, param, A):
    k = np.arange(1, len(np.asarray(sdd21).ravel()) + 1)
    return 0.003 * k, 0.004 * k, SimpleNamespace(SNR_ISI_FOM_PDF=3.25, FOM=0.125)


def _oracle_RIL(chdata):
    k = np.arange(1, len(np.asarray(chdata[0].faxis).ravel()) + 1)
    return SimpleNamespace(RILN_dB=0.05 * np.cos(k / 5.0) + 0.01 * k)


def _orun(types=('THRU', 'FEXT', 'NEXT'), f=None, param=None, op=None,
          n_chan=None):
    chd = _oracle_chdata(types, f)
    fax = (f or _ofx())['faxis']
    return FD_Processing(
        chd, SimpleNamespace(),
        param if param is not None else _oracle_param(n_chan or len(types), fax),
        op if op is not None else _oracle_op(), DO_ONCE=True,
        _get_ILN_fn=_oracle_ILN, _get_ILN_cmp_td_fn=_oracle_ILN_td,
        _capture_RIL_RILN_fn=_oracle_RIL)


def _eq(got, want, what, tol=2e-15):
    g = float(np.atleast_1d(np.asarray(got, dtype=float)).ravel()[0])
    if np.isnan(want):
        assert np.isnan(g), '%s: got %r, reference NaN' % (what, g)
        return
    if np.isinf(want):
        assert g == want, '%s: got %r, reference %r' % (what, g, want)
        return
    assert abs(g - want) <= tol * max(abs(want), 1e-300), \
        '%s: got %.17g, reference %.17g' % (what, g, want)


# ---------------------------------------------------------------------------
# COM Octave (FD_Processing, 4p16p0): THRU+FEXT+NEXT on faxis=linspace(0,60e9,32),
# f1=faxis(4), f2=faxis(29), f2_ild=faxis(26), pkg_len_select=[1 2],
# package_testcase_i=1 so A_thru=0.4 / A_fext=0.22 / A_next=0.17.
# ---------------------------------------------------------------------------
_O_P_SIGNAL = 0.2167448649982045
_O_P_SIGMA = 0.46555865903042176
_O_SCMR_CD = 31.875576009091141
_O_SCMR_DC = 35.618511112766654
_O_FOM_ILD = 0.0017878827968865424
_O_ICN_MV = 4.2334999875454953
_O_MDFEXT_ICN = 3.1327406917545484
_O_MDNEXT_ICN = 2.8475354085196098
_O_SNR_MDFEXT = 36.797583729924369
_O_IL_AT_FNQ = 4.8248479324064171
_O_FIT_IL_AT_FNQ = 0.035276041666666667
_O_FIT_IL_AT_F2ILD = 0.027000000000000003
_O_VTF = 4.8248479324064171
_O_D2D = 5.4226936880171497
_O_VIP_VMP = 5.8837383424560601
_O_DELTA_F = 1935483870.9677429


def test_oracle_fd_baseline():
    chd, out = _orun()
    _eq(out.P_signal_FD, _O_P_SIGNAL, 'P_signal_FD')
    _eq(out.P_signal_sigma_FD, _O_P_SIGMA, 'P_signal_sigma_FD')
    _eq(out.SCMR_FD_CD_ch_dB, _O_SCMR_CD, 'SCMR_FD_CD_ch_dB')
    _eq(out.SCMR_FD_DC_ch_dB, _O_SCMR_DC, 'SCMR_FD_DC_ch_dB')
    _eq(out.FOM_ILD, _O_FOM_ILD, 'FOM_ILD')
    _eq(out.ICN_mV, _O_ICN_MV, 'ICN_mV')
    _eq(out.MDFEXT_ICN_92_47_mV, _O_MDFEXT_ICN, 'MDFEXT_ICN_92_47_mV')
    _eq(out.MDNEXT_ICN_92_46_mV, _O_MDNEXT_ICN, 'MDNEXT_ICN_92_46_mV')
    _eq(out.SNR_MDFEXT, _O_SNR_MDFEXT, 'SNR_MDFEXT')
    _eq(out.IL_dB_channel_only_at_Fnq, _O_IL_AT_FNQ, 'IL_dB_channel_only_at_Fnq')
    _eq(out.fitted_IL_dB_at_Fnq, _O_FIT_IL_AT_FNQ, 'fitted_IL_dB_at_Fnq')
    _eq(out.fitted_IL_dB_at_F2_ild, _O_FIT_IL_AT_F2ILD, 'fitted_IL_dB_at_F2_ild')
    _eq(out.VTF_loss_dB_at_Fnq, _O_VTF, 'VTF_loss_dB_at_Fnq')
    _eq(out.IL_db_die_to_die_at_Fnq, _O_D2D, 'IL_db_die_to_die_at_Fnq')
    _eq(out.VIP_to_VMP_IL_dB_at_Fnq, _O_VIP_VMP, 'VIP_to_VMP_IL_dB_at_Fnq')
    _eq(chd[0].delta_f, _O_DELTA_F, 'chdata(1).delta_f')
    _eq(chd[0].A, 0.4, 'chdata(1).A')
    _eq(chd[1].Aicn, 0.19, 'chdata(2).Aicn')
    _eq(chd[2].Aicn, 0.13, 'chdata(3).Aicn')


def test_W_is_bit_identical_to_the_reference():
    """eq 93A-57 transcribed operation for operation, not merely to 1e-12.

    COM Octave, W(linspace(0,60e9,32), 0.75*fb/2, 0.75*fb, fb) with
    Sinc = @(x) sin(pi*x+eps(0))./(pi*x+eps(0)).  The earlier spelling --
    np.sinc(f/(fb+1e-300))**2/(fb+1e-300)/... -- differed in 15 of these 32
    samples at the last bit, because `1/fb * A` is not the same double as
    `A / fb`.
    """
    ref = np.array(_O_W_REF)
    got = np.asarray(_W(np.linspace(0.0, 60e9, _ONF), _OFTR, 0.75 * _OFB, _OFB),
                     dtype=float)
    # The pins were taken on Windows, where numpy's sin agrees with Octave's to
    # the last bit on all 32 samples. Another platform's libm may round a
    # sample differently (glibc does: index 7, 2 ulp), so elsewhere bound it at
    # 1e-15 relative and at most 2 samples; the defect this guards differed in
    # 15 of the 32.
    if os.name != 'nt':
        assert np.all(np.abs(got - ref) <= 1e-15 * np.abs(ref)), 'W beyond 1e-15'
        assert np.count_nonzero(got != ref) <= 2, 'W off the reference form'
        return
    bad = np.nonzero(got != ref)[0]
    assert bad.size == 0, (
        'W differs from the reference at %d of %d samples (first at index %d: '
        '%.17g vs %.17g)' % (bad.size, ref.size, bad[0] if bad.size else -1,
                             got[bad[0]] if bad.size else 0.0,
                             ref[bad[0]] if bad.size else 0.0))


# ---------------------------------------------------------------------------
# COM Octave (FD_Processing, 4p16p0): scd21_orig and sdc21_orig identically
# zero.  EC_CM_*_PWR_RMS is then 0 and 10*log10(P_signal/0) is +Inf.  The port
# divided by (EC + 1e-300) and reported 2993.3594881720082 dB.
# ---------------------------------------------------------------------------
def test_oracle_perfectly_balanced_channel_gives_inf_SCMR():
    f = _ofx()
    f['scd'] = np.zeros(_ONF)
    f['sdc'] = np.zeros(_ONF)
    _, out = _orun(f=f)
    assert out.SCMR_FD_CD_ch_dB == float('inf'), \
        'SCMR_FD_CD_ch_dB=%r; the reference gives Inf, not a plausible ~2993 dB' \
        % out.SCMR_FD_CD_ch_dB
    assert out.SCMR_FD_DC_ch_dB == float('inf'), out.SCMR_FD_DC_ch_dB
    _eq(out.P_signal_FD, _O_P_SIGNAL, 'P_signal_FD')


# ---------------------------------------------------------------------------
# COM Octave (FD_Processing, 4p16p0): FEXT sdd21f identically zero, so
# sqrd_mdfext is 0 and SNR_MDFEXT is +Inf (MDFEXT_ICN stays 0 and the NEXT-only
# ICN is unchanged).  The port reported 2993.3594881720082 dB.
# ---------------------------------------------------------------------------
_O_ICN_NEXT_ONLY = 2.8475354085196098


def test_oracle_no_fext_coupling_gives_inf_SNR_MDFEXT():
    f = _ofx()
    f['fext'] = np.zeros(_ONF)
    _, out = _orun(f=f)
    assert out.SNR_MDFEXT == float('inf'), \
        'SNR_MDFEXT=%r; the reference gives Inf' % out.SNR_MDFEXT
    _eq(out.MDFEXT_ICN_92_47_mV, 0.0, 'MDFEXT_ICN_92_47_mV')
    _eq(out.ICN_mV, _O_ICN_NEXT_ONLY, 'ICN_mV')


# ---------------------------------------------------------------------------
# COM Octave (FD_Processing, 4p16p0): a Nyquist query outside the frequency
# axis.  faxis = linspace(30e9,60e9,32) while fnq = fb/2 = 26.5625 GHz, so
# every interp1 in the function is asked for a point below faxis(1) and returns
# NaN.  np.interp CLAMPS and gave the first bin's loss (1.766, 2.374, 2.840 dB
# and a fitted 0.049) instead.
# ---------------------------------------------------------------------------
def test_oracle_nyquist_above_band_is_nan_not_clamped():
    fax = np.linspace(30e9, 60e9, _ONF)
    f = _ofx(fax)
    _, out = _orun(f=f, param=_oracle_param(3, fax))
    for name in ('IL_dB_channel_only_at_Fnq', 'fitted_IL_dB_at_Fnq',
                 'VTF_loss_dB_at_Fnq', 'IL_db_die_to_die_at_Fnq',
                 'VIP_to_VMP_IL_dB_at_Fnq'):
        v = float(np.atleast_1d(np.asarray(getattr(out, name), float)).ravel()[0])
        assert np.isnan(v), \
            '%s=%r; interp1 returns NaN outside [faxis(1), faxis(end)] -- ' \
            'np.interp clamps to the edge value' % (name, v)


# ---------------------------------------------------------------------------
# COM Octave (FD_Processing, 4p16p0): a zero S-parameter bin next to Nyquist.
# faxis = linspace(0,60e9,32) brackets fnq=26.5625 GHz between index 14 and 15
# (1-based).  -20*log10(0) is +Inf there, and interp1's `s*dy(k)+y(k)` gives
#   Inf when the Inf is the UPPER bracket (s*Inf + finite), and
#   NaN when it is the LOWER bracket (s*(-Inf) + Inf).
# The port's `|x| + 1e-300` made both finite (4345.04 dB), and np.interp's own
# NaN-avoidance retry turned the second case into +Inf.
# ---------------------------------------------------------------------------
def test_oracle_zero_bin_above_nyquist_gives_inf():
    f = _ofx()
    t = f['thru'].copy(); t[14] = 0.0
    f['thru'] = t
    p = f['p2p'].copy(); p[14] = 0.0
    f['p2p'] = p
    _, out = _orun(f=f)
    for name in ('IL_dB_channel_only_at_Fnq', 'VTF_loss_dB_at_Fnq',
                 'VIP_to_VMP_IL_dB_at_Fnq'):
        v = float(np.asarray(getattr(out, name), float).ravel()[0])
        assert v == float('inf'), \
            '%s=%r; -20*log10(0) is +Inf in the reference, not ~4345 dB' % (name, v)


def test_oracle_zero_bin_below_nyquist_gives_nan():
    f = _ofx()
    t = f['thru'].copy(); t[13] = 0.0
    f['thru'] = t
    p = f['p2p'].copy(); p[13] = 0.0
    f['p2p'] = p
    _, out = _orun(f=f)
    for name in ('IL_dB_channel_only_at_Fnq', 'VTF_loss_dB_at_Fnq',
                 'VIP_to_VMP_IL_dB_at_Fnq'):
        v = float(np.asarray(getattr(out, name), float).ravel()[0])
        assert np.isnan(v), \
            '%s=%r; interp1 gives NaN when the LOWER bracket is Inf ' \
            '(np.interp retries from the right and returns Inf)' % (name, v)


# ---------------------------------------------------------------------------
# COM Octave (FD_Processing, 4p16p0): param.number_of_s4p_files=2 with three
# channels present.  The reference stops after the FEXT, so the NEXT never
# enters the PSXT power sum and MDNEXT_ICN_92_46_mV is never written at all.
# The port iterated every element of chdata and reported ICN_mV 4.2335.
# ---------------------------------------------------------------------------
_O_ICN_FEXT_ONLY = 3.1327406917545484


def test_oracle_respects_number_of_s4p_files():
    chd = _oracle_chdata(('THRU', 'FEXT', 'NEXT'))
    _, out = FD_Processing(chd, SimpleNamespace(),
                           _oracle_param(2, _make_oracle_faxis()), _oracle_op(),
                           DO_ONCE=True, _get_ILN_fn=_oracle_ILN)
    _eq(out.ICN_mV, _O_ICN_FEXT_ONLY, 'ICN_mV')
    _eq(out.MDFEXT_ICN_92_47_mV, _O_MDFEXT_ICN, 'MDFEXT_ICN_92_47_mV')
    assert not hasattr(out, 'MDNEXT_ICN_92_46_mV'), \
        'the third channel is past number_of_s4p_files and must not be processed'
    _eq(out.VTF_loss_dB_at_Fnq, _O_VTF, 'VTF_loss_dB_at_Fnq')


# ---------------------------------------------------------------------------
# COM Octave (FD_Processing, 4p16p0): OP.COMPUTE_RILN / OP.COMPUTE_TDILN with
# capture_RIL_RILN and get_ILN_cmp_td stubbed identically on both sides.
# FOM_RILN sums PWF(index_f1:index_f2-1), i.e. it stops one bin SHORT of f2.
# ---------------------------------------------------------------------------
_O_FOM_RILN = 0.040200291687977445


def test_oracle_riln_and_tdiln():
    _, out = _orun(op=_oracle_op(COMPUTE_RILN=True, COMPUTE_TDILN=True))
    _eq(out.FOM_RILN, _O_FOM_RILN, 'FOM_RILN')
    _eq(out.FOM_TDILN, 3.25, 'FOM_TDILN')
    _eq(out.FOM_ILD, _O_FOM_ILD, 'FOM_ILD')


# ---------------------------------------------------------------------------
# COM Octave (FD_Processing, 4p16p0): OP.WC_PORTZ=1 with param.Tx_rd_sel=2
# (A_thru = a_thru(2) = 0.6), OP.include_pcb=1, and a NOISE third channel that
# gets A=Aicn=1 and is skipped by the `continue`.
# ---------------------------------------------------------------------------
_O_CABLE_LOSS = 5.0894132470815201
_O_LOSS_WITH_PCB = 4.4154916519998713


def test_oracle_wc_portz_include_pcb_and_noise_channel():
    chd, out = _orun(types=('THRU', 'FEXT', 'NOISE'),
                     param=_oracle_param(3, _make_oracle_faxis(), Tx_rd_sel=2),
                     op=_oracle_op(WC_PORTZ=True, include_pcb=True))
    _eq(chd[0].A, 0.6, 'chdata(1).A (Tx_rd_sel=2)')
    _eq(chd[2].A, 1.0, 'NOISE A')
    _eq(chd[2].Aicn, 1.0, 'NOISE Aicn')
    _eq(out.cable__assembley_loss, _O_CABLE_LOSS, 'cable__assembley_loss')
    _eq(out.loss_with_PCB, _O_LOSS_WITH_PCB, 'loss_with_PCB')
    _eq(out.ICN_mV, _O_ICN_FEXT_ONLY, 'ICN_mV (NOISE skipped)')
    _eq(out.FOM_ILD, _O_FOM_ILD, 'FOM_ILD')



# ---------------------------------------------------------------------------
# 4p17p0 L1868-1872: with OP.ACBW the THRU channel's apparent bandwidth is
# computed, get_ACBW(sdd21f, faxis/1e9, OP, param), and reported as
# output_args.ACBW_GHz. The fit goes into a stray variable in the reference
# (CICP_fit_chdata(i).db), so chdata gets no fit field. get_ACBW's own values are
# pinned against COM Octave in its test; this pins the call.
# ---------------------------------------------------------------------------
def test_acbw_is_called_for_the_thru_only_when_asked(monkeypatch):
    import com_functions.fn.FD_Processing.py_impl as fdp
    calls = []

    def spy(Hch, fGHz, OP, param):
        calls.append((np.asarray(Hch).copy(), np.asarray(fGHz).copy()))
        return 42.5, np.zeros(3), np.ones(3), np.full(3, 2.0), np.arange(4.0), np.zeros(3)
    monkeypatch.setattr(fdp, '_get_ACBW', spy)
    for acbw in (1, 0):
        calls.clear()
        param = _make_param(n_chan=2)
        OP = _make_op()
        OP.ACBW = acbw
        chdata = [_make_ch('THRU'), _make_ch('FEXT')]
        chdata, out = FD_Processing(chdata, SimpleNamespace(), param, OP, DO_ONCE=True,
                                    _get_ILN_fn=_stub_get_ILN)
        if acbw:
            assert len(calls) == 1, 'get_ACBW once, for the THRU'
            np.testing.assert_array_equal(calls[0][0], chdata[0].sdd21f)
            np.testing.assert_array_equal(calls[0][1], chdata[0].faxis / 1e9)
            assert out.ACBW_GHz == 42.5 and chdata[0].Bch_GHz == 42.5
            np.testing.assert_array_equal(chdata[0].CICP_alpha, np.arange(4.0))
            assert not hasattr(chdata[0], 'CICP_fit_db'), 'the reference stores no fit on chdata'
        else:
            assert not calls and not hasattr(out, 'ACBW_GHz')
