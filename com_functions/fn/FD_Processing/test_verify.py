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


def _make_param(fb=53.125e9, N=64):
    faxis = _make_faxis(N, fb)
    return SimpleNamespace(
        package_testcase_i=1,
        number_of_s4p_files=1,
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
    param = _make_param()
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
    param = _make_param()
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
    param = _make_param()
    OP = _make_op()

    out1 = SimpleNamespace()
    FD_Processing([_make_ch('THRU'), _make_ch('FEXT')], out1, param, OP,
                   DO_ONCE=True, _get_ILN_fn=_stub_get_ILN)

    out2 = SimpleNamespace()
    FD_Processing([_make_ch('THRU'), _make_ch('FEXT'), _make_ch('NOISE')], out2, param, OP,
                   DO_ONCE=True, _get_ILN_fn=_stub_get_ILN)

    assert hasattr(out1, 'ICN_mV')
    assert hasattr(out2, 'ICN_mV')
    assert float(out1.ICN_mV) == pytest.approx(float(out2.ICN_mV), rel=1e-9), \
        "NOISE channel should not change ICN_mV"
