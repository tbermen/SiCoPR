"""Verification tests for OptFom_Update_Best_Setttings() (note: MATLAB typo).

# ============================================================
# MATLAB GROUND TRUTH (lines 3935-3975)
# Copies THIS/param fields to BEST. Computes IR via FFE when not TDMODE.
# ============================================================
"""
import numpy as np
from types import SimpleNamespace
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Update_Best_Setttings.py_impl import OptFom_Update_Best_Setttings


def _THIS():
    return SimpleNamespace(
        txffe=[0, 0, 1, 0, 0],
        tx_index_vector=[0, 0, 1, 0, 0],
        ctle_index=1,
        g_dc=-3.0,
        g_LP_index=0,
        FOM=-25.0,
        cursor_i=40,
        itick=5,
        sigma_N=0.01,
        h_J=0.001,
        A_s=0.9,
        A_p=1.0,
        ISI_N=0.05,
        tail_RSS=0.001,
        dfetaps=np.array([0.1, 0.05]),
    )


def _param(M=8):
    return SimpleNamespace(
        current_ffegain=1.0,
        cursor_index=5,
        samples_per_ui=M,
        use_bmax=[1.0],
        use_bmin=[-1.0],
        Floating_DFE=False,
        Floating_RXFFE=False,
    )


def _op(tdmode=False, rxffe=False):
    return SimpleNamespace(TDMODE=tdmode, RxFFE=rxffe, RxFFE_with_MMSE=False)


def _chdata(M=8):
    ir = np.zeros(200)
    ir[40] = 1.0
    return [SimpleNamespace(ctle_imp_response=ir)]


def test_fom_set():
    """BEST.FOM is updated from THIS.FOM."""
    BEST = SimpleNamespace()
    result = OptFom_Update_Best_Setttings(BEST, _THIS(), np.zeros(200), _chdata(), _param(), _op())
    assert result.FOM == _THIS().FOM


def test_txffe_set():
    """BEST.txffe is copied from THIS."""
    BEST = SimpleNamespace()
    result = OptFom_Update_Best_Setttings(BEST, _THIS(), np.zeros(200), _chdata(), _param(), _op())
    assert result.txffe == _THIS().txffe


def test_ir_set_when_not_tdmode():
    """BEST.IR is computed via FFE when not TDMODE."""
    BEST = SimpleNamespace()
    result = OptFom_Update_Best_Setttings(BEST, _THIS(), np.zeros(200), _chdata(), _param(), _op(tdmode=False))
    assert hasattr(result, 'IR')
    assert len(result.IR) == 200


def test_ir_not_set_in_tdmode():
    """BEST.IR is not set when TDMODE=True."""
    BEST = SimpleNamespace()
    result = OptFom_Update_Best_Setttings(BEST, _THIS(), np.zeros(200), _chdata(), _param(), _op(tdmode=True))
    assert not hasattr(result, 'IR')


def test_sigma_n_set():
    """BEST.sigma_N is copied from THIS."""
    BEST = SimpleNamespace()
    result = OptFom_Update_Best_Setttings(BEST, _THIS(), np.zeros(200), _chdata(), _param(), _op())
    assert result.sigma_N == _THIS().sigma_N


# ============================================================
# COM Octave 4p16p0 oracle pins.
# OptFom_Update_Best_Setttings extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m), with FFE taken from
# matlab/com_ieee8023_4p16p0.m because the compat FFE is a rewritten speed
# variant.  Fixtures are the ones above: M=8, chdata(1).ctle_imp_response
# 200 samples, THIS.cursor_i=40, param.use_bmax=[1], use_bmin=[-1].
#
# Everything else in this function is a plain field copy, so what is worth
# pinning is BEST.IR = FFE(THIS.txffe, cursor_index-1, M, ctle_imp_response),
# which carries both the tap-to-sample spacing and the cursor offset.
# ============================================================
import pytest


def _decaying_chdata(N=200):
    """A response with a tail, so every tap contributes to every sample."""
    ir = np.concatenate([np.zeros(30), np.exp(-np.arange(40) / 6.0),
                         np.zeros(N - 70)])
    return [SimpleNamespace(ctle_imp_response=ir)]


_TXFFE5 = [-0.05, 0.2, 1.0, -0.15, 0.03]

# BEST.IR for _decaying_chdata with _TXFFE5 and cursor_index=5
_OCT_IR_SUM = 6.7007604563798742
_OCT_IR_NORM = 1.950231717829163
_OCT_IR_HEAD = [-0.035826565528689465, -0.030326532985631673,
                -0.025670855951629601, -0.02172991042535391,
                -0.018393972058572117, -0.015570161195729884]
_OCT_IR_30_42 = [0.063365610783497797, 0.053637831514762493,
                 0.045403444140008305, 0.038433185711608878,
                 0.032532989334203975, 0.027538580927464935,
                 0.023310905484520292, 0.019732255483298801,
                 0.016766625347552036, 0.01419264194479054,
                 0.012013812034181184, 0.010169472333205301]
_OCT_IR_60_66 = [-6.4277121887592882e-05, -5.4409409006413807e-05,
                 0.00014483849981494323, 0.000122603143153922,
                 0.00010378132009394275, 8.7848990844545657e-05]
_OCT_IR_ARGMAX = 14
_OCT_IR_MAX = 1.0492452550620053


def test_octave_IR_from_a_decaying_response():
    """COM Octave: BEST.IR for a five-tap Tx FFE over a decaying response.

    The whole 200-sample vector is pinned through its sum, its 2-norm, three
    windows and its peak, which together fix both the tap spacing (M samples)
    and the cursor offset (cursor_index-1 precursor taps).
    """
    THIS = _THIS()
    THIS.txffe = list(_TXFFE5)
    out = OptFom_Update_Best_Setttings(SimpleNamespace(), THIS, np.zeros(200),
                                       _decaying_chdata(), _param(), _op())
    IR = np.asarray(out.IR).ravel()
    assert len(IR) == 200
    assert float(np.sum(IR)) == pytest.approx(_OCT_IR_SUM, rel=1e-13)
    assert float(np.linalg.norm(IR)) == pytest.approx(_OCT_IR_NORM, rel=1e-13)
    np.testing.assert_allclose(IR[0:6], _OCT_IR_HEAD, rtol=1e-13)
    np.testing.assert_allclose(IR[30:42], _OCT_IR_30_42, rtol=1e-13)
    np.testing.assert_allclose(IR[60:66], _OCT_IR_60_66, rtol=1e-13)
    assert int(np.argmax(IR)) == _OCT_IR_ARGMAX
    assert float(np.max(IR)) == pytest.approx(_OCT_IR_MAX, rel=1e-13)


def test_octave_IR_tap_positions_for_an_impulse_response():
    """COM Octave, an impulse at sample 40 with cursor_index=3: the taps land
    at 40 + (i-1-cmx)*M for cmx=2, i.e. samples 24, 32, 40, 48, 56, each
    carrying its own tap value."""
    THIS = _THIS()
    THIS.txffe = list(_TXFFE5)
    p = _param()
    p.cursor_index = 3
    out = OptFom_Update_Best_Setttings(SimpleNamespace(), THIS, np.zeros(200),
                                       _chdata(), p, _op())
    IR = np.asarray(out.IR).ravel()
    np.testing.assert_array_equal(np.nonzero(IR)[0], [24, 32, 40, 48, 56])
    np.testing.assert_allclose(IR[[24, 32, 40, 48, 56]], _TXFFE5, rtol=1e-15)


def test_octave_cursor_index_shifts_the_main_tap():
    """COM Octave: with cursor_index=5 the single unity tap at position 3 of
    five lands at sample 24, and with cursor_index=1 at sample 56."""
    out = OptFom_Update_Best_Setttings(SimpleNamespace(), _THIS(),
                                       np.zeros(200), _chdata(), _param(),
                                       _op())
    np.testing.assert_array_equal(np.nonzero(np.asarray(out.IR).ravel())[0],
                                  [24])
    p = _param()
    p.cursor_index = 1
    out = OptFom_Update_Best_Setttings(SimpleNamespace(), _THIS(),
                                       np.zeros(200), _chdata(), p, _op())
    np.testing.assert_array_equal(np.nonzero(np.asarray(out.IR).ravel())[0],
                                  [56])


def test_octave_all_zero_txffe_leaves_IR_a_scalar_zero():
    """COM Octave: FFE initialises V0=0 and skips every zero tap, so an
    all-zero Tx FFE returns the scalar 0, not a zero vector."""
    THIS = _THIS()
    THIS.txffe = [0.0] * 5
    out = OptFom_Update_Best_Setttings(SimpleNamespace(), THIS, np.zeros(200),
                                       _chdata(), _param(), _op())
    assert np.asarray(out.IR).size == 1
    assert float(np.asarray(out.IR).ravel()[0]) == 0.0


def _THIS_full():
    t = _THIS()
    t.floating_tap_locations = np.array([9.0, 14.0])
    t.floating_tap_coef = np.array([0.02, -0.01])
    t.C = np.array([0.1, 1.0, -0.2])
    t.PSD_results = SimpleNamespace(a=1)
    t.MMSE_results = SimpleNamespace(b=2)
    return t


def test_octave_optional_field_blocks():
    """COM Octave: Floating_DFE adds locations and coefficients,
    Floating_RXFFE adds locations only, and OP.RxFFE adds RxFFE, PSD_results
    and MMSE_results."""
    p = _param()
    p.Floating_DFE = True
    out = OptFom_Update_Best_Setttings(SimpleNamespace(), _THIS_full(),
                                       np.zeros(200), _chdata(), p, _op())
    assert hasattr(out, 'floating_tap_locations')
    assert hasattr(out, 'floating_tap_coef')

    p = _param()
    p.Floating_RXFFE = True
    out = OptFom_Update_Best_Setttings(SimpleNamespace(), _THIS_full(),
                                       np.zeros(200), _chdata(), p, _op())
    assert hasattr(out, 'floating_tap_locations')
    assert not hasattr(out, 'floating_tap_coef')

    out = OptFom_Update_Best_Setttings(SimpleNamespace(), _THIS_full(),
                                       np.zeros(200), _chdata(), _param(),
                                       _op(rxffe=True))
    assert hasattr(out, 'RxFFE')
    assert hasattr(out, 'PSD_results')
    assert hasattr(out, 'MMSE_results')
    out = OptFom_Update_Best_Setttings(SimpleNamespace(), _THIS_full(),
                                       np.zeros(200), _chdata(), _param(),
                                       _op())
    assert not hasattr(out, 'RxFFE')


def test_octave_copied_scalars():
    """COM Octave: the plain field copies, including the ISI_N -> ISI and
    g_LP_index -> G_high_pass renames and param's two bound vectors."""
    THIS = _THIS()
    out = OptFom_Update_Best_Setttings(SimpleNamespace(), THIS, np.zeros(200),
                                       _chdata(), _param(), _op())
    assert out.ffegain == 1.0
    assert out.ctle == THIS.ctle_index
    assert out.gdc == THIS.g_dc
    assert out.G_high_pass == THIS.g_LP_index
    assert out.FOM == THIS.FOM
    assert out.cursor_i == THIS.cursor_i
    assert out.itick == THIS.itick
    assert out.A_s == THIS.A_s
    assert out.A_p == THIS.A_p
    assert out.ISI == THIS.ISI_N
    assert out.tail_RSS == THIS.tail_RSS
    np.testing.assert_array_equal(out.bmax, [1.0])
    np.testing.assert_array_equal(out.bmin, [-1.0])
    np.testing.assert_array_equal(out.dfetaps, THIS.dfetaps)
