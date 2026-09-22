"""Verification tests for add_brd().

# ============================================================
# MATLAB GROUND TRUTH (lines 4768-4820)
# Adds board trace (TX+caps and RX+caps) to chdata S-params.
# include_pcb=1: tx || channel || rx cascade.
# include_pcb=2: channel || rx cascade.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.add_brd.py_impl import add_brd


def _freq(N=50):
    return np.linspace(0, 50e9, N)


def _chdata(ctype='THRU', N=50):
    f = _freq(N)
    cd = SimpleNamespace(
        type=ctype,
        faxis=f,
        sdd11_raw=np.zeros(N, dtype=complex),
        sdd12_raw=np.ones(N, dtype=complex) * 0.9,
        sdd21_raw=np.ones(N, dtype=complex) * 0.9,
        sdd22_raw=np.zeros(N, dtype=complex),
    )
    return cd


def _param():
    return SimpleNamespace(
        Z0=50.0,
        C_0=np.array([0.1e-12, 0.1e-12]),
        C_1=np.array([0.05e-12, 0.05e-12]),
        brd_Z_c=np.array([50.0, 50.0]),
        brd_gamma0_a1_a2=np.array([0.0, 0.01, 0.001]),
        brd_tau=0.0,
        z_bp_tx=0.01, z_bp_rx=0.01,
        z_bp_next=0.01, z_bp_fext=0.01,
    )


def _op(include_pcb=1):
    return SimpleNamespace(include_pcb=include_pcb)


def test_returns_four_arrays():
    """add_brd returns (s11, s12, s21, s22)."""
    result = add_brd(_chdata(), _param(), _op())
    assert len(result) == 4


def test_output_length_matches_freq():
    """Output arrays have same length as faxis."""
    N = 50
    s11, s12, s21, s22 = add_brd(_chdata(N=N), _param(), _op())
    assert len(s21) == N


def test_include_pcb2():
    """include_pcb=2: rx-only cascade returns valid arrays."""
    s11, s12, s21, s22 = add_brd(_chdata(), _param(), _op(include_pcb=2))
    assert len(s21) > 0


def test_matched_line_no_reflection():
    """Matched transmission line (Z_c=Z0, zero length) → s11 ≈ 0 at DC."""
    cd = _chdata()
    p = _param()
    p.brd_Z_c = np.array([50.0, 50.0])
    p.z_bp_tx = 0.0
    p.z_bp_rx = 0.0
    s11, s12, s21, s22 = add_brd(cd, p, _op())
    assert np.abs(s11[0]) < 0.1


def test_fext_type():
    """FEXT type uses z_bp_fext for TX side."""
    result = add_brd(_chdata(ctype='FEXT'), _param(), _op())
    assert len(result) == 4


# ============================================================
# COM Octave oracle values (2026-09-22)
# add_brd, synth_tline and combines4p extracted verbatim from the reference
# and run under Octave on a 5-point axis starting at DC.
#
# Pinned at rtol=1e-14, not bit-for-bit, and deliberately so: numpy and Octave
# disagree at 1 ULP on complex MULTIPLY and complex DIVIDE themselves --
# a bare `a.*b` on random complex vectors differed in 9 of 12 elements, and
# `a./b` in 6 of 12.  The arithmetic FORM here is token-for-token the
# reference's (unary minus first, then the left-to-right chain, and the same
# cascade order), and the residual after that is the two libraries' complex
# kernels, not a rearrangement.  Bit-exactness against Octave is unreachable
# for any function that multiplies or divides complex numbers.
# ============================================================

OCT_F = np.linspace(0.0, 40e9, 5)
OCT_S11 = np.array([0.05, 0.04 + 0.01j, -0.03 + 0.02j, 0.01 - 0.04j, -0.02 - 0.03j])
OCT_S12 = np.array([0.9, 0.8 - 0.2j, 0.6 - 0.4j, 0.4 - 0.5j, 0.2 - 0.6j])
OCT_S22 = np.array([0.04, 0.03 - 0.01j, -0.02 + 0.03j, 0.02 - 0.01j, -0.01 - 0.02j])


def _oct_chdata(ctype='THRU'):
    return SimpleNamespace(type=ctype, faxis=OCT_F, sdd11_raw=OCT_S11,
                           sdd12_raw=OCT_S12, sdd21_raw=OCT_S12,
                           sdd22_raw=OCT_S22)


def _oct_param():
    return SimpleNamespace(
        z_bp_tx=0.0254, z_bp_rx=0.0381, z_bp_next=0.0127, z_bp_fext=0.0508,
        Z0=50, C_0=[0.6e-12, 0.4e-12], C_1=[0.3e-12, 0.2e-12],
        brd_Z_c=[95, 105], brd_gamma0_a1_a2=[0, 1.734e-3, 1.455e-4],
        brd_tau=5.79e-9)


OCT_THRU_PCB1 = (
    [0.050000000000000003-5.1559416071812601e-26j, -0.71697642575468135-0.31499479823832482j, -0.79530888430289992-0.34430692829304937j, -0.94288996209131282-0.2417032342616868j, -0.97433342437025683-0.17902174933614795j],
    [0.90000000000000002-4.9252191060379891e-26j, -0.026209000362420933-0.40167969411892257j, -0.20078955794691405+0.014167120224578205j, -0.040230294459337471+0.031086159035835861j, -0.011650857735409326+0.02054517998546938j],
    [0.90000000000000002-4.9252191060379891e-26j, -0.026209000362420933-0.40167969411892257j, -0.20078955794691405+0.014167120224578205j, -0.040230294459337471+0.031086159035835861j, -0.011650857735409326+0.02054517998546938j],
    [0.040000000000000001-4.8061432755159741e-26j, -0.63380465723412283-0.22004715600756408j, -0.57772668596412746-0.45080056034888838j, -0.87518844424341224-0.35709830688424649j, -0.94370881182752353-0.26595502013299022j])

OCT_THRU_PCB2 = (
    [0.050000000000000003-1.6951040900321952e-26j, -0.39497682209362805-0.13168621057576049j, -0.39010246704843787+0.32566180557538066j, -0.035773253498531027+0.33564423094507284j, 0.21376384332741855+0.27968571387440444j],
    [0.90000000000000002-1.9587869484816477e-26j, 0.31261177268352497-0.498858601154133j, -0.040277706789730791-0.34523021293357981j, -0.11001988393156417-0.17777047653769909j, -0.13419823354223406-0.092063013386012749j],
    [0.90000000000000002-1.9587869484816477e-26j, 0.31261177268352497-0.498858601154133j, -0.040277706789730791-0.34523021293357981j, -0.11001988393156417-0.17777047653769909j, -0.13419823354223406-0.092063013386012749j],
    [0.040000000000000001-2.2634871404676816e-26j, -0.4747635651514423-0.51487061734201167j, -0.77208865029674967-0.41416941343671027j, -0.8910396077696634-0.31490366559716354j, -0.93420308045355449-0.2464071754075032j])

OCT_NEXT_PCB1 = (
    [0.050000000000000003-5.1559416071812601e-26j, -0.71695544131707833-0.31495016228522033j, -0.79525727557147607-0.34434945088593749j, -0.94285939592754631-0.24171584480105296j, -0.97429358499711582-0.17903110463645175j],
    [0.90000000000000002-4.9252191060379891e-26j, -0.026169413629413829-0.40171518802176204j, -0.20082887975807537+0.014173148323083834j, -0.040235942278404151+0.03108763126212236j, -0.011655409943754032+0.020545753008207941j],
    [0.90000000000000002-4.9252191060379891e-26j, -0.026169413629413829-0.40171518802176204j, -0.20082887975807537+0.014173148323083834j, -0.040235942278404151+0.03108763126212236j, -0.011655409943754032+0.020545753008207941j],
    [0.040000000000000001-4.8061432755159741e-26j, -0.63395008176677003-0.22009496559679692j, -0.57775016641577248-0.45075031954956779j, -0.87525600901699308-0.35708398797632862j, -0.94380242175351492-0.26593913858736229j])

OCT_FEXT_PCB1 = (
    [0.050000000000000003-5.1559416071812601e-26j, -0.7168611840703788-0.31496944125665871j, -0.79528249339339552-0.34433754002154493j, -0.94282531766472033-0.24171358411195584j, -0.97424772224011802-0.17903725631208192j],
    [0.90000000000000002-4.9252191060379891e-26j, -0.026288707018810344-0.40163732191366369j, -0.20073209966124517+0.014203262631416537j, -0.040207820394343508+0.031100078443160338j, -0.011629156944353122+0.020549296800130096j],
    [0.90000000000000002-4.9252191060379891e-26j, -0.026288707018810344-0.40163732191366369j, -0.20073209966124517+0.014203262631416537j, -0.040207820394343508+0.031100078443160338j, -0.011629156944353122+0.020549296800130096j],
    [0.040000000000000001-4.8061432755159741e-26j, -0.633734474540949-0.22008375110985068j, -0.57780527153534011-0.45076529898761275j, -0.87518725555638355-0.35708365003548243j, -0.9437032971125412-0.26595114457411623j])


@pytest.mark.parametrize('ctype,ipcb,want', [
    ('THRU', 1, OCT_THRU_PCB1),
    ('THRU', 2, OCT_THRU_PCB2),
    ('NEXT', 1, OCT_NEXT_PCB1),
    ('FEXT', 1, OCT_FEXT_PCB1),
])
def test_oracle_cascade(ctype, ipcb, want):
    """The three board-length mappings give three different cascades, so these
    also pin the switch: NEXT reads z_bp_rx/z_bp_next, FEXT z_bp_fext/z_bp_rx."""
    got = add_brd(_oct_chdata(ctype), _oct_param(), _op(ipcb))
    for g, w, name in zip(got, want, ('s11', 's12', 's21', 's22')):
        np.testing.assert_allclose(g, w, rtol=1e-14, atol=1e-25,
                                   err_msg='%s %s pcb=%d' % (name, ctype, ipcb))


def test_oracle_noise_type_is_the_thru_mapping():
    """case {'THRU' 'NOISE'} share a branch, so NOISE must match THRU."""
    a = add_brd(_oct_chdata('THRU'), _oct_param(), _op(1))
    b = add_brd(_oct_chdata('NOISE'), _oct_param(), _op(1))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_oracle_include_pcb_zero_has_no_case():
    """The switch has no otherwise, so nothing is assigned.

    COM Octave, OP.include_pcb=0: "error: element number 1 undefined in
    return list".
    """
    with pytest.raises(ValueError, match='include_pcb'):
        add_brd(_oct_chdata(), _oct_param(), _op(0))


def test_oracle_unknown_channel_type_has_no_case():
    """COM Octave, chdata.type='BOGUS': "error: 'z_bp_tx' undefined"."""
    with pytest.raises(ValueError, match='type'):
        add_brd(_oct_chdata('BOGUS'), _oct_param(), _op(1))
