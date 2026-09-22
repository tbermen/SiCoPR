"""Tests for force (RxFFE tap solver).

MATLAB GROUND TRUTH:
  force(V, param, OP, ix, C=[]) solves for FFE taps.
  With pre-set C: Vfiltered = FFE(C, cmx, spui, V).
  With C=[]: solves VV*C = FV where FV[cmx]=V[ix].
  Tap constraint 'unity cursor': Cmod[cmx] = 1.
  FFE with 1 tap [1.0] at cmx=0: Vfiltered == V.
  FFE with cursor-only: Vfiltered ≈ V (identity filter).
"""
import pytest
import numpy as np
from types import SimpleNamespace
from com_functions.fn.force.py_impl import force, _FFE


def _make_param(cmx=0, cpx=0, spui=32, ndfe=0, bmax=None):
    p = SimpleNamespace()
    p.RxFFE_cmx = cmx
    p.RxFFE_cpx = cpx
    p.samples_per_ui = spui
    p.ndfe = ndfe
    p.bmax = bmax if bmax is not None else np.array([1.0])
    p.N_bg = 0
    p.RxFFE_stepz = 0
    return p


def _make_OP(constraint='unity cursor', method='FORCE'):
    op = SimpleNamespace()
    op.RXFFE_TAP_CONSTRAINT = constraint
    op.FFE_OPT_METHOD = method
    return op


def test_ffe_identity_single_tap():
    """FFE with single tap C=[1] and cmx=0: output equals input."""
    V = np.sin(np.linspace(0, 2 * np.pi, 64))
    out = _FFE(np.array([1.0]), 0, 1, V)
    assert np.allclose(out, V)


def test_ffe_passthrough_with_preset_C():
    """force with preset C=[1.0], cmx=0, cpx=0: Vfiltered = V."""
    V = np.random.randn(128)
    param = _make_param(cmx=0, cpx=0, spui=1)
    Vfiltered, Cmod, _ = force(V, param, _make_OP(), C=np.array([1.0]))
    assert np.allclose(Vfiltered, V)


def test_returns_three_outputs():
    """force always returns (Vfiltered, Cmod, idx)."""
    V = np.ones(64)
    param = _make_param(cmx=1, cpx=1, spui=4)
    result = force(V, param, _make_OP())
    assert len(result) == 3


def test_unity_cursor_constraint():
    """Tap constraint 'unity cursor': Cmod[cmx] should be 1."""
    np.random.seed(42)
    V = np.random.randn(128)
    param = _make_param(cmx=2, cpx=1, spui=8)
    op = _make_OP(constraint='unity cursor')
    # ix is given rather than left to argmax: a cursor in the last few UI
    # leaves too little pulse for the VV window, which MATLAB rejects. See
    # test_octave_vv_window_past_the_end_errors.
    _, Cmod, _ = force(V, param, op, ix=64)
    assert abs(float(Cmod[2]) - 1.0) < 1e-6


def test_cmod_length_matches_num_taps():
    """len(Cmod) = cmx + cpx + 1."""
    np.random.seed(7)
    V = np.random.randn(256)
    param = _make_param(cmx=2, cpx=3, spui=16)
    _, Cmod, _ = force(V, param, _make_OP(), ix=128)
    assert len(Cmod) == 6   # 2+3+1


def test_return_v_false_skips_filter():
    """return_V=0: Vfiltered is empty array."""
    np.random.seed(3)
    V = np.random.randn(64)
    param = _make_param(cmx=0, cpx=0, spui=4)
    Vfiltered, _, _ = force(V, param, _make_OP(), ix=32, return_V=0)
    assert len(Vfiltered) == 0


def test_wiener_hopf_raises():
    """FFE_OPT_METHOD='WIENER-HOPF' raises NotImplementedError."""
    np.random.seed(11)
    V = np.random.randn(64)
    param = _make_param(cmx=1, cpx=1, spui=4)
    op = _make_OP(method='WIENER-HOPF')
    with pytest.raises(NotImplementedError):
        force(V, param, op, ix=32)


# ============================================================
# COM Octave oracle values (2026-09-22)
# force run verbatim from octave/com_ieee8023_4p16p0_octave_compat.m via
# tools/octave_oracle.py, with findbankloc as its subfunction and return_V=0
# so the rewritten compat FFE is never reached. MATLAB's ix is 1-BASED and
# this port's is 0-based, so every ix below is the MATLAB one and the port is
# called with ix-1.
#
# Three divergences these pin:
#   1. the pre-cursor sample run used mod(ix_0based, spui) instead of
#      mod(ix_1based, spui), so whenever mod(ix,spui)==1 the first UI sample
#      was dropped and every tap moved -- by 674 in the worst case below;
#   2. `ix < length(V)` is on the 1-based ix, so the last sample of V takes
#      the other branch;
#   3. a VV window running past the end of vsampled is a subscript error in
#      MATLAB; the port zero-filled the column and solved anyway.
# ============================================================

SPUI_ORACLE = 4
# V is rebuilt from its formula rather than pasted: 40 and 64 doubles each.
_T40 = np.arange(40, dtype=float)
V_ORACLE_A = np.round(np.exp(-((_T40 - 17.0) ** 2) / 12.0)
                      + 0.4 * np.exp(-_T40 / 6.0), 6)
_T64 = np.arange(64, dtype=float)
V_ORACLE_B = np.round(np.exp(-((_T64 - 21.0) ** 2) / 10.0)
                      + 0.35 * np.exp(-_T64 / 9.0)
                      + 0.12 * np.exp(-((_T64 - 38.0) ** 2) / 40.0), 6)


def _oracle_param(**over):
    p = SimpleNamespace(RxFFE_cmx=2, RxFFE_cpx=3, N_bg=0, N_bmax=6,
                        RxFFE_stepz=0, ndfe=1, samples_per_ui=SPUI_ORACLE,
                        bmax=np.array([0.7, 0.2]), N_tail_start=4, N_bf=2,
                        bmaxg=0.5, current_ffegain=0)
    for k, v in over.items():
        setattr(p, k, v)
    return p


def _oracle_op(float_ctl='isi'):
    return SimpleNamespace(FFE_OPT_METHOD='FORCE', RXFFE_FLOAT_CTL=float_ctl,
                           RXFFE_TAP_CONSTRAINT='unity cursor')


@pytest.mark.parametrize('ix_matlab,Cmod', [
    (5, [-341.89160969271609, 175.53399024843429, 1, -635.91554204003887,
         673.14361134055139, -176.56717777892155]),
    (9, [-1.2321309683675987, -1.3081031001048375, 1, 0.38551959217519999,
         -5.1405548452839707, 0.96258011510098007]),
    (13, [-0.11512342612999953, 2.2116407114724996, 1, -1.0514608282384619,
          -0.44165659028491894, 5.0262359405496611]),
    (17, [-0.07475104980403105, -0.14468700427095862, 1,
          -0.038139197458714878, -0.0051933878153574342,
          -7.965862684300772e-08]),
])
def test_octave_precursor_sampling_at_mod_one(ix_matlab, Cmod):
    """COM Octave: every one of these has mod(ix,spui)==1, the case the
    0-based modulus got wrong. V = exp(-(t-17)^2/12) + 0.4*exp(-t/6),
    t = 0:39, rounded to 6 places, spui=4, cmx=2, cpx=3, ndfe=1."""
    _, got, idx = force(V_ORACLE_A, _oracle_param(), _oracle_op(),
                        ix_matlab - 1, None, 0)
    np.testing.assert_allclose(np.asarray(got).ravel(), Cmod, rtol=1e-8,
                               atol=1e-12)
    assert len(idx) == 0


def test_octave_ix_equal_to_length_takes_the_other_branch():
    """COM Octave, ix = length(V) = 64: MATLAB's `ix < length(V)` is false, so
    vsampled_raw is the plain V(mod(ix,spui):spui:end) run and ivs is the
    position of max(vsampled).
        Cmod = [-0.18067895556163135 0.11702634291529526 1
                -0.23555318964606944 0.20961948438175879 -0.23982944546538501]
    """
    _, got, _ = force(V_ORACLE_B, _oracle_param(), _oracle_op(), 64 - 1,
                      None, 0)
    np.testing.assert_allclose(
        np.asarray(got).ravel(),
        [-0.18067895556163135, 0.11702634291529526, 1, -0.23555318964606944,
         0.20961948438175879, -0.23982944546538501], rtol=1e-8, atol=1e-12)


def test_octave_vv_window_past_the_end_errors():
    """COM Octave, ix=63 with length(V)=64:
        error: vsampled(26): out of bound 25 (dimensions are 1x25)
    The port zero-filled the column and solved on a VV the reference never
    builds."""
    with pytest.raises(IndexError):
        force(V_ORACLE_B, _oracle_param(), _oracle_op(), 63 - 1, None, 0)


@pytest.mark.parametrize('float_ctl,Cmod,idx', [
    ('isi',
     [-0.026586629196229745, -0.2150046200966502, 1, 0.021491770426800675,
      0.0078667065079489635, -0.020066175642932692, -0.080817572042429328,
      -0.074493308939815234, 0, 0.0019265632221625608, 0.011330081846410604,
      0, 0],
     [4, 5, 7, 8]),
    ('taps',
     [-0.026586629196229745, -0.2150046200966502, 1, 0.021491770426800675,
      0.0078667065079489635, -0.020066175642932692, 0, 0,
      -0.022543568376185742, 0.0019265632221625608, 0.011330081846410604,
      0.013563725722441812, 0],
     [6, 7, 8, 9]),
])
def test_octave_floating_tap_banks(float_ctl, Cmod, idx):
    """COM Octave, N_bg=2, N_bf=2, N_bmax=10, N_tail_start=4, bmaxg=0.5,
    ix=22: the bank positions and the zeroed non-bank taps."""
    p = _oracle_param(N_bg=2, N_bmax=10)
    _, got, got_idx = force(V_ORACLE_B, p, _oracle_op(float_ctl), 22 - 1,
                            None, 0)
    np.testing.assert_allclose(np.asarray(got).ravel(), Cmod, rtol=1e-8,
                               atol=1e-12)
    np.testing.assert_array_equal(np.asarray(got_idx).ravel(), idx)


def test_octave_tap_quantisation():
    """COM Octave, RxFFE_stepz=0.03125, ix=22:
        Cmod = [-0 -0.21875 1 -0 -0 -0.03125]"""
    p = _oracle_param(RxFFE_stepz=0.03125)
    _, got, _ = force(V_ORACLE_B, p, _oracle_op(), 22 - 1, None, 0)
    np.testing.assert_allclose(np.asarray(got).ravel(),
                               [0, -0.21875, 1, 0, 0, -0.03125], atol=1e-15)
