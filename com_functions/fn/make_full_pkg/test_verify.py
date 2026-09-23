"""Tests for make_full_pkg.

MATLAB GROUND TRUTH:
  Zero-length transmission line (Len=0): S11=0, S21=1 for lossless matched line.
  Single capacitor (Cpad>0, others 0): S11 = -jwCZ/(2+jwCZ), S21 = 2/(2+jwCZ).
  TX vs RX: uses different Pkg_len_TX vs Pkg_len_RX.
  DC mode: Z0/=2, Cpad*=2, Cball*=2, etc.
  Returns 4 arrays each of length len(faxis).
"""
import pytest
import numpy as np
from types import SimpleNamespace
from com_functions.fn.make_full_pkg.py_impl import make_full_pkg, _synth_tline, _combines4p


def _make_param(mele=1, Z0=100.0, Len_TX=0.0, Len_RX=0.0, Z_c=100.0,
                Cpad=0.0, Cball=0.0, Cbump=0.0, Lcomp=0.0, kappa1=1.0, kappa2=1.0):
    p = SimpleNamespace()
    p.Z0 = Z0
    p.C_diepad = np.array([Cpad, Cpad])
    p.C_pkg_board = np.array([Cball, Cball])
    p.L_comp = np.array([Lcomp, Lcomp])
    p.C_bump = np.array([Cbump, Cbump])
    if mele == 1:
        p.z_p_next_cases = np.array([[Z_c]])
        p.pkg_Z_c = np.array([Z_c, Z_c])
        p.Pkg_len_TX = np.array([Len_TX])
        p.Pkg_len_RX = np.array([Len_RX])
        p.Pkg_len_NEXT = np.array([Len_TX])
        p.Pkg_len_FEXT = np.array([Len_TX])
    p.pkg_tau = 0.0
    p.pkg_gamma0_a1_a2 = np.array([0.0, 0.0, 0.0])
    p.PKG_NAME = None
    p.kappa1 = kappa1
    p.kappa2 = kappa2
    return p


def test_synth_tline_zero_len_matched():
    """Zero length + matched impedance: S11=0, S21=1."""
    f = np.array([1e9, 5e9, 10e9])
    s11, s12, s21, s22 = _synth_tline(f, 100.0, 50.0, [0, 0, 0], 0, 0)
    assert np.allclose(s11, 0, atol=1e-10)
    assert np.allclose(np.abs(s21), 1.0, atol=1e-10)


def test_combines4p_identity_cascade():
    """Cascading two identity networks gives identity."""
    f = np.array([1e9, 5e9])
    s11 = np.zeros(2)
    s21 = np.ones(2)
    s11o, s12o, s21o, s22o = _combines4p(s11, s21, s21, s11, s11, s21, s21, s11)
    assert np.allclose(s21o, 1.0, atol=1e-10)
    assert np.allclose(s11o, 0.0, atol=1e-10)


def test_make_full_pkg_returns_four_arrays():
    faxis = np.linspace(1e9, 25e9, 50)
    param = _make_param()
    s11, s12, s21, s22 = make_full_pkg('TX', faxis, param, 'THRU')
    assert len(s11) == 50
    assert len(s21) == 50


def test_make_full_pkg_lossless_zero_len():
    """Zero-length lossless line + zero caps: S21=1 everywhere."""
    faxis = np.linspace(1e9, 25e9, 20)
    param = _make_param(Len_TX=0.0, Z_c=100.0, Z0=50.0)
    _, _, s21, _ = make_full_pkg('TX', faxis, param, 'THRU')
    assert np.allclose(np.abs(s21), 1.0, atol=1e-8)


def test_tx_vs_rx_different_length():
    """TX and RX use different length params."""
    faxis = np.linspace(1e9, 10e9, 20)
    param = _make_param(Len_TX=0.01, Len_RX=0.02, Z_c=100.0, Z0=50.0)
    param.pkg_gamma0_a1_a2 = np.array([0.0, 0.5, 0.0])
    _, _, s21_tx, _ = make_full_pkg('TX', faxis, param, 'THRU')
    _, _, s21_rx, _ = make_full_pkg('RX', faxis, param, 'THRU')
    # Different lengths should give different attenuation
    assert not np.allclose(s21_tx, s21_rx, atol=1e-4)


def test_dc_mode_halves_z0():
    """DC mode uses Z0/2 internally (doesn't change param externally)."""
    faxis = np.linspace(1e9, 10e9, 10)
    param = _make_param(Z0=100.0, Cpad=1e-13)
    Z0_before = float(param.Z0)
    make_full_pkg('TX', faxis, param, 'THRU', mode='dc')
    # param.Z0 should not be modified (make_full_pkg works on copy)
    assert float(param.Z0) == pytest.approx(Z0_before)


def test_rx_type_raises_for_wrong_type():
    faxis = np.linspace(1e9, 10e9, 10)
    param = _make_param()
    with pytest.raises((ValueError, AttributeError)):
        make_full_pkg('INVALID', faxis, param, 'THRU')


# ---------------------------------------------------------------------------
# COM Octave oracle tests.
#
# Values below were produced by running make_full_pkg (with make_pkg,
# synth_tline and combines4p) verbatim out of
# octave/com_ieee8023_4p16p0_octave_compat.m via tools/octave_oracle.py, on the
# inputs each test builds.  The extracted bodies were first checked identical
# to matlab/com_ieee8023_4p16p0.m.
#
# Tolerance: the residual against the reference is set by exp()/log() differing
# between Octave's libm and numpy, amplified by the 1-exp(-2*gamma*d)
# cancellation at low frequency.  Measured worst case over these probes is
# 1.5e-15 relative, so 1e-13 leaves margin while still catching every
# structural divergence found here, all of which were 1e-4 or larger.
# ---------------------------------------------------------------------------

_OCT_F = np.array([0.0, 1e9, 26.5625e9, 53e9])


def _oct_param(**over):
    """The param struct the oracle probes were run on."""
    p = SimpleNamespace()
    p.PKG_NAME = None
    p.C_diepad = np.array([1.1e-13, 1.3e-13])
    p.L_comp = np.array([1.1e-10, 1.3e-10])
    p.C_pkg_board = np.array([5e-14, 6e-14])
    p.C_bump = np.array([1.5e-13, 1.7e-13])
    p.C_v = np.array([3e-14, 4e-14])
    p.Z0 = 50.0
    p.pkg_tau = 6.141e-3
    p.pkg_gamma0_a1_a2 = np.array([0.0, 1.734e-3, 1.455e-4])
    # The workbook stores package_Z_c as cases-by-[Tx Rx] and transposes it,
    # so the shipped shape is 2-by-mele (MATLAB line 10403).
    p.pkg_Z_c = np.array([[87.5, 92.5, 90.0, 95.0], [88.0, 93.0, 91.0, 96.0]])
    p.z_p_next_cases = np.array([[1.0, 2.0, 3.0, 4.0]])
    p.Pkg_len_TX = np.array([12.0, 1.0, 2.0, 3.0])
    p.Pkg_len_RX = np.array([14.0, 1.5, 2.5, 3.5])
    p.Pkg_len_NEXT = np.array([12.0, 1.0, 2.0, 3.0])
    p.Pkg_len_FEXT = np.array([12.0, 1.0, 2.0, 3.0])
    for k, v in over.items():
        setattr(p, k, v)
    return p


def _assert_pin(got, s11, s12, s21, s22, rtol=1e-13):
    for name, g, want in zip(('s11', 's12', 's21', 's22'), got,
                             (s11, s12, s21, s22)):
        np.testing.assert_allclose(np.asarray(g).ravel(), np.array(want),
                                   rtol=rtol, err_msg=name)


def test_oracle_rx_mele1_linear_index_is_column_major():
    """MATLAB pkg_Z_c(2) on a 2x4 is element (2,1), not numpy's ravel()[1].

    param.pkg_Z_c = [87.5 92.5 90 95 ; 88 93 91 96], so the reference picks
    88 for RX while a row-major ravel picks 92.5.  COM Octave s21 at 1 GHz is
    0.78218388281501161-0.57074890734687478j; the row-major read gave
    0.78497930629192492-0.57106813157164604j, a 3.5e-3 relative error.
    """
    p = _oct_param(z_p_next_cases=np.array([[12.0]]),
                   Pkg_len_RX=np.array([14.0]))
    got = make_full_pkg('RX', _OCT_F, p, 'THRU')
    _assert_pin(
        got,
        [-1.4657971806028317e-15-1.4663028174694183e-15j, -0.054400438725395135-0.091658813078847087j, -0.48728883566244874-0.2411943356758541j, 0.28006165609813394-0.59845538320563474j],
        [0.99999999999998845-1.1532835635203662e-14j, 0.78218388281501161-0.57074890734687478j, -0.68417016919262053+0.09807238862404502j, 0.4969900260076407-0.1724699244428704j],
        [0.99999999999998845-1.1532835635203662e-14j, 0.78218388281501161-0.57074890734687478j, -0.68417016919262053+0.09807238862404502j, 0.4969900260076407-0.1724699244428704j],
        [-1.4657971806028317e-15-1.4663028174694183e-15j, -0.073446533368760406-0.076479393379752056j, 0.21199432664766948-0.35924828737522807j, -0.45746594173471666-0.40492805722971792j])


def test_oracle_tx_mele1_linear_index():
    """pkg_Z_c(1) is element (1,1) for TX, the same in either order.

    Pinned so a fix to the RX read cannot silently move the TX one.
    """
    p = _oct_param(z_p_next_cases=np.array([[12.0]]),
                   Pkg_len_TX=np.array([14.0]))
    got = make_full_pkg('TX', _OCT_F, p, 'THRU')
    _assert_pin(
        got,
        [-1.5315130116480955e-15-1.5320413176427729e-15j, -0.053627986648265978-0.089240048418431375j, -0.4284899793928536-0.27589288030815878j, 0.24937630064314276+0.14573373271789258j],
        [0.99999999999998845-1.1541377926184807e-14j, 0.78719386104622313-0.56422791652669557j, -0.71055183080670692-0.037037683608168413j, 0.63269646334550622+0.26193005933033048j],
        [0.99999999999998845-1.1541377926184807e-14j, 0.78719386104622313-0.56422791652669557j, -0.71055183080670692-0.037037683608168413j, 0.63269646334550622+0.26193005933033048j],
        [-1.5315130116480955e-15-1.5320413176427729e-15j, -0.070497795261506221-0.07603133248176315j, 0.25844953096183132-0.24860992130738227j, -0.27727969517690904-0.26128601207718355j])


def test_oracle_mele4_vector_diepad_tx_and_rx():
    """The four-segment package on the shipped vector C_d / L_comp syntax."""
    _assert_pin(
        make_full_pkg('TX', _OCT_F, _oct_param(), 'THRU'),
        [-1.6702504680723933e-15-1.6752294757632258e-15j, -0.064816809058259969-0.088261189622661482j, -0.61857481288062532-0.22736738756910541j, -0.01325455272050558+0.453187878444968j],
        [0.99999999999998523-1.4809930372277587e-14j, 0.67840147322295674-0.68025681176180952j, 0.34655449944856043-0.46699208150613891j, -0.099030742325424428-0.53028701938113398j],
        [0.99999999999998523-1.4809930372277587e-14j, 0.67840147322295674-0.68025681176180952j, 0.34655449944856043-0.46699208150613891j, -0.099030742325424428-0.53028701938113398j],
        [-1.6702504680723939e-15-1.6752294757632138e-15j, -0.089665966091815236-0.060358204137111224j, -0.32058958505958307-0.47089276251289564j, -0.29747890158887269-0.51996026865288736j])
    _assert_pin(
        make_full_pkg('RX', _OCT_F, _oct_param(), 'THRU'),
        [-1.8611204049143168e-15-1.8650377355939512e-15j, -0.078006874783546343-0.086821827342870458j, -0.6015778099982686-0.039058371555088206j, 0.37337902791250333-0.4997348990209759j],
        [0.99999999999998235-1.7675704133575356e-14j, 0.56465418600081951-0.76848503799500367j, -0.028774208070366767+0.58728866673993096j, -0.46066067332126848+0.12505787770323815j],
        [0.99999999999998235-1.7675704133575356e-14j, 0.56465418600081951-0.76848503799500367j, -0.028774208070366767+0.58728866673993096j, -0.46066067332126848+0.12505787770323815j],
        [-1.8611204049143168e-15-1.8650377355939318e-15j, -0.10550150573230511-0.044743951009476479j, -0.44789004191267617-0.1648452767292316j, -0.43667441052842471-0.23978193536448736j])


def test_oracle_mele4_matrix_diepad_keeps_every_die_section():
    """C_d / L_comp as a 2xN matrix: [Cd_Tx 0 0 0] is N+3 long, num_blocks=5."""
    mat = dict(C_diepad=np.array([[1.1e-13, 0.9e-13], [1.3e-13, 1.0e-13]]),
               L_comp=np.array([[1.1e-10, 0.7e-10], [1.3e-10, 0.8e-10]]))
    _assert_pin(
        make_full_pkg('TX', _OCT_F, _oct_param(**mat), 'THRU'),
        [-1.6702504680723933e-15-1.6752294757653881e-15j, -0.068066205538104385-0.095494804765564864j, -0.62555301731099788+0.10321486975291276j, 0.30909950272593145-0.71279053373430212j],
        [0.99999999999998523-1.4809930372281702e-14j, 0.66553458033790847-0.6916621366855551j, 0.17473082517301794-0.57183693347497577j, -0.37413222231777671-0.072282637418878415j],
        [0.99999999999998523-1.4809930372281702e-14j, 0.66553458033790847-0.6916621366855551j, 0.17473082517301794-0.57183693347497577j, -0.37413222231777671-0.072282637418878415j],
        [-1.6702504680723939e-15-1.675229475765376e-15j, -0.09863960098480648-0.060138537730984938j, -0.35896520501187718-0.41652647329764236j, -0.24085499752426029-0.651381533662169j])
    _assert_pin(
        make_full_pkg('RX', _OCT_F, _oct_param(**mat), 'THRU'),
        [-1.8611204049143168e-15-1.8650377355963229e-15j, -0.081549921595061456-0.094170484791021539j, -0.39865807599866249+0.29966447454789774j, -0.27645797333193184-0.84656443210995214j],
        [0.99999999999998235-1.767570413357996e-14j, 0.54874224391999715-0.77880555537565155j, 0.24398659896583838+0.59033786911716435j, -0.14944362136096093+0.23417943575002356j],
        [0.99999999999998235-1.767570413357996e-14j, 0.54874224391999715-0.77880555537565155j, 0.24398659896583838+0.59033786911716435j, -0.14944362136096093+0.23417943575002356j],
        [-1.8611204049143168e-15-1.8650377355963039e-15j, -0.11468041869928713-0.041615259093142176j, -0.38889816266561089-0.11641850927685632j, -0.55208801365923721-0.10991498262173682j])


def test_oracle_dc_and_no_die_modes():
    """mode='dc' halves Z0 and scales the LC values; include_die=0 zeros them."""
    _assert_pin(
        make_full_pkg('TX', _OCT_F, _oct_param(), 'THRU', 'dc'),
        [2.4120100950673843e-14+2.4208756812009071e-14j, 0.66215009612209519+0.24095251564072265j, -0.77992352272798016-0.35371886200611968j, -0.01493816412700952+0.85825818696374134j],
        [0.99999999999997169-2.833004677403711e-14j, 0.29657545006909347-0.5972065166774787j, 0.19809080835582671-0.2643845855404206j, -0.062124087901968265-0.18668296763794637j],
        [0.99999999999997169-2.833004677403711e-14j, 0.29657545006909347-0.5972065166774787j, 0.19809080835582671-0.2643845855404206j, -0.062124087901968265-0.18668296763794637j],
        [2.4120100950673837e-14+2.420875681200908e-14j, 0.62094542314700107+0.33352561797058805j, -0.022731589492086094-0.72183241196183112j, -0.20534008362933592-0.86711107633366724j])
    _assert_pin(
        make_full_pkg('TX', _OCT_F, _oct_param(), 'THRU', 'dd', 0),
        [-1.6702504680723933e-15-1.6752294757556918e-15j, -0.057942162343380149-0.060241798757403042j, 0.062293298657083204-0.14867903677897648j, 0.22356596391260064-0.022305751936988229j],
        [0.99999999999998523-1.4809930372266981e-14j, 0.71055083759041993-0.65020832164210418j, 0.76092855479202282+0.057407337059902024j, 0.53407188860133725+0.25016803594223164j],
        [0.99999999999998523-1.4809930372266981e-14j, 0.71055083759041993-0.65020832164210418j, 0.76092855479202282+0.057407337059902024j, 0.53407188860133725+0.25016803594223164j],
        [-1.6702504680723939e-15-1.6752294757556797e-15j, -0.058313655666379693-0.058930349318190071j, -0.10682833043582721-0.24979502342244828j, -0.27918293296469199-0.43504177666704591j])


def test_oracle_dc_point_is_clamped_to_eps_not_tiny():
    """make_pkg's `f(f<eps)=eps` is eps(1)=2.22e-16, not the smallest double.

    At f=0 the clamp is the only thing that makes the package lossy, so the
    constant is visible as 1-s21(1).  COM Octave gives 1.4765966227514582e-14
    for the TX four-segment case and 1.7652546091539989e-14 for RX;
    np.finfo(float).tiny leaves 1.1102230246251565e-16, two orders smaller.

    abs=0 matters: pytest.approx defaults to abs=1e-12, which swallows the
    whole difference and lets the test pass on the tiny constant.  rel is 5%
    because the deviation is one part in 1e14 of s21, so a half-ulp rounding
    difference between Octave and numpy is already 0.8% of it.
    """
    _, _, s21, _ = make_full_pkg('TX', _OCT_F, _oct_param(), 'THRU')
    assert (1.0 - s21[0].real) == pytest.approx(1.4765966227514582e-14,
                                                rel=0.05, abs=0)
    _, _, s21r, _ = make_full_pkg('RX', _OCT_F, _oct_param(), 'THRU')
    assert (1.0 - s21r[0].real) == pytest.approx(1.7652546091539989e-14,
                                                 rel=0.05, abs=0)


def test_oracle_type_switch_is_case_sensitive():
    """`switch type / case 'TX'` is case-sensitive, unlike the strcmpi above it.

    COM Octave, make_full_pkg('Tx', ...): "error: 'Cball' undefined near line
    143, column 22" -- the switch matches nothing and nothing is assigned.
    """
    for bad in ('Tx', 'tx', 'Rx', 'rx'):
        with pytest.raises(ValueError, match='TX or RX'):
            make_full_pkg(bad, _OCT_F, _oct_param(), 'THRU')


def test_oracle_unknown_channel_type_has_no_case():
    """COM Octave, channel_type='BOGUS': "error: 'Len' undefined near line 145"
    for TX and for RX alike; the reference does not fall back to a default."""
    for typ in ('TX', 'RX'):
        with pytest.raises(ValueError, match='channel_type'):
            make_full_pkg(typ, _OCT_F, _oct_param(), 'BOGUS')
    # 'NOISE' is a case on the RX side and must still answer.
    make_full_pkg('RX', _OCT_F, _oct_param(), 'NOISE')


def test_oracle_short_parameter_vectors_are_out_of_bound():
    """A parameter shorter than the reference indexes is an error, not a zero.

    COM Octave messages, in the order the assertions below run:
      "error: Len(2): out of bound 1 (dimensions are 1x1)"
      "error: C_bump(2): out of bound 1 (dimensions are 1x1)"
      "error: C_pkg_board(2): out of bound 1 (dimensions are 1x1)"
      "error: param(2): out of bound 1 (dimensions are 1x1)"      (C_v)
      "error: C_diepad(2): out of bound 1 (dimensions are 1x1)"
      "error: L_comp(2): out of bound 1 (dimensions are 1x1)"
      "error: param(2,_): out of bound 1 (dimensions are 1x4)"     (pkg_Z_c)
    """
    with pytest.raises(IndexError, match=r'Len\(2\)'):
        make_full_pkg('TX', _OCT_F, _oct_param(Pkg_len_TX=np.array([12.0])),
                      'THRU')
    with pytest.raises(IndexError, match=r'C_bump\(2\)'):
        make_full_pkg('RX', _OCT_F, _oct_param(C_bump=np.array([1.5e-13])),
                      'THRU')
    with pytest.raises(IndexError, match=r'C_pkg_board\(2\)'):
        make_full_pkg('RX', _OCT_F,
                      _oct_param(C_pkg_board=np.array([5e-14])), 'THRU')
    with pytest.raises(IndexError, match=r'C_v\(2\)'):
        make_full_pkg('RX', _OCT_F, _oct_param(C_v=np.array([3e-14])), 'THRU')
    with pytest.raises(IndexError, match=r'C_diepad\(2\)'):
        make_full_pkg('TX', _OCT_F, _oct_param(C_diepad=np.array([1.1e-13])),
                      'THRU')
    with pytest.raises(IndexError, match=r'L_comp\(2\)'):
        make_full_pkg('TX', _OCT_F, _oct_param(L_comp=np.array([1.1e-10])),
                      'THRU')
    with pytest.raises(IndexError, match=r'pkg_Z_c\(2,:\)'):
        make_full_pkg('RX', _OCT_F,
                      _oct_param(pkg_Z_c=np.array([87.5, 92.5, 90.0, 95.0])),
                      'THRU')
