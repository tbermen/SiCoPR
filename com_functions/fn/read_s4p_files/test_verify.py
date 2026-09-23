"""Verification tests for read_s4p_files().

# ============================================================
# MATLAB GROUND TRUTH (lines 10594-10767)
# Reads S4P/S2P files; populates chdata with faxis, sddXX_raw,
# sddXX_orig, sdcXX, scdXX, sccXX fields.
# INC_PACKAGE=0: sdd21 = sdd21_raw.copy() (bypass package).
# Validates same frequency axis across all channels.
# S2P: sdd11_raw only; S4P: all four 2×2 mixed-mode sub-matrices.
# ============================================================
"""
import warnings

import pytest
import numpy as np
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.read_s4p_files.py_impl import read_s4p_files


def _write_s4p(path, freqs_GHz, S_per_freq, Z0=100.0):
    """Write 4-port Touchstone RI file (row-major)."""
    with open(path, 'w') as f:
        f.write(f'# GHz S RI R {Z0}\n')
        for i, fg in enumerate(freqs_GHz):
            S = S_per_freq[i]
            row = f'{fg}'
            for r in range(4):
                for c in range(4):
                    v = S[r, c]
                    row += f' {v.real} {v.imag}'
            f.write(row + '\n')


def _make_param(tmp_path, fb=28e9):
    p = SimpleNamespace()
    p.Z0 = 100.0
    p.fb = fb
    p.flim = float('inf')
    p.Txpskew = 0.0; p.Txnskew = 0.0; p.Rxpskew = 0.0; p.Rxnskew = 0.0
    p.snpPortsOrder = [1, 3, 2, 4]
    p.max_start_freq = 0.1e9
    p.FLAG = SimpleNamespace(S2P=False)
    p.package_testcase_i = 1
    return p


def _make_OP():
    return SimpleNamespace(
        INC_PACKAGE=0,
        RX_CALIBRATION=0,
        include_pcb=0,
        DISPLAY_WINDOW=True,
        ZERO_PAD=False,
        IDEAL_TX_TERM=False,
        IDEAL_RX_TERM=False,
        T_r_filter_type=0,
        PSDRXCAL=False,
    )


def _s4p_identity(freqs_GHz):
    """Identity S4P: S21=S12=1, all others=0."""
    S = np.zeros((4, 4), dtype=complex)
    S[1, 0] = 1.0; S[0, 1] = 1.0
    return [S.copy() for _ in freqs_GHz]


def _make_chdata(path):
    ch = SimpleNamespace()
    ch.filename = str(path)
    ch.ext = '.s4p'
    ch.type = 'THRU'
    ch.faxis = None
    return ch


def test_s4p_faxis_populated(tmp_path):
    """After read, ch.faxis is set to Hz."""
    freqs = np.array([1.0, 5.0, 10.0])
    p = str(tmp_path / 'ch.s4p')
    _write_s4p(p, freqs, _s4p_identity(freqs))
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    OP = _make_OP()
    chdata, _, _, _ = read_s4p_files(param, OP, [ch])
    assert chdata[0].faxis is not None
    assert chdata[0].faxis[0] == pytest.approx(1e9)
    assert chdata[0].faxis[-1] == pytest.approx(10e9)


def test_s4p_sdd21_raw_set(tmp_path):
    """sdd21_raw field is set after reading."""
    freqs = np.array([1.0, 5.0])
    p = str(tmp_path / 'ch.s4p')
    _write_s4p(p, freqs, _s4p_identity(freqs))
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    OP = _make_OP()
    chdata, _, _, _ = read_s4p_files(param, OP, [ch])
    assert hasattr(chdata[0], 'sdd21_raw')
    assert len(chdata[0].sdd21_raw) == 2


def test_s4p_orig_fields_present(tmp_path):
    """_orig copies exist for sdd21, sdd11, sdd12, sdd22."""
    freqs = np.array([1.0, 5.0])
    p = str(tmp_path / 'ch.s4p')
    _write_s4p(p, freqs, _s4p_identity(freqs))
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    OP = _make_OP()
    chdata, _, _, _ = read_s4p_files(param, OP, [ch])
    for field in ('sdd21_orig', 'sdd12_orig', 'sdd11_orig', 'sdd22_orig'):
        assert hasattr(chdata[0], field), f'Missing: {field}'


def test_inc_package_0_sdd21_equals_raw(tmp_path):
    """INC_PACKAGE=0: ch.sdd21 = sdd21_raw."""
    freqs = np.array([1.0, 5.0, 10.0])
    p = str(tmp_path / 'ch.s4p')
    _write_s4p(p, freqs, _s4p_identity(freqs))
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    OP = _make_OP()  # INC_PACKAGE=0
    chdata, _, _, _ = read_s4p_files(param, OP, [ch])
    assert np.allclose(chdata[0].sdd21, chdata[0].sdd21_raw)


def test_unsupported_ext_raises(tmp_path):
    """Unsupported file extension raises ValueError."""
    ch = SimpleNamespace(filename='test.s3p', ext='.s3p', type='THRU', faxis=None)
    param = _make_param(tmp_path)
    OP = _make_OP()
    with pytest.raises(ValueError, match='unsupported extension'):
        read_s4p_files(param, OP, [ch])


def test_returns_four_outputs(tmp_path):
    """r4p15p0: read_s4p_files returns (chdata, SDDch, SDDp2p, param)."""
    freqs = np.array([1.0, 5.0])
    p = str(tmp_path / 'ch.s4p')
    _write_s4p(p, freqs, _s4p_identity(freqs))
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    OP = _make_OP()
    result = read_s4p_files(param, OP, [ch])
    assert len(result) == 4
    assert result[3] is param  # 4th output is param


def test_empty_portorder_auto_detects_and_persists(tmp_path):
    """r4p15p0: empty snpPortsOrder -> auto-detect; resolved order saved to param."""
    freqs = np.array([1.0, 5.0, 10.0, 20.0])
    S_per = []
    for _ in freqs:
        S = np.zeros((4, 4), dtype=complex)
        for d in range(4):
            S[d, d] = 0.05
        S[0, 1] = S[1, 0] = 0.9   # ports 1<->2
        S[2, 3] = S[3, 2] = 0.9   # ports 3<->4
        S_per.append(S)
    p = str(tmp_path / 'auto.s4p')
    _write_s4p(p, freqs, S_per)
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    param.snpPortsOrder = np.array([])  # empty -> auto
    OP = _make_OP()
    chdata, _, _, param_out = read_s4p_files(param, OP, [ch])
    assert list(param_out.snpPortsOrder) == [1, 3, 2, 4]
    assert chdata[0].faxis is not None


def test_bad_portorder_length_raises(tmp_path):
    """r4p15p0: non-empty port order whose length != 4 raises."""
    freqs = np.array([1.0, 5.0])
    p = str(tmp_path / 'ch.s4p')
    _write_s4p(p, freqs, _s4p_identity(freqs))
    ch = _make_chdata(p)
    param = _make_param(tmp_path)
    param.snpPortsOrder = [1, 2, 3]  # length 3
    OP = _make_OP()
    with pytest.raises(ValueError, match='does not match'):
        read_s4p_files(param, OP, [ch])


def test_second_file_different_freq_raises(tmp_path):
    """Crosstalk file with different freq axis raises ValueError."""
    freqs1 = np.array([1.0, 5.0, 10.0])
    freqs2 = np.array([2.0, 6.0, 11.0])
    p1 = str(tmp_path / 'ch1.s4p')
    p2 = str(tmp_path / 'ch2.s4p')
    _write_s4p(p1, freqs1, _s4p_identity(freqs1))
    _write_s4p(p2, freqs2, _s4p_identity(freqs2))
    ch1 = _make_chdata(p1)
    ch2 = SimpleNamespace(filename=p2, ext='.s4p', type='FEXT', faxis=None)
    param = _make_param(tmp_path)
    OP = _make_OP()
    with pytest.raises(ValueError, match='frequency axis'):
        read_s4p_files(param, OP, [ch1, ch2])


# ============================================================
# COM Octave oracle values — the reference's own predicates evaluated under
# Octave on the same operands (the body around them is file I/O, which an
# Octave probe cannot say anything useful about).  Pinned 2026-09-22.
#
# Two divergences these pin:
#  * MATLAB warns COM:read_s4p:FreqStepTooHigh when
#    max(diff(freq)) - param.max_freq_step > 1.  The port had no such check at
#    all, so an under-sampled channel was read without a word.
#  * The sdc21 package call sits under a bare `if 1` in MATLAB with no error
#    handling, so a failure stops the run.  The port wrapped it in
#    `except Exception: pass`, turning a broken AC-common-mode calculation
#    into sigma_ACCM_at_tp0 == 0 -- a silent zero noise contribution.
#
# COM Octave:
#   f=0:100e6:1e9, max_freq_step=50e6 -> max(diff(f))-max_freq_step > 1 TRUE
#   f=0:50e6:1e9,  max_freq_step=50e6 -> FALSE   (exactly at the limit)
#   f=0:10e6:1e9,  max_freq_step=50e6 -> FALSE
#   f a single point -> diff is 0x0, the comparison is empty and `if` is false
# ============================================================

def _uniform_s4p(tmp_path, step_GHz, n=11, name='fs.s4p'):
    freqs = np.arange(n) * step_GHz + step_GHz
    p = str(tmp_path / name)
    _write_s4p(p, freqs, _s4p_identity(freqs))
    return p, freqs


def test_octave_freq_step_too_high_warns(tmp_path):
    """max(diff(f)) = 100 MHz against max_freq_step = 50 MHz must warn."""
    p, _ = _uniform_s4p(tmp_path, 0.1)          # 100 MHz steps
    param = _make_param(tmp_path)
    param.max_freq_step = 50e6
    with pytest.warns(UserWarning, match='larger than the recommended'):
        read_s4p_files(param, _make_OP(), [_make_chdata(p)])


def test_octave_freq_step_exactly_at_limit_does_not_warn(tmp_path):
    """Guard the other side: a 50 MHz step against a 50 MHz limit is FALSE."""
    p, _ = _uniform_s4p(tmp_path, 0.05)         # 50 MHz steps
    param = _make_param(tmp_path)
    param.max_freq_step = 50e6
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        read_s4p_files(param, _make_OP(), [_make_chdata(p)])
    assert not [w for w in rec if 'larger than the recommended' in str(w.message)]


def test_octave_sdc21_package_failure_is_not_swallowed(tmp_path, monkeypatch):
    """MATLAB has no try/catch around the 'cd' s21_pkg call."""
    import com_functions.fn.read_s4p_files.py_impl as mod

    def _fake_s21_pkg(chdata, param, OP, channel_number, mode='dd', include_die=1):
        if mode == 'cd':
            raise RuntimeError('AC-CM package construction failed')
        n = len(chdata.faxis)
        return np.ones(n, dtype=complex), np.zeros((n, 2, 2), dtype=complex), 0.0

    monkeypatch.setattr(mod, '_s21_pkg', _fake_s21_pkg)
    freqs = np.array([1.0, 5.0, 10.0])
    p = str(tmp_path / 'cd.s4p')
    _write_s4p(p, freqs, _s4p_identity(freqs))
    OP = _make_OP()
    OP.INC_PACKAGE = 1
    with pytest.raises(RuntimeError, match='AC-CM package construction failed'):
        read_s4p_files(_make_param(tmp_path), OP, [_make_chdata(p)])


# --------------------------------------------------------------------------
# Renormalisation and the mixed-mode transform: the reference's FORM, exactly.
#
# Both of these are "the same matrix in exact arithmetic" cases, where the
# expression the reference writes and the one that is numerically better give
# answers ~5e-16 apart. No tolerance test can tell them apart and no oracle
# can either, because numpy does not reproduce Octave's LAPACK bit for bit.
# What CAN be pinned is the form, by replaying the reference expression on the
# same input and demanding exact equality.
#
#   ML read_Nport_touchstone 215:  inv(eye(p) - rho*s_old) * (s_old - rho*eye(p))
#     an EXPLICIT inverse, not a solve. Here the reference is the LESS accurate
#     form, and matching it is the job.
#
#   ML read_p4_s4params 10923:     W = T * (Snew / T)
#   ML read_p2_s2params 10801:     W = T * (S / T)
#     MATLAB's / SOLVES. inv(T) is exactly representable for these T, which is
#     why T @ Snew @ inv(T) looked interchangeable; it is not, and it was what
#     this function's hand-inlined copies did while read_p4_s4params, the
#     canonical translation, already used mrdivide.
#
# ML 209 guards renormalisation with ~isequal, which is exact. The port used
# abs(Z0 - Z_renorm) > 1e-9, so a pair a nano-ohm apart renormalised in MATLAB
# and not here.
#
# Each test carries its own negative control: it asserts that the form the
# reference does NOT use gives a different answer on this very input, so a
# test that stopped discriminating would say so rather than stay green.
# --------------------------------------------------------------------------

_T4 = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, -1.0]])


def _renorm_s4p(tmp_path, file_Z0, param_Z0):
    """Read a 4-port at file_Z0 with param.Z0 = param_Z0, and the raw matrix."""
    # Deliberately unstructured. An S with the usual reciprocal symmetry made
    # the two transform forms agree exactly at D(4,2), which would have left
    # the negative control below unable to fail.
    rng = np.random.default_rng(20260924)
    freqs = np.array([1.0, 2.0, 3.0, 4.0])
    S = [(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) * 0.3
         for _ in range(len(freqs))]
    p = str(tmp_path / 'renorm.s4p')
    _write_s4p(p, freqs, S, Z0=file_Z0)
    param = _make_param(tmp_path)
    param.Z0 = param_Z0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        chdata, _, _, _ = read_s4p_files(param, _make_OP(), [_make_chdata(p)])
    return chdata[0], np.array(S), param


def _expected_sdd21(S_file, file_Z0, param_Z0, port_order=(1, 3, 2, 4),
                    renorm='inv', transform='mrdivide'):
    """Replay the reference pipeline: renormalise, reorder ports, transform."""
    nf = len(S_file)
    sp = np.transpose(np.array(S_file), (1, 2, 0)).copy()
    if file_Z0 != param_Z0:
        rho = (param_Z0 - file_Z0) / (param_Z0 + file_Z0)
        eye = np.eye(4)
        for k in range(nf):
            s_old = sp[:, :, k]
            if renorm == 'inv':                       # ML 215
                sp[:, :, k] = np.linalg.inv(eye - rho * s_old) @ (s_old - rho * eye)
            else:                                     # the form NOT used
                sp[:, :, k] = np.linalg.solve(eye - rho * s_old, s_old - rho * eye)
    sch = np.transpose(sp, (2, 0, 1))
    po = [p - 1 for p in port_order]
    sch = sch[:, po, :][:, :, po]
    out = np.empty(nf, dtype=complex)
    for i in range(nf):
        if transform == 'mrdivide':                   # ML 10923
            D = _T4 @ np.linalg.solve(_T4.T, sch[i].T).T
        else:                                         # the form NOT used
            D = _T4 @ sch[i] @ np.linalg.inv(_T4)
        out[i] = D[3, 1]                              # SDD(2,1) = D(4,2)
    return out


def test_renormalisation_uses_the_reference_explicit_inverse(tmp_path):
    """ML 215 writes inv(A)*B, and inv(A)*B is what has to come out."""
    ch, S_file, _ = _renorm_s4p(tmp_path, file_Z0=50.0, param_Z0=100.0)
    got = np.asarray(ch.sdd21_raw, dtype=complex)
    want = _expected_sdd21(S_file, 50.0, 100.0, renorm='inv')
    assert np.array_equal(got, want), (
        'worst |delta| = %.3e against the reference form'
        % float(np.max(np.abs(got - want))))

    # negative control: the solve form differs on this very input, so this
    # test is discriminating rather than passing on any implementation
    other = _expected_sdd21(S_file, 50.0, 100.0, renorm='solve')
    assert not np.array_equal(want, other), (
        'solve and inv agree here, so this test proves nothing; pick an input '
        'where they do not')


def test_mixed_mode_transform_is_mrdivide_not_inv(tmp_path):
    """ML 10923 writes T*(Snew/T). MATLAB's / solves; it does not multiply."""
    ch, S_file, _ = _renorm_s4p(tmp_path, file_Z0=100.0, param_Z0=100.0)
    got = np.asarray(ch.sdd21_raw, dtype=complex)
    want = _expected_sdd21(S_file, 100.0, 100.0, transform='mrdivide')
    assert np.array_equal(got, want), (
        'worst |delta| = %.3e against the reference form'
        % float(np.max(np.abs(got - want))))

    other = _expected_sdd21(S_file, 100.0, 100.0, transform='inv')
    assert not np.array_equal(want, other), (
        'T @ S @ inv(T) and T @ (S / T) agree here, so this test proves '
        'nothing')


def test_renormalisation_guard_is_exact_not_a_tolerance(tmp_path):
    """ML 209 is ~isequal. A nano-ohm difference renormalises in MATLAB.

    The port used abs(Z0 - Z_renorm) > 1e-9, which skips this case. rho is
    about 5e-12 here, so the effect is small -- but the set of files that get
    renormalised at all is decided by this line, and it has to be the
    reference's set.
    """
    ch, S_file, _ = _renorm_s4p(tmp_path, file_Z0=100.0, param_Z0=100.0 + 1e-10)
    got = np.asarray(ch.sdd21_raw, dtype=complex)
    renormalised = _expected_sdd21(S_file, 100.0, 100.0 + 1e-10, renorm='inv')
    skipped = _expected_sdd21(S_file, 100.0, 100.0, renorm='inv')
    assert not np.array_equal(renormalised, skipped), (
        'renormalising at this Z0 changes nothing, so the test cannot tell '
        'the exact guard from the tolerance')
    assert np.array_equal(got, renormalised), (
        'the tolerance guard is back: this file was NOT renormalised, where '
        'the reference renormalises it')


# --------------------------------------------------------------------------
# The inlined Touchstone readers, against COM Octave.
#
# read_s4p_files does not call read_p4_s4params; it carries a hand-inlined
# copy. That copy had drifted on the skew binding: ML 10921 declares Sigfct as
# @(sigma2, sigma1, sigma4, sigma3), so the parameter NAMES are transposed
# ("need to swap sigma for 1 and 3 and 2 and 4", RIM 12/29/2023) and calling
# it with (Txp, Txn, Rxp, Rxn) binds sigma1 = Txn and sigma2 = Txp. The copy
# read the names in call order. That is invisible while the p and n skews
# match, and on the fixture below it was 0.190 out in SDC -- against values of
# order 0.1 -- while read_p4_s4params, which has the COM Octave test that
# pins this, was right.
#
# read_s4p_files is the reader the engine actually calls, so the copy is the
# one that has to be checked, not only the canonical. Same fixture and same
# COM Octave literals as com_functions/fn/read_p4_s4params/test_verify.py.
# --------------------------------------------------------------------------

_SKEW_OCT = {
    'SDD': [-0.032170553032775787 + 0.036856464388770618j,
            -0.0099320706699547873 + 0.026423876089297249j,
            -0.036814532001131786 + 0.028058824918477748j,
            -0.016135221446448689 + 0.017599923348547073j],
    'SDC': [-0.079295103639119721 - 0.16362413819436167j,
            -0.089593198765815674 - 0.18466005323126555j,
            -0.066561629068211337 - 0.14455614293989621j,
            -0.067296645166451813 - 0.16035136734195049j],
    'SCC': [0.24420280467838895 + 0.076185425059713777j,
            0.27369177389836907 + 0.034665667922157584j,
            0.35226547577827622 + 0.17017215344708619j,
            0.38409228576250043 + 0.14071853474954765j],
    'SCD': [-0.0093832848809085774 - 0.014732944371761664j,
            -0.03216097759250254 + 0.016097580822253696j,
            0.014684503693481302 - 0.050344332044631956j,
            -0.021537271164006 - 0.0023566532186473331j],
}


def _skew_fixture_s(k):
    S = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            S[i, j] = ((0.1 * (i + 1) + 0.01 * (j + 1)) / (1 + 0.3 * k)
                       + 1j * (0.02 * (i + 1) - 0.03 * (j + 1)) * (1 + 0.1 * k))
    return S


def _run_skew_fixture(tmp_path):
    from com_functions.fn.read_s4p_files.py_impl import _read_p4_s4params_inline
    path = str(tmp_path / 'skew.s4p')
    _write_s4p(path, [0.0, 10.0, 25.0, 50.0],
               [_skew_fixture_s(k) for k in range(4)], Z0=50.0)
    param = SimpleNamespace(Z0=50.0, flim=100e9, Txpskew=3.0, Txnskew=-1.0,
                            Rxpskew=2.5, Rxnskew=0.5, fb=28e9,
                            max_start_freq=1e9)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _read_p4_s4params_inline(path, [1, 3, 2, 4], param,
                                        SimpleNamespace(DISPLAY_WINDOW=0))


@pytest.mark.parametrize('name,slot', [('SDD', 1), ('SDC', 2), ('SCC', 3),
                                       ('SCD', 4)])
def test_octave_inlined_reader_asymmetric_skew(tmp_path, name, slot):
    """Txp 3 ps, Txn -1 ps, Rxp 2.5 ps, Rxn 0.5 ps: every p and n differs."""
    out = _run_skew_fixture(tmp_path)
    got = np.asarray(out[slot][2], dtype=complex).ravel()
    want = np.array(_SKEW_OCT[name], dtype=complex)
    worst = float(np.max(np.abs(got - want)))
    assert worst <= 1e-15, (
        '%s[2] is %.3e from COM Octave. 0.19 means the sigma binding is back '
        'in call order; ML 10921 transposes the parameter names.'
        % (name, worst))


def _write_s2p_matrix(path, freqs_GHz, S_per_freq, Z0):
    with open(path, 'w') as f:
        f.write('# GHz S RI R %g\n' % Z0)
        for i, fg in enumerate(freqs_GHz):
            S = S_per_freq[i]
            row = '%g' % fg
            for r in range(2):
                for c in range(2):
                    row += ' %.17g %.17g' % (float(S[r, c].real),
                                             float(S[r, c].imag))
            f.write(row + '\n')


def test_inlined_two_port_reader_matches_the_canonical(tmp_path):
    """_read_p2_s2params_inline must agree with read_p2_s2params, bit for bit.

    The 2-port path carries the same mrdivide transform (ML 10801,
    W = T * (S / T)) and the same ML 209 renormalisation as the canonical
    function, in a hand-inlined copy. The 4-port copy had drifted on the skew
    binding and was 0.19 out; the only way to know this one has not drifted is
    to run both on the same file and require exact equality, not a tolerance,
    since every difference of this kind starts at 1e-16.
    """
    from com_functions.fn.read_s4p_files.py_impl import _read_p2_s2params_inline
    from com_functions.fn.read_p2_s2params.py_impl import read_p2_s2params

    rng = np.random.default_rng(20260924)
    freqs = np.array([1.0, 2.0, 3.0, 4.0])
    S = [(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))) * 0.3
         for _ in freqs]
    path = str(tmp_path / 'two_port.s2p')
    # file at 50 ohm, read at 100: renormalisation runs too
    _write_s2p_matrix(path, freqs, S, 50.0)
    param = SimpleNamespace(Z0=100.0, flim=100e9, fb=28e9, max_start_freq=1e9)
    OP = SimpleNamespace(DISPLAY_WINDOW=0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        inline = _read_p2_s2params_inline(path, [1, 2], param, OP)
        canon = read_p2_s2params(path, 0, 0, [1, 2], OP, param)

    np.testing.assert_array_equal(np.asarray(inline[0].freq),
                                  np.asarray(canon[0].freq))
    for name, i in (('SDD', 1), ('SDC', 2), ('SCC', 3), ('SCD', 4)):
        a = np.asarray(inline[i], dtype=complex)
        b = np.asarray(canon[i], dtype=complex)
        assert a.shape == b.shape, '%s shape %s vs %s' % (name, a.shape, b.shape)
        assert np.array_equal(a, b), (
            'the inlined copy has drifted: %s differs by up to %.3e'
            % (name, float(np.max(np.abs(a - b)))))
