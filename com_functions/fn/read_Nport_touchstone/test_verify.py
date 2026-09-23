"""Tests for read_Nport_touchstone.

MATLAB GROUND TRUTH:
  Reads Touchstone SnP files; RI/MA/DB formats.
  2-port: swaps S12/S21 per Touchstone 1.x spec.
  Port reordering via port_order (1-based).
  Renormalization applied when file Z0 != Z_renorm.
  Returns (sch, schFreqAxis) where sch.shape = (nfreq, nport, nport).
"""
import pytest
import numpy as np
import os
import tempfile
from com_functions.fn.read_Nport_touchstone.py_impl import read_Nport_touchstone


def _write_s2p(path, freqs_GHz, s11_re, s11_im, s21_re, s21_im, fmt='RI', Z0=50.0):
    """Write a simple 2-port Touchstone file (RI format)."""
    with open(path, 'w') as f:
        f.write(f'# GHz S {fmt} R {Z0}\n')
        for i, fg in enumerate(freqs_GHz):
            f.write(f'{fg} {s11_re[i]} {s11_im[i]} {s21_re[i]} {s21_im[i]} '
                    f'{s21_re[i]} {s21_im[i]} {s11_re[i]} {s11_im[i]}\n')


def test_returns_correct_shape(tmp_path):
    freqs = np.array([1.0, 5.0, 10.0])
    s11r = np.zeros(3)
    s11i = np.zeros(3)
    s21r = np.full(3, 0.9)
    s21i = np.zeros(3)
    p = str(tmp_path / 'test.s2p')
    _write_s2p(p, freqs, s11r, s11i, s21r, s21i)
    sch, freq, port_order = read_Nport_touchstone(p, [1, 2], 50.0)
    assert sch.shape == (3, 2, 2)
    assert len(freq) == 3
    assert list(port_order) == [1, 2]  # r4p15p0: 3rd return echoes supplied order


def test_frequency_in_Hz(tmp_path):
    freqs = np.array([1.0, 5.0])  # GHz
    p = str(tmp_path / 'test.s2p')
    _write_s2p(p, freqs, np.zeros(2), np.zeros(2), np.ones(2), np.zeros(2))
    _, freq, _ = read_Nport_touchstone(p, [1, 2], 50.0)
    assert freq[0] == pytest.approx(1e9)
    assert freq[1] == pytest.approx(5e9)


def test_RI_format_parses_correctly(tmp_path):
    """s21 = 0.8+0.0j should be preserved in RI format."""
    freqs = np.array([1.0])
    p = str(tmp_path / 'test.s2p')
    _write_s2p(p, freqs, [0.0], [0.0], [0.8], [0.0])
    sch, _, _ = read_Nport_touchstone(p, [1, 2], 50.0)
    # S12 and S21 are swapped back after read (Touchstone 1.x spec)
    assert abs(sch[0, 1, 0] - 0.8) < 1e-6 or abs(sch[0, 0, 1] - 0.8) < 1e-6


def test_port_reorder_swaps_ports(tmp_path):
    """Port ordering [2,1] should swap ports."""
    freqs = np.array([1.0])
    p = str(tmp_path / 'test.s2p')
    _write_s2p(p, freqs, [0.1], [0.0], [0.7], [0.0])
    sch_normal, _, _ = read_Nport_touchstone(p, [1, 2], 50.0)
    sch_swapped, _, _ = read_Nport_touchstone(p, [2, 1], 50.0)
    # s11 of swapped = s22 of normal (after 2-port swap)
    assert abs(sch_swapped[0, 0, 0] - sch_normal[0, 1, 1]) < 1e-6


def test_MA_format(tmp_path):
    """MA format: magnitude/angle (degrees)."""
    freqs = np.array([1.0])
    mag = 0.9
    ang_deg = 45.0
    expected = mag * np.exp(1j * ang_deg * np.pi / 180)
    with open(str(tmp_path / 'test.s2p'), 'w') as fh:
        fh.write('# GHz S MA R 50\n')
        fh.write(f'1.0 0.0 0.0 {mag} {ang_deg} {mag} {ang_deg} 0.0 0.0\n')
    sch, _, _ = read_Nport_touchstone(str(tmp_path / 'test.s2p'), [1, 2], 50.0)
    val = sch[0, 0, 1] if abs(sch[0, 0, 1]) > 0.1 else sch[0, 1, 0]
    assert abs(abs(val) - mag) < 1e-4


# ---------------------------------------------------------------------------
# Against COM Octave, on a Touchstone file the test writes itself -- synthetic,
# so it ships, and no IEEE channel is involved.
#
# This reader has a defect history (Octave's textscan silently truncating large
# files), and its coverage was 48%. The values below are COM Octave's own
# read_Nport_touchstone on the same file, including the [1 3 2 4] port swap and
# renormalisation to 100 ohms.
# ---------------------------------------------------------------------------

_TS_FREQ = [0.0, 1e9, 5e9, 10e9, 20e9]
_OCT_ROW0_F0 = [0.05 + 0j,
                0.019923893961834912 + 0.0017431148549531634j,
                0.90 + 0j,
                0.019923893961834912 + 0.0017431148549531634j]
_OCT_ROW0_F1 = [0.050999300904618759 - 0.00026703415540239856j,
                0.019923893961834912 + 0.0017431148549531634j,
                0.88458511418682484 - 0.030890392868349451j,
                0.019923893961834912 + 0.0017431148549531634j]


def _write_touchstone(path):
    """A 4-port in magnitude/angle form, 100 ohm reference, starting at DC."""
    lines = ['! synthetic 4-port for unit testing', '# Hz S MA R 100']
    for fi in _TS_FREQ:
        row = ['%.6g' % fi]
        for i in range(4):
            for j in range(4):
                if i == j:
                    m, a = 0.05 + 0.001 * fi / 1e9, -3.0 * fi / 1e10
                elif abs(i - j) == 1:
                    m, a = 0.9 * np.exp(-fi / 6e10), -20.0 * fi / 1e10
                else:
                    m, a = 0.02, 5.0
                row += ['%.9g' % m, '%.6g' % a]
        lines.append(' '.join(row))
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return str(path)


def test_matches_com_octave(tmp_path):
    p = _write_touchstone(tmp_path / 'synth4.s4p')
    out = read_Nport_touchstone(p, [1, 3, 2, 4], 100)
    sch, fx = np.asarray(out[0]), np.asarray(out[1]).ravel()

    assert list(fx) == _TS_FREQ, 'frequency axis is %r' % list(fx)
    assert sch.shape == (len(_TS_FREQ), 4, 4), 'sch shape is %s' % (sch.shape,)
    for k, want in ((0, _OCT_ROW0_F0), (1, _OCT_ROW0_F1)):
        got = sch[k, 0, :]
        worst = float(np.max(np.abs(got - np.array(want))))
        assert worst < 1e-13, (
            'sch[%d,0,:] worst difference from COM Octave is %.2e: %r'
            % (k, worst, list(got)))


def test_reads_every_frequency_in_the_file(tmp_path):
    """The defect this reader is known for is silent truncation, so the row
    count is worth asserting on its own."""
    p = _write_touchstone(tmp_path / 'synth4.s4p')
    fx = np.asarray(read_Nport_touchstone(p, [1, 3, 2, 4], 100)[1]).ravel()
    assert fx.size == len(_TS_FREQ), (
        'read %d frequencies from a %d-row file' % (fx.size, len(_TS_FREQ)))


def test_port_order_actually_swaps(tmp_path):
    """[1 3 2 4] must not return the same matrix as [1 2 3 4]."""
    p = _write_touchstone(tmp_path / 'synth4.s4p')
    a = np.asarray(read_Nport_touchstone(p, [1, 3, 2, 4], 100)[0])
    b = np.asarray(read_Nport_touchstone(p, [1, 2, 3, 4], 100)[0])
    assert not np.allclose(a, b), 'the port order had no effect on the matrix'


# --------------------------------------------------------------------------
# Renormalisation: ML 209-216, exactly as written.
#
#   if ~isequal(Spar.Z0, Z_renorm)          <- EXACT, not a tolerance
#       rho = (Z_renorm - Spar.Z0)/(Z_renorm + Spar.Z0);
#       Spar.S(:,:,k) = inv(eye(p) - rho*s_old(:,:,k)) * (s_old(:,:,k) - rho*eye(p));
#
# An EXPLICIT inverse. np.linalg.solve is the same matrix in exact arithmetic
# and ~4e-16 away in floating point, and numpy reproduces neither Octave nor
# MATLAB bit for bit, so no tolerance test and no oracle can tell the two
# forms apart. What can be pinned is the form, by replaying the reference
# expression here and demanding exact equality. read_s4p_files carries the
# same block inlined twice and is tested the same way.
#
# Each test carries its own negative control: it asserts that the form the
# reference does NOT use gives a different answer on this very input.
# --------------------------------------------------------------------------

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


def _renorm_case(tmp_path, file_Z0, Z_renorm):
    # unstructured on purpose: a reciprocal S makes the two forms agree
    # exactly at some elements, which would leave the control unable to fail
    rng = np.random.default_rng(20260924)
    freqs = np.array([1.0, 2.0, 3.0, 4.0])
    S = [(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))) * 0.3
         for _ in freqs]
    p = str(tmp_path / 'renorm.s2p')
    _write_s2p_matrix(p, freqs, S, file_Z0)
    sch = read_Nport_touchstone(p, [1, 2], Z_renorm)[0]
    return np.asarray(sch, dtype=complex), S


def _expect(s_old_all, file_Z0, Z_renorm, how):
    """Replay ML 209-216 on the matrix the reader itself returns unrenormalised.

    Taking s_old from a Z_renorm == file_Z0 read rather than re-deriving it
    from the file keeps this test about the renormalisation expression and
    free of any assumption about port order or token layout.
    """
    out = np.empty_like(s_old_all)
    eye = np.eye(s_old_all.shape[1])
    rho = (Z_renorm - file_Z0) / (Z_renorm + file_Z0)
    for k in range(s_old_all.shape[0]):
        s_old = s_old_all[k]
        if how == 'inv':
            out[k] = np.linalg.inv(eye - rho * s_old) @ (s_old - rho * eye)
        else:
            out[k] = np.linalg.solve(eye - rho * s_old, s_old - rho * eye)
    return out


def test_renormalisation_uses_the_reference_explicit_inverse(tmp_path):
    """ML 215 writes inv(A)*B, and inv(A)*B is what has to come out."""
    got, _ = _renorm_case(tmp_path, file_Z0=50.0, Z_renorm=100.0)
    s_old, _ = _renorm_case(tmp_path, file_Z0=50.0, Z_renorm=50.0)
    want = _expect(s_old, 50.0, 100.0, 'inv')
    assert np.array_equal(got, want), (
        'worst |delta| = %.3e against the reference form'
        % float(np.max(np.abs(got - want))))
    other = _expect(s_old, 50.0, 100.0, 'solve')
    assert not np.array_equal(want, other), (
        'inv and solve agree on this input, so the test proves nothing')


def test_renormalisation_guard_is_exact_not_a_tolerance(tmp_path):
    """ML 209 is ~isequal: a tenth of a nano-ohm still renormalises."""
    got, _ = _renorm_case(tmp_path, file_Z0=100.0, Z_renorm=100.0 + 1e-10)
    skipped, _ = _renorm_case(tmp_path, file_Z0=100.0, Z_renorm=100.0)
    renormalised = _expect(skipped, 100.0, 100.0 + 1e-10, 'inv')
    assert not np.array_equal(renormalised, skipped), (
        'renormalising at this Z0 changes nothing, so the test cannot tell '
        'an exact guard from a tolerance')
    assert np.array_equal(got, renormalised), (
        'this file was NOT renormalised, where the reference renormalises it')
