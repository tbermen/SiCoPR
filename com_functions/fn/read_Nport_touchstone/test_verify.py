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
