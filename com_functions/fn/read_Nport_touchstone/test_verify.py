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
