"""Tests for plot_pie_com (MATLAB lines ~9011).

MATLAB GROUND TRUTH:
  COM_per_noise = COM * (maxn^2 / sum(maxn^2)).
  COM < 0 → returns early (no output).
  All PDFs zero → function handles gracefully.
"""
import numpy as np
import pytest
from types import SimpleNamespace
from com_functions.fn.plot_pie_com.py_impl import plot_pie_com


def _make_pdf(sigma=0.1, BinSize=1e-4):
    n = int(6 * sigma / BinSize) * 2 + 1
    x_min = -(n // 2)
    x = np.arange(x_min, -x_min + 1) * BinSize
    y = np.exp(-0.5 * (x / sigma) ** 2)
    y = y / np.sum(y)
    return SimpleNamespace(BinSize=BinSize, Min=int(x_min), y=y, x=x)


def _param():
    p = SimpleNamespace()
    p.specBER = 1e-4
    p.delta_IL = 3.0
    p.pass_threshold = 0.0
    return p


def test_runs_without_error():
    pdf = _make_pdf(0.05)
    plot_pie_com(None, 1.0, pdf, pdf, pdf, pdf, pdf, 1e-4, _param())


def test_returns_none():
    pdf = _make_pdf(0.05)
    result = plot_pie_com(None, 1.0, pdf, pdf, pdf, pdf, pdf, 1e-4, _param())
    assert result is None


def test_low_signal_no_error():
    # max_signal << noise → COM < 0 → returns early
    pdf = _make_pdf(1.0)
    plot_pie_com(None, 0.001, pdf, pdf, pdf, pdf, pdf, 1e-4, _param())


def test_prints_percentages(capsys):
    pdf = _make_pdf(0.05)
    plot_pie_com(None, 1.0, pdf, pdf, pdf, pdf, pdf, 1e-4, _param())
    out = capsys.readouterr().out
    assert '%' in out


def test_different_pdfs_no_error(capsys):
    sci = _make_pdf(0.1)
    cci = _make_pdf(0.05)
    noise = _make_pdf(0.02)
    combined = _make_pdf(0.12)
    isi_and_xtalk = _make_pdf(0.08)
    plot_pie_com(None, 1.0, sci, cci, isi_and_xtalk, noise, combined, 1e-4, _param())
