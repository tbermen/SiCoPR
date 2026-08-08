"""Verification tests for plot_bathtub_curves().

# ============================================================
# MATLAB GROUND TRUTH (lines 8865-8913)
# Plots voltage bathtub curves on hax (matplotlib Axes).
# cursors PDF: two-point at ±max_signal, prob 0.5 each.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.plot_bathtub_curves.py_impl import plot_bathtub_curves, _d_cpdf, _conv_fct


def _gaussian_pdf(sigma=0.1, bin_size=0.001, n_sigma=5):
    x = np.arange(-round(n_sigma * sigma / bin_size), round(n_sigma * sigma / bin_size) + 1) * bin_size
    y = np.exp(-x**2 / (2 * sigma**2))
    y /= y.sum()
    return SimpleNamespace(BinSize=bin_size, Min=int(round(x[0] / bin_size)), y=y, x=x)


def _dummy_pdf(bin_size=0.001):
    return SimpleNamespace(BinSize=bin_size, Min=0, y=np.array([1.0]), x=np.array([0.0]))


def test_d_cpdf_two_points():
    """_d_cpdf creates PDF with two points at ±max_signal."""
    p = _d_cpdf(0.001, np.array([-0.5, 0.5]), np.array([0.5, 0.5]))
    assert hasattr(p, 'y')
    assert abs(np.sum(p.y) - 1.0) < 1e-10


def test_conv_fct_combines_pdfs():
    """_conv_fct convolves two PDFs; output length is len(p1)+len(p2)-1."""
    p1 = _dummy_pdf()
    p2 = _dummy_pdf()
    result = _conv_fct(p1, p2)
    assert len(result.y) == len(p1.y) + len(p2.y) - 1


def test_plot_runs_without_error():
    """plot_bathtub_curves runs without exceptions given valid inputs."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        bs = 0.001
        g_pdf = _gaussian_pdf(sigma=0.05, bin_size=bs)
        d_pdf = _dummy_pdf(bs)
        plot_bathtub_curves(ax, 0.5, g_pdf, d_pdf, g_pdf, g_pdf, d_pdf, g_pdf, bs)
        plt.close('all')
    except ImportError:
        pytest.skip('matplotlib not available')


def test_plot_adds_lines():
    """plot_bathtub_curves adds lines to the axes."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        bs = 0.001
        g_pdf = _gaussian_pdf(sigma=0.05, bin_size=bs)
        d_pdf = _dummy_pdf(bs)
        plot_bathtub_curves(ax, 0.5, g_pdf, d_pdf, g_pdf, g_pdf, d_pdf, g_pdf, bs)
        assert len(ax.lines) > 0
        plt.close('all')
    except ImportError:
        pytest.skip('matplotlib not available')
