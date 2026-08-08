"""Verification tests for Bathtub_Contribution_Wrapper().

# ============================================================
# MATLAB GROUND TRUTH (lines 977-1028)
# COM_CONTRIBUTION_CURVES=False -> plot_bathtub_curves; True -> plot_pie_com.
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.Bathtub_Contribution_Wrapper.py_impl import (
    Bathtub_Contribution_Wrapper, _plot_bathtub_curves, _conv_fct, _d_cpdf)


def _gaussian_pdf(sigma=0.05, bin_size=0.001, n_sigma=5):
    x = np.arange(-round(n_sigma * sigma / bin_size), round(n_sigma * sigma / bin_size) + 1) * bin_size
    y = np.exp(-x**2 / (2 * sigma**2))
    y /= y.sum()
    return SimpleNamespace(BinSize=bin_size, Min=int(round(x[0] / bin_size)), y=y, x=x)


def _dummy_pdf(bin_size=0.001):
    return SimpleNamespace(BinSize=bin_size, Min=0, y=np.array([1.0]), x=np.array([0.0]))


def _make_structs(bs=0.001):
    g = _gaussian_pdf(bin_size=bs)
    d = _dummy_pdf(bs)
    SNR = SimpleNamespace(
        A_s=0.5,
        combined_interference_and_noise_pdf=g,
        COM=3.0,
    )
    Noise = SimpleNamespace(
        sci_pdf=d, cci_pdf=d, isi_and_xtalk_pdf=g,
        noise_pdf=g, jitt_pdf=d,
    )
    param = SimpleNamespace(
        package_testcase_i=1,
        specBER=1e-6,
        delta_y=bs,
    )
    OP = SimpleNamespace(
        pkg_len_select=[1],
        COM_CONTRIBUTION_CURVES=False,
        DEBUG=False,
        DISPLAY_WINDOW=False,
        RX_CALIBRATION=0,
    )
    chdata = [SimpleNamespace(base='test_channel')]
    return SNR, Noise, param, OP, chdata


def test_pie_com_branch_calls_plot_pie_com(monkeypatch):
    """COM_CONTRIBUTION_CURVES=True branch calls plot_pie_com (no NotImplementedError)."""
    pytest.importorskip('matplotlib')
    import com_functions.fn.Bathtub_Contribution_Wrapper.py_impl as mod
    called = {}

    def stub(hax, A_s, sci, cci, isi, noise, comb, dy, param):
        called['args'] = (A_s, dy)

    monkeypatch.setattr(mod, 'plot_pie_com', stub, raising=False)
    SNR, Noise, param, OP, chdata = _make_structs()
    OP.COM_CONTRIBUTION_CURVES = True
    result = mod.Bathtub_Contribution_Wrapper(SNR, Noise, param, chdata, OP)
    assert result is None
    assert called.get('args') == (SNR.A_s, param.delta_y)


def test_d_cpdf_normalised():
    """_d_cpdf creates a normalised PDF."""
    p = _d_cpdf(0.001, np.array([-0.5, 0.5]), np.array([0.5, 0.5]))
    assert abs(np.sum(p.y) - 1.0) < 1e-10


def test_conv_fct_length():
    """_conv_fct output length equals sum of input lengths minus 1."""
    p1 = _dummy_pdf()
    p2 = _dummy_pdf()
    result = _conv_fct(p1, p2)
    assert len(result.y) == len(p1.y) + len(p2.y) - 1


def test_no_display_returns_none():
    """Bathtub_Contribution_Wrapper returns None (display-only function)."""
    pytest.importorskip('matplotlib')
    SNR, Noise, param, OP, chdata = _make_structs()
    result = Bathtub_Contribution_Wrapper(SNR, Noise, param, chdata, OP)
    assert result is None


def test_plot_bathtub_adds_lines():
    """_plot_bathtub_curves adds lines to the axes."""
    mpl = pytest.importorskip('matplotlib')
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    bs = 0.001
    g = _gaussian_pdf(bin_size=bs)
    d = _dummy_pdf(bs)
    _plot_bathtub_curves(ax, 0.5, d, d, g, g, d, g, bs)
    assert len(ax.lines) > 0
    plt.close('all')
