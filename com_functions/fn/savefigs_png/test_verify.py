"""Verification tests for savefigs_png().

# ============================================================
# MATLAB GROUND TRUTH (lines 11216-11255)
# Same as savefigs but saves PNG instead of .fig.
# ============================================================
"""
import os
import numpy as np
from types import SimpleNamespace
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.savefigs_png.py_impl import savefigs_png


def _param():
    return SimpleNamespace(base='test')


def _op(result_dir='.'):
    return SimpleNamespace(RUNTAG='run1', RESULT_DIR=result_dir, SAVE_FIGURES=0, SAVE_FIGURE_to_CSV=0)


def test_returns_iterable():
    """savefigs_png returns something iterable."""
    result = savefigs_png(_param(), _op())
    assert hasattr(result, '__len__') or hasattr(result, '__iter__')


def test_no_figures_returns_empty():
    """With no open figures, returns empty."""
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        result = savefigs_png(_param(), _op())
        assert len(result) == 0
    except ImportError:
        pass


def test_saves_png(tmp_path):
    """With SAVE_FIGURES=1, a .png file is created."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.close('all')
        fig = plt.figure(label='test_png')
        plt.plot([1, 2, 3], [1, 4, 9])
        op = SimpleNamespace(RUNTAG='', RESULT_DIR=str(tmp_path),
                             SAVE_FIGURES=1, SAVE_FIGURE_to_CSV=0)
        savefigs_png(_param(), op)
        pngs = [f for f in os.listdir(str(tmp_path)) if f.endswith('.png')]
        assert len(pngs) >= 1
        plt.close('all')
    except ImportError:
        pass
