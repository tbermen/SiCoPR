"""Verification tests for savefigs().

# ============================================================
# MATLAB GROUND TRUTH (lines 11175-11215)
# Returns list of open figures. Saves .fig and .csv when flags set.
# ============================================================
"""
import numpy as np
from types import SimpleNamespace
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.savefigs.py_impl import savefigs


def _param():
    return SimpleNamespace(base='test')


def _op():
    return SimpleNamespace(RUNTAG='run1', RESULT_DIR='.', SAVE_FIGURES=0, SAVE_FIGURE_to_CSV=0)


def test_returns_list():
    """savefigs returns something iterable (list or array)."""
    result = savefigs(_param(), _op())
    assert hasattr(result, '__len__') or hasattr(result, '__iter__')


def test_no_figures_returns_empty():
    """With no open figures, returns empty list."""
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        result = savefigs(_param(), _op())
        assert len(result) == 0
    except ImportError:
        pass  # matplotlib not available — trivially passes


def test_with_figure_returns_nonempty(tmp_path):
    """With open figures, returns non-empty list."""
    try:
        import matplotlib.pyplot as plt
        plt.figure(label='test_fig')
        plt.plot([1, 2, 3], [1, 4, 9])
        op = SimpleNamespace(RUNTAG='run1', RESULT_DIR=str(tmp_path),
                             SAVE_FIGURES=0, SAVE_FIGURE_to_CSV=0)
        result = savefigs(_param(), op)
        assert len(result) > 0
        plt.close('all')
    except ImportError:
        pass
