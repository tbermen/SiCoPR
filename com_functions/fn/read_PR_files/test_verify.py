"""Verification tests for read_PR_files().

# ============================================================
# MATLAB GROUND TRUTH (lines 9569-9600)
# Reads .csv files; prepends 3*M zero-precursor samples.
# Computes uneq_imp_response from step-filtered pulse.
# Other file types are ignored (no-op).
# ============================================================
"""
import os
import numpy as np
import pytest
import tempfile
from types import SimpleNamespace
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.read_PR_files.py_impl import read_PR_files


def _param(M=4):
    return SimpleNamespace(samples_per_ui=M)


def _op():
    return SimpleNamespace(DISPLAY_WINDOW=False)


def _csv_file(tmp_path, N=20, dt=1e-12):
    t = np.arange(N) * dt
    v = np.zeros(N)
    v[N // 2] = 1.0
    data = np.column_stack([t, v])
    fpath = os.path.join(str(tmp_path), 'test.csv')
    np.savetxt(fpath, data)
    return fpath


def test_csv_sets_uneq_pulse_response(tmp_path):
    """Reading a CSV populates uneq_pulse_response on chdata."""
    M = 4
    fpath = _csv_file(tmp_path)
    cd = SimpleNamespace(filename=fpath, ext='.csv')
    chdata, param = read_PR_files(_param(M), _op(), [cd])
    assert hasattr(chdata[0], 'uneq_pulse_response')
    assert len(chdata[0].uneq_pulse_response) > 0


def test_csv_prepends_guard_zeros(tmp_path):
    """CSV reader prepends 3*M zeros before the file data."""
    M = 4
    fpath = _csv_file(tmp_path, N=20)
    cd = SimpleNamespace(filename=fpath, ext='.csv')
    chdata, _ = read_PR_files(_param(M), _op(), [cd])
    upr = chdata[0].uneq_pulse_response
    assert np.all(upr[:3 * M] == 0.0)


def test_csv_sets_uneq_imp_response(tmp_path):
    """CSV reader also sets uneq_imp_response."""
    fpath = _csv_file(tmp_path)
    cd = SimpleNamespace(filename=fpath, ext='.csv')
    chdata, _ = read_PR_files(_param(4), _op(), [cd])
    assert hasattr(chdata[0], 'uneq_imp_response')


def test_non_csv_is_noop(tmp_path):
    """Non-.csv extension leaves chdata unmodified."""
    cd = SimpleNamespace(filename='dummy.s4p', ext='.s4p')
    chdata, _ = read_PR_files(_param(4), _op(), [cd])
    assert not hasattr(cd, 'uneq_pulse_response')
