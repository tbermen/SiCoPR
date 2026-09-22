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


# ============================================================
# COM Octave 4p16p0 oracle pins.
# read_PR_files extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m) and run under Octave on the very files these
# tests write.
#
# Fixture: eight samples at dt = 1 ps, v = [0 0 0.25 1 0.5 0.125 0 0],
# samples_per_ui = 4, so 3*M = 12 guard zeros go in front.
# ============================================================
_PIN_ROWS = [(0.0, 0.0), (1e-12, 0.0), (2e-12, 0.25), (3e-12, 1.0),
             (4e-12, 0.5), (5e-12, 0.125), (6e-12, 0.0), (7e-12, 0.0)]

_OCT_UPR = [0] * 14 + [0.25, 1, 0.5, 0.125, 0, 0]
_OCT_T = [k * 1e-12 for k in range(20)]
_OCT_IMP = [0] * 14 + [0.25, 0.75, -0.5, -0.375, 0.125, 0.75]


def _pr_file(tmp_path, sep=' ', name='pin.csv'):
    path = os.path.join(str(tmp_path), name)
    with open(path, 'w', newline='\n') as f:
        for t, v in _PIN_ROWS:
            f.write('%.12g%s%.12g\n' % (t, sep, v))
    return path


def _run(path, ext='.csv', M=4):
    cd = SimpleNamespace(filename=path, ext=ext)
    chdata, _ = read_PR_files(SimpleNamespace(samples_per_ui=M),
                              SimpleNamespace(DISPLAY_WINDOW=False), [cd])
    return chdata[0]


def test_octave_pulse_time_and_impulse_response(tmp_path):
    """COM Octave: 3*M guard zeros in front of the file's second column, a
    time axis that starts at zero and shifts the file's own times by 3*M*dt,
    and an impulse response that is the M-tap step response differenced, with
    sample 1 copied from sample 2."""
    ch = _run(_pr_file(tmp_path))
    np.testing.assert_allclose(ch.uneq_pulse_response, _OCT_UPR, rtol=1e-15,
                               atol=0)
    np.testing.assert_allclose(ch.t, _OCT_T, rtol=1e-14, atol=0)
    np.testing.assert_allclose(ch.uneq_imp_response, _OCT_IMP, rtol=1e-14,
                               atol=0)


def test_octave_load_takes_comma_and_tab_as_well_as_space(tmp_path):
    """COM Octave reads the file with load(), which accepts space, comma or
    tab -- and since the extension is .csv, commas are the expected case.
    np.loadtxt splits on whitespace only and stopped with 'could not convert
    string' on a comma-separated file the reference read without complaint."""
    for sep, name in ((' ', 'sp.csv'), (',', 'comma.csv'), ('\t', 'tab.csv')):
        ch = _run(_pr_file(tmp_path, sep=sep, name=name))
        np.testing.assert_allclose(ch.uneq_pulse_response, _OCT_UPR,
                                   rtol=1e-15, atol=0), name
        np.testing.assert_allclose(ch.uneq_imp_response, _OCT_IMP,
                                   rtol=1e-14, atol=0), name


def test_octave_extension_match_is_case_sensitive(tmp_path):
    """COM Octave: `switch chdata(i).ext; case '.csv'` does no case folding,
    so a channel recorded as '.CSV' falls straight through and keeps only
    filename and ext.  Lower-casing the extension read the file instead."""
    path = _pr_file(tmp_path, name='upper.csv')
    ch = _run(path, ext='.CSV')
    assert not hasattr(ch, 'uneq_pulse_response')
    assert not hasattr(ch, 't')
    assert not hasattr(ch, 'uneq_imp_response')
    # the same file under the exact extension is read
    ch = _run(path, ext='.csv')
    assert hasattr(ch, 'uneq_pulse_response')


def test_octave_guard_length_follows_samples_per_ui(tmp_path):
    """COM Octave: the guard is 3*param.samples_per_ui samples long."""
    path = _pr_file(tmp_path, name='m8.csv')
    for M, guard in ((4, 12), (8, 24), (32, 96)):
        ch = _run(path, M=M)
        assert len(ch.uneq_pulse_response) == guard + len(_PIN_ROWS), M
        assert np.all(ch.uneq_pulse_response[:guard] == 0.0), M
        assert ch.t[guard] == pytest.approx(guard * 1e-12, rel=1e-14), M


def test_octave_extra_columns_are_ignored(tmp_path):
    """COM Octave reads vt(:,1) and vt(:,2) only."""
    path = os.path.join(str(tmp_path), 'three.csv')
    with open(path, 'w', newline='\n') as f:
        for t, v in _PIN_ROWS:
            f.write('%.12g %.12g 9.9\n' % (t, v))
    ch = _run(path)
    np.testing.assert_allclose(ch.uneq_pulse_response, _OCT_UPR, rtol=1e-15,
                               atol=0)
