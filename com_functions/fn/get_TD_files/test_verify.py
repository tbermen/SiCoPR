"""Tests for get_TD_files (MATLAB lines ~7221).

MATLAB GROUND TRUTH:
  file_list with 1 THRU + 2 FEXT + 1 NEXT:
    chdata[0].type='THRU', chdata[1].type='FEXT', chdata[3].type='NEXT'
    param.base set from first file
  Empty file_list: raises NotImplementedError.
"""
import pytest
from com_functions.fn.get_TD_files.py_impl import get_TD_files
from types import SimpleNamespace


def _param():
    p = SimpleNamespace()
    p.fb = 26.5625e9
    p.f_v = 1.0
    p.f_f = 1.0
    p.f_n = 1.0
    p.base = ''
    return p


def _op():
    return SimpleNamespace(RUNTAG='run1', DISPLAY_WINDOW=False)


def test_basic_1thru_2fext_1next():
    file_list = [
        r'C:\data\chan\thru.s4p',
        r'C:\data\chan\fext1.s4p',
        r'C:\data\chan\fext2.s4p',
        r'C:\data\chan\next1.s4p',
    ]
    chdata, param = get_TD_files(_param(), _op(), 2, 1, file_list)
    assert len(chdata) == 4
    assert chdata[0].type == 'THRU'
    assert chdata[1].type == 'FEXT'
    assert chdata[2].type == 'FEXT'
    assert chdata[3].type == 'NEXT'


def test_param_base_set():
    file_list = [r'C:\data\chan\thru.s4p']
    chdata, param = get_TD_files(_param(), _op(), 0, 0, file_list)
    assert param.base != ''
    assert 'thru' in param.base.lower()


def test_thru_ftr():
    p = _param()
    p.fb = 10e9
    p.f_v = 0.75
    file_list = [r'C:\data\thru.s4p']
    chdata, param = get_TD_files(p, _op(), 0, 0, file_list)
    assert abs(chdata[0].ftr - 10e9 * 0.75) < 1


def test_fext_ftr():
    p = _param()
    p.fb = 10e9
    p.f_f = 0.5
    file_list = [r'C:\data\thru.s4p', r'C:\data\fext.s4p']
    chdata, param = get_TD_files(p, _op(), 1, 0, file_list)
    assert abs(chdata[1].ftr - 10e9 * 0.5) < 1


def test_empty_file_list_raises():
    with pytest.raises(NotImplementedError):
        get_TD_files(_param(), _op(), 0, 0, [])


def test_next_ftr():
    p = _param()
    p.fb = 10e9
    p.f_n = 0.4
    file_list = [r'C:\data\thru.s4p', r'C:\data\next.s4p']
    chdata, param = get_TD_files(p, _op(), 0, 1, file_list)
    assert chdata[1].type == 'NEXT'
    assert abs(chdata[1].ftr - 10e9 * 0.4) < 1
