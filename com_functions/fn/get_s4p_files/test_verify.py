"""Tests for get_s4p_files.

MATLAB GROUND TRUTH:
  get_s4p_files builds chdata list from file names.
  THRU: type='THRU', ftr = fb * f_v
  FEXT: type='FEXT', ftr = fb * f_v  (NOT f_f)
  NEXT: type='NEXT', ftr = fb * f_v  (NOT f_n)
  PSDRXCAL noise channel appended when OP.PSDRXCAL=True.
  Empty file_list raises NotImplementedError.
"""
import pytest
import os
from types import SimpleNamespace
from com_functions.fn.get_s4p_files.py_impl import get_s4p_files


def _make_param():
    p = SimpleNamespace()
    p.fb = 26.5625e9
    p.f_v = 1.0
    p.f_f = 2.0   # should NOT be used by get_s4p_files
    p.f_n = 2.0   # should NOT be used
    return p


def _make_OP(psdrxcal=False):
    op = SimpleNamespace()
    op.PSDRXCAL = psdrxcal
    op.RUNTAG = 'run1'
    return op


def _file_list(n):
    return [os.path.join('dir', f'ch{i}.s4p') for i in range(n)]


def test_thru_type():
    chdata, _ = get_s4p_files(_make_param(), _make_OP(), 0, 0, _file_list(1))
    assert chdata[0].type == 'THRU'


def test_fext_type():
    chdata, _ = get_s4p_files(_make_param(), _make_OP(), 2, 0, _file_list(3))
    assert chdata[1].type == 'FEXT'
    assert chdata[2].type == 'FEXT'


def test_next_type():
    chdata, _ = get_s4p_files(_make_param(), _make_OP(), 0, 1, _file_list(2))
    assert chdata[1].type == 'NEXT'


def test_ftr_uses_f_v_not_f_f():
    p = _make_param()
    chdata, _ = get_s4p_files(p, _make_OP(), 1, 0, _file_list(2))
    # FEXT ftr must be fb*f_v, NOT fb*f_f
    assert chdata[1].ftr == pytest.approx(p.fb * p.f_v)


def test_psdrxcal_noise_channel():
    chdata, _ = get_s4p_files(_make_param(), _make_OP(psdrxcal=True), 0, 0, _file_list(2))
    assert len(chdata) == 2
    assert chdata[1].type == 'NOISE'


def test_empty_list_raises():
    with pytest.raises(NotImplementedError):
        get_s4p_files(_make_param(), _make_OP(), 0, 0, [])


def test_count_without_noise():
    chdata, _ = get_s4p_files(_make_param(), _make_OP(), 1, 1, _file_list(3))
    assert len(chdata) == 3
