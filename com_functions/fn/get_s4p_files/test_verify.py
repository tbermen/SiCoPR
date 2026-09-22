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


# ============================================================
# COM Octave 4p16p0 oracle pins.
# get_s4p_files extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m) and run under Octave with a non-empty
# file_list, so only the CLI path is exercised -- the picker branch calls
# uigetfile and is deliberately a NotImplementedError here.
#
# The Octave run was on Windows, where filesep is '\', so the filenames it
# returned are spelled with backslashes.  The pins below build the expected
# value with os.path.join so they say the same thing on either platform,
# which is exactly what fullfile does.
#
# One difference left in place and reported rather than patched: a path with
# a DOUBLED separator ('dir\\ch0.s4p') gives Octave an empty directory
# component, base 'run1 --ch0', where os.path collapses it to 'dir'.
# ============================================================


def _J(*parts):
    return os.path.join(*parts)


def test_octave_filenames_are_rebuilt_with_the_os_separator():
    """COM Octave: chdata.filename is fullfile(filepath,[basename fileext]),
    so every separator comes back as filesep whatever the caller passed."""
    for given in ('dir/ch0.s4p', 'dir\\ch0.s4p'):
        chdata, _ = get_s4p_files(_make_param(), _make_OP(), 0, 0, [given])
        assert chdata[0].filename == _J('dir', 'ch0.s4p'), given
        assert chdata[0].base == 'run1 dir--ch0'
    chdata, _ = get_s4p_files(_make_param(), _make_OP(), 0, 0,
                              ['a/b\\c/ch0.s4p'])
    assert chdata[0].filename == _J('a', 'b', 'c', 'ch0.s4p')
    assert chdata[0].base == 'run1 c--ch0'      # dirname is the LAST component


def test_octave_four_channel_names_and_types():
    """COM Octave, one thru, two FEXT and one NEXT."""
    files = ['dir/ch%d.s4p' % i for i in range(4)]
    chdata, param = get_s4p_files(_make_param(), _make_OP(), 2, 1, files)
    assert [c.type for c in chdata] == ['THRU', 'FEXT', 'FEXT', 'NEXT']
    assert [c.filename for c in chdata] == [_J('dir', 'ch%d.s4p' % i)
                                            for i in range(4)]
    assert [c.base for c in chdata] == ['run1 dir--ch%d' % i for i in range(4)]
    assert [c.ext for c in chdata] == ['.s4p'] * 4
    for c in chdata:
        assert c.ftr == pytest.approx(26562500000.0, rel=1e-15)
    assert param.base == 'run1 dir--ch0'


def test_octave_bare_aggressor_name_inherits_the_previous_directory():
    """COM Octave: every channel after the thru runs
    `if isempty(filepath), filepath=lastfilepath; end`, so a bare file name is
    opened from the previous channel's directory, and the inheritance
    propagates down the list.  file_list {'d1/ch0.s4p','ch1.s4p','ch2.s4p'}
    with one FEXT and one NEXT gave 'd1\\ch1.s4p' and 'd1\\ch2.s4p', base
    'run1 d1--ch1' and 'run1 d1--ch2'.  Parsing each name on its own opened
    'ch1.s4p' from the working directory instead."""
    chdata, _ = get_s4p_files(_make_param(), _make_OP(), 1, 1,
                              ['d1/ch0.s4p', 'ch1.s4p', 'ch2.s4p'])
    assert [c.filename for c in chdata] == [_J('d1', 'ch0.s4p'),
                                            _J('d1', 'ch1.s4p'),
                                            _J('d1', 'ch2.s4p')]
    assert [c.base for c in chdata] == ['run1 d1--ch0', 'run1 d1--ch1',
                                        'run1 d1--ch2']


def test_octave_bare_noise_channel_inherits_too():
    """COM Octave: the PSDRXCAL noise channel carries the same fallback."""
    chdata, _ = get_s4p_files(_make_param(), _make_OP(psdrxcal=True), 0, 0,
                              ['d1/ch0.s4p', 'noise.s4p'])
    assert chdata[1].type == 'NOISE'
    assert chdata[1].filename == _J('d1', 'noise.s4p')
    assert chdata[1].base == 'run1 d1--noise'


def test_octave_bare_thru_name_has_no_fallback():
    """COM Octave: the thru has no isempty(filepath) check, so a bare name
    stays bare and dirname is empty."""
    chdata, _ = get_s4p_files(_make_param(), _make_OP(), 0, 0, ['ch0.s4p'])
    assert chdata[0].filename == 'ch0.s4p'
    assert chdata[0].base == 'run1 --ch0'


def test_octave_extension_splitting():
    """COM Octave uses fileparts: the extension is the LAST dot onward, and a
    name with no dot has an empty extension."""
    chdata, _ = get_s4p_files(_make_param(), _make_OP(), 0, 0,
                              ['dir/ch0.v2.s4p'])
    assert chdata[0].ext == '.s4p'
    assert chdata[0].base == 'run1 dir--ch0.v2'
    chdata, _ = get_s4p_files(_make_param(), _make_OP(), 0, 0, ['dir/ch0'])
    assert chdata[0].ext == ''
    assert chdata[0].base == 'run1 dir--ch0'


def test_octave_absolute_path():
    """COM Octave: 'C:/data/chan/ch0.s4p' -> 'C:\\data\\chan\\ch0.s4p',
    base 'run1 chan--ch0'."""
    chdata, _ = get_s4p_files(_make_param(), _make_OP(), 0, 0,
                              ['C:/data/chan/ch0.s4p'])
    assert chdata[0].base == 'run1 chan--ch0'
    # not os.path.join: that reads 'C:' as a drive-relative prefix
    assert chdata[0].filename == os.sep.join(['C:', 'data', 'chan', 'ch0.s4p'])


def test_octave_short_file_list_is_refused():
    """COM Octave: file_list{nxi} past the end errors 'file_list(2): out of
    bound 1'."""
    with pytest.raises(ValueError):
        get_s4p_files(_make_param(), _make_OP(), 2, 0, ['dir/ch0.s4p'])
    with pytest.raises(ValueError):
        get_s4p_files(_make_param(), _make_OP(), 0, 1, ['dir/ch0.s4p'])
    with pytest.raises(ValueError):
        get_s4p_files(_make_param(), _make_OP(psdrxcal=True), 0, 0,
                      ['dir/ch0.s4p'])
