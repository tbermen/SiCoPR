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
    return SimpleNamespace(RUNTAG='RUN1', DISPLAY_WINDOW=False)


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


# ---------------------------------------------------------------------------
# COM Octave oracle (2026-09-22): get_TD_files run verbatim under Octave from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m for this function), OP.RUNTAG='RUN1',
# param.fb=26.5625e9, f_v=1, f_f=0.75, f_n=0.6.
#
# Only the file_list branch is oracled.  The other branch is uigetfile, and an
# Octave probe of a modal file dialog proves nothing; the CLI always supplies
# file_list and the port raises NotImplementedError there.
#
# Three divergences it found, all in the path arithmetic:
#
# 1. `[~, dirname] = fileparts(filepath)` splits an extension off the last
#    component even when that component is a DIRECTORY.  os.path.basename does
#    not.  A version-numbered directory therefore named every report file
#    differently: 'rev1' under MATLAB, 'rev1.2' in the port.
# 2. chdata.filename is fullfile(filepath, [basename fileext]) -- the path
#    REBUILT -- and on Windows fullfile rewrites forward slashes as
#    backslashes.  The port returned the caller's string unchanged.
# 3. MATLAB's fileparts drops exactly one separator from the head, so a
#    doubled separator leaves a trailing one and the directory NAME is empty.
#    os.path.split strips them all and the port answered 'chan'.
# ---------------------------------------------------------------------------
_OCT_CASES = [
    # (input path, filename, ext, base)
    (r'C:\data\chan\thru.s4p', r'C:\data\chan\thru.s4p', '.s4p', 'RUN1 chan--thru'),
    (r'C:\data\rev1.2\thru.s4p', r'C:\data\rev1.2\thru.s4p', '.s4p', 'RUN1 rev1--thru'),
    ('C:/data/chan/thru.s4p', r'C:\data\chan\thru.s4p', '.s4p', 'RUN1 chan--thru'),
    ('thru.s4p', 'thru.s4p', '.s4p', 'RUN1 --thru'),
    (r'C:\data\chan\thru.v2.s4p', r'C:\data\chan\thru.v2.s4p', '.s4p', 'RUN1 chan--thru.v2'),
    (r'C:\data\chan\noext', r'C:\data\chan\noext', '', 'RUN1 chan--noext'),
    (r'C:\data\chan' + '\\\\' + 'thru.s4p', r'C:\data\chan\thru.s4p', '.s4p',
     'RUN1 --thru'),
]


@pytest.mark.parametrize('path,filename,ext,base', _OCT_CASES)
def test_octave_path_parsing(path, filename, ext, base):
    """COM Octave chdata(1).filename / .ext / .base for one THRU file."""
    chdata, param = get_TD_files(_param(), _op(), 0, 0, [path])
    assert chdata[0].filename == filename
    assert chdata[0].ext == ext
    assert chdata[0].base == base
    assert param.base == base


def test_octave_three_files_and_ftr():
    """COM Octave, 1 THRU + 1 FEXT + 1 NEXT: types, bases and ftr.

    The FEXT sits in a different directory, so its base must come from its own
    path, not the THRU's."""
    p = _param()
    p.f_v, p.f_f, p.f_n = 1.0, 0.75, 0.6
    chdata, param = get_TD_files(p, _op(), 1, 1, [
        r'C:\data\chan\thru.s4p', r'C:\other\dir2\fx.s4p', r'C:\data\chan\next1.s4p'])
    assert [c.type for c in chdata] == ['THRU', 'FEXT', 'NEXT']
    assert [c.base for c in chdata] == ['RUN1 chan--thru', 'RUN1 dir2--fx',
                                        'RUN1 chan--next1']
    assert [c.ftr for c in chdata] == [26562500000.0, 19921875000.0, 15937500000.0]
