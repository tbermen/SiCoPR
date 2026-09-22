# ============================================================
# MATLAB→Python translation notes for get_TD_files
# MATLAB lines: 7221–7319
# ============================================================
# file_list: list of file path strings (no GUI mode; OP.DISPLAY_WINDOW path not implemented).
# filepath/basename/fileext come from _fileparts, MATLAB's fileparts, which is
#   NOT os.path.split + os.path.splitext in two places -- see _fileparts and
#   _dirname below.
# chdata[i].filename: fullfile(filepath, [basename fileext]), i.e. the path
#   REBUILT, not the caller's string.
# chdata[i].base: RUNTAG + ' ' + dirname + '--' + basename.
# chdata[i].type: 'THRU' (index 0), 'FEXT' (1..num_fext), 'NEXT' (num_fext+1..).
# chdata[i].ftr: param.fb * param.f_v/f_f/f_n.
# chdata[i].ext: file extension.
# param.base set from first file.
# GUI path (empty file_list): raises NotImplementedError.
# ============================================================

import os
from types import SimpleNamespace


def _fileparts(p):
    """MATLAB fileparts(): (path, name, ext).

    Splits at the LAST separator and drops exactly that one separator, where
    os.path.split() strips ALL trailing separators from the head. The
    difference shows on a doubled separator:
      COM Octave  fileparts('C:\\data\\chan\\\\thru.s4p')
        -> path 'C:\\data\\chan\\'  (os.path.split gives 'C:\\data\\chan')
    and that trailing separator is what makes _dirname come out empty, as
    MATLAB reports it.
    """
    i = max(p.rfind('\\'), p.rfind('/'))
    if i >= 0:
        head, tail = p[:i], p[i + 1:]
    elif len(p) == 2 and p[1] == ':':
        head, tail = p, ''      # a bare drive is all path and no name
    else:
        head, tail = '', p
    name, ext = os.path.splitext(tail)
    return head, name, ext


def _dirname(filepath):
    """MATLAB `[~, dirname] = fileparts(filepath)` -- the NAME of the directory.

    fileparts splits an extension off the last component whether or not that
    component is a file, so a dotted directory loses everything from its last
    dot.  os.path.basename does not, and chdata.base (which names every report
    file) carried the difference.
      COM Octave  'C:\\data\\rev1.2\\thru.s4p' -> base 'RUN1 rev1--thru'
                  (os.path.basename gave 'RUN1 rev1.2--thru')
      COM Octave  'C:\\data\\chan\\\\thru.s4p' -> base 'RUN1 --thru'
                  (os.path.basename gave 'RUN1 chan--thru')
    """
    return _fileparts(filepath)[1]


def _fullfile(head, tail):
    """MATLAB fullfile(): join with filesep, collapsing a duplicate separator,
    and on Windows rewrite every forward slash as a backslash.

    COM Octave  fullfile('C:/data/chan', 'thru.s4p')
      -> 'C:\\data\\chan\\thru.s4p'   (os.path.join gives 'C:/data/chan\\thru.s4p')
    """
    if not head:
        return tail
    sep = os.sep
    p = head + ('' if head.endswith(('\\', '/')) else sep) + tail
    if sep == '\\':
        p = p.replace('/', '\\')
    return p


def _one(fpath, runtag):
    """Common per-file work: filename, ext and base, as MATLAB builds them."""
    filepath, basename, fileext = _fileparts(fpath)
    ch = SimpleNamespace()
    ch.filename = _fullfile(filepath, basename + fileext)
    ch.ext = fileext
    ch.base = '%s %s--%s' % (runtag, _dirname(filepath), basename)
    return ch


def get_TD_files(param, OP, num_fext, num_next, file_list):
    """Parse file names and build chdata structure (MATLAB lines 7221-7319).

    file_list: list of file path strings. Returns (chdata, param).
    """
    if not file_list or len(file_list) == 0:
        raise NotImplementedError('get_TD_files: GUI file selection not supported; provide file_list')

    chdata = []
    runtag = str(getattr(OP, 'RUNTAG', ''))

    # THRU file (index 0 in MATLAB = 0 in Python)
    ch = _one(str(file_list[0]).replace('\\', os.sep), runtag)
    ch.type = 'THRU'
    ch.ftr = float(param.fb) * float(param.f_v)
    chdata.append(ch)
    param.base = ch.base
    nxi = 1

    # FEXT files
    for i in range(int(num_fext)):
        fi = nxi + i
        if fi >= len(file_list):
            raise ValueError(f'Not enough FEXT files; expected {num_fext}')
        ch_i = _one(str(file_list[fi]).replace('\\', os.sep), runtag)
        ch_i.ftr = float(param.fb) * float(param.f_f)
        ch_i.type = 'FEXT'
        chdata.append(ch_i)

    nxi += int(num_fext)

    # NEXT files
    for i in range(int(num_next)):
        fi = nxi + i
        if fi >= len(file_list):
            raise ValueError(f'Not enough NEXT files; expected {num_next}')
        ch_i = _one(str(file_list[fi]).replace('\\', os.sep), runtag)
        ch_i.ftr = float(param.fb) * float(param.f_n)
        ch_i.type = 'NEXT'
        chdata.append(ch_i)

    return chdata, param
