# ============================================================
# MATLAB→Python translation notes for get_s4p_files
# MATLAB lines: 7664–7795
# ============================================================
# Nearly identical to get_TD_files but:
#   - FEXT and NEXT channels use param.fb * param.f_v (NOT f_f / f_n)
#   - Optional PSDRXCAL noise channel appended at end if OP.PSDRXCAL
#   - GUI path (empty file_list): raises NotImplementedError
# ============================================================

import os
from types import SimpleNamespace


def get_s4p_files(param, OP, num_fext, num_next, file_list):
    """Parse S4P file names into chdata structure (MATLAB lines 7664-7795).

    Returns (chdata, param).
    """
    if not file_list or len(file_list) == 0:
        raise NotImplementedError('get_s4p_files: GUI file selection not supported; provide file_list')

    chdata = []
    runtag = str(getattr(OP, 'RUNTAG', ''))

    def _parse(fpath_raw):
        # MATLAB stores fullfile(filepath, [basename fileext]), and fullfile
        # rewrites every separator to filesep.  COM Octave on Windows,
        # 'a/b\c/ch0.s4p' -> chdata(1).filename 'a\b\c\ch0.s4p', where this
        # kept the mixture the caller passed.
        fpath = str(fpath_raw).replace('\\', os.sep).replace('/', os.sep)
        filepath = os.path.dirname(fpath)
        base_ext = os.path.basename(fpath)
        basename, fileext = os.path.splitext(base_ext)
        dirname = os.path.basename(filepath) if filepath else ''
        return filepath, base_ext, basename, fileext, dirname

    def _fullfile(filepath, base_ext):
        return os.path.join(filepath, base_ext) if base_ext else filepath

    # THRU (index 0 in Python).  The thru has no lastfilepath fallback.
    filepath, base_ext, basename, fileext, dirname = _parse(file_list[0])
    ch = SimpleNamespace()
    ch.filename = _fullfile(filepath, base_ext)
    ch.ext = fileext
    ch.base = f'{runtag} {dirname}--{basename}'
    ch.type = 'THRU'
    ch.ftr = float(param.fb) * float(param.f_v)
    chdata.append(ch)
    param.base = ch.base
    nxi = 1

    def _aggressor(fi, kind, what):
        """One FEXT/NEXT/NOISE entry, carrying the directory forward.

        MATLAB keeps `filepath` alive across the loops and, for every channel
        after the thru, does `if isempty(filepath), filepath=lastfilepath; end`
        -- so a bare file name is opened from the PREVIOUS channel's
        directory, and that inheritance propagates down the list.  COM Octave,
        file_list {'d1/ch0.s4p','ch1.s4p','ch2.s4p'} with one FEXT and one
        NEXT: chdata(2).filename 'd1\\ch1.s4p' and chdata(3).filename
        'd1\\ch2.s4p', base 'run1 d1--ch1' and 'run1 d1--ch2'.  Parsing each
        entry independently opened 'ch1.s4p' from the working directory.
        """
        nonlocal filepath
        if fi >= len(file_list):
            raise ValueError(f'Not enough {what} files')
        fp, be, bn, ext, dn = _parse(file_list[fi])
        if not fp:
            fp = filepath                       # lastfilepath
            dn = os.path.basename(fp) if fp else ''
        filepath = fp
        ch_i = SimpleNamespace()
        ch_i.filename = _fullfile(fp, be)
        ch_i.ext = ext
        ch_i.base = f'{runtag} {dn}--{bn}'
        ch_i.ftr = float(param.fb) * float(param.f_v)
        ch_i.type = kind
        chdata.append(ch_i)

    for i in range(int(num_fext)):
        _aggressor(nxi + i, 'FEXT', 'FEXT')
    nxi += int(num_fext)

    for i in range(int(num_next)):
        _aggressor(nxi + i, 'NEXT', 'NEXT')
    nxi += int(num_next)

    # Optional noise channel for PSDRXCAL
    if getattr(OP, 'PSDRXCAL', False):
        _aggressor(nxi, 'NOISE', 'noise channel')

    return chdata, param
