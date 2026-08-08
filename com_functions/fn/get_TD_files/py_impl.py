# ============================================================
# MATLAB→Python translation notes for get_TD_files
# MATLAB lines: 7221–7319
# ============================================================
# file_list: list of file path strings (no GUI mode; OP.DISPLAY_WINDOW path not implemented).
# filepath/basename/fileext: Python os.path.splitext + os.path.split equivalent.
# chdata[i].filename: full path.
# chdata[i].base: RUNTAG + ' ' + dirname + '--' + basename.
# chdata[i].type: 'THRU' (index 0), 'FEXT' (1..num_fext), 'NEXT' (num_fext+1..).
# chdata[i].ftr: param.fb * param.f_v/f_f/f_n.
# chdata[i].ext: file extension.
# param.base set from first file.
# GUI path (empty file_list): raises NotImplementedError.
# ============================================================

import os
from types import SimpleNamespace


def get_TD_files(param, OP, num_fext, num_next, file_list):
    """Parse file names and build chdata structure (MATLAB lines 7221-7319).

    file_list: list of file path strings. Returns (chdata, param).
    """
    if not file_list or len(file_list) == 0:
        raise NotImplementedError('get_TD_files: GUI file selection not supported; provide file_list')

    chdata = []
    nxi = 0  # 0-based index

    # THRU file (index 0 in MATLAB = 0 in Python)
    fpath = str(file_list[0]).replace('\\', os.sep)
    filepath = os.path.dirname(fpath)
    basename_ext = os.path.basename(fpath)
    basename, fileext = os.path.splitext(basename_ext)
    dirname = os.path.basename(filepath) if filepath else ''
    runtag = str(getattr(OP, 'RUNTAG', ''))

    ch = SimpleNamespace()
    ch.filename = fpath
    ch.ext = fileext
    ch.base = f'{runtag} {dirname}--{basename}'
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
        fpath_i = str(file_list[fi]).replace('\\', os.sep)
        fp_i = os.path.dirname(fpath_i)
        bn_i, ext_i = os.path.splitext(os.path.basename(fpath_i))
        dn_i = os.path.basename(fp_i) if fp_i else ''
        ch_i = SimpleNamespace()
        ch_i.filename = fpath_i
        ch_i.ext = ext_i
        ch_i.base = f'{runtag} {dn_i}--{bn_i}'
        ch_i.ftr = float(param.fb) * float(param.f_f)
        ch_i.type = 'FEXT'
        chdata.append(ch_i)

    nxi += int(num_fext)

    # NEXT files
    for i in range(int(num_next)):
        fi = nxi + i
        if fi >= len(file_list):
            raise ValueError(f'Not enough NEXT files; expected {num_next}')
        fpath_i = str(file_list[fi]).replace('\\', os.sep)
        fp_i = os.path.dirname(fpath_i)
        bn_i, ext_i = os.path.splitext(os.path.basename(fpath_i))
        dn_i = os.path.basename(fp_i) if fp_i else ''
        ch_i = SimpleNamespace()
        ch_i.filename = fpath_i
        ch_i.ext = ext_i
        ch_i.base = f'{runtag} {dn_i}--{bn_i}'
        ch_i.ftr = float(param.fb) * float(param.f_n)
        ch_i.type = 'NEXT'
        chdata.append(ch_i)

    return chdata, param
