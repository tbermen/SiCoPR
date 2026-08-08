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
        fpath = str(fpath_raw).replace('\\', os.sep)
        filepath = os.path.dirname(fpath)
        base_ext = os.path.basename(fpath)
        basename, fileext = os.path.splitext(base_ext)
        dirname = os.path.basename(filepath) if filepath else ''
        return fpath, filepath, basename, fileext, dirname

    # THRU (index 0 in Python)
    fpath, filepath, basename, fileext, dirname = _parse(file_list[0])
    ch = SimpleNamespace()
    ch.filename = fpath
    ch.ext = fileext
    ch.base = f'{runtag} {dirname}--{basename}'
    ch.type = 'THRU'
    ch.ftr = float(param.fb) * float(param.f_v)
    chdata.append(ch)
    param.base = ch.base
    nxi = 1

    # FEXT channels
    for i in range(int(num_fext)):
        fi = nxi + i
        if fi >= len(file_list):
            raise ValueError(f'Not enough FEXT files; expected {num_fext}')
        fp, fpe, bn, ext, dn = _parse(file_list[fi])
        ch_i = SimpleNamespace()
        ch_i.filename = fp
        ch_i.ext = ext
        ch_i.base = f'{runtag} {dn}--{bn}'
        ch_i.ftr = float(param.fb) * float(param.f_v)
        ch_i.type = 'FEXT'
        chdata.append(ch_i)
    nxi += int(num_fext)

    # NEXT channels
    for i in range(int(num_next)):
        fi = nxi + i
        if fi >= len(file_list):
            raise ValueError(f'Not enough NEXT files; expected {num_next}')
        fp, fpe, bn, ext, dn = _parse(file_list[fi])
        ch_i = SimpleNamespace()
        ch_i.filename = fp
        ch_i.ext = ext
        ch_i.base = f'{runtag} {dn}--{bn}'
        ch_i.ftr = float(param.fb) * float(param.f_v)
        ch_i.type = 'NEXT'
        chdata.append(ch_i)
    nxi += int(num_next)

    # Optional noise channel for PSDRXCAL
    if getattr(OP, 'PSDRXCAL', False):
        noise_idx = nxi  # 0-based
        if noise_idx >= len(file_list):
            raise ValueError('Not enough files: PSDRXCAL requires a noise channel file')
        fp, fpe, bn, ext, dn = _parse(file_list[noise_idx])
        ch_n = SimpleNamespace()
        ch_n.filename = fp
        ch_n.ext = ext
        ch_n.base = f'{runtag} {dn}--{bn}'
        ch_n.ftr = float(param.fb) * float(param.f_v)
        ch_n.type = 'NOISE'
        chdata.append(ch_n)

    return chdata, param
