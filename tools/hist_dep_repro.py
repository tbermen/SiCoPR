"""Root-cause the history dependence: same winning operating point, different COM
depending on how many candidates were evaluated.

Strategy: run the SAME winning EQ point two ways --
  A) 1 candidate evaluated   (TXFFE pinned to the winning row)
  B) 27 candidates evaluated (same ctle/lp, all TXFFE rows, LOCAL_SEARCH=0)
Both must select txffe row 8. If COM(A) != COM(B) we have a fast repro, and the
captured state diff at the moment COM computation begins names the leak.
"""
import hashlib
import os
import sys
from types import SimpleNamespace

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
os.chdir(_ROOT)
import com
from sweep_compare import apply_grid_reduction, _extract_case
import fom_com_probe as p

CONFIG = 'config_com_dj_200G_CAKR_178_PKGA_06_2_2025__Case1_TXLEOn.xlsx'
THRU = 'akinwale_3dj_01_2310/Tx_NPC_250mm_32AWG_BPK_500mm_27AWG_BPK_250mm_32AWG_NPC_Rx_thru1.s4p'
MAX_CTLE, MAX_TAP = 3, 3
WIN = {'ctle_index': '0', 'lp_index': '3', 'gffe_index': '0',
       'txffe_index': '8', 'tx_taps': '[1 1 1 3 3]'}


def _h(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def summarize(v, depth=0):
    """Compact, comparable fingerprint of an arbitrary value."""
    if v is None:
        return 'None'
    if isinstance(v, (bool, int, float, str)):
        return f'{v!r}'
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return f'ndarray{v.shape} empty'
        try:
            return (f'ndarray{v.shape} {v.dtype} h={_h(v)} '
                    f'min={np.nanmin(v):.6g} max={np.nanmax(v):.6g}')
        except (TypeError, ValueError):
            return f'ndarray{v.shape} {v.dtype} h={_h(v)}'
    if isinstance(v, (list, tuple)):
        if depth > 0:
            return f'{type(v).__name__}(len={len(v)})'
        return f'{type(v).__name__}[' + ', '.join(summarize(x, depth + 1) for x in v) + ']'
    if isinstance(v, SimpleNamespace):
        if depth > 1:
            return 'ns{...}'
        return 'ns{' + ', '.join(f'{k}={summarize(getattr(v, k), depth + 1)}'
                                 for k in sorted(vars(v))) + '}'
    return f'{type(v).__name__}'


def snapshot(param, fom_result, chdata, OP):
    s = {}
    for tag, obj in (('param', param), ('fom_result', fom_result), ('OP', OP)):
        for k in sorted(vars(obj)):
            s[f'{tag}.{k}'] = summarize(getattr(obj, k))
    for i, cd in enumerate(chdata):
        for k in sorted(vars(cd)):
            s[f'chdata[{i}].{k}'] = summarize(getattr(cd, k))
    return s


def run(mode):
    """mode 'A' = TXFFE pinned to winner; mode 'B' = all TXFFE rows."""
    orig_read = com.read_ParamConfigFile
    orig_build = com.OptFom_Build_TXFFE
    orig_apply = com.Apply_EQ
    cap = {}

    def patched_read(cf, OP):
        param, OP = orig_read(cf, OP)
        param.LOCAL_SEARCH = 0
        param.NonZeroLSMethod = 0
        apply_grid_reduction(param, MAX_CTLE, MAX_TAP)
        p._pin_index(param, ('ctle_gdc_values', 'CTLE_fp1', 'CTLE_fp2', 'CTLE_fz'),
                     int(WIN['ctle_index']))
        p._pin_index(param, ('g_DC_HP_values',), int(WIN['lp_index']))
        p._pin_index(param, ('cursor_gain',), int(WIN['gffe_index']))
        return param, OP

    def patched_apply(param, fom_result, chdata, OP):
        if not cap:
            cap.update(snapshot(param, fom_result, chdata, OP))
        return orig_apply(param, fom_result, chdata, OP)

    com.read_ParamConfigFile = patched_read
    com.Apply_EQ = patched_apply
    if mode == 'A':
        com.OptFom_Build_TXFFE = p._pinned_build(orig_build, WIN)
    try:
        res = com._run_com(CONFIG, 0, 0, [THRU], export_mat=False)
    finally:
        com.read_ParamConfigFile = orig_read
        com.OptFom_Build_TXFFE = orig_build
        com.Apply_EQ = orig_apply

    r = _extract_case(res)
    return float(getattr(r, 'COM_dB', float('nan'))), cap


if __name__ == '__main__':
    comA, sA = run('A')
    print(f'\n### A (1 candidate)   COM = {comA:.10f}', flush=True)
    comB, sB = run('B')
    print(f'### B (27 candidates) COM = {comB:.10f}', flush=True)
    print(f'### dCOM = {comB - comA:+.10f} dB\n')

    keys = sorted(set(sA) | set(sB))
    diffs = [k for k in keys if sA.get(k, '<absent>') != sB.get(k, '<absent>')]
    print(f'{len(diffs)} differing state keys at the moment COM computation begins:\n')
    for k in diffs:
        print(f'  {k}')
        print(f'     A: {sA.get(k, "<absent>")[:150]}')
        print(f'     B: {sB.get(k, "<absent>")[:150]}')
