# ============================================================
# MATLAB→Python translation notes for TDR_ERL_Processing
# MATLAB lines: 4498–4592
# ============================================================
# package_testcase_i is 1-based in MATLAB. Function guards with ==1.
# OP.TDR, OP.ERL, OP.ERL_ONLY, OP.TDR_W_TXPKG, OP.BREAD_CRUMBS: boolean flags.
# OP.AUTO_TFX: whether to include tfx_estimate.
# param.FLAG.S2P: 2-port file (no port 2 data).
# OP.Report_Modal_ERL: 'enable'/'provisional' → include modal ERL fields.
# str2csv inlined: joins base strings.
# ERL output: [ERL11, ERL22] or [ERL11, nan] or [nan, ERL22].
# ERL_ONLY block: sets output_args fields OP/param/chdata/fom_result/Z_t/file_names.
# ============================================================

import numpy as np
from com_functions.fn.str2csv.py_impl import str2csv as _str2csv
from types import SimpleNamespace


def _mcat(*parts):
    """MATLAB's `[a b]` on numeric scalars: an EMPTY operand contributes
    nothing, it does not become an element.
    COM Octave: [nan, []] -> nan, numel 1    (the port built [nan, []], 2 long)
                [[], nan] -> nan, numel 1
                [nan, 5]  -> [nan 5], numel 2
    """
    return np.concatenate(
        [np.atleast_1d(np.asarray(p, dtype=float).ravel()) for p in parts])


def _mmin2(a, b):
    """MATLAB's two-argument min(): NaN is SKIPPED unless both are NaN.
    Python's min() just compares, so it is order-dependent and wrong.
    COM Octave: min(NaN, 5) -> 5   (python min gave nan)
                min(5, NaN) -> 5
                min(NaN,NaN)-> NaN
    """
    return float(np.fmin(np.float64(a), np.float64(b)))


def TDR_ERL_Processing(output_args, OP, package_testcase_i, chdata, param):
    """Fill TDR/ERL fields in output_args (MATLAB lines 4498-4592).

    Returns (output_args, ERL, min_ERL).
    """
    # Fill TDR data — only on first test case
    if package_testcase_i == 1:
        if OP.TDR:
            output_args.Z11est = chdata[0].TDR11.avgZport
            if not param.FLAG.S2P:
                output_args.Z22est = chdata[0].TDR22.avgZport
            else:
                output_args.Z22est = []
            if OP.AUTO_TFX:
                output_args.tfx_estimate = param.tfx[1]  # MATLAB tfx(2) → 0-based [1]
            else:
                output_args.tfx_estimate = []
        else:
            output_args.Z11est = []
            output_args.Z22est = []
            output_args.tfx_estimate = []

    # Process ERL — only on first test case
    if package_testcase_i == 1:
        _modal = (str(OP.Report_Modal_ERL).lower() in ('enable', 'provisional'))
        if OP.ERL:
            output_args.ERL11 = chdata[0].TDR11.ERL
            if _modal:
                output_args.ERL11_CD = chdata[0].TDR11.ERL_CD
                output_args.ERL11_DC = chdata[0].TDR11.ERL_DC
                output_args.ERL11_CC = chdata[0].TDR11.ERL_CC
            if not param.FLAG.S2P:
                output_args.ERL22 = chdata[0].TDR22.ERL
                if _modal:
                    output_args.ERL22_CD = chdata[0].TDR22.ERL_CD
                    output_args.ERL22_DC = chdata[0].TDR22.ERL_DC
                    output_args.ERL22_CC = chdata[0].TDR22.ERL_CC
            else:
                output_args.ERL22 = []
                output_args.ERL22_CD = []
                output_args.ERL22_DC = []
                output_args.ERL22_CC = []
        else:
            output_args.ERL11 = []
            output_args.ERL22 = []
            if not _modal:
                output_args.ERL11_CD = []
                output_args.ERL22_CD = []
                output_args.ERL11_DC = []
                output_args.ERL22_DC = []
                output_args.ERL11_CC = []
                output_args.ERL22_CC = []

    # Compute ERL summary
    erl11 = getattr(output_args, 'ERL11', [])
    erl22 = getattr(output_args, 'ERL22', [])

    def _is_empty(v):
        if v is None:
            return True
        if isinstance(v, (list, np.ndarray)) and len(v) == 0:
            return True
        return False

    if OP.ERL:
        if OP.TDR_W_TXPKG:
            min_ERL = erl22
            ERL = _mcat(np.nan, erl22)
        else:
            if _is_empty(erl22):
                min_ERL = erl11
                ERL = _mcat(erl11, np.nan)
            else:
                min_ERL = _mmin2(erl11, erl22)
                ERL = _mcat(erl11, erl22)
        output_args.ERL = min_ERL
    else:
        min_ERL = []
        ERL = []
        output_args.ERL = []

    if OP.ERL_ONLY:
        if OP.BREAD_CRUMBS:
            output_args.OP = OP
            output_args.param = param
            output_args.chdata = chdata
            fom_result = SimpleNamespace(ran=0)
            output_args.fom_result = fom_result
        output_args.Z_t = param.Z_t
        # MATLAB passes str2csv the ONE-element cell {chdata(1).base}; joining
        # every chdata entry listed the crosstalk files too.
        # COM Octave: str2csv({'a.s4p'}) -> 'a.s4p';
        #   sprintf('"%s"','a.s4p') -> '"a.s4p"'
        fileset_str = _str2csv([str(chdata[0].base)])
        output_args.file_names = f'"{fileset_str}"'

    return output_args, ERL, min_ERL
