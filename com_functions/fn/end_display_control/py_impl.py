# ============================================================
# MATLAB→Python translation notes for end_display_control
# MATLAB lines: 5545–5721
# ============================================================
# Pure display/reporting function.
# MATLAB msgbox → print(msg) (GUI not available in Python).
# MATLAB fprintf(...) → print(f'...') to stdout.
# MATLAB fprintf(2,...) → print(f'...', file=sys.stderr) for FAIL messages.
# MATLAB size(param.z_p_tx_cases) → np.asarray(...).shape
# param.flex set based on mele (2/4/1).
# ERL message uses ERL (two-element array) spread across format args.
# ============================================================

import sys
import numpy as np


def end_display_control(msg, param, OP, output_args, COM, min_ERL, ERL, VEO_mV, VEC_dB,
                        threshold_DER, DISPLAY_WINDOW):
    ztx = np.asarray(param.z_p_tx_cases)
    if ztx.ndim == 1:
        ncases = len(ztx)
        mele = 1
    else:
        ncases, mele = ztx.shape[0], ztx.shape[1]

    if mele == 2:
        param.flex = 2
    elif mele == 4:
        param.flex = 4
    elif mele == 1:
        param.flex = 1
    else:
        raise ValueError('config file syntax error: unexpected z_p_tx_cases shape')

    if not getattr(param, 'FLAG', None) or not getattr(param.FLAG, 'S2P', False):
        if not OP.TDMODE:
            print(f'SCMR_CH =  {output_args.SCMR_CH:.4g} dB ')

    if DISPLAY_WINDOW and not OP.RX_CALIBRATION:
        msgcolor = 'g'
        if not OP.ERL_ONLY:
            phy = str(getattr(OP, 'PHY', 'C2C'))
            if phy == 'C2M':
                if VEO_mV >= param.Min_VEO and VEO_mV <= param.Max_VEO:
                    msg = f'{msg}: EH = {VEO_mV:.3f} mV (pass)\n'
                else:
                    msg = f'{msg}: EH = {VEO_mV:.3f} mV (FAIL)\n'
                    msgcolor = 'r'
                if VEC_dB <= param.VEC_pass_threshold:
                    msg = f'{msg}: VEC = {VEC_dB:.3f} dB (pass)\n'
                else:
                    msg = f'{msg}: VEC = {VEC_dB:.3f} dB (FAIL)\n'
                    msgcolor = 'r'
            elif phy == 'C2C':
                if COM >= param.pass_threshold:
                    msg = f'{msg}: COM = {COM:.3f} dB (pass)\n'
                else:
                    msg = f'{msg}: COM = {COM:.3f} dB (FAIL)\n'
                    msgcolor = 'r'
                msg = f'{msg}: DER = {threshold_DER:.3e} at COM threshold \n'
            elif phy == 'C2Mcom':
                if VEO_mV >= param.Min_VEO and VEO_mV <= param.Max_VEO:
                    msg = f'{msg}: EH = {VEO_mV:.3f} mV (pass)\n'
                else:
                    msg = f'{msg}: EH = {VEO_mV:.3f} mV (FAIL)\n'
                    msgcolor = 'r'
                if VEC_dB <= param.VEC_pass_threshold:
                    msg = f'{msg}: VEC = {VEC_dB:.3f} dB (pass)\n'
                else:
                    msg = f'{msg}: VEC = {VEC_dB:.3f} dB (FAIL)\n'
                    msgcolor = 'r'
                if COM >= param.pass_threshold:
                    msg = f'{msg}: COM = {COM:.3f} dB (pass)\n'
                else:
                    msg = f'{msg}: COM = {COM:.3f} dB (FAIL)\n'
                    msgcolor = 'r'
                msg = f'{msg}: DER = {threshold_DER:.3e} at COM threshold \n'

        if OP.ERL:
            if ERL is not None and len(np.asarray(ERL)) > 0:
                erl_arr = np.asarray(ERL).ravel()
                if min_ERL >= param.ERL_pass_threshold:
                    msg = (f'{msg}: PASS ... ERL = {min_ERL:.3f} dB '
                           f'({erl_arr[0]:.3f} dB,{erl_arr[1]:.3f} dB) \n')
                else:
                    msg = (f'{msg}: FAIL ... ERL = {min_ERL:.3f} dB '
                           f'({erl_arr[0]:.3f} dB,{erl_arr[1]:.3f} dB) \n')
                    msgcolor = 'r'

        # MATLAB msgbox → print (GUI not available)
        revision = getattr(output_args, 'code_revision', '')
        print(f'[COM r{revision} results] {msg}')
    else:
        if not OP.ERL_ONLY:
            phy = str(getattr(OP, 'PHY', 'C2C'))
            if phy == 'C2C':
                if COM >= param.pass_threshold:
                    print(f'{msg} PASS ... COM = {COM:.3f} dB')
                else:
                    print(f'{msg} FAIL ... COM = {COM:.3f} dB', file=sys.stderr)
                print(f'{msg} DER = {threshold_DER:.3e} at COM threshold ')
            elif phy == 'C2Mcom':
                if VEC_dB <= param.VEC_pass_threshold:
                    print(f'{msg} PASS ... VEC = {VEC_dB:.3f} dB')
                else:
                    print(f'{msg} FAIL ... VEC = {VEC_dB:.3f} dB', file=sys.stderr)
                if VEO_mV >= param.Min_VEO and VEO_mV <= param.Max_VEO:
                    print(f'{msg} PASS ... EH = {VEO_mV:.3f} mV')
                else:
                    print(f'{msg} FAIL ... EH = {VEO_mV:.3f} mV', file=sys.stderr)
                if COM >= param.pass_threshold:
                    print(f'{msg} PASS ... COM = {COM:.3f} dB')
                else:
                    print(f'{msg} FAIL ... COM = {COM:.3f} dB', file=sys.stderr)
                print(f'{msg} DER = {threshold_DER:.3e} at COM threshold ')
            elif phy == 'C2M':
                if VEC_dB <= param.VEC_pass_threshold:
                    print(f'{msg} PASS ... VEC = {VEC_dB:.3f} dB')
                else:
                    print(f'{msg} FAIL ... VEC = {VEC_dB:.3f} dB', file=sys.stderr)
                if VEO_mV >= param.Min_VEO and VEO_mV <= param.Max_VEO:
                    print(f'{msg} PASS ... EH = {VEO_mV:.3f} mV')
                else:
                    print(f'{msg} FAIL ... EH = {VEO_mV:.3f} mV', file=sys.stderr)

        if OP.ERL:
            if ERL is not None and len(np.asarray(ERL)) > 0:
                erl_arr = np.asarray(ERL).ravel()
                if min_ERL >= param.ERL_pass_threshold:
                    print(f'{msg}: PASS ... ERL = {min_ERL:.3f} dB '
                          f'({erl_arr[0]:.3f} dB, {erl_arr[1]:.3f} dB)')
                else:
                    print(f'{msg}: FAIL ... ERL = {min_ERL:.3f} dB '
                          f'({erl_arr[0]:.3f} dB, {erl_arr[1]:.3f} dB)', file=sys.stderr)
                report_modal = str(getattr(OP, 'Report_Modal_ERL', '')).lower()
                if report_modal in ('enable', 'provisional'):
                    print(f'ERL11 CD DC CC = [{output_args.ERL11_CD:.4g} '
                          f'{output_args.ERL11_DC:.4g} {output_args.ERL11_CC:.4g}] dB    '
                          f'ERL22 CD DC CC = [{output_args.ERL22_CD:.4g} '
                          f'{output_args.ERL22_DC:.4g} {output_args.ERL22_CC:.4g}] dB')
    return msg
