# ============================================================
# MATLAB→Python translation notes for read_package_parameters
# MATLAB lines: 10525–10593
# ============================================================
# Reads package parameters from one parameter block (a `.START`/`.END` region of
# the configuration sheet).
#
# `xls_parameter` is inlined below as `_xls_param`, following the project rule
# that a callee is inlined rather than imported from a sibling py_impl.
#
# NOTE: this function is fully implemented and raises nothing. Earlier revisions
# of this header said it "raises NotImplementedError for the full-spreadsheet
# path"; that was never true of the body and the wording made `assemble_sicopr.py`
# report it as an unimplemented function on every build. It is also currently
# uncalled: `read_ParamConfigFile` parses `.START` blocks with its own inlined
# `__read_pkg_params`.
# ============================================================
# z_p_tx_cases shape: MATLAB transposes → shape (ncases, mele).
# mele==2 → flex=2; mele==4 → flex=4; mele==1 → flex=1; else → ValueError.
# When mele==2: expand to 4-column by appending zeros for TX/NEXT/FEXT/RX.
# pkg_Z_c similarly expanded to 4 when mele==2.
# ============================================================

import numpy as np
from types import SimpleNamespace


_MANDATORY = object()   # xls_parameter called with nargin<4: no default exists


def _xls_param(parameter, key, default=_MANDATORY):
    """Minimal xls_parameter: looks up key in a dict or SimpleNamespace.

    MATLAB's xls_parameter takes the default as an optional FOURTH argument
    and, when it is absent, calls missingParameter(param_name) rather than
    inventing one.  COM Octave, read_package_parameters with 'C_p' left out of
    the parameter cell: "error: The data for mandatory parameter C_p is
    missing or incorrect".  The port supplied a default for every key,
    including the nine the reference treats as mandatory, and answered.
    """
    missing = object()
    if isinstance(parameter, dict):
        val = parameter.get(key, missing)
    else:
        attr = key.replace(' ', '_').replace('(', '').replace(')', '')
        val = getattr(parameter, attr, missing)
    if val is missing or val is None:
        if default is _MANDATORY:
            raise KeyError('The data for mandatory parameter %s is missing '
                           'or incorrect' % key)
        val = default
    return np.atleast_1d(np.asarray(val, dtype=float))


def read_package_parameters(parameter, param_struct=None):
    """Read package parameters from a parameter block (MATLAB lines 10525-10593).

    parameter: dict or SimpleNamespace with parameter fields.
    param_struct: optional existing struct to append to (creates new if omitted).
    Returns updated param_struct.
    """
    if param_struct is None:
        param_struct = SimpleNamespace()

    def xp(key, default=_MANDATORY):
        return _xls_param(parameter, key, default)

    # No fourth argument in MATLAB for any of these: they are mandatory.
    param_struct.C_pkg_board = xp('C_p') * 1e-9
    param_struct.R_diepad = xp('R_d')
    param_struct.a_thru = xp('A_v')
    param_struct.a_fext = xp('A_fe')
    param_struct.a_next = xp('A_ne')

    # z_p_tx_cases: MATLAB transposes → shape (ncases, mele).
    # The spreadsheet stores rows = package segments, columns = cases; the engine
    # indexes [case, :]. MATLAB applies .' to all four z_p keywords
    # (com_ieee8023_4p15p0.m L10678/10689/10695/10701).
    raw = xp('z_p (TX)')
    z_p_tx = np.atleast_2d(raw).T
    ncases, mele = z_p_tx.shape
    if mele == 2:
        param_struct.flex = 2
    elif mele == 4:
        param_struct.flex = 4
    elif mele == 1:
        param_struct.flex = 1
    else:
        raise ValueError('config file syntax error: z_p (TX) must have 1, 2, or 4 columns')
    param_struct.z_p_tx_cases = z_p_tx

    def _load_zp(key):
        raw2 = xp(key)                       # mandatory, like z_p (TX)
        arr = np.atleast_2d(raw2).T          # same transpose as z_p (TX) above
        if arr.shape != (ncases, mele):
            raise ValueError('All TX, NEXT, FEXT, Rx cases must agree')
        return arr

    param_struct.z_p_next_cases = _load_zp('z_p (NEXT)')
    param_struct.z_p_fext_cases = _load_zp('z_p (FEXT)')
    param_struct.z_p_rx_cases = _load_zp('z_p (RX)')

    param_struct.pkg_gamma0_a1_a2 = xp('package_tl_gamma0_a1_a2', np.array([0.0, 1.734e-3, 1.455e-4]))
    param_struct.pkg_tau = xp('package_tl_tau', np.array([6.141e-3]))
    # MATLAB: param_struct.pkg_Z_c = xls_parameter(..., 78.2).' -- TRANSPOSED,
    # like the four z_p keywords, and the default is the scalar 78.2.  The
    # port kept the sheet's orientation, so a 4x2 package_Z_c stayed (4,2)
    # where the reference gives (2,4), and the mele check compared the wrong
    # axis: COM Octave read the shipped
    # [92 92 ; 70 70; 80 80; 100 100] into pkg_Z_c(2,4) = [92 70 80 100;
    # 92 70 80 100] while the port raised 'tx rx pairs must have the same
    # number element entries'.  make_full_pkg indexes pkg_Z_c(1,:) for TX and
    # (2,:) for RX, so the orientation is load-bearing.
    raw_zc = xp('package_Z_c', 78.2)
    pkg_Z_c = np.atleast_2d(raw_zc).T
    if pkg_Z_c.shape[1] != mele:
        raise ValueError('tx rx pairs must have the same number element entries as TX, NEXT, FEXT, Rx')
    param_struct.pkg_Z_c = pkg_Z_c

    if mele == 2:
        zeros4 = np.zeros((ncases, 2))
        param_struct.z_p_fext_casesx = np.hstack([param_struct.z_p_fext_cases, zeros4])
        param_struct.z_p_next_casesx = np.hstack([param_struct.z_p_next_cases, zeros4])
        param_struct.z_p_tx_casesx = np.hstack([param_struct.z_p_tx_cases, zeros4])
        param_struct.z_p_rx_casesx = np.hstack([param_struct.z_p_rx_cases, zeros4])
        param_struct.z_p_fext_cases = param_struct.z_p_fext_casesx
        param_struct.z_p_next_cases = param_struct.z_p_next_casesx
        param_struct.z_p_tx_cases = param_struct.z_p_tx_casesx
        param_struct.z_p_rx_cases = param_struct.z_p_rx_casesx
        # MATLAB: [pkg_Z_c' ; [100 100 ; 100 100]]' -- transpose, stack two
        # rows of 100 underneath, transpose back, so a (2,2) pkg_Z_c becomes
        # (2,4).  Stacking on the untransposed array made it (4,2) instead:
        # COM Octave gives [92 70 100 100 ; 92 70 100 100].
        extra = np.array([[100.0, 100.0], [100.0, 100.0]])
        param_struct.pkg_Z_c = np.vstack([param_struct.pkg_Z_c.T, extra]).T

    return param_struct
