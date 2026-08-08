# ============================================================
# MATLAB→Python translation notes for read_package_parameters
# MATLAB lines: 10525–10593
# ============================================================
# xls_parameter inlined as _xls_param — calls into the existing xls_parameter py_impl via import,
# but per project protocol no cross-py_impl imports: inline a minimal version here.
# Actually, xls_parameter is a complex function. We raise NotImplementedError unless
# we can call it.  Per protocol we inline helpers.
#
# This function reads package parameters from a parameter block (spreadsheet struct).
# xls_parameter(parameter, key, bool) — inlined as _xls_param below, but since xls_parameter
# is already implemented, we import it directly (it is a sibling function, but since it is
# a pure utility and the project rule is "no imports from sibling py_impl files" only for
# callee inlining, and xls_parameter is verified — we raise NotImplementedError for the
# full-spreadsheet path, noting this is a parameter-file-loading function).
#
# For test coverage: the function can be called with a dict-like parameter block.
# xls_parameter is inlined as _xls_parameter below (minimal version).
# ============================================================
# z_p_tx_cases shape: MATLAB transposes → shape (ncases, mele).
# mele==2 → flex=2; mele==4 → flex=4; mele==1 → flex=1; else → ValueError.
# When mele==2: expand to 4-column by appending zeros for TX/NEXT/FEXT/RX.
# pkg_Z_c similarly expanded to 4 when mele==2.
# ============================================================

import numpy as np
from types import SimpleNamespace


def _xls_param(parameter, key, optional=True, default=None):
    """Minimal xls_parameter: looks up key in a dict or SimpleNamespace."""
    if isinstance(parameter, dict):
        val = parameter.get(key, default)
    else:
        val = getattr(parameter, key.replace(' ', '_').replace('(', '').replace(')', ''), default)
    if val is None:
        if optional:
            return default
        raise KeyError(f'Required key {key!r} not found in parameter block')
    return np.atleast_1d(np.asarray(val, dtype=float))


def read_package_parameters(parameter, param_struct=None):
    """Read package parameters from a parameter block (MATLAB lines 10525-10593).

    parameter: dict or SimpleNamespace with parameter fields.
    param_struct: optional existing struct to append to (creates new if omitted).
    Returns updated param_struct.
    """
    if param_struct is None:
        param_struct = SimpleNamespace()

    def xp(key, default=None):
        return _xls_param(parameter, key, optional=True, default=default)

    param_struct.C_pkg_board = xp('C_p', np.array([0.0])) * 1e-9
    param_struct.R_diepad = xp('R_d', np.array([50.0]))
    param_struct.a_thru = xp('A_v', np.array([1.0]))
    param_struct.a_fext = xp('A_fe', np.array([0.0]))
    param_struct.a_next = xp('A_ne', np.array([0.0]))

    # z_p_tx_cases: MATLAB transposes → shape (ncases, mele)
    raw = xp('z_p (TX)', np.array([[0.0, 0.0]]))
    z_p_tx = np.atleast_2d(raw)
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
        raw2 = xp(key, np.zeros_like(z_p_tx))
        arr = np.atleast_2d(raw2)
        if arr.shape != (ncases, mele):
            raise ValueError('All TX, NEXT, FEXT, Rx cases must agree')
        return arr

    param_struct.z_p_next_cases = _load_zp('z_p (NEXT)')
    param_struct.z_p_fext_cases = _load_zp('z_p (FEXT)')
    param_struct.z_p_rx_cases = _load_zp('z_p (RX)')

    param_struct.pkg_gamma0_a1_a2 = xp('package_tl_gamma0_a1_a2', np.array([0.0, 1.734e-3, 1.455e-4]))
    param_struct.pkg_tau = xp('package_tl_tau', np.array([6.141e-3]))
    raw_zc = xp('package_Z_c', np.array([[78.2, 78.2]]))
    pkg_Z_c = np.atleast_2d(raw_zc)
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
        extra = np.array([[100.0, 100.0], [100.0, 100.0]])
        param_struct.pkg_Z_c = np.vstack([param_struct.pkg_Z_c, extra])

    return param_struct
