# ============================================================
# MATLAB→Python translation notes for process_sxp
# MATLAB lines: 9093–9389
# ============================================================
# Indexing: MATLAB 1-based → 0-based Python.
#   i=1..num_files → i=0..num_files-1
#   ipsl=1..len(port_sel) → ipsl=0..len(port_sel)-1
#   izt=1..len(param.Z_t) → izt=0..len(param.Z_t)-1
#   First-channel guard: MATLAB i==1 → Python i==0
#
# S.Parameters storage: MATLAB S.Parameters(row,col,freq) [1-based NumPorts×NumPorts×Nf]
#   → Python shape (Nf, NumPorts, NumPorts), access S.Parameters[freq_idx, row, col]
#
# Sfield: '_orig' (default, no board) or '_raw' (when OP.SHOW_BRD=True)
#
# MATLAB struct arrays TDR_results(izt,ipsl) → Python list-of-lists tdr_results[izt][ipsl]
# MATLAB chdata(i).TDR11(izt) → Python chdata[i].TDR11 list indexed by izt (0-based)
#
# OP.Report_Modal_ERL: MATLAB strcmpi(x,'enable') || strcmpi(x,'provisional')
#   → x.lower() in ('enable', 'provisional')
#
# All display/plot code (DISPLAY_WINDOW, DEBUG) is omitted — no output.
#
# get_RAW_FIR stub: returns zeros (OP.AUTO_TFX logic uses it only optionally)
# plot_modal stub: returns empty dict (no modal return struct)
# add_pkg_with_die stub: returns S unchanged and TX_RL=0
# get_TDR is the primary callee — must be stubbed here, no import from get_TDR/py_impl
# ============================================================

import numpy as np
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Callee stubs
# ---------------------------------------------------------------------------

def _plot_modal(param, OP, chdata):
    """Stub: no modal return struct."""
    return {}


def _add_pkg_with_die(S, mode, param, OP):
    """Stub: returns S unchanged and TX_RL=0 complex array."""
    TX_RL = np.zeros(len(S.Frequencies), dtype=complex)
    return S, TX_RL


def _get_RAW_FIR(sdd21, faxis, OP, param):
    """Stub: returns zeros impulse and time axis."""
    N = 64
    dt = param.sample_dt
    ir = np.zeros(N)
    t = np.arange(N) * dt
    return ir, t


def _get_TDR(S, OP, param, ZT, nport):
    """Stub: returns a minimal TDR_results struct."""
    N = max(8, len(S.Frequencies) // 4)
    return SimpleNamespace(
        tdr=np.ones(N) * ZT,
        t=np.linspace(0, 1e-9, N),
        avgZport=float(ZT),
        ERL=0.0,
        ERLRMS=0.0,
        ptdr_RL=np.zeros(N),
        WC_ptdr_samples_t=np.zeros(1),
        WC_ptdr_samples=np.zeros(1),
    )


# ---------------------------------------------------------------------------
# Helper: build S-parameter struct from chdata fields
# ---------------------------------------------------------------------------

def _build_S_struct(ch, Sfield, mode, param, Nf):
    """Build S-parameter SimpleNamespace for a given mode ('dd', 'cd', 'dc', 'cc').

    Sfield: '_orig' or '_raw'
    mode: 'dd', 'cd', 'dc', 'cc'
    Returns S with .Frequencies, .Parameters (Nf×NumPorts×NumPorts), .Impedance, .NumPorts
    """
    Z0 = float(param.Z0)
    if mode == 'dd':
        Zref = Z0 * 2
        port_map = [('sdd11', 'sdd12', 'sdd21', 'sdd22')]
    elif mode == 'cd':
        Zref = Z0 * 2
        port_map = [('scd11', 'scd12', 'scd21', 'scd22')]
    elif mode == 'dc':
        Zref = Z0 / 2
        port_map = [('sdc11', 'sdc12', 'sdc21', 'sdc22')]
    elif mode == 'cc':
        Zref = Z0 / 2
        port_map = [('scc11', 'scc12', 'scc21', 'scc22')]
    else:
        raise ValueError(f'Unknown mode: {mode}')

    field_names = port_map[0]  # (s11, s12, s21, s22) prefixes

    S = SimpleNamespace()
    S.Frequencies = ch.faxis
    S.Impedance = Zref

    if int(param.FLAG.S2P) == 0:
        # 2-port (4-port channel file)
        S.NumPorts = 2
        S.Parameters = np.zeros((Nf, 2, 2), dtype=complex)
        S.Parameters[:, 0, 0] = np.asarray(getattr(ch, field_names[0] + Sfield)).ravel()
        S.Parameters[:, 0, 1] = np.asarray(getattr(ch, field_names[1] + Sfield)).ravel()
        S.Parameters[:, 1, 0] = np.asarray(getattr(ch, field_names[2] + Sfield)).ravel()
        S.Parameters[:, 1, 1] = np.asarray(getattr(ch, field_names[3] + Sfield)).ravel()
    else:
        # 1-port (2-port channel file)
        S.NumPorts = 1
        S.Parameters = np.zeros((Nf, 1, 1), dtype=complex)
        S.Parameters[:, 0, 0] = np.asarray(getattr(ch, field_names[0] + Sfield)).ravel()

    return S


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def process_sxp(param, OP, chdata, SDDch,
                _plot_modal_fn=None,
                _add_pkg_with_die_fn=None,
                _get_RAW_FIR_fn=None,
                _get_TDR_fn=None):
    """Process S-parameter data: compute TDR/ERL for the first channel.

    MATLAB lines 9093–9389.

    param: SimpleNamespace — Z0, FLAG.S2P, RL_sel, Z_t, tfx, package_testcase_i,
           TR_TDR, ui, samples_per_ui, sample_dt, etc.
    OP: SimpleNamespace — TDR, PTDR, DISPLAY_WINDOW, DEBUG, SHOW_BRD, AUTO_TFX,
        TDR_W_TXPKG, ERL, Report_Modal_ERL, etc.
    chdata: list of SimpleNamespace — each has faxis, sddXX_orig/_raw, etc.
    SDDch: unused (kept for signature compatibility with MATLAB caller)

    Returns (chdata, param) — chdata[0] is updated with TDR11/TDR22 sub-structs.
    Dependency injection via optional *_fn params.
    """
    pm_fn = _plot_modal_fn or _plot_modal
    pkg_fn = _add_pkg_with_die_fn or _add_pkg_with_die
    raw_fir_fn = _get_RAW_FIR_fn or _get_RAW_FIR
    tdr_fn = _get_TDR_fn or _get_TDR

    num_files = len(chdata)

    for i in range(num_files):
        ch = chdata[i]

        if int(param.package_testcase_i) == 1 and i == 0:
            if getattr(OP, 'TDR', False) and i == 0:
                Nf = len(ch.faxis)

                # ---- Determine Sfield ----
                Sfield = '_raw' if getattr(OP, 'SHOW_BRD', False) else '_orig'

                # ---- Build S-parameter structs ----
                S = _build_S_struct(ch, Sfield, 'dd', param, Nf)
                SCD = _build_S_struct(ch, Sfield, 'cd', param, Nf)
                SDC = _build_S_struct(ch, Sfield, 'dc', param, Nf)
                SCC = _build_S_struct(ch, Sfield, 'cc', param, Nf)

                # ---- Modal plots / CM mask report (returns a SimpleNamespace or None) ----
                return_struct = pm_fn(param, OP, chdata)
                if return_struct is not None:
                    for field in ('Rlcc_179mm', 'Rlcc_179mm_fail', 'Rlcc_178mm',
                                  'Rlcc_178mm_fail', 'Rlcd_179mm', 'Rlcd_179mm_fail',
                                  'Rldc_179mm', 'Rldc_179mm_fail'):
                        if hasattr(return_struct, field):
                            setattr(chdata[0], field, getattr(return_struct, field))

                # ---- TX package insertion ----
                if getattr(OP, 'TDR_W_TXPKG', False):
                    if getattr(OP, 'ERL', 0) == 2:
                        raise RuntimeError("Cannot add package to s2p files. ERL==2 not supported with TDR_W_TXPKG=1")
                    S, chdata[i].TX_RL = pkg_fn(S, 'dd', param, OP)

                # ---- Port selection ----
                if int(param.FLAG.S2P):
                    port_sel = [0]  # 0-based (MATLAB: [1])
                else:
                    port_sel = [0, 1]  # 0-based (MATLAB: [1 2])
                    if getattr(OP, 'AUTO_TFX', False):
                        fir4del, tu = raw_fir_fn(
                            np.asarray(ch.sdd12_orig).ravel(), S.Frequencies, OP, param)
                        # MATLAB find(fir4del==max(fir4del),1): max() SKIPS
                        # NaN and a NaN never equals the max, so the search
                        # lands on the first non-NaN peak.  np.argmax returns
                        # the index of the first NaN instead.
                        # COM Octave: find(x==max(x),1) on [NaN 1 3 2] -> 3
                        #   (1-based; np.argmax gave 0); on [1 3 NaN 3] -> 2
                        #   (np.argmax gave 2, the NaN); on [NaN NaN] -> EMPTY.
                        fir4del = np.asarray(fir4del)
                        if np.all(np.isnan(fir4del)):
                            raise ValueError(
                                'process_sxp: AUTO_TFX - get_RAW_FIR returned '
                                'an all-NaN response, so find(...,1) is empty '
                                'and param.tfx(2) cannot be set')
                        pix = int(np.nanargmax(fir4del))
                        param.tfx[1] = 2 * tu[pix]

                # MATLAB passes OP BY VALUE, so the TDR-only overrides below never
                # escape process_sxp — 4p15p0 L9311 says so outright:
                #   "Only for TDR not returned out of process_sxp function"
                # Python passes by reference, so assigning to OP here leaked the TDR
                # settings into every later stage. In particular the truncation
                # threshold went 1e-3 (config default) -> 1e-5 for the rest of the
                # run, which keeps far more impulse-response tail, lengthens the
                # pulse response, inflates residual ISI and therefore sigma_e, and
                # biased FOM LOW on 95.7% of the 208 reference cases.
                # Rebinding to a shallow copy reproduces MATLAB's by-value scope:
                # the rest of process_sxp sees the TDR values, the caller does not.
                OP = SimpleNamespace(**vars(OP))
                OP.impulse_response_truncation_threshold = 1e-5
                Z_t = np.atleast_1d(param.Z_t)
                n_zt = len(Z_t)

                # Initialize TDR result arrays on chdata[0]
                if not hasattr(chdata[0], 'TDR11'):
                    chdata[0].TDR11 = [SimpleNamespace() for _ in range(n_zt)]
                if not hasattr(chdata[0], 'TDR22'):
                    chdata[0].TDR22 = [SimpleNamespace() for _ in range(n_zt)]

                # Allocate 2D list for TDR results [izt][ipsl]
                tdr_results = [[None] * len(port_sel) for _ in range(n_zt)]
                tdr_cd_results = [[None] * len(port_sel) for _ in range(n_zt)]
                tdr_dc_results = [[None] * len(port_sel) for _ in range(n_zt)]
                tdr_cc_results = [[None] * len(port_sel) for _ in range(n_zt)]
                can_plot_cd = [[False] * len(port_sel) for _ in range(n_zt)]

                report_modal = (str(getattr(OP, 'Report_Modal_ERL', '')).lower()
                                in ('enable', 'provisional'))

                # ---- Interp override ----
                OP.interp_sparam_mag = 'linear_trend_to_DC'
                OP.interp_sparam_phase = 'extrap_cubic_to_dc_linear_to_inf'

                for ipsl_idx, ipsl in enumerate(port_sel):
                    param.RL_sel = ipsl  # 0-based

                    for izt in range(n_zt):
                        ZT = float(Z_t[izt])
                        tdr_results[izt][ipsl_idx] = tdr_fn(S, OP, param, ZT, ipsl)

                        # MATLAB isequal() is EXACT; the 1e-6 tolerance the
                        # port used ran the CD/DC TDR on impedances the
                        # reference treats as different (and reports NaN for).
                        # COM Octave: isequal(100, 100.0000005) -> 0
                        #   (abs(a-b) < 1e-6 said True);
                        #   isequal(100, 100) -> 1
                        if float(param.Z0) == ZT:
                            can_plot_cd[izt][ipsl_idx] = True
                            tdr_cd_results[izt][ipsl_idx] = tdr_fn(SCD, OP, param, ZT, ipsl)
                            tdr_dc_results[izt][ipsl_idx] = tdr_fn(SDC, OP, param, ZT / 4, ipsl)
                        else:
                            can_plot_cd[izt][ipsl_idx] = False
                            tdr_cd_results[izt][ipsl_idx] = SimpleNamespace(ERL=float('nan'), ERLRMS=float('nan'))
                            tdr_dc_results[izt][ipsl_idx] = SimpleNamespace(ERL=float('nan'), ERLRMS=float('nan'))

                        tdr_cc_results[izt][ipsl_idx] = tdr_fn(SCC, OP, param, ZT / 4, ipsl)

                    # ---- Store results in chdata ----
                    for izt in range(n_zt):
                        ZSR_field = tdr_results[izt][ipsl_idx]
                        if ipsl_idx == 0:  # port 1
                            chdata[i].TDR11[izt].ZSR = ZSR_field.tdr
                            chdata[i].TDR11[izt].t = ZSR_field.t
                            chdata[i].TDR11[izt].avgZport = float(ZSR_field.avgZport)
                            if getattr(OP, 'PTDR', False):
                                if not hasattr(chdata[i], 'PDTR11'):
                                    chdata[i].PDTR11 = [SimpleNamespace() for _ in range(n_zt)]
                                if hasattr(ZSR_field, 'ptdr_RL'):
                                    chdata[i].PDTR11[izt].ptdr = ZSR_field.ptdr_RL
                        else:  # port 2
                            chdata[i].TDR22[izt].ZSR = ZSR_field.tdr
                            chdata[i].TDR22[izt].t = ZSR_field.t
                            chdata[i].TDR22[izt].avgZport = float(ZSR_field.avgZport)
                            if getattr(OP, 'PTDR', False):
                                if not hasattr(chdata[i], 'PDTR22'):
                                    chdata[i].PDTR22 = [SimpleNamespace() for _ in range(n_zt)]
                                if hasattr(ZSR_field, 'ptdr_RL'):
                                    chdata[i].PDTR22[izt].ptdr = ZSR_field.ptdr_RL

                        # ERL/ERLRMS storage
                        if getattr(OP, 'PTDR', False) and i == 0:
                            if ipsl_idx == 0:
                                chdata[i].TDR11[izt].ERL = getattr(ZSR_field, 'ERL', [])
                                chdata[i].TDR11[izt].ERLRMS = getattr(ZSR_field, 'ERLRMS', [])
                                if report_modal:
                                    chdata[i].TDR11[izt].ERL_CD = getattr(tdr_cd_results[izt][0], 'ERL', [])
                                    chdata[i].TDR11[izt].ERL_DC = getattr(tdr_dc_results[izt][0], 'ERL', [])
                                    chdata[i].TDR11[izt].ERL_CC = getattr(tdr_cc_results[izt][0], 'ERL', [])
                                    chdata[i].TDR11[izt].ERLRMS_CD = getattr(tdr_cd_results[izt][0], 'ERLRMS', [])
                                    chdata[i].TDR11[izt].ERLRMS_DC = getattr(tdr_dc_results[izt][0], 'ERLRMS', [])
                                    chdata[i].TDR11[izt].ERLRMS_CC = getattr(tdr_cc_results[izt][0], 'ERLRMS', [])
                            else:
                                if int(param.FLAG.S2P) == 0:
                                    chdata[i].TDR22[izt].ERL = getattr(ZSR_field, 'ERL', [])
                                    chdata[i].TDR22[izt].ERLRMS = getattr(ZSR_field, 'ERLRMS', [])
                                    if report_modal:
                                        chdata[i].TDR22[izt].ERL_CD = getattr(tdr_cd_results[izt][1], 'ERL', [])
                                        chdata[i].TDR22[izt].ERL_DC = getattr(tdr_dc_results[izt][1], 'ERL', [])
                                        chdata[i].TDR22[izt].ERL_CC = getattr(tdr_cc_results[izt][1], 'ERL', [])
                                        chdata[i].TDR22[izt].ERLRMS_CD = getattr(tdr_cd_results[izt][0], 'ERLRMS', [])
                                        chdata[i].TDR22[izt].ERLRMS_DC = getattr(tdr_dc_results[izt][0], 'ERLRMS', [])
                                        chdata[i].TDR22[izt].ERLRMS_CC = getattr(tdr_cc_results[izt][0], 'ERLRMS', [])
                                else:
                                    chdata[i].TDR22[izt].ERL = []
                                    chdata[i].TDR22[izt].ERLRMS = []
                                    if report_modal:
                                        for fld in ('ERL_CD', 'ERL_DC', 'ERL_CC',
                                                    'ERLRMS_CD', 'ERLRMS_DC', 'ERLRMS_CC'):
                                            setattr(chdata[i].TDR22[izt], fld, [])
                        else:
                            chdata[i].TDR11[izt].ERL = []
                            chdata[i].TDR11[izt].ERLRMS = []
                            chdata[i].TDR22[izt].ERL = []
                            chdata[i].TDR22[izt].ERLRMS = []
                            if report_modal:
                                for tdr_field in (chdata[i].TDR11[izt], chdata[i].TDR22[izt]):
                                    for fld in ('ERL_CD', 'ERL_DC', 'ERL_CC',
                                                'ERLRMS_CD', 'ERLRMS_DC', 'ERLRMS_CC'):
                                        setattr(tdr_field, fld, [])

    # TDR_ERL_Processing expects chdata[0].TDR11 as a flat namespace (MATLAB
    # struct access without index). Unwrap the first Z_t element.
    if hasattr(chdata[0], 'TDR11') and isinstance(chdata[0].TDR11, list):
        chdata[0].TDR11 = chdata[0].TDR11[0]
    if hasattr(chdata[0], 'TDR22') and isinstance(chdata[0].TDR22, list):
        chdata[0].TDR22 = chdata[0].TDR22[0]

    return chdata, param


if __name__ == '__main__':
    import numpy as np
    from types import SimpleNamespace

    N = 32
    f = np.linspace(0.1e9, 26.5625e9, N)
    zeros = np.zeros(N, dtype=complex)
    ones = np.ones(N, dtype=complex) * 0.01

    ch = SimpleNamespace(
        faxis=f,
        sdd11_orig=ones.copy(), sdd12_orig=ones.copy(),
        sdd21_orig=ones.copy(), sdd22_orig=ones.copy(),
        scd11_orig=zeros.copy(), scd12_orig=zeros.copy(),
        scd21_orig=zeros.copy(), scd22_orig=zeros.copy(),
        sdc11_orig=zeros.copy(), sdc12_orig=zeros.copy(),
        sdc21_orig=zeros.copy(), sdc22_orig=zeros.copy(),
        scc11_orig=zeros.copy(), scc12_orig=zeros.copy(),
        scc21_orig=zeros.copy(), scc22_orig=zeros.copy(),
        base='test_ch',
    )
    param = SimpleNamespace(
        package_testcase_i=1, FLAG=SimpleNamespace(S2P=0),
        RL_sel=0, Z_t=np.array([100.0]), Z0=50.0,
        TR_TDR=0.025, tfx=np.array([0.0, 0.0]),
        ui=1.0/53.125e9, ndfe=4, N_bx=4, beta_x=0.0, Grr=1, rho_x=0.1,
        levels=4, specBER=1e-4, Tukey_Window=0,
        samples_per_ui=4, sample_dt=1.0/(2*26.5625e9),
        fb=53.125e9, fb_BT_cutoff=0.473,
    )
    OP = SimpleNamespace(
        TDR=True, PTDR=False, DISPLAY_WINDOW=False, DEBUG=False,
        SHOW_BRD=False, AUTO_TFX=False, TDR_W_TXPKG=False,
        ERL=1, Report_Modal_ERL='disable',
        N=10, T_k=1e-9, BinSize=1e-3, cb_Guassian=True,
    )
    chdata_out, param_out = process_sxp(param, OP, [ch], None)
    print(f"TDR11[0].ZSR length: {len(chdata_out[0].TDR11[0].ZSR)}")
    print('Smoke test PASSED')
