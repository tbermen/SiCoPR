"""
FD_Processing: Frequency-domain metrics — IL fit, FOM_ILD, ICN, FEXT/NEXT noise.
MATLAB lines 1684–2025.
"""

import numpy as np
from types import SimpleNamespace
from scipy.special import erfcinv


_EPS0 = np.nextafter(0.0, 1.0)   # MATLAB eps(0) = 4.9406564584124654e-324


def _W(f, ft, fr, fb):
    """Power weighting function, eq 93A-57: sinc^2 * 4th-order * 8th-order rolloff.

    Transcribed operation for operation from the reference, which is
        Sinc = @(x) sin(pi*x+eps(0))./(pi*x+eps(0));
        W = @(f,ft,fr,fb) 1/fb * Sinc(f/fb).^2 .* (1./(1+(f/ft).^4)) .* (1./(1+(f/fr).^8));
    The order matters: `1/fb * A` rounds the reciprocal first and is not the
    same double as `A / fb`, and the earlier `+1e-300` guards on fb/ft/fr were
    an invented floor.  The previous spelling differed from the reference in
    15 of 32 samples at the last bit; this one is bit-identical.
    """
    x = f / fb
    sinc = np.sin(np.pi * x + _EPS0) / (np.pi * x + _EPS0)
    return (1.0 / fb * sinc**2
            * (1.0 / (1.0 + (f / ft)**4))
            * (1.0 / (1.0 + (f / fr)**8)))


def _neg20log10(x):
    """MATLAB -20*log10(abs(x)).

    No epsilon floor: at |x| = 0 the reference returns +Inf, and every metric
    built on it (a ratio in dB, an interp1 through the bin) inherits that Inf.
    `-20*log10(|x| + 1e-300)` returns ~6000 dB instead, which is finite and
    plausible and therefore worse than the Inf.
    """
    with np.errstate(divide='ignore'):
        return -20.0 * np.log10(np.abs(x))


def _db10_ratio(num, den):
    """MATLAB 10*log10(num/den) -- +Inf at den == 0, with no epsilon floor."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(10.0 * np.log10(np.float64(num) / np.float64(den)))


def _interp1(x, v, xq):
    """MATLAB interp1(x, v, xq) at one point: linear, NaN outside the data.

    Two things np.interp does differently, both of which showed up against the
    reference:

    * np.interp CLAMPS outside [x[0], x[-1]] and returns the end value, where
      interp1 with no extrapolation argument returns NaN.  A channel whose
      frequency axis starts above Nyquist came back with the first bin's loss
      instead of NaN.
    * np.interp carries a NaN-avoidance retry: when `slope*(xq-x[i]) + v[i]`
      is NaN it recomputes from the right-hand end.  interp1 has no such
      fallback, so where v[i] is +Inf (a zero S-parameter bin) it returns NaN
      and np.interp returned +Inf.
      COM Octave: interp1([0 1 2 3],[1 Inf 3 4],[0 0.5 1 1.5 2]) ->
        [NaN Inf NaN NaN 3];  np.interp gave [1 Inf Inf Inf 3].

    So spell out interp1's own `s*dy(k) + v(k)` on the bracket `lookup` picks.
    """
    xa = np.asarray(x, dtype=float).ravel()
    va = np.asarray(v, dtype=float).ravel()
    if xq < xa[0] or xq > xa[-1]:
        return float('nan')
    k = int(np.searchsorted(xa, xq, side='right')) - 1
    k = min(max(k, 0), len(xa) - 2)
    s = (xq - xa[k]) / (xa[k + 1] - xa[k])
    with np.errstate(invalid='ignore'):      # Inf-Inf is a result, not a warning
        return float(s * (va[k + 1] - va[k]) + va[k])


def _find_idx_ge(faxis, f):
    """0-based index of first element >= f; clamps to len-1 if none found."""
    idx = int(np.searchsorted(faxis, f, side='left'))
    return min(idx, len(faxis) - 1)


def _find_idx_le(faxis, f):
    """0-based index of last element <= f; clamps to 0 if none found."""
    idx = int(np.searchsorted(faxis, f, side='right')) - 1
    return max(idx, 0)


def FD_Processing(chdata, output_args, param, OP, SDDp2p=None, DO_ONCE=True,
                  _get_ILN_fn=None, _get_ILN_cmp_td_fn=None,
                  _capture_RIL_RILN_fn=None,
                  _Bessel_Thomson_Filter_fn=None, _Butterworth_Filter_fn=None,
                  _Raised_Cosine_Filter_fn=None, _Tx_FFE_Filter_fn=None):

    # ── Amplitude assignment ────────────────────────────────────────────────
    ptc_i = int(param.package_testcase_i)  # 1-based
    if getattr(OP, 'WC_PORTZ', False):
        tx_sel = int(param.Tx_rd_sel) - 1  # convert to 0-based
        A_thru = float(np.asarray(param.a_thru).ravel()[tx_sel])
        A_fext = float(np.asarray(param.a_fext).ravel()[tx_sel])
        A_next = float(np.asarray(param.a_next).ravel()[tx_sel])
    else:
        pkg_tc = int(np.asarray(OP.pkg_len_select).ravel()[ptc_i - 1]) - 1  # 0-based index
        A_thru = float(np.asarray(param.a_thru).ravel()[pkg_tc])
        A_fext = float(np.asarray(param.a_fext).ravel()[pkg_tc])
        A_next = float(np.asarray(param.a_next).ravel()[pkg_tc])

    # MATLAB loops `for i=1:param.number_of_s4p_files`, not over every element of
    # chdata: with PSDRXCAL the noise path sits past that count and the reference
    # never touches it.  Iterating the whole list gave it an A, a PWF and -- for a
    # crosstalk type -- a place in the ICN power sum (ICN_mV 4.2335 vs 3.1327 on a
    # 3-channel/count-2 case).
    n_files = int(param.number_of_s4p_files)
    chdata_in_scope = chdata[:n_files]

    for ch in chdata_in_scope:
        if ch.type == 'THRU':
            ch.A = A_thru
            ch.Aicn = A_thru
        elif ch.type == 'FEXT':
            ch.A = A_fext
            ch.Aicn = float(param.a_icn_fext)
        elif ch.type == 'NEXT':
            ch.A = A_next
            ch.Aicn = float(param.a_icn_next)
        else:  # NOISE
            ch.A = 1.0
            ch.Aicn = 1.0

    if getattr(OP, 'TDMODE', False):
        for ch in chdata_in_scope:
            # MATLAB indices 11,10 (1-based) → Python 10,9 (0-based)
            ch.delta_f = float(np.asarray(ch.faxis).ravel()[10] - np.asarray(ch.faxis).ravel()[9])

    if not DO_ONCE:
        return chdata, output_args

    # ── Initialize output fields ────────────────────────────────────────────
    for field in ('fitted_IL_dB_at_Fnq', 'cable__assembley_loss', 'loss_with_PCB',
                  'VIP_to_VMP_IL_dB_at_Fnq', 'IL_dB_channel_only_at_Fnq',
                  'VTF_loss_dB_at_Fnq', 'IL_db_die_to_die_at_Fnq',
                  'FOM_TDILN', 'TD_ILN', 'FOM_RILN', 'FOM_ILD'):
        setattr(output_args, field, [])

    if not getattr(OP, 'GET_FD', True):
        return chdata, output_args

    # ── Apply receiver filter (INCLUDE_FILTER path) ─────────────────────────
    if getattr(OP, 'INCLUDE_FILTER', False):
        for ch in chdata_in_scope:
            f = np.asarray(ch.faxis).ravel()
            H_bt = _Bessel_Thomson_Filter_fn(param, f, OP.Bessel_Thomson)
            H_bw = _Butterworth_Filter_fn(param, f, OP.Butterworth)
            H_rc = _Raised_Cosine_Filter_fn(param, f, OP.Raised_Cosine)
            H_tx = _Tx_FFE_Filter_fn(param, f, param.Pkg_TXFFE_preset)
            ch.sdd21 = np.asarray(ch.sdd21).ravel() * H_bw * H_bt * H_rc * H_tx

    # ── Main FD loop ────────────────────────────────────────────────────────
    f2 = float(param.f2)
    f2_ild = float(param.f2_ild)
    f1 = float(param.f1)
    fr = float(param.f_r) * float(param.fb)
    fb = float(param.fb)

    MDFEXT_ICN = 0.0
    MDNEXT_ICN = 0.0
    P_signal = 0.0
    ILD_magft = np.array([])

    nf = len(np.asarray(chdata[0].sdd21f).ravel())
    PSXT = np.zeros(nf)
    MDFEXT = np.zeros(nf)
    MDNEXT = np.zeros(nf)

    for ch in chdata_in_scope:
        faxis = np.asarray(ch.faxis).ravel()

        # Frequency boundary indices (0-based)
        idx_f2 = _find_idx_ge(faxis, f2)
        idx_f1 = _find_idx_le(faxis, f1)
        idx_f2_ild = _find_idx_ge(faxis, f2_ild)

        # Power weighting function (eq 93A-57)
        temp_angle = float(param.samples_per_ui) * float(param.sample_dt) * np.pi * faxis
        if faxis[0] == 0.0:
            temp_angle[0] = 1e-20
        SINC = np.sin(temp_angle) / temp_angle
        PWF = SINC**2 / (1.0 + (faxis / float(ch.ftr))**4) / (1.0 + (faxis / fr)**8)

        delta_f = float(faxis[10] - faxis[9])
        ch.delta_f = delta_f

        if ch.type == 'THRU':
            sdd21f = np.asarray(ch.sdd21f).ravel()
            Il_dB = _neg20log10(sdd21f)
            fslice = faxis[idx_f1:idx_f2 + 1]

            P_signal = 2.0 * delta_f * np.sum(
                _W(fslice, float(ch.ftr), fr, fb) * 10.0**(-Il_dB[idx_f1:idx_f2 + 1] / 10.0)
            )
            output_args.P_signal_sigma_FD = float(np.sqrt(max(P_signal, 0.0)))
            output_args.P_signal_FD = float(P_signal)

            # SCMR_FD (common-mode rejection metrics)
            Q_inv = lambda x: erfcinv(2.0 * x) * np.sqrt(2.0)
            sigma_X = float(param.sigma_X)
            qv2 = Q_inv(float(param.P_peak))**2
            scd21_o = np.asarray(ch.scd21_orig).ravel()[idx_f1:idx_f2 + 1]
            sdc21_o = np.asarray(ch.sdc21_orig).ravel()[idx_f1:idx_f2 + 1]
            W_slice = _W(fslice, float(ch.ftr), fr, fb)
            EC_CD = 2.0 * delta_f * np.sum(W_slice * np.abs(scd21_o)**2 * sigma_X**2 * qv2)
            EC_DC = 2.0 * delta_f * np.sum(W_slice * np.abs(sdc21_o)**2 * sigma_X**2 * qv2)
            # A perfectly balanced channel has scd21_orig/sdc21_orig identically
            # zero, so EC_CD/EC_DC are 0 and the reference reports Inf dB of
            # common-mode rejection.  `/(EC + 1e-300)` reported ~2993 dB.
            output_args.SCMR_FD_CD_ch_dB = _db10_ratio(P_signal, EC_CD)
            output_args.SCMR_FD_DC_ch_dB = _db10_ratio(P_signal, EC_DC)

            # ILD fit over [f1, f2_ild]
            ILD_magft, ch.fit_f2_ild = _get_ILN_fn(
                sdd21f[idx_f1:idx_f2_ild + 1], faxis[idx_f1:idx_f2_ild + 1])
            _, ch.fit_orig = _get_ILN_fn(sdd21f, faxis)

            fnq = 1.0 / (float(param.ui) * 2.0)
            if faxis[idx_f2] >= fnq:
                ch.fit_ILatNq = _interp1(faxis, -np.asarray(ch.fit_orig).ravel(), fnq)
                output_args.fitted_IL_dB_at_Fnq = ch.fit_ILatNq
            else:
                ch.fit_ILatNq = []
                output_args.fitted_IL_dB_at_Fnq = []

            fit_f2_ild_arr = np.asarray(ch.fit_f2_ild).ravel()
            output_args.fitted_IL_dB_at_F2_ild = float(-fit_f2_ild_arr[-1])

            if faxis[-1] >= fnq:
                ch.ILatNq = _interp1(faxis, _neg20log10(sdd21f), fnq)
            else:
                ch.ILatNq = []
            output_args.IL_dB_channel_only_at_Fnq = ch.ILatNq if isinstance(ch.ILatNq, list) else float(ch.ILatNq)

            if getattr(OP, 'include_pcb', False):
                output_args.cable__assembley_loss = _interp1(
                    faxis, _neg20log10(np.asarray(ch.sdd21_orig).ravel()), fnq)
                output_args.loss_with_PCB = _interp1(
                    faxis, _neg20log10(np.asarray(ch.sdd21_raw).ravel()), fnq)

            # TD/RILN metrics
            if getattr(OP, 'COMPUTE_TDILN', False) or getattr(OP, 'COMPUTE_RILN', False):
                ILD_td, ch.fit, TD_ILN = _get_ILN_cmp_td_fn(
                    sdd21f[idx_f1:idx_f2 + 1], faxis[idx_f1:idx_f2 + 1], OP, param, ch.A)
                FOM_TDILN = float(TD_ILN.SNR_ISI_FOM_PDF)

            if getattr(OP, 'COMPUTE_TDILN', False):
                output_args.FOM_TDILN = FOM_TDILN
                output_args.TD_ILN = TD_ILN

            if getattr(OP, 'COMPUTE_RILN', False):
                RIL_struct = _capture_RIL_RILN_fn(chdata)
                RILN_dB = np.asarray(RIL_struct.RILN_dB).ravel()
                f_span = float(param.f2) - float(param.f1)
                FOM_RILN = float(np.sqrt(delta_f / f_span *
                                          np.sum(PWF[idx_f1:idx_f2] * RILN_dB[idx_f1:idx_f2]**2)))
                output_args.FOM_RILN = FOM_RILN

            # eq 93A-56: FOM_ILD
            FOM_ILD = float(np.sqrt(delta_f / fb *
                                     np.sum(PWF[idx_f1:idx_f2_ild + 1] * np.asarray(ILD_magft)**2)))
            output_args.FOM_ILD = FOM_ILD

        elif ch.type in ('FEXT', 'NEXT'):
            sdd21f = np.asarray(ch.sdd21f).ravel()

            if ch.type == 'FEXT':
                MDFEXT = np.sqrt(np.abs(sdd21f)**2 + MDFEXT**2)
                MDFEXT_dBloss = _neg20log10(MDFEXT)
                MDFEXT_ICN = float(np.sqrt(
                    2.0 * delta_f / fb *
                    np.sum(ch.Aicn**2 * PWF[idx_f1:idx_f2 + 1] *
                           np.abs(MDFEXT[idx_f1:idx_f2 + 1])**2)))
                output_args.MDFEXT_ICN_92_47_mV = MDFEXT_ICN * 1000.0

                PS_MDFEXT = _W(faxis, float(ch.ftr), fr, fb) * 10.0**(-MDFEXT_dBloss / 10.0)
                sqrd_mdfext = 2.0 * delta_f * np.sum(PS_MDFEXT[idx_f1:idx_f2 + 1]) * float(param.sigma_X)**2
                # No FEXT coupling at all makes sqrd_mdfext 0, where the
                # reference reports Inf dB SNR; `+1e-300` reported ~2993 dB.
                output_args.SNR_MDFEXT = _db10_ratio(P_signal, sqrd_mdfext)

            else:  # NEXT
                MDNEXT = np.sqrt(np.abs(sdd21f)**2 + MDNEXT**2)
                MDNEXT_ICN = float(np.sqrt(
                    2.0 * delta_f / fb *
                    np.sum(ch.Aicn**2 * PWF[idx_f1:idx_f2 + 1] *
                           np.abs(MDNEXT[idx_f1:idx_f2 + 1])**2)))
                output_args.MDNEXT_ICN_92_46_mV = MDNEXT_ICN * 1000.0

            PSXT = np.sqrt((np.abs(sdd21f) * ch.Aicn)**2 + PSXT**2)
            ICN = float(np.sqrt(
                2.0 * delta_f / fb *
                np.sum(PWF[idx_f1:idx_f2 + 1] * np.abs(PSXT[idx_f1:idx_f2 + 1])**2)))
            output_args.ICN_mV = ICN * 1000.0

        # else NOISE → skip (no PSXT update, as in MATLAB `continue`)

    # ── Final Nyquist loss interpolation ────────────────────────────────────
    # MATLAB reads chdata(i).faxis, where i is whatever the loop above left
    # behind -- the LAST channel it visited, which is number_of_s4p_files, not
    # the last element of chdata.
    fax_last = np.asarray(chdata_in_scope[-1].faxis).ravel()
    fnq = 1.0 / (float(param.ui) * 2.0)
    ch0 = chdata[0]

    def _il_at_fnq(sig):
        # interp1 returns NaN outside the data range, which the reference leaves
        # unguarded here; _interp1 reproduces that rather than clamping.
        return _interp1(fax_last, _neg20log10(np.asarray(sig).ravel()), fnq)

    output_args.VTF_loss_dB_at_Fnq = _il_at_fnq(ch0.sdd21)
    output_args.IL_db_die_to_die_at_Fnq = _il_at_fnq(ch0.sdd21p_nodie)
    output_args.VIP_to_VMP_IL_dB_at_Fnq = _il_at_fnq(ch0.sdd21p)

    return chdata, output_args
