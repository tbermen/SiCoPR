# ============================================================
# MATLAB→Python translation notes for get_TDR
# MATLAB lines: 6904–7220
# ============================================================
# Parameter `np` renamed to `nport` (avoids shadowing numpy).
# S.Parameters shape convention: (NumFreq, NumPorts, NumPorts) in Python
#   (transposed from MATLAB's (NumPorts, NumPorts, NumFreq)).
#   Access: S.Parameters[i_freq, row, col]  ← matches MATLAB S.Parameters(row,col,i)+1
# RL_sel: 0-based in Python (MATLAB 1-based → subtract 1 at call site).
#   Internally: other_port = 1 - rl_sel (for 2-port; only ports 0 and 1).
# MATLAB find(t>=x,1,'first') → np.searchsorted(t, x) → 0-based index
# MATLAB filter(ones(1,M),1,x) → lfilter(ones(M),1,x)
# MATLAB 1-based loop `for ki=1:M` → Python `for ki in range(M)` with 0-based slice
# Tukey_Window stub: returns ones (matching MATLAB override in this function)
# TDR_RL formula: direct translation of long algebraic expression
# PTDR phase search: 0-based ki, PTDR.pulse[ki::M]
# iscolumn checks: replaced by .ravel() for all frequency-domain arrays
# ============================================================

import numpy as np
from scipy.signal import lfilter
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Callee stubs
# ---------------------------------------------------------------------------

def _Bessel_Thomson_Filter(param, faxis, enable):
    if not enable:
        return np.ones(len(faxis))
    f0 = param.fb_BT_cutoff * param.fb
    s = 1j * np.asarray(faxis) / (f0 + 1e-300)
    b = [105, 105, 45, 10, 1]
    return np.abs(b[0] / np.polyval(b, s))


def _Butterworth_Filter(param, faxis, enable):
    if not enable:
        return np.ones(len(faxis))
    f0 = param.fb / 2.0
    return 1.0 / np.sqrt(1.0 + (np.asarray(faxis) / (f0 + 1e-300)) ** 8)


def _Tukey_Window(faxis, param):
    """Stub: ones (MATLAB overrides to ones in get_TDR)."""
    return np.ones(len(faxis))


def _s21_to_impulse_DC(sdd21, faxis, sample_dt, OP, param):
    N = len(sdd21)
    S = np.zeros(2 * N - 2, dtype=complex)
    S[:N] = sdd21
    S[N:] = np.conj(sdd21[-2:0:-1])
    ir = np.real(np.fft.ifft(S))
    t = np.arange(len(ir)) * sample_dt
    return ir, t, 0.0, 0.0


def _get_StepR(ir, param, cb_step, ZT):
    """Stub: step response via cumsum; ZSR = (1 + cumsum(ir)) * ZT."""
    step = np.cumsum(ir)
    ZSR = (1.0 + step) * ZT
    return SimpleNamespace(ZSR=ZSR, step=step)


def _get_PulseR(ir, param, cb_step, ZT):
    """Stub: pulse response = ir (delta = derivative of step)."""
    M = int(param.samples_per_ui)
    pulse = lfilter(np.ones(M), 1, ir)
    return SimpleNamespace(pulse=pulse, pulse_orig=pulse.copy())


def _get_pdf_from_sampled_signal(samples, levels, bin_size, flag):
    rms = float(np.sqrt(np.mean(np.asarray(samples, dtype=float)**2))) + 1e-30
    n = max(32, int(8 * rms / (bin_size + 1e-30)))
    x = np.linspace(-4 * rms, 4 * rms, n)
    dx = x[1] - x[0] if n > 1 else 1.0
    y = np.exp(-0.5 * (x / rms)**2) / (rms * np.sqrt(2 * np.pi)) * dx
    y = y / (y.sum() + 1e-300)
    return SimpleNamespace(x=x, y=y)


# ---------------------------------------------------------------------------
# TDR_RL: 4-port reflection normalization formula
# ---------------------------------------------------------------------------

def _TDR_RL(Zin, Zout, s11, s12, s21, s22):
    """Re-normalize reflection coefficient from S-parameters (eq from MATLAB)."""
    num = (Zin**2 * s11 + Zin**2 * s22 + Zout**2 * s11 + Zout**2 * s22 +
           Zin**2 - Zout**2 +
           Zin * Zout * s11 * 2.0 - Zin * Zout * s22 * 2.0 +
           Zin**2 * s11 * s22 - Zin**2 * s12 * s21 -
           Zout**2 * s11 * s22 + Zout**2 * s12 * s21)
    den = (Zin * Zout * 2.0 + Zin**2 * s11 + Zin**2 * s22 -
           Zout**2 * s11 - Zout**2 * s22 + Zin**2 + Zout**2 +
           Zin**2 * s11 * s22 - Zin**2 * s12 * s21 +
           Zout**2 * s11 * s22 - Zout**2 * s12 * s21 -
           Zin * Zout * s11 * s22 * 2.0 + Zin * Zout * s12 * s21 * 2.0)
    return num / (den + 1e-300)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_TDR(S, OP, param, ZT, nport,
            _Bessel_Thomson_Filter_fn=None,
            _Butterworth_Filter_fn=None,
            _Tukey_Window_fn=None,
            _s21_to_impulse_DC_fn=None,
            _get_StepR_fn=None,
            _get_PulseR_fn=None,
            _get_pdf_fn=None):
    """Compute TDR (time-domain reflectometry) metrics.

    MATLAB lines 6904–7220.

    S: SimpleNamespace with:
        Frequencies: 1D float array (Hz)
        Parameters: array of shape (Nf, NumPorts, NumPorts) or (NumPorts, NumPorts, Nf)
                    Access convention: S.Parameters[freq_idx, row_0based, col_0based]
        Impedance: reference impedance (Ohms)
        NumPorts: number of ports
    OP: SimpleNamespace — flags: TDR, PTDR, DISPLAY_WINDOW, TDR_Butterworth, BinSize,
        TDR_duration, N (number of UI for maxtime), RL_norm_test, T_k, cb_Guassian, etc.
    param: SimpleNamespace — samples_per_ui, sample_dt, TR_TDR, tfx (1D, indexed by nport),
           FLAG.S2P, RL_sel (0-based), ui, ndfe, N_bx, beta_x, Grr, rho_x,
           levels, specBER, Tukey_Window, fb, fb_BT_cutoff
    ZT: target impedance (Ohms), typically 50 * sqrt(2) for differential
    nport: 0-based port index (MATLAB uses 1-based `np`)

    Dependency injection via optional *_fn params.
    Returns TDR_results SimpleNamespace.
    """
    bt_fn = _Bessel_Thomson_Filter_fn or _Bessel_Thomson_Filter
    bw_fn = _Butterworth_Filter_fn or _Butterworth_Filter
    tw_fn = _Tukey_Window_fn or _Tukey_Window
    s21_fn = _s21_to_impulse_DC_fn or _s21_to_impulse_DC
    step_fn = _get_StepR_fn or _get_StepR
    pulse_fn = _get_PulseR_fn or _get_PulseR
    pdf_fn = _get_pdf_fn or _get_pdf_from_sampled_signal

    TDR_results = SimpleNamespace()

    f = np.asarray(S.Frequencies, dtype=float).ravel()
    TDR_results.f = f
    f9 = f / 1e9

    # ---- RL computation ----
    rl_sel = int(param.RL_sel)  # 0-based
    Zref = float(S.Impedance)
    Nf = len(f)

    RL = np.zeros(Nf, dtype=complex)
    if int(param.FLAG.S2P) == 0:
        # 4-port S-params; S.Parameters shape (Nf, NumPorts, NumPorts)
        # RL_sel=0 → other_port=1; RL_sel=1 → other_port=0
        other_port = 1 - rl_sel
        Params = np.asarray(S.Parameters)
        is_4port = (S.NumPorts > 1)
        for i in range(Nf):
            s11 = complex(Params[i, rl_sel, rl_sel])
            if is_4port:
                s12 = complex(Params[i, rl_sel, other_port])
                s21 = complex(Params[i, other_port, rl_sel])
                s22 = complex(Params[i, other_port, other_port])
            else:
                s12 = 1.0
                s21 = 1.0
                s22 = complex(Params[i, rl_sel, rl_sel])
            RL[i] = _TDR_RL(Zref, 2 * ZT, s11, s12, s21, s22)
    else:
        # 2-port / s2p mode
        rho = (2 * ZT - Zref) / (2 * ZT + Zref)
        Params = np.asarray(S.Parameters)
        interim = np.sqrt(max(0.0, 1 - abs(rho)**2)) * (1 - rho) / (abs(1 - rho) + 1e-300)
        for i in range(Nf):
            s11_i = complex(Params[i, rl_sel, rl_sel])
            # fix B06-D9 (MATLAB rev 4p15p0 lines 7080-7081): MATLAB uses left-division
            # interim \ (s11-rho) = (s11-rho)/interim (interim then cancels), giving
            # RL = (s11-rho)/(1-rho*s11). Python previously used right-division of interim.
            RL[i] = (s11_i - rho) / interim / (1 - rho * s11_i + 1e-300) * interim

    RL = RL.ravel()

    # ---- Duration and delay ----
    TDR_duration = float(getattr(OP, 'TDR_duration', 5))
    if not hasattr(OP, 'DISPLAY_WINDOW'):
        OP.DISPLAY_WINDOW = 1
    TDR_results.delay = 500e-12  # seconds

    tr = float(param.TR_TDR)  # nanoseconds (transition time)
    tfx = float(np.atleast_1d(param.tfx)[nport])

    try:
        N_op = int(OP.N)
        maxtime = N_op * float(param.ui)
    except Exception:
        maxtime = 2e-9

    if int(getattr(OP, 'N', 1)) == 0 and S.NumPorts > 1:
        # Use get_RAW_FIR to estimate maxtime — stubbed: use default
        maxtime = 2e-9

    # ---- Transmitter filter with delay and Gaussian edge ----
    # MATLAB L7040: exp(-(1j)*2*pi*f9*TDR_results.delay/1e-9).  `*` and `/` are
    # equal precedence and left-associative, so the 1e-9 division happens LAST,
    # after the multiply by delay.  Pre-computing delay/1e-9 = 0.5 and
    # multiplying reorders the rounding: at f9=2.5 the reference gives
    # 1.1943401194869635e-15-1j and the pre-divided form 3.0616169978683831e-16-1j.
    # That is ~4e-15 relative on tx_filter and it propagates into the impulse.
    H_t = (np.exp(-2 * (np.pi * f9 * tr / 1.6832)**2) *
           np.exp(-1j * 2 * np.pi * f9 * TDR_results.delay / 1e-9) *
           np.exp(-1j * 2 * np.pi * f9 * tr * 3))

    Use_gaussian = bool(getattr(OP, 'cb_Guassian', True))
    if Use_gaussian:
        RLf = RL.ravel() * H_t.ravel()
    else:
        RLf = RL.ravel() * np.exp(-1j * 2 * np.pi * f9 * TDR_results.delay / 1e-9)

    # ---- Receiver filters (BT disabled, BW optional, Tukey optional) ----
    OP.TDR_Bessel_Thomson = 0
    H_bt = bt_fn(param, f, 0)  # disabled

    if hasattr(OP, 'TDR_Butterworth'):
        H_bw = bw_fn(param, f, OP.TDR_Butterworth)
    else:
        H_bw = np.ones(len(f))

    if float(getattr(param, 'Tukey_Window', 0)) != 0:
        H_tw = tw_fn(f, param)
    else:
        H_tw = np.ones(len(f))

    # Ensure row-like 1D
    H_bt = H_bt.ravel()
    H_bw = H_bw.ravel()
    H_tw = H_tw.ravel()
    RLf = RLf.ravel()

    TDR_results.Rx_filter = H_bt * H_bw * H_tw
    RLf = RLf * TDR_results.Rx_filter
    TDR_results.tx_filter = H_t

    # 4p16p0 L7460-7474: bail out on a channel with essentially no reflection.
    # "some test fixtures have almost zero CM and will cause TD conversion to
    # fail" -- 4p15p0 ran the conversion regardless. Returns a degenerate result
    # with ERL = inf (infinitely good return loss) and ERLRMS = -300 dB.
    if (str(getattr(param, 'matlab_version', '4p15p0')) >= '4p16p0'   # and 4p17p0
            and float(np.mean(np.abs(RL))) < 1e-6):
        dt = float(param.sample_dt)
        M = int(param.samples_per_ui)
        TDR_results.delay = 0
        TDR_results.tdr = np.ones(1000)
        # MATLAB 0:dt:999*dt is inclusive of both ends -> exactly 1000 samples.
        TDR_results.t = np.arange(1000) * dt
        TDR_results.x = 0
        TDR_results.y = 0
        TDR_results.avgZport = 0
        TDR_results.RL = np.zeros(1000)
        TDR_results.ptdr_RL = np.zeros(1000)
        # 0:dt*M:999*dt -- step dt*M, last value <= 999*dt, so floor(999/M)+1
        # samples (32 when M = 32), NOT 1000.
        TDR_results.WC_ptdr_samples_t = np.arange(0, 999 * dt + 1e-18, dt * M)
        TDR_results.WC_ptdr_samples = np.zeros(len(TDR_results.WC_ptdr_samples_t))
        TDR_results.ERL = np.inf
        TDR_results.ERLRMS = -300
        return TDR_results

    # ---- Impulse response ----
    IR, t, causality_dB, truncation_dB = s21_fn(RLf, f, param.sample_dt, OP, param)

    # ---- Time windowing ----
    t = t - TDR_results.delay
    # MATLAB L7098-7101:  tend = find(t>=maxtime+tfx,1);  IR = IR(1:tend)
    # The MATLAB 1-based index IS the Python exclusive bound, so the sample AT
    # the threshold is KEPT.  Slicing with the 0-based index instead dropped it,
    # making every windowed array (tdr, t, ptdr_RL, WC_ptdr_samples) one sample
    # short and shifting avgZport and ERLRMS with it.
    tend_arr = np.where(t >= maxtime + tfx)[0]
    tend = int(tend_arr[0]) + 1 if len(tend_arr) > 0 else len(t)
    IR = IR[:tend]
    t = t[:tend]

    tstart_arr = np.where(t >= tr * 1e-9)[0]
    tstart = int(tstart_arr[0]) if len(tstart_arr) > 0 else 0
    # MATLAB `tstart >= tend` compares two 1-based indices; tstart is 0-based here
    # and tend is the exclusive bound (== the MATLAB 1-based tend after the cut).
    if tstart + 1 >= tend:
        tend = len(t)
        tstart = 0

    cb_step = int(getattr(OP, 'cb_step', 0))
    ch_step = step_fn(IR[tstart:tend], param, cb_step, ZT)
    TDR_results.tdr = ch_step.ZSR
    TDR_results.t = t[tstart:tend]

    PTDR = pulse_fn(IR[tstart:tend], param, cb_step, ZT)
    PTDR.pulse_orig = PTDR.pulse.copy()

    # ---- Average impedance (OP.TDR or OP.PTDR path) ----
    if getattr(OP, 'TDR', False) or getattr(OP, 'PTDR', False):
        try:
            # MATLAB (L7184-7187):
            #   tfstart = find(t >= 3*tr*1e-9, 1);        % index into the FULL t
            #   x = TDR_results.t(tfstart:end);           % applied to the WINDOWED arrays
            # The index is derived from the full, delay-shifted time vector but used to
            # slice t(tstart:tend), so the weighted average effectively begins tstart
            # samples later than 3*tr. Searching the windowed vector instead (as Python
            # did) starts at a different point and biases avgZport -> Z11est/Z22est by a
            # constant ~1.4% independent of package case. Reproduce MATLAB exactly.
            #
            # No clamp: when tfstart runs past the windowed vector MATLAB's slice is
            # EMPTY, x(1) then errors and the catch sets avgZport = 0. Clamping to the
            # last sample instead returned that one sample's impedance.
            tfstart_arr = np.where(t >= 3 * tr * 1e-9)[0]
            # find(...) empty -> MATLAB `t([]:end)` is also empty, same path.
            tfstart = (int(tfstart_arr[0]) if len(tfstart_arr) > 0
                       else len(TDR_results.t))
            T_k = float(getattr(OP, 'T_k', 1e-9))
            x = TDR_results.t[tfstart:]
            y = TDR_results.tdr[tfstart:]
            # MATLAB L7185-7186 assigns x/y BEFORE the average, and with the tdr and t
            # vectors SWAPPED relative to the local x/y it just built.
            TDR_results.x = np.asarray(TDR_results.tdr).ravel()
            TDR_results.y = np.asarray(TDR_results.t).ravel()
            w = np.exp(-(x - x[0]) / (T_k + 1e-300))
            TDR_results.avgZport = float(np.mean(y * w) / (np.mean(w) + 1e-300))
        except Exception:
            TDR_results.avgZport = 0.0
        TDR_results.RL = RL

    # ---- PTDR: ERL computation ----
    if getattr(OP, 'PTDR', False):
        L = int(param.levels)
        BinSize = float(getattr(OP, 'BinSize', 1e-3))
        N_bx = int(getattr(param, 'N_bx', int(param.ndfe)))
        beta_x = float(getattr(param, 'beta_x', 0.0))
        Grr_mode = int(getattr(param, 'Grr', 1))
        rho_x = float(getattr(param, 'rho_x', 0.0))
        specBER = float(getattr(param, 'specBER', 1e-4))
        ui = float(param.ui)
        M = int(param.samples_per_ui)
        t_ptdr = TDR_results.t

        # Find ntx (start of gate, accounting for Gaussian delay)
        ntx_arr = np.where(t_ptdr >= tfx + 3 * tr * 1e-9)[0]
        ntx = int(ntx_arr[0]) if len(ntx_arr) > 0 else 0

        # Find ndfex (end of DFE gate)
        ndfex_arr = np.where(t_ptdr > (N_bx + 1) * ui + tfx + 3 * tr * 1e-9)[0]
        ndfex = int(ndfex_arr[0]) if len(ndfex_arr) > 0 else len(t_ptdr)
        tk = ui * (N_bx + 1) + tfx + 3 * tr * 1e-9

        # Build fctrx gain array.
        # MATLAB (L7230-7243):
        #   switch param.Grr
        #     case 0: fctrx(1:length(PTDR.pulse_orig)) = (1+rho_x)*rho_x;
        #     case 1: fctrx(1:length(PTDR.pulse_orig)) = 1;
        #     case 2: fctrx(1:length(PTDR.pulse_orig)) = 1;
        #   end
        #   fctrx(1:ntx) = 0;              % only the LEAD-IN is zeroed
        #   for ii = ntx:ndfex ...         % INCLUSIVE of ndfex
        #
        # fctrx is pre-filled across the WHOLE array, so beyond the DFE gate
        # (ii > ndfex) it retains that fill value — it is NOT zero. Allocating
        # np.zeros here discarded all reflection energy past the gate, which
        # understates the reflection and overstates ERL (23.1 vs 16.2 dB).
        fill = (1.0 + rho_x) * rho_x if Grr_mode == 0 else 1.0
        fctrx = np.full(len(PTDR.pulse), fill, dtype=float)
        fctrx[:ntx] = 0.0
        for ii in range(ntx, min(ndfex + 1, len(fctrx))):
            x_ii = (t_ptdr[ii] - tfx - 3 * tr * 1e-9) / (ui + 1e-300)
            if N_bx > 0 and beta_x != 0:
                Gloss_ii = 10.0 ** (beta_x * (t_ptdr[ii] - tk) / 20)
            else:
                Gloss_ii = 1.0
            if Grr_mode in (0, 1):
                Grr_ii = ((1 + rho_x) * rho_x *
                          np.exp(-((x_ii - N_bx - 1)**2) / ((1 + N_bx)**2 + 1e-300)))
            else:  # mode 2
                Grr_ii = rho_x
            fctrx[ii] = Gloss_ii * Grr_ii

        PTDR.pulse = PTDR.pulse * fctrx
        FAST_NOISE_CONV = 0

        RL_norm_test = bool(getattr(OP, 'RL_norm_test', False))
        RL_equiv = -np.inf
        best_ki = 0
        best_erl = -np.inf
        best_pdf = None

        for ki in range(M):
            tps = PTDR.pulse[ki::M]
            if len(tps) == 0:
                continue
            if RL_norm_test:
                rl_fom = float(np.linalg.norm(tps))
            else:
                testpdf = pdf_fn(tps, L, BinSize * 10, FAST_NOISE_CONV)
                cdf_test = np.cumsum(testpdf.y)
                idx_ber = int(np.searchsorted(cdf_test, specBER))
                if idx_ber < len(testpdf.x):
                    rl_test = -testpdf.x[idx_ber]
                else:
                    rl_test = 0.0
                rl_fom = rl_test

            if rl_fom > RL_equiv:
                RL_equiv = rl_fom
                best_ki = ki
            # UPSTREAM DEFECT B06-D10 (com_ieee8023_4p14p0/4p15p0/4p16p0, and the
            # adaptive-local-search build, all identical): the reference's
            #     if ~OP.RL_norm_test
            #         best_erl=rl_test; best_pdf=testpdf; best_cdf=cdf_test;
            #     end
            # sits OUTSIDE the `rl_fom > RL_equiv` guard, so ERL is reported for
            # the LAST phase while WC_ptdr_samples still come from best_ki.
            # Reproduced deliberately: COM Octave on a case whose worst phase is
            # not the last gives ERL = 35.391021572434525 dB (last phase,
            # rl_test = 0.017) where picking best_ki gives 32.39577516576788 dB
            # (rl_test = 0.024). Not reached with the default ERL_FOM = 1, which
            # takes the post-loop recompute below.
            if not RL_norm_test:
                best_erl = rl_test
                best_pdf = testpdf

        if RL_norm_test:
            tps = PTDR.pulse[best_ki::M]
            testpdf = pdf_fn(tps, L, BinSize * 10, FAST_NOISE_CONV)
            cdf_test = np.cumsum(testpdf.y)
            idx_ber = int(np.searchsorted(cdf_test, specBER))
            if idx_ber < len(testpdf.x):
                best_erl = -testpdf.x[idx_ber]
            else:
                best_erl = 0.0

        # MATLAB `rms = @(x) norm(x)/sqrt(length(x))`, evaluated BEFORE the ki loop.
        ERLRMS = float(np.linalg.norm(PTDR.pulse) / np.sqrt(len(PTDR.pulse)))
        TDR_results.ptdr_RL = PTDR.pulse
        TDR_results.WC_ptdr_samples_t = t_ptdr[best_ki::M]
        TDR_results.WC_ptdr_samples = PTDR.pulse[best_ki::M]
        # MATLAB `db = @(x) 20*log10(abs(x))` — NO epsilon floor. At best_erl = 0
        # (the degenerate single-bin pdf) the reference reports ERL = +Inf; the
        # +1e-300 floor reported a plausible-looking 6000 dB instead.
        with np.errstate(divide='ignore'):
            TDR_results.ERL = -20.0 * np.log10(abs(best_erl))
            TDR_results.ERLRMS = -20.0 * np.log10(abs(ERLRMS))

    return TDR_results


if __name__ == '__main__':
    import numpy as np
    from types import SimpleNamespace

    N = 64
    f = np.linspace(0.1e9, 26.5625e9, N)
    # Simple diagonal S-params (pure 50 ohm match)
    Params = np.zeros((N, 2, 2), dtype=complex)
    for i in range(N):
        Params[i, 0, 0] = 0.01  # small reflection
        Params[i, 1, 1] = 0.01
        Params[i, 0, 1] = 0.9
        Params[i, 1, 0] = 0.9

    S = SimpleNamespace(
        Frequencies=f, Parameters=Params, Impedance=50.0, NumPorts=2,
    )
    param = SimpleNamespace(
        FLAG=SimpleNamespace(S2P=0), RL_sel=0,
        TR_TDR=0.025, tfx=np.array([0.0, 0.0]), ui=1.0/53.125e9,
        ndfe=4, N_bx=4, beta_x=0.0, Grr=1, rho_x=0.1,
        levels=4, specBER=1e-4, Tukey_Window=0,
        samples_per_ui=4, sample_dt=1.0/(2*26.5625e9),
        fb=53.125e9, fb_BT_cutoff=0.473,
    )
    OP = SimpleNamespace(
        N=10, TDR=True, PTDR=False, DISPLAY_WINDOW=False,
        RL_norm_test=False, T_k=1e-9, BinSize=1e-3, cb_Guassian=True,
    )
    result = get_TDR(S, OP, param, 50.0 * np.sqrt(2), 0)
    print(f'TDR length: {len(result.tdr)}')
    print(f'avgZport: {result.avgZport:.2f}')
    print('Smoke test PASSED')
