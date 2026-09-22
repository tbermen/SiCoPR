# ============================================================
# MATLAB→Python translation notes for OptFom_Calc_FOM
# MATLAB lines: 2817–2873
# ============================================================
# do_C2M=False: FOM = 20*log10(A_s / total_noise_rms) — equation 93A-36.
# do_C2M=True: normal_dist/COM_eye_width path per MATLAB lines 2826-2873.
# skip_loop: set to 1 if eye is too small (only in do_C2M path).
# ============================================================

import numpy as np
from types import SimpleNamespace
from scipy.special import erfcinv
from com_functions.fn.normal_dist.py_impl import normal_dist as _normal_dist
from com_functions.fn.COM_eye_width.py_impl import COM_eye_width as _COM_eye_width


def _m20log10(num, den):
    """MATLAB 20*log10(num/den) for real scalars.

    Two ends of the reference's behaviour were lost: Python's `/` RAISES on a
    zero divisor where MATLAB carries Inf, and numpy's log10 of a negative
    float is NaN where MATLAB goes complex.
    COM Octave: 20*log10(0.5/0)     -> Inf
                20*log10(0/0)       -> NaN
                20*log10(-0.5/0.05) -> 20 + 27.287527076836827i
                20*log10(-0.25/0.05)-> 13.979400086720377 + 27.287527076836827i
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.float64(num) / np.float64(den)
        if ratio < 0:                       # False for NaN, so NaN stays real
            return complex(20.0 * np.log10(complex(ratio)))
        return float(20.0 * np.log10(ratio))


def OptFom_Calc_FOM(chdata, do_C2M, THIS, param, OP, sbr, _COM_eye_width_fn=None):
    skip_loop = 0
    A_s = float(THIS.A_s)
    total_noise_rms = float(THIS.total_noise_rms)

    if not do_C2M:
        # Equation 93A-36
        FOM = _m20log10(A_s, total_noise_rms)
        return FOM, skip_loop

    # ── C2M path (MATLAB lines 2826–2873) ──────────────────────────────────
    if float(getattr(param, 'Noise_Crest_Factor', 0)) == 0:
        ber_q = float(np.sqrt(2) * erfcinv(2 * float(param.specBER)))
    else:
        ber_q = float(param.Noise_Crest_Factor)

    if getattr(OP, 'force_pdf_bin_size', False):
        delta_y = float(OP.BinSize)
    else:
        delta_y = float(min(A_s / 1000.0, float(OP.BinSize)))

    ne_noise_pdf = _normal_dist(0, ber_q, delta_y)
    cci_pdf      = _normal_dist(0, ber_q, delta_y)

    # MATLAB L3120 writes chdata(1).eq_pulse_response=sbr HERE, before the
    # EH_1st skip test. MATLAB passes chdata by value and this function returns
    # only [FOM, skip_loop] (L3097), so that write is local and the caller never
    # sees it. Python shares the list and its namespaces, so element 0 is
    # replaced with a copy first -- the process_sxp remedy, see
    # tests/test_reference_leaks.py for what this defect class cost.
    #
    # The copy also contains COM_eye_width's timing_bathtub side-channel, which
    # it writes to chdata[0] below. Without it, a losing EQ candidate's bathtub
    # is left on the caller's chdata for com_plots and com_mat_export to read
    # whenever the driver's own COM_eye_width call is gated off (OP.EW ~= 1 or
    # OP.MLSE ~= 0) -- the wrong curve, silently.
    chdata = list(chdata)
    chdata[0] = SimpleNamespace(**vars(chdata[0]))
    chdata[0].eq_pulse_response = np.asarray(sbr, dtype=float)

    tmp_result = SimpleNamespace(t_s=int(THIS.cursor_i), A_s=A_s)

    EH_1st = 2.0 * (A_s - float(erfcinv(float(param.specBER) * 2)) * 2.0 / np.sqrt(2) * total_noise_rms)
    if EH_1st <= float(getattr(param, 'Min_VEO_Test', 0)) / 1000.0 - 0.001:
        return None, 1

    Struct_Noise = SimpleNamespace(
        sigma_N      = float(THIS.sigma_N),
        sigma_TX     = float(THIS.sigma_TX),
        cci_pdf      = cci_pdf,
        ber_q        = ber_q,
        ne_noise_pdf = ne_noise_pdf,
    )

    ew_fn = _COM_eye_width_fn or _COM_eye_width
    Left_EW, Right_EW, eye_contour, EH_T_C2M, EH_B_C2M = \
        ew_fn(chdata, delta_y, tmp_result, param, OP, Struct_Noise, 1)

    EH = float(EH_T_C2M) - float(EH_B_C2M)
    N_i = (A_s * 2.0 - EH) / 2.0

    if EH <= float(getattr(param, 'Min_VEO_Test', 0)) / 1000.0:
        return None, 1

    # No N_i<=0 guard: the reference has none, and skipping where it returns a
    # value (Inf at N_i==0, complex below it) is a divergence, not a safeguard.
    FOM = _m20log10(A_s, N_i)
    return FOM, skip_loop
