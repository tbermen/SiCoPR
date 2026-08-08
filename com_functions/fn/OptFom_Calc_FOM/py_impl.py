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


def OptFom_Calc_FOM(chdata, do_C2M, THIS, param, OP, sbr, _COM_eye_width_fn=None):
    skip_loop = 0
    A_s = float(THIS.A_s)
    total_noise_rms = float(THIS.total_noise_rms)

    if not do_C2M:
        FOM = 20 * np.log10(A_s / total_noise_rms)
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

    tmp_result = SimpleNamespace(t_s=int(THIS.cursor_i), A_s=A_s)

    EH_1st = 2.0 * (A_s - float(erfcinv(float(param.specBER) * 2)) * 2.0 / np.sqrt(2) * total_noise_rms)
    if EH_1st <= float(getattr(param, 'Min_VEO_Test', 0)) / 1000.0 - 0.001:
        return None, 1

    chdata[0].eq_pulse_response = np.asarray(sbr, dtype=float)

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

    if N_i <= 0:
        return None, 1

    FOM = 20.0 * np.log10(A_s / N_i)
    return FOM, skip_loop
