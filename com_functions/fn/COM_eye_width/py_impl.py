# ============================================================
# MATLAB→Python translation notes for COM_eye_width
# MATLAB lines: 1362–1551
# ============================================================
# Indexing: MATLAB 1-based → 0-based Python
#   half_UI: MATLAB ceil(samp_UI/2) [1-based] → samp_UI//2 [0-based]
#   pdf_range: MATLAB 1:samp_UI → Python range(samp_UI)
#   start_sample/end_sample: MATLAB 1-based → subtract 1 for Python
#   h_j_full(:,j): MATLAB col j [1-based] → h_j_full[:, j] [0-based]
#   A_s_vec(j): MATLAB 1-based → A_s_vec[j] [0-based]
#   pdf_full{n}(j): MATLAB 1-based both → pdf_full[n][j] [0-based]
#   eye_contour{n}(half_UI, :): MATLAB 1-based row → eye_contour[n][half_UI, :]
#
# Levels vector: MATLAB Levels=2*(0:L-1)/(L-1)-1 → [-1,...,+1] (L values)
# A_s_vec scaling: A_s_vec *= (levels-1)   [MATLAB: A_s_vec*(param.levels-1)]
#
# eye_contour output: samp_UI × 2*(levels-1) matrix
#   col (n-1)*2+1 (1-based) = A_ni_top{n+1} → col (n)*2 (0-based) = A_ni_top[n+1]
#   col (n-1)*2+2 (1-based) = A_ni_bottom{n} → col (n)*2+1 (0-based) = A_ni_bottom[n]
#
# out_VT/out_VB: only computed when param.T_O != 0 (windowed histogram)
# Histogram window types: gaussian, triangle, rectangle, dual_rayleigh
# ============================================================

import numpy as np
from com_functions.fn.get_center_of_UI.py_impl import get_center_of_UI as _get_center_of_UI

def _histogram_window(T_O, QL, hw_type):
    """ML 1502-1524: the VEC histogram window, one vector of 2*T_O+1 weights.

    Lifted out of COM_eye_width so it can be driven on its own. The whole
    behaviour of OP.Histogram_Window_Weight lives here, and inline it was
    unreachable by any test: COM_eye_width needs the full get_pdf_full chain to
    run at all, so the three non-default window types sat in
    test_option_coverage.py's KNOWN_UNCOVERED with nothing exercising them.

    Verified against the reference by lifting ML 1502-1524 verbatim into an
    Octave function and running both on the same (T_O, QL). All four window
    types agree at T_O = 3, 5, 7, 8 and QL = 1.5, 2.5, 4.0: exact for
    gaussian, dual_rayleigh and rectangle, and within 3.4e-16 for triangle,
    whose MATLAB colon expression accumulates differently from arange.
    """
    T_O = int(T_O)
    if hw_type in ('gaussian', 'norm', 'normal', 'guassian'):
        QL_sigma = T_O / (QL + 1e-300)
        idx_arr = np.arange(-T_O, T_O + 1)
        return np.exp(-0.5 * (idx_arr / (QL_sigma + 1e-300)) ** 2)
    if hw_type == 'triangle':
        t_slope = 1.0 / T_O
        weights = np.concatenate([
            np.arange(0, 1 + t_slope, t_slope),
            np.arange(1 - t_slope, -t_slope, -t_slope)])
        return weights[:2 * T_O + 1]
    if hw_type == 'dual_rayleigh':
        QL_sigma = T_O / (QL + 1e-300)
        X = np.arange(-T_O, T_O + 1, dtype=float)
        weights = ((X + T_O) / QL_sigma ** 2 *
                   np.exp(-0.5 * ((X + T_O) / QL_sigma) ** 2) -
                   (X - T_O) / QL_sigma ** 2 *
                   np.exp(-0.5 * ((X - T_O) / QL_sigma) ** 2))
        return weights / (_mmax(weights) + 1e-300)
    if hw_type == 'rectangle':
        return np.ones(2 * T_O + 1)
    # ML 1523: otherwise -> error('%s not recognized for
    # Histogram_Window_Weight'). The port used to fall through to rectangle,
    # answering where the reference stops on a misspelled option.
    raise ValueError('%s not recognized for Histogram_Window_Weight' % hw_type)


def _mextreme_complex(a, take):
    """MATLAB orders complex values by magnitude, then by angle; numpy orders
    them lexicographically by real part, so max([3+4i, 5]) is 3+4i in MATLAB
    and 5 in numpy. take is -1 for max, 0 for min."""
    f = np.asarray(a).ravel()
    good = ~np.isnan(np.abs(f))
    if not good.any():
        return f[0]
    g = f[good]
    return g[np.lexsort((np.angle(g), np.abs(g)))[take]]


def _mmax(a):
    """MATLAB max(): a NaN is skipped unless every element is NaN, and complex
    values are ordered by magnitude then angle.

    np.max propagates a NaN, so one bad sample swallows the result where MATLAB
    ignores it. np.nanmax matches MATLAB but warns on an all-NaN input, where
    MATLAB quietly returns NaN. The isnan test also keeps the ordinary no-NaN
    case on np.max's faster path.
    """
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, -1)
    if a.dtype.kind != 'f':
        return np.max(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.max(a)
    return np.nanmax(a)


def _mmin(a):
    """MATLAB min(): the mirror of _mmax."""
    a = np.asarray(a)
    if a.dtype.kind == 'c':
        return _mextreme_complex(a, 0)
    if a.dtype.kind != 'f':
        return np.min(a)
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return np.min(a)
    return np.nanmin(a)

from scipy.signal import fftconvolve
import copy
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Callee stubs
# ---------------------------------------------------------------------------


# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.
_CONV_FFT_MIN = 128


def _conv1d(a, b):
    """Convolve two 1-D PDFs, choosing direct or FFT by operand size."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if min(a.size, b.size) >= _CONV_FFT_MIN:
        return fftconvolve(a, b)
    return np.convolve(a, b)


def _get_pdf_full(chdata_0, delta_y, t_s, param, OP, pdf_range):
    """Stub: returns list of trivial PDFs, zero ISI, unit signal."""
    samp_UI = int(param.samples_for_C2M)
    n_bins = max(32, int(2.0 / delta_y))
    x0 = np.linspace(-1.0, 1.0, n_bins)
    dx = x0[1] - x0[0]
    y0 = np.ones(n_bins) / n_bins
    pdf_list = [SimpleNamespace(x=x0.copy(), y=y0.copy(), BinSize=dx, Min=-n_bins//2)
                for _ in range(samp_UI)]
    # h_j_full: ISI matrix shape (1, samp_UI) — minimal
    h_j_full = np.zeros((1, samp_UI))
    A_s_vec = np.ones(samp_UI) * 0.5  # representative signal amplitude
    return pdf_list, h_j_full, A_s_vec


def _normal_dist(sigma, n_sigma, delta_y):
    """Stub: Gaussian PDF."""
    n = max(32, int(n_sigma * sigma / (delta_y + 1e-300) * 2))
    x = np.linspace(-n_sigma * sigma, n_sigma * sigma, n)
    dx = x[1] - x[0] if n > 1 else 1.0
    y = np.exp(-0.5 * (x / (sigma + 1e-300))**2) / ((sigma + 1e-300) * np.sqrt(2 * np.pi)) * dx
    y /= y.sum() + 1e-300
    return SimpleNamespace(x=x, y=y, BinSize=dx, Min=int(x[0] / (dx + 1e-300)))


def _conv_fct(pdf_a, pdf_b):
    """Stub: convolve two PDFs (same x axis assumption)."""
    y_c = _conv1d(pdf_a.y, pdf_b.y)
    dx = float(getattr(pdf_a, 'BinSize', 1e-3))
    x_c = np.arange(len(y_c)) * dx + float(pdf_a.x[0]) + float(pdf_b.x[0])
    total = y_c.sum()
    y_c = y_c / (total + 1e-300)
    return SimpleNamespace(x=x_c, y=y_c, BinSize=dx, Min=int(x_c[0] / (dx + 1e-300)))


def _conv_fct_MeanNotZero(pdf_a, pdf_b):
    return _conv_fct(pdf_a, pdf_b)


def _get_pdf_from_sampled_signal(samples, levels, delta_y):
    """Stub: Gaussian from samples."""
    rms = float(np.sqrt(np.mean(np.asarray(samples, dtype=float)**2))) + 1e-30
    n = max(32, int(4 * rms / (delta_y + 1e-300)))
    x = np.linspace(-4 * rms, 4 * rms, n)
    dx = x[1] - x[0] if n > 1 else delta_y
    y = np.exp(-0.5 * (x / rms)**2) / (rms * np.sqrt(2 * np.pi)) * dx
    y /= y.sum() + 1e-300
    return SimpleNamespace(x=x, y=y, BinSize=dx, Min=int(x[0] / (dx + 1e-300)))


def _pdf_to_cdf(pdf):
    cdf_y = np.cumsum(pdf.y)
    return SimpleNamespace(x=pdf.x.copy(), y=cdf_y, BinSize=pdf.BinSize, Min=pdf.Min)


def _cdf_to_ber_contour(cdf, specBER):
    """Find top (V where CDF(V) >= 1-specBER) and bottom (V where CDF(V) >= specBER) voltages."""
    idx_top = int(np.searchsorted(cdf.y, 1.0 - specBER))
    idx_bot = int(np.searchsorted(cdf.y, specBER))
    x = cdf.x
    top = float(x[min(idx_top, len(x) - 1)])
    bot = float(x[min(idx_bot, len(x) - 1)])
    return top, bot


def _find_eye_width(eye_contour_n, half_UI, samp_UI, vref):
    """Stub: return half of samp_UI as symmetric eye width."""
    half = samp_UI // 4
    return float(half), float(half)


def _combine_pdf_same_voltage_axis(pdf_a, pdf_b):
    """Stub: add y values (assumes same x axis)."""
    if len(pdf_a.y) == len(pdf_b.y):
        y_combined = pdf_a.y + pdf_b.y
    else:
        y_combined = pdf_a.y.copy()
    return SimpleNamespace(x=pdf_a.x.copy(), y=y_combined, BinSize=pdf_a.BinSize, Min=pdf_a.Min)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def COM_eye_width(chdata, delta_y, fom_result, param, OP, Struct_Noise, pdf_range_flag,
                  _get_center_of_UI_fn=None,
                  _get_pdf_full_fn=None,
                  _normal_dist_fn=None,
                  _conv_fct_fn=None,
                  _conv_fct_MNZ_fn=None,
                  _get_pdf_ss_fn=None,
                  _pdf_to_cdf_fn=None,
                  _cdf_to_ber_fn=None,
                  _find_eye_width_fn=None,
                  _combine_pdf_fn=None):
    """Compute eye width and contours for the COM figure of merit.

    MATLAB lines 1362–1551.

    chdata: list of SimpleNamespace (chdata[0] is the primary channel)
    delta_y: voltage bin size for PDFs
    fom_result: SimpleNamespace with t_s field (cursor sample index, 0-based)
    param: SimpleNamespace with samples_for_C2M, T_O, levels, specBER, sigma_RJ,
           sigma_X, A_DD, QL
    OP: SimpleNamespace with Histogram_Window_Weight, ber_q
    Struct_Noise: SimpleNamespace with sigma_N, sigma_TX, ne_noise_pdf, cci_pdf, ber_q
    pdf_range_flag: bool — if True, limit PDF computation to T_O window around center

    Returns (Left_EW, Right_EW, eye_contour, out_VT, out_VB)
        Left_EW, Right_EW: arrays of length (levels-1), time in samples
        eye_contour: (samp_UI, 2*(levels-1)) array
        out_VT, out_VB: scalars or [] (only set when T_O != 0)
    """
    gcu_fn = _get_center_of_UI_fn or _get_center_of_UI
    gpf_fn = _get_pdf_full_fn or _get_pdf_full
    nd_fn = _normal_dist_fn or _normal_dist
    cf_fn = _conv_fct_fn or _conv_fct
    cfmz_fn = _conv_fct_MNZ_fn or _conv_fct_MeanNotZero
    gpss_fn = _get_pdf_ss_fn or _get_pdf_from_sampled_signal
    ptc_fn = _pdf_to_cdf_fn or _pdf_to_cdf
    ber_fn = _cdf_to_ber_fn or _cdf_to_ber_contour
    few_fn = _find_eye_width_fn or _find_eye_width
    cpdf_fn = _combine_pdf_fn or _combine_pdf_same_voltage_axis

    samp_UI = int(param.samples_for_C2M)
    levels = int(param.levels)
    specBER = float(param.specBER)
    sigma_RJ = float(param.sigma_RJ)
    sigma_X = float(param.sigma_X)
    A_DD = float(param.A_DD)
    ber_q = float(getattr(Struct_Noise, 'ber_q', getattr(OP, 'ber_q', 7.0)))

    # 0-based center index
    half_UI = gcu_fn(samp_UI)  # 0-based

    T_O = int(np.floor((param.T_O / 1000.0) * samp_UI))
    start_sample = half_UI - T_O
    end_sample = half_UI + T_O

    # pdf_range: get_pdf_full expects 1-based MATLAB indices (matching MATLAB COM_eye_width)
    if pdf_range_flag:
        pdf_range_pass = [start_sample + 1, end_sample + 1]
    else:
        pdf_range_pass = []

    pdf_full_1, h_j_full, A_s_vec = gpf_fn(
        chdata[0], delta_y, fom_result.t_s, param, OP, pdf_range_pass)

    # Resolve pdf_range to full list of 0-based indices
    if not pdf_range_flag or not pdf_range_pass:
        pdf_range = list(range(samp_UI))
    else:
        pdf_range = list(range(min(start_sample, end_sample),
                               max(start_sample, end_sample) + 1))

    # Normalized signal amplitude levels: -1, ..., +1
    Levels = 2.0 * np.arange(levels) / (levels - 1) - 1.0  # shape (levels,)
    A_s_vec = np.asarray(A_s_vec) * (levels - 1)  # scale

    # ---- Build per-level shifted PDFs ----
    # pdf_full[n][j]: copy of pdf_full_1[j] with x shifted by A_s_vec[j] * Levels[n]
    pdf_full = []
    for n in range(levels):
        pdf_n = [copy.deepcopy(pdf_full_1[j]) for j in range(samp_UI)]
        for j in pdf_range:
            shift = float(A_s_vec[j]) * float(Levels[n])
            pdf_n[j].x = pdf_n[j].x + shift
            dx = float(pdf_n[j].BinSize)
            pdf_n[j].Min = int(pdf_n[j].x[0] / (dx + 1e-300))
        pdf_full.append(pdf_n)

    # ---- Build noise + ISI combined PDFs ----
    sigma_G_full = np.zeros(samp_UI)
    combined_pdf = [[None] * samp_UI for _ in range(levels)]
    combined_cdf = [[None] * samp_UI for _ in range(levels)]

    sigma_N = float(Struct_Noise.sigma_N)
    sigma_TX = float(Struct_Noise.sigma_TX)

    for n in range(levels):
        for j in pdf_range:
            h_col = h_j_full[:, j].ravel()
            sigma_G_full[j] = float(np.linalg.norm([
                sigma_RJ * sigma_X * float(np.linalg.norm(h_col)),
                sigma_N,
                sigma_TX,
            ]))
            g_pdf = nd_fn(sigma_G_full[j], ber_q, delta_y)
            g_pdf = cf_fn(g_pdf, Struct_Noise.ne_noise_pdf)
            dd_pdf = gpss_fn(A_DD * h_col, levels, delta_y)
            noise_pdf = cf_fn(g_pdf, dd_pdf)
            isi_xtalk_pdf = cfmz_fn(pdf_full[n][j], Struct_Noise.cci_pdf)
            comb_pdf = cfmz_fn(isi_xtalk_pdf, noise_pdf)
            combined_pdf[n][j] = comb_pdf
            combined_cdf[n][j] = ptc_fn(comb_pdf)

    # ---- BER contour: top and bottom voltages ----
    A_ni_top = [np.zeros(samp_UI) for _ in range(levels)]
    A_ni_bot = [np.zeros(samp_UI) for _ in range(levels)]
    for n in range(levels):
        for j in pdf_range:
            top, bot = ber_fn(combined_cdf[n][j], specBER)
            A_ni_top[n][j] = top
            A_ni_bot[n][j] = bot

    # ---- Eye contour: (levels-1) eyes ----
    # MATLAB: eye_contour{n}(:,1)=A_ni_top{n+1};  eye_contour{n}(:,2)=A_ni_bottom{n};
    n_eyes = levels - 1
    eye_contour_list = []
    for n in range(n_eyes):
        ec_n = np.column_stack([A_ni_top[n + 1], A_ni_bot[n]])  # shape (samp_UI, 2)
        eye_contour_list.append(ec_n)

    # ---- Eye width ----
    Left_EW = np.zeros(n_eyes)
    Right_EW = np.zeros(n_eyes)
    for n in range(n_eyes):
        EH_top = float(eye_contour_list[n][half_UI, 0])
        EH_bot = float(eye_contour_list[n][half_UI, 1])
        vref = EH_top / 2.0 + EH_bot / 2.0
        lew, rew = few_fn(eye_contour_list[n], half_UI, samp_UI, vref)
        Left_EW[n] = lew
        Right_EW[n] = rew

    # ---- Convert eye_contour cell to matrix ----
    eye_contour = np.zeros((samp_UI, 2 * n_eyes))
    for n in range(n_eyes):
        eye_contour[:, n * 2: n * 2 + 2] = eye_contour_list[n]

    # ---- Timing-bathtub data for plotting (final full-eye call only) ----
    # BER vs sample phase at each eye's center threshold (the horizontal scan).
    # Stored on chdata[0] as a side-channel so the return signature (and MATLAB
    # fidelity) is unchanged; consumed by com_plots when OP.SAVE_FIGURES.
    if not pdf_range_flag:
        # pdf_to_cdf returns THREE curves and picking the wrong one silently
        # produces a flat 0.5:
        #     yB = cumsum(pdf)        = P(V <= v)   (bottom-eye tail)
        #     yT = reverse cumsum     = P(V >= v)   (top-eye tail)
        #     y  = min(yB, yT)        -- necessarily ~0.5 mid-distribution
        # A threshold crossing is an error only in one direction per level: the
        # UPPER level errs by falling below the threshold (yB), the LOWER level
        # by rising above it (yT). An earlier version of this block used `y` and
        # `1 - y`, which reads the minimum of two half-probabilities and pinned
        # every bathtub at BER ~ 0.5 regardless of how open the eye was.
        def _tail(cdf_obj, which):
            """P(V<=v) for 'B' or P(V>=v) for 'T', with a safe fallback."""
            arr = getattr(cdf_obj, 'yB' if which == 'B' else 'yT', None)
            if arr is None:                      # minimal stub without yB/yT
                y = np.asarray(cdf_obj.y, dtype=float)
                arr = np.cumsum(y) if which == 'B' else np.flip(np.cumsum(np.flip(y)))
            return np.asarray(arr, dtype=float).ravel()

        def _eye_ber(up, lo, j, vth):
            """Symbol-error probability at threshold vth for the eye up/lo."""
            cuj, clj = combined_cdf[up][j], combined_cdf[lo][j]
            if cuj is None or clj is None:
                return np.nan
            ber_up = float(np.interp(vth, np.asarray(cuj.x, dtype=float),
                                     _tail(cuj, 'B')))
            ber_lo = float(np.interp(vth, np.asarray(clj.x, dtype=float),
                                     _tail(clj, 'T')))
            return 0.5 * (ber_up + ber_lo)

        phase_UI = (np.arange(samp_UI) - half_UI) / float(samp_UI)

        # --- timing bathtub: sweep sample phase at each eye's own threshold ---
        ber_eyes = np.full((n_eyes, samp_UI), np.nan)
        vth_eyes = np.zeros(n_eyes)
        for n in range(n_eyes):
            up, lo = n + 1, n
            vth = 0.5 * (A_ni_bot[lo][half_UI] + A_ni_top[up][half_UI])
            vth_eyes[n] = vth
            for j in range(samp_UI):
                ber_eyes[n, j] = _eye_ber(up, lo, j, vth)

        # --- voltage bathtub: sweep the threshold at the centre phase ---------
        # One curve per eye, each spanning its own eye and therefore centred on
        # that eye's level rather than on a single +/-A_s pair.
        # The axis must run well BEYOND the outer thresholds, not just span the
        # eye edges: each outer eye's bathtub keeps rising past its own level,
        # and an axis that stops at the outermost threshold clips those two
        # curves mid-slope. Extend by a full eye spacing on each side so every
        # curve reaches its ~0.5 shoulders.
        v_lo = min(float(A_ni_top[n + 1][half_UI]) for n in range(n_eyes))
        v_hi = max(float(A_ni_bot[n][half_UI]) for n in range(n_eyes))
        vth_min, vth_max = float(_mmin(vth_eyes)), float(_mmax(vth_eyes))
        if n_eyes > 1:
            pad = (vth_max - vth_min) / (n_eyes - 1)      # one eye spacing
        else:
            pad = max(v_hi - v_lo, 1e-6)
        lo = min(v_lo, vth_min - pad)
        hi = max(v_hi, vth_max + pad)
        v_axis = np.linspace(lo, hi, 601)
        ber_v = np.full((n_eyes, v_axis.size), np.nan)
        for n in range(n_eyes):
            up, lo = n + 1, n
            for k, vth in enumerate(v_axis):
                ber_v[n, k] = _eye_ber(up, lo, half_UI, float(vth))

        chdata[0].timing_bathtub = {"phase_UI": phase_UI, "ber_eyes": ber_eyes,
                                    "eye_contour": eye_contour,
                                    "eye_threshold_V": vth_eyes,
                                    "vbt_threshold_V": v_axis,
                                    "vbt_ber": ber_v}

    # ---- Windowed VEC (out_VT, out_VB) when T_O != 0 ----
    out_VT = []
    out_VB = []
    if int(param.T_O) != 0:
        T_O_nonzero = T_O if T_O > 0 else 1
        hw_type = str(getattr(OP, 'Histogram_Window_Weight', 'rectangle')).lower()
        weights = _histogram_window(T_O_nonzero, float(param.QL), hw_type)

        # Build weighted combined PDF for each level
        out_pdf_levels = [None] * levels
        for n in range(levels):
            out_pdf_n = None
            for j_idx, j in enumerate(range(start_sample, end_sample + 1)):
                if j < 0 or j >= samp_UI:
                    continue
                if combined_pdf[n][j] is None:
                    continue
                target = copy.deepcopy(combined_pdf[n][j])
                w = float(weights[j_idx])
                target.y = target.y * w
                if out_pdf_n is None:
                    out_pdf_n = target
                else:
                    out_pdf_n = cpdf_fn(out_pdf_n, target)
            if out_pdf_n is not None:
                s = out_pdf_n.y.sum()
                out_pdf_n.y = out_pdf_n.y / (s + 1e-300)
            out_pdf_levels[n] = out_pdf_n

        out_cdf_levels = [ptc_fn(p) if p is not None else None for p in out_pdf_levels]

        A_ni_top_O = np.zeros(levels)
        A_ni_bot_O = np.zeros(levels)
        for n in range(levels):
            if out_cdf_levels[n] is not None:
                top, bot = ber_fn(out_cdf_levels[n], specBER)
            else:
                top, bot = 0.0, 0.0
            A_ni_top_O[n] = top
            A_ni_bot_O[n] = bot

        OUT_VT_L = np.zeros((n_eyes, 2))
        for n in range(n_eyes):
            OUT_VT_L[n, 0] = A_ni_top_O[n + 1]
            OUT_VT_L[n, 1] = A_ni_bot_O[n]

        EH_VT = OUT_VT_L[:, 0] - OUT_VT_L[:, 1]
        min_idx = int(np.argmin(EH_VT))
        out_VT = float(OUT_VT_L[min_idx, 0])
        out_VB = float(OUT_VT_L[min_idx, 1])

    return Left_EW, Right_EW, eye_contour, out_VT, out_VB


if __name__ == '__main__':
    import numpy as np
    from types import SimpleNamespace

    samp_UI = 32
    levels = 4
    delta_y = 0.01
    specBER = 1e-4

    param = SimpleNamespace(
        samples_for_C2M=samp_UI, T_O=0, levels=levels, specBER=specBER,
        sigma_RJ=0.01, sigma_X=1.0, A_DD=0.05, QL=2.5,
    )
    ne_pdf = SimpleNamespace(x=np.array([0.0]), y=np.array([1.0]), BinSize=delta_y, Min=0)
    cci_pdf = SimpleNamespace(x=np.array([0.0]), y=np.array([1.0]), BinSize=delta_y, Min=0)
    Struct_Noise = SimpleNamespace(
        sigma_N=0.01, sigma_TX=0.01, ber_q=7.0,
        ne_noise_pdf=ne_pdf, cci_pdf=cci_pdf,
    )
    OP = SimpleNamespace(Histogram_Window_Weight='rectangle', ber_q=7.0)
    fom_result = SimpleNamespace(t_s=samp_UI // 2)
    ch = SimpleNamespace()

    Left_EW, Right_EW, eye_contour, out_VT, out_VB = COM_eye_width(
        [ch], delta_y, fom_result, param, OP, Struct_Noise, False)
    print(f'Left_EW = {Left_EW}')
    print(f'Right_EW = {Right_EW}')
    print(f'eye_contour shape = {eye_contour.shape}')
    print('Smoke test PASSED')
