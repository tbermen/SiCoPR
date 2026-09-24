# ============================================================
# MATLAB→Python translation notes for get_PSDs
# MATLAB lines: 6380–6654
# ============================================================
# cursor_i: caller passes 0-based Python index. MATLAB code uses 1-based.
#   Internal conversion: cursor_i_m = cursor_i + 1 where MATLAB 1-based needed.
#   mod(cursor_i_m - 1, M) + 1 → cursor_i % M  (0-based phase)
#   mod(cursor_i_m, M) → (cursor_i + 1) % M
# filter(ones(1,M), 1, x) → lfilter(ones(M), 1, x)
# reshape(v, num_ui, M) MATLAB col-major → v.reshape(num_ui, M, order='F')
# sum(A.') where A is num_ui×M → np.sum(A.T, axis=0) giving shape (num_ui,)
# Single→double-sided: [re(S[0]), S[1:-1], re(S[-1]), conj(S[-2:0:-1])]
# fvec: single-sided, length = num_ui*M//2 + 1
# isfield → hasattr
# Output: result SimpleNamespace with S_rn, S_xn, S_tn, S_jn, S_qn, S_n, etc.
# ============================================================

import collections as _collections
import hashlib as _hashlib

import numpy as np
from com_functions.fn.CDF_inv_ev.py_impl import CDF_inv_ev as _CDF_inv_ev
from com_functions.fn.get_pdf_from_sampled_signal.py_impl import get_pdf_from_sampled_signal as _pdf_uncached
from com_functions.fn.normal_dist.py_impl import normal_dist as _normal_dist
from com_functions.fn.Init_PDF_Fast.py_impl import Init_PDF_Fast as _Init_PDF_Fast
from com_functions.fn.conv_fct.py_impl import conv_fct as _conv_fct
from com_functions.fn.d_cpdf.py_impl import d_cpdf as _d_cpdf

from scipy.signal import lfilter, fftconvolve
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Callee stubs (minimal)
# ---------------------------------------------------------------------------


# PDF convolutions are extremely skewed in size: ~79% of the arithmetic sits in
# ~1% of the calls (both operands long), while most calls have a kernel of a few
# bins. Direct convolution wins for tiny kernels and loses badly for long ones
# (measured 2.7x slower at 600, 19x at 9000, >1000x at 20000+), so dispatch on
# size. The FFT path agrees with the direct path to ~1e-15 relative.


# get_PSDs calls get_pdf_from_sampled_signal once per tick per EQ setting, and
# consecutive ticks feed it the SAME sampled vector, so most builds are exact
# repeats. Each build costs ~120 convolutions over a 4096-point vector, so
# memoising them is worthwhile. Keyed on the input bytes, so a hit is
# bit-identical by construction; small LRU because repeats are temporally local
# (within one equalizer setting's tick sweep).
#
# This wrapper is the reason get_PSDs keeps a local name for the function
# rather than calling the canonical directly. It is NOT a duplicate
# translation: it returns exactly what the canonical returns, which is why the
# differential in tests/test_inlined_copies.py reports it as identical. It was
# collapsed onto a bare import on 2026-09-22 and had to be restored -- a copy
# being behaviourally equivalent does not make it redundant.
_PDF_CACHE = _collections.OrderedDict()
_PDF_CACHE_MAX = 64


def _detach(pdf):
    """Hand out a PDF that shares nothing mutable with the cached entry.

    Copying the namespace alone is not enough: the arrays inside would still be
    shared, so a caller doing `pdf.y *= k` (rather than `pdf.y = pdf.y * k`)
    would corrupt the cache and silently poison every later hit. Copying the
    arrays costs far less than recomputing the PDF, so the speed-up stands.
    """
    out = SimpleNamespace(**vars(pdf))
    for _k, _v in vars(out).items():
        if isinstance(_v, np.ndarray):
            setattr(out, _k, _v.copy())
    return out


def _get_pdf_from_sampled_signal(input_vector, L, BinSize, FAST_NOISE_CONV=0):
    _arr = np.ascontiguousarray(np.asarray(input_vector, dtype=float))
    _key = (_hashlib.blake2b(_arr.tobytes(), digest_size=16).digest(),
            int(L), float(BinSize), int(FAST_NOISE_CONV))
    _hit = _PDF_CACHE.get(_key)
    if _hit is not None:
        _PDF_CACHE.move_to_end(_key)
        return _detach(_hit)
    _res = _pdf_uncached(input_vector, L, BinSize, FAST_NOISE_CONV)
    _PDF_CACHE[_key] = _res
    if len(_PDF_CACHE) > _PDF_CACHE_MAX:
        _PDF_CACHE.popitem(last=False)
    return _detach(_res)


def _S_RN(fvec, G_DC, G_DC2, param):
    """Stub for S_RN — returns flat receiver noise PSD (V^2/GHz)."""
    eta_0 = param.eta_0  # V^2/GHz
    return eta_0 * np.ones(len(fvec))


def _S_IN(fvec, H_noise, G_DC, G_DC2, param, OP):
    """Stub for S_IN — returns zeros."""
    return np.zeros(len(fvec))


def _H_interp(sdd21p, faxis, fvec, fb):
    """Stub for H_interp — interpolates linearly."""
    return np.interp(fvec, faxis, np.abs(sdd21p))


# ---------------------------------------------------------------------------
# Helper: build conjugate-symmetric double-sided vector
# ---------------------------------------------------------------------------

def _check_h_index(idx, n):
    """Refuse an h() index the reference would refuse, with its own message.

    `idx` is 0-based; the message quotes the 1-based index MATLAB reports.
    """
    idx = np.asarray(idx)
    # Which offending index the reference names: the FIRST non-positive one,
    # but the LARGEST over-bound one.
    #   COM Octave: h=1:10; h([3 -5 -1]) -> "h(-5): ...";  h([-1 -5 3]) ->
    #   "h(-1): ...";  h([3 12 15]) and h([3 15 12]) both -> "h(15): out of
    #   bound 10".
    bad = idx[idx < 0]
    if bad.size:
        raise IndexError(
            'get_PSDs: h(%d): subscripts must be either integers 1 to '
            '(2^63)-1 or logicals - the jitter sampling window starts '
            'before the pulse response' % (int(bad[0]) + 1))
    if idx.size and int(idx.max()) >= n:
        raise IndexError(
            'get_PSDs: h(%d): out of bound %d - the jitter sampling window '
            'runs past the pulse response' % (int(idx.max()) + 1, n))


def _to_double_sided(S_ss):
    """Convert single-sided spectrum to double-sided (conjugate symmetric).

    Input: S_ss of length N = num_ui*M/2 + 1
    Output: length 2*(N-1) = num_ui*M
    """
    return np.concatenate([
        [np.real(S_ss[0])],
        S_ss[1:-1],
        [np.real(S_ss[-1])],
        np.conj(S_ss[-2:0:-1])
    ])


def _fold_psd(full_psd, num_ui, M):
    """Fold full double-sided PSD (num_ui*M elements) to decimated num_ui-element PSD.

    Equivalent to MATLAB: sum(reshape(psd, num_ui, M).')
    Then take single-sided and mirror back.
    """
    # MATLAB column-major reshape: each column of length num_ui
    mat = full_psd.reshape(num_ui, M, order='F')  # (num_ui, M)
    # Transpose and sum columns → length num_ui
    S_folded = np.sum(mat.T, axis=0)
    return S_folded


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Inlined helpers for the B01-D2 'slow' ADC-clip path (MATLAB 4p15p0 6716-6725).
# Faithful copies of the audited canonical get_pdf_from_sampled_signal / conv_fct /
# CDF_inv_ev (and their sub-helpers). No cross-py_impl imports per build protocol.
# ---------------------------------------------------------------------------


# The ADC-clip signal PDF depends only on the sampled pulse response, which is a
# pure function of the equalizer setting and the sampling PHASE (cursor_i % M).
# The itick sweep visits 49 ticks but only M=32 distinct phases, so ~35% of these
# builds are exact repeats. Each build costs ~120 convolutions over a 4096-point
# vector, so memoising them is worthwhile. Keyed on the input bytes, so a hit is
# bit-identical by construction; small LRU because repeats are temporally local
# (within one equalizer setting's tick sweep).


def get_PSDs(result, h, cursor_i, txffe, G_DC, G_DC2, param, chdata, OP,
             _S_RN_fn=None, _S_IN_fn=None, _H_interp_fn=None):
    """Compute power spectral densities for all noise sources.

    MATLAB lines 6380–6654.

    cursor_i: 0-based Python index of the cursor in the equalized pulse response h.
    h: 1D array — equalized pulse response (all channels, CTLE+TxFFE applied).
    result: SimpleNamespace to accumulate PSD fields.
    txffe: TX FFE tap array.
    G_DC, G_DC2: CTLE DC gain parameters.
    param, chdata, OP: standard COM structs.
    Dependency injection via optional *_fn params.
    Returns result (modified in place).
    """
    S_RN_fn = _S_RN_fn if _S_RN_fn is not None else _S_RN
    S_IN_fn = _S_IN_fn if _S_IN_fn is not None else _S_IN
    H_interp_fn = _H_interp_fn if _H_interp_fn is not None else _H_interp

    # MATLAB passes [] for an uninitialised result; assigning a field to [] there
    # auto-creates a struct. Replicate that: None -> new namespace.
    if result is None:
        result = SimpleNamespace()

    # ---- Setup ----
    num_ui = int(param.num_ui_RXFF_noise)
    M = int(param.samples_per_ui)
    L = param.levels
    fb = param.fb
    SNR_TX = param.SNR_TX
    dw = param.RxFFE_cmx
    bmax = np.atleast_1d(np.asarray(param.bmax, dtype=float))
    bmin = np.atleast_1d(np.asarray(param.bmin, dtype=float))
    Nb = int(param.ndfe)
    sigma_X2 = (L**2 - 1) / (3.0 * (L - 1)**2)
    eta_0 = param.eta_0  # V^2/GHz
    T_b = 1.0 / fb
    delta_f = fb / num_ui  # Hz
    # Single-sided frequency vector: num_ui*M/2 + 1 points
    n_fvec = num_ui * M // 2 + 1
    fvec = np.arange(n_fvec) * delta_f
    result.fvec = fvec

    # ---- H_rxffe (eq 178A-28) ----
    if OP.COMPUTE_COM:
        H_rxffe = np.zeros(len(fvec), dtype=complex)
        w_vec = np.atleast_1d(np.asarray(result.w, dtype=complex))
        for nn, w_n in enumerate(w_vec):
            H_rxffe += w_n * np.exp(-1j * 2 * np.pi * fvec * T_b * (nn - dw - 1))
        # MATLAB L6408-6410: take the symbol-rate band (first num_ui/2+1 points)
        # of |H_rxffe|^2 and double-side to num_ui. Do NOT fold/alias — folding
        # sums M copies and inflates the noise enhancement ~Mx (closes the eye).
        H_rxffe_2_of_f = np.abs(H_rxffe) ** 2  # length num_ui*M//2+1
        H_rxffe_2_ss = H_rxffe_2_of_f[:num_ui // 2 + 1]
        H_rxffe_2 = _to_double_sided(H_rxffe_2_ss)  # length num_ui
    else:
        H_rxffe_2 = 1

    result.H_rxffe_2 = H_rxffe_2

    if OP.WO_TXFFE:
        # ---- S_rn (eq 178A-15): receiver thermal noise ----
        if not OP.COMPUTE_COM:
            S_RN_of_f = S_RN_fn(fvec, G_DC, G_DC2, param)
            rxn_psd = _to_double_sided(S_RN_of_f) / 1e9  # V^2/GHz → V^2/Hz
            rxn_rms = np.sqrt(np.sum(rxn_psd) * delta_f)
            S_rn_full = _fold_psd(rxn_psd, num_ui, M)
            S_rn_ss = S_rn_full[:num_ui // 2 + 1]
            result.S_rn = _to_double_sided(S_rn_ss)
            result.S_rn_rms = np.sqrt(np.sum(result.S_rn) * delta_f)

            # ---- S_in (eq 178A-24): input noise ----
            if OP.PSDRXCAL:
                # MATLAB L7215 keeps the interpolated noise-path VTF on the
                # result struct; the port computed it into a local and dropped
                # it, so result.H_noise never existed.
                result.H_noise = H_interp_fn(chdata[-1].sdd21p, chdata[0].faxis,
                                             fvec, fb)
                S_IN_of_f = S_IN_fn(fvec, result.H_noise, G_DC, G_DC2, param, OP)
                inn_psd = _to_double_sided(S_IN_of_f)
                inn_rms = np.sqrt(np.sum(inn_psd) * delta_f)
                S_in_full = _fold_psd(inn_psd, num_ui, M)
                S_in_ss = S_in_full[:num_ui // 2 + 1]
                result.S_in = _to_double_sided(S_in_ss)
                result.S_in_rms = np.sqrt(np.sum(result.S_in) * delta_f)
            else:
                result.S_in = 0
                result.S_in_rms = 0
        else:
            # Apply H_rxffe to cached PSDs
            result.S_rn = result.S_rn * H_rxffe_2
            result.S_rn_rms = np.sqrt(np.sum(result.S_rn) * delta_f)
            result.S_in = result.S_in * H_rxffe_2
            result.S_in_rms = np.sqrt(np.sum(result.S_in) * delta_f)

    else:
        # ---- S_xn (eq 178A-16): crosstalk PSD ----
        if not OP.COMPUTE_COM:
            result.S_xn = 0.0
            n_channels = len(chdata)
            if OP.PSDRXCAL:
                num_channel_files = n_channels - 1
            else:
                num_channel_files = n_channels

            if num_channel_files != 1:
                hk_list = [None] * n_channels  # 0-indexed placeholder
                # ML: [~, iphase(xchan)] = max(hxn) for xchan=2:num_channel_files
                # on a variable never initialised, so MATLAB grows it to
                # num_channel_files and leaves iphase(1), the THRU, at 0. In
                # the port's 0-based form that is -1. get_pdf never reads the
                # THRU's entry (it samples the thru at the cursor phase).
                # COM Octave, checkpoint 05 of wXtalk_T4_R24: iphase [0 8 8 6].
                iphase = np.full(num_channel_files, -1, dtype=int)
                for xchan in range(1, num_channel_files):  # 0-based: channels 1..num_channel_files-1
                    ch = chdata[xchan]
                    # r4p15p0: dropped the unused pulse_ctle length calc; pad/truncate
                    # the crosstalk pulse response (hk_k) directly to num_ui*M.
                    hk_k = np.asarray(ch.pulse_response_w_CFT_TXFFE_noRxFFE).ravel()
                    # Pad or truncate to num_ui*M
                    if num_ui * M > len(hk_k):
                        hk_k = np.concatenate([hk_k, np.zeros(num_ui * M - len(hk_k))])
                    else:
                        hk_k = hk_k[:num_ui * M]

                    hxn = np.array([np.linalg.norm(hk_k[i::M]) for i in range(M)])
                    iphase[xchan] = int(np.argmax(hxn))  # 0-based phase

                    hrn = hk_k[iphase[xchan]::M]
                    S_xn_chan = sigma_X2 * (np.abs(np.fft.fft(hrn))) ** 2 / fb
                    # ML: hk(xchan).k, .hrn, .S_xn, then result.hk=hk after the
                    # loop, so the result carries all three, not hrn alone.
                    hk_list[xchan] = SimpleNamespace(k=hk_k, hrn=hrn, S_xn=S_xn_chan)
                    result.__dict__.setdefault('hk', [None] * num_channel_files)
                    result.hk[xchan] = hk_list[xchan]
                    result.S_xn = result.S_xn + S_xn_chan

                result.iphase = iphase
                result.S_xn_rms = np.sqrt(np.sum(result.S_xn) * delta_f)
            else:
                result.S_xn = 0.0
                result.hk = []
                result.iphase = 0      # ML: result.iphase=1, 1-based
                result.S_xn_rms = 0.0
        else:
            result.S_xn = result.S_xn * H_rxffe_2
            result.S_xn_rms = np.sqrt(np.sum(result.S_xn) * delta_f)

        # ---- S_tn (eq 178A-17): transmitter noise PSD ----
        if not OP.COMPUTE_COM:
            if not OP.TDMODE:
                htn = lfilter(np.ones(M), 1, chdata[0].ctle_imp_response)
            else:
                if hasattr(chdata[0], 'ctle_pulse_response'):
                    htn = chdata[0].ctle_pulse_response
                else:
                    htn = lfilter(np.ones(M), 1, chdata[0].ctle_imp_response)

            # Align to sample point (0-based cursor_i)
            phase_0 = cursor_i % M  # 0-based phase start
            htn = htn[phase_0:].ravel()
            htn = htn[::M]  # resample every M

            len_htn = len(htn)
            if num_ui > len_htn:
                hext = np.concatenate([htn, np.zeros(num_ui - len_htn)])
            else:
                hext = htn[:num_ui]

            S_tn_w_AM = int(getattr(param, 'S_tn_w_AM', 0))
            result.S_tn = (sigma_X2 ** S_tn_w_AM) * 10 ** (-SNR_TX / 10) * \
                           (np.abs(np.fft.fft(hext))) ** 2 / fb
            result.S_tn_rms = np.sqrt(np.sum(result.S_tn) * delta_f)
        else:
            result.S_tn = result.S_tn * H_rxffe_2
            result.S_tn_rms = np.sqrt(np.sum(result.S_tn) * delta_f)

        # ---- S_jn (eq 93A-28): jitter noise PSD ----
        if not OP.COMPUTE_COM:
            # Sampling offset logic (MATLAB 1-based → 0-based Python)
            cursor_i_m = cursor_i + 1  # 1-based equivalent
            sampling_offset = cursor_i_m % M  # MATLAB: mod(cursor_i, M)
            if sampling_offset <= 1:
                sampling_offset += M
            # MATLAB: cursors_early_sample = h(sampling_offset-1:M:end) (1-based)
            # → 0-based: h[sampling_offset-2::M]
            # MATLAB: cursors_late_sample = h(sampling_offset+1:M:end) (1-based)
            # → 0-based: h[sampling_offset::M]
            if OP.LIMIT_JITTER_CONTRIB_TO_DFE_SPAN:
                # h(cursor_i-1+M*(-1:ndfe)) (1-based) → h[cursor_i + M*(-1:ndfe)] (0-based)
                # fix B01-D1 (MATLAB rev 4p15p0 lines 6676-6677): early/late samples are
                # h(cursor_i-1+M*(-1:ndfe)) / h(cursor_i+1+M*(-1:ndfe)), centered at the cursor.
                idx_early = cursor_i - 1 + M * np.arange(-1, Nb + 1)
                idx_late = cursor_i + 1 + M * np.arange(-1, Nb + 1)
                # No masking: MATLAB indexes h() with the whole vector and
                # refuses an index off either end rather than quietly taking
                # the jitter slope from fewer UI.  Dropping the out-of-range
                # entries left a shorter h_J that still FFTs to a plausible
                # S_jn, which is the worst kind of wrong.
                # COM Octave: cursor_i=4 (1-based), M=4 -> "error: h(-1):
                #   subscripts must be either integers 1 to (2^63)-1 or
                #   logicals"; ndfe=14 with len(h)=64 -> "error: h(69): out of
                #   bound 64 (dimensions are 64x1)".
                _check_h_index(idx_early, len(h))
                _check_h_index(idx_late, len(h))
                cursors_early_sample = h[idx_early]
                cursors_late_sample = h[idx_late]
            else:
                cursors_early_sample = h[sampling_offset - 2::M]
                cursors_late_sample = h[sampling_offset::M]

            # Ensure equal length
            n_eq = min(len(cursors_early_sample), len(cursors_late_sample))
            cursors_early_sample = cursors_early_sample[:n_eq]
            cursors_late_sample = cursors_late_sample[:n_eq]

            h_J = (cursors_late_sample - cursors_early_sample) / 2.0 * M
            h_J = h_J.ravel()
            if num_ui > len(h_J):
                h_J = np.concatenate([h_J, np.zeros(num_ui - len(h_J))])
            else:
                h_J = h_J[:num_ui]

            A_DD = float(param.A_DD)
            sigma_RJ = float(param.sigma_RJ)
            result.S_jn = sigma_X2 * (A_DD**2 + sigma_RJ**2) * \
                           (np.abs(np.fft.fft(h_J))) ** 2 / fb
            result.S_jn_rms = np.sqrt(np.sum(result.S_jn) * delta_f)
            result.S_rj_jn = sigma_X2 * (sigma_RJ**2) * \
                              (np.abs(np.fft.fft(h_J))) ** 2 / fb
            result.S_rj_rms = np.sqrt(np.sum(result.S_rj_jn) * delta_f)
        else:
            result.S_jn = result.S_jn * H_rxffe_2
            result.S_jn_rms = np.sqrt(np.sum(result.S_jn) * delta_f)
            result.S_rj_jn = result.S_rj_jn * H_rxffe_2
            result.S_rj_rms = np.sqrt(np.sum(result.S_rj_jn) * delta_f)

        # ---- S_qn: quantization noise ----
        N_qb = int(param.N_qb)
        if N_qb != 0:
            pulse_for_quantization = chdata[0].pulse_response_w_CFT_TXFFE_noRxFFE
            sample_idx_0 = cursor_i % M  # 0-based
            sampled_pr = pulse_for_quantization[sample_idx_0:]
            sampled_pr = sampled_pr.ravel()[::M]
            if num_ui > len(sampled_pr):
                sampled_pr = np.concatenate([sampled_pr, np.zeros(num_ui - len(sampled_pr))])
            else:
                sampled_pr = sampled_pr[:num_ui]

            if str(getattr(param, 'clip_method', 'Fast')).lower() == 'slow':
                # fix B01-D2 (MATLAB rev 4p15p0 lines 6716-6725): adc_clip is the P_qc
                # quantile of the signal+Gaussian-noise PDF, and ctle_signal_sigma is set.
                sig_pdf = _get_pdf_from_sampled_signal(sampled_pr, int(param.levels), OP.BinSize)
                sigma_noise = np.sqrt(
                    getattr(result, 'S_in_rms', 0)**2 +
                    getattr(result, 'S_rn_rms', 0)**2 +
                    getattr(result, 'S_xn_rms', 0)**2 +
                    getattr(result, 'S_tn_rms', 0)**2 +
                    getattr(result, 'S_rj_rms', 0)**2
                )
                noise_pdf = SimpleNamespace(**vars(sig_pdf))   # copy x-axis/Min/BinSize, replace y
                noise_pdf.y = (1.0 / (np.sqrt(2 * np.pi) * sigma_noise)
                               * np.exp(-sig_pdf.x**2 / (2 * sigma_noise**2)) * OP.BinSize)
                sig_noise_pdf = _conv_fct(sig_pdf, noise_pdf)
                sig_noise_cdf = np.cumsum(sig_noise_pdf.y)
                ctle_signal_sigma = float(np.sqrt(np.sum(sig_noise_pdf.x**2 * sig_noise_pdf.y)))
                adc_clip = float(-_CDF_inv_ev(param.P_qc, sig_noise_pdf, sig_noise_cdf))
                result.ctle_signal_sigma = ctle_signal_sigma
            else:
                adc_clip = float(np.sum(np.abs(sampled_pr)))

            adc_lsb = 2.0 * adc_clip / (2**N_qb - 1)
            sigma_Q = adc_lsb / np.sqrt(12.0)
            result.adc_clip = adc_clip
            result.S_qn = sigma_Q**2 / fb * np.ones(num_ui)
            result.S_qn_rms = np.sqrt(np.sum(result.S_qn) * delta_f)
        else:
            result.S_qn = 0.0
            result.S_qn_rms = 0.0

        # ---- Total S_n ----
        result.S_n = (result.S_rn + result.S_tn + result.S_xn +
                      result.S_jn + result.S_qn + result.S_in)
        result.S_n_rms = np.sqrt(np.sum(result.S_n) * delta_f)

        # ---- S_isi, S_G, Sn_rho (COMPUTE_COM path only) ----
        if OP.COMPUTE_COM:
            # 0-based: samp_idx = phase..len(h) step M
            samp_phase = cursor_i % M
            samp_idx = np.arange(samp_phase, len(h), M)
            cursor_n_arr = np.where(samp_idx == cursor_i)[0]
            if len(cursor_n_arr) == 0:
                # cursor_i not in the sample grid → find nearest
                cursor_n = int(np.searchsorted(samp_idx, cursor_i))
            else:
                cursor_n = int(cursor_n_arr[0])

            hisi = h[samp_idx].copy()
            if num_ui > len(hisi):
                hisi = np.concatenate([hisi, np.zeros(num_ui - len(hisi))])
            else:
                hisi = hisi[:num_ui]
            hisi = hisi.reshape(1, -1).ravel()

            # Apply DFE clipping to ISI taps (eq 178A-29)
            if cursor_n < len(hisi):
                cursor_val = hisi[cursor_n]
                hisi[cursor_n] = 0.0
                for ii in range(cursor_n + 1, cursor_n + Nb + 1):
                    if ii < len(hisi):
                        ib = ii - cursor_n - 1
                        if ib < len(bmax) and hisi[ii] >= bmax[ib] * cursor_val:
                            hisi[ii] -= bmax[ib] * cursor_val
                        elif ib < len(bmin) and hisi[ii] <= bmin[ib] * cursor_val:
                            hisi[ii] -= bmin[ib] * cursor_val
                        else:
                            hisi[ii] = 0.0

            result.S_isi = sigma_X2 * (np.abs(np.fft.fft(hisi))) ** 2 / fb
            result.S_isi_rms = np.sqrt(np.sum(result.S_isi) * delta_f)

            result.S_G = result.S_tn + result.S_rj_jn + result.S_rn + result.S_in
            result.S_G_rms = np.sqrt(np.sum(result.S_G) * delta_f)
            result.Sn_rho = result.S_isi + result.S_n
            result.Sn_rho_rms = np.sqrt(np.sum(result.Sn_rho) * delta_f)

    return result


if __name__ == '__main__':
    from types import SimpleNamespace
    import numpy as np

    # Smoke test: single THRU channel, no crosstalk
    M = 4
    num_ui = 16
    fb = 53.125e9
    N = num_ui * M

    param = SimpleNamespace(
        num_ui_RXFF_noise=num_ui, samples_per_ui=M, levels=4, fb=fb,
        SNR_TX=30.0, RxFFE_cmx=0, bmax=np.array([0.7, 0.2, 0.2, 0.2]),
        bmin=np.array([-0.7, -0.2, -0.2, -0.2]), ndfe=4, eta_0=1.7e-4,
        A_DD=0.05, sigma_RJ=0.01, N_qb=0, clip_method='Fast',
        S_tn_w_AM=0,
    )
    ch = SimpleNamespace(
        ctle_imp_response=np.zeros(N * 2),
        pulse_response_w_CFT_TXFFE_noRxFFE=np.zeros(N * 2),
        faxis=np.linspace(0, fb / 2, 100),
    )
    ch.ctle_imp_response[0] = 1.0
    ch.pulse_response_w_CFT_TXFFE_noRxFFE[M * 3] = 1.0

    OP = SimpleNamespace(
        COMPUTE_COM=False, WO_TXFFE=False, TDMODE=False, PSDRXCAL=False,
        LIMIT_JITTER_CONTRIB_TO_DFE_SPAN=False,
    )
    result = SimpleNamespace(S_rn=0, S_in=0, S_xn=0, S_tn=0, S_jn=0)
    h = np.zeros(N * 2)
    h[M * 3] = 1.0

    get_PSDs(result, h, M * 3, None, -1.0, None, param, [ch], OP)
    print(f'S_tn_rms = {result.S_tn_rms:.4e}')
    print(f'S_jn_rms = {result.S_jn_rms:.4e}')
    print(f'S_n_rms  = {result.S_n_rms:.4e}')
    print('Smoke test PASSED')
