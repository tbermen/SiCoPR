# ============================================================
# MATLAB GROUND TRUTH
# get_PSDs computes power spectral densities for IEEE 802.3 COM noise sources.
# MATLAB lines 6380–6654
#
# Key invariants verified analytically:
# 1. S_tn: from CTLE-filtered pulse response, resampled at cursor phase
#    10^(-SNR_TX/10) * sigma_X2 * |FFT(hext)|^2 / fb
#    For unit impulse ctle_imp_response, hext = [1, 0, ...] → |FFT|^2 = 1
#    S_tn_rms = sqrt(N * 10^(-SNR_TX/10) * sigma_X2 / fb * delta_f)
#
# 2. S_jn: from finite-difference derivative of pulse at cursor
#    For unit impulse at position M*3, derivative at that phase = 0 (single spike)
#
# 3. S_xn: crosstalk from channels 1..N-1
#    Sum of sigma_X2 * |FFT(hrn)|^2 / fb for each xtalk channel
#    For single THRU channel, S_xn=0 (no xtalk)
#
# 4. S_n = S_rn + S_tn + S_xn + S_jn + S_qn + S_in
#
# 5. S_qn = 0 when N_qb=0
#
# 6. H_rxffe_2 = |H_rxffe|^2 is stored in result.H_rxffe_2 for COMPUTE_COM
# ============================================================

import numpy as np
import pytest
from types import SimpleNamespace
from scipy.signal import lfilter

from com_functions.fn.get_PSDs.py_impl import get_PSDs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_param(M=4, num_ui=16, fb=53.125e9, SNR_TX=30.0, Nb=4):
    sigma_X2 = (4**2 - 1) / (3.0 * (4 - 1)**2)  # L=4
    return SimpleNamespace(
        num_ui_RXFF_noise=num_ui, samples_per_ui=M, levels=4, fb=fb,
        SNR_TX=SNR_TX, RxFFE_cmx=0,
        bmax=np.array([0.7, 0.2, 0.2, 0.2]),
        bmin=np.array([-0.7, -0.2, -0.2, -0.2]),
        ndfe=Nb, eta_0=1.7e-4, A_DD=0.05, sigma_RJ=0.01,
        N_qb=0, clip_method='Fast', S_tn_w_AM=0,
    )


def _base_op(COMPUTE_COM=False, WO_TXFFE=False, TDMODE=False,
              PSDRXCAL=False, LIMIT_JITTER=False):
    return SimpleNamespace(
        COMPUTE_COM=COMPUTE_COM, WO_TXFFE=WO_TXFFE, TDMODE=TDMODE,
        PSDRXCAL=PSDRXCAL, LIMIT_JITTER_CONTRIB_TO_DFE_SPAN=LIMIT_JITTER,
    )


def _single_channel(M=4, N_total=128, cursor=12):
    """Single THRU channel: unit impulse ctle, pulse positioned at cursor."""
    ch = SimpleNamespace(
        ctle_imp_response=np.zeros(N_total),
        pulse_response_w_CFT_TXFFE_noRxFFE=np.zeros(N_total),
        faxis=np.linspace(0, 26.5625e9, 100),
    )
    ch.ctle_imp_response[0] = 1.0
    ch.pulse_response_w_CFT_TXFFE_noRxFFE[cursor] = 1.0
    return ch


def _base_result():
    return SimpleNamespace(S_rn=0, S_in=0, S_xn=0, S_tn=0, S_jn=0,
                           S_rj_jn=0, S_qn=0)


# ---------------------------------------------------------------------------
# Test 1 – Nominal: S_tn non-zero for non-trivial cursor, correct formula
# ---------------------------------------------------------------------------

def test_S_tn_formula():
    """S_tn = (sigma_X2**S_tn_w_AM) * 10^(-SNR_TX/10) * |FFT(hext)|^2 / fb.

    With S_tn_w_AM=0, sigma_X2 factor is not applied (sigma_X2**0 = 1).
    With unit ctle impulse (ctle_imp_response[0]=1), the running-sum pulse
    is lfilter(ones(M), 1, e0) = [1,1,1,1,0,...].  At cursor_i=0 (phase 0),
    resampling gives hext=[1,0,0,...,0] (length num_ui).
    FFT([1,0,...]) = [1,1,...,1] so |FFT|^2 = ones(num_ui).
    S_tn[k] = 10^(-3) / fb for all k (num_ui terms).
    S_tn_rms^2 = sum(S_tn) * delta_f = num_ui * 10^(-3)/fb * fb/num_ui = 10^(-3).
    """
    M = 4
    num_ui = 8
    fb = 53.125e9
    param = _base_param(M=M, num_ui=num_ui, fb=fb, SNR_TX=30.0)
    op = _base_op(WO_TXFFE=False)
    ch = _single_channel(M=M, N_total=num_ui * M * 2, cursor=0)
    h = np.zeros(num_ui * M * 2)
    h[0] = 1.0
    result = _base_result()

    def stub_S_RN(fvec, g_dc, g_dc2, p):
        return p.eta_0 * np.ones(len(fvec))

    get_PSDs(result, h, 0, None, -1.0, None, param, [ch], op,
             _S_RN_fn=stub_S_RN)

    # S_tn_w_AM=0 → sigma_X2**0 = 1, so expected = 10^(-SNR_TX/10) = 0.001
    expected_rms2 = 10**(-30.0 / 10)
    assert abs(result.S_tn_rms**2 - expected_rms2) / expected_rms2 < 0.01, \
        f"S_tn_rms^2={result.S_tn_rms**2:.4e}, expected≈{expected_rms2:.4e}"


# ---------------------------------------------------------------------------
# Test 2 – Nominal: S_xn = 0 with a single channel (no crosstalk)
# ---------------------------------------------------------------------------

def test_S_xn_zero_single_channel():
    """With only one channel and no PSDRXCAL, S_xn should be zero."""
    M = 4
    num_ui = 8
    param = _base_param(M=M, num_ui=num_ui)
    op = _base_op(WO_TXFFE=False, PSDRXCAL=False)
    ch = _single_channel(M=M, N_total=num_ui * M * 2, cursor=M * 2)
    h = np.zeros(num_ui * M * 2)
    h[M * 2] = 1.0
    result = _base_result()

    get_PSDs(result, h, M * 2, None, -1.0, None, param, [ch], op)
    # S_xn = 0 scalar or array of zeros
    xn = result.S_xn
    if hasattr(xn, '__len__'):
        assert np.allclose(xn, 0, atol=1e-30)
    else:
        assert xn == 0.0


# ---------------------------------------------------------------------------
# Test 3 – Nominal: S_n = sum of components
# ---------------------------------------------------------------------------

def test_S_n_is_sum_of_components():
    """S_n = S_rn + S_tn + S_xn + S_jn + S_qn + S_in (all sources)."""
    M = 4
    num_ui = 8
    param = _base_param(M=M, num_ui=num_ui)
    op = _base_op(WO_TXFFE=False)
    ch = _single_channel(M=M, N_total=num_ui * M * 2, cursor=0)
    h = np.zeros(num_ui * M * 2)
    h[0] = 1.0
    result = _base_result()

    get_PSDs(result, h, 0, None, -1.0, None, param, [ch], op)

    expected = result.S_tn + result.S_xn + result.S_jn + result.S_qn + result.S_rn + result.S_in
    np.testing.assert_allclose(result.S_n, expected, atol=1e-30)


# ---------------------------------------------------------------------------
# Test 4 – Nominal: S_qn = 0 when N_qb = 0
# ---------------------------------------------------------------------------

def test_S_qn_zero_when_N_qb_zero():
    """N_qb=0 → S_qn = 0 scalar, S_qn_rms = 0."""
    M = 4
    num_ui = 8
    param = _base_param(M=M, num_ui=num_ui)
    param.N_qb = 0
    op = _base_op(WO_TXFFE=False)
    ch = _single_channel(M=M, N_total=num_ui * M * 2, cursor=0)
    h = np.zeros(num_ui * M * 2)
    h[0] = 1.0
    result = _base_result()

    get_PSDs(result, h, 0, None, -1.0, None, param, [ch], op)

    assert result.S_qn_rms == 0.0


# ---------------------------------------------------------------------------
# Test 5 – Nominal: WO_TXFFE path returns S_rn (receiver noise)
# ---------------------------------------------------------------------------

def test_wo_txffe_sets_S_rn():
    """WO_TXFFE=True (and not COMPUTE_COM) → S_rn computed from S_RN stub."""
    M = 4
    num_ui = 8
    param = _base_param(M=M, num_ui=num_ui)
    op = _base_op(WO_TXFFE=True, COMPUTE_COM=False, PSDRXCAL=False)
    ch = _single_channel(M=M, N_total=num_ui * M * 2, cursor=0)
    h = np.zeros(num_ui * M * 2)
    h[0] = 1.0
    result = _base_result()

    eta_0 = param.eta_0

    def flat_S_RN(fvec, g, g2, p):
        return eta_0 * np.ones(len(fvec))

    get_PSDs(result, h, 0, None, -1.0, None, param, [ch], op,
             _S_RN_fn=flat_S_RN)

    # S_rn_rms > 0 (eta_0 = 1.7e-4 V^2/GHz)
    assert result.S_rn_rms > 0, f"S_rn_rms={result.S_rn_rms}"


# ---------------------------------------------------------------------------
# Test 6 – Boundary: fvec length is num_ui*M//2 + 1
# ---------------------------------------------------------------------------

def test_fvec_length():
    """result.fvec has length num_ui*M//2 + 1 (single-sided)."""
    M = 4
    num_ui = 8
    param = _base_param(M=M, num_ui=num_ui)
    op = _base_op(WO_TXFFE=False)
    ch = _single_channel(M=M, N_total=num_ui * M * 2, cursor=0)
    h = np.zeros(num_ui * M * 2)
    h[0] = 1.0
    result = _base_result()

    get_PSDs(result, h, 0, None, -1.0, None, param, [ch], op)

    expected_len = num_ui * M // 2 + 1
    assert len(result.fvec) == expected_len, \
        f"fvec len={len(result.fvec)}, expected {expected_len}"


# ---------------------------------------------------------------------------
# Test 7 – Boundary: COMPUTE_COM path computes H_rxffe_2 from result.w
# ---------------------------------------------------------------------------

def test_compute_com_sets_H_rxffe_2():
    """COMPUTE_COM=True → result.H_rxffe_2 is set and non-negative array."""
    M = 4
    num_ui = 8
    param = _base_param(M=M, num_ui=num_ui)
    op = _base_op(WO_TXFFE=True, COMPUTE_COM=True, PSDRXCAL=False)
    ch = _single_channel(M=M, N_total=num_ui * M * 2, cursor=0)
    h = np.zeros(num_ui * M * 2)
    h[0] = 1.0
    result = _base_result()
    result.S_rn = np.ones(num_ui) * 1e-10
    result.S_in = np.zeros(num_ui)
    # Simple RxFFE: cursor tap w=[1.0] (unity, no equalization)
    result.w = np.array([1.0])

    get_PSDs(result, h, 0, None, -1.0, None, param, [ch], op)

    assert hasattr(result, 'H_rxffe_2'), "H_rxffe_2 not set"
    H2 = np.atleast_1d(result.H_rxffe_2)
    assert np.all(np.real(H2) >= -1e-15), f"H_rxffe_2 has negative values"


# ---------------------------------------------------------------------------
# B01-D1 regression: LIMIT_JITTER early/late sampling is centered AT the cursor.
# MATLAB 4p15p0 lines 6676-6677:
#   cursors_early_sample = h(cursor_i-1 + M*(-1:ndfe))
#   cursors_late_sample  = h(cursor_i+1 + M*(-1:ndfe))
# => 0-based: idx_early = cursor_i-1 + M*k, idx_late = cursor_i+1 + M*k.
# A narrow symmetric triangle peaking exactly at the cursor has zero centered
# slope at every sampled UI, so h_J = 0 and S_jn = 0. Pre-fix Python sampled one
# UI late (cursor_i+M*k / cursor_i+2+M*k), giving a nonzero slope -> S_jn != 0.
# ---------------------------------------------------------------------------
def test_B01_D1_jitter_slope_centered_at_cursor():
    M, num_ui, cursor, N = 4, 8, 12, 64
    param = _base_param(M=M, num_ui=num_ui, Nb=4)
    op = _base_op(COMPUTE_COM=False, LIMIT_JITTER=True)
    h = np.zeros(N)
    h[cursor - 1], h[cursor], h[cursor + 1] = 1.0, 2.0, 1.0   # symmetric triangle, peak at cursor
    ch = _single_channel(M=M, N_total=N, cursor=cursor)
    result = _base_result()

    def stub_S_RN(fvec, g_dc, g_dc2, p):
        return p.eta_0 * np.ones(len(fvec))

    get_PSDs(result, h, cursor, None, -1.0, None, param, [ch], op, _S_RN_fn=stub_S_RN)

    assert result.S_jn_rms < 1e-12, (
        "S_jn_rms=%g; a symmetric triangle peaking at the cursor has zero centered "
        "jitter slope (MATLAB 6676-6677 sample cursor_i-/+1), so S_jn must be 0. "
        "Pre-fix one-UI-late sampling makes it nonzero." % result.S_jn_rms)


# ---------------------------------------------------------------------------
# B01-D2 regression: ADC 'slow' clip = P_qc quantile of the signal+noise PDF,
# and result.ctle_signal_sigma is set (MATLAB 4p15p0 lines 6716-6725).
# Pre-fix Python used a max(|.|)+3*sigma heuristic and never set ctle_signal_sigma.
# Oracle: the audited canonical get_pdf_from_sampled_signal/conv_fct/CDF_inv_ev.
# ---------------------------------------------------------------------------
def test_B01_D2_slow_clip_is_pdf_quantile_and_sets_sigma():
    from com_functions.fn.get_pdf_from_sampled_signal.py_impl import get_pdf_from_sampled_signal as _ref_gpss
    from com_functions.fn.conv_fct.py_impl import conv_fct as _ref_conv
    from com_functions.fn.CDF_inv_ev.py_impl import CDF_inv_ev as _ref_cdfinv

    M, num_ui, cursor, N = 4, 8, 12, 64
    BinSize, P_qc, N_qb = 1e-4, 1e-4, 6
    param = _base_param(M=M, num_ui=num_ui, Nb=4)
    param.N_qb = N_qb
    param.clip_method = 'slow'
    param.P_qc = P_qc
    op = _base_op(COMPUTE_COM=False)
    op.BinSize = BinSize

    pulse = np.zeros(N)
    pulse[cursor - M] = 0.05
    pulse[cursor] = 1.0
    pulse[cursor + M] = 0.2
    pulse[cursor + 2 * M] = 0.1
    ch = _single_channel(M=M, N_total=N, cursor=cursor)
    ch.pulse_response_w_CFT_TXFFE_noRxFFE = pulse.copy()
    h = np.zeros(N)
    h[cursor] = 1.0
    result = _base_result()

    def stub_S_RN(fvec, g_dc, g_dc2, p):
        return p.eta_0 * np.ones(len(fvec))

    get_PSDs(result, h, cursor, None, -1.0, None, param, [ch], op, _S_RN_fn=stub_S_RN)

    # (1) ctle_signal_sigma must be set on the slow path (MATLAB 6725).
    assert hasattr(result, 'ctle_signal_sigma'), \
        "slow clip path must set result.ctle_signal_sigma (MATLAB line 6725)"

    # (2) adc_clip must equal the P_qc quantile of the signal+noise PDF, not the
    #     max(|.|)+3*sigma heuristic. Build the oracle from the audited helpers,
    #     using the same sampled pulse and the sigma_noise the run produced.
    sample_idx0 = cursor % M
    sampled_pr = pulse[sample_idx0:].ravel()[::M]
    sampled_pr = np.concatenate([sampled_pr, np.zeros(max(0, num_ui - len(sampled_pr)))])[:num_ui]
    # sigma_noise exactly as get_PSDs computes it (getattr with 0 default; the sigma_noise
    # expression is pre-existing and unchanged by B01-D2, which fixes the clip METHOD).
    sigma_noise = np.sqrt(getattr(result, 'S_in_rms', 0)**2 + getattr(result, 'S_rn_rms', 0)**2
                          + getattr(result, 'S_xn_rms', 0)**2 + getattr(result, 'S_tn_rms', 0)**2
                          + getattr(result, 'S_rj_rms', 0)**2)
    sig_pdf = _ref_gpss(sampled_pr, param.levels, BinSize)
    noise_pdf = SimpleNamespace(**vars(sig_pdf))
    noise_pdf.y = (1.0 / (np.sqrt(2 * np.pi) * sigma_noise)
                   * np.exp(-sig_pdf.x**2 / (2 * sigma_noise**2)) * BinSize)
    sig_noise = _ref_conv(sig_pdf, noise_pdf)
    ref_adc = -_ref_cdfinv(P_qc, sig_noise, np.cumsum(sig_noise.y))

    assert abs(result.adc_clip - ref_adc) <= 1e-9 * max(1.0, abs(ref_adc)), (
        "slow adc_clip=%g; expected P_qc quantile %g (MATLAB 6724). Pre-fix "
        "max(|.|)+3*sigma heuristic differs." % (result.adc_clip, ref_adc))
