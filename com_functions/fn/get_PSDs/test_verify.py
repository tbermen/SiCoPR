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


# ===========================================================================
# Oracle-backed cases.
#
# Everything above this line was written from a READING of the MATLAB, which is
# what this section exists to replace: a reading cannot tell
# `sum(reshape(v,num_ui,M).')` from a row-major fold, nor a 1-based sampling
# phase from a 0-based one.  The blocks below were produced by EXECUTING the
# reference under Octave on these exact inputs (tools/octave_oracle.py) and
# pinning what came back.
#
# Two things the fixtures below are deliberate about:
#
#  * They inject the REAL S_RN / S_IN / H_interp, exactly as sicopr.py's
#    `_wired_get_PSDs = partial(get_PSDs, _S_RN_fn=S_RN, _S_IN_fn=S_IN,
#    _H_interp_fn=H_interp)` does.  get_PSDs falls back to three flat stubs
#    when a dependency is not injected (_S_RN returns eta_0*ones, _S_IN returns
#    zeros, _H_interp is a linear np.interp), and a value pinned from a bare
#    call would be pinning the stub rather than the port.
#  * The pulse responses are REAL-valued, so no abs() of a complex number is
#    involved and the comparisons can be exact rather than tolerance-fitted.
# ===========================================================================

from com_functions.fn.S_RN.py_impl import S_RN as _real_S_RN
from com_functions.fn.S_IN.py_impl import S_IN as _real_S_IN
from com_functions.fn.H_interp.py_impl import H_interp as _real_H_interp

_M, _NUM_UI, _FB = 4, 8, 53.125e9
_N = _NUM_UI * _M * 2          # 64 samples
_CURSOR = 13                   # 0-based; MATLAB 1-based cursor_i = 14


def _fx():
    """The exact vectors handed to Octave."""
    t = np.arange(_N, dtype=float)
    return dict(
        pr1=np.exp(-((t - _CURSOR) / 5.0) ** 2) * 0.9 + 0.03 * np.sin(t / 3.0),
        pr2=0.11 * np.exp(-((t - 20.0) / 7.0) ** 2) + 0.004 * np.cos(t / 2.0),
        pr3=0.07 * np.exp(-((t - 9.0) / 4.0) ** 2) - 0.006 * np.sin(t / 5.0),
        cir1=0.25 * np.exp(-((t - 6.0) / 3.0) ** 2) + 0.01 * np.cos(t / 4.0),
        h=0.8 * np.exp(-((t - _CURSOR) / 4.5) ** 2) + 0.02 * np.sin(t / 2.5) - 0.01,
    )


def _oracle_param(**kw):
    p = SimpleNamespace(
        num_ui_RXFF_noise=_NUM_UI, samples_per_ui=_M, levels=4, fb=_FB,
        SNR_TX=30.0, RxFFE_cmx=0,
        bmax=np.array([0.7, 0.2, 0.2, 0.2]),
        bmin=np.array([-0.7, -0.2, -0.2, -0.2]),
        ndfe=4, eta_0=1.7e-4, A_DD=0.05, sigma_RJ=0.01,
        N_qb=0, clip_method='Fast', S_tn_w_AM=0, P_qc=1e-4,
        CTLE_fp1=np.array([1.0e10]), CTLE_fz=np.array([0.6e10]),
        CTLE_fp2=np.array([3.0e10]), f_HP=np.array([0.6e9]), f_r=0.75)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _oracle_chdata(n_chan=3):
    f = _fx()
    chs = [SimpleNamespace(ctle_imp_response=f['cir1'].copy(),
                           pulse_response_w_CFT_TXFFE_noRxFFE=f['pr1'].copy())]
    for k in ('pr2', 'pr3')[:n_chan - 1]:
        chs.append(SimpleNamespace(pulse_response_w_CFT_TXFFE_noRxFFE=f[k].copy()))
    return chs


def _oracle_op(**kw):
    o = SimpleNamespace(COMPUTE_COM=False, WO_TXFFE=False, TDMODE=False,
                        PSDRXCAL=False, LIMIT_JITTER_CONTRIB_TO_DFE_SPAN=False)
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def _run(result=None, op=None, param=None, chdata=None, cursor=_CURSOR):
    """get_PSDs wired the way sicopr.py wires it."""
    return get_PSDs(result if result is not None else _base_result(),
                    _fx()['h'], cursor, None, -6.0, -6.0,
                    param if param is not None else _oracle_param(),
                    chdata if chdata is not None else _oracle_chdata(3),
                    op if op is not None else _oracle_op(),
                    _S_RN_fn=_real_S_RN, _S_IN_fn=_real_S_IN,
                    _H_interp_fn=_real_H_interp)


def _same(got, want, what):
    g = np.atleast_1d(np.asarray(got, dtype=float)).ravel()
    w = np.atleast_1d(np.asarray(want, dtype=float)).ravel()
    assert g.shape == w.shape, '%s: shape %s, reference %s' % (what, g.shape, w.shape)
    rel = np.max(np.abs(g - w) / np.maximum(np.abs(w), 1e-300))
    assert rel <= 2e-15, '%s: max rel %.3e\n got  %r\n want %r' % (what, rel, g, w)


# ---------------------------------------------------------------------------
# COM Octave (get_PSDs, 4p16p0): the main branch -- OP.WO_TXFFE=0,
# OP.COMPUTE_COM=0, param.N_qb=0, three channels, cursor_i=14 (1-based).
# ---------------------------------------------------------------------------
_A_S_XN = [1.3632078927433882e-12, 5.6123818281844662e-13, 6.560816315341924e-14, 1.2472809581321203e-14, 4.9239122273379246e-15, 1.2472809581321203e-14, 6.560816315341924e-14, 5.6123818281844662e-13]
_A_S_XN_RMS = 0.13257529106916757
_A_S_TN = [3.5258358929362313e-14, 2.4777188293801312e-14, 1.4794371388569545e-14, 3.6060366903689332e-15, 4.4559220435872906e-16, 3.6060366903689332e-15, 1.4794371388569545e-14, 2.4777188293801312e-14]
_A_S_TN_RMS = 0.028470142295443777
_A_S_JN = [6.3614782157115998e-19, 2.8664958883693906e-14, 2.7481521155844963e-14, 8.7817032334236115e-15, 5.5855660240570342e-20, 8.7817032334236115e-15, 2.7481521155844963e-14, 2.8664958883693906e-14]
_A_S_JN_RMS = 0.02936549045103205
_A_S_RJ_JN = [2.4467223906583075e-20, 1.1024984186036116e-15, 1.0569815829171139e-15, 3.3775781667013888e-16, 2.1482946246373207e-21, 3.3775781667013888e-16, 1.0569815829171139e-15, 1.1024984186036116e-15]
_A_S_RJ_RMS = 0.0057590464936936531
_A_S_N = [1.3984668878205721e-12, 6.1468032999594184e-13, 1.0788405569783375e-13, 2.486054950511375e-14, 5.3695602873568943e-15, 2.486054950511375e-14, 1.0788405569783375e-13, 6.1468032999594184e-13]
_A_S_N_RMS = 0.13874108560129894
_A_FVEC = [0, 6640625000, 13281250000, 19921875000, 26562500000, 33203125000, 39843750000, 46484375000, 53125000000, 59765625000, 66406250000, 73046875000, 79687500000, 86328125000, 92968750000, 99609375000, 106250000000]
# result.hk(2).hrn and result.hk(3).hrn -- the crosstalk pulse resampled at the
# max-norm phase, which is where a 0-based/1-based phase slip would show.
_A_HRN2 = [0.0035797998099917809, -0.0020898094502243029, 0.0084668495679615721, 0.044373089031770757, 0.089134777465193854, 0.10587570641192351, 0.070032227648538808, 0.01964122189750277]
_A_HRN3 = [9.0078737441025292e-05, 0.020702734973153586, 0.06415691421473084, 0.022658552651072181, 0.0028153413343723799, 0.0052380933207675975, 0.0057535535254410611, 0.0027876130774547002]


def test_oracle_main_branch():
    r = _run()
    _same(r.fvec, _A_FVEC, 'fvec')
    _same(r.S_xn, _A_S_XN, 'S_xn')
    _same(r.S_xn_rms, _A_S_XN_RMS, 'S_xn_rms')
    _same(r.S_tn, _A_S_TN, 'S_tn')
    _same(r.S_tn_rms, _A_S_TN_RMS, 'S_tn_rms')
    _same(r.S_jn, _A_S_JN, 'S_jn')
    _same(r.S_jn_rms, _A_S_JN_RMS, 'S_jn_rms')
    _same(r.S_rj_jn, _A_S_RJ_JN, 'S_rj_jn')
    _same(r.S_rj_rms, _A_S_RJ_RMS, 'S_rj_rms')
    _same(r.S_n, _A_S_N, 'S_n')
    _same(r.S_n_rms, _A_S_N_RMS, 'S_n_rms')
    assert r.S_qn == 0.0 and r.S_qn_rms == 0.0


def test_oracle_crosstalk_phase_and_hrn():
    """The max-norm sampling phase per crosstalk channel, via hk(k).hrn.

    result.iphase is the one field the port keeps 0-based on purpose: get_pdf
    consumes it as a 0-based column (`ixphase`), the way cursor_i is 0-based
    throughout.  MATLAB's iphase(2:3) here is [2 2]; the decimated response it
    selects is what hrn pins, so the phase is checked by its consequence.
    """
    r = _run()
    _same(r.hk[1].hrn, _A_HRN2, 'hk[1].hrn')
    _same(r.hk[2].hrn, _A_HRN3, 'hk[2].hrn')
    assert list(np.asarray(r.iphase).ravel()[1:]) == [1, 1], \
        'iphase (0-based) should be MATLAB [2 2] minus one'


# ---------------------------------------------------------------------------
# COM Octave (get_PSDs + S_RN, 4p16p0): OP.WO_TXFFE=1, OP.COMPUTE_COM=0,
# OP.PSDRXCAL=0, G_DC=-6, G_DC2=-6.  S_RN is the real reference function, not
# the flat eta_0*ones stub get_PSDs falls back to.
# ---------------------------------------------------------------------------
_B_S_RN = [1.5435975879093839e-14, 9.9287852406385035e-14, 1.7330528432947109e-13, 2.1312545479015342e-13, 2.2613655822533443e-13, 2.1312545479015342e-13, 1.7330528432947109e-13, 9.9287852406385035e-14]
_B_S_RN_RMS = 0.089750446533663736


def test_oracle_wo_txffe_receiver_noise():
    r = _run(op=_oracle_op(WO_TXFFE=True))
    _same(r.S_rn, _B_S_RN, 'S_rn')
    _same(r.S_rn_rms, _B_S_RN_RMS, 'S_rn_rms')
    assert r.S_in == 0 and r.S_in_rms == 0


# ---------------------------------------------------------------------------
# COM Octave (get_PSDs + S_RN + S_IN + N_s + H_interp, 4p16p0): OP.WO_TXFFE=1,
# OP.PSDRXCAL=1, OP.RIT_REF_PTR='clause_179', sigma_ns=0.007, f_hp=0.7e9,
# chdata(1).faxis = linspace(0,60e9,40), chdata(3).sdd21p = 0.6*exp(-f/9e10)+0.02.
# H_noise is the field MATLAB L7215 hangs on result and the port used to drop.
# ---------------------------------------------------------------------------
_R_H_NOISE = [0.62, 0.57732298387321801, 0.53768151063929959, 0.50085967286507294, 0.46665691389488884, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
_R_S_IN = [0, 3.0763272304913908e-16, 4.3121573852493479e-16, 3.7764463871369111e-16, 5.5610491806079807e-16, 3.7764463871369111e-16, 4.3121573852493479e-16, 3.0763272304913908e-16]
_R_S_IN_RMS = 0.0043036389497371139


def test_oracle_psdrxcal_input_noise():
    chs = _oracle_chdata(3)
    fax = np.linspace(0.0, 6e10, 40)
    chs[0].faxis = fax
    chs[2].sdd21p = 0.6 * np.exp(-fax / 9e10) + 0.02
    r = _run(op=_oracle_op(WO_TXFFE=True, PSDRXCAL=True,
                           RIT_REF_PTR='clause_179'),
             param=_oracle_param(sigma_ns=0.007, f_hp=0.7e9), chdata=chs)
    assert hasattr(r, 'H_noise'), \
        'MATLAB L7215 assigns result.H_noise; the port dropped it into a local'
    _same(np.real(r.H_noise), _R_H_NOISE, 'H_noise')
    _same(r.S_in, _R_S_IN, 'S_in')
    _same(r.S_in_rms, _R_S_IN_RMS, 'S_in_rms')


# ---------------------------------------------------------------------------
# COM Octave (get_PSDs, 4p16p0): OP.COMPUTE_COM=1 with OP.WO_TXFFE=0 --
# H_rxffe_2 from result.w=[0.13 0.92 -0.21 0.05] at RxFFE_cmx=1, the four
# cached PSDs scaled by it, and S_isi / S_G / Sn_rho.
# ---------------------------------------------------------------------------
_C_H_RXFFE_2 = [0.79210000000000025, 0.78177225124899763, 0.87250000000000016, 1.0380277487510023, 1.1025, 1.0380277487510023, 0.87250000000000016, 0.78177225124899763]
_C_S_XN = [2.4555100000000007e-12, 2.5016712039967926e-12, 2.8792500000000007e-12, 3.5292943457534081e-12, 3.8587499999999999e-12, 3.7368998955036081e-12, 3.2282500000000005e-12, 2.9707345547461911e-12]
_C_S_TN = [1.9010400000000006e-14, 2.1889623034971933e-14, 2.7920000000000009e-14, 3.7368998955036083e-14, 4.4100000000000009e-14, 4.5673220945044103e-14, 4.1880000000000007e-14, 4.0652157064947883e-14]
_C_S_ISI = [1.5479044589255987e-12, 1.470743456920914e-12, 1.2685908544081077e-12, 1.043995439283364e-12, 9.4439162466674204e-13, 1.043995439283364e-12, 1.2685908544081077e-12, 1.470743456920914e-12]
_C_S_ISI_RMS = 0.25845261060864938
_C_S_G = [2.9485660000000007e-14, 3.2436863610846233e-14, 3.861800000000001e-14, 4.8303223928911989e-14, 5.5202500000000011e-14, 5.6815051468670208e-14, 5.2927000000000011e-14, 5.1668460991571578e-14]
_C_S_G_RMS = 0.04926318399043042
_C_SN_RHO = [4.0419300589255995e-12, 4.0152490954701645e-12, 4.199720854408108e-12, 4.6393432834693264e-12, 4.8792916246667424e-12, 4.8594051662045383e-12, 4.5696608544081083e-12, 4.5124562472645268e-12]
_C_SN_RHO_RMS = 0.48701497191351945
_C_S_N_RMS = 0.41277818612119033


def _com_seed():
    r = _base_result()
    r.w = np.array([0.13, 0.92, -0.21, 0.05])
    k = np.arange(1, 9)
    r.S_xn = np.ones(8) * 3e-12 + k * 1e-13
    r.S_tn = np.ones(8) * 2e-14 + k * 4e-15
    r.S_jn = np.ones(8) * 1e-14 + k * 2e-15
    r.S_rj_jn = np.ones(8) * 5e-16 + k * 1e-16
    r.S_rn = np.ones(8) * 7e-15
    r.S_in = np.ones(8) * 3e-15
    return r


def test_oracle_compute_com_branch():
    r = _run(result=_com_seed(), op=_oracle_op(COMPUTE_COM=True),
             param=_oracle_param(RxFFE_cmx=1))
    _same(r.H_rxffe_2, _C_H_RXFFE_2, 'H_rxffe_2')
    _same(r.S_xn, _C_S_XN, 'S_xn')
    _same(r.S_tn, _C_S_TN, 'S_tn')
    _same(r.S_isi, _C_S_ISI, 'S_isi')
    _same(r.S_isi_rms, _C_S_ISI_RMS, 'S_isi_rms')
    _same(r.S_G, _C_S_G, 'S_G')
    _same(r.S_G_rms, _C_S_G_RMS, 'S_G_rms')
    _same(r.Sn_rho, _C_SN_RHO, 'Sn_rho')
    _same(r.Sn_rho_rms, _C_SN_RHO_RMS, 'Sn_rho_rms')
    _same(r.S_n_rms, _C_S_N_RMS, 'S_n_rms')


def test_oracle_H_rxffe_2_is_not_folded():
    """H_rxffe_2 is the symbol-rate band double-sided, NOT the aliased fold.

    Folding M copies would multiply the noise enhancement by ~M and close the
    eye; the reference takes H_rxffe_2_of_f(1:num_ui/2+1) and mirrors it.
    """
    r = _run(result=_com_seed(), op=_oracle_op(COMPUTE_COM=True),
             param=_oracle_param(RxFFE_cmx=1))
    assert len(np.asarray(r.H_rxffe_2).ravel()) == _NUM_UI
    _same(r.H_rxffe_2, _C_H_RXFFE_2, 'H_rxffe_2')


# ---------------------------------------------------------------------------
# COM Octave (get_PSDs, 4p16p0): OP.LIMIT_JITTER_CONTRIB_TO_DFE_SPAN=1, where
# the early/late samples are h(cursor_i-/+1 + M*(-1:ndfe)) rather than the
# whole decimated response.
# ---------------------------------------------------------------------------
_E_S_JN = [4.4257413018827847e-16, 2.2219256472989513e-14, 2.9883160066809375e-14, 1.1342714154896401e-14, 1.19255475543099e-16, 1.1342714154896401e-14, 2.9883160066809375e-14, 2.2219256472989513e-14]
_E_S_JN_RMS = 0.029092293511589658
_E_S_RJ_JN = [1.7022081930318404e-17, 8.5458678742267362e-16, 1.1493523102618989e-15, 4.3625823672678461e-16, 4.5867490593499614e-18, 4.3625823672678461e-16, 1.1493523102618989e-15, 8.5458678742267362e-16]
_E_S_RJ_RMS = 0.0057054681657986318


def test_oracle_limit_jitter_branch():
    r = _run(op=_oracle_op(LIMIT_JITTER_CONTRIB_TO_DFE_SPAN=True))
    _same(r.S_jn, _E_S_JN, 'S_jn')
    _same(r.S_jn_rms, _E_S_JN_RMS, 'S_jn_rms')
    _same(r.S_rj_jn, _E_S_RJ_JN, 'S_rj_jn')
    _same(r.S_rj_rms, _E_S_RJ_RMS, 'S_rj_rms')


def test_limit_jitter_refuses_index_before_the_pulse():
    """MATLAB refuses h(cursor_i-1+M*(-1)) when that lands before the array.

    COM Octave, cursor_i=4 (1-based) with M=4:
      error: h(-1): subscripts must be either integers 1 to (2^63)-1 or logicals
    The port masked the out-of-range entries out and returned a shorter h_J,
    whose FFT is a perfectly plausible S_jn computed from the wrong samples.
    """
    with pytest.raises(IndexError, match=r'h\(-1\)'):
        _run(op=_oracle_op(LIMIT_JITTER_CONTRIB_TO_DFE_SPAN=True), cursor=3)


def test_limit_jitter_refuses_index_past_the_pulse():
    """COM Octave, param.ndfe=14 with len(h)=64:
      error: h(69): out of bound 64 (dimensions are 64x1)
    Note 69, not the first offender 65: for an over-bound index the reference
    names the LARGEST one (h=1:10; h([3 12 15]) also reports 15).
    """
    p = _oracle_param(ndfe=14, bmax=0.2 * np.ones(14), bmin=-0.2 * np.ones(14))
    with pytest.raises(IndexError, match=r'h\(69\): out of bound 64'):
        _run(op=_oracle_op(LIMIT_JITTER_CONTRIB_TO_DFE_SPAN=True), param=p)


# ---------------------------------------------------------------------------
# COM Octave (get_PSDs, 4p16p0): quantisation noise, both clip methods.
# 'Fast' is sum(abs(sampled_pulse_response)); 'Slow' is the P_qc quantile of
# the signal-plus-Gaussian-noise PDF (needs get_pdf_from_sampled_signal,
# conv_fct, CDF_inv_ev), with OP.BinSize=1e-3, P_qc=1e-4, N_qb=6 and a seeded
# result.S_rn_rms=0.004 / result.S_in_rms=0.002.
# ---------------------------------------------------------------------------
_F_ADC_CLIP = 2.0461245773740449
_F_S_QN0 = 6.6185498968126354e-15
_F_S_QN_RMS = 0.018751278976863718
_G_ADC_CLIP = 2.226
_G_CTLE_SIGMA = 0.83361527302522165
_G_S_QN_RMS = 0.020399709511366781


def test_oracle_quantisation_fast_clip():
    r = _run(param=_oracle_param(N_qb=6, clip_method='Fast'))
    _same(r.adc_clip, _F_ADC_CLIP, 'adc_clip')
    _same(r.S_qn, np.full(_NUM_UI, _F_S_QN0), 'S_qn')
    _same(r.S_qn_rms, _F_S_QN_RMS, 'S_qn_rms')


def test_oracle_quantisation_slow_clip():
    res = _base_result()
    res.S_rn_rms = 0.004
    res.S_in_rms = 0.002
    op = _oracle_op()
    op.BinSize = 1e-3
    op.FAST_NOISE_CONV = 0
    r = _run(result=res, op=op,
             param=_oracle_param(N_qb=6, clip_method='Slow', P_qc=1e-4))
    _same(r.adc_clip, _G_ADC_CLIP, 'adc_clip')
    _same(r.ctle_signal_sigma, _G_CTLE_SIGMA, 'ctle_signal_sigma')
    _same(r.S_qn_rms, _G_S_QN_RMS, 'S_qn_rms')


# ---------------------------------------------------------------------------
# COM Octave (get_PSDs, 4p16p0): OP.TDMODE=1 with chdata(1).ctle_pulse_response
# present, which bypasses the filter(ones(1,M),1,...) running sum; and the
# single-channel case, where S_xn stays scalar 0 and iphase stays 1.
# ---------------------------------------------------------------------------
_I_S_TN = [1.9427194898547924e-14, 1.075024379074675e-14, 2.1590322818502463e-15, 2.8777202421444819e-16, 2.6878233837280667e-17, 2.8777202421444819e-16, 2.1590322818502463e-15, 1.075024379074675e-14]
_I_S_TN_RMS = 0.017448796503785655
_H_S_TN_RMS = 0.028470142295443777
_H_S_N_RMS = 0.040900868349614178


def test_oracle_tdmode_uses_ctle_pulse_response():
    chs = _oracle_chdata(3)
    chs[0].ctle_pulse_response = _fx()['pr1'] * 0.5
    r = _run(op=_oracle_op(TDMODE=True), chdata=chs)
    _same(r.S_tn, _I_S_TN, 'S_tn')
    _same(r.S_tn_rms, _I_S_TN_RMS, 'S_tn_rms')


def test_oracle_single_channel_no_crosstalk():
    r = _run(chdata=_oracle_chdata(1))
    assert r.S_xn == 0 and r.S_xn_rms == 0
    assert r.hk == [] and r.iphase == 1
    _same(r.S_tn_rms, _H_S_TN_RMS, 'S_tn_rms')
    _same(r.S_n_rms, _H_S_N_RMS, 'S_n_rms')
