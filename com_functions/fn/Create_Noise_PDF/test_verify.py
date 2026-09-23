"""Verification tests for Create_Noise_PDF().

# ============================================================
# MATLAB GROUND TRUTH (lines 1552-1680)
# Computes combined interference and noise PDF/CDF and NS noise struct.
# Non-MMSE, RX_CALIBRATION=0: uses sigma_TX from SNR_TX param.
# N_qb=0: skips quantization step.
# Returns (PDF, CDF, NS).
# ============================================================
"""
import numpy as np
import pytest
from types import SimpleNamespace
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import com_functions.fn.Create_Noise_PDF.py_impl as _mod
from com_functions.fn.Create_Noise_PDF.py_impl import Create_Noise_PDF


def _gaussian_pdf(sigma=0.05, bin_size=0.001):
    x = np.arange(-round(5*sigma/bin_size), round(5*sigma/bin_size)+1) * bin_size
    y = np.exp(-x**2/(2*sigma**2)); y /= y.sum()
    return SimpleNamespace(BinSize=bin_size, Min=int(round(x[0]/bin_size)), y=y, x=x)


def _delta_pdf(bin_size=0.001):
    return SimpleNamespace(BinSize=bin_size, Min=0, y=np.array([1.0]), x=np.array([0.0]))


def _param():
    bs = 0.001
    return SimpleNamespace(
        levels=4, specBER=1e-6,
        R_LM=0.4, SNR_TX=30.0,
        sigma_RJ=0.01, sigma_X=1.0,
        A_DD=0.01,
        delta_y=bs,
        N_qb=0,
        Noise_Crest_Factor=0,
        CTLE_type='CL93',
        ctle_gdc_values=np.array([-6.0, 0.0]),
        CTLE_fp1=np.array([10e9]), CTLE_fp2=np.array([20e9]),
        CTLE_fz=np.array([5e9]),
        g_DC_HP_values=np.array([0.0]),
        f_HP=np.array([1e9]),
        f_HP_Z=np.array([1e9]), f_HP_P=np.array([2e9]),
        number_of_s4p_files=1,
    )


def _fom_result():
    return SimpleNamespace(
        sigma_N=0.02,
        ctle=1,
        best_G_high_pass=1,
        txffe=np.array([0.0, 1.0, 0.0]),
        cur=2,
        h_J=np.zeros(5),
    )


def _OP():
    return SimpleNamespace(
        RX_CALIBRATION=0,
        FFE_OPT_METHOD='',
        RxFFE=False,
        SNR_TXwC0=False,
        PSDRXCAL=False,
        force_BBN_Q_factor=False,
        BBN_Q_factor=7.0,
    )


def _chdata():
    bs = 0.001
    pdfr = _delta_pdf(bs)
    return [SimpleNamespace(type='THRU', pdfr=pdfr)]


def test_returns_three_outputs():
    """Create_Noise_PDF returns (PDF, CDF, NS)."""
    PDF, CDF, NS = Create_Noise_PDF(0.5, _param(), _fom_result(), _chdata(), _OP(), 0.0)
    assert hasattr(PDF, 'y')
    assert len(CDF) > 0
    assert hasattr(NS, 'sigma_N')


def test_pdf_y_sums_to_one():
    """Combined PDF.y sums to 1."""
    PDF, CDF, NS = Create_Noise_PDF(0.5, _param(), _fom_result(), _chdata(), _OP(), 0.0)
    assert float(np.sum(PDF.y)) == pytest.approx(1.0, abs=1e-4)


def test_cdf_monotone():
    """CDF is non-decreasing."""
    _, CDF, _ = Create_Noise_PDF(0.5, _param(), _fom_result(), _chdata(), _OP(), 0.0)
    assert np.all(np.diff(CDF) >= -1e-12)


def test_sigma_TX_positive():
    """NS.sigma_TX > 0."""
    _, _, NS = Create_Noise_PDF(0.5, _param(), _fom_result(), _chdata(), _OP(), 0.0)
    assert NS.sigma_TX > 0


def test_ns_has_required_fields():
    """NS has sigma_N, sigma_TX, sigma_G, ber_q, noise_pdf."""
    _, _, NS = Create_Noise_PDF(0.5, _param(), _fom_result(), _chdata(), _OP(), 0.0)
    for field in ['sigma_N', 'sigma_TX', 'sigma_G', 'ber_q', 'noise_pdf',
                  'sci_pdf', 'gaussian_noise_pdf', 'p_DD', 'jitt_pdf']:
        assert hasattr(NS, field), f'Missing: {field}'


def test_N_qb_nonzero_invokes_quantization(monkeypatch):
    """N_qb != 0 routes through adjust_Rx_noise_for_quantization (no longer a stub).

    The quantization helper is a top-level fn in the assembled module; here we
    inject a spy to confirm the dispatch reaches it and the combined PDF/CDF are
    still returned. (The helper itself is unit-tested in its own directory.)"""
    p = _param()
    p.N_qb = 4
    calls = {'n': 0}

    def spy_adjust(pdf, NS, chdata, fom_result, param, OP):
        calls['n'] += 1
        return chdata, NS, pdf  # pass the combined PDF through unchanged

    monkeypatch.setattr(_mod, 'adjust_Rx_noise_for_quantization', spy_adjust, raising=False)
    PDF, CDF, NS = Create_Noise_PDF(0.5, p, _fom_result(), _chdata(), _OP(), 0.0)
    assert calls['n'] == 1
    assert hasattr(PDF, 'y')
    assert len(CDF) == len(PDF.y)


# ---------------------------------------------------------------------------
# COM Octave oracle (2026-09-23): the CTLE_type switch at MATLAB L1585-1594.
# It only runs under OP.RX_CALIBRATION, where it builds the low-frequency
# factor H_low2 of the calibration transfer function H_ctf2, which goes into
# get_sigma_noise and comes back as sigma_ne. sigma_ne is a local, so the only
# way it is observable from outside is NS.ne_noise_pdf and everything
# convolved with it -- NS.gaussian_noise_pdf, NS.noise_pdf, PDF and CDF. Those
# are what is pinned below.
#
# Create_Noise_PDF was run verbatim under Octave from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m for this function) with the deps
# get_sigma_noise, normal_dist, conv_fct, d_cpdf, get_pdf_from_sampled_signal
# and Init_PDF_Fast, plus octave/patches/erfcinv.m (Octave's own erfcinv is
# inaccurate in the tail; the patch is bit-identical to scipy's).
#
# ctle = 2 and best_G_high_pass = 1 (MATLAB 1-based) are deliberately
# different, and every parameter vector holds two distinct values, so reading
# one index where the reference reads the other changes sigma_ne.
#
# Not bit-exact: the largest per-element relative difference is 1.5e-15,
# from complex abs()/mean() accumulation order. Pinned at rtol=1e-13.
# ---------------------------------------------------------------------------
_RX_TOL = dict(rtol=1e-13, atol=0.0)
_RX_FAXIS = np.arange(0, 41) * 1e9
_RX_SDD21 = (0.8 * np.exp(-_RX_FAXIS / 30e9)
             * np.exp(-1j * 2 * np.pi * _RX_FAXIS * 1e-10))


def _rx_pdfr(bs=1e-3):
    """d_cpdf(1e-3, [-0.05 0.05], [0.5 0.5]) -- verified bit-identical to the
    Octave d_cpdf the oracle run used, so the pin does not depend on it."""
    y = np.zeros(101)
    y[0] = 0.5
    y[-1] = 0.5
    return SimpleNamespace(BinSize=bs, Min=-50, y=y,
                           x=np.arange(-50, 51) * bs)


def _rx_inputs(ctle_type):
    p = SimpleNamespace(
        fb=25e9, f_r=0.75, f_hp=0.5e9,
        CTLE_fz=np.array([5e9, 6e9]), CTLE_fp1=np.array([10e9, 11e9]),
        CTLE_fp2=np.array([20e9, 21e9]), ctle_gdc_values=np.array([0.0, -3.0]),
        CTLE_type=ctle_type, g_DC_HP_values=np.array([-4.0, -6.0]),
        f_HP=np.array([0.667e9, 1.5e9]), f_HP_Z=np.array([1e9, 2e9]),
        f_HP_P=np.array([2e9, 3e9]),
        levels=4, specBER=1e-6, R_LM=0.95, SNR_TX=31.0, sigma_RJ=0.01,
        sigma_X=0.3, A_DD=0.02, delta_y=1e-3, N_qb=0, Noise_Crest_Factor=0,
        number_of_s4p_files=2)
    fom = SimpleNamespace(sigma_N=0.02, ctle=2, best_G_high_pass=1,
                          txffe=np.array([0.0, 1.0, 0.0]), cur=2,
                          h_J=np.array([0.01, -0.02, 0.005, 0.0, 0.0]))
    pdfr = _rx_pdfr()
    cd = [SimpleNamespace(type='THRU', faxis=_RX_FAXIS, sdd21=_RX_SDD21,
                          pdfr=pdfr),
          SimpleNamespace(type='FEXT', faxis=_RX_FAXIS, sdd21=_RX_SDD21,
                          pdfr=pdfr)]
    op = SimpleNamespace(RX_CALIBRATION=1, FFE_OPT_METHOD='', RxFFE=0,
                         SNR_TXwC0=0, PSDRXCAL=0, force_BBN_Q_factor=0,
                         BBN_Q_factor=7.0)
    return p, fom, cd, op


# COM Octave, CTLE_type='CL120d', OP.RX_CALIBRATION=1, sigma_bn=0.005.
_OCT_D_NE_PDF_Y = [
    6.3928783250548288e-21, 1.3009984145160735e-19, 2.3820013139894876e-18,
    3.9236668645195453e-17, 5.8146979230111387e-16, 7.7525948694703177e-15,
    9.2993356169943148e-14, 1.0035560998559365e-12, 9.743529153351904e-12,
    8.5109065816292965e-11, 6.6883695190183247e-10, 4.7287852875941378e-09,
    3.0079030359887667e-08, 1.7213253699458252e-07, 8.8623114425235333e-07,
    4.105026331657868e-06, 1.710683975134868e-05, 6.4136983132385855e-05,
    0.00021633768056832933, 0.00065650904105637271, 0.0017923967765437004,
    0.0044026331350060077, 0.0097291688853302968, 0.019343002474019121,
    0.034598473125241551, 0.055676878644132899, 0.080607922312486049,
    0.10499424027194763, 0.12303767739123002, 0.1297166349538624,
    0.12303767739123002, 0.10499424027194763, 0.080607922312486049,
    0.055676878644132899, 0.034598473125241551, 0.019343002474019121,
    0.0097291688853302968, 0.0044026331350060077, 0.0017923967765437004,
    0.00065650904105637271, 0.00021633768056832933, 6.4136983132385855e-05,
    1.710683975134868e-05, 4.105026331657868e-06, 8.8623114425235333e-07,
    1.7213253699458252e-07, 3.0079030359887667e-08, 4.7287852875941378e-09,
    6.6883695190183247e-10, 8.5109065816292965e-11, 9.743529153351904e-12,
    1.0035560998559365e-12, 9.2993356169943148e-14, 7.7525948694703177e-15,
    5.8146979230111387e-16, 3.9236668645195453e-17, 2.3820013139894876e-18,
    1.3009984145160735e-19, 6.3928783250548288e-21,
]
# (index, value) probes into the returned CDF, which is cumsum(PDF.y).
_OCT_D_CDF = [
    (0, 5.9798017425457086e-43),
    (100, 2.4543371282393799e-16),
    (300, 2.0554366088140698e-05),
    (543, 0.50241845909415406),
    (800, 0.99999452763368302),
    (1086, 0.99999999999999956),
]


def test_octave_rx_calibration_CL120d():
    """COM Octave, CTLE_type='CL120d' under RX_CALIBRATION.

    H_low2 uses param.g_DC_HP_values and param.f_HP, both indexed by
    fom_result.best_G_high_pass (= 1), not by fom_result.ctle (= 2)."""
    p, fom, cd, op = _rx_inputs('CL120d')
    PDF, CDF, NS = Create_Noise_PDF(0.5, p, fom, cd, op, 0.005)

    # sigma_ne is a local; ne_noise_pdf is its fingerprint.
    assert NS.ne_noise_pdf.Min == -29
    np.testing.assert_allclose(NS.ne_noise_pdf.y, _OCT_D_NE_PDF_Y, **_RX_TOL)

    assert len(CDF) == 1087
    for i, v in _OCT_D_CDF:
        assert CDF[i] == pytest.approx(v, rel=1e-13), 'CDF[%d]' % i

    assert PDF.Min == -543
    # PDF.y probes. Not argmax: the two-spike sci_pdf makes the combined PDF
    # symmetric, so bins 525 and 561 tie and the winner is last-bit noise.
    for i, v in [(200, 6.4572257798199416e-11),
                 (400, 0.00066973591689569888),
                 (525, 0.004844231770843881),
                 (543, 0.0048369181883080089),
                 (700, 0.00037238993823096849)]:
        assert PDF.y[i] == pytest.approx(v, rel=1e-13), 'PDF.y[%d]' % i


def test_octave_rx_calibration_CL120d_scalars():
    """COM Octave, the same run: the NS scalars, all bit-exact.

    NS.sigma_hp closes get_sigma_noise's second output, which is only ever
    computed on the RX_CALIBRATION path."""
    p, fom, cd, op = _rx_inputs('CL120d')
    _, _, NS = Create_Noise_PDF(0.5, p, fom, cd, op, 0.005)
    assert NS.sigma_hp == 0.004522199050777595
    assert NS.sigma_N == 0.02
    assert NS.sigma_TX == 0.044500783125228212
    assert NS.sigma_G == 0.04878856857665119
    assert NS.sigma_rjit == 6.8738635424337596e-05
    assert NS.ber_q == 4.7534243088228987
    assert NS.thru_peak_interference_at_BER == 0.050000000000000003
    assert NS.sci_sigma == 0.010518732760127113
    assert NS.peak_interference_at_BER == 0.050000000000000003


def test_octave_rx_calibration_CL120e_is_undefined_upstream():
    """CTLE_type='CL120e' cannot run: MATLAB L1592 reads bare f_HP_P/f_HP_Z.

    COM Octave, same inputs as the CL120d run above:
        error: 'f_HP_P' undefined near line 16, column 45
        error: called from
            Create_Noise_PDF at line 16 column 13

    Six other sites in the reference write param.f_HP_P / param.f_HP_Z; this
    one does not, and neither bare name exists in the function's scope. The
    port used to substitute param.f_HP_P/param.f_HP_Z and return a number
    where the reference stops."""
    p, fom, cd, op = _rx_inputs('CL120e')
    with pytest.raises(ValueError, match='f_HP_P'):
        Create_Noise_PDF(0.5, p, fom, cd, op, 0.005)


def test_CL120e_only_blocked_under_rx_calibration():
    """The defective line is inside `if OP.RX_CALIBRATION`, so CTLE_type
    'CL120e' is harmless with the flag clear -- the reference never reaches
    it either. Guards the raise above from over-reaching."""
    p, fom, cd, op = _rx_inputs('CL120e')
    op.RX_CALIBRATION = 0
    PDF, CDF, NS = Create_Noise_PDF(0.5, p, fom, cd, op, 0.005)
    assert len(CDF) == len(PDF.y)
