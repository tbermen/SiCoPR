"""Verification tests for get_cm_noise().

# ============================================================
# MATLAB GROUND TRUTH
# For ki in 1..M: tps = PR(ki:M:end)  [1-based → Python ki-1::M]
# PR_fom = -testpdf.x(find(cdf_test>=BER,1,'first'))  when CM_norm_test=0
# results.CMn = best PR_fom over all ki
# results.CMn_p2p = max(PR) - min(PR)
#
# M=1: single sub-phase; tps=PR; testpdf built from PR; CMn computed once
# CM_norm_test=1: PR_fom = norm(tps); results.CMn = max over ki of norm(tps_ki)
# ============================================================
"""
import numpy as np
import pytest
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.get_cm_noise.py_impl import get_cm_noise


def test_m1_norm_test():
    """M=1, CM_norm_test=1: CMn = norm(PR)."""
    PR = np.array([0.1, -0.2, 0.3, -0.1, 0.05])
    op = SimpleNamespace(CM_norm_test=1)
    results = get_cm_noise(1, PR, 4, 1e-6, op)
    assert results.CMn == pytest.approx(float(np.linalg.norm(PR)), rel=1e-10)


def test_cmnp2p():
    """CMn_p2p = max(PR) - min(PR)."""
    PR = np.array([0.1, -0.3, 0.2])
    op = SimpleNamespace(CM_norm_test=1)
    results = get_cm_noise(1, PR, 4, 1e-6, op)
    assert results.CMn_p2p == pytest.approx(0.5)


def test_default_op():
    """OP=None: defaults to CM_norm_test=0; results has CMn and CMn_p2p."""
    PR = np.array([0.5, -0.5, 0.3, -0.3, 0.1, -0.1])
    results = get_cm_noise(1, PR, 4, 1e-6)
    assert hasattr(results, 'CMn')
    assert hasattr(results, 'CMn_p2p')
    assert np.isfinite(results.CMn)


def test_op_missing_cm_norm_test():
    """OP without CM_norm_test field → defaults to 0."""
    PR = np.array([0.4, -0.4, 0.2, -0.2])
    op = SimpleNamespace()
    results = get_cm_noise(1, PR, 4, 1e-6, op)
    assert hasattr(results, 'CMn')


def test_multi_phase_picks_best():
    """M=2: selects the sub-phase with higher norm as best."""
    PR = np.array([0.1, 0.9, 0.1, 0.9])   # odd indices have large values
    op = SimpleNamespace(CM_norm_test=1)
    results = get_cm_noise(2, PR, 4, 1e-6, op)
    # sub-phase 0: PR[0::2]=[0.1,0.1]; sub-phase 1: PR[1::2]=[0.9,0.9]
    best_expected = float(np.linalg.norm([0.9, 0.9]))
    assert results.CMn == pytest.approx(best_expected, rel=1e-10)


# ============================================================
# COM Octave oracle values (tools/octave_oracle.py, get_cm_noise from the
# 4p16p0 compat file; get_pdf_from_sampled_signal / d_cpdf / Init_PDF_Fast /
# conv_fct taken from matlab/com_ieee8023_4p16p0.m, because the compat copy of
# get_pdf_from_sampled_signal is a speed rewrite).  Generated 2026-09-22.
# ============================================================

# PR used by the oracle probes below.
_PR_ORACLE = np.array([
    0.020409, -0.025557, 0.004181, -0.005678, -0.004526, -0.002156,
    -0.0202, -0.002319, -0.008652, 0.03323, 0.002258, -0.003526,
    -0.002813, -0.00668, -0.010552, -0.003908, 0.004819, -0.002386,
    0.009578, -0.001998])


def test_oracle_nominal():
    """COM Octave, get_cm_noise(4, PR, 4, 1e-5, OP) with OP.CM_norm_test=0:
        results.CMn     = 0.07010000000000001
        results.CMn_p2p = 0.058787000000000006
    """
    r = get_cm_noise(4, _PR_ORACLE, 4, 1e-5, SimpleNamespace(CM_norm_test=0))
    assert r.CMn == pytest.approx(0.07010000000000001, rel=1e-14)
    assert r.CMn_p2p == pytest.approx(0.058787000000000006, rel=1e-14)


def test_ber_above_cdf_leaves_best_at_minus_inf():
    """find(cdf_test>=BER,1,'first') can come back empty.

    MATLAB then has PRn_test=[], so `if PR_fom > PR_fom_best` is false and the
    running best is untouched.

    COM Octave, get_cm_noise(4, PR, 4, 2, OP), OP.CM_norm_test=0:
        results.CMn = -Inf
    np.argmax on an all-false mask returned 0, which made Python answer
    -testpdf.x(1) = 0.07010000000000001 instead.
    """
    r = get_cm_noise(4, _PR_ORACLE, 4, 2.0, SimpleNamespace(CM_norm_test=0))
    assert r.CMn == -np.inf


def test_empty_sub_phase_takes_the_delta_pdf_branch():
    """M larger than the sample count leaves late sub-phases empty.

    get_pdf_from_sampled_signal guards with `if max(abs(v)) > BinSize`; max([])
    is [] and `if []` is false, so MATLAB falls through to the delta PDF.

    COM Octave, get_cm_noise(5, PR(1:3), 4, 1e-5, OP), OP.CM_norm_test=0:
        results.CMn     = 0.025600000000000001
        results.CMn_p2p = 0.045966
    Python raised "zero-size array to reduction operation maximum".
    """
    r = get_cm_noise(5, _PR_ORACLE[:3], 4, 1e-5, SimpleNamespace(CM_norm_test=0))
    assert r.CMn == pytest.approx(0.025600000000000001, rel=1e-14)
    assert r.CMn_p2p == pytest.approx(0.045966, rel=1e-14)


def test_m_zero_is_refused():
    """`for ki=1:0` never creates `results`, so MATLAB errors at the caller's
    assignment.

    COM Octave, get_cm_noise(0, PR, 4, 1e-5, OP):
        error: value on right hand side of assignment is undefined
    Python returned an empty struct, so a caller reading results.CMn would have
    failed far from the cause.
    """
    with pytest.raises(ValueError):
        get_cm_noise(0, _PR_ORACLE, 4, 1e-5, SimpleNamespace(CM_norm_test=0))
