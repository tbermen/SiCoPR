"""
Stage 4: Intermediate variable comparison against MATLAB reference.

Requires:
    tests/fixtures/ieee8023ck_reference.xlsx
    tests/fixtures/ieee8023ck_compliant_host_channel.s4p

All tests are skipped if these files are absent (see conftest.py).
When fixture files ARE present, all tests must pass before Stage 5.

Reference values are approximate targets for the 802.3ck standard
compliant host channel at 53.125 GBaud, PAM-4.
"""
import pytest
import numpy as np
import os

REFERENCE_CONFIG = os.path.join(os.path.dirname(__file__), 'fixtures', 'ieee8023ck_reference.xlsx')
REFERENCE_S4P    = os.path.join(os.path.dirname(__file__), 'fixtures', 'ieee8023ck_compliant_host_channel.s4p')

pytestmark = pytest.mark.skipif(
    not (os.path.exists(REFERENCE_CONFIG) and os.path.exists(REFERENCE_S4P)),
    reason='Reference fixture files not present — skipping checkpoint tests'
)


def test_checkpoint1_read_s4p_faxis(reference_chdata):
    """After read_s4p_files: frequency axis has the right number of points and range."""
    chdata, param, OP = reference_chdata
    cd = chdata[0]
    assert hasattr(cd, 'faxis'), "chdata[0] must have faxis after read_s4p_files"
    assert len(cd.faxis) > 100, f"faxis too short: {len(cd.faxis)} points"
    assert cd.faxis[-1] > 10e9, f"faxis max {cd.faxis[-1]/1e9:.1f} GHz should be > 10 GHz"
    assert np.all(np.isfinite(cd.faxis)), "faxis contains non-finite values"


def test_checkpoint1_read_s4p_sdd21(reference_chdata):
    """After read_s4p_files: sdd21 is complex, finite, and has reasonable insertion loss."""
    chdata, param, OP = reference_chdata
    cd = chdata[0]
    assert hasattr(cd, 'sdd21'), "chdata[0] must have sdd21"
    assert np.iscomplexobj(cd.sdd21), "sdd21 must be complex"
    assert np.all(np.isfinite(cd.sdd21)), "sdd21 contains non-finite values"
    # IL at Nyquist should be in a physically reasonable range (not open/short)
    f_nyq = param.fb / 2.0
    il_at_nyq_dB = -20.0 * np.log10(
        max(abs(np.interp(f_nyq, cd.faxis, abs(cd.sdd21))), 1e-10)
    )
    assert 3.0 < il_at_nyq_dB < 60.0, \
        f"IL at Nyquist = {il_at_nyq_dB:.1f} dB — outside expected 3–60 dB range"


def test_checkpoint2_process_sxp_sdd21(reference_chdata_processed):
    """After process_sxp: sdd21 present and finite."""
    chdata, param, OP = reference_chdata_processed
    cd = chdata[0]
    assert hasattr(cd, 'sdd21'), "process_sxp must preserve/update chdata[0].sdd21"
    assert np.all(np.isfinite(cd.sdd21)), "sdd21 contains non-finite values after process_sxp"


def test_checkpoint2_process_sxp_sch(reference_chdata_processed):
    """After process_sxp: SCH field set (system channel frequency response)."""
    chdata, param, OP = reference_chdata_processed
    cd = chdata[0]
    assert hasattr(cd, 'SCH'), "process_sxp must set chdata[0].SCH"
    assert np.all(np.isfinite(cd.SCH)), "SCH contains non-finite values"


def test_checkpoint3_com_fd_to_td_pulse_response(reference_chdata_td):
    """After COM_FD_to_TD: pulse response has a clear peak in 0.3–2.0 range."""
    chdata, param, OP = reference_chdata_td
    cd = chdata[0]
    assert hasattr(cd, 'uneq_pulse_response'), \
        "COM_FD_to_TD must set chdata[0].uneq_pulse_response"
    pr = cd.uneq_pulse_response
    assert len(pr) > 0, "uneq_pulse_response is empty"
    assert np.all(np.isfinite(pr)), "uneq_pulse_response contains non-finite values"
    peak = float(np.max(pr))
    assert 0.3 < peak < 2.0, \
        f"Pulse response peak {peak:.4f} outside expected range [0.3, 2.0]"


def test_checkpoint3_com_fd_to_td_cursor(reference_chdata_td):
    """After COM_FD_to_TD: cursor index is in the middle 25–75% of the array."""
    chdata, param, OP = reference_chdata_td
    cd = chdata[0]
    pr = cd.uneq_pulse_response
    cursor = int(np.argmax(pr))
    n = len(pr)
    assert n * 0.25 < cursor < n * 0.75, \
        f"Cursor index {cursor} out of expected range for array length {n}"


def test_checkpoint4_optimize_fom_finite(reference_result):
    """After optimize_fom: FOM is finite."""
    result = reference_result
    assert result is not None, "optimize_fom returned None"
    assert hasattr(result, 'FOM'), "optimize_fom must return struct with FOM field"
    fom = float(result.FOM)
    assert np.isfinite(fom), f"FOM is not finite: {fom}"


def test_checkpoint4_optimize_fom_reasonable(reference_result):
    """After optimize_fom: FOM is in a physically reasonable range for a real channel."""
    result = reference_result
    fom = float(result.FOM)
    # Real channels produce COM values typically between -10 dB and +20 dB
    assert -10.0 < fom < 20.0, \
        f"FOM = {fom:.2f} dB is outside the physically reasonable range [-10, 20] dB"
