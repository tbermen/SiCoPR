"""
Stage 5: End-to-end COM result comparison.

Requires:
    tests/fixtures/ieee8023ck_reference.xlsx
    tests/fixtures/ieee8023ck_compliant_host_channel.s4p

EXPECTED_COM_DB must be set to the value MATLAB produces for the exact reference input.
The test is skipped if fixture files are absent.

802.3ck standard: compliant host channel COM must be ≥ 3.0 dB.
"""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

REFERENCE_CONFIG = os.path.join(os.path.dirname(__file__), 'fixtures', 'ieee8023ck_reference.xlsx')
REFERENCE_S4P    = os.path.join(os.path.dirname(__file__), 'fixtures', 'ieee8023ck_compliant_host_channel.s4p')

# Set to the value MATLAB produces for this exact reference channel+config.
# Until a MATLAB run is available, this is set to the standard minimum pass threshold.
# Replace with the actual MATLAB output once known.
EXPECTED_COM_DB  = None   # Set to float once MATLAB reference value is known
COM_TOLERANCE_DB = 0.5    # ±0.5 dB

pytestmark = pytest.mark.skipif(
    not (os.path.exists(REFERENCE_CONFIG) and os.path.exists(REFERENCE_S4P)),
    reason='Reference fixture files not present — skipping end-to-end test'
)


def test_com_result_is_finite():
    """
    Full end-to-end COM must produce a finite result.
    This test passes even without knowing the exact expected value.
    """
    import com
    results = com.com_ieee8023(REFERENCE_CONFIG, 0, 0, REFERENCE_S4P)
    assert results is not None and len(results) > 0, "com_ieee8023 returned empty results"
    r = results[0]
    assert r is not None, "com_ieee8023 returned None for case 1"
    com_val = float(getattr(r, 'COM', np.nan))
    assert np.isfinite(com_val), f"COM result is not finite: {com_val}"
    print(f"\nCOM = {com_val:.3f} dB")


@pytest.mark.skipif(EXPECTED_COM_DB is None, reason='EXPECTED_COM_DB not set — run MATLAB first')
def test_com_result_matches_matlab():
    """
    Full end-to-end COM must match the MATLAB reference within ±0.5 dB.
    Set EXPECTED_COM_DB to the value MATLAB produces for this exact input.
    """
    import com
    results = com.com_ieee8023(REFERENCE_CONFIG, 0, 0, REFERENCE_S4P)
    r = results[0]
    com_val = float(getattr(r, 'COM', np.nan))
    assert np.isfinite(com_val), f"COM result is not finite: {com_val}"
    assert abs(com_val - EXPECTED_COM_DB) <= COM_TOLERANCE_DB, (
        f"COM = {com_val:.3f} dB, expected {EXPECTED_COM_DB:.3f} ± {COM_TOLERANCE_DB} dB. "
        f"Difference: {com_val - EXPECTED_COM_DB:.3f} dB. "
        f"Check Stage 4 checkpoints to find where divergence begins."
    )


def test_com_standard_compliance():
    """
    For a compliant 802.3ck host channel, COM must be ≥ 3.0 dB per the standard.
    This test validates the pass/fail decision, independent of exact MATLAB match.
    """
    import com
    results = com.com_ieee8023(REFERENCE_CONFIG, 0, 0, REFERENCE_S4P)
    r = results[0]
    com_val = float(getattr(r, 'COM', np.nan))
    assert np.isfinite(com_val), f"COM result is not finite: {com_val}"
    assert com_val >= 3.0, (
        f"COM = {com_val:.3f} dB fails 802.3ck 3.0 dB threshold. "
        f"Channel labeled 'compliant' but COM < 3.0 dB."
    )
