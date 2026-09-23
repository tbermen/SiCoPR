"""
Session-scoped fixtures for integration tests.
These fixtures run the real COM pipeline on the reference channel.
All fixtures skip if the reference fixture files are not present.
"""
# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# The 802.3ck reference pair is not redistributable, so it is not committed and
# these fixtures skip by default. Point COM_TEST_FIXTURES at a directory holding
# both files to run them. See tests/fixtures/README.md.
FIXTURE_DIR = os.environ.get(
    'COM_TEST_FIXTURES', os.path.join(os.path.dirname(__file__), 'fixtures'))

REFERENCE_CONFIG = os.path.join(FIXTURE_DIR, 'ieee8023ck_reference.xlsx')
REFERENCE_S4P    = os.path.join(FIXTURE_DIR, 'ieee8023ck_compliant_host_channel.s4p')

_FIXTURES_PRESENT = os.path.exists(REFERENCE_CONFIG) and os.path.exists(REFERENCE_S4P)


@pytest.fixture(scope='session')
def reference_chdata():
    """Run sicopr.py up through get_s4p_files on the reference channel."""
    if not _FIXTURES_PRESENT:
        pytest.skip('Reference fixture files not present')
    import sicopr
    from types import SimpleNamespace
    OP = SimpleNamespace()
    OP.TDMODE = False
    OP.DEBUG = False
    OP.DISPLAY_WINDOW = False
    OP.GET_FD = False
    OP.RxFFE = False
    OP.RxFFE_with_MMSE = False
    OP.FFE_OPT_METHOD = 'FOM'
    OP.RX_CALIBRATION = False
    OP.PSDRXCAL = False
    OP.ERL_ONLY = False
    OP.WC_PORTZ = False
    OP.SNDR_REF = False
    OP.RESULT_DIR = ''
    param, OP = sicopr.read_ParamConfigFile(REFERENCE_CONFIG, OP)
    chdata, param = sicopr.get_s4p_files(param, OP, 0, 0, [REFERENCE_S4P])
    return chdata, param, OP


@pytest.fixture(scope='session')
def reference_chdata_processed(reference_chdata):
    """Run process_sxp on the reference channel."""
    import sicopr
    chdata, param, OP = reference_chdata
    chdata_xt = []
    chdata_out, param_out = sicopr.process_sxp(param, OP, chdata, chdata_xt)
    return chdata_out, param_out, OP


@pytest.fixture(scope='session')
def reference_chdata_td(reference_chdata_processed):
    """Run COM_FD_to_TD on the reference channel."""
    import sicopr
    chdata, param, OP = reference_chdata_processed
    # Inject the REAL dependencies, exactly as sicopr's own run path does at
    # its _wired_COM_FD_to_TD. Called bare, COM_FD_to_TD falls back to the
    # module-level stubs it carries for unit testing -- including a "minimal
    # s21->impulse via ifft" that stands in for the whole of
    # s21_to_impulse_DC. Every checkpoint below, and reference_result which
    # chains off this fixture, was therefore asserting on a stubbed pipeline
    # while reading as though it validated the engine.
    chdata_out = sicopr.COM_FD_to_TD(
        chdata, param, OP,
        _s21_to_impulse_DC_fn=sicopr.s21_to_impulse_DC,
        _Bessel_Thomson_Filter_fn=sicopr.Bessel_Thomson_Filter,
        _Butterworth_Filter_fn=sicopr.Butterworth_Filter,
        _get_cm_noise_fn=sicopr.get_cm_noise)
    return chdata_out, param, OP


@pytest.fixture(scope='session')
def reference_result(reference_chdata_td):
    """Run optimize_fom on the reference channel and return the best result."""
    import sicopr
    chdata, param, OP = reference_chdata_td
    sigma_bn = 0.0
    result = sicopr.optimize_fom(OP, param, chdata, sigma_bn, do_C2M=0)
    return result
