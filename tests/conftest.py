"""
Session-scoped fixtures for integration tests.
These fixtures run the real COM pipeline on the reference channel.
All fixtures skip if the reference fixture files are not present.
"""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

REFERENCE_CONFIG = os.path.join(os.path.dirname(__file__), 'fixtures', 'ieee8023ck_reference.xlsx')
REFERENCE_S4P    = os.path.join(os.path.dirname(__file__), 'fixtures', 'ieee8023ck_compliant_host_channel.s4p')

_FIXTURES_PRESENT = os.path.exists(REFERENCE_CONFIG) and os.path.exists(REFERENCE_S4P)


@pytest.fixture(scope='session')
def reference_chdata():
    """Run com.py up through get_s4p_files on the reference channel."""
    if not _FIXTURES_PRESENT:
        pytest.skip('Reference fixture files not present')
    import com
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
    param, OP = com.read_ParamConfigFile(REFERENCE_CONFIG, OP)
    chdata, param = com.get_s4p_files(param, OP, 0, 0, [REFERENCE_S4P])
    return chdata, param, OP


@pytest.fixture(scope='session')
def reference_chdata_processed(reference_chdata):
    """Run process_sxp on the reference channel."""
    import com
    chdata, param, OP = reference_chdata
    chdata_xt = []
    chdata_out, param_out = com.process_sxp(param, OP, chdata, chdata_xt)
    return chdata_out, param_out, OP


@pytest.fixture(scope='session')
def reference_chdata_td(reference_chdata_processed):
    """Run COM_FD_to_TD on the reference channel."""
    import com
    chdata, param, OP = reference_chdata_processed
    chdata_out = com.COM_FD_to_TD(chdata, param, OP)
    return chdata_out, param, OP


@pytest.fixture(scope='session')
def reference_result(reference_chdata_td):
    """Run optimize_fom on the reference channel and return the best result."""
    import com
    chdata, param, OP = reference_chdata_td
    sigma_bn = 0.0
    result = com.optimize_fom(OP, param, chdata, sigma_bn, do_C2M=0)
    return result
