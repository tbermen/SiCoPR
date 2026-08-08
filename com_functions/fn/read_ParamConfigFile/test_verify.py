# ============================================================
# MATLAB GROUND TRUTH
# read_ParamConfigFile reads a spreadsheet/CSV and populates param and OP structs.
# All expected values below are derived analytically from the MATLAB source
# (lines 9601-10258) — primarily parameter pass-through with unit conversion.
#
# Key conversions from config CSV values:
#   fb  = f_b * 1e9           (GHz → Hz)
#   max_start_freq = f_min * 1e9
#   max_freq_step  = Delta_f * 1e9
#   CTLE_fp1 = f_p1 * 1e9  (default = fb/4)
#   CTLE_fp2 = f_p2 * 1e9  (default = fb)
#   CTLE_fz  = f_z  * 1e9  (default = fb/4)
#
# For f_b = 53.125 GHz → fb = 53.125e9 Hz
# CTLE_fp1 default = 53.125e9/4 = 13.28125e9
# CTLE_fp2 default = 53.125e9
# CTLE_fz  default = 53.125e9/4 = 13.28125e9
#
# bmax shape: ndfe=4 → np.array([0.7, 0.2, 0.2, 0.2])
# bmin shape: ndfe=4 → np.array([-0.7, -0.2, -0.2, -0.2])
# Floating DFE flags: N_bg=0 → Floating_DFE=False, Floating_RXFFE=False
# OP.RxFFE: ffe_pre_tap_len=0, ffe_post_tap_len=0 → False
# ============================================================

import os
import tempfile
import pytest
import numpy as np
from types import SimpleNamespace

from com_functions.fn.read_ParamConfigFile.py_impl import read_ParamConfigFile

# ---------------------------------------------------------------------------
# Fixture: build a temporary CSV with all mandatory parameters
# ---------------------------------------------------------------------------

MINIMAL_CSV = '''\
f_b,53.125
f_min,0.05
Delta_f,0.01
c(0),0.68
N_b,4
b_max(1),0.7
L,4
DER_0,1e-4
sigma_RJ,0.01
A_DD,0.05
eta_0,1.7e-4
R_LM,1
Include PCB,0
g_DC,-1
SNR_TX,30
'''

FULL_CSV = '''\
f_b,53.125
f_min,0.05
Delta_f,0.01
c(0),0.68
N_b,4
b_max(1),0.7
b_max(2..N_b),0.15
b_min(1),-0.5
b_min(2..N_b),-0.15
L,4
DER_0,1e-4
sigma_RJ,0.01
A_DD,0.05
eta_0,1.7e-4
R_LM,1
Include PCB,0
g_DC,-1
SNR_TX,30
f_p1,7.0
f_p2,25.0
f_z,5.0
N_bg,2
N_bf,6
R_0,50
A_v,0.4
A_fe,0.4
A_ne,0.4
kappa1,1
kappa2,1
'''


@pytest.fixture
def minimal_csv_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(MINIMAL_CSV)
        name = f.name
    yield name
    os.unlink(name)


@pytest.fixture
def full_csv_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(FULL_CSV)
        name = f.name
    yield name
    os.unlink(name)


def make_op():
    return SimpleNamespace(GET_FD=False, TDMODE=0, CONFIG2MAT_ONLY=False, SAVE_CONFIG2MAT=False)


# ---------------------------------------------------------------------------
# Test 1 – Nominal: frequency unit conversions
# ---------------------------------------------------------------------------

def test_frequency_conversion(minimal_csv_file):
    """f_b in GHz → fb in Hz; f_min, Delta_f similarly converted."""
    param, OP = read_ParamConfigFile(minimal_csv_file, make_op())
    assert abs(param.fb - 53.125e9) < 1.0, f"fb={param.fb}"
    assert abs(param.max_start_freq - 0.05e9) < 1.0, f"max_start_freq={param.max_start_freq}"
    assert abs(param.max_freq_step - 0.01e9) < 1.0, f"max_freq_step={param.max_freq_step}"


# ---------------------------------------------------------------------------
# Test 2 – Nominal: CTLE defaults (no f_p1/f_p2/f_z in minimal CSV)
# ---------------------------------------------------------------------------

def test_ctle_defaults(minimal_csv_file):
    """Without explicit CTLE poles/zeros, defaults are fb/4, fb, fb/4."""
    param, _ = read_ParamConfigFile(minimal_csv_file, make_op())
    fb = 53.125e9
    assert abs(param.CTLE_fp1 - fb / 4) < 1e3, f"CTLE_fp1={param.CTLE_fp1}"
    assert abs(param.CTLE_fp2 - fb) < 1e3, f"CTLE_fp2={param.CTLE_fp2}"
    assert abs(param.CTLE_fz - fb / 4) < 1e3, f"CTLE_fz={param.CTLE_fz}"


# ---------------------------------------------------------------------------
# Test 3 – Nominal: DFE bmax/bmin shape and values
# ---------------------------------------------------------------------------

def test_dfe_bmax_bmin_shape(minimal_csv_file):
    """bmax[0]=0.7 and bmax[1:]=0.2 (default); bmin mirrors negatives."""
    param, _ = read_ParamConfigFile(minimal_csv_file, make_op())
    assert param.ndfe == 4
    assert len(param.bmax) == 4
    assert abs(param.bmax[0] - 0.7) < 1e-9
    assert all(abs(param.bmax[1:] - 0.2) < 1e-9)
    assert abs(param.bmin[0] - (-0.7)) < 1e-9


# ---------------------------------------------------------------------------
# Test 4 – Full CSV: explicit CTLE poles override defaults
# ---------------------------------------------------------------------------

def test_ctle_explicit_values(full_csv_file):
    """Explicit f_p1=7.0 GHz, f_p2=25.0 GHz, f_z=5.0 GHz → converted to Hz."""
    param, _ = read_ParamConfigFile(full_csv_file, make_op())
    assert abs(param.CTLE_fp1 - 7.0e9) < 1e3, f"CTLE_fp1={param.CTLE_fp1}"
    assert abs(param.CTLE_fp2 - 25.0e9) < 1e3, f"CTLE_fp2={param.CTLE_fp2}"
    assert abs(param.CTLE_fz - 5.0e9) < 1e3, f"CTLE_fz={param.CTLE_fz}"


# ---------------------------------------------------------------------------
# Test 5 – Full CSV: explicit bmax/bmin override defaults
# ---------------------------------------------------------------------------

def test_dfe_explicit_limits(full_csv_file):
    """Explicit b_max(2..N_b)=0.15, b_min(1)=-0.5, b_min(2..N_b)=-0.15."""
    param, _ = read_ParamConfigFile(full_csv_file, make_op())
    assert abs(param.bmax[0] - 0.7) < 1e-9
    assert all(abs(param.bmax[1:] - 0.15) < 1e-9)
    assert abs(param.bmin[0] - (-0.5)) < 1e-9
    assert all(abs(param.bmin[1:] - (-0.15)) < 1e-9)


# ---------------------------------------------------------------------------
# Test 6 – Edge: N_b=0 → empty bmax/bmin
# ---------------------------------------------------------------------------

def test_ndfe_zero():
    """N_b=0 → bmax and bmin are empty arrays."""
    csv_content = MINIMAL_CSV.replace('N_b,4', 'N_b,0')
    csv_content = csv_content.replace('b_max(1),0.7', 'b_max(1),0.7')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        fname = f.name
    try:
        # When ndfe=0, b_max(1) is not needed (bmax = empty array)
        # Build a CSV without b_max(1) for ndfe=0
        no_bmax_csv = '\n'.join(
            line for line in MINIMAL_CSV.splitlines()
            if not line.startswith('N_b,') and not line.startswith('b_max(1),')
        ) + '\nN_b,0\n'
        with open(fname, 'w') as f2:
            f2.write(no_bmax_csv)
        param, _ = read_ParamConfigFile(fname, make_op())
        assert param.ndfe == 0
        assert len(param.bmax) == 0
        assert len(param.bmin) == 0
    finally:
        os.unlink(fname)


# ---------------------------------------------------------------------------
# Test 7 – Floating DFE flag set when N_bg > 0
# ---------------------------------------------------------------------------

def test_floating_dfe_flag(full_csv_file):
    """N_bg=2 in full CSV → Floating_DFE=True (no RxFFE)."""
    param, OP = read_ParamConfigFile(full_csv_file, make_op())
    assert param.N_bg == 2
    assert param.Floating_DFE is True
    assert param.Floating_RXFFE is False
    assert OP.RxFFE is False


# ---------------------------------------------------------------------------
# Test 8 – Boundary: OP.TDMODE=1 disables FD fields
# ---------------------------------------------------------------------------

def test_tdmode_disables_fd(minimal_csv_file):
    """When TDMODE=1 in OP (pre-set), FD-related flags are cleared."""
    # TDMODE is read from config; pass it via a csv override
    tdmode_csv = MINIMAL_CSV + 'TDMODE,1\n'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(tdmode_csv)
        fname = f.name
    try:
        op = make_op()
        _, OP = read_ParamConfigFile(fname, op)
        assert OP.TDMODE
        assert OP.GET_FD is False
        assert OP.ERL == 0
        assert OP.TDR == 0
    finally:
        os.unlink(fname)


# ---------------------------------------------------------------------------
# Test 9 – Default OP flags are set correctly for minimal CSV
# ---------------------------------------------------------------------------

def test_op_default_flags(minimal_csv_file):
    """Default OP flags match MATLAB defaults when not present in CSV."""
    _, OP = read_ParamConfigFile(minimal_csv_file, make_op())
    assert OP.INC_PACKAGE is True or OP.INC_PACKAGE == 1
    assert OP.Butterworth == 1
    assert OP.CDR == 'MM'
    assert OP.FFE_OPT_METHOD == 'MMSE'
    assert OP.PMD_type == 'C2C'


# ---------------------------------------------------------------------------
# Test 10 – START/END block: package parameters parsed into param.PKG
# ---------------------------------------------------------------------------

def test_start_end_block_parsing():
    """PKG_NAME + .START/.END blocks populate param.PKG correctly."""
    pkg_csv = '''\
f_b,53.125
f_min,0.05
Delta_f,0.01
c(0),0.68
N_b,4
b_max(1),0.7
L,4
DER_0,1e-4
sigma_RJ,0.01
A_DD,0.05
eta_0,1.7e-4
R_LM,1
Include PCB,0
g_DC,-1
SNR_TX,30
PKG_NAME,TXPKG RXPKG
.START,TXPKG
C_p,0.1
R_d,50
A_v,0.4
.END,TXPKG
.START,RXPKG
C_p,0.2
R_d,60
A_v,0.45
.END,RXPKG
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(pkg_csv)
        fname = f.name
    try:
        param, _ = read_ParamConfigFile(fname, make_op())
        assert hasattr(param, 'PKG')
        assert hasattr(param.PKG, 'TXPKG')
        assert hasattr(param.PKG, 'RXPKG')
        # After PKG_NAME substitution, C_pkg_board[0]=TXPKG value, [1]=RXPKG value
        assert abs(float(np.atleast_1d(param.C_pkg_board)[0]) - 0.1e-9) < 1e-20
        assert abs(float(np.atleast_1d(param.C_pkg_board)[1]) - 0.2e-9) < 1e-20
    finally:
        os.unlink(fname)
