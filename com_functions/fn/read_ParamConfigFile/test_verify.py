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
from com_functions.fn.read_ParamConfigFile.py_impl import _parse_matlab_matrix

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
M,32
R_0,50
RESULT_DIR,results
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
M,32
RESULT_DIR,results
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
    """Without explicit CTLE poles/zeros, the reference's defaults are 1e9
    times fb/4, fb and fb/4 -- see the block comment further down.

    COM Octave on this exact config:
        param.CTLE_fp1 = 1.328125e+19
        param.CTLE_fp2 = 5.3125e+19
        param.CTLE_fz  = 1.328125e+19
    """
    param, _ = read_ParamConfigFile(minimal_csv_file, make_op())
    for name, want in (('CTLE_fp1', 1.328125e+19),
                       ('CTLE_fp2', 5.3125e+19),
                       ('CTLE_fz', 1.328125e+19)):
        got = float(np.ravel(np.asarray(getattr(param, name)))[0])
        assert abs(got - want) <= 1e-12 * want, (
            '%s is %.17g, COM Octave gives %.17g' % (name, got, want))


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
M,32
R_0,50
RESULT_DIR,results
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


# ============================================================
# COM Octave oracle values (2026-09-22)
# read_ParamConfigFile run verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m via tools/octave_oracle.py, with
# xls_parameter, read_package_parameters, xls_parameter_txffe and csvread4com
# as its subfunctions, on the example workbook
# examples/akinwale_CR_22dB_VendorX/config_com_dj_..._Case1.xlsx exported to
# CSV, and then on that config broken in one way at a time.
#
# The whole-config run now agrees on 210 of 210 param fields and 100 of 100 OP
# fields. Before this pass it differed in these ways:
#   1. param.C_0 and param.C_1 were never set. They were the only two fields
#      the reference had and the port did not, and their absence was what made
#      parameter_size_adjustment's make_length2 loop skip a missing field
#      rather than error (see the note at sicopr.py L14234);
#   2. M, R_0 and RESULT_DIR are MANDATORY in the reference and the port gave
#      them defaults. ML 10235 reads `xls_parameter(parameter,'M',32)` and ML
#      10273 `xls_parameter(parameter,'R_0',50)`: the third argument of
#      xls_parameter is eval_if_string, not default_value, so those numbers
#      are truthy eval flags and neither keyword has a default. R_0's flag
#      also means a string value is evaluated, which the port did not do;
#   3. a blank value cell came back as '' instead of NaN, so the first
#      arithmetic on it raised TypeError where the reference carries NaN;
#   4. param.c, param.alen and param.az were Python lists, not vectors.
#
# NOT reproduced, reported instead: a non-numeric value in a numeric keyword.
# COM Octave, f_b = abc: param.fb = [9.7e10 9.8e10 9.9e10], because MATLAB
# multiplies the char's ASCII codes. Matching that would mean emulating
# implicit char->double arithmetic through the whole parser for a value that
# is meaningless either way, so the port still raises there.
# ============================================================


def _write_csv(tmp_path, text, name='cfg.csv'):
    p = tmp_path / name
    p.write_text(text, encoding='utf-8')
    return str(p)


def test_octave_C_0_and_C_1_default_to_zero(minimal_csv_file):
    """COM Octave: param.C_0 and param.C_1 exist on every read and are 0 when
    the keywords are absent. The port set neither."""
    param, _ = read_ParamConfigFile(minimal_csv_file, make_op())
    assert float(np.atleast_1d(param.C_0)[0]) == 0.0
    assert float(np.atleast_1d(param.C_1)[0]) == 0.0


def test_octave_C_0_and_C_1_scale_by_1e9(tmp_path):
    """COM Octave: C_0 and C_1 are given in nF, so 0.3 -> 3e-10."""
    path = _write_csv(tmp_path, MINIMAL_CSV + 'C_0,0.3\nC_1,0.25\n')
    param, _ = read_ParamConfigFile(path, make_op())
    assert float(np.atleast_1d(param.C_0)[0]) == pytest.approx(0.3e-9)
    assert float(np.atleast_1d(param.C_1)[0]) == pytest.approx(0.25e-9)


@pytest.mark.parametrize('keyword', ['M,32\n', 'R_0,50\n', 'RESULT_DIR,results\n'])
def test_octave_mandatory_keywords(tmp_path, keyword):
    """COM Octave: dropping any of M, R_0 or RESULT_DIR stops the reference
    (it reaches its own missing-parameter path, which calls the undefined
    missingParameter). The port supplied 32, 50 and '' instead."""
    path = _write_csv(tmp_path, MINIMAL_CSV.replace(keyword, ''))
    with pytest.raises(KeyError):
        read_ParamConfigFile(path, make_op())


def test_mandatory_keyword_survives_another_module_sentinel(tmp_path, monkeypatch):
    """Finding F05 (2026-10-03). In the assembled sicopr.py, xls_parameter's own
    `_SENTINEL = object()` came after this module's and rebound the name, so
    `default_value is _SENTINEL` was never true and a missing M came back as a
    bare object, failing later with an unrelated TypeError. Rebinding the old
    module-level name here does what assembly did."""
    import com_functions.fn.read_ParamConfigFile.py_impl as impl
    monkeypatch.setattr(impl, '_SENTINEL', object(), raising=False)
    path = _write_csv(tmp_path, MINIMAL_CSV.replace('M,32\n', ''))
    with pytest.raises(KeyError, match='Mandatory parameter "M"'):
        read_ParamConfigFile(path, make_op())


def test_octave_R_0_string_is_evaluated(tmp_path):
    """COM Octave: R_0 = '[50 50]' gives param.Z0 = [50 50], because
    xls_parameter's eval_if_string argument is the truthy 50."""
    path = _write_csv(tmp_path, MINIMAL_CSV.replace('R_0,50\n', 'R_0,[50 50]\n'))
    param, _ = read_ParamConfigFile(path, make_op())
    np.testing.assert_array_equal(np.atleast_1d(param.Z0), [50.0, 50.0])


def test_octave_blank_value_cell_is_nan(tmp_path):
    """COM Octave: f_b with an empty value cell gives param.fb = NaN, not an
    error. xlsread's raw output is NaN for a blank, which is also why ML 9660
    can ask `if isnan(param.PKG_NAME)`."""
    path = _write_csv(tmp_path, MINIMAL_CSV.replace('f_b,53.125\n', 'f_b,\n'))
    param, _ = read_ParamConfigFile(path, make_op())
    assert np.isnan(float(np.atleast_1d(param.fb)[0]))


def test_octave_placeholder_vectors_are_arrays(minimal_csv_file):
    """ML 10161-10163: param.c, param.alen and param.az are double row
    vectors. A Python list multiplies by repeating instead of scaling."""
    param, _ = read_ParamConfigFile(minimal_csv_file, make_op())
    for name, want in (('c', [0.4e-12, 0.4e-12]), ('alen', [20.0, 30.0, 550.0]),
                       ('az', [100.0, 120.0, 100.0])):
        got = getattr(param, name)
        assert isinstance(got, np.ndarray), '%s is %r' % (name, type(got))
        np.testing.assert_array_equal(got, want)


def test_octave_duplicate_keyword_is_an_error(tmp_path):
    """COM Octave: 2 occurrences of "f_b" found. Please recheck spreadsheet."""
    path = _write_csv(tmp_path, MINIMAL_CSV + 'f_b,99\n')
    with pytest.raises(ValueError):
        read_ParamConfigFile(path, make_op())


def test_octave_start_without_end_is_an_error(tmp_path):
    """COM Octave: Number of .START and .END must be the same."""
    path = _write_csv(tmp_path, MINIMAL_CSV + '.START,BLK\nC_p,0.1\n')
    with pytest.raises(ValueError):
        read_ParamConfigFile(path, make_op())


def test_octave_pkg_name_without_blocks_is_an_error(tmp_path):
    """COM Octave: PKG_NAME can only be used if .START blocks for package
    parameters are used."""
    path = _write_csv(tmp_path, MINIMAL_CSV + 'PKG_NAME,NOSUCH\n')
    with pytest.raises(ValueError):
        read_ParamConfigFile(path, make_op())


def test_octave_pkg_name_unknown_block_is_an_error(tmp_path):
    """COM Octave: Package Block "NOSUCH" not found."""
    path = _write_csv(tmp_path, MINIMAL_CSV + 'PKG_NAME,NOSUCH\n'
                      '.START,TXPKG\nC_p,0.1\n.END,TXPKG\n')
    with pytest.raises(ValueError):
        read_ParamConfigFile(path, make_op())


# ---------------------------------------------------------------------------
# CTLE_type selection, against COM Octave.
#
# 'CL120d' and 'CL120e' sat in test_option_coverage.py's KNOWN_UNCOVERED: no
# test wrote a config that selects either. ML 10263-10264 is
#     if ~isempty(param.g_DC_HP_values) ; param.CTLE_type='CL120d'; end
#     if ~isempty(param.f_HP_Z)         ; param.CTLE_type='CL120e'; end
# so the two overrides are ORDERED -- f_HP_Z wins over g_DC_HP, and both win
# over the configured CTLE_type. All three cases are checked below.
# ---------------------------------------------------------------------------

_CL120D_CSV = MINIMAL_CSV + 'g_DC_HP,-1\nf_HP_PZ,0.6625\n'
_CL120E_CSV = MINIMAL_CSV + 'f_HP_Z,0.6625\nf_HP_P,1.5\n'
_BOTH_CSV = MINIMAL_CSV + 'g_DC_HP,-1\nf_HP_PZ,0.6625\nf_HP_Z,0.6625\nf_HP_P,1.5\n'


def _param_from(csv_text):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv',
                                     delete=False) as f:
        f.write(csv_text)
        name = f.name
    try:
        param, _ = read_ParamConfigFile(name, make_op())
        return param
    finally:
        os.unlink(name)


def test_ctle_type_CL120d_selected_by_g_DC_HP():
    """A g_DC_HP entry overrides the default CL93."""
    param = _param_from(_CL120D_CSV)
    assert param.CTLE_type == 'CL120d', param.CTLE_type
    assert np.size(param.g_DC_HP_values) > 0
    assert np.size(param.f_HP) > 0


def test_ctle_type_CL120e_selected_by_f_HP_Z():
    """An f_HP_Z entry overrides the default CL93."""
    param = _param_from(_CL120E_CSV)
    assert param.CTLE_type == 'CL120e', param.CTLE_type


def test_ctle_type_CL120e_wins_over_CL120d():
    """ML 10263-10264 apply in order, so f_HP_Z overrides g_DC_HP.

    Without this the two `if`s could be written as an if/elif and CL120d would
    win, which no single-option test would notice.
    """
    param = _param_from(_BOTH_CSV)
    assert param.CTLE_type == 'CL120e', param.CTLE_type


def test_ctle_type_CL120e_divides_fz_by_the_dc_gain():
    """ML 10281: in CL120e the zero has already been adjusted for gain, so
    CTLE_fz is divided by 10^(g_DC/20). g_DC is -1 in this config."""
    param = _param_from(_CL120E_CSV)
    fz = np.atleast_1d(np.asarray(param.CTLE_fz, dtype=float))
    # The base CTLE_fz here is the REFERENCE's default, 1e9*(fb/4), not fb/4;
    # this test is about the /10^(g_DC/20) division, not the default itself.
    base = 1e9 * (53.125e9 / 4)
    assert np.allclose(fz, base / (10.0 ** (-1.0 / 20.0)), rtol=1e-12)


# ---------------------------------------------------------------------------
# The CTLE pole and zero defaults, RULED 2026-09-23: match the reference.
#
# ML 10268-10270 multiply by 1e9 to turn a GHz config entry into Hz, but the
# DEFAULT is param.fb/4, and param.fb is already in Hz (ML 10175). So an
# omitted key gives 1e9 * 1.328e10 = 1.328e19 Hz, and the trailing comments on
# those lines read "fp1 is in GHz". It looks like the defaults were meant to be
# param.fb/4/1e9.
#
# The port used to write it that way and get the physically sensible value.
# That is a silent correction of an upstream defect, which docs/VERIFICATION.md
# says not to do: an upstream defect should be visible, not quietly improved
# upon. The owner ruled to match the reference and report it, so
# test_ctle_defaults above now pins COM Octave's own numbers.
#
# Consequence worth knowing: a 1.328e19 Hz pole makes the CTLE transfer
# function identically 1, i.e. no filtering at all. Every shipped workbook
# supplies f_p1, f_p2 and f_z, so a normal run never reaches the defaults.
# Reported for the COM ad hoc as item A12.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Default emulated release (owner's call): 4p16p0 from 2026-09-27, 4p17p0 from
# 2026-10-03, from VERSION.json.
# VERSION.json said 4p16p0 from 2026-09-22 while this module's constant still
# said 4p15p0, so the generated header announced one release and a plain run
# emulated the other. These pin the two to each other and to what a config
# without a 'COM Version' keyword actually gets.
# ---------------------------------------------------------------------------
def test_default_version_is_the_one_VERSION_json_names():
    import json
    import com_functions.fn.read_ParamConfigFile.py_impl as impl
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    with open(os.path.join(root, 'VERSION.json'), encoding='utf-8') as fh:
        want = json.load(fh)['default_matlab_version']
    assert want == '4p17p0'          # the owner's call, 2026-10-03
    assert impl.COM_MATLAB_VERSION == want


def test_config_without_version_keyword_emulates_4p17p0(minimal_csv_file):
    param, _ = read_ParamConfigFile(minimal_csv_file, make_op())
    assert param.matlab_version == '4p17p0'


# ---------------------------------------------------------------------------
# Whole numbers written as '32.0' (2026-09-28). A workbook resaved by a tool
# that writes every number with a decimal point reached the engine with
# M = 32.0, and np.ones(M) in Apply_EQ raised; MATLAB, where every number is a
# double, runs the same file. CSV configs had the same exposure: float('32').
# The reader now hands the engine an int for a whole-number cell, which is
# what openpyxl returned for the workbooks the port was verified with.
# ---------------------------------------------------------------------------
def test_whole_number_cell_values_reach_the_engine_as_int(minimal_csv_file):
    param, _ = read_ParamConfigFile(minimal_csv_file, make_op())
    assert param.samples_per_ui == 32
    assert isinstance(param.samples_per_ui, int), type(param.samples_per_ui)


def test_parse_cell_keeps_fractions_and_non_finite_as_float():
    from com_functions.fn.read_ParamConfigFile.py_impl import _parse_cell
    assert type(_parse_cell(32.0)) is int and _parse_cell(32.0) == 32
    assert type(_parse_cell('32')) is int
    assert _parse_cell(0.4) == 0.4 and type(_parse_cell(0.4)) is float
    assert type(_parse_cell(1e300)) is float        # beyond int64: left alone
    assert np.isnan(_parse_cell(None))
    assert np.isinf(_parse_cell(float('inf')))
    assert _parse_cell(True) is True


# ---------------------------------------------------------------------------
# Count keywords are conditioned in _xls_param, whatever the source: with every
# numeric cell forced to float (the Drive artefact, and worse), the counts still
# reach the engine as int; a fractional count is an error naming the keyword.
# ---------------------------------------------------------------------------
def test_count_keywords_are_int_even_when_every_cell_is_float(minimal_csv_file, monkeypatch):
    import com_functions.fn.read_ParamConfigFile.py_impl as impl
    monkeypatch.setattr(impl, '_whole_to_int',
                        lambda v: float(v) if isinstance(v, int) and not isinstance(v, bool) else v)
    param, _ = read_ParamConfigFile(minimal_csv_file, make_op())
    for name in ('samples_per_ui', 'ndfe', 'levels', 'N_bmax', 'N_v'):
        v = getattr(param, name)
        assert isinstance(v, int), (name, type(v))


def test_fractional_count_keyword_is_a_named_error(tmp_path):
    p = tmp_path / 'frac.csv'
    p.write_text(MINIMAL_CSV.replace('M,32', 'M,32.5'))
    with pytest.raises(ValueError, match='"M" must be a whole number'):
        read_ParamConfigFile(str(p), make_op())


# ---------------------------------------------------------------------------
# 4p17p0 keywords (L11034, L11207 of com_ieee8023_4p17p0.m): T_dev, default 1,
# and ACBW, default 0, which turns on the apparent channel bandwidth in
# FD_Processing. Earlier releases have neither, so a workbook that sets ACBW
# must change nothing under them.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('ver,acbw,tdev', [('4p17p0', 1, 0.4), ('4p16p0', 0, None),
                                           ('4p15p0', 0, None)])
def test_acbw_keywords_follow_the_release(tmp_path, ver, acbw, tdev):
    path = _write_csv(tmp_path, MINIMAL_CSV + 'COM Version,%s\nACBW,1\nT_dev,0.4\n' % ver)
    param, OP = read_ParamConfigFile(path, make_op())
    assert OP.ACBW == acbw
    assert getattr(param, 'T_dev', None) == tdev


def test_acbw_defaults_on_4p17p0(tmp_path):
    path = _write_csv(tmp_path, MINIMAL_CSV + 'COM Version,4p17p0\n')
    param, OP = read_ParamConfigFile(path, make_op())
    assert OP.ACBW == 0 and param.T_dev == 1


def test_4p17p0_keeps_the_4p16p0_clip_default(tmp_path):
    """A 4p16p0 change carries into 4p17p0: the Clip Method default stays Slow."""
    path = _write_csv(tmp_path, MINIMAL_CSV + 'COM Version,4p17p0\n')
    param, _ = read_ParamConfigFile(path, make_op())
    assert param.clip_method == 'Slow'


def _same_load(a, b):
    """param and OP from two loads, field for field (arrays by value, NaN equal)."""
    for x, y in zip(a, b):
        dx, dy = vars(x), vars(y)
        assert sorted(dx) == sorted(dy)
        for k in dx:
            u, v = dx[k], dy[k]
            if isinstance(u, SimpleNamespace):
                _same_load((u,), (v,))
            elif isinstance(u, (np.ndarray, list, float)) or isinstance(v, (np.ndarray, list, float)):
                np.testing.assert_array_equal(np.asarray(u, dtype=object if isinstance(u, list) else None),
                                              np.asarray(v, dtype=object if isinstance(v, list) else None),
                                              err_msg=k)
            else:
                assert u == v, k


def test_mat_config_loads_like_the_same_grid_as_csv(tmp_path):
    """Finding F04 (2026-10-03). The reference loads a .mat config holding the
    `parameter` cell array (4p17p0 L10620, `load(matcongfile)`) and reads it with
    the same code as a workbook. scipy hands each cell back as a small array
    (1x1 double, 1-element char), which reached the arithmetic as is and failed:
    "unsupported operand type(s) for *: 'object' and 'float'". The grid here is
    MINIMAL_CSV's, written as tools/xlsx_to_com_mat.py writes a .mat for Octave:
    numbers as doubles, text as char, a blank as NaN."""
    import csv as _csv
    import scipy.io
    csv_path = _write_csv(tmp_path, MINIMAL_CSV)
    rows = [r for r in _csv.reader(MINIMAL_CSV.splitlines()) if r]
    ncol = max(len(r) for r in rows)
    grid = np.empty((len(rows), ncol), dtype=object)
    for i, r in enumerate(rows):
        for j in range(ncol):
            c = r[j] if j < len(r) else ''
            try:
                grid[i, j] = float(c) if c.strip() else float('nan')
            except ValueError:
                grid[i, j] = c
    mat_path = str(tmp_path / 'cfg.mat')
    scipy.io.savemat(mat_path, {'parameter': grid}, format='5', oned_as='row')
    _same_load(read_ParamConfigFile(mat_path, make_op()),
               read_ParamConfigFile(csv_path, make_op()))


# ============================================================
# COM Octave (Octave 11.3.0, 2026-10-06): eval of each config string, as the
# reference's xls_parameter does with eval_if_string. MATLAB's colon is
# start + k*step, counted with a tolerant floor, and the last element is set to
# the limit only when that sum overshoots it. Findings F06 (a range built by
# linspace over a rounded count, so [0:0.3:1] reached 1) and F12 (no clamp, so
# [0:0.1:0.3] ended at 0.30000000000000004).
# Octave is the proxy here: MATLAB forms the second half of a range from its
# end, so a few elements can differ from MATLAB by one ulp. Not pinned.
# ============================================================
COLON_PINS = [
    ('[0:0.3:1]', [0.0, 0.3, 0.6, 0.8999999999999999]),
    ('[-0.2:0.05:0.05]', [-0.2, -0.15000000000000002, -0.1, -0.04999999999999999, 0.0,
                          0.04999999999999999]),
    ('[0:0.1:0.3]', [0.0, 0.1, 0.2, 0.3]),
    ('[-0.3:0.1:0]', [-0.3, -0.19999999999999998, -0.09999999999999998, 0.0]),
    ('[1:0]', []),
    ('[0.14:-.02:0]', [0.14, 0.12000000000000001, 0.1, 0.08000000000000002, 0.06000000000000001,
                       0.04000000000000001, 0.020000000000000018, 0.0]),
    ('[0.1e-4:0.5e-5:0.5e-4]', [1e-05, 1.5000000000000002e-05, 2e-05, 2.5000000000000005e-05,
                                3.0000000000000004e-05, 3.5000000000000004e-05, 4e-05, 4.5e-05,
                                5e-05]),
    ('[0:0.25:1 ; 1:0.25:2]', [[0.0, 0.25, 0.5, 0.75, 1.0], [1.0, 1.25, 1.5, 1.75, 2.0]]),
    ('[ -0.34:.02:0]', [-0.34, -0.32, -0.30000000000000004, -0.28, -0.26, -0.24000000000000002,
                        -0.22000000000000003, -0.2, -0.18000000000000002, -0.16000000000000003,
                        -0.14, -0.12000000000000002, -0.10000000000000003, -0.08000000000000002,
                        -0.06, -0.040000000000000036, -0.020000000000000018, 0.0]),
    ('[-12:4:0]', [-12.0, -8.0, -4.0, 0.0]),
    ('[0:0.1:1]', [0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001,
                   0.7000000000000001, 0.8, 0.9, 1.0]),
    ('[1:3]', [1.0, 2.0, 3.0]),
]


@pytest.mark.parametrize('text,want', COLON_PINS, ids=[t for t, _ in COLON_PINS])
def test_octave_colon_ranges(text, want):
    got = np.asarray(_parse_matlab_matrix(text), dtype=float)
    want = np.asarray(want, dtype=float)
    if want.size == 0:
        assert got.size == 0
        return
    assert got.shape == want.shape
    assert got.tolist() == want.tolist()        # bit for bit
