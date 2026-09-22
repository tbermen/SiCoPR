"""Verification tests for OptFom_Initialize_Loop_Struct().

# ============================================================
# MATLAB GROUND TRUTH (lines 3538-3579)
# Returns THIS with FOM=0 and all other fields as empty lists [].
# ============================================================
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.OptFom_Initialize_Loop_Struct.py_impl import OptFom_Initialize_Loop_Struct


def test_fom_is_zero():
    THIS = OptFom_Initialize_Loop_Struct()
    assert THIS.FOM == 0


def test_all_required_fields_present():
    THIS = OptFom_Initialize_Loop_Struct()
    for field in ('tx_index_vector', 'ctle_index', 'g_LP_index', 'itick',
                  'g_dc', 'g_DC_low', 'H_ctf', 'txffe', 'cursor_i',
                  'A_s', 'A_p', 'far_cursors', 'precursors',
                  'dfetaps', 'tail_RSS', 'floating_tap_coef', 'excess_dfe_cursors',
                  'C', 'MMSE_results', 'PSD_results', 'floating_tap_locations',
                  'sigma_N', 'sigma_TX', 'total_noise_rms', 'ISI_N', 'h_J', 'sigma_ne'):
        assert hasattr(THIS, field), f'missing field: {field}'


def test_non_fom_fields_are_empty():
    THIS = OptFom_Initialize_Loop_Struct()
    for field in ('tx_index_vector', 'txffe', 'dfetaps', 'C'):
        val = getattr(THIS, field)
        assert len(val) == 0, f'{field} should be empty'


def test_returns_new_instance_each_call():
    """Each call returns an independent struct."""
    t1 = OptFom_Initialize_Loop_Struct()
    t2 = OptFom_Initialize_Loop_Struct()
    t1.FOM = 99
    assert t2.FOM == 0


# ============================================================
# COM Octave oracle values (tools/octave_oracle.py, 4p16p0 compat file,
# OptFom_Initialize_Loop_Struct).  Generated 2026-09-22.
# No divergence was found; this pins the exact field list and its order, which
# MATLAB fixes at creation and which struct2cell/fieldnames consumers depend on.
#
#   fieldnames(OptFom_Initialize_Loop_Struct()) -> the 28 names below, in order
#   THIS.FOM = 0;  every other field is [] (0x0)
# ============================================================

_OCTAVE_FIELDS = [
    'FOM', 'tx_index_vector', 'ctle_index', 'g_LP_index', 'itick', 'g_dc',
    'g_DC_low', 'H_ctf', 'txffe', 'cursor_i', 'A_s', 'A_p', 'far_cursors',
    'precursors', 'dfetaps', 'tail_RSS', 'floating_tap_coef',
    'excess_dfe_cursors', 'C', 'MMSE_results', 'PSD_results',
    'floating_tap_locations', 'sigma_N', 'sigma_TX', 'total_noise_rms',
    'ISI_N', 'h_J', 'sigma_ne']


def test_field_names_and_order_match_matlab():
    THIS = OptFom_Initialize_Loop_Struct()
    assert list(vars(THIS).keys()) == _OCTAVE_FIELDS


def test_only_fom_is_non_empty():
    THIS = OptFom_Initialize_Loop_Struct()
    assert THIS.FOM == 0
    for name in _OCTAVE_FIELDS[1:]:
        assert getattr(THIS, name) == [], name


def test_each_call_returns_independent_lists():
    """MATLAB structs are values; two calls must not share the same list."""
    a = OptFom_Initialize_Loop_Struct()
    b = OptFom_Initialize_Loop_Struct()
    a.dfetaps.append(1.0)
    assert b.dfetaps == []
