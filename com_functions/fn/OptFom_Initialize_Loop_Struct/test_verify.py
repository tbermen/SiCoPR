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
