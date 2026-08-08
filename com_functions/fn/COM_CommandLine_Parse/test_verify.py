"""Verification tests for COM_CommandLine_Parse().

# ============================================================
# MATLAB GROUND TRUTH (lines 1149-1200)
# Parses command-line arguments for COM.
# Legacy mode: config_file, num_fext, num_next.
# TD mode: sets OP.TDMODE=True, OP.GET_FD=False.
# Config2Mat mode: sets OP.CONFIG2MAT_ONLY=True.
# Empty varargin: returns defaults.
# ============================================================
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.COM_CommandLine_Parse.py_impl import COM_CommandLine_Parse
from types import SimpleNamespace


def _OP():
    return SimpleNamespace()


def test_empty_args_defaults():
    """Empty varargin returns default OP with TDMODE=False."""
    cfg, nfext, nnext, kw, OP, rest = COM_CommandLine_Parse(_OP())
    assert OP.TDMODE is False
    assert OP.GET_FD is True
    assert cfg == ''


def test_legacy_mode_config_file():
    """Legacy mode parses config_file as first positional arg."""
    cfg, nfext, nnext, kw, OP, rest = COM_CommandLine_Parse(_OP(), 'my_config.csv', 2, 1)
    assert cfg == 'my_config.csv'
    assert nfext == 2
    assert nnext == 1


def test_td_mode_sets_flags():
    """TD keyword sets TDMODE=True and GET_FD=False."""
    cfg, nfext, nnext, kw, OP, rest = COM_CommandLine_Parse(_OP(), 'TD', 'ch.csv', 0, 0)
    assert OP.TDMODE is True
    assert OP.GET_FD is False
    assert cfg == 'ch.csv'


def test_config2mat_mode():
    """Config2Mat keyword sets CONFIG2MAT_ONLY=True."""
    cfg, nfext, nnext, kw, OP, rest = COM_CommandLine_Parse(_OP(), 'Config2Mat', 'cfg.xlsx')
    assert OP.CONFIG2MAT_ONLY is True
    assert cfg == 'cfg.xlsx'


def test_remember_keyword_td():
    """Remember_keyword is 'TD' when TD keyword used."""
    _, _, _, kw, OP, _ = COM_CommandLine_Parse(_OP(), 'TD', 'cfg.csv', 0, 0)
    assert kw.upper() == 'TD'


def test_legacy_no_fext_next_when_string_follows():
    """When next arg is non-numeric string, num_fext/num_next default to 0."""
    cfg, nfext, nnext, kw, OP, rest = COM_CommandLine_Parse(_OP(), 'ch.csv', 'extra')
    assert nfext == 0
    assert nnext == 0
    assert cfg == 'ch.csv'
