"""Verification tests for save_cmd_line().

# ============================================================
# MATLAB GROUND TRUTH (lines 11165-11174)
# cmd_str = cli_name('config_file',num_fext, num_next,'chdata(1).filename'
# for i=2:len(chdata): append ,'chdata(i).filename'
# append ')'
#
# Result: cli_name('config',5, 3,'file1','file2')
# ============================================================
"""
import sys
import os
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.save_cmd_line.py_impl import save_cmd_line


def _chdata(filenames):
    return [SimpleNamespace(filename=fn) for fn in filenames]


def test_single_channel_format():
    """Single channel: no extra commas from loop."""
    ch = _chdata(['thru.csv'])
    result = save_cmd_line('cfg.xlsx', ch, 2, 1, 'run_com')
    assert result == "run_com('cfg.xlsx',2, 1,'thru.csv')"


def test_multiple_channels():
    """Multiple channels: comma-separated filenames after first."""
    ch = _chdata(['thru.csv', 'fext1.csv', 'fext2.csv'])
    result = save_cmd_line('cfg.xlsx', ch, 2, 1, 'run_com')
    assert result == "run_com('cfg.xlsx',2, 1,'thru.csv','fext1.csv','fext2.csv')"


def test_ends_with_paren():
    """Output always ends with ')'."""
    ch = _chdata(['a.s4p'])
    result = save_cmd_line('cfg.xlsx', ch, 0, 0, 'com')
    assert result.endswith(')')


def test_starts_with_cli_name():
    """Output starts with cli_name."""
    ch = _chdata(['f.csv'])
    result = save_cmd_line('c.xlsx', ch, 1, 1, 'my_cli')
    assert result.startswith('my_cli(')


def test_config_file_quoted():
    """config_file appears in single quotes."""
    ch = _chdata(['f.csv'])
    result = save_cmd_line('my_config.xlsx', ch, 1, 2, 'com')
    assert "'my_config.xlsx'" in result
