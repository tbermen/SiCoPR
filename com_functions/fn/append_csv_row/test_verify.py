# ============================================================
# MATLAB GROUND TRUTH for append_csv_row (L5157-5187)
#   - First call on a non-existent file writes the header row.
#   - Numeric cells formatted with %.6g; strings quoted with ".
#   - Empty row_cells writes only the header (no data line).
# ============================================================
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from com_functions.fn.append_csv_row.py_impl import append_csv_row


def test_writes_header_then_row(tmp_path):
    p = str(tmp_path / 'log.csv')
    append_csv_row(p, ['iter', 'L1_w', 'reason'], [])
    append_csv_row(p, ['iter', 'L1_w', 'reason'], [3, 1.234567, 'Evaluate Candidate'])
    lines = open(p).read().splitlines()
    assert lines[0] == 'iter,L1_w,reason'
    assert lines[1] == '3,1.23457,"Evaluate Candidate"'


def test_header_only_when_row_empty(tmp_path):
    p = str(tmp_path / 'log.csv')
    append_csv_row(p, ['a', 'b'], [])
    assert open(p).read().splitlines() == ['a,b']


def test_appends_without_rewriting_header(tmp_path):
    p = str(tmp_path / 'log.csv')
    append_csv_row(p, ['a'], [1])
    append_csv_row(p, ['a'], [2])
    lines = open(p).read().splitlines()
    # header written once (file did not exist on first call)
    assert lines == ['a', '1', '2']


def test_non_numeric_non_string_becomes_empty_quotes(tmp_path):
    p = str(tmp_path / 'log.csv')
    append_csv_row(p, ['a', 'b'], [None, 5])
    assert open(p).read().splitlines()[1] == '"",5'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
