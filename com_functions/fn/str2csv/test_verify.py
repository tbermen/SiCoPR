"""Verification tests for str2csv() — list of strings to comma-separated string.

# ============================================================
# MATLAB GROUND TRUTH
# The 2×n cell trick interleaves strings with commas, last comma
# replaced by '' before strcat:
#
#   str2csv({'a','b','c'})  → strcat('a',',','b',',','c','') = 'a,b,c'
#   str2csv({'hello','world'}) → 'hello,world'
#   str2csv({'single'})     → strcat('single','') = 'single'
#   str2csv({})             → strcat() = ''
# ============================================================
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.str2csv.py_impl import str2csv


def test_three_elements():
    assert str2csv(['a', 'b', 'c']) == 'a,b,c'


def test_two_elements():
    assert str2csv(['hello', 'world']) == 'hello,world'


def test_single_element():
    """Single element — no comma at all."""
    assert str2csv(['single']) == 'single'


def test_empty_list():
    assert str2csv([]) == ''


def test_no_trailing_comma():
    result = str2csv(['x', 'y', 'z'])
    assert not result.endswith(',')


def test_no_leading_comma():
    result = str2csv(['x', 'y', 'z'])
    assert not result.startswith(',')


def test_comma_count():
    """n elements → n-1 commas."""
    result = str2csv(['a', 'b', 'c', 'd'])
    assert result.count(',') == 3
