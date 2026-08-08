"""Verification tests for missingParameter() — mandatory-parameter error helper.

# ============================================================
# MATLAB GROUND TRUTH
# MATLAB error() always raises MException; the message format is:
#   'The data for mandatory parameter %s is missing or incorrect'
# with %s substituted by parameterName.
#
# Examples:
#   missingParameter('sigma')
#     → error message contains 'sigma'
#   missingParameter('Ts')
#     → error message contains 'Ts'
# ============================================================
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from com_functions.fn.missingParameter.py_impl import missingParameter


def test_raises_value_error():
    with pytest.raises(ValueError):
        missingParameter("sigma")


def test_message_contains_param_name():
    with pytest.raises(ValueError, match="sigma"):
        missingParameter("sigma")


def test_message_contains_different_param():
    with pytest.raises(ValueError, match="Ts"):
        missingParameter("Ts")


def test_message_text():
    """Full message matches the MATLAB format string."""
    with pytest.raises(ValueError, match="missing or incorrect"):
        missingParameter("any_param")
