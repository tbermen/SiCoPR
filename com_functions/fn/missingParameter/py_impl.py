# ============================================================
# MATLAB→Python translation notes for missingParameter
# MATLAB lines: 8406–8409
# ============================================================
# 1-based vs 0-based indexing: not applicable.
# Column-major vs row-major: not applicable.
# MATLAB error(): raises an exception with an identifier and message.
#   Python equivalent: raise ValueError with the same message text.
# Output shape: no return value — function always raises.
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================


def missingParameter(parameterName):
    """Raise ValueError for a missing or incorrect mandatory parameter.

    Mirrors MATLAB error('error:badParameterInformation', ...).
    """
    raise ValueError(
        f"The data for mandatory parameter {parameterName} is missing or incorrect"
    )


if __name__ == "__main__":
    try:
        missingParameter("sigma")
    except ValueError as e:
        print("Caught:", e)
