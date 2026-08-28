# ============================================================
# MATLAB→Python translation notes for get_center_of_UI
# MATLAB lines: 7320–7330
# ============================================================
# 1-based vs 0-based indexing: MATLAB [~, half_UI] = min(...) returns a
#   1-based index.  This Python translation returns a 0-based index.
#   Callers must NOT subtract 1 — the Python convention is already applied.
#   Relationship: python_result = matlab_result - 1.
# Range: MATLAB 0:1/N:1-1/N produces exactly N points.  np.arange(N)/N
#   is equivalent and avoids floating-point accumulation errors.
# Output shape: scalar int.
# Known discrepancy from prior sicopr.py attempt: none found.
# ============================================================
import numpy as np


def get_center_of_UI(samples_per_UI):
    """Return the 0-based index of the UI sample closest to the centre (0.5).

    UI_window = [0, 1/N, 2/N, ..., (N-1)/N]
    Returns argmin(|UI_window - 0.5|) — 0-based.
    (MATLAB returns the same value 1-based.)
    """
    N = int(samples_per_UI)
    UI_window = np.arange(N) / N          # MATLAB line 7328
    half_UI = int(np.argmin(np.abs(UI_window - 0.5)))  # MATLAB line 7330
    return half_UI


if __name__ == "__main__":
    for n in [1, 2, 4, 5, 8, 32]:
        print(f"get_center_of_UI({n}) = {get_center_of_UI(n)}")
