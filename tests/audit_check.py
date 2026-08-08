"""Shared check helper for audit test scripts (see AUDIT_PROMPT_com_conversion.md
section 6). Each audit test file does:

    from audit_check import check, finish
    check("name_of_check", cond, "reason shown on FAIL")
    ...
    finish()   # prints summary, exits nonzero if any check failed

Every check appends a row to tests/results.csv:
    timestamp, test_name, result, reason
Not a pytest module. Runnable audit scripts import it when executed as
`python tests/test_<area>.py` from the repo root or from tests/.
"""
import csv
import datetime
import os
import sys

RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results.csv")
_results = []


def _ensure_header():
    if not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0:
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "test_name", "result",
                                    "reason"])


def check(name, cond, reason=""):
    """Record one named boolean check. Prints PASS/FAIL, logs to results.csv."""
    cond = bool(cond)
    status = "PASS" if cond else "FAIL"
    if cond:
        print("PASS %s" % name)
    else:
        print("FAIL %s: %s" % (name, reason))
    _ensure_header()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            name, status, "" if cond else reason])
    _results.append(cond)
    return cond


def finish():
    """Print a summary and exit nonzero if any check failed."""
    n = len(_results)
    n_fail = _results.count(False)
    print("%d checks, %d failed" % (n, n_fail))
    sys.exit(1 if n_fail else 0)
