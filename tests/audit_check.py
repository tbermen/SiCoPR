"""Shared check helper for audit test scripts (see AUDIT_PROMPT_com_conversion.md
section 6). Each audit test file does:

    from audit_check import check, xcheck, finish
    check("name_of_check", cond, "reason shown on FAIL")
    xcheck("known_divergence", cond, "why com.py differs from MATLAB")
    ...
    finish()   # prints summary, exits nonzero on any UNEXPECTED outcome

Use `check` for behaviour that must match MATLAB, and `xcheck` for a divergence
that has been reviewed and accepted. An xcheck that starts passing fails the run,
so the divergence ledger cannot silently go stale.

Every check appends a row to tests/results.csv:
    timestamp, test_name, result, reason
where result is PASS / FAIL / XFAIL / XPASS.
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
_xresults = []


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


def xcheck(name, cond, reason=""):
    """Record a KNOWN, accepted divergence from the MATLAB reference.

    `cond` is the same "agrees with MATLAB" condition `check` takes, so an
    xcheck reads identically to the check it replaces. The difference is which
    outcome is news:

        cond False -> XFAIL. The documented divergence is still there. Expected;
                      does not fail the run.
        cond True  -> XPASS. The divergence is GONE. The ledger entry is stale,
                      so the run FAILS to force someone to update it.

    Without this, every known divergence sat in the suite as a permanent FAIL,
    the totals never reached zero, and a genuine regression was invisible in the
    noise. Two entries had in fact gone stale unnoticed (the get_TDR s2p RL
    formula and B16-D20's A_p offset, both fixed in the 8-defect correlation commit) — XPASS exists to
    catch exactly that.
    """
    cond = bool(cond)
    status = "XPASS" if cond else "XFAIL"
    if cond:
        print("XPASS %s: divergence resolved — update the ledger and promote "
              "this to check()" % name)
    else:
        print("XFAIL %s (known divergence): %s" % (name, reason))
    _ensure_header()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            name, status, reason if not cond else "divergence resolved"])
    _xresults.append(cond)
    return cond


def finish():
    """Summarise and exit nonzero on any unexpected outcome.

    Unexpected means a check that failed, or an xcheck that passed (a stale
    ledger entry). Known divergences that are still present are expected and
    keep the run green.
    """
    n_fail = _results.count(False)
    n_stale = _xresults.count(True)
    print("%d checks, %d failed; %d known divergences, %d resolved"
          % (len(_results), n_fail, len(_xresults), n_stale))
    if n_stale:
        print("  -> a documented divergence no longer reproduces; the ledger "
              "entry above is out of date")
    sys.exit(1 if (n_fail or n_stale) else 0)
