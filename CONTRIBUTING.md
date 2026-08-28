# Contributing

Thanks for looking at this. One rule matters more than all the others:

> ## `com.py` is generated. Never edit it.
>
> It is assembled from `com_functions/fn/<name>/py_impl.py` by `assemble_com.py`.
> A change made directly to `com.py` is silently discarded the next time anyone
> runs the assembler — including CI. If your diff touches `com.py` and nothing
> under `com_functions/fn/`, it will be rejected.

---

## How to propose a change

**Nobody pushes to `master`, including the maintainer.** Every change arrives as a
pull request and is merged after review. If you have never done this on GitHub,
the whole sequence is:

```bash
# 1. Fork on github.com (button, top right). You now own a copy.
# 2. Clone YOUR fork
git clone https://github.com/<you>/SiCoPR.git
cd SiCoPR

# 3. Branch. Name it after the change, not after yourself.
git checkout -b fix-ctle-index-base

# 4. Work. Follow "The loop" below, and make sure tests/run_all.ps1 is green.
git add -A
git commit          # see "Commit messages"

# 5. Push to your fork and open the pull request
git push -u origin fix-ctle-index-base
# GitHub prints a link; follow it, or use the "Compare & pull request" button.
```

You do **not** need permission to do any of this, and you cannot break anything:
a fork is your own copy, and a pull request is a proposal, not a change.

**Open an issue first if the change is large or changes a number.** A defect
report costs you five minutes and may save you a week — the answer is sometimes
"the MATLAB does that too, on purpose" (see *Prefer fidelity over improvement*).
For a typo or an obvious bug, skip straight to the pull request.

### What happens to your pull request

1. **CI runs automatically** on a clean clone with no correlation data: the unit
   suite, the cross-check scripts, a check that `com.py` matches its sources, and
   a check that the licence notice is intact. All of it must pass. You can run
   the same thing locally first — that is what `tests/run_all.ps1` is.
2. **The maintainer reviews it.** Expect questions about which MATLAB lines the
   new behaviour matches; that is the review, not scepticism about you.
3. **Numbers get checked against the reference set** if the change can move a
   COM, FOM or sampling-phase value. That run needs data that is not in this
   repository, so the maintainer does it. It takes a few hours; be patient.
4. **Merge.** Your commits keep your name on them.

A pull request that is stalled is usually waiting on a question, not rejected.
Ask.

### Reporting a defect without fixing it

That is a real contribution and is welcome. What makes a report actionable:

- the configuration and channel involved, or the smallest input that shows it;
- what you expected, what you got, and **where the MATLAB says the expected
  thing** if you know;
- the full traceback, if it crashed.

If the input is not yours to share, say so — a description of the shape of the
problem is still worth having.

---

## The loop

```bash
# 1. edit the per-function source
#    com_functions/fn/<name>/py_impl.py

# 2. test that function alone
python -m pytest com_functions/fn/<name>/test_verify.py -q

# 3. regenerate the engine
python assemble_com.py

# 4. run everything
powershell -ExecutionPolicy Bypass -File tests/run_all.ps1     # Windows
```

`tests/run_all.ps1` is the gate. Note that **`pytest tests/` runs almost nothing
and exits 0** — most files in `tests/` are standalone audit scripts that the
runner invokes directly. Use the runner.

## What a good change looks like

**Cite the MATLAB.** This is a port. Most functions carry a comment naming the
MATLAB source lines they translate; keep those accurate, because they are how
the code gets reviewed. If you change behaviour, say which MATLAB lines the new
behaviour matches.

**Prefer fidelity over improvement.** If the MATLAB does something inefficient,
awkward, or numerically second-best, the port does it too. There is a real
example in the history: `Init_PDF_Fast` was written to compute a PDF grid a few
ulp *more accurately* than MATLAB's colon operator, and that extra accuracy
changed which side of a bin boundary a lookup fell on. Matching the reference is
the goal; making it better is a different project.

**Watch three specific traps.** They account for most defects found so far:

| trap | why it bites |
|---|---|
| MATLAB assigns structs **by value**, Python binds a **reference** | a function that writes to a parameter it never returns changes nothing in MATLAB and everything in Python |
| MATLAB is **1-based**, Python **0-based** | several structs deliberately store 1-based indices; `tests/test_index_base.py` holds the declared base for each and will fail if you index with one raw |
| MATLAB `round()` is **half away from zero**, Python's is **banker's** | they differ on exact ties, and ties are not rare when the input is a ratio of integers |

**One function can exist as many copies.** The assembler inlines helpers, so
there are ~178 inlined copies of ~70 functions. A fix applied to one
`py_impl.py` reaches the canonical copy only. `tests/test_inlined_copies.py`
compares copies behaviourally and will tell you which ones you missed.

## Tests

New behaviour needs a test that fails without your change. Prove it: reintroduce
the bug, watch the test fail, then fix it again. Several guards in `tests/` were
written this way and say so in their docstrings.

Prefer an oracle that does not depend on reading the MATLAB — analytic values,
physical invariants, or agreement between two independent code paths. Tests that
merely re-assert the same reading of MATLAB that produced the code cannot catch a
misreading, and at least one defect in this project's history was *encoded* into
its own unit-test fixtures.

## Correlation data

The 208-case correlation runs against IEEE 802.3dj channel S-parameters and
configuration workbooks that are **not** in this repository — they are not ours
to redistribute. `README.md` names the contributions to download and where the
harness expects them. Tests that need that data skip cleanly when it is absent,
so a fresh clone still runs the full suite.

## Licensing

This is a derivative of the IEEE 802.3 COM MATLAB reference, which is
BSD-3-Clause. Contributions are accepted under the same license. Keep the
copyright notice in `LICENSE` and in the generated `com.py` header intact, and
do not describe the project — in code, docs, or commit messages — as endorsed by
or affiliated with IEEE or the 802-COM Authors. Clause 3 forbids it.

## Commit messages

Say what changed and **what it bought**, with the evidence. "Fixed rounding" is
not useful; "`nui` used banker's rounding where MATLAB rounds half away from
zero — one tie in 98,145 calls, dropped one ISI sample, closed the last COM
miss" is. `docs/FIX_SUMMARY.md` is the running ledger; add an entry there for
anything that changes a number.
