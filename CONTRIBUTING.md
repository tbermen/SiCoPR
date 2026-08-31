# Contributing to SiCoPR

Thanks for looking at this. One rule matters more than all the others:

> ## `sicopr.py` is generated. Never edit it.
>
> It is assembled from `com_functions/fn/<name>/py_impl.py` by `assemble_sicopr.py`.
> A change made directly to `sicopr.py` is silently discarded the next time anyone
> runs the assembler — including CI. If your diff touches `sicopr.py` and nothing
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
   suite, the cross-check scripts, a check that `sicopr.py` matches its sources, and
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
python assemble_sicopr.py

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
there are 177 inlined copies of 70 functions. A fix applied to one `py_impl.py`
reaches the canonical copy only. `tests/test_inlined_copies.py` compares copies
behaviourally and will tell you which ones you missed — but it can only drive
136 of the 177, so check `com_functions/inlined_copies.json` for the rest rather
than assuming the harness has you covered.

## Changing the config editor (`gui/`)

The GUI plays by different rules from the engine. It is **not** generated, it is
**not** a port of anything, and `assemble_sicopr.py` does not touch it — edit
`gui/` files directly.

```powershell
python gui/app.py --no-browser     # http://127.0.0.1:8765
python tests/test_config_roundtrip.py
python tests/test_gui_server.py
python tests/test_gui_static.py
```

Three things to know before changing it:

**`gui/config_io.py` writes spreadsheets, and that is the risky part.** Nine
value cells in a stock config are spreadsheet *formulas*, and the engine reads
cached values — openpyxl can preserve one or the other, never both, so the
writer edits the sheet XML directly. `.START` package blocks are separate
namespaces. A value's type is semantics: `0` is a fixed tap, `[ -0.34:.02:0]` is
a ~198-point sweep. `tests/test_config_roundtrip.py` gates all of it by handing
source and rewrite to the real `read_ParamConfigFile` and requiring identical
`param` and `OP`. See [`gui/README.md`](gui/README.md) for the details.

**`gui/static/app.js` must parse.** A syntax error there breaks the entire page
while every server-side test still passes, because the server is not the thing
that is wrong. `tests/test_gui_static.py` parses it — with `node --check` if you
have node, otherwise with `esprima` (`pip install esprima`, already in
`requirements.txt` as test-only). It warns loudly rather than passing quietly if
it has neither. This is not belt-and-braces: a mangled escape sequence shipped
exactly that failure once.

**The server reads and writes files by path and spawns processes.** Every path
parameter is containment-checked against the repo, file types are whitelisted,
and commands are built as argv lists without a shell. If you add an endpoint
that takes a path, add the guard and a test that plants a real file outside the
repo — a test that asks for a *non-existent* outside path passes whether or not
the guard exists, which is a mistake this suite has made and corrected twice.

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
copyright notice in `LICENSE` and in the generated `sicopr.py` header intact, and
do not describe the project — in code, docs, or commit messages — as endorsed by
or affiliated with IEEE or the 802-COM Authors. Clause 3 forbids it.

Provenance of the upstream — exact versions, SHA-256 checksums, and what is and
is not covered by its licence — is recorded in [`NOTICE`](NOTICE).

### Which header a new file gets

Two forms, and the choice is about accuracy, not preference. Naming the 802-COM
Authors on a file they had no hand in attributes copyright to the wrong people,
which is the same mistake as omitting them from a file they wrote.

**Derived from the upstream MATLAB** — a translation, a transcribed formula, or a
test whose expected values come from the reference:

```python
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause
```

**Original work** — the GUI, the R analysis, the test harness plumbing. The GUI is
not a port of anything, and its files say so:

```python
# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause
```

The header goes *after* the module docstring, so the docstring stays first for
`help()`. Two exceptions: empty `__init__.py` markers get nothing, and the 157
`com_functions/fn/*/py_impl.py` files get nothing — the assembler inlines their
leading comments, so a header there would appear 157 times inside the generated
engine. [`com_functions/fn/README.md`](com_functions/fn/README.md) covers that
tree collectively.

### Sign your commits off (DCO)

Every commit must carry a `Signed-off-by:` line. Git adds it for you:

```bash
git commit -s -m "your message"
```

which appends

```
Signed-off-by: Your Name <your@email>
```

That line means you agree to the [Developer Certificate of Origin
1.1](https://developercertificate.org/) — in short: that you wrote the change, or have
the right to submit it, and that you are contributing it under this project's
licence.

**Why this is asked for.** BSD-3-Clause, unlike Apache-2.0, contains no clause
covering *inbound* contributions. Without a sign-off there is no record that a
contributor agreed to the licence their code is being distributed under. The DCO
is the lightweight way to have that record: no paperwork, no CLA to sign, one
line per commit.

Use your real name and an email you control. **Please use a personal address
rather than an employer one** — a corporate address in the commit record invites
the question of whether the work was done within the scope of employment, and
that is a question worth not raising.

CI checks this on every pull request. If you forget, `git commit --amend -s` on
the last commit, or `git rebase --signoff origin/master` for a series, then
force-push your branch.

## Commit messages

Say what changed and **what it bought**, with the evidence. "Fixed rounding" is
not useful; "`nui` used banker's rounding where MATLAB rounds half away from
zero — one tie in 98,145 calls, dropped one ISI sample, closed the last COM
miss" is. `docs/FIX_SUMMARY.md` is the running ledger; add an entry there for
anything that changes a number.
