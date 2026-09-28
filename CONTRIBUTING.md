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

**Watch four specific traps.** They account for most defects found so far:

| trap | why it bites |
|---|---|
| MATLAB assigns structs **by value**, Python binds a **reference** | a function that writes to a parameter it never returns changes nothing in MATLAB and everything in Python |
| MATLAB is **1-based**, Python **0-based** | several structs deliberately store 1-based indices; `tests/test_index_base.py` holds the declared base for each and will fail if you index with one raw |
| MATLAB `round()` is **half away from zero**, Python's is **banker's** | they differ on exact ties, and ties are not rare when the input is a ratio of integers |
| MATLAB holds **every number as a double**; Python needs an **int** for a count | a workbook that stores `32` as `32.0` (any tool that re-saves it) runs in MATLAB and crashed `np.ones(M)` here. Config counts are conditioned in `read_ParamConfigFile._xls_param`: a new keyword used as a count, size or index belongs in `_COUNT_KEYWORDS` |

**One function can exist as many copies.** The assembler inlines helpers, so
there are 62 inlined copies of 45 functions. A fix applied to one `py_impl.py`
reaches the canonical copy only. `tests/test_inlined_copies.py` compares copies
behaviourally and will tell you which ones you missed — but it can only drive
28 of the 62, so check `com_functions/inlined_copies.json` for the rest rather
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
parameter is containment-checked against the roots the user gave it (the repo,
`--dir`/`--run-dir`/`SICOPR_DIRS`, and directories opened in the session; see
`_safe_path` in `gui/app.py`), file types are whitelisted,
and commands are built as argv lists without a shell. If you add an endpoint
that takes a path, add the guard and a test that plants a real file outside the
repo — a test that asks for a *non-existent* outside path passes whether or not
the guard exists, which is a mistake this suite has made and corrected twice.

## Tests

New behaviour needs a test that fails without your change. Prove it: reintroduce
the bug, watch the test fail, then fix it again. Several guards in `tests/` were
written this way and say so in their docstrings.

Prefer an oracle that does not depend on reading the MATLAB — the executed
reference first (see *Verifying a function port* below), then analytic values,
physical invariants, or agreement between two independent code paths. Tests that
merely re-assert the same reading of MATLAB that produced the code cannot catch a
misreading, and at least one defect in this project's history was *encoded* into
its own unit-test fixtures.

**A change made for speed** is accepted only if `tools/equivalence_check.py`
passes on all 28 checkpoint cases: strict outputs bit-identical, noise and DER
fields within 1e-12 relative per element, Octave agreement no worse. Two August
speed-ups that were verified only against corpus statistics turned out not to be
equivalent and were withdrawn.

No runtime comparison with MATLAB is offered: the timings that exist were taken
on different machines and, at the time, on different search spaces. The August
2026 Python-vs-Python figure (roughly 4.5–5×) is withdrawn pending a
like-for-like re-measurement; its FFT convolution and hoisted Gram matrix were
not equivalent (the FFT lost the far tail of the noise CDF that DER is read from)
and were undone on 2026-09-23/24. Five speed-ups have been re-earned under the
rule above; on one case the run went from 304 s to 50 s against the accurate
baseline. No corpus-level multiplier is current.

## The test suite in detail

`tests/` holds **two kinds of file**, and the difference matters:

| | how to run |
|---|---|
| `test_smoke.py`, `test_checkpoints.py`, `test_end_to_end.py`, `test_export_columns.py` | pytest modules |
| every other `test_*.py` | standalone scripts: `python tests/test_x.py` |

Do **not** run `pytest tests` over the whole directory. The audit scripts call
`sys.exit()` at import, which aborts collection: pytest reports `no tests ran`
**and still exits 0**, so nothing runs and nothing complains. `run_all.ps1` runs
the pre-flight audit, assembly, interface checks, the unit tests and every
cross-check script, dispatching each kind correctly.

The scripts record two outcomes. `check()` is behaviour that must match MATLAB;
`xcheck()` is a reviewed, accepted divergence, which reports `XFAIL` while it
persists and **fails the run if it starts passing**, so a divergence that gets
fixed cannot leave a stale entry behind in the ledger.

State on 2026-09-26: **2025** per-function tests across 157 functions, and **44
audit scripts**; the 41 that print a tally total **628 checks** with 20 accepted
divergences. Whether any function-level verification is open is answered by
`python com_functions/verification/report.py`, per
[`docs/VERIFICATION.md`](docs/VERIFICATION.md).

**Skips on a fresh clone are normal.** Tests that need the correlation data (see
*Correlation data* below) skip and say what they wanted. Some skip messages name
a local directory such as `tests\1_IEEE_802p3dj_COM_Spreadsheets`, or point at a
README section that has since moved here; that data is deliberately absent from
the repository, and the skip is expected.

### What the coverage does *not* reach

The assembler inlines helpers into their callers. Most calls between translated
functions became imports on 2026-09-22, but the engine still contains **62
inlined copies of 45 functions**. `tests/test_inlined_copies.py` compares each
copy against its canonical top-level version and reports plainly how far it gets:

```
62 inlined copies of 45 functions; 28 comparison(s) made, 34 skipped
```

**34 of those 62 copies have no behavioural verification.** They are not skipped
by choice: the harness drives both sides from synthetic inputs, and for these it
cannot build any without a populated `param`/`OP` struct or a real Touchstone
file. For them the only check is that the copy still accepts the same arguments
as the canonical.

The correlation result says nothing about those 34. The reference cases exercise
the *canonical* implementations, which `_run_com` injects; the inlined copies are
`or`-fallbacks that a normal run never reaches. Several are deliberately narrow
(a Gaussian fitted to the sample RMS where the canonical builds an exact PDF, for
instance), and 17 such divergences are catalogued with reasons in
`test_inlined_copies.py`'s `KNOWN_BEHAVIOUR`. Five divergences were found only
when coverage was raised from 113 copies to 136 (before the copies were collapsed
onto imports), which is the argument for treating the remaining 34 as unverified
rather than as probably fine: every time this harness has been pointed at more
copies, it has found more divergences.

`com_functions/inlined_copies.json`, written by the assembler, records where every
copy came from: which helper, which caller, and the upstream MATLAB line range of
each. It counts a wider population than the 62 above: the assembler reports 245
inlined helper copies, 62 of which duplicate a translated function; the rest are
private helpers with no canonical top-level function to compare against.

The practical risk is not in what runs today. It is that a future caller which
forgets to inject would silently get the approximation, with no error and a
plausible number.

### Cross-cutting guards

Most of the scripts exist because the per-function tests structurally cannot
catch the defect classes that actually got through. Each was built from a real
failure and verified by re-introducing it:

| script | guards against | why |
|---|---|---|
| `test_reference_leaks.py` | writing to a parameter the function never returns | MATLAB passes structs **by value**, Python by reference. **Five of the original eight** correlation defects were this class, and five of the eighteen correlation-era entries in [`docs/FIX_SUMMARY.md`](docs/FIX_SUMMARY.md). Caught a new instance during the 4p16p0 port. |
| `test_inlined_copies.py` | an inlined copy drifting from its canonical function | there are **62 copies of 45 functions**; a fix to `py_impl.py` reaches only one of them. Engine defect #6 lived in three copies. |
| `test_optimization_invariants.py` | the speed work silently breaking | cache transparency and key completeness, the verified Gram gather, direct (never FFT) convolution, shared buffers. Found a live cache-aliasing defect. |
| `test_matlab_stage_oracles.py` | drift from real MATLAB values | pins **208 cases × 35 scalars + 14 vector families** taken from the reference workbooks. The oracle file itself is not tracked (it *is* reference data); point `COM_STAGE_ORACLES` at a local copy, and without it the test skips. |
| `test_octave_checkpoints.py` | a stage drifting from the reference code | compares 10 stage structs on 28 cases against COM Octave goldens (`COM_OCTAVE_CHECKPOINTS`; `COM_CHECKPOINT_CASES=all` for every case). The goldens are local-only, so it skips in a clone. See [`docs/VERIFICATION.md`](docs/VERIFICATION.md). |
| `test_abort_path_leaks.py` | writing into a caller's struct before an early return | the sibling of the leak above that the leak guard cannot see: the function *does* return the struct, but commits values on a path MATLAB never commits on. Ledger #10b. |
| `test_sort_stability.py` | `np.argsort` reordering ties | MATLAB's `sort` is stable, NumPy's default is not. Ties were measured in **81% of argsort calls** rather than assumed rare. |
| `test_integer_ratio_rounding.py` | banker's rounding on a ratio of integers | MATLAB rounds half away from zero. The audit dismissed this as measure-zero, which is true for continuous data and false for `a/b` with both integral (ledger #16). |
| `test_structural_invariants.py` | properties no single function owns | shapes, index bases and struct field sets that only go wrong between functions. |
| `test_stage_figures.py` | a pipeline stage losing its figures | fails if any of the seven stages stops emitting a figure; `STAGE_INDEX.md` in each case directory lists a stage with no figure as such rather than omitting it. |

The configuration editor under `gui/` has its own three, described in
*Changing the config editor* above: `test_config_roundtrip.py` (the config writer
damaging a workbook), `test_gui_server.py` (the HTTP layer: payload shape, path
containment, process control, and the results and dashboard views) and
`test_gui_static.py` (`app.js` failing to parse).

### Tooling for a new MATLAB release

```powershell
python tools/matlab_version_diff.py OLD.m NEW.m   # -> which py_impl files to re-check
python tools/matlab_version_diff.py --self-check  # validates the differ itself
```

[`docs/VERSIONS.md`](docs/VERSIONS.md) records what the last release changed.

## Correlation data

**The port ships; the data it was verified against does not.** The engine, its
tests, the tooling, the MATLAB reference sources and the documentation are all
here. The channel S-parameters the port was correlated against are IEEE 802.3
contributions and are not ours to redistribute, and neither are the outputs:
reference values, comparison tables, generated figures and result files are
either derived from that data or produced by running on it, and are excluded on
the same grounds. CI enforces the exclusion. The COM configuration workbooks are
a different case: each carries a `License Notice` sheet placing it under the
same BSD-3-Clause licence as the reference code, which is why one ships in
[`examples/`](examples/) together with the results both engines produced on it.

The channels themselves are public: the sets used come from the IEEE 802.3dj
public area, whose [channel and tool page](https://www.ieee802.org/3/dj/public/tools/index.html)
lists the CR and KR contributions by name, and the configuration workbooks from
the COM ad hoc.

**This repository documents the verification; it does not offer to reproduce
it.** The harness that ran the 208-case and 1368-case comparisons is not in this
repository; it is kept with the data it needs. Anyone wanting to check the result
independently would supply their own channels and configs, run their own MATLAB
or COM Octave (`tools/octave_compare.py` does the Octave side for one case), and
compare against this engine's output. The agreement statistics in
[`MATLAB_Correlation_Review.md`](MATLAB_Correlation_Review.md) are stated
precisely enough to support that. Tests that need correlation data skip cleanly
when it is absent, so a fresh clone still runs the full suite.

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

## Verifying a function port

`docs/VERIFICATION.md` is the contract. Read it before any function-level
verification work.

The short form: a reading of the MATLAB is not verification. `sigma = std(x)`
and `sigma = np.std(x)` read alike and are not alike, and a reading is exactly
what produced that defect. Execute the reference with `tools/octave_oracle.py`,
pin the values it returns as literals in the test, and head the block with the
string `COM Octave` so the coverage tool can see it.

COM MATLAB is the reference behaviour. COM Octave is a proxy for it that can be
run at any level, from a single function to a whole case. Where Octave diverges
from MATLAB, fix Octave (see `octave/make_octave_compat.py`) and document the
fix, so it stays trustworthy as the oracle.

Run `tests\run_all.ps1`, never bare `pytest tests`, which collects almost
nothing and exits 0. Then `python com_functions/verification/report.py` answers
"are there any opens?" from the suite rather than from memory.
