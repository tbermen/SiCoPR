# SiCoPR Function-Port Verification: Objective and Contract

## Why this document exists

This process has been run once before, and it reported success while the port
was wrong.

`dev/state/audit_ledger.csv` (2026-07) is a per-function ledger of exactly the
shape proposed below: one row per function, a `verdict` field, an `evidence`
field, a `test_files` field. It concluded **147 EQUIVALENT, 12 DIVERGENT** over
159 rows.

On 2026-09-22 an Octave-oracle pass found real divergences in 27 functions.
**All 27 were marked EQUIVALENT in that ledger.** None was flagged.

The ledger was not dishonest and the reviewers were not careless. The defect is
structural: `verdict` was a judgment, written once, stored, and never
re-evaluated. A stored judgment decays silently. The code moves, an assumption
turns out to be wrong, and the row still says `pass`.

Everything below follows from one rule:

> **Status is computed on every run. It is never stored, and never written by a
> human or an agent.** The only things written down are decisions, and every
> decision must be falsifiable by the suite.

If a future revision of this document reintroduces a writable `status` field,
it has reintroduced the 2026-07 outcome.

## Where this document lives

Adopted into the repository on 2026-09-23, when `verification/report.py` began
answering "are there any opens?" from the suite rather than from anyone's
recollection. It sat in `G:\si\dev\prompts\` while under review, which is
outside every repository and therefore the wrong home for a process that
governs code: a contract a contributor cannot read, cannot review in a pull
request and cannot version alongside what it governs decays into exactly the
2026-07 ledger, a document describing a process nobody can see running.

It is referenced from `CONTRIBUTING.md`.

## Objective

Reach a state where "are there any opens?" is answered by running
`verification/report.py`, per supported MATLAB version, and it prints zero.
Completion is never asserted from judgment. The report is the answer, and the
report derives its answer from executing the suite, not from reading a file
that records what someone once concluded.

## Reference resolution (never hard-coded here)

- The set of MATLAB versions under verification is
  `VERSION.json:supported_matlab_versions`. The default is
  `VERSION.json:default_matlab_version`.
- The reference `.m` file for each version is resolved from
  `VERSION.json:references`, which already exists and is keyed by version. Do
  not add version strings or file names to this document, to `methods.md`, or
  to `report.py`.
- The Octave oracle identity is recorded in `verification/oracle.json`: Octave
  version, tree commit hash, shim set hash. Every row closed by an
  Octave-based method stores the `oracle_id` that closed it.

## Layout (repo-relative)

```
com_functions/verification/
  methods.md                 shared across versions
  mutations.py               the defect-class operator catalogue
  equivalent_mutants.md      mutants that provably change nothing, with reasons
  oracle.json                current Octave oracle identity
  octave_deviations.md       verified-Octave vs MATLAB differences, each with a probe
  escapes.md                 discrepancies found outside the inventory
  report.py                  deterministic, command-line
  approved/<date>-<sha>.txt  approved reports, the anchor for every "since"
  <version>/inventory.json   generated, not authored
  <version>/findings.md
  <version>/audit_state.json
```

## The two kinds of state

Keeping these apart is the whole design.

**Derived state** is recomputed from the reference `.m`, the `py_impl.py`
sources and the test tree every time the report runs. It includes the row set,
every row's status, every count in the report, and every row's discrimination
score. It is never edited and never committed as a result, only as code that
produces it.

**Decided state** is the small set of human choices: a waiver and its reason,
an accepted divergence and its reason, a baseline being ratcheted. It is
committed. Every decided item must be **falsifiable**: the suite fails if the
decision stops describing the code.

The repository already has the mechanism for decided state and this process
should use it rather than invent another:

- `tests/audit_check.py` provides `check` / `xcheck` / `finish`. An `xcheck`
  records an accepted divergence, and **an `xcheck` that starts passing fails
  the run**, so an accepted divergence cannot quietly stop being real.
- Ratcheted baselines (`BASELINE_ORACLE_BACKED`, `BASELINE_SHAPE_CHECKED`,
  `BASELINE_UNDRIVABLE`, `BASELINE_WEAK_COMPOSITES`) fail in both directions: a
  regression fails outright, and an improvement fails until the baseline is
  lowered, so a gain is held.
- Pinned sets (`KNOWN_UNCOVERED`, `KNOWN_ARITY`, `KNOWN_BEHAVIOUR`) name the
  exact exceptions, so a new one is reported by name rather than absorbed into
  a count.

## The inventory is the contract (one per version)

### The row set is generated

The inventory is produced by parsing the version's reference `.m` together with
the `py_impl.py` sources and the test tree. It is not authored. This matters
because an authored inventory can be short, and a short inventory reports zero
opens for the wrong reason.

`tools/translation_coverage.py` already does this for row type (a): it parses
the reference, builds the call graph, classifies each function leaf or
composite, and grades it `oracle` / `values pinned` / `shape only` /
`no numeric result` / `no test` / `not translated`. That grade is the status.
`report.py` should extend that tool, not duplicate it.

### Row types, and where each denominator comes from

- **a. Translated function.** Denominator: every `function` defined in the
  version's reference `.m`. Cannot be short, because it is parsed from the
  file.
- **b. Builtin or operator seam.** Denominator: **every** MATLAB builtin and
  operator used in the reference, each classified as "same semantics" or
  "differs, and here is the rule". Not "the ones known to differ". "Known" is
  the reviewer's memory, and memory is what failed in 2026-07.

  This is a **gate, not a list**: parse the reference, diff the set of builtins
  used against the classification file, and fail on anything unclassified.
  Enforced that way it needs no adherence at all, and a new MATLAB version
  introducing a new builtin fails the suite until someone classifies it.

  A construct classified as "differs" carries its rule and the operator id that
  encodes it, and **every row whose Python touches that construct closes only
  by an oracle-backed method.** A reading cannot close it; a reading is what
  produced `np.std`.
- **c. Unit-conversion or index-base seam** between two ported functions.
- **d. Inlined copy.** One row per surviving copy of a translated function.
  A row that closes against the canonical proves nothing about the copies. On
  2026-09-22 a pass fixed 26 canonical functions and left 62 stale copies
  behind with the suite green. 73 copies remain, 23 of which nothing drives
  behaviourally. See `tests/test_inlined_copies.py` and
  `com_functions/inlined_copies.json`.

### Row fields

`id`, `matlab_ref` (line numbers in that version's file), `python_ref`
(`py_impl` path and function), `method`, `test_id`, `evidence`, `oracle_id`
(if Octave-closed), `manual_discrimination` (the named exception only, see
below), `notes`.

There is no `status` field and no writable `discriminates` field. Both are
computed.

Rows are added freely and never deleted. A row that does not apply gets a
waiver with a reason; pending waivers are listed in the report for approval.

## What it takes to close a row

All four, or the row is open.

1. **A registered method.** Named in `methods.md`, with what it proves, what it
   cannot prove, tolerance rules, and whether it depends on the Octave oracle.
   If the row touches a construct classified as semantically different, the
   method **must** be oracle-backed. This is checkable: the classification
   names the construct, the row names its `py_impl` site, and the method
   declares whether it executes the reference.
2. **A bound `test_id` that actually executes in `tests\run_all.ps1`.** If the
   named test does not run in the suite, the row is open regardless of what the
   evidence says. Note that `pytest tests` collects almost nothing and exits 0,
   so "the tests passed" is not self-evident in this repository.
3. **Demonstrated discrimination.** The bound test must be shown, by the suite
   itself, to fail when the implementation is broken. This is computed, not
   attested. See the next section.
4. **Evidence** sufficient for a reader to reproduce the closure without
   rerunning the reasoning.

## Discrimination: proving a test is not asleep

A test that passes on broken code is not evidence. This repository has shipped
two such checks (2026-09-04) that looked like proof, and a coverage grade that
keyed off a comment marker, so stripping a test's assertions left its grade
untouched.

The obvious remedy, "record that someone reintroduced the bug and watched the
test fail", is **wrong for the same reason a stored `status` is wrong.** It is
a judgment written once and never re-evaluated. Gut the test's assertions later
and the note still says it discriminates. Discrimination belongs in derived
state.

### The mutation catalogue

`verification/mutations.py` holds an operator catalogue. For each operator the
report applies the mutation to the relevant `py_impl.py`, runs that function's
bound test, and records whether the test failed. A test that still passes did
not discriminate.

Generic mutation operators (flip `+` to `-`) are mostly noise. **Every operator
in this catalogue is a defect that actually happened in this port**, with the
commit that fixed it:

| operator | the real defect |
|---|---|
| `np.std(x)` to and from `ddof=1` | N vs N-1 inside a membership threshold |
| `_mround` to `np.round` | half away from zero vs half to even, 41 sites |
| `_mmax` / `_mmin` to `np.max` / `np.min` | NaN handling, 48 sites |
| `_colon_x` to `arange()*binsize` | 24.6% of PDF axis elements, by one ulp |
| delete `chdata = list(chdata)` | by-reference leak, 5 of 8 defects in 2026-08 |
| `!= 0` to `> 0` | `d_cpdf` counting NaN as support |
| `_length(f)` to `len(f)` | `Butterworth_Filter`, `Bessel_Thomson_Filter` |
| drop `'extrap'` | `H_interp` outside the data range |
| `np.roll(V.T, s, axis=0)` to `np.roll(V, s)` | `FFE` on 2-D input |
| index shift by one | ADC clip off-by-one, `get_pdf_full` double correction |
| delete `.copy()` | `stot` / `ttos` mutating the caller's array |
| `np.linalg.solve` to and from `lstsq` | `force` on a singular VV |

The catalogue is cheap to run. Each operator applies to only a handful of files
(`np.std` 2, `_colon_x` 2, `'extrap'` 6, `_mmax` 25), and running only the
affected function's test keeps a full pass near a minute against a 17 s fn
suite. Apply an operator only where its pattern is present.

### Every fix becomes an operator

This is the escape rule's twin, and it is what makes the catalogue converge:

> **A fix that produces no new mutation operator is a process failure.**

When a defect is fixed, the defect itself is added to the catalogue. The suite
then permanently tests its own ability to catch that class again, for every
function, not just the one that was fixed. The catalogue only grows, and it
grows toward the defects this port actually produces rather than toward a
textbook list.

### The named exception

Some defects no operator can express. `findbankloc` was a "pick the highest
power non-overlapping banks" approximation where MATLAB runs a real badV/goodV
admissibility loop: a whole-algorithm misreading, not a mutable token. Those
rows carry `manual_discrimination` instead, recording how the control was
demonstrated. The report lists them separately so the exception stays visible
and stays small. A growing list of manual entries is a signal that the
catalogue is missing an operator.

### Equivalent mutants

Some mutations provably change nothing (`ddof=1` on a single-element array).
They read as "not caught" and are not test failures. They go in
`equivalent_mutants.md` with the reason, as decided state. Unlike a stored
`discriminates` note this is falsifiable: if the mutant ever stops being
equivalent, a test starts catching it and the entry is contradicted.

### Which signal is load-bearing

Two numbers come out of this, and they are not equal in standing.

**The mutation score is primary.** Rows whose bound test survived an applicable
mutation is derived entirely from the code and the suite. It depends on nobody
remembering anything, and it is the number that catches a sleeping test.

**Catalogue health is secondary.** It answers only whether the catalogue is
still growing, and it reads records that people write. If it ever goes
unreliable, the primary signal is unaffected. A green catalogue-health line is
not evidence that the tests are awake; only the mutation score is that. Do not
let the two be quoted as if they were interchangeable.

### What this does and does not prove

Catching a mutant proves the test is **not vacuous**. It does not prove the
test is correct or complete. That is strictly more than the suite can currently
assert about itself, and it should not be described as more than it is.

## Methods are yours, but registered

- Every method lives in `methods.md` before it is used.
- Every row names the method or methods that closed it.
- A new method states which existing rows it should be applied to
  retroactively, and the report tracks that backfill until it is done.
- A method whose closures cannot be given a negative control is not a
  verification method. It may still be useful as a triage aid; it does not
  close rows.

## Octave oracle rules

- Octave-based methods are the function-level oracle because COM Octave
  requires far less porting than the Python port and is available in the test
  environment; MATLAB is not.
- **A row carrying a classified semantic difference closes ONLY by an
  oracle-backed method.** See "Why the oracle is mandatory" below. This is the
  document's sharpest rule and the one that answers the defect that prompted
  it.

### The suite must run without Octave

The gate has to be runnable by someone who does not have Octave installed, or
the project cannot take outside contributions once it is public.

The split is already the repository's practice and should stay that way:
oracle values are **generated** with Octave, then **pinned as literals** in the
test with a `COM Octave` provenance comment naming what produced them. No test
file imports the oracle at run time, and the end-to-end Octave comparison is
opt-in behind an environment variable.

So there are two tiers. The **gate** runs anywhere, from pinned literals.
**Oracle generation and re-validation** run where Octave is installed, and
produce the literals plus the `oracle_id` that dates them. An implementation
that makes the gate require Octave has broken the public path; do not.
- **Octave is not MATLAB, and the differences are silent.** Known cases as of
  2026-09-22: `A\b` on a singular square matrix (MATLAB warns and returns
  `Inf`, Octave returns a minimum-norm least-squares solution);
  `interp1(...,'pchip')` outside the data range (MATLAB extrapolates, Octave
  returns `NA` unless `'extrap'` is given); `sum()` accumulation order (MATLAB
  is SIMD-blocked, Octave is strictly left to right, numpy is pairwise, so
  Octave is a third answer rather than a tiebreak).
- Record each in `octave_deviations.md` with the MATLAB line and the reason,
  **and give each one a probe that fails if the deviation stops being true.** A
  prose list of language differences rots exactly the way a stored verdict
  rots.
- Every deviation carries a computed **exposure set** (below). Rows in that set
  close only against a documented resolution, not against Octave alone.
- Top-level MATLAB results from the correlation corpus are the tiebreaker for a
  **system-level** disagreement. They are end to end, so in general they cannot
  localise a function-level one. Do not invoke the corpus where it cannot help.

### Exposure: which rows a change to the Octave build touches

When the Octave build changes, whether to fix a deviation or for any other
reason, the question that has to be answerable is *which functions are exposed*
so they can be re-run and updated.

Invalidating every oracle-closed row on any change to the tree is safe and
useless: a one-function patch would reopen all of them, and the re-run cost
would make the framework something people work around. Compute the exposure set
instead.

- `oracle.json` records a hash **per extracted function**, not only a hash of
  the tree. `tools/octave_oracle.py` already extracts a named function verbatim
  from the compat file, so this is available for free.
- On a change, the exposure set is every function whose extracted Octave body
  changed, plus the transitive closure of its **callers** in the reference call
  graph, which `tools/translation_coverage.py` already builds.
- Only rows in the exposure set go stale. `report.py` names the set, and names
  the change that produced it.

Worked example: patching `CDF_ev` in the Octave build exposes `CDF_ev` and its
callers `MLSE` and `MLSE_U1_c_178A`. Three rows re-run, not every oracle-closed
row in the inventory.

### A corpus disagreement is an escape

Octave-versus-MATLAB differences cannot be found by this framework; they are
found by the MATLAB correlation corpus, outside it. That does not put them
outside the process, it makes them escapes, and the escape rule applies in
full. Such a finding must produce:

1. an entry in `octave_deviations.md`, with its probe
2. the computed exposure set for that function
3. every row in that set reopened

An Octave/MATLAB disagreement that produces no deviation entry and no reopened
rows is the same process failure as any other escape that changes nothing.

### Why the oracle is mandatory, and why mutation is not enough

Mutation testing proves a test is **sensitive to the code**. It does not prove
the code is **right**, and the difference is exactly the defect that prompted
this document.

Suppose the port writes `sigma = np.std(x)` where MATLAB normalises by N-1, and
the test pins the value that the port produces. Apply the operator: flip to
`ddof=1`, the test fails, the row scores as discriminating. Green. The defect is
untouched, and the catalogue has certified a test that enshrines it. A test
written from a *reading* of the MATLAB will pin whatever the port does, and
mutation will confirm the test is awake while both are wrong together.

Only executing the reference separates them. So the chain is:

1. Row type (b) forces the construct to be classified at all.
2. Classification forces oracle closure for any row that touches it.
3. The oracle catches N versus N-1.
4. Mutation proves that oracle test does not later go to sleep.

Each link does one job and none substitutes for another. "Preferred" is not
enough for step 2, because a reading is precisely what produced `np.std`.

## Row status vocabulary

Computed: `open`, `pass`, `finding`, `unreachable`, `stale`.
Decided: `waived` (with reason and recorded approval), `upstream`.

`upstream` is for a defect in the reference MATLAB itself, reported to the COM
ad hoc. The 2026-09-22 pass produced nine, for example `applyDFEbk` guarding
`dfe_delta` on `nargin<6` when it is the seventh argument, and `force` calling
`WIENER_HOPF_MMSE`, which is not defined anywhere in 4p16p0. These are not
port failures and must not read as `finding`, but they are also not `pass`.
An `upstream` row names where it was reported and carries the port's chosen
behaviour with its justification.

Closed means `pass`, `unreachable`, approved `waived`, or `upstream` with a
recorded resolution. `finding` and `stale` are not closed.

## Waiver approval

A waiver is decided state, so it needs a record, not a conversation. Each
waiver carries: reason, who approved it, the date, and the condition under
which it should be revisited. A waiver with no recorded approval is pending and
counts as open. Pending waivers are listed individually in the report, never as
a count.

## Three-way disagreement rule

For rows verified by documented expectation vs Octave vs Python, the row closes
only when all three agree, or the disagreement is resolved in the row's notes
with MATLAB as ground truth. Octave and Python agreeing while the documented
expectation differs is a finding, not a pass.

`tests/test_documented_contracts.py` is the reference implementation: it states
the MATLAB rule from first principles, runs the reference under Octave, runs
the port, and carries a probe per primitive proving the conventions actually
differ. Copy its shape rather than inventing another.

## Escape rule (what makes this converge)

When a discrepancy is found by anything other than the inventory's own tests
(integration sweep, three-way COM run, casual read), log two items in
`escapes.md`:

1. the discrepancy, as a finding in that version's `findings.md`
2. the inventory gap: which row should have caught it, why it did not, and what
   row or method was missing
3. a `disposition`, which is how the catalogue grows (below)

Then add the missing rows or method and apply them across the whole inventory
for every supported version. **An escape that produces no inventory change is a
process failure; state that explicitly.**

### Disposition (on every finding and every escape)

Every entry in `findings.md` and `escapes.md` carries one of:

- `operator: <id>` naming an entry in `mutations.py`
- `not-mutable: <reason>`, the `manual_discrimination` path

This is what makes "a fix that produces no new operator is a process failure"
checkable. The report does not try to infer from git history which commits were
defect fixes, because that is exactly the judgment call that would make the
check unreliable. It reads the records the process already requires, and
verifies referential integrity in both directions: no entry without a
disposition, no disposition naming an operator that does not exist, no operator
citing an entry that does not exist.

A defect that is fixed without ever being written down escapes this, which is
what the `Finding:` commit trailer in Session discipline exists to catch.

This rule is the most valuable part of the document. It is the only mechanism
that makes the inventory grow toward completeness instead of toward comfort.

It has a twin in the mutation catalogue: an escape grows the inventory, and a
fix grows the catalogue. Together they mean a defect that gets past this
process once cannot get past it the same way twice. Neither works without the
other, because a complete inventory closed by sleeping tests reports zero.

## Coverage backstop

"Zero opens" means zero opens against the inventory. It does not prove the
inventory is complete. Three instruments, none of them line coverage:

1. **Function-level translation coverage** (`tools/translation_coverage.py`):
   per function, is it translated, is a value pinned, is a shape checked, and
   is it checked against the *executed* reference rather than a reading of it.
   The last is the one that catches a library default, and as of 2026-09-22 it
   stands at 57 of 146.
2. **Option-branch coverage** (`tests/test_option_coverage.py`): fails when a
   function's test never passes an option its implementation branches on.
3. **Inlined-copy differential** (`tests/test_inlined_copies.py`): every copy
   driven against its canonical, with a raise counted as a result, on the
   inputs where behaviour actually differs. Copies that nothing drives are
   pinned so the blind spot cannot grow.

Line coverage is deliberately excluded. It moves the wrong way in this port:
adding a correct helper lowers it, so it penalises the fix. Where a line-level
question matters, express it as an option branch or an unreachable row naming
the config option that would reach it.

These three ask whether the inventory is complete. The mutation catalogue asks
a different question, whether the tests behind it are awake, and neither
substitutes for the other. A row can be covered by an instrument above and
still be closed by a test that catches nothing.

## Onboarding a new version

When a version is added to `VERSION.json`:

- Derive its inventory from the nearest prior version's inventory.
- A row carries over as closed only if the MATLAB function body and its
  `py_impl` are unchanged between versions, **and** the version's branch inside
  that `py_impl` (where behaviour is keyed on `COM_MATLAB_VERSION`) has
  actually been exercised. A byte-identical file with an unexercised version
  branch does not carry over. Record the carry-over decision in `notes`.
- Functions new to the version get new rows. Functions removed are waived with
  reason "absent in <version>".
- `report.py` shows the new version with its true open count. Nothing is
  inherited silently.

## The approved report (the anchor for every "since")

Several checks below are phrased "since the last approved report". That phrase
is worthless unless approval is a thing in the repository rather than a thing
in someone's memory, so it is one:

`report.py --approve` writes `verification/approved/<UTC-date>-<sha>.txt`
containing the full report output and the git sha of the tree it was computed
from. "Since the last approved report" then means the literal range
`git log <sha>..HEAD`, resolved from the newest file in that directory.

Two properties follow, and both matter:

- Every "since" query is a concrete commit range, not a judgment about what has
  happened lately.
- An approved report is falsifiable after the fact. Anyone can check out that
  sha, rerun `report.py`, and compare against the stored text. A report that no
  longer reproduces at its own sha means the report is not deterministic, which
  is itself a finding.

If no approved report exists yet, "since" means the whole history and every
count starts from zero approvals. That is the honest starting state; do not
seed it with a backdated approval.

## Report

`report.py [--matlab-version V | --all]`, default from `VERSION.json`:

- per version: total, closed, open, finding, stale, unreachable, upstream,
  waived (pending, listed individually)
- every non-closed row with `id` and a one-line reason
- current `oracle_id` and the count of rows closed against older oracles
- escapes logged since the anchor sha
- rows whose `test_id` did not execute in the last suite run
- rows whose bound test survived an applicable mutation, named operator by
  operator, excluding entries in `equivalent_mutants.md`
- rows relying on `manual_discrimination`, listed individually
- catalogue health, as three referential-integrity checks rather than an
  inference over git history:
  - findings or escapes carrying no `disposition`
  - a `disposition` naming an operator that is not in `mutations.py`
  - an operator citing a finding or escape that does not exist
- commits since the anchor sha that touch `com_functions/fn/*/py_impl.py` and
  carry no `Finding:` trailer

The mutation lines are the ones that would have caught 2026-07: that ledger's
159 rows all cited evidence and test files, and 27 of the functions it cleared
had defects a mutation would have exposed.

`report.py` must be deterministic: same tree, same output. It reads the suite's
results rather than re-deriving conclusions, and it pins the inputs it depends
on by hash.

## How to answer "are there any opens?"

Run `tests\run_all.ps1`, then `report.py --all`, and paste the output. Every
"since" in the list below resolves against the anchor sha of the newest file in
`verification/approved/`.

If every supported version shows zero open, zero finding, zero stale, zero
pending waivers, zero rows with an unexecuted `test_id`, zero rows that
survived an applicable mutation, zero findings or escapes without a
`disposition`, zero dangling operator references, zero `py_impl.py` commits
without a `Finding:` trailer, and no escapes since the anchor, say "no opens"
and stop. Otherwise the list is the answer; do not summarize or soften it.

A non-zero `manual_discrimination` count is not an open, but it is reported
every time, because it is the part of the process still resting on someone's
word.

## Session discipline

- `audit_state.json` per version for resumability.
- Audit sessions are read-only against `py_impl.py` and `sicopr.py`. Fixes are
  separate sessions, one finding per gate.
- A commit touching `com_functions/fn/*/py_impl.py` carries a `Finding: <id>`
  trailer, or `Finding: none (<reason>)` for a genuine non-fix change such as a
  refactor, a new function or a comment. This is the one new convention in this
  document, and it exists to catch the single case the disposition checks
  cannot see: a defect fixed without ever being written down.
  **Check it in the suite, not in a git hook.** A hook is bypassable with
  `--no-verify`, which CONTRIBUTING forbids but cannot prevent, and the suite
  is where the gate actually lives. Scope the check to that one path and to
  commits after the anchor sha.
- Closed rows are re-audited only when an escape, an oracle change, or a new
  method reopens them, and the reason is recorded.
- `sicopr.py` is generated; never edit it directly. Edit
  `com_functions/fn/<name>/py_impl.py` and re-run `assemble_sicopr.py`.
- Call another translated function with an import, not a hand-written copy, so
  new row type (d) entries are not created by accident.
- No em dashes in any generated file.
