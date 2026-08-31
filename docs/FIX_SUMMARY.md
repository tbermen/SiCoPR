# Fix ledger — every correctness fix applied to COM Python

**One-stop history.** Root-cause investigations come and go and get their own
documents; this file keeps a permanent one-line-plus-summary entry for every fix
that changed a number, with a pointer to the detail. If you want to know *what
has been fixed and what it bought*, read this file and nothing else.

**Adding an entry.** When a fix lands, add a row to the ledger and a short
subsection under the matching date. Record what it *bought* — a fix with no
measured effect is a claim, not a result — and link the detailed write-up rather
than reproducing it. Keep entries even when the underlying investigation doc is
later deleted.

> Scope: fixes to the engine (`com_functions/fn/*/py_impl.py`, from which
> `sicopr.py` is assembled) and to the settings the engine is run with. Tooling,
> reporting and documentation changes are not tracked here, with one exception
> noted below.

**How this file is organised.** Four parts, in this order:

1. **Current correlation status** — where the port stands against the MATLAB
   reference today. If you read one thing, read this.
2. **The ledger** — one numbered row per fix that changed a number, oldest first,
   with what it *bought*. This is the index; everything after it is detail.
3. **Dated sections** — one per ledger date, expanding the rows into root cause,
   evidence and what was rejected. Two entries here changed no number but were
   still engine changes, and say so.
4. **Test ROI**, then an **appendix** pointing at the archived earlier fix pass.

Not everything here is a success. Two sections record investigations whose fix
was **tried and reverted** — the DER residual in the Noise stage, and the
`argsort` blanket edit — because knowing what was ruled out is worth as much as
knowing what landed.

---

## Current correlation status

208 MATLAB reference cases (`com_ieee8023_4p15p0`, 26 channels × 4 packages ×
with/without crosstalk), each condition run on the configuration its own
MATLAB reference was produced with — the pairing the MATLAB author confirmed
on 2026-08-24 ([`TXFFE_SWEEP_ROOT_CAUSE.md`](TXFFE_SWEEP_ROOT_CAUSE.md)).

| | |
|---|---|
| FOM bit-exact | **208 / 208** |
| COM bit-exact | **208 / 208** |
| sampling phase (`itick`) exact | **208 / 208** |
| max \|ΔCOM\| | **3.29e-14 dB** |
| pass/fail disagreements at 3 dB | **0** |

Starting point before any of the fixes below: **max \|ΔCOM\| = 6.256 dB**.

---

## The ledger

| # | date | fix | class | found by | bought | commit |
|---|---|---|---|---|---|---|
| 1 | 2026-08-14 | `z_p` transpose for RX / NEXT / FEXT | config parse | column-ranked error | RX package built from a matrix row → ~15 dB spurious loss | `git log --grep="Fix 8 engine defects"` |
| 2 | 2026-08-14 | Insertion-loss fit solved at effective rank 2 of 4 | numerics | column-ranked error | `lstsq` silently truncated rank; half the fit basis restored | `git log --grep="Fix 8 engine defects"` |
| 3 | 2026-08-14 | RxFFE floating-tap array sized by tap COUNT, not SPAN | allocation | column-ranked error | floating taps past index 23 were being discarded | `git log --grep="Fix 8 engine defects"` |
| 4 | 2026-08-14 | `get_TDR` `tfstart` index base | 1- vs 0-based | column-ranked error | Z11est/Z22est 1.4e-2 → 4e-15 | `git log --grep="Fix 8 engine defects"` |
| 5 | 2026-08-14 | `get_TDR` `fctrx` initialisation | translation | column-ranked error | ERL11/ERL22/ERL 4e-1 → ~1e-15 | `git log --grep="Fix 8 engine defects"` |
| 6 | 2026-08-14 | Package die network truncated to 1 of 3 LC sections | translation | column-ranked error | restored ~15 ps of die delay (a 53-sample pulse shift) | `git log --grep="Fix 8 engine defects"` |
| 7 | 2026-08-14 | Cursor index base in `optimize_fom` (audit B16-D20) | 1- vs 0-based | audit + #6 | was reverted early as "worse"; it and #6 were **compensating** | `git log --grep="Fix 8 engine defects"` |
| 8 | 2026-08-14 | `process_sxp` leaked a TDR-only setting into the whole run | **by-reference** | sign of the bias | **the systematic FOM bias** — Python low on 199/208 (95.7%) | `git log --grep="Fix 8 engine defects"` |
| 9 | 2026-08-19 | `BEST.PSD_results` aliased a struct MATLAB copies by value | **by-reference** | stage-6 noise gap | COM bit-exact 135 → **170**, pass/fail flips 2 → **1** | `git log --grep="BEST.PSD_results aliased"` |
| 10 | 2026-08-20 | Four latent fidelity defects (unstable sort ×2, abort-path leak, `_findbankloc` stubs ×2) | mixed | itick investigation | **inert on this corpus by design** — real under other settings | `git log --grep="latent MATLAB-fidelity defects"` |
| 11 | 2026-08-20 | Off-by-one in the ADC-clip sampling phase | 1- vs 0-based | variance decomposition | COM bit-exact 170 → **198**, pass/fail flips 1 → **0** | `git log --grep="ADC-clip sampling phase"` |
| 12 | 2026-08-20 | `Overwrite_Min_Radius` honoured in both version paths | silent discard | Tx FFE investigation | a config setting it under 4p15p0 had it silently discarded | `git log --grep="all 10 Tx FFE cases reproduce"` |
| 13 | 2026-08-21 | `TXLE_taps_1..4`, `Pre2Pmax`, mixed-mode ERL absent from results.xlsx | reporting | user review | columns claimed to agree were never exported | `git log --grep="Tx FFE tap columns were unpopulated"`, `git log --grep="refresh the as-supplied set"` |
| 14 | 2026-08-21 | `BEST.ctle` / `BEST.G_high_pass` used 1-based as 0-based in `OptFom_Update_BEST_Post_Optimize` | **1- vs 0-based** | corpus sweep crash | reporting-only, but it **hard-crashed** any run whose winning CTLE was last in the list | (this commit) |
| 15 | 2026-08-22 | `floating_tap_locations` had a **producer-dependent base** — 0-based from `floatingDFE`, 1-based from `MMSE`/`force` | **1- vs 0-based** | index registry | the FDFE cursor-time vector was one UI early under `Floating_DFE` | (this commit) |
| 16 | 2026-08-22 | `nui = round(len/M)` used Python's banker's rounding where MATLAB rounds half away from zero | **rounding** | Noise-stage chart | **closed the last COM miss** — COM 207 → **208 / 208**, max \|ΔCOM\| 0.0076 → **3.3e-14** | (this commit) |
| 17 | 2026-08-27 | adaptive-search radius floor forced to `1` on the 4p15p0 path, where the reference behaves as the 4p16p0 rule | **inferred setting** | the last open question | the `--min-radius 2` switch is **no longer needed** to reproduce the corpus | (this commit) |
| 18 | 2026-08-28 | `findbankloc` indexed `ndiff` with bank-member positions; MATLAB grows an array on out-of-range assignment, NumPy raises | **index space** | a flaky gate run | **IndexError on 4.4% of inputs** wherever floating DFE taps are used; inert on this corpus, real under other settings | (this commit) |

#17 is a settings deduction rather than a translation defect, and is written up in
[`MIN_RADIUS_ASSUMPTION.md`](MIN_RADIUS_ASSUMPTION.md) — including what it does
not establish and what would overturn it. It is the **second** case of the supplied
configuration snapshots post-dating the run that produced the reference results;
the Tx FFE grid was the first. When a config and its own results disagree, suspect
the config.

**Five of the eighteen are the same root class**: MATLAB assigns structs **by
value**, Python binds a **reference**. #8, #9 and part of #10 are direct
instances; #3 and #6 are the same failure to carry a whole structure across a
boundary. This is the single most productive thing to check first in this port.

---

## 2026-08-14 — the eight defects found by the 208-case correlation (`git log --grep="Fix 8 engine defects"`)

Max \|ΔCOM\| 6.256 dB → ~0.18 dB. Detail in
[`../MATLAB_Correlation_Review.md`](../MATLAB_Correlation_Review.md) §2–§4.

**Method note worth keeping.** Every one of these was found by ranking all
comparable output columns by relative error and letting the data localise the
fault. Reading code to guess causes failed repeatedly.

#8 is the one to remember. `process_sxp` set
`OP.impulse_response_truncation_threshold = 1e-5` on the shared `OP`, and the
MATLAB source says in a comment at L9311 that it is "Only for TDR not returned
out of process_sxp function" — true under by-value semantics, false under
Python's. A 100× tighter truncation retained excess impulse-response tail,
lengthened the pulse response, inflated residual ISI and biased FOM low on
95.7% of cases, while the whole unit suite stayed green.

#7 is the cautionary one: applying it alone made agreement 5–20× *worse*, so it
was reverted as wrong. It was correct all along — it and #6 were compensating.
**A fix that makes things worse is not necessarily the wrong fix.**

## 2026-08-19 — `BEST.PSD_results` aliasing (`git log --grep="BEST.PSD_results aliased"`)

`BEST.PSD_results = THIS.PSD_results` copies by value in MATLAB and binds a
reference in Python, while `get_PSDs` mutates its `result` argument in place and
`optimize_fom` reuses one object per CTLE block. The reported noise therefore
came from the **last tick swept (+24)**, not the winner's.

**Why it hid**: `ts_sample_adj_range` is [−24, 24] and `samples_per_ui` is 32,
and **24 ≡ −8 (mod 32)** — so the stale arrays were *correct* exactly when the
winning tick was −8. The data showed `sgm_TX` exact on 10 of 208 cases, all 10
at `itick = −8`, with the error growing with distance from −8.

`sgm_TX` went from 2/208 exact to 42/48 on the re-run subset. This fix does
**not** move `itick`; that was a separate cause (#12 and the Tx FFE settings).

## 2026-08-20 — four latent defects (`git log --grep="latent MATLAB-fidelity defects"`)

All four are **unreachable in the 208-case configs**, verified: the corpus
re-ran bit-identical on `com_py`, `fom_py`, `itick_py` and `tick_match` across
all 208. That is the point — they are real under other settings and the corpus
cannot catch a regression in them, so they were fixed deliberately rather than
left for a future config to trip over.

1. `np.argsort` defaults to quicksort, which is **not stable**; MATLAB's `sort`
   is. Live under `TS_SRCH_MODE='middle'`. Same defect inside `findbankloc`'s
   `argsort(-ndiff)`, where ties are not hypothetical — the ISI tail is mostly
   zeros.
2. By-reference leak on `OptFom_Calc_Noise`'s abort path — the caller saw the
   *aborted* tick's values where MATLAB sees the last successfully scored one.
3. `MMSE._findbankloc` and `force._findbankloc` were "simplified" stubs picking
   the highest-power non-overlapping banks. The real routine ranks bank starts
   by `ndiff = h0n - h1n` and runs a `badV`/`goodV` admissibility loop that can
   reject the strongest bank outright. A faithful port already existed and had
   simply never been wired into these two copies.

**Deliberately not done at the time**: ~25 other `argsort` sites, several using
`argsort(x)[::-1]`, which inverts tie order relative to a stable descending sort
even with `kind='stable'` added. They needed individual checks against the MATLAB
they came from, not a blanket edit.

> **Superseded 2026-08-28.** Those checks were done and all **39** `argsort` calls
> now pass `kind='stable'`, with `tests/test_sort_stability.py` failing the run on
> any new unstable site. See ledger row 10a in *Test ROI* below. This paragraph is
> kept because the reasoning for not doing it as a blanket edit still applies to
> the next person who finds a sort.

## 2026-08-20 — ADC-clip sampling phase (`git log --grep="ADC-clip sampling phase"`)

Detail in [`COM_PDF_RESIDUAL.md`](COM_PDF_RESIDUAL.md). The last engine-level
disagreement, and the only one large enough to move a case across the 3 dB
threshold.

MATLAB's `mod(t_s-1,M)+1` takes a **1-based** `t_s`; Python's `fom_result.t_s`
is already 0-based, so `(t_s - 1) % M` sampled **one sample early**. The port was
already inconsistent with itself — `get_PSDs` does the same decimation correctly,
and that inconsistency is what confirmed the reading.

Localised by **convolution adds variances**: `sigma_before_clip² = σ_signal² +
σ_combined²`, so measuring Python's `σ_combined` and subtracting isolated the
signal PDF, which has exactly one input — the sampled pulse.

A lead recorded as **wrong** so it is not re-followed: the non-stable
`argsort(...)[::-1]` in `get_pdf_from_sampled_signal` is *not* this defect. A
stable descending sort gives a bit-identical sigma.

## 2026-08-20 — `Overwrite_Min_Radius` version path (`git log --grep="all 10 Tx FFE cases reproduce"`)

Found while resolving the ten Tx FFE cases. The keyword was read only under
4p16p0, so a config setting it while emulating 4p15p0 had it **silently
discarded** — the same silent-config-discard class as the Tx FFE settings
mismatch itself. Guarded since by `tests/test_config_search_space.py`.

The accompanying settings finding is not an engine fix and is recorded in
[`TXFFE_SWEEP_ROOT_CAUSE.md`](TXFFE_SWEEP_ROOT_CAUSE.md): the supplied configs
pin Tx FFE `c(-1)` to a single zero, so Python searched 1 candidate where MATLAB
searched ~1584.

## 2026-08-21 — results.xlsx columns (`git log --grep="Tx FFE tap columns were unpopulated"`, `git log --grep="refresh the as-supplied set"`)

Not an engine fix, but listed because it invalidated a *claim*, which is worse
than a wrong number: `TXLE_taps_1..4` was reported as agreeing with MATLAB on
all 208 cases while never being exported at all, and the mixed-mode ERL variants
and `Pre2Pmax` were described as "not computed" by a hand-maintained note long
after they were. All are now populated and bit-exact against MATLAB, and guarded
by `tests/test_export_columns.py`, which fails when a column the review compares
is left blank.

---

## 2026-08-21 — `BEST.ctle` index base in post-optimize

`THIS.ctle_index` and `THIS.g_LP_index` are set **1-based to match MATLAB**, and
`OptFom_Update_Best_Setttings` copies them straight into `BEST.ctle` and
`BEST.G_high_pass`. `OptFom_Update_BEST_Post_Optimize` then indexed
`param.CTLE_fz` / `f_HP` / `g_DC_HP_values` with them under a comment claiming
`# 0-based`, reading one entry too high on every run since the initial commit.

**Why it survived the 208-case correlation.** The mis-indexed values feed only
`OptFom_Plot_Best_Results`; the COM path recomputes `ctle_gain` inside
`optimize_fom`, which *does* convert. Verified rather than assumed: re-running
`wXtalk_T1_R16` after the fix left **all 218 numeric output fields
bit-identical**, `COM_dB` included. No published result changes.

**Why it surfaced now.** An over-long CTLE list turns the error into a silently
wrong lookup instead of an `IndexError`; it only raises when the winning CTLE is
the **last** in the list. The 7-channel sweep runs `--max-ctle 3`, which
truncates the list to exactly the loop count, and the August engine fixes moved
the winning CTLE onto the last entry. It crashed on channel 1.

Guarded by two checks in `tests/test_cursor_indexing.py` — one that the call
survives a winner at the last index, one that the pole/zero actually used is the
selected entry's. The first version of that test passed against the reintroduced
bug because a missing `OP.Butterworth` raised before the lookup was reached;
mutation testing caught the vacuous pass, and the fixture now disables all three
filters.

---

## 2026-08-22 — index-base conformance checking (`tests/test_index_base.py`)

Not a fix; a guard against the largest recurring class. Four of the fourteen
entries above are index-base errors, and they share a shape that defeats
ordinary testing: a 1-based value used as a 0-based subscript reads the **wrong
element** while the array is long enough, and only raises `IndexError` when the
index lands on the last entry. #14 carried that from the initial commit until a
`--max-ctle 3` sweep happened to put the winning CTLE last.

Approaches measured and rejected before settling:

| approach | result |
|---|---|
| `int` subclass raising in `__index__` | **no enforcement at all** — CPython/numpy use a C fast path; `big[OneBased(3)]` silently returned element 3 |
| mypy / pyright | defeated by the data model — 177 `SimpleNamespace` constructions, 462 attributes across `param`/`OP`/`result`/`BEST`/`THIS`, all `Any` |
| contradiction detection (field used both ways) | 1 of 4 recall — needs the bug to coexist with a correct use |
| **declaration conformance** | **3 of 4**, adopted |

The registry declares each index field's base *once*, with the evidence for it,
and three rules prove every use conforms: **A** a 1-based field reaching a
subscript unconverted, **B** an explicit `− 1` on a 0-based field, **C** an
index-shaped struct attribute with no declared base.

Rule C is what makes it scale — a new feature cannot introduce an undeclared
index without failing the run. It caught three on its first execution
(`DFE_taps_i`, `start_max_idx`, `end_max_idx`), all since declared.

Replayed against the historical versions in git: the version before the 8-defect fix flags `cursor_i` in
`optimize_fom` (#7); the version before the ADC-clip fix flags `t_s` in
`Apply_EQ` (#11) and `ctle`/`G_high_pass` (#14); the version before the
`BEST.ctle` fix flags #14; HEAD is clean. The miss is #4,
which is not a base error — a valid index applied to the wrong array frame.

It runs over the assembled `sicopr.py`, so it covers all inlined copies of a
function at once, and it sees paths no test executes.

**A real finding came out of building the registry** — see #15 below, which also
corrects how it was first described here.

---

## 2026-08-22 — `floating_tap_locations` producer-dependent base

Found by writing the index registry. The field had **no single base**: three
functions produce it and they disagreed.

| producer | Python base | MATLAB |
|---|---|---|
| `floatingDFE` | **0-based** ("all using 0-based indices") | 1-based (`tap_loc` indexes `hisi`) |
| `MMSE` | 1-based (`idx + RxFFE_cmx + 1`) | 1-based (ML 2576) |
| `force` | 1-based (`# idx is 1-based`) | 1-based |

Both consumers sit behind `if param.Floating_DFE`, so they read the
**`floatingDFE`** value — the 0-based one:

- `BEST.FDFE_taps_mV = BEST.DFE_taps_mV[floc]` — 0-based subscript of a 0-based
  value, which is **correct**, matching MATLAB's 1-based `DFE_taps_mV(floc)` at
  ML 4143.
- `BEST.sampled_sbr_fdfecursors_t = ((cursor_i + 1)/M + floc) * ui` — MATLAB
  4133 is `(cursor_i/M + floc)*ui` with `floc` 1-based, so Python was **one UI
  early**. That was the defect.

> **Correction.** The first version of this entry, and the commit message that
> introduced the checker, named the wrong line: they said the `DFE_taps_mV`
> subscript at `sicopr.py:5865` was defective. It was not — its producer is
> 0-based, so the raw subscript was right. The time vector was the broken one.
> The registry flagged the field correctly; the initial reading of *why* was wrong.

Fixed by normalising at the single boundary where `floatingDFE`'s output becomes
the shared field (`OptFom_Compute_DFE`), so the field is **1-based whatever
produced it** — matching MATLAB and the other two producers. The consumer now
converts for its subscript, and the time vector matches ML 4133 unchanged. The
`MMSE` comment that claimed "in Python we keep as 0-based", directly above code
adding `+ 1`, is corrected — that contradiction is what made the base ambiguous.

**Verification.** All 249 output fields bit-identical on re-run cases, and
`floating_tap_locations` = [22…29] in both Python and MATLAB on
`wXtalk_T1_R16` — so the RxFFE/MMSE path is exercised by the corpus and its
1-based convention is confirmed against the reference. The `Floating_DFE` path
is not reachable by any of the 208 configs, so the time-vector correction rests
on matching ML 4133 line-for-line rather than on measured data. Both fields are
write-only in the port (MATLAB uses them for a stem plot), so nothing downstream
moves either way.

---

## 2026-08-22 — oracle-free guards, layers 1 and 2

Built from the review of how each correlation-found defect could have been
caught **without MATLAB results**. The measured starting point: all 156
per-function tests cite MATLAB, but only **18** use an oracle independent of
reading MATLAB. When the misreading *is* the bug, the test encodes it — which is
literally what happened to #1, whose commit records that "the unit-test fixtures
were written transposed, so they encoded the bug."

### Layer 1 — snapshot isolation (`tests/test_snapshot_isolation.py`)

Targets #9. The invariant needs no MATLAB data, only MATLAB's *semantics*:
assignment copies, so **a best recorded at candidate N cannot be altered by
candidate N+1**. The test records a best, mutates every mutable field of the
live candidate in place, and asserts the record did not move — so it *discovers*
which fields alias instead of relying on a maintained list.

First run found **13 aliased fields** in `OptFom_Update_Best_Setttings` and 7 in
the EQ-failed variant (`PSD_results` correctly absent — it was already copied).
`_value_copy` was generalised to be recursive and type-general, and applied to
all of them. Copying is now uniform rather than case-by-case, because "this
object happens never to be mutated today" is not a property anyone should have
to re-verify on every edit.

**Verified inert:** all 249 output fields bit-identical on four re-run cases
(~1000 comparisons).

### Layer 2 — behavioural coverage of inlined copies

Targets #10c. `test_inlined_copies.py` already compared copies behaviourally,
but only for the 10 functions that had synthetic input factories — **50 of 178
copies, 28%**. `findbankloc`, where #10c lived, had no factory, so its four
copies were never driven.

Ten more factories, plus arity-adaptive driving (copies legitimately take fewer
arguments than the canonical), took coverage to **113 of 178, 63%**.

That surfaced **12 divergences**, every one checked against sicopr.py's `_wired_*`
partials before being recorded. None is live, but two findings are worth having:

- The `get_TDR` Bessel fallback stub is not subtly wrong — it hardcodes the
  4th-order coefficients **without reversing them**, giving DC gain **105
  instead of 1**, and returns a magnitude where MATLAB returns a complex
  response (ML 1033-1040 uses `fliplr`). Harmless only for as long as the
  injection at `sicopr.py:132` holds.
- `findbankloc` has the **producer-dependent base** disease of #15: the
  canonical returns 0-based, the `MMSE` and `force` copies return 1-based
  because their callers mirror ML 2576 arithmetic verbatim. Each is locally
  correct; the hazard is that one name now means two bases. Not changed —
  resolving it needs an oracle for the floating-tap paths, which the 208 configs
  do not exercise.

---

## 2026-08-22 — the DER residual in the Noise stage: mechanism found, fix rejected

The "6 Noise" pipeline stage sits at 95.1% of columns exact where every other
stage is at 99.7% or better. Investigated on the reading that a stage short of
100% is a defect waiting to be root-caused. **The mechanism was found and
proven. It is not a defect, and the fix attempted for it was reverted as a net
regression.** Both halves of that are worth recording.

### What the shortfall is made of

Measured per column across the corpus:

| column | inexact | median relative error | direction |
|---|---|---|---|
| `DER_MLSE` | 154 / 204 | 1.36 % | Python low on **133 of 154** |
| `DER_DFE` | 125 / 208 | 1.45 % | Python low on **123 of 125** |
| `DER_thresh` | 4 / 208 | 2.4e-4 | low on 4 of 4 |
| `sgm_isi` chain | 4 / 208 | 1.7e-4 | see the separate item below |

A 98% one-directional split at a stable ~1.5% median is the signature that
located the `process_sxp` leak, so it was treated as a bias.

### The mechanism

`DER_DFE = CDF_ev(A_s, PDF, CDF)`, and `CDF_ev` is a **discrete bin lookup**:

```matlab
index = find(PDF.x >= -val, 1, 'first');   % ML 1144-1146
```

`BinSize` is derived from `A_s` such that `A_s / BinSize == 1000` **exactly**, so
`-A_s` lands precisely on bin -1000 on essentially every case. The `>=` is
therefore decided on an exact tie, and the tie is broken by the last bits of the
grid. Proven at bit level on `woXtalk_T1_R02`:

```
(-A_s)              = -0x1.11f683ba71df9p-8
Python x[i]         = -0x1.11f683ba71df9p-8   bit-identical -> >= TRUE  -> bin i
start + k*BinSize   = -0x1.11f683ba71e00p-8   7 ulp lower   -> >= FALSE -> bin i+1
```

and MATLAB's reported `DER_DFE` equals this port's `CDF[i+1]` **to 11
significant digits**. One bin of a decaying CDF tail is ~1.5%, which is the
entire observed error.

`A_s` itself is never bit-identical to MATLAB (208/208 differ, max 2.3e-11
relative) — ordinary double-precision accumulation through the pipeline. So
which side of the tie a case lands on is decided by the 11th digit of `A_s`.

### The fix that was tried, and why it was reverted

MATLAB has two grid idioms that are algebraically identical and different in
floating point:

```matlab
p.x = (p.Min*p.BinSize : p.BinSize : pMax*p.BinSize)   % ML 2247, 5808, 5828
pdf.x = (pdf.Min:-pdf.Min)*binsize                     % ML 5700, 5954, 8835
```

The port writes both as `arange(...) * BinSize`. Reproducing the first as naive
accumulation (`start + k*step`) at all 25 scaled-colon sites — the canonical
`conv_fct`, `conv_fct_MeanNotZero` and `Init_PDF_Fast` plus every inlined copy —
**made agreement worse**, measured over 52 cases:

| column | before | after |
|---|---|---|
| `DER_DFE` exact | 12 | **9** |
| `DER_MLSE` exact | 8 | **3** |
| COM / FOM / VEO / VEC / `sgm_*` | unchanged | unchanged |

Reverted. The useful negative result: **MATLAB's colon is more accurate than
naive `start + k*step`** — it is endpoint-corrected — so the port's existing
integer form is the *closer* approximation of the two, and emulating the colon
as accumulation is wrong in the other direction. Matching MATLAB here would
require reproducing its colon bit-exactly, which cannot be verified without
MATLAB.

### Conclusion

The `DER_*` columns are **quantisation-limited, not wrong**. Everything in this
stage that is not a bin lookup — `sgm_*`, `sigma_before_clip`, `peak_clip`, and
downstream COM / FOM / VEO / VEC / `itick` — agrees exactly. Chasing the Noise
stage to 100% on the current metric means chasing which side of an exact tie the
11th digit of `A_s` falls on, and that is not a property the port can control.

The honest reporting change is to score `DER_*` as agreeing when Python's value
equals MATLAB's within **one CDF bin**, which is a physically meaningful
statement about a discrete lookup. That has NOT been applied, because it should
not be applied while a real residual is still outstanding in the same stage —
see the `sgm_isi` item, which is a genuine target and the last COM miss.

---

## 2026-08-22 — `nui` rounding: the last COM miss

Found by asking why the **"6 Noise"** pipeline stage sat at 95.1% when every
other stage was at 99.7% or better — on the reading that a stage short of 100%
is a defect waiting to be root-caused. It was.

`get_pdf` computes

```python
nui = round(len(residual_response) / M)
```

which sets the row count of the `vs` sampling matrix and therefore how many ISI
samples enter the residual-ISI PDF. **Python's `round()` is banker's rounding;
MATLAB's rounds half away from zero.** Instrumenting every `round()` call in a
full run found **exactly one of 98,145** on a tie — `2360.500000`, MATLAB 2361,
Python 2360. One row lost, one ISI sample dropped, `sgm_isi` low, and the deficit
dilutes out through `sgm_isi_xt`, `sgm_Ani__isi_xt_noise` and
`sigma_before_clip`.

Reached only 4 of 208 case-instances (`T1_R23`, `T4_R10`, both crosstalk
conditions) because the tie needs the response length to be an odd multiple of
`M/2` — a function of channel **and** package, so each affected channel appears
in 8 cases and diverges in 2.

**The audit had already found this site and dismissed it.**
`docs/AUDIT_FINDINGS.md` lists `nui=round(len/M)` among the bare-`round` sites
and argues they "differ from MATLAB only for exact half-integer inputs
(measure-zero for continuous data)". True of continuous inputs; false here,
because `len(residual_response)` and `M` are both **integers**, so the quotient
is a rational that lands on `.5` exactly. Second audit verdict this correlation
has overturned.

### Result

| | before | after |
|---|---|---|
| COM bit-exact | 207 / 208 | **208 / 208** |
| max \|ΔCOM\| | 0.007623 dB | **3.3e-14 dB** |
| `wXtalk_T4_R10` output columns exact | 204 / 217 | **217 / 217** |
| Noise-stage columns exact | 95.1 % | **95.6 %** |
| COM-stage columns exact | 96.6 % | **100 %** |

FOM and `itick` are unchanged, as they must be — the defect is downstream of the
equalizer and the sampling-point choice.

### What is left in the Noise stage, and why it stays

After this fix, **every inexact column-instance in the Noise stage is `DER_MLSE`
or `DER_DFE` — 275 of 275, 100%.** All 29 other noise columns are exact on all
208 cases. Those two are `CDF_ev` bin lookups landing on an exact tie, proven
quantisation-limited in the section above; they are not chasable. The Noise bar
will not reach 100% on the current metric, and that is now a fully explained
end state rather than an open question.

---

## Test ROI — what the added guards have actually caught

Added because the same defect classes kept recurring. This section exists to let
that investment be judged on evidence rather than on the assumption that more
tests must help, so it records the **direct yield honestly, including where it is
low**.

### Direct yield: new defects found by the new tests

| test | new defects found | what else it produced |
|---|---|---|
| `test_index_base.py` | **1** — #15, `floating_tap_locations` producer-dependent base, a real one-UI error in the FDFE cursor-time vector | 3 undeclared index fields forced into declaration (`DFE_taps_i`, `start_max_idx`, `end_max_idx`) |
| `test_snapshot_isolation.py` | **0 live** | 20 aliased fields found and detached (13 in `OptFom_Update_Best_Setttings`, 7 in the EQ-failed variant); all verified numerically inert today |
| `test_inlined_copies.py` (coverage 28% → 63%) | **0 live** | 12 divergences documented, incl. the `get_TDR` Bessel stub at DC gain 105 instead of 1, and `findbankloc` returning two different bases under one name |
| `test_export_columns.py` | 0 (the miss was found by review) | now guards #13 against recurrence |

**One new defect.** That is the honest headline, and on its own it is a thin
return for the effort.

### Indirect yield: regression coverage of the existing ledger

The stronger case is what would now be caught *automatically* if it were
reintroduced — i.e. whether the next cycle needs the MATLAB corpus to find these
classes again. Verified by replaying the rules against the historical `sicopr.py`
in git, or by reintroducing the defect and watching the test fail:

| # | defect | caught today? | how verified |
|---|---|---|---|
| 1 | `z_p` transpose | **✓** | `test_structural_invariants` — non-square shape AND square-input values; mutation-verified |
| 2 | `lstsq` rank truncation | **✓** | `test_analytic_recovery` — input built from the fit basis; mutation-verified |
| 3 | RxFFE array count vs span | **✓** | `test_structural_invariants` — sizing rule swept, widening term asserted; mutation-verified |
| 4 | `get_TDR` `tfstart` frame | **✓** | `test_analytic_recovery` — constructed impedance profile; mutation-verified |
| 5 | `get_TDR` `fctrx` init | **✓** | `test_analytic_recovery` — known energy past the gate; mutation-verified |
| 6 | die network 1 of 3 sections | **✓** | `test_structural_invariants` — 2x1 vs 2x3 `C_d` must differ; `make_full_pkg` is drivable after all; mutation-verified |
| 7 | cursor index base | **✓** | replay: flags `cursor_i` in `optimize_fom` on the pre-8-defect version |
| 8 | `process_sxp` leak | **✓** | `test_reference_leaks`, mutation-verified |
| 9 | `BEST.PSD_results` aliasing | **✓** | reintroduced; snapshot test fails naming `PSD_results` |
| 10a | unstable sort | **✓** | `test_sort_stability` — all 39 `argsort` calls made `kind='stable'`; lint fails on any new unstable site; mutation-verified |
| 10b | abort-path leak | **✓** | `test_abort_path_leaks` — AST lint for writes to a caller struct before an early return; mutation-verified by restoring the removed `THIS` writes |
| 10c | `_findbankloc` stubs | **✓** | now drivable; flags the `MMSE` and `force` copies |
| 11 | ADC-clip sampling phase | **✓** | replay: Rule B flags `t_s` in `Apply_EQ` on the pre-ADC-clip version |
| 12 | `Overwrite_Min_Radius` discard | **✓** | `test_config_search_space` — no config keyword may be read only inside a `_v416` gate; mutation-verified by re-gating the read |
| 13 | results.xlsx columns | **✓** | mutation-verified |
| 14 | `BEST.ctle` index base | **✓** | replay + mutation |
| 15 | `floating_tap_locations` | **✓** | the test that found it |
| 16 | `nui` banker's rounding | **✓** | `test_integer_ratio_rounding` — AST lint for `round()` on an int/int ratio; found and fixed a second live site (audit B11-D15); mutation-verified |
| 17 | adaptive radius floor | **✓** | `OptFom_Adaptive_Local_Search/test_verify` — three tests incl. one that fails if the 4p15p0 path is 'corrected' back to a forced 1 |
| 18 | `findbankloc` index space | **✓** | `findbankloc/test_verify` — seeded regression over three failing draws; mutation-verified |

**20 of 20 rows caught automatically** (2026-08-29). The table has 20 rows for 18
ledger entries because #10 bundled three separate defects and is split here.
Before any of this work the number was effectively 2 (`test_reference_leaks` and
the config keyword check); everything else required a MATLAB run to notice.

Every row was verified by reintroducing the defect and watching the guard fail,
then reverting — not by inspection. Two of the guards found something while
being built: the rounding lint found a second live site (audit B11-D15), and the
sort lint found 28 unstable calls where the assumption "ties are rare" turned out
to be false in 81% of calls.

> **Correction.** An earlier revision of this line read "14 of 15, 1 partial, 0
> not" while the table it summarised still showed 10a and 10b as ✗. The count was
> wrong in the direction that flattered the work, which is the direction that
> matters.
>
> **Both are now closed.** 10a first; 10b since — `OptFom_Calc_Noise` returns on
> the abort path with `THIS` untouched (py_impl L101) and commits `h_J`,
> `sigma_TX`, `ISI_N`, `sigma_N` and `total_noise_rms` only at the end
> (L153–157), which is where MATLAB commits them. `test_abort_path_leaks.py`
> holds the line.

## 2026-08-29 — the config file stayed locked after being read

Not a correctness defect — no number moves — but an engine change, so it belongs
in this ledger rather than only in a commit message.

`read_ParamConfigFile`'s Excel loader (`__load_excel`) opened the workbook with
`openpyxl.load_workbook(..., read_only=True)` and never closed it. `read_only`
keeps the underlying zip handle open, so on Windows the configuration file
remained **locked for the life of the process**: anything that read a config and
then tried to rewrite or delete it failed with `WinError 32`. Found while
building the configuration editor, which validates a generated config by parsing
it and then has to be able to replace the file.

Every row is materialised before the close, so there is nothing to lose by
closing immediately; the reader now does so in a `finally`.

**Verified inert.** `tools/bench_com.py --against` reported **ALL IDENTICAL**
across all three benchmark cases, and the generated diff in `sicopr.py` was 20
lines, all inside `__load_excel`. The symptom itself was checked directly: a
config can now be deleted immediately after being read, where before that
raised.

> Timing in that comparison is not usable — the full test suite was running
> concurrently on the first case (0.71x). The other two, run clean, were 0.99x
> and 1.02x. The accuracy verdict is unaffected by CPU contention; the speed
> numbers are.

## 2026-08-29 — recorded divergences that never read the engine

Three `xcheck` records computed BOTH sides of their comparison inside the test
file and never called `sicopr`. They reported "divergent" unconditionally, so
they said nothing about the engine at any point — and two of them were still
reporting a divergence that had already been fixed.

| record | state | what it actually was |
|---|---|---|
| `optfom_triple_transit_uses_matlab_half_away` | was reporting a live divergence | recomputed `round(...)` inline; the site was then genuinely fixed (B11-D15) |
| `s21_grid_round_half_away_from_zero` | **stale** | condition was `round(200.5) == 201`, which is false for Python's builtin whatever the engine does. The engine had been fixed (B03-D7) and the record never noticed |
| `mlse_gaussian_qfunc_arg_matches_matlab` | unsound | hand-wrote both forms from invented values (alpha=0.3, main=0.1, L=4, sigma=0.02) |

All three now read the engine — its helper, or its own source line — so they
resolve when the code does. Each was mutation-verified by reverting the site.
Recorded divergences went from 28 to 25.

**The MLSE one is left open rather than closed.** MATLAB 2331 reads
`(1-2*alpha)*main/(L-1)*sigma_noise`, which by MATLAB's left-associative
precedence multiplies by `sigma_noise` where the port divides. But the
expression feeds `delta_COM`, and `delta_COM` matches the MATLAB reference to 15
significant digits on all 208 cases (1.371137901447263 vs 1.37113790144726). So
the divergence is not observable in any reported output on this corpus — which
is evidence, not proof, since the corpus is one channel family. The form the
engine uses is pinned instead, with a note to re-run the correlation and check
`delta_COM` if anyone changes it.

**The general lesson.** A test that computes its own expected value AND its own
actual value tests arithmetic, not software. Three of twelve divergence records
had that shape. `docs/FIX_SUMMARY.md` already recorded that defect #1's fixtures
encoded the bug; this is the same failure at the level of the record rather than
the fixture.

What the guards do NOT claim:

- They were designed knowing the defects. The generalisable ones — analytic
  recovery, structural invariants, the three lints — need no knowledge of a
  particular defect and should catch new instances of their class; that is still
  unverified, and the way to verify it is whether the next correlation cycle
  finds anything already guarded.
- Coverage of the *ledger* is not coverage of the *engine*. 20 of 20 known
  defects are guarded; the config space the corpus never exercises remains the
  larger hole.
- ~~**16's class.**~~ Closed 2026-08-29. `tests/test_integer_ratio_rounding.py`
  scans all 94 `round()`/`np.round()` calls and flags any whose argument is a
  ratio of two integers. It found exactly one live site with no false positives:
  `triple_transit_time = round(2*sbr_peak_i/samples_per_ui)`, which the audit had
  recorded as B11-D15 and dismissed as "measure-zero" — the same reasoning the
  `nui` correction had already shown to be wrong for integer ratios. Fixed to
  half-away; `bench_com --against` reports ALL IDENTICAL, so it was inert on the
  benchmark cases but is no longer a latent divergence.

The six that were uncovered are now covered by the two layers built for exactly
them: `tests/test_analytic_recovery.py` (#2, #4, #5) and
`tests/test_structural_invariants.py` (#1, #3, #6). Every one was verified by
reintroducing the defect and watching the check fail, not by inspection.

### What the evidence does and does not support

**Supported.** The recurring classes — index bases, by-reference aliasing,
divergent duplicate copies — are now covered, and covered *statically or
synthetically*, so they no longer consume a correlation cycle. #14 is the
concrete case: it crashed only under a sweep configuration the 208-case corpus
never runs, and would otherwise still be latent.

**Not supported.** "More tests improve accuracy" is not what the data shows.
Nineteen of the twenty aliasing fixes changed no number, and none of the 12
copy divergences was live. What the work bought is **the cost of finding the
next one**, not a measurable accuracy gain today. The accuracy gains in this
cycle all came from the MATLAB correlation.

**The honest caveat.** These detectors were designed knowing the defects. The
claim worth defending is not "we would have caught them," it is that three of
the families involved — analytic recovery, structural invariants, redundancy —
require no knowledge of any particular defect and therefore generalise. That
assumption is still unverified, and the way to verify it is whether the next
correlation cycle finds anything in a class already guarded.

### The measurable gap — closed 2026-08-28

This section used to read: *"Six ledger defects remain uncovered, and they
cluster: #2, #4 and #5 all need analytic recovery (build the input from the
known answer), #1, #3 and #6 all need structural invariants from the config.
Those are layers 3 and 4 of the plan."*

Both layers are built and all six are covered. Two things are worth recording
because they were not obvious in advance:

**`make_full_pkg` turned out to be drivable.** The table above had recorded it
as "currently undrivable", which is why #6 had no guard. It needs a fifteen-field
`param` and nothing else; the existing per-function test already had two thirds
of that factory. The barrier was assumed rather than measured.

**A shape assertion would not have caught #1.** The z_p transpose is invisible
to shape checks on a square matrix, which is what let it through the first time.
The guard therefore uses a non-square input *and* asserts values on a square
one — and the mutation test confirms the square case is the one that catches it.

The number still worth watching is independent-oracle coverage of the
per-function suite, 18 of 156 files. These two layers add oracle-free coverage
at the `tests/` level rather than inside that suite, so they raise the ledger
coverage without moving that ratio.

---

---

## Appendix — the gated fix pass (closed 2026-08-17)

An earlier, differently-structured fix pass, overtaken by the 208-case
correlation. Its working state has been moved out of this ledger to
[`ARCHIVE_gated_fix_pass.md`](ARCHIVE_gated_fix_pass.md), because it was written
in the present tense — "pending reassembly", "Fix queue", "Next:" — and read
like a live backlog when it is nothing of the kind. **There is no outstanding
engine work in it.**
