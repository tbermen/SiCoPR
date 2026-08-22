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
> `com.py` is assembled) and to the settings the engine is run with. Tooling,
> reporting and documentation changes are not tracked here, with one exception
> noted below.

---

## Current correlation status

208 MATLAB reference cases (`com_ieee8023_4p15p0`, 26 channels × 4 packages ×
with/without crosstalk). The two reference workbooks were produced with
different Tx FFE settings, so both readings are reported — see
[`TXFFE_SWEEP_ROOT_CAUSE.md`](TXFFE_SWEEP_ROOT_CAUSE.md).

| | configs as supplied | settings aligned |
|---|---|---|
| FOM bit-exact | 198 / 208 | **208 / 208** |
| COM bit-exact | 198 / 208 | **207 / 208** |
| sampling phase (`itick`) exact | 200 / 208 | **208 / 208** |
| max \|ΔCOM\| | 0.1852 dB | **0.0076 dB** |
| pass/fail disagreements at 3 dB | 0 | **0** |

Starting point before any of the fixes below: **max \|ΔCOM\| = 6.256 dB**.

---

## The ledger

| # | date | fix | class | found by | bought | commit |
|---|---|---|---|---|---|---|
| 1 | 2026-08-14 | `z_p` transpose for RX / NEXT / FEXT | config parse | column-ranked error | RX package built from a matrix row → ~15 dB spurious loss | `b2b2621` |
| 2 | 2026-08-14 | Insertion-loss fit solved at effective rank 2 of 4 | numerics | column-ranked error | `lstsq` silently truncated rank; half the fit basis restored | `b2b2621` |
| 3 | 2026-08-14 | RxFFE floating-tap array sized by tap COUNT, not SPAN | allocation | column-ranked error | floating taps past index 23 were being discarded | `b2b2621` |
| 4 | 2026-08-14 | `get_TDR` `tfstart` index base | 1- vs 0-based | column-ranked error | Z11est/Z22est 1.4e-2 → 4e-15 | `b2b2621` |
| 5 | 2026-08-14 | `get_TDR` `fctrx` initialisation | translation | column-ranked error | ERL11/ERL22/ERL 4e-1 → ~1e-15 | `b2b2621` |
| 6 | 2026-08-14 | Package die network truncated to 1 of 3 LC sections | translation | column-ranked error | restored ~15 ps of die delay (a 53-sample pulse shift) | `b2b2621` |
| 7 | 2026-08-14 | Cursor index base in `optimize_fom` (audit B16-D20) | 1- vs 0-based | audit + #6 | was reverted early as "worse"; it and #6 were **compensating** | `b2b2621` |
| 8 | 2026-08-14 | `process_sxp` leaked a TDR-only setting into the whole run | **by-reference** | sign of the bias | **the systematic FOM bias** — Python low on 199/208 (95.7%) | `b2b2621` |
| 9 | 2026-08-19 | `BEST.PSD_results` aliased a struct MATLAB copies by value | **by-reference** | stage-6 noise gap | COM bit-exact 135 → **170**, pass/fail flips 2 → **1** | `00529f8` |
| 10 | 2026-08-20 | Four latent fidelity defects (unstable sort ×2, abort-path leak, `_findbankloc` stubs ×2) | mixed | itick investigation | **inert on this corpus by design** — real under other settings | `6994b4a` |
| 11 | 2026-08-20 | Off-by-one in the ADC-clip sampling phase | 1- vs 0-based | variance decomposition | COM bit-exact 170 → **198**, pass/fail flips 1 → **0** | `62dbce6` |
| 12 | 2026-08-20 | `Overwrite_Min_Radius` honoured in both version paths | silent discard | Tx FFE investigation | a config setting it under 4p15p0 had it silently discarded | `abdba31` |
| 13 | 2026-08-21 | `TXLE_taps_1..4`, `Pre2Pmax`, mixed-mode ERL absent from results.xlsx | reporting | user review | columns claimed to agree were never exported | `cb28dca`, `3f0404e` |
| 14 | 2026-08-21 | `BEST.ctle` / `BEST.G_high_pass` used 1-based as 0-based in `OptFom_Update_BEST_Post_Optimize` | **1- vs 0-based** | corpus sweep crash | reporting-only, but it **hard-crashed** any run whose winning CTLE was last in the list | (this commit) |
| 15 | 2026-08-22 | `floating_tap_locations` had a **producer-dependent base** — 0-based from `floatingDFE`, 1-based from `MMSE`/`force` | **1- vs 0-based** | index registry | the FDFE cursor-time vector was one UI early under `Floating_DFE` | (this commit) |

**Five of the fifteen are the same root class**: MATLAB assigns structs **by
value**, Python binds a **reference**. #8, #9 and part of #10 are direct
instances; #3 and #6 are the same failure to carry a whole structure across a
boundary. This is the single most productive thing to check first in this port.

---

## 2026-08-14 — the eight defects found by the 208-case correlation (`b2b2621`)

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

## 2026-08-19 — `BEST.PSD_results` aliasing (`00529f8`)

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

## 2026-08-20 — four latent defects (`6994b4a`)

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

**Deliberately not done**: ~25 other `argsort` sites, several using
`argsort(x)[::-1]`, which inverts tie order relative to a stable descending sort
even with `kind='stable'` added. They need individual checks against the MATLAB
they came from, not a blanket edit.

## 2026-08-20 — ADC-clip sampling phase (`62dbce6`)

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

## 2026-08-20 — `Overwrite_Min_Radius` version path (`abdba31`)

Found while resolving the ten Tx FFE cases. The keyword was read only under
4p16p0, so a config setting it while emulating 4p15p0 had it **silently
discarded** — the same silent-config-discard class as the Tx FFE settings
mismatch itself. Guarded since by `tests/test_config_search_space.py`.

The accompanying settings finding is not an engine fix and is recorded in
[`TXFFE_SWEEP_ROOT_CAUSE.md`](TXFFE_SWEEP_ROOT_CAUSE.md): the supplied configs
pin Tx FFE `c(-1)` to a single zero, so Python searched 1 candidate where MATLAB
searched ~1584.

## 2026-08-21 — results.xlsx columns (`cb28dca`, `3f0404e`)

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

Replayed against the historical versions in git: `b2b2621~1` flags `cursor_i` in
`optimize_fom` (#7); `62dbce6~1` flags `t_s` in `Apply_EQ` (#11) and
`ctle`/`G_high_pass` (#14); `bc4cecb~1` flags #14; HEAD is clean. The miss is #4,
which is not a base error — a valid index applied to the wrong array frame.

It runs over the assembled `com.py`, so it covers all inlined copies of a
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
> subscript at `com.py:5865` was defective. It was not — its producer is
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

That surfaced **12 divergences**, every one checked against com.py's `_wired_*`
partials before being recorded. None is live, but two findings are worth having:

- The `get_TDR` Bessel fallback stub is not subtly wrong — it hardcodes the
  4th-order coefficients **without reversing them**, giving DC gain **105
  instead of 1**, and returns a magnitude where MATLAB returns a complex
  response (ML 1033-1040 uses `fliplr`). Harmless only for as long as the
  injection at `com.py:132` holds.
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

Measured per column across the settings-aligned set:

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
classes again. Verified by replaying the rules against the historical `com.py`
in git, or by reintroducing the defect and watching the test fail:

| # | defect | caught today? | how verified |
|---|---|---|---|
| 1 | `z_p` transpose | ✗ | — |
| 2 | `lstsq` rank truncation | ✗ | — |
| 3 | RxFFE array count vs span | ✗ | — |
| 4 | `get_TDR` `tfstart` frame | ✗ | not a base error; needs analytic recovery |
| 5 | `get_TDR` `fctrx` init | ✗ | needs analytic recovery |
| 6 | die network 1 of 3 sections | ✗ | would be covered by `make_pkg` / `make_full_pkg` factories (5 copies, currently undrivable) |
| 7 | cursor index base | **✓** | replay: flags `cursor_i` in `optimize_fom` at `b2b2621~1` |
| 8 | `process_sxp` leak | **✓** | `test_reference_leaks`, mutation-verified |
| 9 | `BEST.PSD_results` aliasing | **✓** | reintroduced; snapshot test fails naming `PSD_results` |
| 10a | unstable sort | ✗ | needs the `argsort` lint |
| 10b | abort-path leak | ✗ | different shape from the leak guard |
| 10c | `_findbankloc` stubs | **✓** | now drivable; flags the `MMSE` and `force` copies |
| 11 | ADC-clip sampling phase | **✓** | replay: Rule B flags `t_s` in `Apply_EQ` at `62dbce6~1` |
| 12 | `Overwrite_Min_Radius` discard | partial | keyword parity sees unread keywords, not per-version-path ones |
| 13 | results.xlsx columns | **✓** | mutation-verified |
| 14 | `BEST.ctle` index base | **✓** | replay + mutation |
| 15 | `floating_tap_locations` | **✓** | the test that found it |

**8 of 15 caught automatically, 1 partial, 6 not.** Before this work the number
was effectively 2 (`test_reference_leaks` and the config keyword check), and
everything else required a MATLAB run to notice.

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

### The measurable gap

Six ledger defects remain uncovered, and they cluster: #2, #4 and #5 all need
**analytic recovery** (build the input from the known answer), #1, #3 and #6 all
need **structural invariants** from the config. Those are layers 3 and 4 of the
plan. The per-function suite's independent-oracle coverage — 18 of 156 files —
is the number to watch.

---

## Appendix — the gated fix pass (closed 2026-08-17)

Retained for history. An earlier, differently-structured pass that was overtaken
by the 208-case correlation.


Driven by `FIX_PROMPT_com_conversion_v2.md`. Fixes edit
`com_functions/fn/<name>/py_impl.py` (never `com.py`, which is a generated build
artifact reassembled by `assemble_com.py`). One finding per gate.

> **CLOSED 2026-08-17.** This gated fix pass was overtaken by the 208-case MATLAB
> correlation, which fixed eight engine defects in one commit (`b2b2621`) — see
> [`../MATLAB_Correlation_Review.md`](../MATLAB_Correlation_Review.md). The
> "Next: F6 get_pdf_full" item below was never worked in this format. The stale
> guard flagged at the end of the batch-2 note (`get_TDR_s2p_RL_is_the_wrong_formula`)
> has since been removed, and the divergence ledger now uses `xcheck` so a
> resolved divergence fails the run instead of lingering.

## Status: GATE BATCH-2 ASSEMBLED_VERIFIED (com.py reassembled + integration-tested)

Batch 2 = B01-D2 (get_PSDs ADC 'slow' clip) + B06-D9 (get_TDR s2p RL) -
ASSEMBLED_VERIFIED. assemble_com.py: 157 functions, 5 stubs (baseline match).
Integration: B01-D2 (adc_clip_slow + ctle_signal_sigma) and B06-D9 (s2p_RL) audit
checks flipped FAIL->PASS; fn suite 874 pass / 0 fail; smoke + e2e exit 0.
**get_PSDs fully resolved** (D1/D2 fixed, D3 kicked back). One stale guard flagged
(get_TDR_s2p_RL_is_the_wrong_formula, asserted the old bug). Next: F6 get_pdf_full
(B08-D12), or the always-on bugs B16-D20 / B13-D18.

---

## Prior: GATE BATCH-1 ASSEMBLED_VERIFIED (com.py reassembled + integration-tested)

Revision 4p15p0, dependency order. Batch 1 = B03-D6/D7, B02-D4/D5, B01-D1 -
ASSEMBLED_VERIFIED. `assemble_com.py`: 157 functions, 5 stubs (baseline match).
Integration: the audit "matches-MATLAB" checks for all three flipped FAIL->PASS
(s21 causality, interp_Sparam phase+mag, get_PSDs S_jn LIMIT); com_functions/fn
suite 872 pass / 0 fail (865 baseline + 7 new); tests/test_smoke.py and
tests/test_end_to_end.py exit 0. B01-D3 KICKED_BACK (unreachable). B01-D2 deferred
to its own gate.

Two audit divergence-documentation guards are now stale/invalid and flagged for
replacement (NOT edited): `s21_grid_round_half_away_from_zero` (asserts Python's
builtin `round`, not com.py) and `interp_Sparam_phase_extrap_is_flat` (asserts the
old clamped phase). The real B03-D7/B02-D5 fixes are verified by the fn tests.

Next: B01-D2 (own gate) then F4 get_TDR (B06-D9), or as you direct.

### Gate 0 findings

1. **MATLAB revision mismatch (Section 2 gate).** `registry.json` `matlab_lines`
   are indexed to **4p14p0**; the audit ledger and this pass cite **4p15p0** (the
   source-of-truth `.m` present, and the revision the audit re-derived against).
   Offset grows from +5 (~line 2585) to +130 (past line 7000) because 4p15p0
   added content. All 11 DIVERGENT functions are version-identical (4p14==4p15
   body, audit B13), so re-derivation is valid from either; 4p15p0 is chosen for
   citation consistency. `registry.json` will NOT be edited (dirs are name-keyed;
   the line base is metadata only). **Recommendation: confirm 4p15p0 authoritative.**

2. **Baseline build is not the prompt's literal "clean".** `python assemble_com.py`
   → "Assembled com.py from 157 functions. WARNING: 5 function(s) still contain
   NotImplementedError." The 5 are **deliberate stubs** for non-portable/undefined
   paths (`force` WIENER-HOPF [undefined in MATLAB], `get_s4p_files`/`get_TD_files`
   GUI file-picker, `read_package_parameters` cross-import, `RILN_TD` inlined
   non-zero-IL). All 5 functions' fn tests PASS. This is the recorded baseline
   reference (not a blocker; none are in the fix queue).

3. **git unavailable** → reversibility via `.bak` fallback (copy `py_impl.py` to
   `py_impl.<finding>.bak` before editing).

4. Test convention: `com_functions/fn/<name>/test_verify.py`, pytest, imports
   `py_impl` directly, `SimpleNamespace` inputs, analytic/invariant assertions.

### Fix queue (dependency-ordered, Section 5): 11 DIVERGENT functions

| Rank | Finding | Function | Stage | Trap | Audit recommendation (hypothesis) |
|---|---|---|---|---|---|
| 1 | B03-D6/D7 | s21_to_impulse_DC | FD->TD impulse (high fan-in) | index-base | fix causality window; half-away grid round |
| 2 | B02-D5/D4 | interp_Sparam | S-param interp (high fan-in) | interpolation/eps | linear phase extrap; eps=finfo.eps |
| 3 | B01-D1/D2/D3 | get_PSDs | noise PSD (high fan-in) | index/conditional | jitter +1; ADC-slow clip; stub defaults |
| 4 | B06-D9 | get_TDR | TDR/ERL | mldivide-vs-divide | RL=(s11-rho)/(1-rho*s11) |
| 5 | B06-D11 | TDR_ERL_Processing | TDR/ERL | cosmetic | file_names from first base |
| 6 | B08-D12 | get_pdf_full | C2M PDF | resample grid | match MATLAB grid |
| 7 | B13-D18 | get_xtlk_noise | crosstalk noise | argmax slice off-by-one | index_f2 = argmax+1 / len(f) |
| 8 | B12-D17/D16 | MMSE_FOM | optimizer MMSE | conditional recompute | blim recompute inside w!=wlim |
| 9 | B16-D20/B11-D15 | optimize_fom | optimizer driver | cursor off-by-one | drop the -1 at A_s/A_p/far/pre |
| 10 | B15-D19 | MLSE | post-processing | precedence | sigma_noise to numerator (diagnostic-only) |
| 11 | B05-D8 | s_for_c4 | cascade (UNUSED) | port reorder | apply snp2smp([1 3 2 4]) |

Cross-cutting (deferred, separate decision): **B10-D14** rounding (~50 bare-round
sites, measure-zero; half-away sweep vs accept).

**Recommended first fix:** rank 1, **s21_to_impulse_DC (B03-D6)** - upstream-most,
highest fan-in, so downstream fixes build on corrected TD impulse. (If you prefer
impact-ordering instead of dependency-ordering, the highest always-on COM impact
is B16-D20 optimize_fom and B13-D18 get_xtlk_noise.)

## Fixed (UNIT_VERIFIED, pending reassembly)

| Finding | Function | MATLAB (4p15p0) | Edit | Unit result |
|---|---|---|---|---|
| B03-D6 + B03-D7 | s21_to_impulse_DC | 11232, 11260-11261, 11267 | causality window `[:start_ind]`->`[:start_ind+1]`, `[half:]`->`[half-1:]`, signed `max`; fout `round`->half-away `_mround` | 9 PASS (2 new + 7 pre-existing); 38/38 neighbor tests PASS |
| B02-D4 + B02-D5 | interp_Sparam (+ inlined copy in s21_to_impulse_DC) | 8091, 8125, 8185 | `|S|` floor + HF threshold use machine `eps` (added `eps_mag`; log guards left at `tiny`); phase clamp -> linear end-slope extrapolation | 20 PASS (4 new + F1 + pre-existing); 53 neighbor tests PASS |
| B01-D1 | get_PSDs | 6676-6677 | LIMIT_JITTER early/late sampling centered at cursor: `cursor_i+M*k`->`cursor_i-1+M*k`, `cursor_i+2+M*k`->`cursor_i+1+M*k` | 8 PASS (1 new + 7 pre-existing); 44 neighbor tests PASS |
| B01-D2 | get_PSDs | 6716-6725 | ADC 'slow' clip: inlined get_pdf_from_sampled_signal/conv_fct/CDF_inv_ev; adc_clip = P_qc quantile of signal+noise PDF; set ctle_signal_sigma (else sum(abs) unchanged) | 9 PASS (1 new + 8 pre-existing); 40 neighbor tests PASS |
| B06-D9 | get_TDR | 7080-7081 | s2p RL: `interim/(s11-rho)` -> `(s11-rho)/interim` (MATLAB left-division; interim cancels -> `(s11-rho)/(1-rho*s11)`); s4p path untouched | 7 PASS (1 new + 6 pre-existing); 41 neighbor tests PASS |

Reach: propagates to mainline via the shared `s21_to_impulse_DC` (`COM_FD_to_TD`,
`get_TDR`, `get_RAW_FIR`, `calculate_delay_CausalityEnforcement`, `TD_FD_fillin`,
`RILN_TD`, `get_ILN_cmp_td` all call it). Backup: `py_impl.B03-D6-D7.bak`.

## Kicked back / version-delta / surfaced-new
- **B01-D3 (get_PSDs injection stub defaults) — KICKED_BACK** (human-approved).
  The `_S_RN_fn`/`_S_IN_fn`/`_H_interp_fn` stub defaults do diverge, but the
  assembled pipeline always injects the real functions via `_run_com`, so no
  reachable `com.py` result is affected. It cannot be fixed in `get_PSDs`'s
  `py_impl` (no cross-`py_impl` sibling imports; the stubs exist for standalone
  testability). Recorded as a latent API-robustness caveat, not a conversion bug.
- **B01-D2 (get_PSDs ADC 'slow' clip)** — deferred to its own later gate
  (substantial port; conditional on `N_qb!=0 & clip_method=='slow'`).
- Observation (not a finding): `RILN_TD` and `get_ILN_cmp_td` contain a **dead**
  inlined `_s21_to_impulse_DC` helper (raises for non-zero IL); it is unused - their
  real path calls the shared `s21_to_impulse_DC` - so the fix reaches them and no
  separate fix is required.
