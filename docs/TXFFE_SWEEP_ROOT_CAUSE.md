# The eight sampling-phase cases are a config mismatch, not an engine defect

**Found 2026-08-20, from data already on disk.** The `itick` divergences — and
the two non-exact-FOM cases alongside them — trace to the Tx FFE search space,
not to any arithmetic in the port.

---

## 1. The observation that settles it

MATLAB reports its winning Tx FFE (`TXLE_taps_1..4`). Across the 104
with-crosstalk cases:

| MATLAB's winning Tx FFE | cases | FOM bit-exact | `itick` mismatched |
|---|---|---|---|
| unity `[0, 0, 1, 0]` | 94 | **94 / 94** | 0 |
| **non-unity** | 10 | **0 / 10** | **8** |

Broken out by tap vector:

| tap vector | cases | FOM exact | tick mismatched |
|---|---|---|---|
| `(0, 0, 1, 0)` | 94 | 94 | 0 |
| `(0, −0.02, 0.98, 0)` | 4 | 0 | 2 |
| `(0, −0.04, 0.96, 0)` | 3 | 0 | 3 |
| `(0, −0.1, 0.9, 0)` | 2 | 0 | 2 |
| `(0, −0.06, 0.94, 0)` | 1 | 0 | 1 |

The split is perfect. **Every case where MATLAB selects a Tx FFE with
pre-emphasis is a case Python gets wrong, and every case where MATLAB selects
unity is bit-exact.** The ten non-unity cases are exactly the eight `itick`
mismatches plus `wXtalk_T3_R07` and `wXtalk_T3_R15`, the two known non-exact-FOM
cases. That is the entire residual, accounted for.

## 2. Why Python cannot reach those points

Python's reported `TXLE_taps` is `[1.0]` on all 208 cases, and `Pre2Pmax` — which
MATLAB derives from the same vector — is absent on all 208, because Python's
vector is shorter than the 3 elements that derivation needs.

The cause is in the config. In `COM_Settings` the Tx FFE tap rows are laid out:

| row | A | B | C | D |
|---|---|---|---|---|
| 17 | `c(-1)` | **`0`** | `[ -0.34:.02:0]` | `[min:step:max]` |
| 18 | `c(-2)` | **`0`** | `[ 0.14:.02:0]` | `[min:step:max]` |
| 21 | `c(1)` | **`0`** | `[ -0.2:.02:0]` | `[min:step:max]` |

Both engines read the cell immediately **right of the label** — column B — and
column B holds `0`. Column C is a template showing the available range. So the
supplied configs define a Tx FFE grid with exactly **one** point: unity. All four
configs (Case1–Case4) are identical in this respect.

**This is not a parsing bug.** Python's `__eval_matlab_value` handles the MATLAB
range syntax correctly when it is given it:

```
'[ -0.34:.02:0]'  ->  18 values
'[ -0.2:.02:0]'   ->  11 values
'[ 0.14:.02:0]'   ->   0 values (empty, exactly as MATLAB's 0.14:.02:0)
```

## 3. Confirmation: 10 / 10, to 1e-11

Pinning the CTLE to MATLAB's reported value, enabling the `c(-1)`/`c(1)` sweep
and evaluating **every** Tx FFE candidate on a full grid, then looking up
MATLAB's own tap vector in the result (`scratchpad/txffe_scan.py`):

| case | MATLAB Tx FFE | itick ML | itick PY | rank in Python | FOM ML | ΔFOM |
|---|---|---|---|---|---|---|
| wXtalk_T1_R07 | `[0, −0.04, 0.96, 0]` | −8 | **−8** | #2 | 12.443168 | −1.2e−11 |
| wXtalk_T1_R08 | `[0, −0.04, 0.96, 0]` | −6 | **−6** | #1 | 11.875333 | −1.5e−11 |
| wXtalk_T1_R15 | `[0, −0.02, 0.98, 0]` | −3 | **−3** | #1 | 12.528249 | −3.2e−12 |
| wXtalk_T1_R16 | `[0, −0.1, 0.9, 0]` | 0 | **0** | #1 | 11.691042 | −2.3e−11 |
| wXtalk_T2_R15 | `[0, −0.02, 0.98, 0]` | −4 | **−4** | #1 | 11.128613 | −4.7e−12 |
| wXtalk_T2_R16 | `[0, −0.06, 0.94, 0]` | 1 | **1** | #1 | 9.955886 | −1.6e−11 |
| wXtalk_T3_R07 | `[0, −0.02, 0.98, 0]` | −8 | **−8** | #1 | 10.759523 | −9.7e−12 |
| wXtalk_T3_R15 | `[0, −0.02, 0.98, 0]` | −3 | **−3** | #1 | 10.589143 | −5.0e−12 |
| wXtalk_T3_R16 | `[0, −0.04, 0.96, 0]` | 2 | **2** | #1 | 9.361404 | −1.5e−11 |
| wXtalk_T3_R17 | `[0, −0.1, 0.9, 0]` | 6 | **6** | #9 | 13.801599 | −1.7e−12 |

**Given the same Tx FFE, Python reproduces MATLAB's sampling phase exactly and
its FOM to 1e-11 on all ten.** The COM engine is correct; the residual was
entirely the search space.

The most striking is `wXtalk_T1_R16`. It was the case where "Python cannot reach
MATLAB's FOM at MATLAB's tick under any equalizer setting — short by 5.43 dB".
With `[0, −0.1, 0.9, 0]` in the grid, Python produces 11.691042 at `itick = 0`,
matching MATLAB to 2.3e−11. The equalizer was never in the grid to be found.

### A second finding: MATLAB's own answer is not always its grid's optimum

The "rank" column is Python's full-grid ranking at MATLAB's CTLE. On 8 of 10
MATLAB's choice is rank #1. On two it is not:

- `wXtalk_T1_R07` — rank #2; `[−0.06, 0.94, 0]` scores 12.4454, +0.0023 dB better.
- `wXtalk_T3_R17` — rank #9; seven candidates beat it, including **unity itself**
  at 13.8662 versus MATLAB's 13.8016 (+0.0796 dB).

Unity is certainly in MATLAB's grid (its `c(-1)` range spans 0), so on R17
MATLAB's adaptive local search terminated before evaluating a point 0.08 dB
better. **This is direct evidence, from the reference data itself, that adaptive
local search can stop short of the optimum on a real Tx FFE grid.**

That matters for the proposal, and it qualifies an earlier result: the
"adaptive LS == full grid, bit-identical, 9/9" check was run on the configs as
supplied — i.e. with a **single-point** Tx FFE grid, where the adaptive search has
nothing to prune in that dimension. It should be re-run on a real grid before
being quoted.

## 4. What this replaces

The earlier reading — that the eight were a numerical disagreement about where in
the UI to sample — is wrong. They are cases where the two engines were given
**different search spaces**. Python could not select the equalizer MATLAB
selected, so it settled elsewhere; and a different Tx FFE changes the equalized
pulse, which moves the peak that anchors `itick`.

This supersedes the anchor/basin analysis in `docs/ITICK_SUBSET_FOR_REVIEW.md`
§5–§6 *as the explanation*. The measurements there stand — the anchor really is
re-derived per EQ candidate and really does move 2–4 samples — but the reason the
two engines' anchors differ is the Tx FFE, not an ambiguity in the anchor rule.

## 5. The question for Hansel, restated

Not `cursor_i` any more:

> **1. Which Tx FFE tap ranges were active in the run that produced these
> workbooks?**
>
> **2.** (confirmation only) The workbooks are named `..._AdaptiveLS.xlsx`, so
> adaptive local search was used and Python matches it. Worth one line to confirm
> the implementation was the 4p16p0 one backported into a 4p15p0 build, since
> 4p15p0 as distributed has no adaptive search.

The configs as supplied set `c(-1)`, `c(-2)` and `c(1)` to `0` in the value
column, which yields a single unity Tx FFE — but the results contain non-unity
winners on 10 cases, so the run used a different setting. Specifically: was
`c(-2)` set to `[ 0.14:.02:0]` (which evaluates empty) or to something else, and
were `c(-3)`/`c(-4)` in play?

This is now the **only** open unknown — see §7 and the note on item 2 below.

With that, the corpus can be re-run on a matched search space and the remaining
ten cases should close.

## 6. Consequence for the runtime comparison — flagged, not yet measured

**The published runtime comparison is not like-for-like.** Python has been
searching 1 Tx FFE candidate per CTLE where MATLAB searched a swept grid — 198
combinations from the `c(-1)`/`c(1)` ranges alone, before pruning.

Measured on `wXtalk_T1_R07`:

| configuration | wall clock |
|---|---|
| as supplied (1 Tx FFE point) | **1.1 min** |
| sweep on, adaptive local search | **1.8 min** |
| sweep on, legacy local search | slower again (all 198 visited per CTLE) |

*(An earlier draft of this section put the sweep-enabled figure at 15–20 minutes.
That was wrong — it came from a full-grid pinned-CTLE diagnostic scan, not from a
normal run. The correct penalty with adaptive search is about 1.6×.)*

Any claim that the Python port is faster than MATLAB must still be re-measured on
a matched search space, but the correction matters: the penalty is modest, so a
corrected full-corpus run is affordable.

---

## 7. Re-running the corpus on a *guessed* grid makes correlation worse

The obvious next step is to re-run all 208 with the sweep on. Done
(`--txffe-sweep`, results in `matlab_compare_results/cases_txffesweep/`):

| | FOM bit-exact | COM bit-exact | itick exact | max \|ΔCOM\| | flips |
|---|---|---|---|---|---|
| as supplied (baseline) | **198 / 208** | **170 / 208** | **200 / 208** | 0.1852 | 1 |
| Tx FFE sweep enabled | 189 / 208 | 163 / 208 | 192 / 208 | 0.2029 | 1 |

It gets **worse**, and the reason is instructive rather than discouraging.

Of the ten cases the sweep was meant to fix, **five now match MATLAB exactly**
(bit-exact FOM and identical `itick`): `wXtalk_T1_R15`, `T2_R16`, `T3_R16`,
`T3_R17`, `T3_R15`. On the other five Python's search stops at a different tap
set from MATLAB's:

| case | MATLAB | Python (sweep) | ΔFOM |
|---|---|---|---|
| wXtalk_T1_R07 | −0.04 | −0.02 | −7.1e−3 |
| wXtalk_T1_R08 | −0.04 | −0.02 | −5.8e−3 |
| wXtalk_T1_R16 | −0.10 | −0.06 | −1.2e−3 |
| wXtalk_T2_R15 | −0.02 | unity | −6.9e−3 |
| wXtalk_T3_R07 | −0.02 | unity | −8.7e−4 |

And it costs about fourteen previously-matching cases: Python selects a non-unity
Tx FFE on **22** of 208 where MATLAB selected one on **10**.

### What that tells us

The grid is a guess, and a guess is not good enough. Two things about the real
run are still unknown, and both change which candidates a local search visits:

1. **The exact tap ranges.** MATLAB's `TXLE_taps` has four elements, so the
   `c(-2)` slot survived the trim in `OptFom_Build_TXFFE` — meaning it held at
   least two values. My injection sets only `c(-1)` and `c(1)`, so my grid has a
   different *shape*, which changes `num_txffe_runs`, the sweep-index ordering
   and hence the pruning.
2. ~~The local-search method.~~ **Not a mismatch — corrected 2026-08-20.** The
   reference workbooks are named `..._AdaptiveLS.xlsx`, so the run used adaptive
   local search and Python's `Non-zero Local Search Method = 1` matches it.
   (`com_ieee8023_4p15p0.m` *as distributed* contains only `OptFom_Local_Search`;
   the adaptive variant first appears in 4p16p0, so the reference run used a
   4p15p0 build with it backported — expected, from its author.) An earlier
   version of this document listed this as a second mismatch. It is not one.

So there is **one** unknown, not two: the grid shape. A spot check on
`wXtalk_T1_R07` with the legacy method reaches FOM 12.4454 — *above* MATLAB's
12.4432 — confirming Python is not under-searching either way. The two engines
are walking differently-shaped grids.

### Consequence

**The as-supplied run stays the headline correlation.** It is a faithful use of
the configs we were given, the ten outliers are explained and independently
verified (§3), and it is the better number besides. The sweep run is kept
alongside as evidence, not as a replacement.

Closing the last ten end to end needs the two answers in §5 — not more work on
this side.


---

## 8. Looking for the same failure mode elsewhere, and a guard against it

The defect class is **silent search-space collapse**: a parameter that should
define a swept dimension reads as a scalar, nothing errors, and the optimiser
quietly searches one point. Worth asking where else that can happen.

### Audit (`tools/audit_search_space.py`)

Two sweeps over all four shipped configs:

**Adjacent-template** — a keyword whose value cell is a scalar while a real
MATLAB range literal sits in the next cell along, i.e. the exact Tx FFE shape.
Result: **3 findings, all of them the Tx FFE taps** (`c(-1)`, `c(-2)`, `c(1)`),
in each of the four configs. Nothing else in the workbook has this shape.

**Degenerate dimensions** — the five dimensions `optimize_fom` actually loops
over:

| dimension | size | verdict |
|---|---|---|
| `cursor_gain` (Gffe) | 1 | legitimate — see below |
| `ctle_gdc_values` | 21 | swept |
| `g_DC_HP_values` | 7 | swept |
| **Tx FFE grid** | **1** | **the defect** |
| itick range | 49 | swept |

`cursor_gain` deserved a check because it is also 1. It is read from
`'crusor_gain'` — a misspelling in the MATLAB source, which the port reproduces
exactly — its config default is 0, and MATLAB's own comment says "only FFE and
not supported". `length(0) == 1` in MATLAB too, so both engines agree. Not a
defect.

### Keyword parity

One level up, the same class appears as *a keyword the reference honours and the
port ignores*. Comparing every `xls_parameter` call in `com_ieee8023_4p15p0.m`
against `com.py`: **every keyword MATLAB reads is read by the port.** The three
the port reads and 4p15p0 does not are `COM Version` (a port-only version
switch) and `Non-zero Local Search Method` / `Overwrite Minimum Radius`, both of
which exist in 4p16p0 with exactly those spellings.

*(An initial pass reported 8 missing keywords. Four are read through the
package-block helper, two — `Impulse response truncatio threshold` and
`Include PCB (table 92-13)` — are commented out in the MATLAB source, and the
regex was matching inside comments. Fixed.)*

### The guard: `tests/test_config_search_space.py`

Nothing in the suite could have caught this, because every existing test asks
"does this function compute the right answer?" and none asked "is the optimiser
being given anything to search?". The new test asks the second question:

1. **Search dimensions are pinned** per config. If a parser change, a config edit
   or a version switch moves one, the test fails and forces the correlation to be
   re-stated rather than shifting unnoticed. *This is the check that would have
   caught the Tx FFE defect on day one.*
2. **Adjacent-template ledger** — a *new* scalar-beside-a-range keyword fails;
   the three known ones carry a documented reason.
3. **Keyword parity** — a keyword MATLAB reads and the port does not fails.

Verified by mutation: setting the expected Tx FFE dimension to 198 fails all four
configs, and removing `c(-1)` from the ledger fails all four. It is wired into
`tests/run_all.ps1` by auto-discovery.


---

## 9. Adaptive local search is lossy on a real Tx FFE grid

The `adaptive LS == full grid, bit-identical 9/9` result is the centrepiece of the
proposal, and it was measured with the configs as supplied — i.e. on a
**single-point** Tx FFE grid, where the adaptive search has nothing to prune in
that dimension. Repeating the comparison on a real grid.

`wXtalk_T1_R07`, CTLE pinned to MATLAB's −17 / −4, Tx FFE grid of 198 candidates:

| search | Tx FFE | itick | FOM | vs full grid |
|---|---|---|---|---|
| Python **full grid** (rank #1) | `[−0.06, 0.94, 0]` | −9 | 12.4454352312 | — |
| Python **legacy** local search | `[−0.06, 0.94, 0]` | −9 | 12.4454352312 | **0.0000** |
| MATLAB **adaptive** LS (reported) | `[0, −0.04, 0.96, 0]` | −8 | 12.4431675894 | −0.0023 |
| Python **adaptive** LS | `[−0.02, 0.98, 0]` | −7 | 12.4360986990 | −0.0093 |

The clean comparison is the middle pair against the top row, because they are the
same engine on the same grid: **the legacy search finds the full-grid optimum
bit-identically, and the adaptive search stops 0.0093 dB short.** MATLAB's
adaptive search also stops short of that optimum, by 0.0023 dB, though its number
is not strictly comparable because its grid has a different shape (§7).

### What this does and does not say

- It **does** show that "adaptive == full grid" is a property of the
  single-point-grid configuration, not a general property of the algorithm. On a
  real Tx FFE grid the adaptive search demonstrably loses FOM.
- It **does not** quantify the loss in general. This is one case at one CTLE.
  A defensible number needs the comparison re-run across the corpus on a real
  grid — which needs the grid (§5).
- The loss is small in absolute terms (0.009 dB of FOM here) and the runtime
  saving is large. That may well be the right trade; the point is that it *is* a
  trade, and the current framing says it is free.

This is consistent with the independent observation in §3 that MATLAB's own
reported answer is not its grid's optimum on 2 of the 10 cases — including
`wXtalk_T3_R17`, where seven candidates beat it at its own CTLE, unity among them
(13.8662 vs the reported 13.8016, a 0.0796 dB gap).

**Recommendation before the proposal is presented:** re-run the
adaptive-vs-full-grid comparison on a real Tx FFE grid and quote *that* number.
The claim is stronger for being honest about the cost, and an IEEE reviewer with
a swept config will find this immediately.
