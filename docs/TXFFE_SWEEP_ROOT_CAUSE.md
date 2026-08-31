# The eight sampling-phase cases are a config mismatch, not an engine defect

> **RESOLVED.** With the with-crosstalk grid reconstructed (`c(-2)=[0:.02:0.14]`
> alongside `c(-1)`/`c(1)`, plus adaptive-search `min_radius = 2`), all ten
> previously divergent cases reproduce MATLAB exactly. Matched-configuration
> correlation over the full corpus: **FOM 208/208 bit-exact, itick 208/208 exact,
> COM 208/208, max |ΔCOM| 3.3e-14, zero pass/fail disagreements.** See §10–§12.

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
"adaptive LS == full grid" check (now 208/208) was run on the configs as
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
| as supplied (baseline) | **198 / 208** | **198 / 208** | **200 / 208** | 0.1852 | 0 |
| Tx FFE sweep enabled | 189 / 208 | 188 / 208 | 192 / 208 | 0.2101 | 0 |

*(Both rows re-measured 2026-08-21 on the engine including the ADC-clip fix, so
the comparison is like-for-like. The earlier version of this table had the
baseline row refreshed and the sweep row not, which made it apples-to-oranges.)*

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

*Written before the maintainer confirmed the sweep.* At the time, the run on
the configs as received was kept as the headline, because the sweep grid was a
reconstruction. That is no longer the position: the sweep workbooks were
supplied on 2026-08-24, so the with-crosstalk cases run on them and there is a
single correlation — 208/208 on FOM, COM and sampling phase (§12).


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
against `sicopr.py`: **every keyword MATLAB reads is read by the port.** The three
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

## 9. Adaptive local search on a real Tx FFE grid: optimal on 15 of 16

The `adaptive LS == full grid` result (now **208/208 bit-identical**, `MATLAB_Correlation_Review.md` §5)
is the centrepiece of the proposal, and it is measured with the configs as supplied — i.e. on a
**single-point** Tx FFE grid, where the adaptive search has nothing to prune in
that dimension. Repeating it on a real grid.

### Method

A joint full grid over CTLE × Tx FFE is **1.43M evaluations per case** (21 CTLE ×
7 g_DC_HP × 198 Tx FFE × 49 ticks), roughly 198× the as-supplied full grid that
already cost 7–9 h per case. Not available. So the measurement isolates the
dimension that was actually degenerate: **CTLE and `g_DC_HP` pinned to the
adaptive search's own choice, then all 198 Tx FFE candidates evaluated.**

16 cases spanning DAC, BPK twinax and OSFP, both crosstalk conditions, all four
package configurations.

### Result

| | |
|---|---|
| cases | 16 |
| adaptive found the full-grid optimum | **15 / 16** |
| mean loss | 0.000584 dB |
| median loss | **0.000000 dB** |
| max loss | 0.009337 dB |

The single miss is `wXtalk_T1_R07`: adaptive picks `[−0.02, 0.98, 0]`
(FOM 12.436099, rank #4) where the full grid finds `[−0.06, 0.94, 0]`
(12.445435). Every other case lands on rank #1 exactly.

For that one case the legacy local search was also run, and it *does* find the
full-grid optimum bit-identically (12.4454352312), so the miss is specific to the
adaptive pruning rather than to local search in general.

### What to say about it

**Adaptive local search is not bit-identical to full grid once the Tx FFE grid is
real — it missed on 1 of 16 — but the typical loss is zero and the worst observed
is 0.009 dB.** That is a defensible and, for the proposal, a good result.

*An earlier version of this section was written from the `wXtalk_T1_R07` case
alone and concluded that "adaptive search is not free", recommending the claim be
re-measured before the proposal is presented. That over-generalised from a single
case. The recommendation to re-measure was right; the conclusion drawn before
doing so was not.*

### Caveats that remain

- This pins CTLE and `g_DC_HP` and full-grids the Tx FFE only. It does not test
  the adaptive search's pruning in the CTLE dimension — though the 208/208
  full-grid result does exactly that, and finds no loss.
- 16 cases, not 208.
- Separately, and not from this experiment: MATLAB's own reported answer is not
  its grid's optimum on 2 of the 10 non-unity cases (§3), including
  `wXtalk_T3_R17` where seven candidates beat it at its own CTLE. That is
  consistent with a small, occasional pruning loss of the size measured here.


---

## 10. RESOLVED — 10 / 10, with two settings

The ten cases reproduce MATLAB exactly once **both** of the following are set. Each
alone is insufficient.

### Setting 1 — the Tx FFE grid, including c(-2)

```
c(-1) = [ -0.34:.02:0]     18 values
c(-2) = [0:.02:0.14]        8 values     <-- ascending, NOT the backwards form
c(1)  = [ -0.2:.02:0]      11 values
                          ---------
                           1584 candidates
```

The `[ 0.14:.02:0]` printed in the config's **Units** column is written backwards
and evaluates **empty** under the MATLAB colon operator. The ascending form is
the one that reproduces MATLAB's 4-element `TXLE_taps`, because `c(-2)` then
survives `OptFom_Build_TXFFE`'s leading-zero trim.

### Setting 2 — the adaptive-search radius floor

`min_radius = 2`, the 4p16p0 rule for a multi-candidate grid. The 4p15p0 path
forced `1`.

With the grid alone the result was **5 of 10**. Adding the radius floor took it
to **10 of 10**:

| case | MATLAB Tx FFE | Python | itick ML/PY | ΔFOM |
|---|---|---|---|---|
| wXtalk_T1_R07 | `[0, −0.04, 0.96, 0]` | same | −8 / −8 | −1.2e−11 |
| wXtalk_T1_R08 | `[0, −0.04, 0.96, 0]` | same | −6 / −6 | −1.5e−11 |
| wXtalk_T1_R15 | `[0, −0.02, 0.98, 0]` | same | −3 / −3 | −3.2e−12 |
| wXtalk_T1_R16 | `[0, −0.1, 0.9, 0]` | same | 0 / 0 | −2.3e−11 |
| wXtalk_T2_R15 | `[0, −0.02, 0.98, 0]` | same | −4 / −4 | −4.7e−12 |
| wXtalk_T2_R16 | `[0, −0.06, 0.94, 0]` | same | 1 / 1 | −1.6e−11 |
| wXtalk_T3_R16 | `[0, −0.04, 0.96, 0]` | same | 2 / 2 | −1.5e−11 |
| wXtalk_T3_R17 | `[0, −0.1, 0.9, 0]` | same | 6 / 6 | −1.7e−12 |
| wXtalk_T3_R07 | `[0, −0.02, 0.98, 0]` | same | −8 / −8 | −9.7e−12 |
| wXtalk_T3_R15 | `[0, −0.02, 0.98, 0]` | same | −3 / −3 | −5.0e−12 |

### Why the attribution to min_radius is sound

The 4p15p0-vs-4p16p0 comparison over all 208 cases shows COM, FOM, VEO, VEC,
itick and ERL **identical** on a single-point Tx FFE grid. `min_radius` is the
only 4p16p0 change that is `num_txffe_runs`-dependent, so it is the only one that
can act once the grid has more than one point.

### Engine change

`Overwrite_Min_Radius` is now honoured in **both** version paths. It was read only
under 4p16p0, so a config setting it while emulating 4p15p0 had it silently
discarded — the same silent-config-discard class as the Tx FFE defect itself.

## 11. The two reference workbooks were produced with different Tx FFE configs

Traced because the two workbooks report this column differently:

| workbook | column | implies |
|---|---|---|
| WithXtalk | `TXLE_taps_1..4` | 4-element vector, so `c(-2)` survived the trim → **swept** grid |
| WithoutXtalk | single `TXLE_taps` = 1 | fully trimmed `[1.0]` → **single-point** grid |

**Nothing in `com_ieee8023_4p15p0.m` makes that length depend on crosstalk:**

- `param.tx_ffe_c*_values` is assigned in exactly one place, the config read
  (L9858–9864). Nothing else in the file modifies it.
- `OptFom_Build_TXFFE(param)` takes only `param` — no `chdata`, no crosstalk
  flag, no channel count.
- `output_args.TXLE_taps = fom_result.txffe` (L4081) is the only write, with no
  branch.
- `num_pre` counts param *fields*, which `xls_parameter` always creates.
- The workbook is not written by `com_ieee8023_` at all — it only writes a
  keywords CSV — so the column naming comes from an external harness flattening
  whatever length the field has.

So the tap-vector length is a pure function of the config, and the two runs used
different `c(-2)` settings **despite recording the same config filenames**.

That also explains, with no further hypothesis, why COM Python matches woXtalk
**104/104** on FOM, COM and itick while missing exactly 10 wXtalk cases: the
config we were given is the **woXtalk** one.

*(Checked and dismissed: the workbooks record `PKGA` for Cases 3–4 where our files
are named `PKGB`. `Pkg_len_TX_1` is 12/33/30/45 in both, matching our configs, so
the package content agrees and the filename difference is cosmetic.)*

---

## 12. Final result — matched configuration

Running each condition on the config its reference actually used
(produced with `tools/matlab_compare.py --run --modal-erl --jobs 5`, local tooling that is not in the repository — README §1; the standalone comparison script this section originally used has been removed, since there is no longer a second reading to compare against):

| | result |
|---|---|
| cases | 208 |
| **FOM bit-exact** | **208 / 208** |
| **COM bit-exact** | **208 / 208** |
| **sampling phase (`itick`) exact** | **208 / 208** |
| max \|ΔCOM\| | **3.3e-14 dB** |
| rms ΔCOM | **0.000529 dB** |
| pass/fail disagreements | **0** |

There is no COM miss: every case agrees to within 3.3e-14 dB. The last one,
`wXtalk_T4_R10`, was closed on 2026-08-22 by a banker's-rounding fix in
`get_pdf` (`MATLAB_Correlation_Review.md` §4.3).

### The 2×2 control

| condition | config | FOM | COM | itick | max \|ΔCOM\| |
|---|---|---|---|---|---|
| woXtalk | **own** (as supplied) | **104** | **104** | **104** | **0.000000** |
| woXtalk | other (swept) | 88 | 88 | 91 | 0.229710 |
| wXtalk | **own** (swept + min_radius 2) | **104** | **104** | **104** | 0.000000 |
| wXtalk | other (as supplied) | 94 | 94 | 96 | 0.185248 |

Each condition is near-exact on its own config and materially worse on the
other's. That is the evidence that the two reference runs used different Tx FFE
settings — it is not merely that a swept grid helps wXtalk, but that the unswept
grid is *required* for woXtalk.

### Standing caveat

**The wXtalk grid is a reconstruction, not a config we were sent.** It is
supported by reproducing MATLAB's tap vector, sampling phase and FOM on all ten
previously divergent cases, and by the 2×2 above — but it should be confirmed
against the real settings before the numbers are quoted as a like-for-like
correlation.

---

## 13. What MATLAB actually selected — only `c(-1)` ever moves

The sections above establish that `c(-2)` was **in** MATLAB's grid: the
with-crosstalk workbook reports a 4-element `TXLE_taps`, and
`OptFom_Build_TXFFE` trims leading precursor taps that are single-valued zeros,
so a tap that survives the trim was multi-valued.

That is a fact about the **search space**. It is easy to slide from there to "so
`c(-2)` is the tap that differs" — which the data does not support. Measuring
the winners separates the two (`tools/txffe_census.py`, from the reference
workbooks):

| `[ c(-2)  c(-1)  c(0)  c(1) ]` | cases | which |
|---|---|---|
| `[ 0  0  1  0 ]` | 94 | |
| `[ 0  −0.02  0.98  0 ]` | 4 | T1_R15 T2_R15 T3_R07 T3_R15 |
| `[ 0  −0.04  0.96  0 ]` | 3 | T1_R07 T1_R08 T3_R16 |
| `[ 0  −0.10  0.90  0 ]` | 2 | T1_R16 T3_R17 |
| `[ 0  −0.06  0.94  0 ]` | 1 | T2_R16 |

Without crosstalk, all 104 cases are unity.

So across all 208 reference cases:

- **`c(-2)` and `c(1)` win at zero every single time.**
- **`c(-1)` is the only tap that ever moves**, on exactly the 10 divergent
  cases, taking −0.02, −0.04, −0.06 or −0.10.
- `c(0)` is not independent — the cursor is `1 − Σ|other taps|`.

### Why this matters for the ask

The outcome turns on the **range given to `c(-1)`**, which the supplied config
pins to a single zero. `c(-2)` and `c(1)` matter only in that they enlarge the
candidate count, which is what brings `min_radius` into play — a search-reach
effect, not a search-space one.

This narrows the request to Hansel from "send the Tx FFE settings" to "confirm
the `c(-1)` range", which is a question he can answer from one cell.

### What this does *not* establish

That the `c(-2)` and `c(1)` sweeps are unnecessary to *reproduce* the result.
They change the candidate count, and `min_radius` is what made the difference
between 5 of 10 and 10 of 10 — so removing them could plausibly change which
`c(-1)` the adaptive search reaches. The census constrains the *answer*, not the
*search*. Testing that would take a run with `c(-2)` fixed at zero; it has not
been done, and the reconstruction keeps all three swept.
