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

> **Which Tx FFE tap ranges were active in the run that produced these
> workbooks?**

The configs as supplied set `c(-1)`, `c(-2)` and `c(1)` to `0` in the value
column, which yields a single unity Tx FFE — but the results contain non-unity
winners on 10 cases, so the run used a different setting. Specifically: was
`c(-2)` set to `[ 0.14:.02:0]` (which evaluates empty) or to something else, and
were `c(-3)`/`c(-4)` in play?

With that, the corpus can be re-run on a matched search space and the remaining
ten cases should close.

## 6. Consequence for the runtime comparison — flagged, not yet measured

**The published runtime comparison is not like-for-like.** Python has been
searching 1 Tx FFE candidate per CTLE where MATLAB searched a swept grid — 198
combinations from the `c(-1)`/`c(1)` ranges alone, before adaptive pruning. The
one measurement taken so far: `wXtalk_T1_R07` runs in **1.1 min** as supplied and
roughly **15–20 min** with the sweep enabled.

Any claim that the Python port is faster than MATLAB must be re-measured on a
matched search space before it is repeated.

This does **not** affect the adaptive-vs-full-grid result, which compares Python
against Python on one and the same grid.
