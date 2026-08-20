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

## 3. Confirmation by experiment

Injecting the column-C ranges into `param` after the config is read — changing
nothing else — and re-running `wXtalk_T1_R07`:

| | Tx FFE | itick | FOM |
|---|---|---|---|
| Python, as supplied (1 grid point) | `[1]` | −7 | 12.4255 |
| **Python, sweep enabled (198 points)** | `[−0.02, 0.98, 0]` | −7 | **12.4361** |
| MATLAB reference | `[0, −0.04, 0.96, 0]` | −8 | 12.4432 |

Python moves off unity and its FOM rises toward MATLAB's. It does not land
exactly on MATLAB's answer, and the tap vector shows why: Python returns **3**
taps, MATLAB **4**. `OptFom_Build_TXFFE` trims leading single-valued zero taps
until the first non-trivial one, so a 4-element result means MATLAB's grid kept
the `c(-2)` position while mine trimmed it — Hansel's config had something at
`c(-2)` that my injection did not reproduce.

The mechanism is established; the exact grid is not.

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
