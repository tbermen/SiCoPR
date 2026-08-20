# The eight sampling-phase cases — subset for review

Eight of the 208 reference cases select a different sampling phase (`itick`) from
MATLAB. This isolates them: what the correlation looks like without them, what
they have in common, and the one question that would settle it.

Regenerate with `python tools/itick_subset_report.py`.
Machine-readable table: `report_data/itick_subset.csv` (8 rows, 34 columns).

---

## 1. Correlation with and without them

| set | FOM bit-exact | COM bit-exact | itick exact | max \|ΔCOM\| | rms ΔCOM | pass/fail flips |
|---|---|---|---|---|---|---|
| all 208 | 198 / 208 (95.2%) | 135 / 208 (64.9%) | 200 / 208 (96.2%) | 0.175602 | 0.018592 | 2 |
| **the 200 agreeing ticks** | **198 / 200 (99.0%)** | 135 / 200 (67.5%) | 200 / 200 (100%) | **0.034719** | **0.006266** | 2 |
| the 8 mismatched | 0 / 8 | 0 / 8 | 0 / 8 | 0.175602 | 0.089477 | 0 |

Excluding them, **FOM is bit-exact on 99.0%**, the worst COM difference drops
**5×** (0.176 → 0.035 dB) and rms drops **3×** (0.0186 → 0.0063 dB).

Two things worth noting, because they say these eight are not the only story:

- **The 2 pass/fail flips are not in this set.** `wXtalk_T2_R06` and
  `wXtalk_T2_R24` both have matching ticks; they sit at COM 3.007 vs 2.997 and
  3.007 vs 3.000, i.e. within 0.01 dB of the 3 dB threshold. They belong to the
  separate COM-PDF residual, not to this.
- **Only 2 of the 200 have non-exact FOM** — `wXtalk_T3_R07` (ΔFOM −8.7e-4) and
  `wXtalk_T3_R15` (−1.7e-3). Everything else in the agreeing set is exact.

## 2. The eight

| case | test | family | itick py / ml | Δ | ΔCOM | ΔFOM |
|---|---|---|---|---|---|---|
| wXtalk_T1_R07 | 1 | DAC | −7 / −8 | +1 | −0.0180 | −0.0176 |
| wXtalk_T1_R08 | 1 | DAC | −5 / −6 | +1 | −0.0418 | −0.0190 |
| wXtalk_T1_R15 | 1 | DAC | −2 / −3 | +1 | −0.0611 | −0.0052 |
| wXtalk_T1_R16 | 1 | DAC | +5 / 0 | +5 | −0.1756 | −0.0841 |
| wXtalk_T2_R15 | 2 | DAC | −3 / −4 | +1 | −0.0074 | −0.0068 |
| wXtalk_T2_R16 | 2 | DAC | +3 / +1 | +2 | −0.0543 | −0.0283 |
| wXtalk_T3_R16 | 3 | DAC | +4 / +2 | +2 | −0.0704 | −0.0369 |
| wXtalk_T3_R17 | 3 | BPK twinax | −19 / +6 | −25 | +0.1395 | +0.0646 |

Python's tick is **later than or equal to** MATLAB's in seven of eight
(Δ = +1 on four of them). `wXtalk_T3_R17` is the outlier in both magnitude
(Δ = −25) and sign, and is the only non-DAC case.

## 3. What they have in common

**Every one is with-crosstalk.** 8 of 104 wXtalk cases; **0 of 104 woXtalk**.

| axis | distribution |
|---|---|
| condition | wXtalk 8/104 (7.7%) · **woXtalk 0/104 (0.0%)** |
| package test | T1 4/52 · T2 2/52 · T3 2/52 · **T4 0/52** |
| family | DAC 7/96 (7.3%) · BPK twinax 1/56 · **OSFP 0/32 · li_dj CR 0/24** |

Compared **like for like against the other 96 with-crosstalk cases** (comparing
against all 200 would be meaningless, since the 104 woXtalk cases have ICN = 0 by
construction):

| MATLAB-reported quantity | the 8 | wXtalk peers | ratio |
|---|---|---|---|
| MDFEXT_ICN_92_47_mV | 1.2764 | 0.89457 | **1.43** |
| MDNEXT_ICN_92_46_mV | 0.27076 | 0.51719 | **0.52** |
| sgm_isi | 7.114e-4 | 4.943e-4 | **1.44** |
| sgm_xt | 3.118e-4 | 2.278e-4 | 1.37 |
| IL_dB_channel_only_at_Fnq | 30.601 | 28.146 | 1.09 |
| IL_db_die_to_die_at_Fnq | 36.656 | 39.715 | 0.92 |
| ERL | 13.231 | 13.773 | 0.96 |

So: **FEXT-dominated rather than NEXT-dominated, with ~44% more residual ISI**, a
slightly lossier channel but *less* die-to-die loss, and marginally worse return
loss. Consistent with short, low-loss, tightly-coupled assemblies where the
equalizer search is least decisive.

### One apparent difference that is not real

The raw medians suggested a much shorter package (`Pkg_len_TX_1` 21 vs 33). That
is an artifact of the test mix — the eight avoid Test_4 entirely, and Test_4 has
the longest package. Per test the values are identical:

| test | the 8 | peers |
|---|---|---|
| 1 | 12 | 12 |
| 2 | 33 | 33 |
| 3 | 30 | 30 |
| 4 | — | 45 |

Package length is **not** a factor. Included because the pooled number looks like
a strong signal and is not one.

## 4. The question for Hansel

These eight are the cases where the two engines disagree about *where in the UI*
to sample. Three facts constrain the explanation:

1. **Python cannot reach MATLAB's reported FOM at MATLAB's reported tick under
   any equalizer setting** — short by 2.36 dB and 5.43 dB on the two worst —
   **yet the peak FOM values agree to 0.06–0.09 dB.** Both engines find the same
   quality of solution; they disagree about the coordinate it sits at.
2. **Adaptive pruning is not responsible.** An exhaustive full-grid run
   reproduces Python's answer on 9 of 9 of the hardest cases, including
   `wXtalk_T1_R16`, where the full grid independently lands on Python's tick.
3. **A second, independent investigation lands on the same unknown.** Stage-6
   noise terms built by decimating the pulse at `cursor_i % samples_per_ui`
   disagree, while the analytic term that uses no cursor is exact; the error is
   ~1e-12 at `itick = −8` and grows with distance from it. See
   `docs/STAGE6_NOISE_AGREEMENT.md`.

**The request: `cursor_i`, or the absolute `t_s`, reported alongside `itick`.**

One extra column. If `cursor_i − SBR_peak ≠ itick` on these eight, the divergence
is a frame-origin convention rather than a numerical disagreement, and the
stage-6 residual is explained at the same time. The workbooks currently expose
only `itick`, so the origin is unobservable from the outputs.

Falsifiable either way: `sigma_TX` should agree exactly wherever the two cursors
differ by a whole multiple of `samples_per_ui`, and not otherwise.
