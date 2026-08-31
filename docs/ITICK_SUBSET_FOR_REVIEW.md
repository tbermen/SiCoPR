# The eight sampling-phase cases — subset for review

> **SUPERSEDED 2026-08-20 as to CAUSE** — see `docs/TXFFE_SWEEP_ROOT_CAUSE.md`.
> The eight are a **config mismatch**, not an engine defect. MATLAB's winning
> Tx FFE is non-unity on exactly the 10 cases that disagree (94/94 bit-exact
> where it is unity, 0/10 where it is not), and the supplied configs define a
> Tx FFE grid with a single unity point, so Python could never select those
> equalizers. The measurements below stand; the explanation in section 5-6 does not.

Eight of the 208 reference cases select a different sampling phase (`itick`) from
MATLAB. This isolates them: what the correlation looks like without them, what
they have in common, what the search actually does, and the one question that
would settle it.

Headline: the search is not at fault (every tick is evaluated, §4), the crosstalk
model is not at fault (§3b), and `itick` is an offset from an origin that moves
with the equalizer (§5) — so several of the eight may not be sampling at
different instants at all.

Produced with `tools/itick_subset_report.py` (local tooling, not in the repository — README §1).
Machine-readable tables: `report_data/itick_subset.csv` (8 rows, 34 columns) and
`report_data/fom_surface_<case>.csv`, produced with `tools/fom_surface_probe.py` (local tooling).

---

## 1. Correlation with and without them

| set | FOM bit-exact | COM bit-exact | itick exact | max \|ΔCOM\| | rms ΔCOM | pass/fail flips |
|---|---|---|---|---|---|---|
| all 208 | 198 / 208 (95.2%) | 170 / 208 (81.7%) | 200 / 208 (96.2%) | 0.185248 | 0.018660 | 1 |
| **the 200 agreeing ticks** | **198 / 200 (99.0%)** | **170 / 200 (85.0%)** | 200 / 200 (100%) | **0.028201** | **0.004946** | 1 |
| the 8 mismatched | 0 / 8 | 0 / 8 | 0 / 8 | 0.185248 | 0.091876 | 0 |

Excluding them, **FOM is bit-exact on 99.0%**, the worst COM difference drops
**6.6×** (0.185 → 0.028 dB) and rms drops **3.8×** (0.0187 → 0.0049 dB).

> Figures updated 2026-08-20, after the `BEST.PSD_results` value-copy fix
> (`git log --grep="BEST.PSD_results aliased"`, `docs/STAGE6_NOISE_AGREEMENT.md` §8). That fix took COM bit-exact
> from 135 to 170 of 208 and pass/fail flips from 2 to 1. It left FOM (198) and
> `itick` (200) untouched, because the optimiser always rebuilt the tick-dependent
> PSDs correctly — so everything below about the eight still stands unchanged.

Two things worth noting, because they say these eight are not the only story:

- **The remaining pass/fail flip is not in this set.** `wXtalk_T2_R06` and
  `wXtalk_T2_R24` both have matching ticks (one of the two was resolved by the
  `BEST.PSD_results` fix); they sit at COM 3.007 vs 2.997 and
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

## 3b. Is the crosstalk contribution itself the source?

The natural reading of "all eight are with-crosstalk" is that the crosstalk noise
computation is where the discrepancy enters. Tested three ways; all three say no.

**(a) The crosstalk computation is bit-exact.** Every channel-level crosstalk
quantity agrees on all 208 cases:

| quantity | exact |
|---|---|
| `ICN_mV` | 208 / 208 |
| `MDFEXT_ICN_92_47_mV` | 208 / 208 |
| `MDNEXT_ICN_92_46_mV` | 208 / 208 |
| `SNR_MDFEXT` | 104 / 104 (all wXtalk) |

**(b) The post-equalization crosstalk terms disagree only where the sampling
point already differs.** `sgm_xt`, `peak_MDFEXT_interference_at_BER_mV`,
`peak_MDNEXT…` and `peak_MDXTK…` are each exact on 198/208, and the ten that are
not are precisely the 8 tick mismatches plus the 2 cases with non-exact FOM
(`wXtalk_T3_R07`, `wXtalk_T3_R15`). They move because the eye is being sampled
somewhere else — consequence, not cause.

**(c) The one crosstalk-only *decision* does not separate the sets.** `get_PSDs`
chooses each aggressor's sampling phase by `argmax` over M candidate norms —
code that runs only for with-crosstalk cases, and a plausible place for a
discrete flip. Measured on the 8 divergent cases and 6 matched with-crosstalk
controls, the margin between the best and second-best phase is:

| set | median min-margin | range |
|---|---|---|
| divergent (8) | 1.58e-5 | 3.7e-6 … 1.0e-4 |
| control (6) | 2.31e-5 | 3.8e-6 … 1.1e-4 |

The distributions overlap almost completely (83% of control values fall inside
the divergent range) and the divergent median is *larger*, not smaller, than a
tie hypothesis would predict. At ~1e-5 these margins are also ten orders of
magnitude above floating-point noise, so a last-bit difference could not flip
the choice in any case.

### What crosstalk does do

The correlation splits cleanly by condition:

| | FOM bit-exact | itick exact | max \|ΔCOM\| |
|---|---|---|---|
| woXtalk | **104 / 104** | **104 / 104** | 0.034719 |
| wXtalk | 94 / 104 | 96 / 104 | 0.175602 |

Without crosstalk the two engines agree on every FOM and every sampling phase.
So crosstalk is the **enabling condition**, not the defect: `S_xn` enters the
MMSE noise, which changes the RxFFE solve and therefore the shape of FOM versus
sampling phase. It is what makes the tick selection contested enough for a small
underlying difference to change the winner — while the crosstalk arithmetic
itself is exact.

That is worth stating precisely to Hansel, because "it only happens with
crosstalk" invites the assumption that the crosstalk model differs, and the data
says it does not.

## 4. The search is not the cause — every tick is evaluated

Before blaming anything numerical, the search itself had to be ruled out. For
all four 208-case configs:

| switch | value | consequence |
|---|---|---|
| `TS_SRCH_MODE` | `full-sweep` | `middle_search = 0`, so the whole `ts_sample_adj_range` (−24…+24, 49 ticks) is swept in order |
| `LOCAL_SEARCH` | 2 | but `OptFom_Itick_LocalSearch` is gated on `middle_search`, so it never skips a tick |
| `RxFFE` + `FFE_OPT_METHOD` | on + `MMSE` | `RxFFE_with_MMSE = 1` |

`RxFFE_with_MMSE = 1` has three effects that matter here, identical in both
engines (ML 8772-8785, 3613-3622, and `OptFom_Compute_RxFFE`):

- `OptFom_Calc_Noise`'s abort bound (`20log10(A_s/sigma_ISI) < Best_FOM`, the one
  piece of history-dependent pruning in the tick loop) is **disabled**;
- `OptFom_Compute_RxFFE`'s `RXFFE_Illegal` early-out is **skipped**;
- `OptFom_Calc_FOM` is **never called** — the FOM is `MMSE_results.FOM`.

So the winning tick is a **plain argmax over a fully evaluated 49-point
surface**. There is no pruning, no early termination, and no order dependence
left to blame. Measured on four cases, Python scores 49/49 ticks for every EQ
candidate, confirming this. Whatever differs has to be a difference in FOM
*values*, not in which values got looked at.

## 5. `itick` is measured from an origin that moves

`ts_anchor = 1` in all four configs, so the anchor is

```
raw_cursor_i = argmax(sbr[Peak_Search_Range])      (ML 3529-3531)
THIS.cursor_i = raw_cursor_i + THIS.itick          (ML 8766)
```

recomputed **for every EQ candidate**, on that candidate's own equalized pulse.
The peak of the equalized pulse moves as the equalizer changes, so the anchor
moves with it. Measured (`tools/fom_surface_probe.py`):

| case | EQ candidates | distinct anchors | anchor spread |
|---|---|---|---|
| wXtalk_T1_R01 | 17 | 3 | 2 samples |
| wXtalk_T1_R07 | 25 | 5 | 4 samples |
| wXtalk_T1_R16 | 20 | 4 | 3 samples |
| wXtalk_T3_R17 | 20 | 3 | 2 samples |

The consequence is direct. Taking only the EQ candidates within 0.5 dB of the
best, and asking where each one's own optimum sits:

| case | argmax as `itick` | argmax as *absolute* sample |
|---|---|---|
| wXtalk_T1_R16 | 3, 4, 5 (3 labels) | 28357, 28358 (**2 samples**) |
| wXtalk_T1_R07 | −8, −7, −6, −5 (4 labels) | 23971, 23972, 23973 (3 samples) |
| wXtalk_T3_R17 | −20, −19, −18 (3 labels) | 14703, 14704, 14705 (3 samples) |

On R16, seventeen near-optimal equalizer settings agree on **two** physical
sampling instants but disagree across **three** `itick` labels. The label is
noisier than the thing it labels.

**So an `itick` difference of a few samples between two engines that chose
slightly different equalizers is expected, and does not by itself mean they
sampled at different instants.** Five of the eight have Δ = +1 or +2, and three
of the eight land on a different winning CTLE from MATLAB — the regime where this
effect is largest.

How far it goes is worth stating precisely rather than assuming, because the
measured spread does not cover all eight:

| case | Δ | Python absolute cursor | anchor MATLAB would need | Python's anchors | covered? |
|---|---|---|---|---|---|
| wXtalk_T1_R07 | +1 | 23972 (anchor 23979, itick −7) | 23980 | 23977…23981 | **yes** |
| wXtalk_T1_R16 | +5 | 28357 (anchor 28352, itick +5) | 28357 | 28351…28354 | no — 3 beyond |
| wXtalk_T3_R17 | −25 | 14704 (anchor 14723, itick −19) | 14698 | 14722…14724 | no — 24 beyond |

For R07 the anchor MATLAB would need in order to be sampling Python's instant
falls inside the range Python itself produced across its 25 candidates, so anchor
movement is a sufficient explanation. For R16 it is 3 samples beyond anything
Python produced — plausible, since MATLAB won on a different CTLE (−20 vs −19),
but **not** established. For R17 it is 24 samples out, which anchor movement
cannot explain; that case is the bimodal one (§6).

So this mechanism accounts for some of the eight, not all of them. `cursor_i`
would say which.

This also retires the earlier "anchor-origin offset" reading. There is no fixed
frame origin to be offset: with `ts_anchor = 1` the origin is an `argmax` that is
re-derived per candidate, and both engines re-derive it the same way.

## 6. What the FOM surfaces actually look like

`tools/fom_surface_probe.py` dumps FOM vs tick for every EQ candidate. Best
Python FOM at MATLAB's reported tick, maximised over **all** EQ candidates:

| case | Python peak | at tick | MATLAB peak | at tick | Python at MATLAB's tick | short by |
|---|---|---|---|---|---|---|
| wXtalk_T1_R01 (control) | 12.0943 | −6 | 12.0943 | −6 | 12.0943 | 0.0000 |
| wXtalk_T1_R07 | 12.4255 | −7 | 12.4432 | −8 | 12.4015 | 0.0416 |
| wXtalk_T1_R16 | 11.6070 | +5 | 11.6910 | 0 | 6.2573 | **5.4337** |
| wXtalk_T3_R17 | 13.8662 | −19 | 13.8016 | +6 | 11.4398 | **2.3618** |

Two observations:

- **The peak values agree to 0.005–0.084 dB in every case** — including
  `wXtalk_T3_R17`, where Python's peak is *higher* than MATLAB's. Both engines
  find equally good solutions.
- The surfaces are smooth, not notched. R16 climbs monotonically 0→+5
  (6.26, 8.57, 10.79, 11.49, 11.54, 11.61); there is no isolated hole at
  MATLAB's tick. A 5-sample offset of a smooth ramp is what a 5-sample anchor
  difference looks like.

`wXtalk_T3_R17` is a different animal and should be treated separately: its
surface is **bimodal**, with 13.866 at −19 and a second, lower optimum of 13.680
at +2 — two basins 21 ticks apart and 0.187 dB apart in height. MATLAB's reported
13.802 sits between the two, 0.065 dB below Python's peak and 0.12 dB above
Python's secondary one, at tick +6. So the engines settled in different basins on
a surface that has more than one, and the winning heights are close. This is not
an anchor effect (§5 shows the anchor cannot move 24 samples) and not obviously a
defect; it is a genuinely contested optimum.

Note this is compatible with the earlier finding that the landscapes are sharply
peaked — only 2 of 49 ticks lie within 0.1 dB of the peak here too. "Bimodal"
refers to two separated local maxima, not to a flat top.

## 7. What was ruled out in the Python code

Read against the 4p15p0 source, site by site:

| checked | verdict |
|---|---|
| phase arithmetic (`mod(cursor_i-1,M)+1`, `mod(cursor_i,M)`) at all 5 `get_PSDs` sites | equivalent |
| `sampling_offset<=1` guard and early/late cursor slicing | equivalent |
| `MMSE` array build: `dh`, `isi_start`/`isi_end`, zero-pad to `num_ui`, toeplitz row-trim branch | equivalent |
| `MMSE_FOM` clipping branches (`~isequal` exact-equality tests, b-refresh placement) | equivalent |
| `FOM_rxffe_floating_taps` greedy bank search: positional-then-value removal, `max` first-wins tie-break | equivalent (sequential removal == union removal) |
| `Peak_Search_Range` construction and `ts_anchor` handling | equivalent |

Three latent defects were found that are **not** active in these configs, and so
cannot explain the eight, but are real and would bite under other settings:

1. `OptFom_Setup_Sampler_Sweep` uses `np.argsort(np.abs(full_sample_range))`.
   NumPy's default is quicksort, which is **not stable**; MATLAB's `sort` is. On
   a symmetric range every ±k pair ties, and the orders genuinely differ
   (`… −2, 2, 3, −3, −4 …` vs `… −2, 2, −3, 3, −4 …`). Dead here because
   `TS_SRCH_MODE = 'full-sweep'` never reads `si`; live for `'middle'`, where
   visit order drives the pruning.
2. `OptFom_Calc_Noise` writes five fields into `THIS` on its abort path
   (`h_J`, `sigma_TX`, `ISI_N`, `sigma_N`, `total_noise_rms`) where MATLAB's
   early `return` leaves `THIS` untouched — the by-reference-vs-by-value class
   that caused 5 of the original 8 defects. Dead here because
   `RxFFE_with_MMSE = 1` makes the whole block unreachable.
3. `_findbankloc` in `MMSE` is a self-described "simplified" reimplementation of
   MATLAB's `findbankloc`, not a translation of it. Dead here because
   `RXFFE_FLOAT_CTL = 'FOM'`; live for `'ISI'`.

## 8. The question for Hansel

To be clear about what this is and is not: **nothing here suggests a defect in
the MATLAB engine.** Both engines sweep the same 49 ticks with no pruning, and
their peak FOM values agree to within 0.084 dB on all four cases examined in
detail. The problem is that the reported quantity cannot distinguish the two
explanations that remain.

`itick` is an offset from `raw_cursor_i = argmax(sbr[Peak_Search_Range])`, which
is re-derived per EQ candidate and was measured to move by 2–4 samples within a
single case (§5). So two engines that settle on slightly different equalizers
report `itick` on different origins. Given only `itick`, "sampled at a different
instant" and "sampled at the same instant, labelled from a different anchor" look
identical in the outputs.

**The request: `cursor_i` — or the absolute `t_s` — reported alongside `itick`.**

One extra column, and it is diagnostic rather than accusatory: it puts both
engines on a common axis so the eight can be sorted into the two buckets.

Concretely, for each of the eight, compare `cursor_i` directly:

- **equal** → the engines sampled the same instant and the `itick` difference is
  pure anchor movement. Nothing to fix; the correlation is better than the
  headline number suggests.
- **different** → a genuine difference in the FOM surface, and the size of the
  gap points at where. `wXtalk_T3_R17` is already known to be this kind: its
  surface is bimodal with two optima 0.065 dB apart at ticks 25 samples apart,
  so the engines picked different basins (§6).

Useful alongside it, and cheap if the run is being done anyway: `raw_cursor_i`
and the winning CTLE/TxFFE indices, which would let the anchor movement be
confirmed directly rather than inferred.

### Related, and now less coupled than it was

The stage-6 noise residual (`docs/STAGE6_NOISE_AGREEMENT.md`) also turns on the
sampling anchor: terms built by decimating the pulse at
`cursor_i % samples_per_ui` disagree, while the analytic term that uses no cursor
is exact, and the error grows with distance from `itick = −8`. The same column
would settle both. But note §7 — the phase arithmetic itself has now been checked
site by site against 4p15p0 and is equivalent, so a stated-convention mismatch is
no longer the leading candidate there either.
