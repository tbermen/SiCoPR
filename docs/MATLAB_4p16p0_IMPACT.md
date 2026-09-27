# 4p16p0 vs 4p15p0 — measured impact on the 208-case corpus

> **Dated (noted 2026-09-26):** last edited 2026-09-01, so measured on an engine that
> predates the September 2026 oracle fixes and speed-up changes; not re-run since. The run times
> below predate the equivalence rule in [`VERIFICATION.md`](VERIFICATION.md).

Both sweeps run on the same 208 reference cases, same channels, same configs,
same engine build; the only difference is `--matlab-version`. 4p16p0 completed
208 ok / 0 failed in 0.6 h at `--jobs 5`.

Both versions were run over the full case set with the local comparison harness,
then diffed column by column. That tooling is not in the repository (README
section 1).

---

## Result

**210 of 213 numeric output columns are identical on all 208 cases.**

| column | cases changed | new / old | max abs delta |
|---|---|---|---|
| `peak_uneq_pulse_mV` | 208 / 208 | **0.385000** | 143.64 mV |
| `steady_state_voltage_mV` | 208 / 208 | **0.385000** | 235.03 mV |
| `Pmax_by_Vf_est` | 120 / 208 | 1.000000 | 3.3e-16 |

Every headline column is bit-identical across the whole corpus:

```
COM_dB  FOM  VEO_mV  VEC_dB  itick  ERL  DER_MLSE  CTLE_DC_gain_dB  RxFFEgain
```

### Reading the two real changes

The ratio is **exactly 0.385000 on every case**, and the active configs declare
`A_v = 0.385`. So both columns move by precisely the victim channel amplitude —
which is what the 4p16p0 edit does and nothing more:

```matlab
chdata(i).uneq_pulse_response = chdata(i).uneq_pulse_response*chdata(i).A;   % new in 4p16p0
chdata(i).uneq_step_response  = chdata(i).uneq_step_response *chdata(i).A;   % new in 4p16p0
```

In 4p15p0 the pulse response was built from the *unscaled* impulse while only
the impulse itself got the `A` factor, so the two were inconsistent. 4p16p0
makes them agree. The corrected values are the physically meaningful ones —
`peak_uneq_pulse_mV` in 4p15p0 was the pulse of a Tx driving full amplitude
rather than its actual launch amplitude.

`Pmax_by_Vf_est` moving by 3e-16 on 120 cases is last-bit floating-point noise:
it is derived from the pulse response, so rescaling changes the rounding path.
Not a behavioural change.

**The concern that this might reach COM is settled: it does not.** COM is built
from `uneq_imp_response`, which was already scaled in 4p15p0. The change is
confined to two reported columns.

---

## What this sweep does NOT tell you

Three of the six 4p16p0 changes were never exercised, because this corpus does
not create the conditions for them. Stating that plainly matters more than the
clean headline:

| change | why it did not fire here |
|---|---|
| `min_radius` 1 → 2 | **all 208 cases sweep exactly one TXFFE candidate** (`1 gffe x 21 CTLE x 7 g_DC_HP x 1 TXFFE x 49 itick`). The mainline rule is `1 if num_txffe_runs == 1 else 2`, so it stays 1 corpus-wide. |
| `Clip Method` default `Fast` → `Slow` | all four configs set `Clip Method: slow` explicitly, so the default is never consulted. |
| common-mode and TDR degenerate guards | never triggered — `ERL` is finite on all 208 (0 cases with `ERL = inf`), and every SCMR column is unchanged. |

So the honest summary is: **of the behaviour-changing edits, this corpus
exercises the `A` scaling only.** The other two need different inputs to
evaluate:

- **`min_radius`** needs a config with a real Tx FFE grid. The CAKR config used
  for the single-case checks sweeps 1584 TXFFE candidates and would exercise it
  directly. Expect a pruning-radius change, and therefore a possible runtime and
  candidate-count change; the earlier adaptive-vs-full-grid study is the right
  gate for whether the answer moves.
- **`Clip Method`** needs a config that omits the keyword. Worth a targeted A/B
  on one case rather than a corpus sweep, since the mechanism is a one-line
  default.

---

## The two changes the corpus could not reach — measured on CAKR

The CAKR config (`config_com_dj_200G_CAKR_178_PKGA_..._TXLEOn_mod_v1.xlsx`)
sweeps **1584 TXFFE candidates** and **does not set `Clip Method`**, so it
exercises both changes the 208-case corpus structurally could not. Three arms,
same channel and aggressors, chosen so each comparison moves one thing:

| arm | version | Clip Method | min_radius |
|---|---|---|---|
| A | 4p15p0 | Fast (default) | 1 |
| B | 4p16p0 | Slow (default) | 2 (mainline rule) |
| C | 4p16p0 | Slow (default) | 1 (`Overwrite Minimum Radius = 1`) |

### min_radius 1 → 2 — B vs C, nothing else differs

**Identical answer, 4.3x the work.**

| | C (min_radius 1) | B (min_radius 2) | |
|---|---|---|---|
| COM_dB | 5.602443154 | 5.602443154 | same |
| FOM | 14.828351473 | 14.828351473 | same |
| VEO_mV, VEC_dB, itick, CTLE, g_DC_HP, ERL | | | all same |
| candidates evaluated | 70 | **300** | **x4.29** |
| wall clock | 4.9 min | **12.0 min** | **x2.48** |

Every reported value is bit-identical; the larger radius floor simply stops the
search pruning. On this channel the mainline default costs 4.3x the candidate
evaluations and 2.5x the runtime and buys nothing.

That is worth passing back to Hansel: his branch forced `min_radius = 1`, and
this is evidence for that choice over the mainline's 2 — at least on a channel
of this class. It does not prove 2 is never useful, only that it is pure
overhead here.

### Clip Method Fast → Slow — A vs C, min_radius held at 1

**This one does move COM.**

| | A (Fast) | C (Slow) | delta |
|---|---|---|---|
| COM_dB | 5.595759755 | 5.602443154 | **+0.00668** |
| FOM | 14.613204542 | 14.828351473 | **+0.21515** |
| VEO_mV | 7.367591437 | 7.377174260 | +0.00958 |
| VEC_dB | 6.467293099 | 6.459910277 | −0.00738 |
| itick, CTLE_DC_gain_dB, g_DC_HP, ERL | | | all same |
| candidates evaluated | 70 | 70 | same |
| wall clock | 4.0 min | 4.9 min | x1.21 |

The equalizer solution is unchanged — same tick, same CTLE, same high-pass, same
number of candidates evaluated — so this is not a different search outcome. It
is the ADC-clip PDF being computed the exact way instead of the approximate one.
The other 4p16p0 edits in this arm cannot account for it: the `A` scaling was
shown COM-neutral across all 208 cases, and the new guards never fire here.

`Slow` is the more exact path, so 4p16p0 is making the accurate method the
default. The cost is about 20% runtime on this case.

**Consequence for any config that omits `Clip Method`: adopting 4p16p0 shifts
COM by roughly +0.007 dB and FOM by +0.22 dB.** Configs that name the keyword —
including all 208 reference cases — are unaffected.

## Recommendation

All six changes are now measured. Summary of what adopting 4p16p0 does:

| change | effect |
|---|---|
| `A` scaling of pulse/step | `peak_uneq_pulse_mV` and `steady_state_voltage_mV` x A (0.385 here). COM untouched. The new values are the correct ones. |
| `Clip Method` default Fast → Slow | **COM +0.007 dB, FOM +0.22 dB** on configs that omit the keyword; nothing on configs that set it. ~20% slower. |
| `min_radius` 1 → 2 | same answer, **4.3x the candidate evaluations, 2.5x the runtime** |
| new step responses | additive fields only |
| CM / TDR degenerate guards | never fired on any input tested |
| `OptFom_Create_Output`, `get_PSDs` | numerically neutral |

Nothing here blocks adoption. Two reported columns become correct, and the
`Clip Method` shift is toward the more exact computation.

**Keep 4p15p0 as the default anyway**, for one reason that has not changed: the
208-case reference workbooks are 4p15p0 output, and they are what the
correlation result (FOM bit-exact 198/208) rests on. Switching the default
without matching reference data would leave the port's main evidence pointing at
a version it no longer emulates.

Two things to raise with Hansel:

1. **`min_radius = 2` looks like a regression in the mainline.** His branch
   forced 1. On CAKR the mainline default evaluates 4.3x the candidates for a
   bit-identical answer. Worth asking what motivated 2, since it may help on
   channel classes not represented here.
2. **A 4p16p0 run of the same 208 cases** would let the default move. With
   adaptive search now mainline he has more reason to want the comparison than
   when this was a branch.

When that reference data arrives, the same harness handles it:
`--matlab-version 4p16p0` on both the sweep and the report.

---

## Per-field detail

A version-comparison table is written locally — one row per numeric column: cases compared,
cases changed, max absolute delta, and the new/old ratio range.
