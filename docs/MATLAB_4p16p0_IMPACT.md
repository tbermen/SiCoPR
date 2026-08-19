# 4p16p0 vs 4p15p0 — measured impact on the 208-case corpus

Both sweeps run on the same 208 reference cases, same channels, same configs,
same engine build; the only difference is `--matlab-version`. 4p16p0 completed
208 ok / 0 failed in 0.6 h at `--jobs 5`.

```
python tools/matlab_compare.py --run --jobs 5                          # 4p15p0
python tools/matlab_compare.py --run --jobs 5 --matlab-version 4p16p0
python tools/compare_matlab_versions.py
```

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

## Recommendation

Nothing here argues against adopting 4p16p0. Two reported columns get more
correct, no COM/FOM/VEO/VEC/itick/ERL value moves, and the corpus runs clean.

But keep 4p15p0 as the default until the two unexercised changes are measured —
particularly `min_radius`, which alters the search itself rather than a reported
number, and which this corpus structurally cannot test.

The correlation evidence continues to rest on 4p15p0, where the reference
workbooks came from. When 4p16p0 reference results become available, the same
harness compares against them by pointing `--matlab-version` at the new sweep.

---

## Per-field detail

`report_data/version_compare.csv` — one row per numeric column: cases compared,
cases changed, max absolute delta, and the new/old ratio range.
