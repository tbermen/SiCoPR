# The adaptive-search radius floor — a settled assumption

**Status: closed.** The port applies the 4p16p0 mainline rule on both version
paths. This document records what the assumption is, what evidence supports it,
what it cannot prove, and what would overturn it.

```
min_radius = 1 if num_txffe_runs == 1 else 2      # ML 4p16p0 L2782-2786
```

overridden by a positive `Overwrite Minimum Radius` in the configuration
workbook (ML 4p16p0 L2788-2792), on either version path.

---

## 1. What was uncertain

`min_radius` is the floor the adaptive local search may not shrink its pruning
radius below. It matters only when the search has something to prune — that is,
when the Tx FFE grid has more than one candidate.

The two MATLAB sources say different things:

| source | rule |
|---|---|
| `com_ieee8023_4p16p0.m` (mainline, L2782-2786) | `1` if `num_txffe_runs == 1`, else `2` |
| `com_ieee8023_4p15p0_adaptive_local_search.m` (the branch we hold) | `1`, unconditionally |

The reference workbooks were produced with the branch. Taken at face value, that
says the floor was 1. **It was not**, and the port therefore does not model the
branch here.

## 2. The evidence

**Primary — the ten disagreeing cases.** Ten crosstalk cases (`wXtalk`
T1_R07/R08/R15/R16, T2_R15/R16, T3_R07/R15/R16/R17) failed to reproduce the
MATLAB Tx FFE tap vector and sampling phase. Two settings were needed together;
each alone was insufficient:

| floor | Tx FFE grid | cases reproduced |
|---|---|---|
| 1 | single point (as the config literally reads) | 0 of 10 |
| 1 | 1584 candidates (the grid actually swept) | **5 of 10** |
| 2 | 1584 candidates | **10 of 10** |

At floor 2 the ten agree on the same tap vector, the same `itick`, and FOM to
≤ 2.3e-11. Commit `19f7d4d`.

**Corroborating — the floor is the only candidate mechanism.** The full
4p15p0-vs-4p16p0 comparison over all 208 cases found `COM`, `FOM`, `VEO`, `VEC`,
`itick` and `ERL` identical on a single-point Tx FFE grid
([`MATLAB_4p16p0_IMPACT.md`](MATLAB_4p16p0_IMPACT.md)). `min_radius` is the only
change in the 4p16p0 delta that depends on `num_txffe_runs`, so it is the only
one that can act once the grid has more than one point. Nothing else in the
delta could have produced the 5-of-10 → 10-of-10 move.

**Consistency check — the floor is not always outcome-changing.** On the CAKR
single-case study (also a 1584-candidate grid), floors of 1 and 2 gave a
bit-identical answer, with floor 2 costing 4.3× the candidate evaluations and
2.5× the wall clock. That is not a contradiction: a larger floor stops the
search pruning, which changes the answer only where the pruning would have
discarded the eventual winner. It did on those ten cases; it did not on CAKR.
The floor is a search-scope setting, and its effect is channel-dependent.

## 3. Why the branch source disagrees with its own results

The branch file forces `1`. The results it produced behave as `2`. The
reconciling explanation needs no theory about MATLAB being wrong:

The reference runs came out of an experimental workflow, not a version-controlled
release. Configurations and code were edited between experiments and the
supplied copies are snapshots of a later state, not of the state that produced
the results. **This is already established for the same run**: the supplied
configuration workbooks pin Tx FFE `c(-1)` to a single zero, while the MATLAB
that produced the workbooks swept 1584 candidates
([`TXFFE_SWEEP_ROOT_CAUSE.md`](TXFFE_SWEEP_ROOT_CAUSE.md)), confirmed by the COM
maintainer. The radius floor is the second instance of the same drift, in the
same run.

Either of two ordinary situations produces it — a working copy carrying the
mainline rule (the adaptive search was being adopted upstream into 4p16p0 at the
time), or a configuration setting `Overwrite Minimum Radius` that was edited out
before the workbook was shared. **Both are indistinguishable from the outputs,
and both give the same observable behaviour**, which is why the assumption is
stated as the behaviour rather than as a claim about Hansel's tree.

## 4. Why the rule, and not a constant

The correlation could equally be reproduced by forcing the floor to 2 everywhere
— that is what `--min-radius 2` on the harness did before this was settled. The
rule is preferred for three reasons:

1. **It exists in published MATLAB.** `1 if num_txffe_runs == 1 else 2` is the
   4p16p0 mainline; a flat 2 is a number this project invented.
2. **It removes an inferred flag from the reproduction recipe.** Anyone
   reproducing the correlation now runs the documented command with no special
   switch, which matters more for a public repository than for a private one.
3. **It cannot be wrong where it acts differently.** A flat 2 and the rule differ
   only on single-candidate grids, and on those the search has one Tx FFE point
   to consider, so the floor governs nothing that varies.

Point 3 is the load-bearing one, and it was verified rather than argued: see below.

## 5. Verification

The full 208-case correlation was re-run with **no `--min-radius` flag** — the
rule alone — on the confirmed per-condition configurations, 2026-08-27:

Run over the full case set with modal ERL enabled, then exported for comparison.
That tooling is not in the repository (README section 1).

| | rule alone | previous run, `--min-radius 2` |
|---|---|---|
| COM bit-exact | **208 / 208** | 208 / 208 |
| FOM bit-exact | **208 / 208** | 208 / 208 |
| sampling phase `itick` exact | **208 / 208** | 208 / 208 |
| pass/fail disagreements at 3 dB | **0** | 0 |
| max \|ΔCOM\| | **3.286e-14 dB** | 3.3e-14 dB |
| max \|ΔFOM\| | 3.379e-11 dB | — |

Full-column: 43,509 numeric values compared, 67 outside 1e-6 relative, all of
them `DER_DFE` (41 cases) and `DER_MLSE` (26 cases) — the two quantisation-limited
columns that were already the only inexact ones. Nothing moved.

So the claim in §4 point 3 is measured, not argued: the rule and a flat floor of
2 give the same answer on every case, and the rule is the one that needs no
switch. **The single-candidate cases confirm it directly** — those are the ones
where the two settings differ (rule gives 1, flat gives 2), and their COM, FOM
and `itick` are unchanged.

## 6. What this does not establish

- It does not establish that Hansel's tree contained the mainline rule. It
  establishes that his results behave as though it did. If he reports otherwise,
  the explanation moves to `Overwrite Minimum Radius` in the config, and the
  port's behaviour does not change either way.
- It does not establish that a floor of 2 is *better*. The CAKR study is evidence
  that it is often pure overhead. This is a fidelity decision, not a quality one.
- It says nothing about floors above 2, which no evidence here touches.

## 7. What would overturn it

Any of these, and this document should be revisited rather than patched:

- A reference workbook produced from a configuration that demonstrably sets
  `Overwrite Minimum Radius = 1` and still reproduces at floor 2.
- A multi-candidate case that reproduces at floor 1 but not at floor 2.
- The COM maintainer confirming the branch was run unmodified with no config
  override — which would mean the mechanism is something other than the floor,
  and the 5-of-10 → 10-of-10 attribution is wrong.

## 8. Where it is enforced

| | |
|---|---|
| implementation | `com_functions/fn/OptFom_Adaptive_Local_Search/py_impl.py` |
| the rule itself | `test_min_radius_floor_follows_the_txffe_candidate_count` |
| applied on both version paths | `test_min_radius_rule_is_version_independent` |
| config override still wins | `test_overwrite_min_radius_wins_on_both_paths` |

The second test is the one that matters for drift: it fails if someone
"corrects" the 4p15p0 path back to the branch's forced `1`.
