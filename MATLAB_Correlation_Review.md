# COM Python — MATLAB Correlation and Adaptive Local Search

**Status review for Hansel D'Silva — 21 August 2026**

Reference data: `Results_Matlab_COM_v4p15_*_ClipMethodSlow_AdaptiveLS.xlsx` (supplied
12 Aug 2026), 26 IEEE 802.3dj CR/KR channels × 4 package configs × with/without
crosstalk = **208 cases**. MATLAB `code_revision = com_ieee8023_4p15p0.m`.

---

## 1. Headline

All 208 cases run, **0 failures**.

The two reference workbooks were produced with **different Tx FFE settings**
([`docs/TXFFE_SWEEP_ROOT_CAUSE.md`](docs/TXFFE_SWEEP_ROOT_CAUSE.md) §11),
and only the without-crosstalk config was supplied. Both readings are therefore
reported throughout, and every figure is generated in both variants:

| | configs **as supplied** | settings **aligned** |
|---|---|---|
| FOM bit-exact | 198 / 208 | **208 / 208** |
| COM bit-exact | 199 / 208 | **208 / 208** |
| sampling phase (`itick`) exact | 200 / 208 | **208 / 208** |
| max \|ΔCOM\| | 0.1852 dB | **3.3e-14 dB** |
| rms ΔCOM | 0.0177 dB | **4.5e-15 dB** |
| pass/fail disagreements at 3 dB | 0 | **0** |

**Settings aligned** pairs each crosstalk condition with the settings its own
reference used. Its with-crosstalk half is a **reconstruction**, not a config we were
sent — supported by reproducing MATLAB's tap vector, sampling phase and FOM on all
ten previously divergent cases, and by the 2×2 control in that document's §11, but
to be confirmed
before the numbers are quoted as like-for-like. **There is no remaining COM miss:**
every case agrees to within 3.3e-14 dB, which is double-precision noise rather than
agreement to a tolerance.

The rest of this section details the **as-supplied** run.

| metric | FOM | COM |
|---|---|---|
| **bit-exact** | **198 / 208 (95.2 %)** | **198 / 208 (95.2 %)** |
| median \|Δ\| | **0.000000 dB** | **0.000000 dB** |
| rms Δ | 0.0082 dB | 0.0177 dB |
| mean Δ | −0.0007 dB | −0.0013 dB |
| \|Δ\| ≤ 0.01 dB | 202 / 208 | 202 / 208 |
| \|Δ\| ≤ 0.05 dB | 206 / 208 | 203 / 208 |
| max \|Δ\| | 0.0841 dB | 0.1852 dB |
| sampling phase (`itick`) exact | **200 / 208** | |

**Without crosstalk the agreement is exact: FOM 104/104, COM 104/104, `itick`
104/104, max \|ΔCOM\| = 0.0000.**

| condition | n | FOM exact | COM exact | `itick` exact | max \|ΔCOM\| |
|---|---|---|---|---|---|
| with crosstalk | 104 | 94 | 94 | 96 | 0.1852 |
| without crosstalk | 104 | **104** | **104** | **104** | 0.0000 |

Both engines ran **adaptive local search** — the supplied configs set
`Local Search = 2` and `Non-zero Local Search Method = 1`, and MATLAB used those same
config files. This is an adaptive-vs-adaptive comparison.

### What changed in this cycle

| | before | after |
|---|---|---|
| max \|ΔCOM\| | 6.256 dB (start of correlation) | 0.185 dB |
| FOM bit-exact | 8 / 208 (3.8 %) | **198 / 208 (95.2 %)** |
| systematic FOM bias | Python lower in 199/208 (95.7 %) | **eliminated** |

---

## 2. Root cause of the systematic FOM bias

Every case showed Python's FOM slightly **low** — 95.7 % of them, mean −0.0057 dB. A
one-directional bias of that consistency is a defect, not scatter.

**`process_sxp` leaked a TDR-only setting into the entire run.**

```matlab
% com_ieee8023_4p15p0.m, L9311
OP.impulse_response_truncation_threshold = 1e-5;   % Only for TDR not returned out of "process_sxp" function
```

MATLAB passes structs **by value**, so that assignment dies with the function — its own
comment says exactly that. COM Python passes by reference, so the threshold stayed at
**1e-5 instead of reverting to the config default 1e-3** for every downstream stage,
including the whole equalizer search.

A 100× tighter truncation retains far more impulse-response tail → longer pulse
response → larger residual ISI → larger `sigma_e` → **FOM biased low**. Exactly the
observed direction and magnitude.

Fixed by rebinding `OP` to a shallow copy immediately before the TDR overrides, which
reproduces MATLAB's by-value scope. The same leak also carried `interp_sparam_mag` and
`interp_sparam_phase` out of the function; both are inert for this data but are
contained by the same change.

### Why it was hard to see

`sigma_e²` is 19.7 % signal and 80.3 % noise, which pointed at the noise PSDs for a long
time. The signal term is a **catastrophic cancellation**:

```
 w'H'Hw   1.7017381439
 +1       1.0000000000
 b'b      0.7002900000
 -2w'h0  -2.0000000000
 -2w'Hb'b -1.4005800000
 SUM      0.0014481439     <-- 1175x amplification
```

`sig_terms` is the residual ISI energy, so a **6e-6** relative error in the pulse moves
`sigma_e` as much as a **0.36 %** error in `Rnn`. That amplification also explains the
package dependence with no package-dependent defect: low-loss packages equalize better,
shrinking `sig_terms` and raising the amplification.

Two intermediate conclusions had to be discarded along the way, both recorded here
because they are easy traps:

- The RxFFE tap difference (~1e-3) is **irrelevant to FOM** — both tap sets give
  identical FOM, because FOM is stationary in the taps at an optimum. Any test built on
  the tap gradient measures the wrong thing.
- A DFE-tap match (`b = Hb·w`, which agreed to 1e-10) pins only ~27 samples of `h`
  around the cursor, **not** all 4182 rows of `H`. 33 % of the residual ISI comes from
  beyond ±87 UI, which that test never touches.

---

## 3. Engine defects found and fixed

Correlation started at max \|ΔCOM\| **6.256 dB**. Ten fixes took it to 0.185 dB
(0.0088 dB on the 200 cases whose sampling phase agrees), with **zero** pass/fail
disagreements.
Each was found by ranking all comparable output columns by relative error and letting
the data localise the fault; reading code to guess causes failed repeatedly.

| # | defect | effect |
|---|---|---|
| 1 | **`z_p` transpose for RX/NEXT/FEXT.** MATLAB transposes all four `z_p` keywords, Python only TX, so the RX package was built from a matrix *row* — 111 mm instead of 13.8 mm. | ≈ 15 dB spurious loss |
| 2 | **IL fit at effective rank 2 of 4.** MATLAB takes the raw normal-equations inverse of a knowingly near-singular matrix; `np.linalg.lstsq` silently truncates the rank. | fitted IL, FOM_ILD now exact |
| 3 | **RxFFE floating-tap array sized by tap *count* (23), not *span* (87).** | every floating tap past 23 discarded |
| 4 | **`get_TDR` `tfstart` index base** — derived from the full time vector, applied to the windowed arrays. | Z11est/Z22est 1.4e-2 → 4e-15 |
| 5 | **`get_TDR` `fctrx` initialisation** — `np.zeros` discarded all reflection energy past the DFE gate. | ERL → ~1e-15 |
| 6 | **Package die network truncated to 1 of 3 LC sections** (three inlined copies). | ~15 ps die delay, 53-sample pulse shift |
| 7 | **Cursor index base in `optimize_fom`** (audit finding B16-D20). | see note |
| 8 | **`process_sxp` OP leak** (§2). | the systematic FOM bias |
| 9 | **`BEST.PSD_results` stored a *reference* to a struct `get_PSDs` mutates in place**, so the reported noise came from the last sampling phase swept, not the winning one. MATLAB copies that struct by value. | noise stage 55 % → 76 %, COM bit-exact 135 → 170 |
| 10 | **ADC-clip sampling phase off by one** — `(t_s-1) % M` on an already 0-based `t_s`, where MATLAB's `mod(t_s-1,M)+1` takes a 1-based one. Sampled the clip pulse one sample early. | noise 76 % → 91 %, COM 170 → 198, pass/fail flips 1 → 0 |

**A methodological trap worth sharing.** Fix 7 was tried early, made agreement *5–20×
worse*, and was reverted. It was correct all along — it and defect 6 were compensating.
Once the die network was fixed, ΔCOM sat at a suspiciously uniform −0.28 dB, which is
what prompted re-testing it.

> Never judge a fix by end-to-end COM agreement while another defect of similar
> magnitude is still open. Test each fix against the pipeline stage it acts on.

Six of the ten are the same defect class: **MATLAB passes structs by value, Python by
reference.** It is worth grepping the port for any `OP.<field> = ...` inside a function.
Defect 9 is the subtler form: the object *is* returned, and what was missing is the
**copy on store** — which an AST leak lint cannot see.

---

## 4. What remains

Everything in this section is reported under **both** readings, because the two
differ in what is left over, not just by how much.

| | configs as supplied | settings aligned |
|---|---|---|
| sampling-phase (`itick`) divergences | **8** | **0** |
| COM not bit-exact | **9** | **0** |
| pass/fail disagreements at 3 dB | **0** | **0** |

Under settings aligned, **nothing remains**: FOM, COM and sampling phase are
bit-exact on all 208 cases, with max |ΔCOM| = 3.3e-14 dB. Everything else in this section describes the as-supplied run, and every
item in it is explained by the Tx FFE settings difference
([`docs/TXFFE_SWEEP_ROOT_CAUSE.md`](docs/TXFFE_SWEEP_ROOT_CAUSE.md) §11) rather than by
an engine disagreement.

### 4.1 Sampling-phase divergences — 8 cases as supplied, 0 aligned

**Root-caused.** These were an open question through most of this cycle and are
not one any more: the supplied configs pin Tx FFE `c(-1)` to a single zero, so
Python searched one candidate where MATLAB searched ~1584. Give both engines the
same search space and all eight agree.

| case | Δtick | ΔCOM (as supplied) | channel |
|---|---|---|---|
| wXtalk_T1_R07 | +1 | 0.000000 | HN_3in_DAC_X_1p0m |
| wXtalk_T2_R15 | +1 | −0.007440 | HN_3in_DAC_Z_1p0m |
| wXtalk_T1_R08 | +1 | −0.041814 | HN_3in_DAC_X_1p5m |
| wXtalk_T2_R16 | +2 | −0.054311 | HN_3in_DAC_Z_1p5m |
| wXtalk_T1_R15 | +1 | −0.061061 | HN_3in_DAC_Z_1p0m |
| wXtalk_T3_R16 | +2 | −0.070361 | HN_3in_DAC_Z_1p5m |
| wXtalk_T3_R17 | **−25** | +0.132145 | BPK twinax 100 mm |
| wXtalk_T1_R16 | +5 | −0.185248 | HN_3in_DAC_Z_1p5m |

All are DAC/BPK assemblies and all carry crosstalk — the profile of a channel
where Tx FFE pre-emphasis wins (high FEXT, high residual ISI, a contested
equalizer optimum). That clustering is the fingerprint of the missing search
dimension, not an independent risk profile. Two of these eight are among the ten
cases where MATLAB selected a non-unity `c(-1)`; the rest move because the
sampling phase is chosen jointly with the equalizer.

**Under settings aligned, `itick` is exact on 208 of 208.**

### 4.2 COM differences — 10 cases as supplied, 1 aligned

As supplied, the ten cases whose COM is not bit-exact are **exactly the eight
above plus two more**, and every one of them carries a non-zero ΔFOM. That is
the signature of a different equalizer answer, not of a difference in the COM
computation: with the search spaces matched, nine of the ten go to zero.

The residual that *was* in the COM PDF path — a mean +0.0018 dB bias over the
cases with exact FOM and matching `itick`, which used to leave 135 of 198 exact
and produced two knife-edge pass/fail disagreements — has been fixed. It was an
off-by-one in the sampling phase used to build the ADC-clip PDF; see
[`docs/COM_PDF_RESIDUAL.md`](docs/COM_PDF_RESIDUAL.md). On the cases whose FOM
and sampling phase agree, COM is now bit-exact on 197 of 200 and max \|ΔCOM\|
dropped from 0.028201 to 0.008778 dB.

### 4.3 The last COM miss, and how it was closed

`wXtalk_T4_R10` sat at +0.007623 dB with bit-exact FOM and matching `itick` — the
only case not explained by the Tx FFE settings. It is now **bit-exact on all 217
output columns**, along with three sibling case-instances that carried the same
signature.

**Root cause: Python's `round()` is banker's rounding; MATLAB's rounds half away
from zero.** `get_pdf` computes

```python
nui = round(len(residual_response) / M)
```

which sets how many rows the `vs` sampling matrix has, and therefore how many ISI
samples enter the residual-ISI PDF. Instrumenting every `round()` call in a run
found **exactly one of 98,145** sitting on a tie — `2360.500000`, where MATLAB
returns 2361 and Python returns 2360. One row lost, one ISI sample dropped, and
`sgm_isi` comes out low; the deficit then dilutes through `sgm_isi_xt`,
`sgm_Ani__isi_xt_noise` and `sigma_before_clip`.

It reached only 4 of 208 case-instances (`T1_R23` and `T4_R10`, both crosstalk
conditions) because the tie needs the response length to be an odd multiple of
`M/2`, which depends on the channel **and** the package — each affected channel
appears in 8 cases and diverges in only 2.

> **This site was already in the audit ledger and had been dismissed.**
> `docs/AUDIT_FINDINGS.md` lists `nui=round(len/M)` among the bare-`round` sites
> and argues they "differ from MATLAB only for exact half-integer inputs
> (measure-zero for continuous data)". That is correct for continuous inputs and
> wrong here: `len(residual_response)` and `M` are both **integers**, so the
> quotient is a rational that lands on `.5` exactly. It is the second audit
> verdict this correlation has overturned.

### 4.4 Pass/fail disagreements: none

There were two, then one, and now none under either reading. Both former cases
(`wXtalk_T2_R06`, `wXtalk_T2_R24`) sat within 0.011 dB of the 3 dB threshold with
bit-exact FOM and matching `itick`; they were knife-edge instances of the ADC-clip
residual, and both resolved when it was fixed.

---

## 5. Adaptive local search vs full grid

A full-grid baseline (`Local Search = 0`, everything else identical) was run on **all
208 cases**, re-measured 2026-08-21 on the current engine after every fix in this cycle.

**Adaptive local search is bit-identical to exhaustive full grid on FOM, COM *and*
sampling phase — 208 of 208 — at a median 13x runtime saving.**

| | adaptive vs full grid |
|---|---|
| FOM identical | **208 / 208** |
| COM identical | **208 / 208** |
| sampling phase (`itick`) identical | **208 / 208** |
| max \|ΔFOM\|, max \|ΔCOM\| | **0.000e+00** |
| identical, with crosstalk | 104 / 104 |
| identical, without crosstalk | 104 / 104 |
| total runtime | 3.0 h vs 32.4 h |
| median per case | 0.67 min vs 9.47 min |
| runtime penalty | median **13x** (range 5x–20x) |

Sanity check `FOM(full grid) >= FOM(adaptive)` passes **208/208** — full grid searches a
superset of the adaptive candidates, so any violation would indicate broken search
wiring.

> **Two corrections to the previous version of this section.** It reported 9 of 9 cases
> and a median **113x** runtime saving. The agreement result is now much stronger (the
> whole corpus, not nine hand-picked channels). The runtime saving is **smaller**: the
> performance work in this cycle sped full grid up more than it sped adaptive up, so the
> honest figure is 13x, not 113x. Quote 13x.

**Scope.** This is measured with the configs as supplied, i.e. a **single-point Tx FFE
grid**, where the adaptive search has nothing to prune in that dimension. On a real Tx
FFE grid the adaptive search finds the full-grid optimum on 15 of 16 cases, with a worst
observed loss of 0.0093 dB — see `docs/TXFFE_SWEEP_ROOT_CAUSE.md` §9.

**This also exonerates adaptive pruning for the §4.1 divergences.** On R16 the exhaustive
full grid independently arrives at Python's `itick = 5` where MATLAB reported 0. The
divergence is between the two *engines*, not the two *search strategies*.

Full grid costs **7–9 hours per case** versus minutes for adaptive, so a complete
208-case full-grid sweep would take 5–8 days and was deliberately scoped to a
representative subset.

---

## 5b. Runtime

MATLAB reports `rtmin` per case, so a direct comparison is possible. The absolute
numbers are confounded — the MATLAB times are from Hansel's machine — so each engine is
also compared **against itself**, which removes the hardware dependence.

| | MATLAB | COM Python (before) | COM Python (after) |
|---|---|---|---|
| total, 208 cases | 14.40 h | 16.10 h | **3.57 h** |
| vs MATLAB | — | 1.12× slower | **4.03× faster** |
| cases faster than MATLAB | — | 84 / 208 | **104 / 208** |
| slowest single case | 13.3 min | 37.5 min | **2.5 min** |
| wall clock, `--jobs 5` | — | 3.4 h | **0.6 h** |

**≈ 4.5× on CPU time, ≈ 5× on wall clock.** Repeat runs of the same build landed at
3.15 h and 3.57 h, so treat these as ±13 % — the machine has meaningful run-to-run
variance and the speed-up should be read as "roughly 4.5–5×", not a precise figure.

**Every correlation statistic was unchanged by the speed work** — that was verified
at the time against the then-current figures (FOM 198/208, COM 135/208, `itick`
200/208, max \|ΔCOM\| 0.175602 dB, 2 pass/fail disagreements). Those figures have
since improved through later fixes; see §1 for the current numbers. `COM_dB`, `VEO_mV` and `VEC_dB` are bit-identical case by case, verified
by re-running the full corpus after each change rather than trusting a sample.

Four changes, each measured before being kept:

1. **Size-gated FFT convolution.** PDF convolution sizes are extremely skewed: 2.2 % of
   calls carry 85 % of the arithmetic while ~77 000 calls have a kernel of ≤ 16 bins and
   carry 3.4 %. Direct convolution wins for tiny kernels and loses badly for long ones
   (2.7× slower at 600, 19× at 9000, >1000× at 20 000+), so the kernel dispatches on
   size rather than switching wholesale. `conv_fct` exists in **18 copies**; all 17 that
   convolve PDFs share the same kernel.
2. **Hoisted Gram matrix.** `MMSE_FOM` recomputed `H.T @ H` (~2.2 MFLOP, H is ~4182×87)
   on each of its ~130 000 calls per case, though H is fixed and only the column
   selection changes. Computing it once and gathering is **107×** faster on that
   operation.
3. **Memoised ADC-clip PDF.** It depends only on the equalizer setting and the sampling
   *phase*, and the 49-tick sweep visits only 32 distinct phases — 34.8 % of builds were
   exact repeats. Keyed on input bytes, so hits are bit-identical by construction.
4. **Assembly micro-optimisations.** `np.block` → preallocated array (3.2×), `np.ix_` →
   `.take().take()` (2.2×), and `np.eye`/`np.zeros` cached. All bit-identical.

Only (1) perturbs the arithmetic, at ~1e-15 relative — it moves FOM in the 14th
significant digit and leaves every reported COM value untouched.

### Parallelism: memory binds before CPU

Each case holds ~379 MB, and this machine has 16 logical CPUs but ~3 GB free.

| | CPU | free RAM | throughput | wall clock |
|---|---|---|---|---|
| `--jobs 5` | 38 % | ~1.2 GB | 4.00 cases/min | **0.6 h** |
| `--jobs 7` | 61 % | 0.4 GB | 4.67 cases/min | 0.7 h |

`--jobs 7` raises throughput but inflates per-case time by 1.20 × through contention and
gives **no wall-clock gain**, while leaving only 0.4 GB free. **`--jobs 5` is the right
setting on this machine** — CPU headroom is not the limit, memory is.

One further saving is available but not taken: `SAVE_FIGURES = 1` in the Test_3 and
Test_4 configs writes 10 PNGs per case. That is a config choice MATLAB honours too, so
it is left alone, but it is pure overhead in batch correlation runs.

---

## 5c. MATLAB 4p16p0

4p16p0 was published on 2026-08-18 and is supported behind a version switch; **4p15p0
remains the default**, since the reference workbooks this correlation rests on are 4p15p0
output.

The delta is small: **146 of 152 function bodies unchanged, 6 changed, 3 added, 0 removed**.
The three additions are `OptFom_Adaptive_Local_Search`, `compute_hard_cap` and
`append_csv_row` — **adaptive local search has been adopted into the released mainline**
rather than remaining a branch.

The same 208 cases were run in both modes. **210 of 213 output columns are identical on
every case**, and no COM, FOM, VEO, VEC, `itick` or ERL value moves.

| change | measured effect |
|---|---|
| pulse/step now scaled by channel amplitude `A` | `peak_uneq_pulse_mV`, `steady_state_voltage_mV` × A on all 208. 4p15p0 scaled the impulse response but not the pulse built from it; the new values are the corrected ones. COM untouched. |
| `Clip Method` default `Fast` → `Slow` | **COM +0.007 dB, FOM +0.22 dB**, but only for configs that omit the keyword — all 208 reference configs set it |
| `min_radius` 1 → 2 in adaptive search | bit-identical answer, **4.3× the candidate evaluations, 2.5× the runtime** |

The last two could not be exercised by the corpus (every case sweeps one TXFFE candidate,
and every config names `Clip Method`), so they were measured separately on a config with a
1584-point Tx FFE grid.

**Two points for you on this.** First, the mainline sets `min_radius = 2` where your branch
forced 1; on that channel the mainline default evaluates 4.3× the candidates for a
bit-identical answer, so we would be interested in what motivated 2 — it may help on channel
classes not represented here. Second, a 4p16p0 run of these same 208 cases is what would let
us move the default; with adaptive search now mainline it is a more useful comparison than
when it was a branch.

Detail: `docs/MATLAB_4p16p0_CHANGES.md` (what changed) and `docs/MATLAB_4p16p0_IMPACT.md`
(measured effect).

---

## 6. Requests

1. **Which Tx FFE tap ranges were active in the run that produced the workbooks?**

   The four configs supplied set `c(-1)`, `c(-2)` and `c(1)` to `0` in the value column,
   which yields a single unity Tx FFE. The workbooks contain non-unity winners on 10
   cases (`[0, −0.02, 0.98, 0]`, `[0, −0.04, 0.96, 0]`, `[0, −0.06, 0.94, 0]`,
   `[0, −0.1, 0.9, 0]`), so that run used a wider grid — see §4.1.

   Part of it can be inferred: MATLAB's reported `TXLE_taps` has **four** elements, and
   `OptFom_Build_TXFFE` trims leading single-valued zero taps until the first non-trivial
   one, so the `c(-2)` slot survived — meaning it held at least two values. Specifically:
   was `c(-2)` set to `[ 0.14:.02:0]` (which evaluates *empty*) or to something else, and
   were `c(-3)` / `c(-4)` in play?

   With the exact set, the corpus can be re-run on a matched search space.

   *An earlier version of this request asked whether the two workbooks record Tx FFE at
   different granularity, and proposed that the single-column and four-column forms were
   "the same answer written two ways". That reading was wrong: the non-unity values are
   real, and they are the whole explanation for the remaining disagreement.*

2. **Confirmation only: adaptive local search, with a backported implementation?**

   The reference workbooks are named `..._AdaptiveLS.xlsx`, so the run used adaptive
   local search and COM Python's `Non-zero Local Search Method = 1` matches it. Worth
   one line of confirmation because `com_ieee8023_4p15p0.m` **as distributed** contains
   only `OptFom_Local_Search` — `OptFom_Adaptive_Local_Search` first appears in 4p16p0.
   The natural reading is that the run used a 4p15p0 build with the adaptive search
   backported, which is what one would expect from its author.

   *An earlier draft of this section listed the search method as a second mismatch and
   asserted that COM Python "has been running the adaptive search against a legacy
   reference". That was wrong — the workbook filenames say `AdaptiveLS`. The search
   method is not a discrepancy; the only open unknown is the Tx FFE grid in item 1.*

   **What does still matter for the proposal:** MATLAB's own reported answer is not its
   grid's optimum on 2 of the 10 cases. On `wXtalk_T3_R17` seven candidates beat it at
   its own CTLE, including unity itself (13.8662 vs the reported 13.8016). So adaptive
   local search can stop short on a real Tx FFE grid. The "adaptive == full grid"
   result in §5 (now 208/208) is measured with the configs as supplied, i.e. on a
   single-point Tx FFE grid where the adaptive search has nothing to prune in that
   dimension. On a real grid it finds the optimum on 15 of 16, worst loss 0.0093 dB
   (`docs/TXFFE_SWEEP_ROOT_CAUSE.md` §9).

   *`cursor_i` / absolute `t_s` is no longer requested — the frame-origin hypothesis it
   was meant to test has been displaced, and the stage-6 noise question it was also meant
   to settle turned out to be a Python defect (`docs/STAGE6_NOISE_AGREEMENT.md` §8,
   fixed).*

---

## Appendix A — deliverables

```
com_python_results/
    as-supplied/
        Results_COM_Python_as-supplied_wXtalk.xlsx        104 cases
        Results_COM_Python_as-supplied_woXtalk.xlsx       104 cases
    settings-aligned/
        Results_COM_Python_settings-aligned_wXtalk.xlsx   104 cases
        Results_COM_Python_settings-aligned_woXtalk.xlsx  104 cases
```

Two sets, because the two MATLAB reference workbooks were evidently produced with
different Tx FFE settings ([`docs/TXFFE_SWEEP_ROOT_CAUSE.md`](docs/TXFFE_SWEEP_ROOT_CAUSE.md)
§11): **as-supplied** runs every case on the four config
spreadsheets as received; **settings-aligned** runs each condition on the settings its
own reference used. Only the wXtalk half differs between them, and that half is a
reconstruction — each workbook's NOTES sheet says so, and records which result
directories it was built from.

Same four tabs, same 26 rows, same 262-column header in the same order as the MATLAB
workbooks, so the two sets diff column-by-column with no remapping. A NOTES sheet in
each workbook lists the columns blank on every row, derived from the exported data
rather than from a fixed list.

```
COM_Python_MATLAB_Review.pptx   17-slide review deck
MATLAB_Correlation_Review.md    this document
tools/matlab_compare.py         --validate / --run / --report, checkpointed, --jobs N
tools/export_results.py         result workbooks in the reference format
R/correlation_report.R          the figures in the deck
```

## Appendix B — reproducing

```bash
python tools/matlab_compare.py --validate                 # resolve all 208 cases
python tools/matlab_compare.py --run --modal-erl --jobs 5  # ~3.4 h, checkpointed
python tools/matlab_compare.py --run --modal-erl --txffe-sweep --min-radius 2 \n       --only-cond wXtalk --jobs 5                       # the aligned wXtalk half
python tools/export_results.py --set as-supplied          # result workbooks
python tools/export_results.py --set settings-aligned
python tools/export_compare_csv.py                        # tidy CSVs
Rscript R/correlation_report.R                            # figures
python tools/build_review_pptx.py                         # deck

# adaptive-vs-full-grid study (slow: 7-9 h per case)
python tools/matlab_compare.py --run --variant fullgrid --only-cond wXtalk \
       --only-test 1 --rows 16,17,7,15,8,10,24,5,1 --jobs 5
```

`--jobs N` runs N cases in parallel; CPU sits at ~38 % with 5 on a 16-thread machine.

## Appendix C — comparison-harness defects

Three bugs in the *comparison tooling* produced alarming false discrepancies before
being caught. Recorded because anyone building an equivalent harness will hit them.

1. **Per-workbook headers.** Both workbooks are 262 columns but have *different layouts
   from column 40 on*. `itick` is column 231 vs 227, `COM_dB` 260 vs 256. Reusing one
   header row for both shifted every no-crosstalk column by 4.
2. **Duplicate header names.** The no-crosstalk workbook repeats
   `DER_MLSE`/`COM_dB`/`DER_thresh`/`rtmin` at columns 259–262, where the **trailing
   copy holds the with-crosstalk reference values**. `dict(zip(headers, row))` keeps the
   *last* occurrence, so every no-crosstalk case was compared against the with-crosstalk
   answer — fake \|ΔCOM\| up to 2.58 dB and 9 fake pass/fail flips.
3. **Runtime column.** Same duplication; the fallback picked column 262, inflating
   MATLAB's reported total from 14.4 h to 24.2 h.

In (1) and (2) the giveaway was a physically impossible value — a non-integer tick
index, then two different runs reporting COM identical to six decimals. Sanity-check
plausibility before hypothesising an engine defect.

## Appendix D — notes on the port

- `com.py` is **generated** by `assemble_com.py` from 159 per-function
  `com_functions/fn/<name>/py_impl.py` files. Never edit `com.py` directly, and note
  that `assemble_com.py` does not carry per-function imports across.
- Several functions are **also inlined into their callers** — `MMSE`/`MMSE_FOM`,
  `interp_Sparam` (inside `s21_to_impulse_DC`), `make_full_pkg` (inside `read_s4p_files`
  and `s21_pkg`). Patching the standalone module does not affect the inlined copies.
- Some callees exist as **fallback stubs** (`S_RN`, `S_IN`, `H_interp`, `get_TDR`). All
  call sites wire the real functions, but an unwired one would silently substitute a
  flat PSD or a dummy TDR.
- `flim` truncates the data at 67 GHz while `fout` extends to 1.7 THz, so ≈ 96 % of the
  analysis grid is extrapolated.
- The full-grid configs in `tests/1_IEEE_802p3dj_COM_Spreadsheets_fullgrid/` differ from
  the originals only in `Local Search` (2 → 0), plus a 1-ULP round-trip on `f_v`
  (relative 1.2e-16) from rewriting cached formula values with openpyxl.
