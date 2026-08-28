# SiCoPR — MATLAB Correlation and Adaptive Local Search

**Correlation status review — 21 August 2026**

*Written for, and reviewed with, Hansel D'Silva, who produced the MATLAB
reference results this port is measured against. Published here because the
evidence for the correctness claim belongs with the code that makes it.*

Reference data: `Results_Matlab_COM_v4p15_*_ClipMethodSlow_AdaptiveLS.xlsx` (supplied
12 Aug 2026), 26 IEEE 802.3dj CR/KR channels × 4 package configs × with/without
crosstalk = **208 cases**. MATLAB `code_revision = com_ieee8023_4p15p0.m`.

---

## 1. Headline

All 208 cases run, **0 failures**, on the configurations the MATLAB author
confirmed: without-crosstalk on the base workbooks, with-crosstalk on the
`*_sweep_TxFFE` workbooks that sweep the Tx FFE.

| metric | result |
|---|---|
| FOM bit-exact | **208 / 208** |
| COM bit-exact | **208 / 208** |
| sampling phase (`itick`) exact | **208 / 208** |
| max \|ΔCOM\| | **3.3e-14 dB** |
| rms ΔCOM | **1.1e-14 dB** |
| pass/fail disagreements at 3 dB | **0** |

3.3e-14 dB is double-precision arithmetic noise, not agreement to a tolerance.
The reference COM values span −4.82 to +6.68 dB, so the agreement holds across
passing and failing channels alike and across all four package configurations.

Both engines ran **adaptive local search** — the reference workbooks are named
`..._AdaptiveLS.xlsx`, and the configs set `Local Search = 2` with
`Non-zero Local Search Method = 1`. This is an adaptive-vs-adaptive comparison.

> **One set of results.** An earlier revision of this document reported two
> readings, because only one of the two configuration workbooks had been shared
> and the with-crosstalk settings had to be reconstructed. That is over: the
> MATLAB author confirmed the with-crosstalk sweep on 2026-08-24 and supplied the
> workbooks, so there is one configuration and one comparison.
>
> The one setting that had to be deduced — the adaptive search's minimum radius
> — is now settled, and needs no switch. The supplied workbooks set no radius
> keyword and the branch source forces 1, yet the reference behaves as the
> 4p16p0 mainline rule (`1` for a single Tx FFE candidate, `2` otherwise). The
> port applies that rule on both version paths; the evidence, what it does not
> establish, and what would overturn it are in
> [`docs/MIN_RADIUS_ASSUMPTION.md`](docs/MIN_RADIUS_ASSUMPTION.md).

### What changed in this cycle

| | before | after |
|---|---|---|
| max \|ΔCOM\| | 6.256 dB (start of correlation) | **3.3e-14 dB** |
| FOM bit-exact | 8 / 208 (3.8 %) | **208 / 208** |
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

**Nothing.** FOM, COM and sampling phase are bit-exact on all 208 cases.

| | |
|---|---|
| FOM bit-exact | **208 / 208** |
| COM bit-exact | **208 / 208** |
| sampling phase (`itick`) exact | **208 / 208** |
| pass/fail disagreements at 3 dB | **0** |
| max \|ΔCOM\| | **3.29e-14 dB** |
| max \|ΔFOM\| | **3.38e-11 dB** |

Full-column: 43,509 numeric values compared, 67 outside 1e-6 relative, all of them
`DER_DFE` (41 cases) and `DER_MLSE` (26 cases). Those two are CDF bin lookups
landing on an exact tie — `−A_s` falls exactly on a bin because `A_s/BinSize` is
exactly 1000, so which side it lands on is decided by the 11th digit of `A_s`.
Quantisation-limited, not wrong.

### 4.1 How the divergences were closed

Eight sampling-phase divergences and ten COM misses were open for most of this
cycle. All are closed, and it is worth recording what closed them, because in
every case the answer was a **settings** difference rather than an engine
disagreement.

| what was diverging | cause | where |
|---|---|---|
| 8 `itick`, and 9 of the 10 COM misses | the supplied configs pin Tx FFE `c(-1)` to a single zero, so Python searched one candidate where MATLAB searched 1584 | [`docs/TXFFE_SWEEP_ROOT_CAUSE.md`](docs/TXFFE_SWEEP_ROOT_CAUSE.md) |
| 5 of those 10, again | the adaptive-search radius floor: the branch source forces 1, the reference behaves as 2 on a multi-candidate grid | [`docs/MIN_RADIUS_ASSUMPTION.md`](docs/MIN_RADIUS_ASSUMPTION.md) |
| a +0.0018 dB mean COM bias, and 2 knife-edge pass/fail flips | an off-by-one in the sampling phase used to build the ADC-clip PDF | [`docs/COM_PDF_RESIDUAL.md`](docs/COM_PDF_RESIDUAL.md) |
| the last COM miss, 4 case-instances | banker's rounding in `nui = round(len/M)` | §4.3 below |

The eight `itick` cases were all DAC or BPK assemblies carrying crosstalk — the
profile of a channel where Tx FFE pre-emphasis wins, which is the fingerprint of
a missing search dimension rather than an independent risk profile. Two of the
eight are among the ten cases where MATLAB selected a non-unity `c(-1)`; the rest
moved because the sampling phase is chosen jointly with the equalizer.


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

There were two, then one, and now none. Both former cases
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

**Scope.** This is measured on a **single-point Tx FFE
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

Those measurements were taken when the whole corpus swept one TXFFE candidate per case.
The with-crosstalk half now sweeps 1584, so `min_radius` **is** exercised by the corpus,
and the port applies the mainline rule rather than the branch's forced 1 (§6.3,
[`docs/MIN_RADIUS_ASSUMPTION.md`](docs/MIN_RADIUS_ASSUMPTION.md)). `Clip Method` still
cannot be exercised here — all four configs name it explicitly.

**Worth noting rather than asking:** on the CAKR channel the larger floor evaluates 4.3×
the candidates for a bit-identical answer, so on that channel class it is pure overhead.
It is not overhead everywhere — it is what reproduces 10 of the 10 crosstalk cases that
turn on it.

Detail: `docs/MATLAB_4p16p0_CHANGES.md` (what changed) and `docs/MATLAB_4p16p0_IMPACT.md`
(measured effect).

---

## 6. Open questions — all closed

Nothing is outstanding. This section used to carry requests; each is recorded
here with how it was settled, because the answers are what the correlation now
rests on.

### 6.1 The Tx FFE grid — answered by the maintainer

The four supplied configs set `c(-1)`, `c(-2)` and `c(1)` to `0` in the value
column, yielding a single unity Tx FFE, while the reference workbooks contain
non-unity winners on 10 cases. Confirmed on 2026-08-24: the with-crosstalk run
**did** sweep the Tx FFE, and the workbooks capturing it were supplied
(`*_sweep_TxFFE.xlsx`, 1584 candidates). The with-crosstalk half of the
correlation now runs on those; the without-crosstalk half runs on the base
workbooks, which is the configuration its own reference used.

Detail: [`docs/TXFFE_SWEEP_ROOT_CAUSE.md`](docs/TXFFE_SWEEP_ROOT_CAUSE.md).

*An earlier version of this item asked whether the two workbooks record Tx FFE at
different granularity, and proposed that the single-column and four-column forms
were "the same answer written two ways". That reading was wrong: the non-unity
values are real, and they were the whole explanation for the disagreement.*

### 6.2 Adaptive local search — answered by the workbook names

The reference workbooks are named `..._AdaptiveLS.xlsx`, so the run used adaptive
local search, and the port's `Non-zero Local Search Method = 1` matches it.
`com_ieee8023_4p15p0.m` **as distributed** contains only `OptFom_Local_Search` —
`OptFom_Adaptive_Local_Search` first appears in 4p16p0 — so the run used a
4p15p0 build with the search backported, which is what one would expect from its
author.

*An earlier draft listed the search method as a second mismatch and asserted that
the port "has been running the adaptive search against a legacy reference". That
was wrong.*

**One finding here is worth keeping, because it is about the method rather than
about the port.** MATLAB's own reported answer is not its grid's optimum on 2 of
those 10 cases. On `wXtalk_T3_R17` seven candidates beat it at its own CTLE,
including unity itself (13.8662 vs the reported 13.8016). So adaptive local
search can stop short on a real Tx FFE grid. The "adaptive == full grid" result
in §5 is measured on a single-point grid, where the search has nothing to prune
in that dimension; on a real grid it finds the optimum on 15 of 16, worst loss
0.0093 dB ([`docs/TXFFE_SWEEP_ROOT_CAUSE.md`](docs/TXFFE_SWEEP_ROOT_CAUSE.md)
§9).

### 6.3 The adaptive-search radius floor — deduced, and closed

The last setting that had to be inferred. The supplied workbooks set no radius
keyword and the branch source forces `1`, yet the reference behaves as `2` on a
multi-candidate grid — a floor of 1 reproduces 5 of the 10 cases, a floor of 2
reproduces 10 of 10.

Rather than ask, the port adopts the rule that already exists in published
MATLAB and produces exactly that behaviour — the 4p16p0 mainline
`1 if num_txffe_runs == 1 else 2` (L2782-2786) — applied on both version paths.
The reproduction command needs no radius switch as a result.

This is the **second** instance of the same drift in the same run: the supplied
configuration snapshots are a later state than the one that produced the
results, which is ordinary for experimental work that is not under revision
control, and is already established for the Tx FFE grid above. The assumption,
its evidence, what it does *not* establish, and what would overturn it:
[`docs/MIN_RADIUS_ASSUMPTION.md`](docs/MIN_RADIUS_ASSUMPTION.md).

### 6.4 Withdrawn

`cursor_i` / absolute `t_s` was requested to test a frame-origin hypothesis. That
hypothesis has been displaced, and the stage-6 noise question the request was
also meant to settle turned out to be a Python defect
([`docs/STAGE6_NOISE_AGREEMENT.md`](docs/STAGE6_NOISE_AGREEMENT.md) §8, fixed).
No further reference data is needed.

---

## Appendix A — deliverables

```
sicopr_results/
    confirmed/
        Results_SiCoPR_confirmed_wXtalk.xlsx      104 cases
        Results_SiCoPR_confirmed_woXtalk.xlsx     104 cases
```

**One set, not two.** Each crosstalk condition is run on the configuration its own
MATLAB reference was produced with: the without-crosstalk cases on the base workbooks,
the with-crosstalk cases on the `*_sweep_TxFFE` workbooks — the pairing the COM
maintainer confirmed on 2026-08-24. The radius floor needs no switch either (§6.3).

Same four tabs, same 26 rows, same 262-column header in the same order as the MATLAB
workbooks, so the two sets diff column-by-column with no remapping. A NOTES sheet in
each workbook lists the columns blank on every row, derived from the exported data
rather than from a fixed list.

```
report_docs/SiCoPR_MATLAB_Review.pptx    review deck (built, not tracked)
MATLAB_Correlation_Review.md    this document
tools/matlab_compare.py         --validate / --run / --report, checkpointed, --jobs N
tools/export_results.py         result workbooks in the reference format
R/correlation_report.R          the figures in the deck
```

## Appendix B — reproducing

```bash
python tools/matlab_compare.py --validate                 # resolve all 208 cases
python tools/matlab_compare.py --run --modal-erl --jobs 5  # ~3.4 h, checkpointed
python tools/matlab_compare.py --run --modal-erl --txffe-sweep \
       --only-cond wXtalk --jobs 5                       # the aligned wXtalk half
python tools/export_results.py                            # result workbooks
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

- `sicopr.py` is **generated** by `assemble_sicopr.py` from 159 per-function
  `com_functions/fn/<name>/py_impl.py` files. Never edit `sicopr.py` directly, and note
  that `assemble_sicopr.py` does not carry per-function imports across.
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
