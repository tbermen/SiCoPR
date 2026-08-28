# Why stage 6 (Noise) sits at 55%, and what would move it

> **SOLVED 2026-08-19.** Root cause: `BEST.PSD_results` aliased a struct MATLAB
> copies by value, so the reported noise terms came from the **last sampling
> phase swept (+24)** rather than the winning one. Since
> `ts_sample_adj_range = [-24, 24]` and `samples_per_ui = 32`, and **24 ≡ −8
> (mod 32)**, the stale arrays were correct exactly when the winning tick was
> −8 — which is precisely the signature this document describes below. Fixed in
> `git log --grep="BEST.PSD_results aliased"`; `sgm_TX` went from 2/208 exact to 42/48 on the validation subset.
> See §8. The itick divergences are a separate matter and are NOT fixed by this.
>
> **Final state 2026-08-22.** Two further fixes moved this stage: the ADC-clip
> sampling phase (`docs/COM_PDF_RESIDUAL.md`) and a banker's-rounding tie in
> `nui = round(len/M)` (`docs/FIX_SUMMARY.md` #16). The Noise stage now sits at
> **95.6%**, and **every** remaining inexact column
> is `DER_MLSE` or `DER_DFE` — 275 of 275. Those two are `CDF_ev` bin lookups
> landing on an exact tie and are quantisation-limited rather than wrong, so this
> stage will not reach 100% on the current metric. All 29 other noise columns are
> exact on all 208 cases.

Investigation prompted by the review deck's stage-agreement chart. The bar is the
share of reported output columns agreeing with MATLAB to better than 1e-9
relative, over all 208 cases. Stage 6 reads 55% where every other stage is 77%
or better.

Reproduce: the scripts used are in the session scratchpad; the inputs are
`matlab_compare_results/cases/*.result.json` (Python) joined to
`tests/2_Results_COM_Matlab/*.xlsx` (MATLAB) via `report_data/compare.csv`.

---

## 1. The 55% is two populations, not one

33 columns match the stage-6 pattern; 30 have comparable values (3 are empty on
both sides and contribute nothing). Splitting them by exactness rate:

| group | columns | exact | rate |
|---|---|---|---|
| **A — healthy** | 18 | 3345 / 3536 | **94.6%** |
| **B — never exact** | 13 | 76 / 2700 | **2.8%** |
| overall | 31 | 3421 / 6236 | 54.9% |

Group A fails only on the ~10 cases already known to diverge (the sampling-phase
set). Group B fails on essentially every case:

```
sigma_before_clip 0/208   P2ptopsigma_clip 0/208   DER_DFE 2/208
DER_MLSE 2/204            sgm_Ani 3/208            DER_thresh 3/208
peak_clip 8/208           sgm_Q 8/208              sgm_G 10/208
sgm_gaussian_noise 10/208 sgm_TX 10/208            sgm_rjit 10/208
sgm_noise__gaussian_noise_p_DD 10/208
```

**If Group B reached Group A's rate the stage would read ~95%.** So this is one
defect to find, not thirteen.

## 2. What the residual is not

- **Not a systematic bias.** Signs are balanced (e.g. `sgm_G` 91 high / 117 low)
  and median ratios are 0.9999–1.0001. There is no scale factor to correct.
- **Not the RxFFE solve.** MATLAB's 87 taps are in the workbooks: Python matches
  them bit-exactly on **198 of 208** cases (median worst-tap error 4.9e-12).
  Restricting to just those 198, `sgm_N` is 198/198 exact while `sgm_G` is still
  10/198 at a median of 3.5e-5. Identical taps, same `H_rxffe_2` scaling, yet one
  term agrees and the other does not.
- **Not duplicated columns inflating the count.** Only `sgm_N`/`sigma_N` are true
  aliases (208/208 identical), and both sit in Group A.
- **Not an index-base slip in the obvious places.** `htn` alignment
  (`mod(cursor_i-1,M)+1` → `cursor_i % M`) and the `h_J` cursor offsets match the
  documented 1-based/0-based conversions.

## 3. What it is: an algorithmic difference, not accumulated rounding

Agreement as a function of tolerance:

| within | share |
|---|---|
| 1e-9 | 54.9% |
| 1e-8 | 54.9% |
| 1e-7 | 54.9% |
| 1e-6 | 55.1% |
| 1e-5 | 58.5% |
| 1e-4 | **67.4%** |
| 1e-3 | 78.7% |
| 1e-2 | 91.7% |

**The curve is flat from 1e-9 to 1e-6 and then climbs.** Nothing sits in the
"almost exact" band. Accumulated floating-point error would produce a gradual
ramp; a bimodal split — exact to 1e-12, or wrong by more than 1e-6 — is the
signature of a genuine difference in how a quantity is computed.

That is encouraging: it means a findable defect exists, not an irreducible
numerical floor.

## 4. Where it must live

`sgm_N` (`S_rn_rms`) agrees; `sgm_TX` (`S_tn_rms`), `sgm_rjit` (`S_rj_rms`) and
`sgm_G` (`S_G_rms = S_tn + S_rj_jn + S_rn + S_in`) do not. All four are the same
integral, `sqrt(sum(S_x) * delta_f)`, and all four are scaled by the same
`H_rxffe_2`. The difference is in the PSD arrays themselves:

- `S_rn` — built from the **analytic** `S_RN(fvec, G_DC, G_DC2, param)` — agrees.
- `S_tn` — FFT of the CTLE pulse response decimated at the symbol rate — does not.
- `S_rj_jn` — FFT of `h_J`, built from early/late cursor samples — does not.

So the suspect is the construction of `htn`/`h_J` or the FFT normalisation, not
the integral and not the equalizer. Code reading has not localised it further:
the index conversions are right, and the error is too large for rounding.

## 5. Localised: the sampling anchor used to decimate the pulse

Continuing with MATLAB's own reported values as the diagnostic:

**(a) Both sides are internally consistent.** The PSDs add, so
`sigma_G^2 = sigma_TX^2 + sigma_rj^2 + sigma_N^2 + sigma_in^2`. That identity holds
to 1e-16 on *both* sides (MATLAB median residual −1.2e-16, Python 0.0). Neither
engine has an inconsistent term.

**(b) `sigma_G` is a consequence, not a cause.** Decomposing its error by each
term's share of `G^2` (TX 23.5%, rjit 3.8%, N 72.7%) predicts the observed
`dG/G` with correlation **0.9998**. Fix TX and rjit and `sigma_G` follows.

**(c) `sigma_N` is exact where it matters.** Its Python/MATLAB ratio is
1.000000000 at both the 5th and 95th percentile. `S_rn` is built from an
**analytic** PSD. `S_tn` and `S_rj_jn` are FFTs of a **pulse decimated at the
symbol rate**. Only the pulse-derived ones disagree.

**(d) The discriminator is `itick`.** Ranking every MATLAB column by how well it
separates the 10 exact cases from the other 198, `itick` wins with **1.0%
overlap** — all ten have `itick = −8`. The exact set is *identical* for
`sgm_TX`, `sgm_rjit` and `sgm_G`, confirming one shared cause.

**(e) The error is a function of distance from that tick:**

| itick | n | median rel err |
|---|---|---|
| −8 | 12 | **1.4e-12** |
| −7 / −9 | 16 | 9e-5 / 1.5e-4 |
| −10 / −6 | 8 | 3.6e-4 / 2.8e-4 |
| −15 | 18 | 1.3e-3 |
| −22 | 13 | 9.4e-4 |

Essentially exact at one tick, growing with distance from it. A *constant* phase
offset would give a roughly constant error; this pattern says the two engines
agree at one sampling anchor and drift apart as the sample point moves away.

**(f) Which lands on `cursor_i`.** `get_PSDs` aligns the transmit-noise pulse
with `phase_0 = cursor_i % M` and builds `h_J` from cursor-adjacent samples;
`S_rn` uses neither. Python and MATLAB call `get_PSDs` with the same argument
(`THIS.cursor_i`) at the same points, and the 1-based/0-based conversions are
correct — verified by probe: for `woXtalk_T1_R14` Python's `phase_0` is 7 and
MATLAB's start index is also 7. Ruled out along the way: the pad-vs-truncate
branch (both pad, `num_ui_RXFF_noise = 4096`) and a phase-zero special case (the
exact case has phase 7, not 0).

So the remaining candidate is that **Python's `cursor_i` and MATLAB's refer to
slightly different frame origins**, coinciding at `itick = −8`. `h_J` is a finite
difference of adjacent cursor samples, which is why `sgm_rjit` scatters ±2%
where `sgm_TX` scatters ±0.2% — a difference operator amplifies exactly this
kind of offset.

### This is the same missing datum as the itick divergences

§4.1 of the correlation review reaches the same place from the other direction:
Python cannot match MATLAB's FOM at MATLAB's reported tick under any equalizer
setting, while the peak values agree — consistent with an anchor-origin offset.
Two independent investigations now converge on one unknown.

**The request to Hansel for `cursor_i` (or absolute `t_s`) alongside `itick` now
resolves two open items, not one.** It is a single extra column.

### Testable prediction, for when that data arrives

If Python's and MATLAB's `cursor_i` differ by `d` samples on a case, then
`phase_0` differs by `d mod 32`, and `sgm_TX` should agree exactly wherever
`d ≡ 0 (mod 32)`. On the ten `itick = −8` cases `d` should be 0 or a multiple of
32; everywhere else it should not be. That is a one-line check once the column
exists, and it either confirms the mechanism or kills it outright.

## 6. Note for the reader

None of these columns is COM. `COM_dB`, `FOM`, `VEO_mV`, `VEC_dB`, `itick` and
`ERL` are unaffected by this residual — stage 7 (COM) is at 77% and the headline
correlation is FOM bit-exact on 198/208. Group B is internal noise
instrumentation, and its worst median error is 6e-3 relative.

## 7. Ruled out: the crosstalk contribution itself

Group B's disagreements are concentrated in with-crosstalk cases, which invites
the reading that `S_xn` — the crosstalk noise term — is where the error enters.
It is not. The crosstalk arithmetic is bit-exact (`ICN_mV`, `MDFEXT_ICN_92_47_mV`,
`MDNEXT_ICN_92_46_mV` all 208/208; `SNR_MDFEXT` 104/104), the post-equalization
crosstalk terms disagree only on cases whose sampling point already differs, and
the one crosstalk-only *decision* — `get_PSDs` choosing each aggressor's phase by
`argmax` over M candidate norms — is nowhere near a tie on the divergent cases
(median top-two margin 1.6e-5, statistically indistinguishable from controls).

Crosstalk is the **enabling condition**: `S_xn` enters `S_n`, which changes the
RxFFE solve and the shape of FOM versus sampling phase, making the tick contested
enough for a small underlying difference to change the winner. Without crosstalk,
FOM and `itick` are exact on 104/104. Full working in
`docs/ITICK_SUBSET_FOR_REVIEW.md` §3b.


## 8. Root cause (2026-08-19)

### What the reported noise terms actually are

In the MMSE path — which is every one of the 208 cases — MATLAB does **not**
compute `sigma_TX` from `A_s`. ML 1594-1596:

```matlab
NS.sigma_TX   = PSD_results.S_tn_rms;
NS.sigma_rjit = PSD_results.S_rj_rms;
```

and the COM stage never rebuilds those PSDs at the final sample point. It copies
them out of the optimiser's result and only rescales by `|H_rxffe|²` (ML 547-553):

```matlab
PSD_results.S_tn    = fom_result.PSD_results.S_tn;
PSD_results.S_jn    = fom_result.PSD_results.S_jn;
PSD_results.S_rj_jn = fom_result.PSD_results.S_rj_jn;
PSD_results.S_xn    = fom_result.PSD_results.S_xn;
```

### The defect

`OptFom_Update_Best_Setttings` did `BEST.PSD_results = THIS.PSD_results`. MATLAB
copies structs **by value**; Python bound a **reference**. `get_PSDs` mutates its
`result` argument in place, and `optimize_fom` reuses one `PSD_results` object
for every txffe/itick candidate inside a CTLE block. `BEST` therefore ended up
pointing at the state left by the **last tick swept**, not the winning one.

### Why the error sat at `itick = −8`

`S_tn` and `S_rj_jn` are built by decimating at phase `cursor_i % 32`. The sweep
ends at `+24`, and **24 ≡ −8 (mod 32)** — so the stale arrays carry the *correct*
phase exactly when the winner is at −8. That is the whole signature §5 described:
error ~1e-12 at −8, growing with distance from it.

It also explains the anomaly that made this findable: `available_signal_after_eq_mV`
(= `1000·A_s`, ML 4013) agrees to **2e-12** while `sgm_TX` did not. A_s is a
direct sample and does not pass through `PSD_results`; `S_tn_rms` is a
decimate-and-truncate product that does.

### Direct evidence

Recording `S_tn_rms` at the moment of each BEST update and again when
`optimize_fom` returns:

| case | itick | at the winning update | in BEST at return | change |
|---|---|---|---|---|
| woXtalk_T1_R01 | −6 | 3.99799578313e-4 | 3.99606119783e-4 | **4.84e-4** |
| woXtalk_T2_R14 | −8 | 3.11617661970e-4 | 3.11617661970e-4 | 0 |

`BEST.PSD_results is THIS.PSD_results` → `True` before the fix, `False` after,
and both changes become 0.

The 4.84e-4 movement on `woXtalk_T1_R01` matches its 4.05e-4 disagreement with
MATLAB — the stale value *is* the disagreement.

### Effect (48-case validation subset)

| column | before (208-wide) | after |
|---|---|---|
| `sgm_TX` | 2/208 exact, median rel 8.0e-4 | **42/48 exact**, median 5.2e-12 |
| `sgm_rjit` | 2/208, median 6.1e-3 | **42/48**, median 5.2e-12 |
| `sgm_N` | 28/208 | **42/48** |
| `sgm_G` | 2/208, median 3.5e-5 | **42/48** |
| `sgm_isi` | 45/208 | **42/48** |
| `sgm_xt` | 128/208 | **18/24** (wXtalk only) |

On tick-matching cases, `max|ΔCOM|` 0.034719 → 0.028201 and rms 0.017846 →
0.014221; `wXtalk_T1_R03` and `wXtalk_T2_R17` became bit-exact in COM.

### What it does not fix

FOM and `itick` are unchanged. The search was never affected — `get_PSDs`
recomputes `S_tn`/`S_jn`/`S_xn`/`S_qn` fresh at every tick, so the FOM surface the
optimiser sees was always correct. Only what `BEST` carried forward was stale.
The eight `itick` divergences stand on the separate anchor/basin analysis in
`docs/ITICK_SUBSET_FOR_REVIEW.md`.

### Why the lint missed it

`tests/test_reference_leaks.py` flags parameters that are written but never
returned. Here the object is legitimately returned; the bug is a **missing copy
on store**. Detect this class by recording a field at the moment of a BEST update
and comparing with what BEST holds at return — any nonzero change means an
in-place-mutated object was stored by reference.
