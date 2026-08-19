# Why stage 6 (Noise) sits at 55%, and what would move it

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

## 5. What would actually move the number

**This investigation did not raise it.** Two honest routes:

1. **Get MATLAB's PSD intermediates for two or three cases** — `S_tn`,
   `S_rj_jn`, `S_G`, and ideally the `htn` and `h_J` vectors that feed them. The
   workbooks expose only the final RMS scalars, so the discrepancy is currently
   unobservable at the point it is created. This is a much smaller request than
   the per-function dump considered earlier: three arrays on two cases would
   localise it immediately, in the same way that ranking output columns by
   relative error localised the original eight defects.

2. **Report the stage at a stated engineering tolerance as well as at 1e-9.**
   Stage 6 is 67% within 1e-4 and 92% within 1e-2. Showing both is more
   informative than one number — but note that this changes what is being
   measured, so it should be presented as an added row, never as a replacement
   that makes the chart look better. The 1e-9 figure is the honest one for
   "bit-exact agreement" and should stay.

Route 1 is the real fix. Route 2 only changes how the same fact is displayed.

## 6. Note for the reader

None of these columns is COM. `COM_dB`, `FOM`, `VEO_mV`, `VEC_dB`, `itick` and
`ERL` are unaffected by this residual — stage 7 (COM) is at 77% and the headline
correlation is FOM bit-exact on 198/208. Group B is internal noise
instrumentation, and its worst median error is 6e-3 relative.
