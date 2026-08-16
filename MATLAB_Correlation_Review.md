# COM Python — MATLAB Correlation and Adaptive Local Search

**Status review for Hansel D'Silva — 15 August 2026**

Reference data: `Results_Matlab_COM_v4p15_*_ClipMethodSlow_AdaptiveLS.xlsx` (supplied
12 Aug 2026), 26 IEEE 802.3dj CR/KR channels × 4 package configs × with/without
crosstalk = **208 cases**. MATLAB `code_revision = com_ieee8023_4p15p0.m`.

---

## 1. Headline

All 208 cases run, **0 failures**.

| metric | FOM | COM |
|---|---|---|
| **bit-exact** | **198 / 208 (95.2 %)** | **135 / 208 (64.9 %)** |
| median \|Δ\| | **0.000000 dB** | **0.000000 dB** |
| rms Δ | 0.0082 dB | 0.0186 dB |
| mean Δ | −0.0007 dB | +0.0004 dB |
| \|Δ\| ≤ 0.01 dB | 202 / 208 | 180 / 208 |
| \|Δ\| ≤ 0.05 dB | 206 / 208 | 203 / 208 |
| max \|Δ\| | 0.0841 dB | 0.1756 dB |
| sampling phase (`itick`) exact | **200 / 208** | |

**Without crosstalk the agreement is exact: FOM 104/104, `itick` 104/104.**

| condition | n | FOM exact | COM exact | `itick` exact | max \|ΔCOM\| |
|---|---|---|---|---|---|
| with crosstalk | 104 | 94 | 65 | 96 | 0.1756 |
| without crosstalk | 104 | **104** | 70 | **104** | 0.0347 |

Both engines ran **adaptive local search** — the supplied configs set
`Local Search = 2` and `Non-zero Local Search Method = 1`, and MATLAB used those same
config files. This is an adaptive-vs-adaptive comparison.

### What changed in this cycle

| | before | after |
|---|---|---|
| max \|ΔCOM\| | 6.256 dB (start of correlation) | 0.176 dB |
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

Correlation started at max \|ΔCOM\| **6.256 dB**. Eight fixes took it to 0.176 dB.
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

**A methodological trap worth sharing.** Fix 7 was tried early, made agreement *5–20×
worse*, and was reverted. It was correct all along — it and defect 6 were compensating.
Once the die network was fixed, ΔCOM sat at a suspiciously uniform −0.28 dB, which is
what prompted re-testing it.

> Never judge a fix by end-to-end COM agreement while another defect of similar
> magnitude is still open. Test each fix against the pipeline stage it acts on.

Five of the eight are the same defect class: **MATLAB passes structs by value, Python by
reference.** It is worth grepping the port for any `OP.<field> = ...` inside a function.

---

## 4. What remains

### 4.1 Sampling-phase divergences — 8 cases, all with crosstalk

| case | Δtick | channel |
|---|---|---|
| wXtalk_T1_R07 | +1 | HN_3in_DAC_X_1p0m |
| wXtalk_T1_R08 | +1 | HN_3in_DAC_X_1p5m |
| wXtalk_T1_R15 | +1 | HN_3in_DAC_Z_1p0m |
| wXtalk_T2_R15 | +1 | HN_3in_DAC_Z_1p0m |
| wXtalk_T2_R16 | +2 | HN_3in_DAC_Z_1p5m |
| wXtalk_T3_R16 | +2 | HN_3in_DAC_Z_1p5m |
| wXtalk_T1_R16 | +5 | HN_3in_DAC_Z_1p5m |
| wXtalk_T3_R17 | **−25** | BPK twinax 100 mm |

All are DAC/BPK assemblies; `HN_3in_DAC_Z_1p5m` diverges in 3 of its 4 configs. **The
`process_sxp` fix did not change this set** — the divergences are a separate phenomenon.

Six hypotheses were tested and rejected: a flat/multimodal FOM surface (only 2 of 49
phases lie within 0.1 dB of the peak), reflections (ERL does not separate the groups),
residual size (`li_dj` has the largest \|ΔFOM\| and zero divergences), `auto_port_order`
(all 208 report `[1 3 2 4]`), anchor ambiguity (a tick-matched control has a *wider*
plateau), and adaptive pruning (see §5 — full grid reproduces Python's answer).

**The measurement that constrains it:** Python cannot reach MATLAB's reported FOM at
MATLAB's reported tick **under any equalizer setting** — short by 2.36 dB and 5.43 dB on
the two worst cases — yet the *peak* FOM values agree to within 0.06–0.09 dB. MATLAB's
reported FOM is consistent with Python's chosen phase, not with MATLAB's own reported
phase. The most probable explanation is a **reporting inconsistency on the MATLAB
side**: the `itick` written to the workbook does not correspond to the FOM written
alongside it.

### 4.2 A small residual in the COM PDF path

Isolating the cases where FOM is bit-exact *and* `itick` matches (198 of 208) leaves a
COM-only difference:

| config | n | COM exact | mean ΔCOM | max \|ΔCOM\| |
|---|---|---|---|---|
| Test_1 | 48 | 25 | +0.0033 | 0.0347 |
| Test_2 | 50 | 29 | +0.0022 | 0.0215 |
| Test_3 | 48 | 36 | +0.0010 | 0.0202 |
| Test_4 | 52 | 45 | +0.0008 | 0.0083 |
| **all** | **198** | **135** | **+0.0018** | **0.0347** |

Still graded by package loss, but ~10× smaller than the pre-fix state and opposite in
sign. This is downstream of the equalizer, in the COM PDF / noise-convolution path.

### 4.3 Two pass/fail disagreements at the 3 dB threshold

| case | Python | MATLAB | Δ | `itick` | ΔFOM |
|---|---|---|---|---|---|
| wXtalk_T2_R06 | 3.00696 | 2.99655 | +0.0104 | −11/−11 | **0.00000** |
| wXtalk_T2_R24 | 3.00696 | 2.99958 | +0.0074 | −10/−10 | **0.00000** |

Both have bit-exact FOM and matching `itick`; both sit within 0.011 dB of the threshold.
They are knife-edge cases of §4.2, not search or equalizer differences. Pre-fix they sat
just *below* 3 dB because Python's COM was biased low.

---

## 5. Adaptive local search vs full grid

A full-grid baseline (`Local Search = 0`, everything else identical) was run on nine
channels covering all four families, chosen to include every case where COM Python and
MATLAB pick a different sampling phase — the hardest cases available.

**Adaptive local search is bit-identical to exhaustive full grid on FOM, COM *and*
sampling phase — 9 of 9 — at a median 113× runtime saving.**

| case | channel | FOM adaptive | FOM full grid | ΔFOM | COM adaptive | COM full grid | ΔCOM | itick | penalty |
|---|---|---|---|---|---|---|---|---|---|
| T1_R01 | OSFP 22 dB | 12.0943 | 12.0943 | **0.0000** | 2.8856 | 2.8856 | **0.0000** | −6/−6 | 170× |
| T1_R05 | DAC X 0.5 m | 14.1699 | 14.1699 | **0.0000** | 5.0410 | 5.0410 | **0.0000** | −12/−12 | 26× |
| T1_R07 | DAC X 1.0 m | 12.4255 | 12.4255 | **0.0000** | 3.2692 | 3.2692 | **0.0000** | −7/−7 | 22× |
| T1_R08 | DAC X 1.5 m | 11.8564 | 11.8564 | **0.0000** | 2.7006 | 2.7006 | **0.0000** | −5/−5 | 167× |
| T1_R10 | DAC Y 1.0 m | 13.7627 | 13.7627 | **0.0000** | 4.6105 | 4.6105 | **0.0000** | −16/−16 | 104× |
| T1_R15 | DAC Z 1.0 m | 12.5230 | 12.5230 | **0.0000** | 3.4966 | 3.4966 | **0.0000** | −2/−2 | 113× |
| T1_R16 | DAC Z 1.5 m | 11.6070 | 11.6070 | **0.0000** | 2.3159 | 2.3159 | **0.0000** | 5/5 | 133× |
| T1_R17 | BPK twinax | 15.3059 | 15.3059 | **0.0000** | 6.0491 | 6.0491 | **0.0000** | −21/−21 | 15× |
| T1_R24 | li_dj CR C | 13.9110 | 13.9110 | **0.0000** | 5.1469 | 5.1469 | **0.0000** | −9/−9 | 236× |

Sanity check `FOM(full grid) ≥ FOM(adaptive)` passes 9/9 — full grid searches a superset
of the adaptive candidates, so any violation would indicate broken search wiring.

Runtime penalty ranges 15×–236×; the spread tracks how much of the equalizer grid the
adaptive search is able to prune on that channel.

**This also exonerates adaptive pruning for the §4.1 divergences.** On R16 the exhaustive
full grid independently arrives at Python's `itick = 5` where MATLAB reported 0. The
divergence is between the two *engines*, not the two *search strategies*.

Full grid costs **7–9 hours per case** versus minutes for adaptive, so a complete
208-case full-grid sweep would take 5–8 days and was deliberately scoped to a
representative subset.

---

## 6. Requests

1. **Which Tx FFE tap set produced the with-crosstalk reference?** The two reference
   workbooks disagree: `wXtalk` reports `TXLE_taps = [0 0 1 0]` (four taps, cursor
   third) and `Pre2Pmax = 0`, while `woXtalk` reports `TXLE_taps = 1` and an empty
   `Pre2Pmax` — **with the same `config_file` recorded in both**.

   The shipped config declares `c(-4)…c(-1) = 0` and `c(1) = 0`. Tracing MATLAB's own
   `OptFom_Build_TXFFE` (L2789–2816) with that input, every all-zero fixed tap is
   dropped and `txffe` collapses to the scalar `[1.0]` — which is what the `woXtalk`
   reference and COM Python both report. COM Python is a faithful port of that logic
   (verified line by line, pinned by unit tests): as soon as any tap is non-trivial the
   `auto_count_trigger` latch retains the whole vector *including its zeros*.

   Numerically inert — no Tx equalization is applied in any of the 208 cases — but worth
   confirming nothing else differed between the two runs.

2. **`cursor_i` (or absolute `t_s`) reported alongside `itick`.** This settles §4.1
   immediately: if `cursor_i − SBR_peak ≠ itick` on those eight cases, the reporting
   inconsistency is confirmed. The workbook currently exposes only `itick`, so the frame
   origin is unobservable.

*(The earlier request for `H` and `Rnn` is withdrawn — the residual it was meant to
diagnose is resolved.)*

---

## Appendix A — deliverables

```
com_python_results/
    Results_COM_Python_adaptiveLS_wXtalk.xlsx     104 cases
    Results_COM_Python_adaptiveLS_woXtalk.xlsx    104 cases
```

Same four tabs, same 26 rows, same 262-column header in the same order as the MATLAB
workbooks, so the two sets diff column-by-column with no remapping. 241 of 262 columns
populated; a NOTES sheet in each workbook lists what is blank and why.

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
python tools/matlab_compare.py --run --jobs 5             # ~3.4 h, checkpointed
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
