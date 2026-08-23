# The COM PDF residual — SOLVED

> **Corpus result: COM bit-exact 170 → 198 of 208, and the pass/fail
> disagreement is gone (1 → 0).** On the 200 cases whose sampling phase
> agrees, COM is now exact on 197 and max |ΔCOM| drops from 0.028201 to
> 0.008778 dB (rms 0.004946 → 0.001022, a 4.8× improvement).

**Root cause: an off-by-one in the sampling phase used to build the ADC-clip
PDF.** Found and fixed 2026-08-20. This was the last engine-level disagreement
with MATLAB, and the only one large enough to move a case across the 3 dB
pass/fail threshold.

---

## 1. The symptom

Of the 208 reference cases, 198 had bit-exact FOM **and** bit-exact `itick` —
everything through the equalizer and the sampling-point choice agreed. Of those,
170 also had bit-exact COM and **28 did not**:

| | value |
|---|---|
| cases | 28 |
| max \|ΔCOM\| | 0.028201 dB |
| mean ΔCOM | **+0.008421 dB** |
| sign | **24 Python-high, 4 Python-low** |

A 24 / 4 split with a non-zero mean is a bias, not scatter — the same signature
that located the `process_sxp` leak. The remaining pass/fail disagreement
(`wXtalk_T2_R06`) was in this set.

## 2. The defect

`Apply_EQ` builds the pulse that feeds the ADC-clip PDF:

```matlab
% MATLAB, t_s is 1-BASED
sample_start = mod(fom_result.t_s-1, param.samples_per_ui)+1;
chdata(i).pulse_sampled_w_tx_ffe_ctle = eq_pulse(sample_start:param.samples_per_ui:end);
```

```python
# Python, fom_result.t_s is BEST.cursor_i and already 0-BASED
sample_start = int((fom_result.t_s - 1) % M)      # WRONG
```

The translation converted the 1-based *result* to 0-based but never converted the
1-based *input*. With `t_s_m = t_s_py + 1`:

```
0-based start = mod(t_s_m - 1, M) = mod(t_s_py, M) = t_s % M
```

so the correct expression is `t_s % M`. **Python sampled one sample early.**

The port was already inconsistent with itself: `get_PSDs` performs the same
decimation and gets it right, using `cursor_i % M` for MATLAB's identical
`mod(cursor_i-1,M)+1`. That inconsistency is what confirmed the reading.

## 3. Why it produced a one-directional COM bias

Sampling one sample off the cursor gives slightly different tap values, so the
signal PDF comes out **narrower**:

```
narrower signal PDF  ->  smaller adc_clip  ->  smaller adc_lsb
                     ->  smaller sigma_Q   ->  less quantization noise
                     ->  HIGHER COM
```

which is exactly the observed direction — Python high on 24 of 28.

It also explains why the effect was *pervasive but usually invisible*.
`sigma_before_clip` disagreed by ~1e-4 on **all** 208 cases, including the 170
whose COM was bit-exact. The quantization noise PDF is a uniform box spanning
bins `[adc_ind_left, adc_ind_right]`, chosen by `argmin`, so a small `adc_lsb`
error usually lands in the same bins and changes nothing — and occasionally
crosses a boundary and moves COM.

## 4. How it was localised

The step that cracked it: **convolution adds variances.**

```
sig_after_ctle_pdf = conv( get_pdf_from_sampled_signal(pulse_sampled_w_tx_ffe_ctle),
                           combined_interference_and_noise_pdf )

sigma_before_clip² = σ_signal² + σ_combined²
```

MATLAB reports `sigma_before_clip`, so the two contributions separate. Measuring
Python's `σ_combined` and subtracting gave the implied MATLAB `σ_signal`, which
differed from Python's by 1.1e-4 and 1.1e-3 on the two probe cases. That pointed
at the **signal** PDF, which has exactly one input — the sampled pulse — rather
than at the much more complicated noise PDF chain.

## 5. Verification

`scratchpad/clip_probe.py`, before and after:

| case | quantity | before | after |
|---|---|---|---|
| woXtalk_T1_R02 | `sigma_before_clip` | 1.13e-4 | **2.3e-11** |
| | `peak_clip` | 8.5e-4 | **2.3e-11** |
| woXtalk_T1_R17 | `sigma_before_clip` | 1.09e-3 | **3.0e-15** |
| | `peak_clip` | 1.56e-2 | **0.0 (exact)** |

`woXtalk_T1_R17` was the worst case in the set (ΔCOM +0.0282 dB).

`tests/run_all.ps1` passes with no xcheck flipped.

## 6. A lead that was wrong, recorded so it is not re-followed

An earlier version of this document flagged
`get_pdf_from_sampled_signal`'s use of `np.argsort(np.abs(v))[::-1]` — which is
not a stable descending sort, where MATLAB's `sort(...,'descend')` is — as a
candidate, with magnitude explicitly unverified.

**It is not the cause.** Rebuilding the signal PDF with a stable descending sort
gives a bit-identical sigma (`0.01125904353108015` both ways on
`woXtalk_T1_R02`). Successive convolution is order-independent, so tie order only
perturbs rounding, as suspected.

It remains a real fidelity gap worth closing, along with the same
`argsort(...)[::-1]` pattern at roughly ten other PDF sites — but it is cosmetic,
not numeric, and it is **not** this defect.


## 7. Corpus result

Full 208-case re-run (208/208 ok):

| | before | after |
|---|---|---|
| FOM bit-exact | 198 / 208 | 198 / 208 |
| **COM bit-exact** | 170 / 208 | **198 / 208** |
| `itick` exact | 200 / 208 | 200 / 208 |
| **pass/fail disagreements** | 1 | **0** |
| rms ΔCOM | 0.018660 | 0.017735 |

On the 200 cases whose sampling phase agrees:

| | before | after |
|---|---|---|
| COM bit-exact | 170 / 200 | **197 / 200** |
| max \|ΔCOM\| | 0.028201 | **0.008778** |
| rms ΔCOM | 0.004946 | **0.001022** |

FOM and `itick` are unchanged, as expected — the defect is downstream of the
equalizer and the sampling-point choice.

By pipeline stage: **Noise 75.8% → 90.8%** (+14.9 pts) and **COM 86.7% → 96.3%**
(+9.5 pts). Stages 1–5 are byte-for-byte identical, which is the right signature
for a fix confined to the COM noise path.

### What the remaining 10 are

Nine are the Tx FFE configuration cases (`docs/TXFFE_SWEEP_ROOT_CAUSE.md`) — the
engines were given different search spaces, so their COM is not comparable.

**One was not**, and it has since been closed. `wXtalk_T4_R10` remained at
+0.007623 dB with exact FOM and exact `itick`, from a 4.1e-5 difference in
`sgm_Ani__isi_xt_noise`. That turned out to be a *separate* defect: Python's
`round()` is banker's rounding where MATLAB's rounds half away from zero, and
`nui = round(len(residual_response) / M)` in `get_pdf` landed on a tie, dropping
one ISI sample. Fixed 2026-08-22; see `MATLAB_Correlation_Review.md` §4.3.
With both fixes in, the settings-aligned correlation is bit-exact on FOM, COM and
sampling phase across all 208 cases (max |ΔCOM| 3.3e-14).
