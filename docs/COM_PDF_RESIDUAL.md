# The last discrepancy: 28 cases, COM-only, systematically positive

**Status 2026-08-20: localised, not root-caused.** This is the only remaining
engine-level disagreement with MATLAB once the Tx FFE configuration difference
(`docs/TXFFE_SWEEP_ROOT_CAUSE.md`) is set aside.

---

## 1. What it is

Of the 208 reference cases, **198 have bit-exact FOM *and* bit-exact `itick`** —
everything through the equalizer and the sampling-point choice agrees. Of those
198, **170 also have bit-exact COM. 28 do not.**

| | value |
|---|---|
| cases | 28 |
| max \|ΔCOM\| | 0.028201 dB |
| rms ΔCOM | 0.013013 dB |
| mean ΔCOM | **+0.008421 dB** |
| median \|ΔCOM\| | 0.010440 dB |
| sign | **24 Python-high, 4 Python-low** |

A 24 / 4 split with a non-zero mean is a bias, not scatter. That is the same
signature that located the `process_sxp` leak, and it is why this is worth
chasing rather than filing as numerical noise.

The single remaining pass/fail disagreement (`wXtalk_T2_R06`, Python 3.0070 vs
MATLAB 2.9965) is in this set.

## 2. Where it enters

Ranking every reported column by median relative error, on the 28 versus the 170
that agree:

| column | the 28 | the 170 OK |
|---|---|---|
| `sgm_Q` / `peak_clip` | 4.05e-3 | 1.82e-3 |
| `P2ptopsigma_clip` | 4.03e-3 | 1.87e-3 |
| `sgm_Ani__isi_xt_noise` | 7.71e-4 | **3.62e-12** |
| `sigma_before_clip` | 1.30e-4 | 1.13e-4 |
| `COM_dB` | 4.09e-3 | 3.01e-15 |
| `sgm_N` | 1.75e-12 | 2.79e-12 |

`sgm_N` is exact, so the aggregate noise PSD agrees. The earliest quantity that
disagrees is the **ADC clip**.

### The mechanism is consistent end to end

`peak_clip` is **smaller in Python on 23 of the 28**. Smaller `adc_clip` →
smaller `adc_lsb` → smaller `sigma_Q` → less quantization noise → **higher COM**.
Python's COM is higher on 24 of 28. The direction matches.

## 3. What has been ruled out

**Not a faithful-port problem.** `adjust_Rx_noise_for_quantization` (ML 4863-4879)
and `CDF_inv_ev` (ML 1147) were compared line by line against the port and are
equivalent, including the by-value PDF copy and the `find(CDF >= val, 1, 'first')`
semantics.

**Not knife-edge bin selection.** `CDF_inv_ev` returns `PDF.x[first bin where
CDF ≥ val]`, so a last-bit CDF difference would shift the answer by exactly one
bin. Measured, in units of `OP.BinSize = 1e-5`:

```
(peak_clip_py - peak_clip_ml) / BinSize  =  -94, -78, -49.0, -48.0, -32.1, ...
integer within 1e-6:  2 of 28
within one bin:       0 of 28
```

The differences are 6 to 94 bins and mostly non-integer, so the PDF itself
differs — this is not a tie-break at the quantile.

## 4. Where the search now stands

`peak_clip` and `sigma_before_clip` are both derived from **one** object:

```matlab
sig_after_ctle_pdf = get_pdf_from_sampled_signal(chdata(1).pulse_sampled_w_tx_ffe_ctle,
                                                 param.levels, param.delta_y);
sig_after_ctle_pdf = conv_fct(sig_after_ctle_pdf, combined_interference_and_noise_pdf);
```

`sigma_before_clip` is that PDF's second moment, and it disagrees by ~1e-4
relative on **both** groups — including the 170 cases whose COM is bit-exact. So
the PDF is pervasively, slightly different everywhere; it only moves COM far
enough to matter on the 28.

That narrows the remaining question to which of the two inputs differs:

- `chdata(1).pulse_sampled_w_tx_ffe_ctle` — the sampled pulse, or
- `combined_interference_and_noise_pdf` — the combined interference + noise PDF.

`sgm_Ani__isi_xt_noise` is the combined PDF's sigma, and it is exact to 3.6e-12
on the 170 while differing at 7.7e-4 on the 28 — which points at the combined PDF
being right in general and wrong on this subset, rather than at a constant error
in the sampled pulse.

## 5. One concrete fidelity gap found on the way

`get_pdf_from_sampled_signal` orders its input by descending magnitude before the
successive-convolution build:

```matlab
[input_vector,index] = sort(abs(input_vector),'descend');   % MATLAB: STABLE
```
```python
sort_idx = np.argsort(np.abs(input_vector))[::-1]           # py_impl.py:114
```

`np.argsort(x)[::-1]` is **not** a stable descending sort: reversing an ascending
sort inverts the relative order of ties, where MATLAB preserves it. Ties in
`|value|` are common in a sampled pulse tail (± pairs, repeated small taps).

**Magnitude unverified.** Successive convolution is mathematically
order-independent, so this should only perturb rounding — of order 1e-15, not the
1e-4 seen in `sigma_before_clip`. It is recorded as a real fidelity gap to fix,
*not* as the explanation. Fixing it and re-measuring is the cheap next step; if
`sigma_before_clip` does not move, the cause is elsewhere.

The same `argsort(...)[::-1]` pattern appears at roughly ten other PDF sites
(`get_pdf`, `Create_Noise_PDF`, `d_cpdf`, `Output_Arg_Fill`, `RILN_TD`, …). They
should be checked together, against the MATLAB each came from, rather than
edited blind.

## 6. Impact, stated plainly

This is a **0.008 dB mean, 0.028 dB worst-case** effect on COM, on 28 of 208
cases, in a quantity whose specification threshold is 3 dB. It moves one case
across that threshold. It does not affect FOM, `itick`, VEO, VEC or ERL, and it
is not implicated in the Tx FFE finding.
