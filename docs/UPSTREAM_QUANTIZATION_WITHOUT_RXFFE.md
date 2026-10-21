# ADC quantization with the Rx FFE off stops the Reference Code

**Found 2026-10-20**, driving SiCoPR 1.1.2 from SiCoPR-Assist. **Status:** a
defect in the COM Reference Code, reproduced by executing it under GNU Octave
11.3 on 4p15p0, 4p16p0 and 4p17p0. SiCoPR reproduces it, as
[`VERIFICATION.md`](VERIFICATION.md) requires, and since this note stops with an
error that names the cause instead of a bare `AttributeError`. Offered to the
COM ad hoc with a proposed fix.

## Summary

A configuration that turns the receiver FFE off (`ffe_pre_tap_len = 0` and
`ffe_post_tap_len = 0`) while keeping ADC quantization on (`N_qb ~= 0`) runs the
whole equalizer search and then stops while building the noise PDF:

```
error: structure has no member 'RxFFE'
error: called from
    com_ieee8023_4p17p0_octave_compat>adjust_Rx_noise_for_quantization at line 5590 column 1
    com_ieee8023_4p17p0_octave_compat>Create_Noise_PDF at line 1834 column 6
    com_ieee8023_4p17p0_octave_compat at line 632 column 10
```

`adjust_Rx_noise_for_quantization` reads `fom_result.RxFFE` unconditionally,
and `optimize_fom` only creates that field when the Rx FFE is on. No shipped
workbook reaches this: every 802.3dj workbook with `N_qb = 6` also has an Rx
FFE. It is reached by a user who switches the Rx FFE off to study a
DFE-only or CTLE-only receiver and leaves `N_qb` as the workbook had it.

## Reproducing it

The configuration is the shipped 802.3dj KR workbook
`config_com_dj_200G_CAKR_178_PKGA_06_2_2025__Case1.xlsx` with these keywords
changed (everything else as shipped, including `N_qb = 6`):

| keyword | shipped | here |
|---|---|---|
| `ffe_pre_tap_len` | 6 | 0 |
| `ffe_post_tap_len` | 8 | 0 |
| `N_bg` | 2 | 0 |
| `N_b` | 1 | 0 |
| `b_max(1)` | 0.85 | 0 |
| `MLSE` | 1 | 0 |
| `g_DC` | `[-20:1:0]` | `[-8:1:0]` |
| `g_DC_HP` | `[-6:1:0]` | `[-2:1:0]` |
| `N_tail_start` | 9 | 1 |

Only the first two matter: with `N_b` left at 14 the outcome is the same. The
channel was a synthetic THRU (19.9 dB insertion loss at Nyquist), no crosstalk.
With the configuration saved as a two-column `.csv` (`keyword,value`):

```
# COM Octave (the release file made Octave-capable; octave/README.md)
octave-cli --eval "addpath('octave'); r = com_ieee8023_4p17p0_octave_compat('cfg.csv', 0, 0, 'thru.s4p');"

# SiCoPR
python -m sicopr cfg.csv thru.s4p --matlab-version 4p17p0
```

Both engines complete the same equalizer search (FOM 9.9 dB, CTLE DC gain
-8 dB in both) and then stop in the same function. Run on all three releases:

| release | COM Octave | SiCoPR before this note | SiCoPR now |
|---|---|---|---|
| 4p15p0 | `structure has no member 'RxFFE'`, compat L5498 | `AttributeError: ... no attribute 'RxFFE'` | `ValueError: N_qb = 6 (ADC quantization) needs an Rx FFE: ...` |
| 4p16p0 | the same, compat L5516 | the same | the same |
| 4p17p0 | the same, compat L5590 | the same | the same |

The function on its own, under Octave on all three releases with a `fom_result`
that has no `RxFFE` field, gives the same error; that is the case pinned in
`com_functions/fn/adjust_Rx_noise_for_quantization/test_verify.py`.

## Root cause in the Reference Code

Line numbers are the original IEEE files in `matlab/`.

| | 4p15p0 | 4p15p0 + adaptive search | 4p16p0 | 4p17p0 |
|---|---|---|---|---|
| `OP.RxFFE` true only if `ffe_pre_tap_len` or `ffe_post_tap_len` is nonzero (`read_ParamConfigFile`) | 9897 | 10211 | 10244 | 10771 |
| `result.RxFFE = BEST.RxFFE` only `if OP.RxFFE` (`OptFom_Create_Output`) | 3455 | 3709 | 3730 | 3793 |
| `if param.N_qb ~=0` calls `adjust_Rx_noise_for_quantization` (`Create_Noise_PDF`) | 1679 | 1679 | 1699 | 1699 |
| `h_rxffe = fom_result.RxFFE(find(fom_result.RxFFE ~= 0));` | 4893 | 5147 | 5165 | 5228 |

The quantization step convolves the uniform ADC-noise PDF, scaled by each Rx
FFE tap other than the cursor, into the noise PDF (Equation 93A-37 territory;
the function was added in r480 beta1). It assumes an Rx FFE exists. The guard
on the call is `N_qb ~= 0` alone; it does not test `OP.RxFFE`, and the function
does not either. When the Rx FFE is off, `optimize_fom` returns a `result`
without the field, and the first read of it is an error. 4p14p0 has the same
line (L4888), so the defect predates every release SiCoPR emulates.

## A related observation: a zero pre-cursor tap shifts the skipped tap

The same line drops zero taps before the loop indexes the cursor:

```matlab
h_rxffe = fom_result.RxFFE(find(fom_result.RxFFE ~= 0));
cursor_tap = param.ffe_pre_tap_len+1;
for irxffe = 1:length(h_rxffe)
    if irxffe ~= cursor_tap
```

`cursor_tap` is the cursor's position in the full tap vector, but the loop
counts positions in the compressed one. If any pre-cursor tap is exactly zero
the cursor moves left in `h_rxffe`, so the loop skips a post-cursor tap and
convolves in the cursor instead. Executed under COM Octave 4p17p0 with
`ffe_pre_tap_len = 2`, `N_qb = 4` and the same other inputs:

| `RxFFE` | taps convolved in | quantization-noise sigma / (LSB/sqrt 12) |
|---|---|---|
| `[1e-9 0.2 1.0 -0.3]` | 1e-9, 0.2, -0.3 (cursor skipped, as intended) | 0.350 |
| `[0 0.2 1.0 -0.3]` | 0.2, 1.0 (cursor counted, -0.3 skipped) | 1.020 |

Expected for the intended taps: sqrt(0.2^2 + 0.3^2) = 0.361 and, for the
second row, sqrt(0.2^2 + 1^2) = 1.020. Exact zeros are common with floating
taps and tap limits, so this can move COM on runs that complete normally.
SiCoPR reproduces it (the reference decides), and records it here.

## Recommended fix to the Reference Code

In `adjust_Rx_noise_for_quantization` (4p17p0 L5228-5235):

```diff
-h_rxffe = fom_result.RxFFE(find(fom_result.RxFFE ~= 0));
+if OP.RxFFE
+    h_rxffe = fom_result.RxFFE;
+else
+    h_rxffe = 1; % no Rx FFE: the cursor alone
+end
 cursor_tap = param.ffe_pre_tap_len+1;
 for irxffe = 1:length(h_rxffe)
-    if irxffe ~= cursor_tap
+    if irxffe ~= cursor_tap && h_rxffe(irxffe) ~= 0
         quantization_noise_in_pdf_scale = scalePDF(quantization_noise_in_pdf, abs(h_rxffe(irxffe)));
         quantization_noise_pdf = conv_fct(quantization_noise_pdf, quantization_noise_in_pdf_scale);
     end
 end
```

The first hunk removes the stop; the second keeps zero taps out of the
convolution, as the `find` intended, without moving the cursor index. The two
are independent: the first changes no completed run, the second moves COM on
runs with an exactly zero pre-cursor tap and is the ad hoc's call.

What the first hunk gives, measured with a scratch copy of the 4p17p0 compat
file patched as above (not part of this repository) on the reproducing case:
the run completes with COM -0.29881 dB, FOM 9.9124 dB, `itick` 0, and
`sgm_Q` 2.045e-4. COM is the `N_qb = 0` value below to 3e-15 dB.

With the Rx FFE off, the cursor is the only tap and the loop skips it, so the
quantization-noise PDF stays a delta and N_qb changes no COM (it still sets the
reported `sgm_Q`, `sigma_before_clip`, `peak_clip` and `P2ptopsigma_clip`).
Whether a receiver with no Rx FFE should carry ADC noise at the cursor is a
modelling question for the ad hoc. Today the reference counts it only through
`get_PSDs`' `S_qn`, which feeds the MMSE FOM and so only runs with an Rx FFE.

## What a user can do now

- **Set `N_qb = 0`** when the Rx FFE is off. On the reproducing case both
  engines complete: COM -0.29881 dB, FOM 9.9124 dB, `itick` 0, CTLE DC gain
  -8 dB; SiCoPR against COM Octave, COM within 2.0e-15 dB and FOM within
  2.1e-13 dB.
  This is what the first hunk of the recommended fix would give, because with no
  Rx FFE the reference adds no quantization noise to the COM PDF either way. The
  four ADC fields above are then not reported.
- **Or keep one Rx FFE tap** (`ffe_post_tap_len = 1`, say) if quantization noise
  is to be modelled; that is a different receiver, and COM will differ.

## What SiCoPR does

It stops where the reference stops, on the same condition, with a `ValueError`
that names `N_qb`, the Rx FFE and this note
(`com_functions/fn/adjust_Rx_noise_for_quantization/py_impl.py`). That is the
"make its occurrence detectable" of [`VERIFICATION.md`](VERIFICATION.md): the
outcome is the reference's, only the message is clearer. The check sits at the
point of failure rather than at configuration time, because a configuration that
never reaches the noise PDF (`ERL_ONLY`, for one) runs to completion in the
reference and must here too. The zero-tap behaviour is reproduced unchanged.
