# Supporting MATLAB `com_ieee8023_4p17p0`

Written 2026-10-03 from `tools/matlab_version_diff.py`, diffing
`matlab/com_ieee8023_4p16p0.m` → `matlab/com_ieee8023_4p17p0.m`.

```
identical bodies : 149
CHANGED          :   7   (all seven are ported functions)
added in new     :   4   (the apparent-channel-bandwidth family)
removed in new   :   0
stale line refs  :   1   (s_for_c4, a citation only)
```

The file is release `4.17.0` of https://opensource.ieee.org/802-com/com_code
(tag `4.17.0`, commit `47459695`, 2026-09-23), byte-identical to
`release/com_ieee8023_4p17p0.m` at that tag; SHA-256 in [`../NOTICE`](../NOTICE).
Upstream's own history between `4.16.0` and `4.17.0` is six commits: the
`MMSE_FOM` HH speed-up (Adam Gregory), the apparent-bandwidth functions (Rich
Mellitz), and the `msgbox` in `MLSE_U1_c_178A` commented out.

4p17p0 is the default emulated release from 2026-10-03 ([`VERSIONS.md`](VERSIONS.md)).

---

## 1. `MMSE_FOM`, `MMSE`, `FOM_rxffe_floating_taps`: the Gram matrix by lag

4p16p0 formed `HH = H'*H` from the selected columns of `H` on every call:
about 130,000 calls per case from the floating-tap search. 4p17p0 forms
`HH_val = H(:,1)'*H` once in `MMSE` (L2584), passes it down through
`FOM_rxffe_floating_taps` (a new trailing argument, L2099), and in `MMSE_FOM`
(L2640-2702) builds `HH` by lag over the kept columns:

```matlab
keep_idx = 1:Nfix;  if param.N_bg ~= 0, keep_idx = [1:Nfix idx+param.RxFFE_cmx+1]; end
lag_idx  = abs(keep_idx(:) - keep_idx) + 1;
HH       = HH_val(lag_idx);
Hb = H(d+2:d+Nb+1, keep_idx);  h0 = H(d+1, keep_idx);
```

**When it equals `H'*H`.** `H(:,i)'*H(:,j)` is a function of `|i-j|` only when
every column of `H` holds all of `h`'s nonzero samples. `MMSE` builds `H`
from a truncated `h` when `length(samp_idx) < num_ui`, but what it drops there
is only the zero padding of `h` (padded out to `num_ui`), so every `H` `MMSE`
builds qualifies and the two forms agree in exact arithmetic. What changes is
summation order: COM, FOM and taps move by ~1e-14. The upstream comment calls
the block equivalent, and in the reference's own use it is.

**The port.** It follows the reference's form exactly under 4p17p0 and keeps
`H'*H` for 4p15p0 and 4p16p0. `MMSE` used to carry its own copy of `MMSE_FOM`;
that copy was replaced by an import of the canonical (behaviourally identical,
checked) so the change lives in one place.

**Verified against COM Octave** running `octave/com_ieee8023_4p17p0_octave_compat.m`:
`MMSE_FOM` on a full Toeplitz `H` and on an `H` cut across nonzero samples (the
case that tells the forms apart: FOM 16.32 dB under 4p17p0, 16.88 dB under
4p16p0, pinned as the negative control), ~1e-13; the whole of `MMSE` with
floating taps placed by the FOM search, ~2e-13.

## 2. `get_ACBW` and three helpers: apparent channel bandwidth (new)

`get_ACBW` (L6759), `get_BW_from_CICP_residual` (L6893),
`get_CICP_fit_residual` (L6997), `get_CICP_fit_sweep` (L7077). The Cumulative
Inverse-Channel Penalty, `cumsum(1./|H|^2)` normalised to its last value in dB,
is fitted with the COM IL basis `a0 + a1*sqrt(f) + a2*f + a3*f^2` over a
sweep of windows `[10, f_upper]` GHz, keeping the window whose residual is
flattest over 20-60 GHz; the bandwidth is the first frequency at or above
10 GHz where the slope of the smoothed residual reaches `T_dev` dB/GHz, or
the last frequency if none does.

Called from `FD_Processing` (L1868-1872), for the THRU channel, only when the
workbook sets **`ACBW = 1`**; reported as `ACBW_GHz`. **Off by default**, so no
existing workbook changes behaviour. New keywords: `ACBW` (default 0) and
`T_dev` (default 1). The port reads both under 4p17p0 only.

MATLAB built-ins numpy does not reproduce, now classified in
[`../com_functions/verification/builtins.md`](../com_functions/verification/builtins.md):
`movmean(x, k, 'omitnan')` (an even window is lopsided, the ends shrink),
`gradient(y, x)` (centred difference, not numpy's non-uniform formula), and
`polyfit` (QR, then a triangular solve).

**Reference behaviour reproduced as it stands** (for the COM ad hoc):

| where | what |
|---|---|
| `get_ACBW` L6847 | `isfield(param,'smooth_window_ghz')` tests the lower-case name and then sets `smooth_window_GHz`, so the smoothing window is always 2 GHz, even if a caller set one |
| `get_CICP_fit_sweep` L7151-7152 | the defaults assign `f_test_min_GHz` twice (20, then 60) and never default `f_test_max_GHz`; harmless because `get_ACBW` always passes both |
| `get_CICP_fit_sweep` L7186-7190 | `smoothed_residual`, `rms_residual_smoothed`, `idx_ref`, `fGHz_fit` computed and never used; so `Nwin` has no effect |
| `FD_Processing` L1870 | the fit is stored into a stray variable, `CICP_fit_chdata(i).db`, so `chdata` gets no fit field |

**Verified against COM Octave**: the bandwidth in both branches (a crossing at
21.80782 GHz; no crossing, reported at 100 GHz), `CICP_db` exact, the
residual slope to 1e-11; `get_BW_from_CICP_residual` on an uneven axis with NaNs
and an odd window to 1e-15, signed zeros included. The fit coefficients agree
to ~1e-11 relative: the 4x4 normal equations have cond ~1e10, so last-bit
differences in forming `fmbg'*fmbg` are amplified, as they would be between
MATLAB and Octave; every decision the fit drives is exact.

## 3. Small changes

| function | change | port |
|---|---|---|
| `get_ILN` | returns `alpha` as a third output; no caller reads it | opt-in `return_alpha=True` |
| `read_ParamConfigFile` | `T_dev`, `ACBW` | read under 4p17p0 only |
| `MLSE_U1_c_178A` | the truncation `msgbox` commented out | no change: the port prints the warning and never had a `msgbox` |

## 4. Octave

`octave/make_octave_compat.py` generates `octave/com_ieee8023_4p17p0_octave_compat.m`;
every patch anchor matched unchanged. One patch is not applied: the
`FOM_rxffe_floating_taps` speed rewrite inlines 4p16p0's `H(:,sel)'*H(:,sel)`,
so on 4p17p0 it would compute the previous release's arithmetic. The release's
own search runs instead, and with it the compiled search kernel stays off
(it is reached only through that patch). One case, the shipped example: Octave
789 s, SiCoPR 374 s.

## 5. Measured effect

On the shipped example under 4p17p0, SiCoPR and COM Octave agree on COM within
8e-15 dB, FOM 1e-13, with the same sampling phase, CTLE and Tx FFE; COM is the
4p16p0 value. The 208-case corpus under 4p17p0, both engines (2026-10-04): COM within
5.2e-14 dB, FOM 6.0e-12 dB, itick, CTLE gain, Tx FFE and ERL identical on 208 / 208;
against the 4p15p0 COM Octave results of the same corpus, COM within 5.0e-14 dB and
itick identical on all 208, so on this corpus the releases agree to arithmetic noise
([`../benchmark/208_case_4p17p0/`](../benchmark/208_case_4p17p0/README.md)).

## 6. Reproducing this report

```powershell
python tools/matlab_version_diff.py matlab/com_ieee8023_4p16p0.m matlab/com_ieee8023_4p17p0.m
python tools/matlab_version_diff.py --self-check
```

The differ missed declarations continued with `...` until this intake (it
reported one new function, not four, and had folded `s21_to_impulse_DC` into its
neighbour in every release since 4p14p0); fixed, with
`tests/test_matlab_version_diff.py`.
