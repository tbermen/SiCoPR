# Supporting MATLAB `com_ieee8023_4p18p0`

Written 2026-11-19 from `tools/matlab_version_diff.py`, diffing
`matlab/com_ieee8023_4p17p0.m` → `matlab/com_ieee8023_4p18p0.m`.

```
identical bodies : 153
CHANGED          :   6   (all six are ported functions)
added in new     :   2   (fast_clip_topK, optimize_fom_BestEQ)
removed in new   :   1   (OptFom_Itick_LocalSearch)
```

A raw line diff agrees: two `function` declarations added, one removed, one
signature changed (`OptFom_Setup_Sampler_Sweep` lost its `middle_search` output).

The file is release `4.18.0` of https://opensource.ieee.org/802-com/com_code
(tag `4.18.0`, commit `5ff60720`, 2026-11-17), byte-identical to
`release/com_ieee8023_4p18p0.m` at that tag; SHA-256 in [`../NOTICE`](../NOTICE).
Upstream's history between `4.17.0` and `4.18.0` is three merged branches: an
itick local search (`Itick_BoxSearch`), a fast top-K Gaussian-mixture ADC clip
solver (`adc_clip_fast_1`), and a fix for Clip Method = Fast with deferred
floating Rx FFE (`OptimizeFOM_OneRun`).

**4p18p0 is supported, not the default.** `--matlab-version 4p18p0` (or a
`COM Version` keyword) selects it; the default stays 4p17p0 until the owner
moves it ([`VERSIONS.md`](VERSIONS.md)).

**Results can change.** Unlike 4p17p0, this release changes the equalizer
search for ordinary workbooks: see section 1.

---

## 1. The itick local search runs in full-sweep mode

In 4p17p0 the sampling-phase (itick) local search, `OptFom_Itick_LocalSearch`,
ran only under `TS_SRCH_MODE = middle`; under the default `full-sweep` it never
pruned anything. 4p18p0 removes that function and its positive/negative itick
bookkeeping and puts the search inline in `optimize_fom` (L10147-10153):

```matlab
if ~isinf(BEST.FOM) && param.LOCAL_SEARCH > 0 && ~box_search && max([THIS.g_LP_index THIS.ctle_index TK]) > 1
    if abs(THIS.itick-BEST.itick) > param.LOCAL_SEARCH
        continue;
    end
end
```

Once a first CTLE / g_DC_HP / Tx FFE setting has been searched over every
itick, later settings try only iticks within `Local Search` of the best so far.
Every workbook in the 208-case corpus and the shipped example sets
`Local Search = 2`, so **this applies to ordinary runs**. It is a search
heuristic: a case whose best equalizer sits at a sampling phase more than two
samples from the running best can now end on a different setting.

Also: `middle` is still accepted but now behaves as `full-sweep` (upstream's
comment: "Removed support for Middle Search"); `box` became a mode (in 4p17p0
it was an error, so the box code could not be reached), with a new keyword
`Itick Box Size` (default 5, read under 4p18p0 only; before, optimize_fom fixed
it at 5) and the sample range padded to a multiple of the box size; a `box`
search with `Local Search > 0` silently runs as `full-sweep`.

**Measured.** On the shipped example the change prunes heavily and agrees: the
same COM, FOM, itick and Tx FFE as 4p17p0, and the SiCoPR run took about 95 s
against about 350 s. The 208-case effect is in section 6.

## 2. `fast_clip_topK`: Clip Method = `Fast_1` (new)

A new quantisation-noise clip solver. `get_PSDs` (L8022-8044) calls it when
`Clip Method` is `Fast_1` (case-blind), with `K_eff = round(log(4096)/log(levels))`
(6 for PAM4). The K strongest cursors are enumerated exactly (`levels^K`
states), the rest and the noise are one Gaussian, and the clip level
`P(X > t) = P_qc` is found by bisection to a width of `OP.BinSize`. Off unless a
workbook asks for it. Before 4p18p0, `Fast_1` is not a method of its own and
falls to the `sum(abs(...))` branch, as it does in those releases.

## 3. `optimize_fom_BestEQ`: one more search at the best setting (new)

`com_ieee8023_` (L509-556) re-runs `optimize_fom` with every sweep variable
fixed at the best value found (Gffe, CTLE gain, g_DC_HP, itick, each Tx FFE
tap) when either:

- **`Defer_Floating_RxFFE`** (new keyword, default 0, 4p18p0 only) is set and
  the workbook has floating Rx FFE taps. `read_ParamConfigFile` (L11589-11600)
  then searches without them (`N_bg = 0`), and the re-run restores them; or
- **`Clip Method = Fast`** with MMSE Rx FFE and `N_qb ~= 0`. The search uses the
  fast `sum(abs(...))` clip; the re-run and everything after it use `Slow`. This
  is the "Bug Fix for Clip Method = Fast" of the upstream log, and **it changes
  results for workbooks that set Clip Method = Fast**: on the shipped example,
  COM 3.466 dB at itick -7 against 3.507 dB at itick -6.

The port runs the re-run on a deep copy of `param` (MATLAB's is a by-value
copy), so the caller's sweep grids survive for `Apply_EQ` and the report.

## 4. Reference behaviour reproduced as it stands (for the COM ad hoc)

| where | what |
|---|---|
| `fast_clip_topK` L6356-6358 | PAM levels come from `1:L/2` but probabilities from `ones(1,L)/L`. For an **odd PAM order** they differ in length and the first tail evaluation errors: "product: nonconformant arguments (op1 is 1x6561, op2 is 1x256)" for PAM3. The port raises the same error |
| `optimize_fom` L10014-10016, via `optimize_fom_BestEQ` | with `ctle_gdc_values` narrowed to one entry, `optimize_fom` reads `CTLE_fz(1)`, `CTLE_fp1(1)`, `CTLE_fp2(1)`: the **first** CTLE's poles, not the winner's. Harmless when the poles are one value for every gain |
| `com_ieee8023_` L534-556 | the re-run's `eq_failed` is not checked |
| `com_ieee8023_` L534-546 | the changes land on the caller's `param` and persist. With more than one package case (or an Rx calibration loop), the second and later cases search **with** floating taps (no deferral), yet still re-run; with Clip Method = Fast every later case searches with `Slow` |
| `optimize_fom` L10147 | `max([THIS.g_LP_index THIS.ctle_index TK])` leaves out the Rx FFE gain index, so each new Rx FFE gain's first setting is searched over every itick again |
| `optimize_fom` L9913-9925 | `num_itick_runs` (the progress count) is taken before the box padding; cosmetic |
| `optimize_fom` L9909 | a `box` search with `Local Search > 0` becomes `full-sweep` without a message |
| `fast_clip_topK` L6455 | `ctle_signal_sigma` is formed from `sigma_X2*sum(h_top.^2)`; the variance of the enumerated states is computed in a comment and not used |

## 5. Verified against COM Octave

`octave/com_ieee8023_4p18p0_octave_compat.m`, every change exercised:

- `fast_clip_topK`: five cases (PAM2/4/6, K above N, tied magnitudes), the
  odd-PAM error. `ctle_signal_sigma` exact; `adc_clip` exact on short pulses and
  1-2 ulp on a 200-sample pulse, which is `sum` accumulation order only (with a
  left-to-right sum the port is bit-exact, asserted in the test).
- `get_PSDs` with `Fast_1`: `adc_clip` and `ctle_signal_sigma` exact.
- `OptFom_Setup_Sampler_Sweep`, `OptFom_Set_Best_Itick`, `read_ParamConfigFile`
  (Defer, Itick Box Size): exact.
- `optimize_fom`: the CALL TRACE, the reference run with stub helpers, for
  full-sweep, middle and box, and a Tx-FFE-only case that pins the 1-based Tx
  FFE index; row for row.
- `optimize_fom_BestEQ`: call trace of what it hands `optimize_fom` and what it
  restores.
- Whole cases, the shipped example without crosstalk, SiCoPR against COM Octave
  on 4p18p0 (2026-11-19), each with the same itick and Tx FFE:

| scenario | COM Octave COM (dB) | SiCoPR − Octave |
|---|---|---|
| as shipped (Local Search 2) | 3.5070028402502937 | 8e-15 |
| `TS_SRCH_MODE = middle` | 3.5070028402502937 | 8e-15 |
| `box`, Local Search 2 (runs as full-sweep) | 3.5070028402502937 | 8e-15 |
| Clip Method = Fast | 3.4663241000181815 | -4e-15 |
| Clip Method = Fast_1 | 3.5070028402502986 | 3e-15 |
| Defer_Floating_RxFFE = 1 | 3.363393109175547 | -8e-15 |
| Defer and Clip Method = Fast | 3.3708264768847447 | 2e-15 |
| `box`, Local Search 0, narrowed grid | 2.9315044826454195 | -5e-15 |
| `box`, Local Search 0, Itick Box Size 4, narrowed grid | 2.9315044826454195 | -5e-15 |
| full-sweep, Local Search 0, same narrowed grid | 2.9315044826454195 | -5e-15 |

With Local Search 0, box mode on the full grid (1584 Tx FFE settings x 147
CTLE combinations) ran past the 2 h limit of `tools/octave_compare.py` in
both engines, so box mode is checked on a narrowed grid of 54 Tx FFE settings
x 15 CTLE combinations (`c(-2)` 0:0.02:0.04, `c(-1)` -0.1:0.02:0, `c(1)`
-0.04:0.02:0, `g_DC` -12:-8, `g_DC_HP` -2:0). On that grid all three runs pick
itick -14, Tx FFE [0, -0.1, 0.9, 0] and g_DC -12 in both engines. Box mode
finds the full-sweep optimum, and the other fields agree with Octave to
within 1e-12. The unit-level call trace covers box mode on its own.

Found on the way, a port defect fixed before any result: the narrowed CTLE and
g_DC_HP grids were numpy scalars, which `OptFom_Calculate_Settings` cannot take
`len()` of (Clip Method = Fast crashed); MATLAB's `x(k)` is 1x1, so they are
one-element arrays now, with a test.

## 6. Measured effect on the 208-case corpus

The 208 cases, run 2026-11-19 at SiCoPR `fc30ae2` with `--matlab-version 4p18p0`,
against COM Octave running 4p18p0 (results in
[`../benchmark/208_case_4p18p0/`](../benchmark/208_case_4p18p0/README.md)).

**SiCoPR against COM Octave, 4p18p0:** no errors. COM agrees to within 5.2e-14 dB
(median 8.9e-15) and FOM to within 6.0e-12 dB. itick, CTLE gain, Tx FFE and ERL
are identical on all 208 cases, the same agreement as 4p17p0.

**4p17p0 against 4p18p0, COM Octave on both sides:** 203 of the 208 cases are
identical on every field. On the other 5, the itick local search of section 1
ends on a different sampling phase, and COM rises in every one:

| case | COM 4p17p0 (dB) | COM 4p18p0 (dB) | itick | Tx FFE |
|---|---|---|---|---|
| wXtalk_T3_R17 | 4.538 | 4.685 | 6 → -20 | [0 -0.1 0.9 0] → [0 0 0.98 -0.02] |
| woXtalk_T1_R20 | 5.969 | 6.317 | -19 → 3 | unchanged |
| woXtalk_T2_R17 | 5.645 | 5.978 | -20 → 3 | unchanged |
| woXtalk_T2_R18 | 5.387 | 5.559 | -20 → 2 | unchanged |
| woXtalk_T3_R17 | 4.875 | 5.131 | -19 → 3 | unchanged |

The largest change is 0.348 dB. All five cases pass the 3 dB threshold under
both releases, so no pass/fail verdict changes. The search maximises FOM, not
COM, so a narrower itick search can end on a higher COM, and here it always did.

The corpus took 4.8 h of Octave case time and 3.5 h of SiCoPR case time, against
30.2 h and 18.1 h under 4p17p0. That is not a like-for-like comparison: the 4p17p0
run used 4 to 5 jobs at once and this one used 2, so the time per case is not
comparable.

## 7. Octave

`octave/make_octave_compat.py` generates the 4p18p0 file; every patch anchor
matched. 4p18p0 changed neither `FOM_rxffe_floating_taps` nor `MMSE_FOM`
(`tests/test_octave_compat.py` checks the bodies are byte-identical to
4p17p0's), so 4p18p0 takes the 4p17p0 variant of the floating-tap speed patch
and the compiled kernels as they are.

## 8. Reproducing this report

```powershell
python tools/matlab_version_diff.py matlab/com_ieee8023_4p17p0.m matlab/com_ieee8023_4p18p0.m
python tools/matlab_version_diff.py --self-check
```
