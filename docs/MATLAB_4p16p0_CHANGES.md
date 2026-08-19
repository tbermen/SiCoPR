# Supporting MATLAB `com_ieee8023_4p16p0`

Generated 2026-08-19 from `tools/matlab_version_diff.py`, diffing
`matlab/com_ieee8023_4p15p0.m` → `matlab/com_ieee8023_4p16p0.m`.

```
identical bodies : 146
CHANGED          :   6   (all six are ported functions)
added in new     :   3   (all three already exist in Python)
removed in new   :   0
stale line refs  :   8
```

The surface is small. Six functions changed and nothing was removed, so this is
an incremental update rather than a re-port.

---

## 0. The headline: adaptive local search is now in the mainline

`OptFom_Adaptive_Local_Search`, `compute_hard_cap` and `append_csv_row` do not
exist in 4p15p0's mainline — they were only in Hansel's
`com_ieee8023_4p15p0_adaptive_local_search.m` branch. In 4p16p0 all three are
first-class functions in the released file. The method this project set out to
support has been adopted upstream.

That also changes what "correlating against MATLAB" means: adaptive search is no
longer a branch to be justified, it is the reference behaviour.

---

## 1. Work required, in priority order

### 1.1 `COM_FD_to_TD` — +20/−2, **highest risk**

Four separate changes:

**(a) Amplitude scaling now reaches the pulse and step responses.** In 4p15p0
`uneq_pulse_response` is built at L1234 from the *unscaled* impulse, and only
`uneq_imp_response` is multiplied by `chdata(i).A` at L1267. 4p16p0 adds:

```matlab
if USE_channel_amplitude
    chdata(i).uneq_imp_response   = chdata(i).uneq_imp_response  *chdata(i).A;  % existed
    chdata(i).uneq_pulse_response = chdata(i).uneq_pulse_response*chdata(i).A;  % NEW
    chdata(i).uneq_step_response  = chdata(i).uneq_step_response *chdata(i).A;  % NEW
end
```

So the pulse response becomes `A` times larger. With a typical `A_v` this is a
factor of ~2, not a rounding difference.

**This is the one item that can move reported numbers.** `uneq_pulse_response`
is read by nine Python modules, including `OptFom_Calculate_Settings` and
`OptFom_Compute_CTLE`, which are on the COM path — so "reporting only" must be
proven, not assumed. Directly affected reported columns include
`peak_uneq_pulse_mV` and `steady_state_voltage_mV`, both of which currently
match 4p15p0 exactly (stage 3 "Pulse TD" is at 100% agreement).

**(b) New step responses** — `uneq_step_response`, `_raw`, `_orig`,
`_orig_filtered`, each `cumsum` of the pulse response decimated by
`samples_per_ui`; plus `uneq_pulse_response_raw_filtered` and
`uneq_step_response_raw_filtered`. Python has none of these
(`grep -c uneq_step_response` → 0). Additive: new fields, no existing value
changes.

**(c) Degenerate common-mode guard.** The `scd21_orig` / `sdc21_orig` impulse
conversions are now wrapped in `if mean(abs(...)) > 1e-6`, with an else branch
setting `t_*_fltr = t_raw` and a zero pulse response — "some test fixtures have
almost zero CM and will cause TD conversion to fail".

**(d) Typo fix**: `truncation__CD_dB` → `truncation_CD_dB` (and `__DC_` → `_DC_`).
Python carries the double-underscore spelling at
`COM_FD_to_TD/py_impl.py:156` and `:161`. Renaming is a breaking change for
anything reading those fields — grep before renaming.

### 1.2 `read_ParamConfigFile` — +3/−1, **COM-affecting**

```matlab
-param.clip_method = xls_parameter(parameter,'Clip Method',false,'Fast');
+param.clip_method = xls_parameter(parameter,'Clip Method',false,'Slow');
+param.NonZeroLSMethod    = xls_parameter(parameter,'Non-zero Local Search Method',true,0);
+param.Overwrite_Min_Radius = xls_parameter(parameter,'Overwrite Minimum Radius',true,[]);
```

- **The `Clip Method` default flips `Fast` → `Slow`.** Any config that does not
  name the keyword changes behaviour. Our 208 reference configs set it
  explicitly (`ClipMethodSlow` in the workbook names), so the correlation corpus
  is unaffected — but other configs are. Python still defaults to `'Fast'`
  (`read_ParamConfigFile/py_impl.py:482`).
- `NonZeroLSMethod` — Python already has it (`:634`).
- `Overwrite_Min_Radius` — **absent from Python entirely.**

### 1.3 `OptFom_Adaptive_Local_Search` — mainline version differs from the branch we ported

Our port follows Hansel's branch. The mainline version is not identical:

| | branch (what Python has) | 4p16p0 mainline |
|---|---|---|
| signature | `(LocalSearch_Value, BEST, THIS, …)` | `(LocalSearch_Value, **Overwrite_Min_Radius**, BEST, THIS, …)` |
| `min_radius` | forced to `1` unconditionally | `1` if `num_txffe_runs == 1` else `2`, then overridden by `Overwrite_Min_Radius` when set |
| init guard | `persistent … initialized`; `isempty(initialized) \|\| iter_count == 1` | `initialized` dropped; plain `iter_count == 1` |
| `vga_index` | `isfield` fallbacks | `THIS.vga_index = 1; BEST.vga_index = 1` outright |

Python hardcodes `min_radius = 1` with the comment *"Hansel forces min_radius = 1"*
(`OptFom_Adaptive_Local_Search/py_impl.py:85`). Under 4p16p0 the default is **2**
whenever more than one TXFFE candidate is swept — which is the normal case. This
changes the pruning radius and therefore which candidates are evaluated.

`compute_hard_cap` and `append_csv_row` are **byte-identical** to the branch, so
those two need no work.

### 1.4 `optimize_fom` — +16/−0

Adds the `FOM_history` / `iter_count` bookkeeping and the
`NonZeroLSMethod == 1` dispatch. Python already has all of this. Two deltas to
apply: thread `Overwrite_Min_Radius` through to the ALS call, and confirm the
history cap matches — MATLAB trims `FOM_history` to `length(loop_range)`.

### 1.5 `get_TDR` — +15/−0

New early return when `mean(abs(RL)) < 1e-6`, yielding a degenerate result
(`tdr = ones(1000,1)`, `ERL = inf`, `ERLRMS = -300`, zeroed arrays). Guards a
perfectly matched or zero-reflection channel. Python has no equivalent.
Self-contained and low risk.

### 1.6 `OptFom_Create_Output` — +5/−6

Stops destructively truncating `PR` (`PR = PR(ibeg:iend)`) and slices at each
use instead. Appears numerically neutral within the function; worth confirming
nothing downstream depended on the truncated `PR`.

### 1.7 `get_PSDs` — +1/−1, no action

`strcmpi(param.clip_method,'slow')` → `'Slow'`. `strcmpi` is case-insensitive so
this is cosmetic, and Python already compares `.lower() == 'slow'`
(`get_PSDs/py_impl.py:480`).

---

## 2. Verification problem to settle first

**We have no 4p16p0 reference results.** The 208-case corpus in
`tests/2_Results_COM_Matlab/` was produced by 4p15p0. Porting these changes
without new reference data means giving up the correlation that currently backs
the port (FOM bit-exact 198/208).

Three options:

1. **Keep 4p15p0 as the correlation baseline** and add 4p16p0 behaviour behind a
   version switch. Preserves the evidence; costs a branch in the code.
2. **Ask Hansel for a 4p16p0 run of the same 208 cases.** Now a much smaller
   favour than the per-function dump discussed earlier — he re-runs an existing
   script — and adaptive search being mainline gives him a reason to want it.
3. **Port and self-verify**, accepting that agreement with 4p15p0 will break on
   the pulse-amplitude change with nothing to check the new behaviour against.

Option 1 or 2. Option 3 discards the project's main asset.

---

## 3. Suggested order

1. Zero-risk additions first: `get_TDR` guard, `COM_FD_to_TD` step responses and
   CM guards, `OptFom_Create_Output` cleanup. Verify with `tools/bench_com.py`
   that no reported field moves.
2. `Overwrite_Min_Radius` end to end: config read → `optimize_fom` → ALS, plus
   the `min_radius` 1-vs-2 rule. Re-run the adaptive-vs-full-grid study; adaptive
   must still equal full grid.
3. The two behaviour changes last, together, since both need reference data:
   the pulse/step `A` scaling and the `Clip Method` default.
4. Re-base the `MATLAB lines:` citations. They currently track **4p14p0** (127 of
   132 first citations), so they are already two versions stale;
   `tools/matlab_version_diff.py` lists the drift.

## 4. Reproducing this report

```bash
python tools/matlab_version_diff.py matlab/com_ieee8023_4p15p0.m \
                                    matlab/com_ieee8023_4p16p0.m --show-diff
python tools/matlab_version_diff.py --self-check      # validates the tool
```
