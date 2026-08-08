# COM — Channel Operating Margin (Python)

A Python port of the IEEE 802.3ck/dj **COM** (Channel Operating Margin) MATLAB reference
tool (`com_ieee8023_4p14p0.m`). It computes COM / VEO / VEC and the supporting equalization
and noise analysis for a serial channel described by Touchstone S-parameter files and an
Excel configuration spreadsheet.

---

## 1. Requirements

- **Python 3.9+** (developed and tested on 3.14)
- Packages in [`requirements.txt`](requirements.txt):

| Package | Required | Used for |
|---|---|---|
| `numpy` | yes | all numerical computation |
| `scipy` | yes | signal processing, interpolation, special functions, linear algebra |
| `openpyxl` | yes | reading the `.xlsx` configuration spreadsheet |
| `matplotlib` | only for plots | figures, written when `SAVE_FIGURES` is enabled in the config |

## 2. Installation

```powershell
# (recommended) create an isolated environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
```

That's it — there's nothing to build. The tool runs directly from `com.py`.

## 3. Running

```powershell
python com.py <config.xlsx> <thru.s4p> [--fext f1.s4p f2.s4p ...] [--next n1.s4p ...]
```

- The **THRU** (victim) channel is the required `<thru.s4p>` argument.
- Crosstalk aggressors are optional: `--fext` for FEXT, `--next` for NEXT (each accepts
  multiple files).

**Example** (the bundled reference channel, no aggressors):

```powershell
python com.py "config_com_ieee8023_93a=3ck_d3p1_120g_C2M_tp1a_9_11_30_21.xlsx" "Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_thru1.s4p"
```

```
--- Case 1 ---
  COM_dB                         = 3.5664
  VEO_mV                         = 11.5800
  Result                         = PASS  (threshold 3.0 dB)
--- Case 2 ---
  COM_dB                         = 3.0184
  Result                         = PASS  (threshold 3.0 dB)
```

One result block is printed per package test case in the config. If `SAVE_FIGURES` /
`CSV_REPORT` are enabled, per-case outputs are written under
`results/<config-name>_<timestamp>/case_NN/` (a `results.csv` plus figures).

### Engineering `.mat` export for R analysis (optional)

Add `--export-mat` to write a per-case MATLAB v5 snapshot next to the standard
outputs (`results/<config-name>_<timestamp>/<config-name>_caseNN.mat`):

```powershell
python com.py <config.xlsx> <thru.s4p> --export-mat
```

This is an **additive debug export** — it does not change any COM result, report,
or figure. It captures the intermediate quantities COM already computed for each
case: the frequency-domain equalizer chain (`H_channel`, `H_ctle`, `H_ch_ctle`,
`H_ffe`, `H_tx`, `H_final`, crosstalk `H_next`/`H_fext`), per-stage impulse and
pulse responses (`h_*`, `pulse_*`), the statistical eye/BER contour, the combined
and component noise PDF/CDF, all COM metrics (`results_full`), a self-describing
`config` struct, and run `meta` (timestamp, versions, input files). It also adds
`FOM_gauss_dB` — a simplified "Gaussian-noise COM": the COM you would get if the
combined interference+noise distribution were purely Gaussian (`A_ni_gauss =
Q^-1(specBER)·sigma_total`); the gap to `COM_dB` quantifies the non-Gaussian tail
penalty. Other extras: `H_rx_filter` (Rx Butterworth×Bessel×raised-cosine
bandwidth filter) and `H_ctle_rx`; `H_ctle_all` + `ctle_gdc_values` (the full
selectable CTLE bank); and `run_summary` (EQ sweep dimensions, number of cases,
wall-clock run time). The file is
readable by `R.matlab::readMat()`. `H_channel`/`H_ctle`/`H_ch_ctle` are genuine
captured FD responses; `H_ffe` is evaluated from the selected Tx-FFE taps and
`H_tx`/`H_final` are their products (DFE is a time-domain term, excluded from the
FD chain) — see `meta.notes` in each file.

Interactive R/Plotly visualisation (channel FD, CTLE, CTLE bank, impulse, pulse,
eye + BER density, equalizer contribution) plus a single combined HTML dashboard:

```powershell
# one-time: install.packages(c("R.matlab", "plotly", "htmltools"))
Rscript R/com_analysis.R results/<...>/<config-name>_case01.mat
# -> writes <config-name>_case01_report.html
```

## 4. Input files

### Configuration spreadsheet (`.xlsx`)
Follows the IEEE 802.3 COM spreadsheet format. The active sheet is **`COM_Settings`**
(parameter / value columns). Key parameters: signalling rate (`f_b`), modulation
(`levels`), DFE taps (`N_b`), BER target (`specBER`), CTLE sweep, package length cases,
and output flags (`SAVE_FIGURES`, `CSV_REPORT`). Start from an IEEE 802.3 working-group
reference spreadsheet.

### S-parameter files (`.s4p`)
4-port Touchstone. The default differential port order is `[1, 3, 2, 4]`; override with
`snpPortsOrder` in the config if your files differ. Crosstalk aggressor files use the same
format (the differential `Sdd21` becomes the coupling response).

## 5. Project layout / development

| Path | What it is |
|---|---|
| `com.py` | **the tool** — assembled, runnable. *Do not edit by hand* |
| `com_functions/fn/<name>/py_impl.py` | per-function source (the editable code) |
| `com_functions/fn/<name>/test_verify.py` | per-function unit tests |
| `assemble_com.py` | concatenates the `py_impl.py` files into `com.py` |
| `com_plots.py` | figure generation (driven by `SAVE_FIGURES`) |
| `com_mat_export.py` | engineering `.mat` export (driven by `--export-mat`) |
| `R/com_analysis.R` | R/Plotly plots + HTML dashboard for the exported `.mat` |
| `com_ieee8023_4p14p0.m` | the MATLAB reference (ground truth) |

To change behaviour: edit the relevant `py_impl.py`, run its `test_verify.py`, then
re-assemble:

```powershell
python assemble_com.py
python -m pytest com_functions/fn/<name>/test_verify.py -q
```

## 6. Verification status & known limitations

Every major feature is **implemented and unit-tested** (full function parity with
`com_ieee8023_4p15p0.m` + the adaptive-local-search branch): TxFFE/CTLE/DFE, **RxFFE
(MMSE)**, floating DFE / floating RxFFE taps, **MLSE**, **crosstalk (FEXT/NEXT, ICN)**,
common-mode modal masks, **RX calibration**, FD processing (ICN/ILD), ERL/TDR, and
TD-ILN/RILN. The `com_functions/fn` unit suite is green (**864 passed, 0 failed**), and the
bundled 802.3ck C2M channel runs **end-to-end** reproducing the documented COM to 4 decimals
(Case 1 = 3.5664 dB, Case 2 = 3.0184 dB).

Note the distinction: the C2M **TxFFE/CTLE/DFE** path is exercised end-to-end by that run;
the other features above are implemented + unit-tested but **not yet exercised end-to-end**
(no bundled config/channel toggles them — RxFFE, MLSE, crosstalk aggressors, GET_FD ICN/ILD,
calibration, modal masks, N_qb). See `MISSING_FEATURES_PLAN.md` §D for how to exercise each.

Honest caveats for anyone relying on the numbers:

- **Numeric parity with MATLAB has not yet been cross-checked** against a MATLAB reference
  run (none was available during development). Verification is by faithful line-by-line
  translation, per-function unit tests, internal consistency, and physical behaviour — plus
  the end-to-end reproduction noted above. (The initial audit's 7 candidate bugs were
  re-checked against the 4p15p0 source and found unfounded; see `audit_state.json`.) A
  one-time MATLAB golden run (same config + the emitted `results.csv` columns) would allow a
  mechanical CSV diff to lock in bit-for-bit parity.
- **`FFE_OPT_METHOD='WIENER-HOPF'`** is intentionally non-functional — its helper
  (`WIENER_HOPF_MMSE`) is undefined in the MATLAB reference itself. Use `'MMSE'`.
- **`FAST_NOISE_CONV`** is an optional speed *approximation* (small noise taps lumped into
  one Gaussian); the default exact path is recommended for reported results.
- **GUI file pickers** are not ported (file lists are always passed on the command line).

## 7. r4p15p0 deltas + adaptive local search

On top of the `com_ieee8023_4p14p0.m` base, the following from `com_ieee8023_4p15p0.m`
and Hansel D'silva's `com_ieee8023_4p15p0_adaptive_local_search.m` branch are incorporated:

- **Automatic port-order detection** (`auto_port_order`). When the config's `Port Order`
  is empty, `0`, contains `NaN`, or is not a single 4-element vector (e.g. a per-package
  `[1 2 3 4; 1 3 2 4]` matrix), the differential port order is auto-detected from the
  S-parameters (through energy + a phase-delay near/far test). The resolved order is
  reported as `output_args.port_order`. Validated on the 802.3dj KR channels (resolves the
  standard `[1 3 2 4]`).
- **Adaptive local search** (`OptFom_Adaptive_Local_Search`, Hansel's branch). Selected with
  the new config keyword **`Non-zero Local Search Method`** (`param.NonZeroLSMethod`): with
  `Local Search > 0`, set it to `1` for the adaptive method or `0` for the legacy
  `OptFom_Local_Search`. `Local Search = 0` is the full grid. This makes the three EQ-search
  methods directly comparable. The adaptive method keeps a persistent search radius that
  grows/shrinks with recent FOM improvement and prunes candidates by a weighted L1/L2
  tap-space distance from the current best. Optional per-candidate CSV logging is available
  via `OptFom_Adaptive_Local_Search.ALS_LOG_CSV` (off by default; the data source for a
  sweep-vs-adaptive comparison plot).
- **Minor r4p15p0 fixes**: `get_PSDs` crosstalk pad length, `adjust_Rx_noise_for_quantization`
  cursor-tap off-by-one, and port-order threading through the Touchstone readers.
- **Two pre-existing gaps fixed while exercising the 802.3dj config** (unrelated to the
  r4p15p0 diff, never hit by the bundled 802.3ck config): named/complex packages
  (`param.PKG`) were accessed by dict subscript instead of attribute, and a 2-row
  `Port Order` matrix now routes to auto-detection.

## 8. EQ-search method comparison & visualization

Tooling to compare the three EQ-search methods (full grid / legacy local search / adaptive
local search) and visualize how each walks the equalizer settings-vs-FOM landscape.

**Instrumentation.** `optimize_fom` has an opt-in, per-candidate logger (off by default, no
effect on any COM result): set `com.SWEEP_LOG_CSV` to a path and `com.SWEEP_METHOD_LABEL` to a
label, and one row is written per TX-FFE candidate considered — method, CTLE/LP indices, tap
vector, candidate FOM, running-best FOM, and `evaluated` (1) vs search-pruned (0). The adaptive
method's per-iteration radius diagnostic is captured separately via `com.ALS_LOG_CSV`.

```powershell
# 1) run all three methods on one channel -> sweep_results/{full_grid,legacy,adaptive}_log.csv
#    --max-ctle / --max-tap-vals cap the (otherwise huge) full grid to a tractable size
python sweep_compare.py <config.xlsx> <thru.s4p> --local-search 2 --max-ctle 3 --max-tap-vals 3

# 2) static matplotlib figures (convergence, effort, tap-space coverage, parallel-coords,
#    adaptive-radius diagnostic, summary table) -> sweep_results/*.png + summary.csv
python plot_sweep_compare.py sweep_results

# 3) interactive R/Plotly dashboard (hover to read exact EQ settings/FOM) -> sweep_results/sweep_compare.html
#    one-time: install.packages(c("plotly", "htmltools"))
Rscript R/sweep_compare.R sweep_results
```

The figures answer: does the adaptive method reach the full-grid optimum (convergence +
summary), at how much less compute (effort + speedup), and which region of the TX-FFE
tap space does each method explore vs skip (coverage scatter / parallel coordinates).
