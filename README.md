# COM — Channel Operating Margin (Python)

A Python port of the IEEE 802.3ck/dj **COM** (Channel Operating Margin) MATLAB reference
tool. It computes COM / VEO / VEC and the supporting equalization and noise analysis for a
serial channel described by Touchstone S-parameter files and an Excel configuration
spreadsheet.

The port is function-for-function: `com.py` follows the structure of the MATLAB source
closely enough to navigate by MATLAB line number, takes the same `.xlsx` + `.s4p` inputs,
and produces the same outputs.

**Two MATLAB releases are supported.** `com.py` emulates **`com_ieee8023_4p15p0`** by
default — that is the release the 208-case reference corpus was produced with, so it is the
version the correlation result in §7 is evidence for — and **`com_ieee8023_4p16p0`** via
`--matlab-version 4p16p0` (or `com.COM_MATLAB_VERSION`, or a `COM Version` config keyword).
4p16p0 is a small delta, and its measured effect on all 208 cases is in §8. Which release a
given `com.py` emulates is recorded in its header and in `VERSION.json`.

On top of the engine there is a study layer (`tools/`, `R/`) built to answer one question:
**does pruning the equalizer search grid change COM?** Results in §5.

> **New to this project?** Start with **[`COM_Python_Tutorial.docx`](COM_Python_Tutorial.docx)**
> — a 40-page tutorial and reference covering installation, architecture, every feature,
> the study layer, the R reports, a COM concepts primer, and a complete index of all 247
> configuration keywords. This README is the quick version.

---

## 1. What ships in this repository — and what doesn't

**The port ships. The correlation data does not.** The engine, its tests, the tooling
and the MATLAB reference sources are all here. The channel S-parameters and
configuration workbooks used to correlate against MATLAB are IEEE 802.3 contributions
and are not ours to redistribute — but every one of them is publicly available, and
this section says exactly which.

| Present | Not present |
|---|---|
| the engine (`com.py` + `com_functions/`), tests, tooling, R reports | channel S-parameters (`tests/0_...`) |
| `matlab/` — the BSD-3-Clause MATLAB reference sources | COM configuration workbooks (`tests/1_...`) |
| `docs/`, `VERSION.json`, `LICENSE`, `CONTRIBUTING.md` | MATLAB reference result workbooks (`tests/2_...`) |
| the correlation harness itself (`tools/matlab_compare.py`) | `tests/oracles/`, `report_data/` — MATLAB reference **values**, distilled from those workbooks |
| the study write-ups (`RESULTS.md`, `STATE.md`) | every generated output: `results/`, `report_figs/`, `corpus_results/*.csv`, `com_python_results/`, `report_docs/` |

**A fresh clone is fully functional without any of it.** The unit suite runs and
passes — 886 per-function tests plus the cross-check scripts — and every test that
needs correlation data skips cleanly and says what is missing. You only need the
data to reproduce the 208-case correlation.

Note the second row of "not present": removing the reference *workbooks* is not
enough on its own, because the same numbers had been extracted into a per-stage
oracle (`tests/oracles/`) and a comparison table (`report_data/`). Those are as
much the reference data as the workbooks are, and they are excluded too. CI
enforces it, so neither can drift back in.

### Obtaining the correlation data

All of it comes from the IEEE 802.3dj public area. Download each contribution and
unpack it into the directory shown.

| Contribution | Contents | Unpack into |
|---|---|---|
| `weaver_3dj_02_2311` | Arista CR channels | `tests/0_IEEE_802p3dj_PublicArea_CR_KR_Channels/1_Arista/` |
| `akinwale_3dj_01_2310` | Intel cable-assembly channels | `tests/0_.../2_Intel/` and `akinwale_3dj_01_2310/` |
| `akinwale_3dj_02_2311` | Intel channels | `tests/0_.../0_Intel/` |
| `lim_3dj_07_2309` | Intel + Molex channels | `tests/0_.../3_Intel_Molex/` |
| `lim_3dj_04_230629` | Intel + TE Connectivity channels | `tests/0_.../4_Intel_TEConnectivity/` |
| `lim_3dj_03_230629` | Intel + Amphenol channels | `tests/0_.../5_Intel_Amphenol/` |

The **COM configuration workbooks** (`config_com_dj_200G_CAKR_178_PKGA/B_*.xlsx`,
including the `_sweep_TxFFE` variants) and the **MATLAB reference result workbooks**
(`Results_Matlab_COM_v4p15_With/WithoutXtalk_ClipMethodSlow_AdaptiveLS.xlsx`) come
from the COM ad hoc rather than from a numbered contribution. Ask on the reflector, or
substitute your own configuration and your own MATLAB run — the harness only needs a
config per package case and a MATLAB result workbook in the standard column layout.

Once the data is in place:

```powershell
python tools/matlab_compare.py --validate      # resolve all 208 cases, run nothing
python tools/matlab_compare.py --run --modal-erl --min-radius 2 --jobs 5
python tools/export_compare_csv.py             # compare.csv, stages.csv
```

`--validate` is the useful first step: it reports which cases it can and cannot
resolve, so a partial download tells you what is missing before you spend hours
running.

## 2. Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
```

Nothing to build — the tool runs directly from `com.py`. Python 3.9+ (developed on 3.14).

The R reports additionally need, inside R:

```r
install.packages(c("plotly", "htmltools", "jsonlite", "R.matlab"))
```

## 3. Run

```powershell
python com.py <config.xlsx> <thru.s4p> [--fext f1.s4p ...] [--next n1.s4p ...]
              [--export-mat] [--matlab-version {4p15p0,4p16p0}] [--eye-under-mlse]
```

The **THRU** (victim) channel is required. Crosstalk aggressors are optional: `--fext` for
FEXT, `--next` for NEXT, each accepting multiple files.

| flag | what it does |
|---|---|
| `--export-mat` | per-case engineering `.mat` snapshot for the R dashboard (§10) |
| `--matlab-version` | which MATLAB release to emulate — **default `4p15p0`** (§8) |
| `--eye-under-mlse` | compute the eye contour and timing bathtub for plotting even when MLSE is on. MATLAB gates the eye on `MLSE == 0`, but MLSE is applied afterwards, so the pre-MLSE eye is well defined. Diagnostic only: no reported value changes. |

```
--- Case 1 ---
  COM_dB                         = 3.4694
  VEO_mV                         = 11.5200
  Result                         = PASS  (threshold 3.0 dB)
```

One result block per package test case. With `SAVE_FIGURES` / `CSV_REPORT` enabled in the
config, per-case outputs land in `results/<config-name>_<timestamp>/case_NN/`.

## 4. Repository layout

| Path | What it is |
|---|---|
| `com.py` | **the engine** — assembled, runnable. *Do not edit by hand* |
| `com_functions/fn/<name>/py_impl.py` | per-function source (the editable code), 157 functions |
| `com_functions/fn/<name>/test_verify.py` | per-function unit tests |
| `assemble_com.py` | concatenates the `py_impl.py` files into `com.py` |
| `com_plots.py`, `com_mat_export.py` | figure generation and `.mat` export — imported *by* `com.py`, so they live beside it |
| `tools/` | study layer (§5) plus the MATLAB-comparison harness, the version differ, and the oracle extractor (§6) |
| `R/` | interactive HTML reports |
| `VERSION.json` | which MATLAB release the port emulates; `assemble_com.py` generates `com.py`'s header from it |
| `matlab/` | MATLAB reference sources (`4p14p0`, `4p15p0`, `4p16p0`, adaptive-local-search branch) |
| `docs/` | audit findings, fix summary, feature plan, 4p16p0 change analysis + measured impact |
| `dev/` | audit and interface-check scripts, plus state ledgers. The development prompts under `dev/prompts/` are kept locally and not published |
| `tests/` | standalone cross-check scripts (run directly, not via pytest — see `CONTRIBUTING.md`) |
| `sweep_results/`, `corpus_results/` | the write-ups of the two EQ-search studies (`STATE.md`, `RESULTS.md`). The data files they describe are generated locally, not tracked |
| `report_docs/` | built decks and their PDF snapshots. Generated by `tools/build_*_pptx.py`; not tracked |

**What is deliberately absent.** The repository carries code, tests and guides — no inputs and no outputs. Channel S-parameters and configuration workbooks are IEEE contributions (§1); result tables, figures, decks, the per-stage MATLAB oracles under `tests/oracles/` and the comparison tables under `report_data/` are all either generated locally or distillations of the MATLAB reference workbooks, which are not ours to redistribute. Every test that needs one of those skips and says so, so a fresh clone runs green: 886 per-function tests plus the cross-check scripts.

## 5. The EQ-search study

Three ways to solve the same channel, differing only in how much of the TX-FFE / CTLE grid
they evaluate:

| Method | Switches |
|---|---|
| **full grid** | `LOCAL_SEARCH = 0` — exhaustive, the reference answer |
| **legacy local search** | `LOCAL_SEARCH = N`, `NonZeroLSMethod = 0` — fixed-radius prune |
| **adaptive local search** | `LOCAL_SEARCH = N`, `NonZeroLSMethod = 1` — Hansel D'silva's branch |

### Result (7 channels, 100–1400 mm, 2026-08-22)

**Adaptive returns COM bit-identical to the exhaustive grid on all seven channels** — ΔCOM
exactly 0 to 17 significant digits, same winning EQ operating point — while evaluating ~8%
of the candidates at a median **11.91× speedup** (range 8.60–12.52×). Legacy also matched
exactly, at 1.28×. No pass/fail flips at the 3 dB threshold, for any method.

The winning EQ setting *moves* with channel length, so the agreement is not an artifact of a
constant answer: adaptive tracks a genuinely shifting optimum while skipping ~92% of the work.

Recomputing true COM for the top-20 candidates by FOM (140 probes) shows the FOM argmax
**is** the COM argmax on 7 of 7 channels — COM regret is **zero everywhere**, with Spearman
ρ from 0.770 to 0.952. FOM ordering is still imperfect deeper in the list (ρ never reaches
1), but the disagreement no longer reaches the top of the ranking, which is the only part a
FOM-driven search uses.

> This corpus was **regenerated on the corrected engine**. The 2026-08-07 run predates the
> engine defects found by the MATLAB correlation and its COM values were materially
> wrong (100 mm moved 3.217 → 6.720 dB). Two of its conclusions changed as a result — most
> notably "FOM is not a faithful proxy", which was substantially an artifact of those
> defects. Both changes are documented in
> [`corpus_results/RESULTS.md`](corpus_results/RESULTS.md).

Full numbers, caveats, and regeneration commands: [`corpus_results/RESULTS.md`](corpus_results/RESULTS.md).

**The corpus is one channel family** (same topology, varying only cable length), one config,
thru-only, and now spans 4.45–6.72 dB — so it contains no cases near the 3 dB threshold.
Those are its main limitations and are documented alongside the results.

### Running sweep commands for com.py

```powershell
# one channel, three methods -> sweep_results/{full_grid,legacy,adaptive}_log.csv + summary.json
python tools/sweep_compare.py <config.xlsx> <thru.s4p> --local-search 2 --max-ctle 3 --max-tap-vals 3

# true COM for the top-K FOM candidates (the FOM-as-proxy measurement)
python tools/fom_com_probe.py <config.xlsx> <thru.s4p> --sweep-dir sweep_results --top-k 20

# N channels, checkpointed and resumable -> corpus_results/{runs,probe}.csv + corpus_summary.json
python tools/corpus_sweep.py <config.xlsx> --channel-dir akinwale_3dj_01_2310 --dry-run
python tools/corpus_sweep.py <config.xlsx> --channel-dir akinwale_3dj_01_2310

# reports
Rscript R/sweep_compare.R sweep_results          # single channel
Rscript R/corpus_report.R corpus_results         # corpus-wide
python tools/build_pptx.py                       # deck -> report_docs/, built from corpus_results/
```

A full 7-channel × 3-method corpus is ~7.5 h. Use `--dry-run` for the estimate,
`--methods full_grid,adaptive` to drop ~40%, and `--probe-top-k 0` to skip the proxy
measurement. Runs are checkpointed per channel, so an interrupted corpus resumes.

**Instrumentation.** `optimize_fom` carries an opt-in per-candidate logger — off by default,
no effect on any COM result. Set `com.SWEEP_LOG_CSV` and `com.SWEEP_METHOD_LABEL` for one row
per TX-FFE candidate (method, EQ indices, tap vector, candidate/best FOM, winning sample
phase, evaluated vs pruned). The adaptive method's per-iteration radius diagnostic is
captured separately via `com.ALS_LOG_CSV`. These logs are what the study measures.

`tools/hist_dep_repro.py` is a diagnostic harness: it solves the same operating point with
different numbers of candidates evaluated and diffs the full engine state at the moment COM
computation begins. Reach for it if COM ever appears to depend on search history.

## 6. Development workflow

`com.py` is generated. To change behaviour, edit the per-function source and re-assemble:

```powershell
python -m pytest com_functions/fn/<name>/test_verify.py -q   # test the change
python assemble_com.py                                        # regenerate com.py
python -m pytest com_functions/fn -q                          # full suite
```

Before committing, run the whole harness — pre-flight audit, assembly, interface
checks, unit tests, and every cross-check script:

```powershell
powershell -ExecutionPolicy Bypass -File tests/run_all.ps1
```

`tests/` holds **two kinds of file**, and the difference matters:

| | how to run |
|---|---|
| `test_smoke.py`, `test_checkpoints.py`, `test_end_to_end.py` | pytest modules |
| every other `test_*.py` | standalone scripts — `python tests/test_x.py` |

Do **not** run `pytest tests` over the whole directory. The audit scripts call
`sys.exit()` at import, which aborts collection: pytest reports `no tests ran`
**and still exits 0**, so nothing runs and nothing complains. `run_all.ps1`
dispatches each kind correctly.

Those scripts record two outcomes. `check()` is behaviour that must match MATLAB;
`xcheck()` is a reviewed, accepted divergence, which reports `XFAIL` while it
persists and **fails the run if it starts passing** — so a divergence that gets
fixed cannot leave a stale entry behind in the ledger.

Current state: **886** per-function tests, and **359 checks across 19 audit scripts**
with 16 accepted divergences.

### Cross-cutting guards

Four of those scripts exist because the per-function tests structurally cannot catch the
defect classes that actually got through. Each was built from a real failure and verified
by re-introducing it:

| script | guards against | why |
|---|---|---|
| `test_reference_leaks.py` | writing to a parameter the function never returns | MATLAB passes structs **by value**, Python by reference. **Five of the original eight** correlation defects were this class, and six of the fifteen engine fixes overall. Caught a new instance during the 4p16p0 port. |
| `test_inlined_copies.py` | an inlined copy drifting from its canonical function | there are **178 copies of 70 functions**; a fix to `py_impl.py` reaches only one of them. Engine defect #6 lived in three copies. |
| `test_optimization_invariants.py` | the speed work silently breaking | cache transparency and key completeness, the hoisted Gram matrix, FFT/direct convolution agreement, shared buffers. Found a live cache-aliasing defect. |
| `test_matlab_stage_oracles.py` | drift from real MATLAB values | pins **208 cases × 35 scalars + 14 vector families** taken from the reference workbooks — the only tests in the repo that assert against MATLAB rather than against Python. The oracle file itself is not tracked (it *is* reference data); the test skips without it. |

### Tooling for a new MATLAB release

```powershell
python tools/matlab_version_diff.py OLD.m NEW.m   # -> which py_impl files to re-check
python tools/matlab_version_diff.py --self-check  # validates the differ itself
python tools/bench_com.py --against <ref>         # speed change + full-precision fingerprint
python tools/compare_matlab_versions.py           # diff two version sweeps, field by field
```

## 7. Verification status & caveats

Every major feature is implemented and unit-tested against `matlab/com_ieee8023_4p15p0.m`
plus the adaptive-local-search branch: TxFFE/CTLE/DFE, RxFFE (MMSE), floating DFE / floating RxFFE
taps, MLSE, crosstalk (FEXT/NEXT, ICN), common-mode modal masks, RX calibration, FD
processing (ICN/ILD), ERL/TDR, and TD-ILN/RILN. The `com_functions/fn` suite is green
(**886 passed, 0 failed**), and the bundled 802.3ck C2M channel runs end-to-end.

**Numeric parity with MATLAB has been established end to end.** 208 reference cases from
Hansel D'silva's `com_ieee8023_4p15p0` runs were compared case by case:

| | result |
|---|---|
| | as supplied | **matched config** |
|---|---|---|
| FOM bit-exact | 198 / 208 | **208 / 208** |
| COM bit-exact | 199 / 208 | **208 / 208** |
| sampling phase (`itick`) exact | 200 / 208 | **208 / 208** |
| max \|ΔCOM\| | 0.185 dB | **3.3e-14 dB** |
| rms ΔCOM | 0.018 dB | **0.00053 dB** |
| pass/fail disagreements | 0 | **0** |

**Matched config** pairs each crosstalk condition with the settings its reference
workbook was actually produced with. The two references differ: the
without-crosstalk run used a single-point Tx FFE grid (the config we were given),
the with-crosstalk run used a swept one. Reproduce with
`python tools/compare_matched_config.py`; full analysis in
[`docs/TXFFE_SWEEP_ROOT_CAUSE.md`](docs/TXFFE_SWEEP_ROOT_CAUSE.md).

The with-crosstalk grid is a **reconstruction** — supported by reproducing
MATLAB's tap vector, sampling phase and FOM on all ten previously divergent
cases, and by a 2×2 control in which each condition is near-exact on its own
config and materially worse on the other's — but not yet confirmed against the
real settings.

Fifteen engine-level defects were found and fixed in the process (the ledger,
with what each one bought, is [`docs/FIX_SUMMARY.md`](docs/FIX_SUMMARY.md)). Reproduce with
`python tools/matlab_compare.py --validate --run --jobs 5`; the full write-up is
[`MATLAB_Correlation_Review.md`](MATLAB_Correlation_Review.md).

**The 10 cases that did not agree were a configuration difference, not an engine
defect, and are now resolved.** MATLAB's winning Tx FFE is non-unity on exactly
those 10, and the supplied config defines a single-point Tx FFE grid. Two settings
together reproduce MATLAB on all ten: the swept grid including
`c(-2) = [0:.02:0.14]`, and adaptive-search `min_radius = 2`. Re-run with
`--txffe-sweep --min-radius 2`.

That result is against **4p15p0**, which is why it stays the default emulation target — see
§8. The same corpus has been run in 4p16p0 mode: 210 of 213 output columns are identical
on all 208 cases, and no COM/FOM/VEO/VEC/itick/ERL value moves.

Honest caveats for anyone relying on the numbers:

- **The eight sampling-phase divergences are resolved** — a Tx FFE search-space
  mismatch, not an engine defect. One question remains open with Hansel: the exact Tx
  FFE tap ranges and `Overwrite Minimum Radius` used for the with-crosstalk run, since
  the config supplied to us matches his without-crosstalk run. The search *method* is
  not a mismatch — the reference workbooks are named `..._AdaptiveLS.xlsx` and the
  port's `Non-zero Local Search Method = 1` matches them.
- **The MATLAB-vs-Python runtime comparison is not like-for-like** and should not be
  quoted until re-measured: Python searched one Tx FFE candidate per CTLE where MATLAB
  swept a grid. The Python-vs-Python speed-up is unaffected.
- **The COM PDF residual is fully closed.** Two separate defects: an off-by-one in the
  ADC-clip sampling phase (`docs/COM_PDF_RESIDUAL.md`), which took COM bit-exact from
  170 → 198 of 208 and cleared the last pass/fail disagreement; and a banker's-rounding
  tie in `nui = round(len/M)` (`MATLAB_Correlation_Review.md` §4.3), which dropped one
  ISI sample on 4 case-instances. With both fixed, the settings-aligned correlation is
  **bit-exact on FOM, COM and sampling phase across all 208 cases**, max |ΔCOM| 3.3e-14.
- **Results produced before August 2026 are not comparable to current output.** The nine
  engine fixes changed COM materially — the largest single correction removed a systematic
  FOM bias affecting 95.7% of cases. Regenerate rather than compare against archived numbers.
- Only the C2M **TxFFE/CTLE/DFE** path is exercised end-to-end; the other features are
  implemented and unit-tested but not covered by a bundled end-to-end config. See
  [`docs/MISSING_FEATURES_PLAN.md`](docs/MISSING_FEATURES_PLAN.md) §D.
- **`FFE_OPT_METHOD='WIENER-HOPF'`** is intentionally non-functional — its helper is
  undefined in the MATLAB reference itself. Use `'MMSE'`.
- **`FAST_NOISE_CONV`** is a speed *approximation*; the default exact path is recommended for
  reported results.
- GUI file pickers are not ported — file lists are always passed on the command line.

The conversion audit (146 EQUIVALENT / 11 DIVERGENT functions, findings D1–D20) is written up
in [`docs/AUDIT_FINDINGS.md`](docs/AUDIT_FINDINGS.md), with the ledger in `dev/state/`.

## 8. MATLAB version support

The port emulates **`4p15p0` by default**, and that default is deliberate: the 208-case
reference workbooks were produced by 4p15p0, so it is the version the correlation result
above is evidence for. `4p16p0` behaviour is opt-in.

```powershell
python com.py <config.xlsx> <thru.s4p> --matlab-version 4p16p0     # per run
python tools/matlab_compare.py --run --jobs 5 --matlab-version 4p16p0   # whole corpus
```

`com.COM_MATLAB_VERSION = '4p16p0'` does the same from Python, and a `COM Version` keyword
in the config wins over both.

**4p16p0 is a small delta**: 146 of 152 function bodies are unchanged, 6 changed, 3 added,
0 removed. The three additions are `OptFom_Adaptive_Local_Search`, `compute_hard_cap` and
`append_csv_row` — Hansel D'silva's adaptive local search, **now adopted into the released
mainline** rather than living in a branch.

Every change has been measured; full detail in
[`docs/MATLAB_4p16p0_CHANGES.md`](docs/MATLAB_4p16p0_CHANGES.md) and
[`docs/MATLAB_4p16p0_IMPACT.md`](docs/MATLAB_4p16p0_IMPACT.md).

| change | measured effect |
|---|---|
| pulse/step now scaled by channel amplitude `A` | `peak_uneq_pulse_mV`, `steady_state_voltage_mV` × A on all 208 cases. **COM, FOM, VEO, VEC, itick, ERL untouched** |
| `Clip Method` default `Fast` → `Slow` | **COM +0.007 dB, FOM +0.22 dB** — but only for configs that omit the keyword. All 208 reference configs set it. |
| `min_radius` 1 → 2 in adaptive search | bit-identical answer, **4.3× the candidate evaluations, 2.5× the runtime** |
| four new step responses | additive fields |
| common-mode / TDR degenerate guards | never fired on any input tested |
| `OptFom_Create_Output`, `get_PSDs` edits | numerically neutral |

Reproduce the comparison with:

```powershell
python tools/matlab_version_diff.py matlab/com_ieee8023_4p15p0.m matlab/com_ieee8023_4p16p0.m
python tools/compare_matlab_versions.py       # after running both sweeps
```

## 9. r4p15p0 deltas + adaptive local search

On top of the `4p14p0` base, incorporated from `4p15p0` and Hansel D'silva's
adaptive-local-search branch:

- **Automatic port-order detection** (`auto_port_order`). When the config's `Port Order` is
  empty, `0`, contains `NaN`, or is not a single 4-element vector (e.g. a per-package
  `[1 2 3 4; 1 3 2 4]` matrix), the differential port order is auto-detected from the
  S-parameters. Validated on the 802.3dj KR channels (resolves the standard `[1 3 2 4]`).
- **Adaptive local search** (`OptFom_Adaptive_Local_Search`), selected by the config keyword
  **`Non-zero Local Search Method`** (`param.NonZeroLSMethod`). It keeps a persistent search
  radius that grows and shrinks with recent FOM improvement, and prunes candidates by a
  weighted L1/L2 tap-space distance from the current best. It carries ~10 hand-tuned
  constants (shrink factors, weights, the L2/L1 ratio, the CTLE window) whose provenance is
  not documented in the source.
- **Minor r4p15p0 fixes**: `get_PSDs` crosstalk pad length,
  `adjust_Rx_noise_for_quantization` cursor-tap off-by-one, port-order threading through the
  Touchstone readers.
- **Two pre-existing gaps** fixed while exercising the 802.3dj config: named/complex packages
  (`param.PKG`) accessed by dict subscript instead of attribute, and a 2-row `Port Order`
  matrix now routing to auto-detection.

## 10. Input files

**Configuration spreadsheet (`.xlsx`)** — IEEE 802.3 COM spreadsheet format, active sheet
`COM_Settings` (parameter / value columns). Key parameters: signalling rate (`f_b`),
modulation (`levels`), DFE taps (`N_b`), BER target (`specBER`), CTLE sweep, package length
cases, output flags (`SAVE_FIGURES`, `CSV_REPORT`). Start from an IEEE 802.3 working-group
reference spreadsheet.

**S-parameter files (`.s4p`)** — 4-port Touchstone. Default differential port order
`[1, 3, 2, 4]`; override with `snpPortsOrder` in the config. Aggressor files use the same
format (differential `Sdd21` becomes the coupling response).

## 11. Engineering `.mat` export (optional)

`--export-mat` writes a per-case MATLAB v5 snapshot alongside the standard outputs. It is an
**additive debug export** — it changes no COM result, report, or figure.

It captures the frequency-domain equalizer chain (`H_channel`, `H_ctle`, `H_ch_ctle`,
`H_ffe`, `H_tx`, `H_final`, crosstalk `H_next`/`H_fext`), per-stage impulse and pulse
responses, the statistical eye / BER contour, combined and component noise PDF/CDF, all COM
metrics (`results_full`), a self-describing `config` struct, and run `meta`. It also adds
`FOM_gauss_dB` — the COM you would get if the combined interference+noise distribution were
purely Gaussian; the gap to `COM_dB` quantifies the non-Gaussian tail penalty.

`H_channel`/`H_ctle`/`H_ch_ctle` are genuine captured FD responses; `H_ffe` is evaluated from
the selected Tx-FFE taps, and `H_tx`/`H_final` are their products (DFE is a time-domain term,
excluded from the FD chain) — see `meta.notes` in each file.

```powershell
Rscript R/com_analysis.R results/<...>/<config-name>_case01.mat
# -> writes <config-name>_case01_report.html
```
