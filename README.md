# COM — Channel Operating Margin (Python)

A Python port of the IEEE 802.3ck/dj **COM** (Channel Operating Margin) MATLAB reference
tool. It computes COM / VEO / VEC and the supporting equalization and noise analysis for a
serial channel described by Touchstone S-parameter files and an Excel configuration
spreadsheet.

The port is function-for-function: `com.py` follows the structure of
`matlab/com_ieee8023_4p15p0.m` closely enough to navigate by MATLAB line number, takes the
same `.xlsx` + `.s4p` inputs, and produces the same outputs.

On top of the engine there is a study layer (`tools/`, `R/`) built to answer one question:
**does pruning the equalizer search grid change COM?** Results in §5.

> **New to this project?** Start with **[`COM_Python_Tutorial.docx`](COM_Python_Tutorial.docx)**
> — a 40-page tutorial and reference covering installation, architecture, every feature,
> the study layer, the R reports, a COM concepts primer, and a complete index of all 247
> configuration keywords. This README is the quick version.

---

## 1. What ships in this repository — and what doesn't

**Code, docs and the IEEE public-area channels ship. Vendor and bulk data do not.**

| Present | Not present (excluded in `.gitignore`) |
|---|---|
| the engine, tests, tooling, R reports | vendor-marked `.s4p` channels + config sheets |
| **96 `.s4p` files** under `tests/0_IEEE_802p3dj_PublicArea_CR_KR_Channels/` | `akinwale_3dj_01_2310/` (210 MB, public IEEE 802.3dj data) |
| reference COM spreadsheets + MATLAB result workbooks (`tests/1_`, `tests/2_`) | `results/` (1.1 GB of generated figures/exports) |
| `docs/`, `dev/`, `matlab/` references, study outputs | `matlab_compare_results/` (regenerable per-case JSON) |

**A fresh clone can run the bundled IEEE channels immediately.** Two things need data you
supply: the canonical example in §3 uses a vendor-marked channel, and the corpus/correlation
commands in §5 assume `akinwale_3dj_01_2310/` is restored from the IEEE 802.3dj contribution
`akinwale_3dj_elec_01_2310`.

To ship the channel data with the repo instead, delete the `akinwale_3dj_01_2310/` line from
`.gitignore`. Review the "vendor / third-party inputs" section of that file before removing
anything there.

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
python com.py <config.xlsx> <thru.s4p> [--fext f1.s4p ...] [--next n1.s4p ...] [--export-mat]
```

The **THRU** (victim) channel is required. Crosstalk aggressors are optional: `--fext` for
FEXT, `--next` for NEXT, each accepting multiple files.

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
| `tools/` | the study layer (§5): sweep, probe, corpus, deck builder |
| `R/` | interactive HTML reports |
| `matlab/` | MATLAB reference sources (`4p14p0`, `4p15p0`, adaptive-local-search branch) |
| `docs/` | audit findings, fix summary, feature plan |
| `dev/` | historical development prompts, state ledgers, one-shot scripts — kept for provenance, not needed to run anything |
| `tests/` | standalone cross-check scripts (run directly, not via pytest) |
| `sweep_results/`, `corpus_results/` | study outputs, each with a `RESULTS.md` / `STATE.md` |

## 5. The EQ-search study

Three ways to solve the same channel, differing only in how much of the TX-FFE / CTLE grid
they evaluate:

| Method | Switches |
|---|---|
| **full grid** | `LOCAL_SEARCH = 0` — exhaustive, the reference answer |
| **legacy local search** | `LOCAL_SEARCH = N`, `NonZeroLSMethod = 0` — fixed-radius prune |
| **adaptive local search** | `LOCAL_SEARCH = N`, `NonZeroLSMethod = 1` — Hansel D'silva's branch |

### Result (7 channels, 100–1400 mm, 2026-08-07)

**Adaptive returns COM bit-identical to the exhaustive grid on all seven channels** — ΔCOM
exactly 0 to 17 significant digits, same winning EQ operating point — while evaluating ~7%
of the candidates at a median **12.9× speedup** (range 11.75–15.10×). Legacy also matched
exactly, at 1.27×. No pass/fail flips at the 3 dB threshold.

The winning EQ setting *moves* with channel length, so the agreement is not an artifact of a
constant answer: adaptive tracks a genuinely shifting optimum while skipping 93% of the work.

A second measurement is arguably more interesting. Recomputing true COM for the top-20
candidates by FOM shows **FOM is not a faithful proxy for COM**: Spearman ρ runs 0.60–0.81,
and on 2 of 7 channels the FOM winner is not the COM winner, costing up to 0.036 dB. That
cost belongs to FOM-driven search itself — the exhaustive full grid pays it too. Adaptive
adds exactly zero on top.

Full numbers, caveats, and regeneration commands: [`corpus_results/RESULTS.md`](corpus_results/RESULTS.md).

**The corpus is one channel family** (same topology, varying only cable length), one config,
thru-only. That is its main limitation and is documented alongside the results.

### Running it

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
python tools/build_pptx.py                       # review deck, built from corpus_results/
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

## 7. Verification status & caveats

Every major feature is implemented and unit-tested against `matlab/com_ieee8023_4p15p0.m`
plus the adaptive-local-search branch: TxFFE/CTLE/DFE, RxFFE (MMSE), floating DFE / floating RxFFE
taps, MLSE, crosstalk (FEXT/NEXT, ICN), common-mode modal masks, RX calibration, FD
processing (ICN/ILD), ERL/TDR, and TD-ILN/RILN. The `com_functions/fn` suite is green
(**876 passed, 0 failed**), and the bundled 802.3ck C2M channel runs end-to-end.

**Numeric parity with MATLAB has been established end to end.** 208 reference cases from
Hansel D'silva's `com_ieee8023_4p15p0` runs were compared case by case:

| | result |
|---|---|
| FOM bit-exact | 198 / 208 |
| COM bit-exact | 135 / 208 |
| sampling phase (`itick`) exact | 200 / 208 |
| max \|ΔCOM\| | 0.176 dB |
| rms ΔCOM | 0.019 dB |
| pass/fail disagreements | 2 (both within 0.02 dB of the 3 dB threshold) |

Eight engine defects were found and fixed in the process. Reproduce with
`python tools/matlab_compare.py --validate --run --jobs 5`; the full write-up is
[`MATLAB_Correlation_Review.md`](MATLAB_Correlation_Review.md).

Honest caveats for anyone relying on the numbers:

- **Eight sampling-phase divergences remain unexplained.** On those cases Python cannot
  reach MATLAB's FOM at MATLAB's tick under any equalizer setting, yet the peak values
  agree — consistent with an anchor-origin offset. Open question with Hansel.
- **A residual ~0.002 dB mean bias remains in the COM PDF path** (max 0.035 dB). It is
  what produces the 2 knife-edge pass/fail disagreements.
- **Results produced before August 2026 are not comparable to current output.** The eight
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

The conversion audit (147 EQUIVALENT / 11 DIVERGENT functions, findings D1–D20) is written up
in [`docs/AUDIT_FINDINGS.md`](docs/AUDIT_FINDINGS.md), with the ledger in `dev/state/`.

## 8. r4p15p0 deltas + adaptive local search

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

## 9. Input files

**Configuration spreadsheet (`.xlsx`)** — IEEE 802.3 COM spreadsheet format, active sheet
`COM_Settings` (parameter / value columns). Key parameters: signalling rate (`f_b`),
modulation (`levels`), DFE taps (`N_b`), BER target (`specBER`), CTLE sweep, package length
cases, output flags (`SAVE_FIGURES`, `CSV_REPORT`). Start from an IEEE 802.3 working-group
reference spreadsheet.

**S-parameter files (`.s4p`)** — 4-port Touchstone. Default differential port order
`[1, 3, 2, 4]`; override with `snpPortsOrder` in the config. Aggressor files use the same
format (differential `Sdd21` becomes the coupling response).

## 10. Engineering `.mat` export (optional)

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
