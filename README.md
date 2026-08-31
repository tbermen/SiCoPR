# SiCoPR

*(pronounced si-copper)* — **Si** (Signal Integrity), **Co** (COM), **P** (Python), and **R**

A Python port of the IEEE 802.3ck/dj **COM** (Channel Operating Margin) MATLAB reference
tool with R programming extensions for reports and visualizations. It computes COM / VEO /
VEC and the supporting equalization and noise analysis for a serial channel described by
Touchstone S-parameter files and an Excel configuration spreadsheet.

The port is function-for-function: `sicopr.py` follows the structure of the MATLAB source
closely enough to navigate by MATLAB line number, takes the same `.xlsx` + `.s4p` inputs,
and produces the same outputs.

**Two MATLAB releases are supported.** `sicopr.py` emulates **`com_ieee8023_4p15p0`** by
default — that is the release the 208-case reference corpus was produced with, so it is the
version the correlation result in §7 is evidence for — and **`com_ieee8023_4p16p0`** via
`--matlab-version 4p16p0` (or `sicopr.COM_MATLAB_VERSION`, or a `COM Version` config keyword).
4p16p0 is a small delta, and its measured effect on all 208 cases is in §8. Which release a
given `sicopr.py` emulates is recorded in its header and in `VERSION.json`.

**The R half of the name.** `R/` turns a run into something you can read: an
interactive HTML dashboard for a single case — channel response, CTLE, equalization
contributions, pulse, eye — and the correlation and study reports that produce the
figures in §5 and §7. The engine writes the data; R renders it. Neither needs the
other to run, and the Python side has no R dependency.

On top of both there is a study layer (`tools/`, `R/`) built to answer one question:
**does pruning the equalizer search grid change COM?** Results in §5.

> **New to this project?** Start with **[`SiCoPR_Tutorial.docx`](SiCoPR_Tutorial.docx)**
> — a 40-page tutorial and reference covering installation, architecture, every feature,
> the study layer, the R reports, a COM concepts primer, and a complete index of all 249
> configuration keywords. This README is the quick version.

---

## Licence and status

**BSD-3-Clause** — [`LICENSE`](LICENSE). The MATLAB reference this is ported from is
BSD-3-Clause (*Copyright 2025 802-COM Authors*); this port is a derivative work released
under the same terms, so the licence travels with it. The reference sources are
redistributed unmodified under `matlab/`, notices intact.

> **Not an IEEE product.** This project is **not** endorsed by, affiliated with, or
> approved by IEEE, the IEEE 802.3 working group, or the 802-COM Authors. It is an
> independent port of the published MATLAB reference. Please describe it that way —
> clause 3 of the licence requires it. "IEEE 802.3" appears here only to identify the
> standard the reference implements.

**Correctness is a measured claim, not a promise.** Against the 208-case reference set the
port reproduces MATLAB bit-for-bit on COM, FOM and sampling phase, with the caveats and the
things *not* covered stated plainly in §7. Read §7 before you rely on it for anything.

**Contributions are welcome** — the process is short and is in
[`CONTRIBUTING.md`](CONTRIBUTING.md). In brief: fork, branch, open a pull request; nobody
pushes to `master` directly, including the maintainer.

---

## 1. What ships in this repository — and what doesn't

**The port ships. The correlation data does not.** The engine, its tests, the tooling
and the MATLAB reference sources are all here. The channel S-parameters and
configuration workbooks used to correlate against MATLAB are IEEE 802.3 contributions
and are not ours to redistribute — but every one of them is publicly available, and
this section says exactly which.

| Present | Not present |
|---|---|
| the engine (`sicopr.py` + `com_functions/`), tests, tooling, R reports | channel S-parameters (`tests/0_...`) |
| `matlab/` — the BSD-3-Clause MATLAB reference sources | COM configuration workbooks (`tests/1_...`) |
| `docs/`, `VERSION.json`, `LICENSE`, `CONTRIBUTING.md` | MATLAB reference result workbooks (`tests/2_...`) |
| the study layer (`tools/sweep_compare.py`, `corpus_sweep.py`, `fom_com_probe.py`) and the config editor (`gui/`) | `tests/oracles/`, `report_data/` — MATLAB reference **values**, distilled from those workbooks |
| `matlab_version_diff.py`, `plot_sweep_compare.py`, `audit_search_space.py` | the correlation harness, exporters and deck builders (`tools/matlab_compare.py`, `export_compare_csv.py`, `export_results.py`, `build_*_pptx.py`) |
| the study write-ups (`RESULTS.md`, `STATE.md`) | every generated output: `results/`, `report_figs/`, `corpus_results/*.csv`, `sicopr_results/`, `report_docs/` |

**A fresh clone is fully functional without any of it.** The unit suite runs and
passes — 890 per-function tests plus the cross-check scripts — and every test that
needs correlation data skips cleanly and says what is missing.

**This repository documents the verification; it does not offer to reproduce it.**
The 208-case comparison against MATLAB was run, and what it found is written up in
§7 and in [`MATLAB_Correlation_Review.md`](MATLAB_Correlation_Review.md). Repeating
it here is not on offer, and not promised: the channel S-parameters, the
configuration workbooks and the MATLAB result workbooks are IEEE 802.3 contributions
that are not ours to redistribute, and the harness that drives the comparison is kept
with them. Anyone wanting to check the result independently would supply their own
channels and configs, run their own MATLAB, and write their own comparison against
this engine's output — which is a reasonable thing to do, and is what the numbers in
§7 are stated precisely enough to support.

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
from the COM ad hoc rather than from a numbered contribution, and are likewise not
redistributed here.

The channels above are worth having for a different reason: they are the ones §7's
numbers were measured on, so running SiCoPR against them puts you on the same inputs
the comparison used — even though the comparison itself is not something this
repository performs.

> **The correlation harness is not published.** `tools/matlab_compare.py` and
> the exporters around it are kept local with the data they drive, because every
> file they read and write is data this repository does not redistribute (§7).
> Obtaining the channel contributions above therefore does **not** let you re-run
> the 208-case comparison — it lets you run SiCoPR on the same channels.
>
> What *is* here: the engine, its tests, the study layer (§5), and the per-stage
> oracle test, which asserts against MATLAB values when you supply them.

## 2. Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
```

Nothing to build — the tool runs directly from `sicopr.py`. Python 3.9+ (developed on 3.14).

The R reports additionally need, inside R:

```r
install.packages(c("plotly", "htmltools", "jsonlite", "R.matlab"))
```

## 3. Run

```powershell
python sicopr.py <config.xlsx> <thru.s4p> [--fext f1.s4p ...] [--next n1.s4p ...]
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

### Config editor (optional)

```powershell
python gui/app.py            # http://127.0.0.1:8765
```

A local, stdlib-only web app that shows a configuration workbook as a channel
schematic — Tx die → package → channel → package → Rx die, plus the aggressors
and the Tx FFE → CTLE → Rx FFE → DFE chain — and lets you edit the settings each
block owns and save a new config.

It writes by editing the sheet XML in place, so formulas, formatting and the
`keywords_*` sheet survive, and it keeps a swept tap (`[ -0.34:.02:0]`, a
~198-point search) distinct from a fixed one (`0`).

It also runs the whole workflow without leaving the page: pick the THRU and
crosstalk channels, review the exact `sicopr.py` command it assembles, run it
with the terminal output streamed live, then refresh to see the new results —
per case, the headline numbers from `results.csv` and the figures grouped by
pipeline stage. An S-parameter tab plots the mixed-mode response of any
Touchstone file in the repo, using the engine's own reader. See
[gui/README.md](gui/README.md).

## 4. Repository layout

| Path | What it is |
|---|---|
| `sicopr.py` | **the engine** — assembled, runnable. *Do not edit by hand* |
| `com_functions/fn/<name>/py_impl.py` | per-function source (the editable code), 157 functions |
| `com_functions/fn/<name>/test_verify.py` | per-function unit tests |
| `assemble_sicopr.py` | concatenates the `py_impl.py` files into `sicopr.py` |
| `com_plots.py`, `com_mat_export.py` | figure generation and `.mat` export — imported *by* `sicopr.py`, so they live beside it |
| `tools/` | study layer (§5) plus the MATLAB-comparison harness, the version differ, and the oracle extractor (§6) |
| `report_figs/case_demo/` | example of a single run's figure set: `s<stage>_*.png` plus `STAGE_INDEX.md` |
| `gui/` | the config editor — a local web app for building configs from a schematic view (§3) |
| `R/` | the R extensions — per-case interactive HTML dashboard, plus the correlation and study reports |
| `VERSION.json` | which MATLAB release the port emulates; `assemble_sicopr.py` generates `sicopr.py`'s header from it |
| `matlab/` | MATLAB reference sources (`4p14p0`, `4p15p0`, `4p16p0`, adaptive-local-search branch) |
| `docs/` | audit findings, fix summary, feature plan, 4p16p0 change analysis + measured impact |
| `dev/` | audit and interface-check scripts, plus state ledgers. The development prompts under `dev/prompts/` are kept locally and not published |
| `tests/` | standalone cross-check scripts (run directly, not via pytest — see `CONTRIBUTING.md`) |
| `sweep_results/`, `corpus_results/` | the write-ups of the two EQ-search studies (`STATE.md`, `RESULTS.md`). The data files they describe are generated locally, not tracked |
| `report_docs/` | built decks and their PDF snapshots. Generated by `tools/build_*_pptx.py`; neither the decks nor their builders are tracked (§1) |

**Per-run stage artifacts.** The interactive R dashboard (`build_dashboard()` in `R/com_analysis.R`) is grouped by pipeline stage, with a
heading and plots for each of the seven — including TDR impedance and ERL, FOM
against sampling phase, the selected equalizer taps, and the individual noise
terms. A plot whose data is absent from the `.mat` renders a visible placeholder
rather than disappearing.

With `SAVE_FIGURES` enabled in the config, each package
test case also writes a figure set named by pipeline stage — `s1_insertion_loss.png`,
`s2_tdr_impedance.png`, `s4_fom_vs_phase.png`, and so on — alongside `STAGE_INDEX.md`,
which maps every stage to its figures and the result columns it owns. A stage with no
figure is listed as **no figure** rather than omitted, so a gap is visible instead of
silent. `tests/test_stage_figures.py` fails if any of the seven stages stops emitting one.

**What is deliberately absent.** The repository carries code, tests and guides — no inputs and no outputs. Channel S-parameters and configuration workbooks are IEEE contributions (§1); result tables, figures, decks, the per-stage MATLAB oracles under `tests/oracles/` and the comparison tables under `report_data/` are all either generated locally or distillations of the MATLAB reference workbooks, which are not ours to redistribute. Every test that needs one of those skips and says so, so a fresh clone runs green: 890 per-function tests plus the cross-check scripts.

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

### Running sweep commands for sicopr.py

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
python tools/build_pptx.py                       # deck -> report_docs/ (local tool, not published)
```

A full 7-channel × 3-method corpus is ~7.5 h. Use `--dry-run` for the estimate,
`--methods full_grid,adaptive` to drop ~40%, and `--probe-top-k 0` to skip the proxy
measurement. Runs are checkpointed per channel, so an interrupted corpus resumes.

**Instrumentation.** `optimize_fom` carries an opt-in per-candidate logger — off by default,
no effect on any COM result. Set `sicopr.SWEEP_LOG_CSV` and `sicopr.SWEEP_METHOD_LABEL` for one row
per TX-FFE candidate (method, EQ indices, tap vector, candidate/best FOM, winning sample
phase, evaluated vs pruned). The adaptive method's per-iteration radius diagnostic is
captured separately via `sicopr.ALS_LOG_CSV`. These logs are what the study measures.

`tools/hist_dep_repro.py` is a diagnostic harness: it solves the same operating point with
different numbers of candidates evaluated and diffs the full engine state at the moment COM
computation begins. Reach for it if COM ever appears to depend on search history.

## 6. Development workflow

`sicopr.py` is generated. To change behaviour, edit the per-function source and re-assemble:

```powershell
python -m pytest com_functions/fn/<name>/test_verify.py -q   # test the change
python assemble_sicopr.py                                        # regenerate sicopr.py
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
| `test_smoke.py`, `test_checkpoints.py`, `test_end_to_end.py`, `test_export_columns.py` | pytest modules |
| every other `test_*.py` | standalone scripts — `python tests/test_x.py` |

The configuration editor under `gui/` is **not** generated and does not go
through the assembler — edit it directly, and see
[`CONTRIBUTING.md`](CONTRIBUTING.md) for what is different about it.

Do **not** run `pytest tests` over the whole directory. The audit scripts call
`sys.exit()` at import, which aborts collection: pytest reports `no tests ran`
**and still exits 0**, so nothing runs and nothing complains. `run_all.ps1`
dispatches each kind correctly.

Those scripts record two outcomes. `check()` is behaviour that must match MATLAB;
`xcheck()` is a reviewed, accepted divergence, which reports `XFAIL` while it
persists and **fails the run if it starts passing** — so a divergence that gets
fixed cannot leave a stale entry behind in the ledger.

Current state: **890** per-function tests, and **656 checks across 31 audit scripts**
with 25 accepted divergences.

### Cross-cutting guards

Most of those scripts exist because the per-function tests structurally cannot catch the
defect classes that actually got through. Each was built from a real failure and verified
by re-introducing it — a check that has never failed for the right reason is not evidence:

| script | guards against | why |
|---|---|---|
| `test_reference_leaks.py` | writing to a parameter the function never returns | MATLAB passes structs **by value**, Python by reference. **Five of the original eight** correlation defects were this class, and six of the fifteen engine fixes overall. Caught a new instance during the 4p16p0 port. |
| `test_inlined_copies.py` | an inlined copy drifting from its canonical function | there are **178 copies of 70 functions**; a fix to `py_impl.py` reaches only one of them. Engine defect #6 lived in three copies. |
| `test_optimization_invariants.py` | the speed work silently breaking | cache transparency and key completeness, the hoisted Gram matrix, FFT/direct convolution agreement, shared buffers. Found a live cache-aliasing defect. |
| `test_matlab_stage_oracles.py` | drift from real MATLAB values | pins **208 cases × 35 scalars + 14 vector families** taken from the reference workbooks — the only tests in the repo that assert against MATLAB rather than against Python. The oracle file itself is not tracked (it *is* reference data); the test skips without it. |
| `test_abort_path_leaks.py` | writing into a caller's struct before an early return | the sibling of the leak above that the leak guard cannot see: the function *does* return the struct, but commits values on a path MATLAB never commits on. Ledger #10b. |
| `test_sort_stability.py` | `np.argsort` reordering ties | MATLAB's `sort` is stable, NumPy's default is not. Ties were measured in **81% of argsort calls** rather than assumed rare. |
| `test_integer_ratio_rounding.py` | banker's rounding on a ratio of integers | MATLAB rounds half away from zero. The audit dismissed this as measure-zero, which is true for continuous data and false for `a/b` with both integral — ledger #16. |
| `test_structural_invariants.py` | properties no single function owns | shapes, index bases and struct field sets that only go wrong between functions. |

The configuration editor under `gui/` has its own three:

| script | guards against |
|---|---|
| `test_config_roundtrip.py` | the config writer damaging a workbook — reads, rewrites, and requires the **engine** to parse both to identical `param`/`OP` |
| `test_gui_server.py` | the HTTP layer: payload shape, path containment, process control, and the results/dashboard views |
| `test_gui_static.py` | `app.js` failing to parse at all — a syntax error there breaks the whole page while every server-side test still passes |

### Tooling for a new MATLAB release

```powershell
python tools/matlab_version_diff.py OLD.m NEW.m   # -> which py_impl files to re-check
python tools/matlab_version_diff.py --self-check  # validates the differ itself
python tools/bench_com.py --against <ref>         # speed + fingerprint (local tool, not published)
python tools/compare_matlab_versions.py           # diff two version sweeps (local tool, not published)
```

## 7. Verification status & caveats

Every major feature is implemented and unit-tested against `matlab/com_ieee8023_4p15p0.m`
plus the adaptive-local-search branch: TxFFE/CTLE/DFE, RxFFE (MMSE), floating DFE / floating
RxFFE taps, MLSE, crosstalk (FEXT/NEXT, ICN), common-mode modal masks, RX calibration, FD
processing (ICN/ILD), ERL/TDR, and TD-ILN/RILN. The `com_functions/fn` suite is green
(**889 passed, 0 failed**).

**Numeric parity with MATLAB is established end to end.** 208 reference cases from Hansel
D'silva's `com_ieee8023_4p15p0` runs, compared case by case:

| | |
|---|---|
| FOM bit-exact | **208 / 208** |
| COM bit-exact | **208 / 208** |
| sampling phase (`itick`) exact | **208 / 208** |
| pass/fail disagreements at 3 dB | **0** |
| max \|ΔCOM\| | **3.29e-14 dB** |
| max \|ΔFOM\| | **3.38e-11 dB** |

That is double-precision arithmetic noise, not agreement to a tolerance: the two
implementations compute the same number. Of 43,509 numeric column-values compared, 67 sit
outside 1e-6 relative and every one is `DER_DFE` or `DER_MLSE` — CDF bin lookups landing on
an exact tie, quantisation-limited rather than wrong.

Each crosstalk condition runs on the configuration its own MATLAB reference was produced
with: the without-crosstalk cases on the base workbooks, the with-crosstalk cases on the
workbooks that sweep the Tx FFE. **That pairing was confirmed by the COM maintainer on
2026-08-24**, so it is the configuration, not one reading among several. Produced with
(local tooling, not part of this repository — §1):

```powershell
# local tools, not published with the repo -- see section 1
python tools/matlab_compare.py --validate --run --modal-erl --jobs 5
python tools/export_compare_csv.py
```

Getting there took **seventeen** engine-level and settings findings, each with what it
bought recorded in [`docs/FIX_SUMMARY.md`](docs/FIX_SUMMARY.md); the full write-up is
[`MATLAB_Correlation_Review.md`](MATLAB_Correlation_Review.md). Two are worth naming here
because they were configuration rather than code, and both were the same failure — a
supplied config snapshot that post-dated the run it came from:

- **The Tx FFE grid.** The supplied workbooks pin `c(-1)`, `c(-2)` and `c(1)` to a single
  zero; the run that produced the reference results swept 1584 candidates. Confirmed by the
  maintainer, who supplied the sweep workbooks.
  ([`docs/TXFFE_SWEEP_ROOT_CAUSE.md`](docs/TXFFE_SWEEP_ROOT_CAUSE.md))
- **The adaptive-search radius floor.** The branch source forces `1`; the reference behaves
  as the 4p16p0 mainline rule (`1` for a single Tx FFE candidate, `2` otherwise). The port
  applies that rule on both version paths, so no switch is needed.
  ([`docs/MIN_RADIUS_ASSUMPTION.md`](docs/MIN_RADIUS_ASSUMPTION.md))

The result is against **4p15p0**, which is why it stays the default emulation target — see
§8. The same corpus has been run in 4p16p0 mode: 210 of 213 output columns are identical on
all 208 cases, and no COM/FOM/VEO/VEC/itick/ERL value moves.

Honest caveats for anyone relying on the numbers:

- **No MATLAB-vs-Python runtime comparison is offered.** The timings that exist were taken
  on different machines and, at the time, on different search spaces. The Python-vs-Python
  speed-up (5.4× on identical work, every output field bit-identical) is unaffected by that
  and is the only speed claim made.
- **Results produced before August 2026 are not comparable to current output.** The engine
  fixes changed COM materially — the largest removed a systematic FOM bias affecting 95.7%
  of cases. Regenerate rather than comparing against archived numbers.
- Only the **TxFFE/CTLE/DFE** path is exercised end to end by a real configuration; the
  other features are implemented and unit-tested but not covered by an end-to-end run. See
  [`docs/MISSING_FEATURES_PLAN.md`](docs/MISSING_FEATURES_PLAN.md) §D.
- **`FFE_OPT_METHOD='WIENER-HOPF'`** is intentionally non-functional — its helper is
  undefined in the MATLAB reference itself. Use `'MMSE'`.
- **`FAST_NOISE_CONV`** is a speed *approximation*; the default exact path is recommended
  for reported results.
- GUI file pickers are not ported — file lists are always passed on the command line.

The conversion audit (146 EQUIVALENT / 11 DIVERGENT functions, findings D1-D20) is written
up in [`docs/AUDIT_FINDINGS.md`](docs/AUDIT_FINDINGS.md).


## 8. MATLAB version support

The port emulates **`4p15p0` by default**, and that default is deliberate: the 208-case
reference workbooks were produced by 4p15p0, so it is the version the correlation result
above is evidence for. `4p16p0` behaviour is opt-in.

```powershell
python sicopr.py <config.xlsx> <thru.s4p> --matlab-version 4p16p0     # per run
python tools/matlab_compare.py --run --jobs 5 --matlab-version 4p16p0   # local tool, not published
```

`sicopr.COM_MATLAB_VERSION = '4p16p0'` does the same from Python, and a `COM Version` keyword
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

Produced with (`matlab_version_diff.py` ships; `compare_matlab_versions.py` is local
tooling — §1):

```powershell
python tools/matlab_version_diff.py matlab/com_ieee8023_4p15p0.m matlab/com_ieee8023_4p16p0.m
python tools/compare_matlab_versions.py       # local tool, not published
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
