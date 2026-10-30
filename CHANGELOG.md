# Changelog

Notable changes to SiCoPR. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Versions follow [Semantic Versioning](https://semver.org/). **1.0.0 is the first
release**; there is no earlier version history, and this file starts there.

Two things this file is *not*:

- **It is not the engineering ledger.** Anything that changes a COM, FOM or
  sampling-phase number is recorded in [`docs/FIX_SUMMARY.md`](docs/FIX_SUMMARY.md)
  with its evidence, and that is the record to read when a number moves.
  `CONTRIBUTING.md` requires an entry there; this file summarises for people who
  are not reading diffs.
- **It is not a statement about the MATLAB reference.** Which upstream release
  the engine emulates is in [`VERSION.json`](VERSION.json), and changing it is a
  change to what the engine *is*, not a routine entry here.

## [1.2.1] - 2026-10-29

Packaging only. **The engine is unchanged**, and so is every file a checkout
runs: COM, FOM and sampling-phase results are the same as 1.2.0's.

### Changed

- **The wheel installs the engine and nothing else**: the modules `sicopr`,
  `com_mat_export` and `com_plots`. A 1.2.0 wheel also installed `gui` and
  `com_functions` as top-level packages in site-packages, where names that
  generic can collide with other software's; the engine imports neither.
  `pip install sicopr` then `python -m sicopr config.xlsx thru.s4p` runs COM.
- **The sdist carries the source tree a checkout has** (`MANIFEST.in`):
  `com_functions/`, `gui/`, `tests/`, `tools/`, `docs/`, `examples/`,
  `matlab/`, `octave/`, `R/`, `benchmark/` and the top-level files. The GUI, the
  worked example, the tests, the Octave route and
  [SiCoPR-Assist](https://github.com/tbermen/SiCoPR-Assist) still need a
  checkout (or the unpacked sdist), as before.
- Package metadata: the licence is the SPDX expression `BSD-3-Clause` with
  `LICENSE` and `NOTICE` as licence files, and the project links add Issues and
  Changelog. The build needs setuptools 77 or newer.
- README, *Install*: says what `pip install sicopr` gives and what needs a
  checkout.

## [1.2.0] - 2026-10-29

GUI panels, and three hardenings of the GUI. Everything here is in `gui/`, the
config editor, a supporting tool: **the engine is unchanged**, and COM, FOM and
sampling-phase results are the same as 1.1.4's (the 208 MATLAB stage-oracle cases
pass unchanged).

### Added

- **GUI panels.** A panel is a side panel that another installed Python package
  adds to the config editor (`python gui/app.py`). It shows on the right of the
  page and works alongside the five tabs: it can load a workbook in Config, see
  edits, assemble the Run tab (channel files and options), take over its
  **Run sicopr.py**, put its own run's output in the Run tab's Output, and open
  a dashboard in Results, dynamic.
  - **For users:** install [SiCoPR-Assist](https://github.com/tbermen/SiCoPR-Assist)
    beside SiCoPR to get its panel; start the editor as usual and the panel is
    there. With no panel installed the editor is exactly as before, and
    `--no-panels` leaves installed panels out.
  - **For package authors:** register an object in the Python entry-point group
    `sicopr.gui_panels`. The editor serves its `panel.js` and `panel.css`, routes
    `/panel/<name>/api/...` to its `handle()` with a narrow view of the server
    (the opened directories, the run directory, whether a run is live), and lists
    it at `/api/panels` (panel API 1). Page and panel talk through named DOM
    events: `sicopr:config`, `sicopr:edit`, `sicopr:view`, `sicopr:run`,
    `sicopr:channels` and `sicopr:browsed` tell the panel what the page did;
    `sicopr:open-config`, `sicopr:overlay`, `sicopr:clear-edits`, `sicopr:browse`,
    `sicopr:set-run`, `sicopr:output` and `sicopr:show` let it act on the page;
    `sicopr:run-request` and `sicopr:stop-request` hand **Run sicopr.py** and
    **Stop** to the panel that assembled the run. The contract is in
    [`gui/README.md`](gui/README.md), *Panels*.
  - **The request guard** of 1.1.2 (`Handler._refused`) covers the panel routes:
    a cross-site, foreign-Origin, DNS-rebinding or text/plain request to
    `/panel/<name>/...` is refused before the panel's `handle()` is reached.
  - `tests/test_gui_server.py` and `tests/test_gui_static.py` pin the contract
    and the no-panel case.

### Security

From a security review of the GUI on 2026-10-29. Each has a check in
`tests/test_gui_server.py` that fails without it.

- **Rscript is never taken from the current folder.** On Windows, Python's
  `shutil.which` looks in the current folder before `PATH`, so an `Rscript.exe`
  placed in the folder the editor was started from would run, with your rights,
  at the first dashboard build. The editor now looks under `%ProgramFiles%\R`
  first on Windows, then on `PATH`, skipping the current folder.
- **Dashboards are sandboxed.** A `*_report.html` is served with
  `Content-Security-Policy: sandbox allow-scripts`, so a dashboard from a run
  directory someone else wrote cannot call the editor's API (open directories,
  save workbooks, start runs) from its scripts. The plots still draw.
- **Every response** carries `X-Content-Type-Options: nosniff` and
  `Content-Security-Policy: frame-ancestors 'self'`, so no other site can frame
  the editor.

### Changed

- The request guard lets one kind of cross-site request through: a dashboard's
  own `lib/` script or stylesheet (a GET under `/rpt/` that is not a page, loaded
  as a script or style), which the browser marks cross-site because the sandboxed
  page has no origin. Without it the dashboards would load no Plotly.
- On Windows, an R install under `%ProgramFiles%\R` is now preferred over an
  `Rscript` found on `PATH`.

## [1.1.4] - 2026-10-29

A security fix, and the config-reader fixes that carry it. **Upgrade if you open
configuration workbooks or run CSVs that someone else wrote.** COM results do not
change for configs written with plain numbers and brackets: the 208 MATLAB
stage-oracle cases and the 28 COM Octave checkpoint cases pass unchanged.

### Security

- **Config cell text could run as Python (1.1.3 and earlier).** For the keywords
  the Reference Code evaluates (for example `kappa1`, the Tx FFE `c(n)` cells,
  and package-block parameters), SiCoPR handed the cell's text to Python's
  unrestricted `eval`. A crafted `.xlsx` workbook or `.csv` run CSV therefore ran
  arbitrary Python as soon as it was loaded, before any COM computation, with the
  permissions of the user running SiCoPR. 1.1.4 reads those cells with its own
  evaluator for the MATLAB subset configs use (below) and refuses anything else
  with an error naming the keyword; Python's `eval` is no longer applied to
  config text. If you open configs from other people, upgrade. No CVE has been
  requested or is claimed.

### Fixed

Evidence: `docs/FIX_SUMMARY.md`, rows 58-60.

- **MATLAB vector expressions in a value cell.** A cell such as
  `[0.3 0.2*ones(1,4) 0.1*ones(1,6)]` (802.3ck's `b_max(2..N_b)`) stopped SiCoPR
  with `could not convert string to float`, because the config reader read only
  numbers and brackets of numbers, where the Reference Code `eval`s the cell. It
  now reads the MATLAB that configs use (numbers, `Inf`, `NaN`, `+ - * / .* ./`,
  parentheses, colon ranges, brackets, `ones` and `zeros`) with MATLAB's
  semantics, in `.xlsx`, `.csv` and `.mat` configs alike, and gives the values
  COM Octave gives. Anything else stops with an error that names the keyword, as
  the Reference Code stops on it; Python's `eval` is no longer used on config
  text.
- **`b_min(2..N_b)` default.** With a vector `b_max(2..N_b)` and no
  `b_min(2..N_b)`, every negative limit was `-b_max(2)`; the reference uses
  `-b_max` tap by tap, and so does SiCoPR now.
- **DFE limits of the wrong length.** A `b_max(2..N_b)` or `b_min(2..N_b)` vector
  that is not `N_b - 1` long was cut short or partly ignored; the Reference Code
  stops on it, and SiCoPR now stops with an error naming the keyword.

### Who is affected

- Configs that write a value as a MATLAB expression: they now run, with the
  reference's values.
- **Behaviour change: an Excel error cell.** A config with `#REF!` or `#N/A` in a
  keyword the Reference Code evaluates: COM Octave stops on such a cell, and
  SiCoPR now does too, with an error naming the keyword, where it used to carry
  the text on. What MATLAB's `xlsread` returns for an Excel error cell is
  unverified here. IEEE's 802.3dj CR `selectable` template has one in
  `z_bp (TX)` (`=Lists!F34`, saved as `#REF!`).
- **Not affected:** configs written with plain numbers and brackets, which
  includes every shipped and corpus workbook: the reader returns the same values
  bit for bit, and the 28 COM Octave checkpoint cases and the 208 MATLAB
  stage-oracle cases pass unchanged.

## [1.1.3] - 2026-10-21

Six port defects, found because IEEE's 802.3ck COM 3.70 C2M workbook stopped
SiCoPR 1.1.2 with an error where the Reference Code runs it, and a clearer error
for a configuration that stops the Reference Code itself. **C2M results can
change.** The reference corpus results do not: the 208 MATLAB stage-oracle cases
and the 28 COM Octave checkpoint cases pass unchanged.

### Who is affected

- **C2M configurations** (`PMD_type` or `PHY` C2M) with MLSE off: COM, FOM, the
  chosen CTLE and Tx FFE, and the reported eye widths can change. On the case that
  found it, the FOM moved by 2e-4 dB and the search chose a different CTLE and
  Tx FFE.
- **Any configuration that computes the eye width** (`EW = 1`) with MLSE off: the
  reported `EW_UI_est` and eye widths change, as below.
- **A two-segment package** (`package_Z_c` given as `[Tx Tx ; Rx Rx]`, as in the
  802.3ck COM 3.70 C2M and MTF workbooks) with `TDR_W_TXPKG = 1`: it now runs
  where it stopped with an error.
- **A workbook with spaced minus signs inside brackets**, such as
  `[ -0.15 - 0.15 - 0.05 ]`: the value is now read as MATLAB reads it.
- **The reported `flim_GHz`**, when the workbook's `flim` lies above the THRU's last
  frequency (the default is 1000 GHz).
- **Not affected:** the 802.3dj KR and CR configurations the correlation corpora
  use. They run MLSE (so no eye width is computed), set `flim` to 67 GHz, use
  four-segment packages and plain brackets, so none of these defects reach them.

### Fixed

What was found (symptom, cause) and the resolution, one per defect. Detail and
evidence: `docs/FIX_SUMMARY.md`, rows 52-57.

- **Two-segment package crash.** `TDR_W_TXPKG = 1` with a two-segment package
  stopped with `make_full_pkg: Zpkg(3): out of bound 2`, because the reader filled
  the package impedance out to four segments in the transposed shape (4x2 where
  the reference has 2x4). It now builds the reference's shape, and the reported
  `pkg_Z_c` has that shape too.
- **Frequency limit leaked to the caller.** `flim_GHz` reported the THRU's last
  frequency (67 GHz on the reproducing case) instead of the reference's 1000 GHz
  default, and files read after the THRU were cut at that frequency, because two
  file readers copied into `read_s4p_files` wrote their frequency limit back to the
  shared parameters where MATLAB changes only a local copy. They now leave the
  caller's value alone, as the main readers already did.
- **Eye widths one sample out.** Every left eye width was one sample short and
  every right one sample long, because `COM_eye_width` passed a 0-based centre
  index to a function that expects MATLAB's 1-based one. It now passes the 1-based
  index.
- **Eye contour one PDF bin out.** The eye contour was one bin (1e-5 V) out at
  about half its points, because `COM_eye_width` truncated each noise PDF's offset
  to an integer where MATLAB keeps it fractional and rounds the sum. On C2M the
  contour sets the noise amplitude, so this moved the FOM the equalizer search
  ranks by, and COM with it: 2e-4 dB of FOM and a different CTLE and Tx FFE on
  the reproducing case. The offset is now kept fractional, as in the reference.
- **One eye width instead of one per eye.** `EW_UI_est` added all the eyes'
  widths into one number (0.67) where the reference reports one per eye
  (`[0.21 0.23 0.21]`). It now reports one per eye.
- **Spaced minus signs in brackets.** `[ -0.15 - 0.15 - 0.05 ]` (the 3.70 C2M
  workbook's `b_min(2..N_b)`) was read as three numbers, because the reader joined
  every `- x` to `-x`; MATLAB reads `a - b` as a subtraction (here -0.35) and only
  `a -b` as two numbers. The reader now follows MATLAB.

Evidence: after these fixes the 3.70 C2M workbook matches COM Octave on 4p15p0,
4p16p0 and 4p17p0, with COM and FOM within 5.3e-14 dB and the sampling phase,
CTLE, `g_DC_HP`, Tx FFE, ERL, `EW_UI` and `flim` identical. Each fix has a test
pinned to COM Octave values that fails with the defect restored. The 208 MATLAB
stage-oracle cases and the 28 COM Octave checkpoint cases still pass unchanged.

### Changed

- **A defect in the Reference Code: ADC quantization with the Rx FFE off.** With
  `N_qb ~= 0` and the Rx FFE off (`ffe_pre_tap_len = ffe_post_tap_len = 0`), the
  Reference Code (4p14p0 to 4p17p0) runs the equalizer search and then stops with
  `structure has no member 'RxFFE'`. SiCoPR stops at the same point, as it must
  to match the reference, but now with a `ValueError` that names the cause and
  the workaround instead of a bare `AttributeError`. **Workaround:** set
  `N_qb = 0` when the Rx FFE is off (with no Rx FFE the reference adds no
  quantization noise to COM either way), or keep one Rx FFE tap if quantization
  noise is to be modelled. No shipped workbook reaches this. Root cause, a related
  observation about a zero pre-cursor tap, and a proposed fix for the COM
  ad hoc: [`docs/UPSTREAM_QUANTIZATION_WITHOUT_RXFFE.md`](docs/UPSTREAM_QUANTIZATION_WITHOUT_RXFFE.md).

## [1.1.2] - 2026-10-16

A security fix for the GUI, a faster COM Octave on 4p17p0, and test and tooling
work. No engine result changes: every COM, FOM and equalizer setting is what
1.1.1 computes.

### Changed

- COM Octave 4p17p0 runs a 4p17p0 variant of the floating-tap search speed patch
  (`octave/patches/FOM_rxffe_floating_taps_4p17p0.m`), and the optional compiled
  kernel has the matching mode: 1.6x to 1.8x faster, bit-identical to the release's
  own search. Rebuild the kernel (`python octave/accel/build_accel.py`) to use it.
- The shipped example runs 4p17p0, and its expected values were regenerated with
  both engines on it (COM unchanged to 1e-14 dB; the engines agree to 1.0e-14 dB).
- Octave speed figures re-measured on 4p17p0 (2026-10-15): with the compiled kernels
  COM Octave runs at 0.87x to 1.04x SiCoPR's time, 1.8x interpreted
  (`octave/README.md`, TUTORIAL 12.5).
- Tests: the unguarded `.copy()` sites the mutation gate reported fell from 31
  functions to 2, both fallback stubs production never calls. New MATLAB
  value-semantics checks (`com_functions/verification/value_semantics.py`) catch 20
  more sites; 28 are argued equivalent in `equivalent_mutants.md`; the gate no longer
  counts sites in `__main__` demo blocks. The verification report's `Finding:`
  trailer check is scoped to `py_impl.py`, as `docs/VERIFICATION.md` specifies.
  No engine result changes.
- `COM_FD_to_TD`, `COM_eye_width`, `get_TDR` and `get_PSDs` no longer carry
  hand-written fallback stubs for their callees: a call that does not inject them
  now uses the canonical functions, the same ones the engine injects. Several stubs
  were badly wrong (a Bessel-Thomson filter with DC gain 105, an eye contour with its
  edges swapped), though the engine never reached them. No engine result changes;
  the last two unguarded `.copy()` gaps went with them.

### Fixed

- The GUI now refuses requests that did not come from its own page. Binding to
  127.0.0.1 kept out the network but not other websites open in the same browser:
  a page on any site could have opened a whole drive (`/api/open` is a GET) and then
  overwritten workbooks on it through a text/plain POST to `/api/save`, and a
  DNS-rebinding page could also have read files from opened directories. Requests
  must now name this server in `Host`, must not be marked cross-site
  (`Sec-Fetch-Site`, `Origin`), and POSTs must be `application/json`, as the page's
  own already are. Found through the first CodeQL code scan.
- CI declares `permissions: contents: read`; its jobs had the repository's broader
  default token permissions.
- The two INFO lines about a channel's frequency range (data ending below the
  signalling rate; data starting above the recommended frequency) now print the
  Reference Code's own text, including its "GH" for "GHz". The conditions were
  already the reference's.

## [1.1.1] - 2026-10-06

### Fixed

- The engine dropped imports its function sources had, so paths no unit test
  reached raised `NameError`: a `.csv` configuration file and
  `writecsv_transposed` (`csv`), a `.csv` pulse-response file (`io`), and the
  apparent channel bandwidth fit when a workbook sets `ACBW = 1` (`cho_factor`,
  `cho_solve`, `solve_triangular`). The Reference Code reads a `.csv`
  configuration, so this was a port defect.
- A configuration missing a mandatory keyword (`M`, `R_0`, `RESULT_DIR`, `f_b`
  and the rest) failed later with an unrelated `TypeError` instead of naming the
  keyword: two modules' private `_SENTINEL` collided once assembled.
- A `.mat` configuration (the `parameter` cell array the Reference Code loads,
  as `tools/xlsx_to_com_mat.py` writes it for COM Octave) failed on its first
  arithmetic, because each cell arrived as a small array. It now loads to the
  same `param` and `OP` as the workbook it came from.
- A MATLAB range in a configuration cell that does not land on its limit is
  now read as MATLAB reads it: `[0:0.3:1]` is 0, 0.3, 0.6, 0.9, where the reader
  stretched it to 0, 0.333, 0.667, 1, and an end that the steps overshoot by
  rounding is set to the limit, as the colon does (`[0:0.1:0.3]` ends at 0.3).
  The translated `xls_parameter_txffe`, which the engine does not call, lacked
  that end clamp and now has it. Both are pinned against COM Octave. Every range in the shipped and reference workbooks reads exactly
  as before, so no reference result moves.
- `tests/test_assembled_names.py` now fails on either kind of defect in the
  assembled engine, and `tests/test_csv_inputs.py` loads every shipped workbook
  as a `.csv` and requires the same `param` and `OP`.

## [1.1.0] - 2026-10-04

### Added

- **COM 4p17p0** (release 4.17.0 of the Reference Code, 2026-09-23), now the
  default emulated release; `--matlab-version 4p16p0` and `4p15p0` remain. The
  release file is in `matlab/`, byte-identical to the tagged release (`NOTICE`
  now records every reference file's origin at
  https://opensource.ieee.org/802-com/com_code). What changed and what the port
  does about each item: [`docs/MATLAB_4p17p0_CHANGES.md`](docs/MATLAB_4p17p0_CHANGES.md).
  - `MMSE_FOM` builds its Gram matrix by lag from `H(:,1)'*H`, formed once in
    `MMSE`, as 4p17p0 does. Results move by summation order only.
  - Apparent channel bandwidth: `get_ACBW` and three helpers, reported as
    `ACBW_GHz` when a workbook sets `ACBW = 1` (threshold `T_dev`). Off by
    default.
- `octave/com_ieee8023_4p17p0_octave_compat.m`, generated by
  `octave/make_octave_compat.py`. It runs the release's own floating-tap search:
  the speed patch for that search predates 4p17p0's change and is not applied.
- `benchmark/208_case_4p17p0/`: the 208-case benchmark under 4p17p0, with its
  COM Octave results and their agreement with SiCoPR.
- `tools/octave_compare.py --resume`: each case's comparison is written as it
  finishes, and a resumed batch skips completed cases.

### Changed

- `MMSE` calls the canonical `MMSE_FOM` instead of carrying a copy of it.
- `tools/translation_coverage.py` measures against the default release named in
  `VERSION.json`; the command line's release choices come from it too.

### Fixed

- `tools/matlab_version_diff.py` did not recognise a function declaration
  continued with `...`, and merged its body into the function above. It missed
  three of 4p17p0's four new functions, and had folded `s21_to_impulse_DC` into
  its neighbour in every release since 4p14p0.

## [1.0.0] - 2026-09-29

First public release. The engine, the test suite and the documentation are the
state described in [`README.md`](README.md); the correlation results it reports
were produced before this file existed and are not restated as changes.

### Added

- `octave/com_ieee8023_4p15p0_octave_compat.m` and `..._4p16p0_octave_compat.m`
  now run under GNU Octave. They are generated from `matlab/` by
  `octave/make_octave_compat.py` with the patch set in `octave/patches/`, and
  `tests/test_octave_compat.py` checks that the committed files are what the
  generator produces and that Octave parses them. Before this they were
  byte-identical to the MATLAB releases.
- The Octave release files run **2.5 to 3 times faster than the same files before
  this work** (measured 2026-09-18: one 4p16p0 case 1418 s to 561 s; one 4p15p0
  case 137 s to 46 s), and every result is unchanged to the last bit. On the
  shipped example (measured 2026-10-01, one case at a time, one BLAS thread),
  Octave takes 2.2 to 2.4 times SiCoPR's run time interpreted and 1.2 to 1.4
  times with the compiled kernels. Part of it is in the patch
  set (the touchstone reader, the floating-tap search, the ISI distribution
  build, FFE, two `isequal` tests); the rest is `octave/accel/`, **optional
  compiled kernels** for the three hottest loops, built once per machine with
  `python octave/accel/build_accel.py` and used automatically from then on
  (`COM_OCTAVE_ACCEL=0` turns them off, which is how the equivalence is
  checked). The `.oct` is per Octave version and platform, so it is not
  committed, and a build that does not match is ignored. Bit-identity is
  enforced by `tests/test_octave_compat.py`, which compares the release's own
  functions with the kernels off and on and fails on each of five planted
  last-bit defects.
- `tools/xlsx_to_com_mat.py` — a COM workbook as the `.mat` configuration
  Octave reads, with `--set KEY=VALUE` for the headless plumbing keywords.
- `tools/octave_compare.py` — the same case through Octave and `python -m
  sicopr`, compared field by field; batch mode over a JSON case list.
- `NOTICE` — upstream provenance, origin and SHA-256 checksums for each of the
  four MATLAB reference files.
- `CITATION.cff`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, this file.
- `com_functions/inlined_copies.json` — a generated manifest recording, for
  every inlined helper copy in the engine, which helper it came from, which
  function it was inlined into, and the upstream MATLAB line range of both.
- Licence headers on the standalone source files (`gui/`, `tools/`, `R/`,
  the assembler and the plotting modules). The 157 `py_impl.py` files are
  covered collectively by [`com_functions/fn/README.md`](com_functions/fn/README.md)
  rather than individually — the assembler inlines their leading comments, so a
  per-file header would appear 157 times inside the generated engine.
- `com_functions/verification/report.py` answers "are there any opens?" from the
  suite; all 146 translated functions are checked against the executed reference.
  The contract is [`docs/VERIFICATION.md`](docs/VERIFICATION.md).
- `tests/test_octave_checkpoints.py`, `tests/test_mutation_score.py`,
  `tools/gen_semantics_pins.py` and `tools/equivalence_check.py`; see
  `docs/VERIFICATION.md`.
- `benchmark/208_case_4p15p0/`: the 208-case benchmark with its COM Octave
  results, the channel list (IEEE URLs and SHA-256, no channel data) and the
  eight configuration workbooks, with the results' provenance and their measured
  agreement with the MATLAB reference results.
- `--exit-on-fail`: exit status 3 if any package case fails its COM threshold,
  for CI. Without it a completed run still exits 0, pass or fail.
- README: the Python entry point, batch use, exit codes and runtime; why a
  Python port when Octave runs the Reference Code; how the port was built (with
  AI assistance) and verified; the compliance statement; the 2026-09-29 ad hoc
  slides and acknowledgements; screenshots of the GUI. A defect-report issue
  template.

### Changed

- The documentation now states the maintainer's commitment as presented to the
  IEEE 802.3 COM ad hoc on 2026-09-29. The commitment is consistency and
  correlation with the Reference Code. SiCoPR follows official releases, and
  between releases the engine changes only for port defects and speed-ups
  proven to change no result. The supporting tools (`gui/`, `R/`, the study
  tools, search instrumentation) are outside it. README, `CONTRIBUTING.md`,
  `docs/MAINTAINING.md`.
- Outside help now comes through issues only. Pull requests are no longer
  accepted, so the DCO sign-off section is gone from `CONTRIBUTING.md`. The CI
  `dco` job remains, and runs only on pull requests.

- Documentation restructured for users. `README.md` is now a short guide:
  install (`pip install -e .`), the worked example, running your own case with
  the actual output file names, the reference code under Octave, the
  correlation headline, limitations and licence. Version detail moved to the new
  [`docs/VERSIONS.md`](docs/VERSIONS.md); test-suite detail, inlined-copy
  coverage and the correlation-data policy moved to `CONTRIBUTING.md`; the
  equalizer-search study and repository layout to `docs/TUTORIAL.md`. Personal
  names replaced by roles, stale example output and figure names corrected.
- Behavioural coverage of the inlined helper copies raised from 113 to 136 of
  177; five further divergences were found in the process and are catalogued in
  `test_inlined_copies.py`'s `KNOWN_BEHAVIOUR`. On 2026-09-22 most copies were
  then replaced by imports: 62 copies of 45 functions remain, 34 of which cannot
  be driven from synthetic inputs and are disclosed as unverified in
  `CONTRIBUTING.md`.
- `results.csv` is written in MATLAB `num2str` format (about five significant
  digits), as the reference writes it. Compare engines at full precision:
  `tools/_sicopr_case.py` writes every result field to JSON, and
  `tools/octave_compare.py` reads that.
- The August speed-ups' FFT convolution and hoisted Gram matrix were withdrawn
  (the FFT lost the far tail of the noise CDF that DER is read from). Speed-ups
  are now accepted only under `tools/equivalence_check.py`: strict outputs
  bit-identical, noise and DER fields within 1e-12 relative per element, on 28
  checkpoint cases. The August 4.5–5× figure is withdrawn pending re-measurement.

### Fixed

- The R dashboard could not be built from any `--export-mat` file written since
  2026-09-05: R.matlab cannot read an empty text field, and the export's
  `meta.git_commit` was empty whenever a run started outside the repository
  (as the GUI's runs do). The export now never writes an empty field, and
  `meta.com_version` names the release the run emulated.
- GUI: static-results captions and the "not produced by this run" line used
  only `/` to find a file name, so on Windows every figure was listed as
  missing; the unsaved-changes bar did not hide at zero; the S-parameter legend
  now names the return-loss traces and its axis labels are larger; the Run tab
  says when figures are drawn.
- The example's COM Octave results no longer record a local path.
- A configuration workbook saved by a tool that writes whole numbers as `32.0`
  crashed the engine (`np.ones(32.0)`); MATLAB, where every number is a double,
  runs it. Whole-number cell values now reach the engine as integers, as they
  did from Excel-saved workbooks, and so do CSV values. Keywords the engine uses
  as counts (`M`, `L`, `N_b`, tap lengths and others) are conditioned where every
  source is read, whatever its type; a fractional count is an error naming the
  keyword instead of a crash. Checked by running the shipped example with every
  number in its workbook forced to float: every pinned value reproduced.
- `__load_excel` left configuration workbooks open, locking them on Windows.
- Three tests asserted *that* something failed rather than *why*, and would have
  passed with the guard they were protecting removed.
