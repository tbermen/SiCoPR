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

## [Unreleased]

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
