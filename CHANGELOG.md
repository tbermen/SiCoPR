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

## [1.0.0] - 2026-09-27

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
  case 137 s to 46 s), and every result is unchanged to the last bit. With the
  compiled kernels, Octave then took about 1.5 times SiCoPR's run time on that
  4p16p0 case (451 s against 299 s); SiCoPR has changed since and the ratio has
  not been re-measured. Part of it is in the patch
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
- DCO sign-off requirement for contributions, enforced by CI
  (`CONTRIBUTING.md` explains why BSD-3-Clause needs one).
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

### Changed

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

- A configuration workbook saved by a tool that writes whole numbers as `32.0`
  crashed the engine (`np.ones(32.0)`); MATLAB, where every number is a double,
  runs it. Whole-number cell values now reach the engine as integers, as they
  did from Excel-saved workbooks. The same applied to CSV configurations.
- `__load_excel` left configuration workbooks open, locking them on Windows.
- Three tests asserted *that* something failed rather than *why*, and would have
  passed with the guard they were protecting removed.
