# Changelog

Notable changes to SiCoPR. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

**This project has not been released yet, and has never carried a version number
or a git tag.** There is therefore no version history to reproduce here, and
inventing one would misrepresent what happened. This file starts at the first
public release and is kept from there.

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

First public release. The engine, the test suite and the documentation are the
state described in [`README.md`](README.md); the correlation results it reports
were produced before this file existed and are not restated as changes.

### Added

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

### Changed

- Behavioural coverage of the inlined helper copies raised from 113 to 136 of
  177. The 41 that remain cannot be driven from synthetic inputs and are
  disclosed as unverified in `README.md`; five further divergences were found in
  the process and are catalogued in `test_inlined_copies.py`'s `KNOWN_BEHAVIOUR`.

### Fixed

- `__load_excel` left configuration workbooks open, locking them on Windows.
- Three tests asserted *that* something failed rather than *why*, and would have
  passed with the guard they were protecting removed.
