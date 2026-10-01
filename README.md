# SiCoPR

*(pronounced si-copper)* — **Si** (Signal Integrity), **Co** (COM), **P** (Python), and **R**
(the R language, used for the reports)

By Todd Bermensolo (Independent), [github.com/tbermen](https://github.com/tbermen).
Questions and defect reports: [GitHub issues](https://github.com/tbermen/SiCoPR/issues).

SiCoPR is a Python port of the IEEE 802.3 **COM** (Channel Operating Margin) reference
code. It reads the same Excel configuration workbook and Touchstone `.s4p` files as the
MATLAB tool and computes COM, VEO/VEC and the supporting equalization and noise analysis,
without MATLAB. It emulates **`com_ieee8023_4p16p0`** by default and `com_ieee8023_4p15p0`
with `--matlab-version 4p15p0`. Licence: BSD-3-Clause, the same as the reference code
([`LICENSE`](LICENSE), provenance in [`NOTICE`](NOTICE)).

The full guide is [`docs/TUTORIAL.md`](docs/TUTORIAL.md): architecture, every feature,
the configuration keyword index, the study layer, the R reports and a COM primer.

## Install

Python 3.10 or newer. From a clone:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

pip install -e .
```

Install the package, not just `requirements.txt`: the example and `python -m sicopr`
import `sicopr`, and without the install they fail with `No module named sicopr` unless
run from the repository root. Required dependencies are numpy, scipy and openpyxl.
Optional extras:

| extra | adds | for |
|---|---|---|
| `pip install -e ".[plots]"` | matplotlib | figures (`SAVE_FIGURES = 1` in the config) |
| `pip install -e ".[report]"` | python-pptx, python-docx | report tooling |
| `pip install -e ".[dev]"` | all of the above, plus pytest | development and tests |

`requirements.txt` carries the same minimums plus the test-only packages (`pytest`,
`esprima`). The R reports under `R/` need, inside R:
`install.packages(c("R.matlab", "ggplot2", "dplyr", "tidyr", "scales", "plotly", "htmltools", "jsonlite"))`.

## Run the worked example

[`examples/`](examples/) holds one configuration workbook and the results both engines in
this repository produced on it. The channel is an IEEE 802.3dj contribution, public but
not redistributed here, so you download it yourself:

1. Download one zip, about 29 MB:
   <https://www.ieee802.org/3/dj/public/tools/CR/akinwale_3dj_02_2311.zip>. It unpacks
   to a single folder, `akinwale_3dj_01_2311/`.
2. Run the example against the unpacked folder:

```powershell
python examples/run_example.py --channels <where you unpacked the zip> --engine sicopr
```

The script checks each channel file against the SHA-256 listed in
[`CHANNEL.md`](examples/akinwale_CR_22dB_VendorX/CHANNEL.md), runs the case without and
with crosstalk, and compares COM, FOM, itick, ERL, VEC, VEO and ICN against the shipped
values. Each SiCoPR run takes about six minutes on one core. Success prints
`all pinned values reproduced`; without crosstalk the case gives COM = 3.5070 dB, with
its five aggressors 2.8959 dB. Leave out `--engine sicopr` to also run the reference code
under GNU Octave, if `octave-cli` is on PATH.
[`EXPECTED.md`](examples/akinwale_CR_22dB_VendorX/EXPECTED.md) has every number.

## Run your own case

```powershell
python -m sicopr <config.xlsx> <thru.s4p> [--fext f1.s4p ...] [--next n1.s4p ...]
```

The THRU (victim) channel is required. Aggressors are optional; `--fext` and `--next`
each take any number of files.

| flag | what it does |
|---|---|
| `--fext F ...` | far-end crosstalk aggressor files |
| `--next N ...` | near-end crosstalk aggressor files |
| `--matlab-version {4p15p0,4p16p0}` | which reference release to emulate; default `4p16p0` ([`docs/VERSIONS.md`](docs/VERSIONS.md)) |
| `--export-mat` | also write a per-case engineering `.mat` snapshot for the R dashboard (TUTORIAL §7.4). Changes no result |
| `--eye-under-mlse` | compute the eye contour and timing bathtub for plotting even when MLSE is on. Diagnostic only; no reported value changes |

The configuration is an IEEE 802.3 COM workbook (sheet `COM_Settings`); start from a
working-group reference workbook rather than building one. A local web editor that shows
a workbook as a channel schematic is in [`gui/`](gui/README.md) (`python gui/app.py`).

**What it writes.** The console prints one block per package case:

```
--- Case 1 ---
  COM_dB                         = 3.5070
  VEO_mV                         = 2.3899
  VEC_dB                         = 9.5721
  FOM_ILD                        = 0.2992
  ICN_mV                         = 0.0000
  Result                         = PASS  (threshold 3.0 dB)
```

(the shipped example, without crosstalk). Files go under the config's `RESULT_DIR`
(`{date}` is replaced by the run date; if `RESULT_DIR` is blank,
`results_<config-name>_<YYYY_MM_DD_HH_MM>/`), relative to the working directory, one
`case_NN/` per package case:

| file | written when | content |
|---|---|---|
| `results.csv` | `CSV_REPORT = 1` | every reported quantity, in MATLAB `num2str` format (about five significant digits), as the reference writes it |
| `s1_*.png` … `s7_*.png` | `SAVE_FIGURES = 1` | figures named by pipeline stage: `s1_insertion_loss`, `s1_return_loss`, `s1_filters`, `s2_tdr_impedance`, `s2_erl_summary`, `s3_sbr_full`, `s3_sbr_zoom`, `s4_fom_vs_phase`, `s5_eq_vs_uneq_sbr`, `s5_fom_convergence`, `s5_eq_taps`, `s6_pdfs`, `s6_noise_terms`, `s6_contribution_pie`, `s7_eye_contour`, `s7_voltage_bathtub`, `s7_timing_bathtub` |
| `STAGE_INDEX.md` | `SAVE_FIGURES = 1` | which figure belongs to which stage; a stage with no figure is listed as such |
| `<config-name>_caseNN.mat` | `--export-mat` | engineering snapshot, written in `RESULT_DIR` itself |

To compare two engines, use full precision rather than `results.csv`:
`tools/_sicopr_case.py` writes every result field to JSON.

### From Python, in batches and in CI

The command line is a thin wrapper around one function, which a script can call directly.
It is the entry point the repository's own tools use (`tools/_sicopr_case.py`); its
leading underscore marks it as not yet a frozen API.

```python
import sicopr

sicopr.COM_MATLAB_VERSION = '4p16p0'            # as --matlab-version; optional
res = sicopr._run_com('config.xlsx', 1, 1,       # number of FEXT, number of NEXT
                      ['thru.s4p', 'fext1.s4p', 'next1.s4p'], export_mat=False)
cases = res if isinstance(res, list) else [res]  # one entry per package case
for r in cases:
    print(r.COM_dB, r.FOM, r.itick)              # every reported field is an attribute
```

A sweep is a loop over that call, one channel or configuration at a time; each call is
independent, so separate processes can run cases in parallel (memory, about 0.5 to 1.5 GB
per case, is usually the limit before CPU). The examples' `run_example.py` is a worked
script of this kind.

**Exit codes.** `python -m sicopr` exits **0 when the run completes, whether COM passes or
fails**, 1 if the engine returns no result, and non-zero with a traceback on any error
(for example an unreadable file, or a configuration count that is not a whole number). A
CI check that should fail on a failing channel reads `COM_dB` from the result or from
`results.csv` and compares it with the threshold itself.

**Runtime.** One case takes from under a minute to about ten minutes on one core,
depending mostly on crosstalk and the size of the equaliser search: the shipped example
took 360 s without crosstalk and 336 s with it (measured 2026-09-26; `run_example.py`
prints the time of every run, so it is also the way to time your own machine). No
runtime comparison with MATLAB is offered, because the timings that exist were taken on
different machines; the comparison with the Reference Code under Octave is in the next
section.

### Reading the output

- **COM** (dB): the ratio of the available signal amplitude to the combined noise and
  interference amplitude at the target detector error ratio. A channel passes when COM
  meets the configured threshold, conventionally 3 dB.
- **FOM**: the figure of merit the equalizer search maximizes, a fast SNR-style estimate.
  The search picks the equalizer setting by FOM; COM is then computed once, for the winner.
- **itick**: the sampling phase the search chose, as a sample offset within the unit
  interval.
- **Tx FFE**, **CTLE**, **DFE**: the equalizers the search sets: transmitter
  feed-forward equalizer taps, receiver continuous-time linear equalizer gains, and
  receiver decision-feedback equalizer taps. Configurations may add a receiver FFE.
- **MLSE**: maximum-likelihood sequence estimation. When enabled, COM includes the
  reference's adjustment for a sequence detector.
- **VEO / VEC**: vertical eye opening (mV) and vertical eye closure (dB), used by
  chip-to-module specifications.
- **ICN** (mV): integrated crosstalk noise from the FEXT/NEXT aggressors. Zero means no
  aggressors were supplied, not that there is no crosstalk.
- **ERL** (dB): effective return loss, a reflection metric computed from the TDR.

TUTORIAL [chapter 6](docs/TUTORIAL.md#6-feature-reference) covers each feature and
[chapter 7](docs/TUTORIAL.md#7-outputs) the outputs; [Appendix A](docs/TUTORIAL.md#appendix-a-com-concepts)
is a COM primer.

## The reference code under GNU Octave (optional)

`octave/` carries the COM 4p15p0 and 4p16p0 release files, generated from `matlab/` by a
small patch set that makes them run under GNU Octave and is a no-op under MATLAB. With
Octave installed you can run the reference code itself on any case and compare it with
SiCoPR field by field (`tools/octave_compare.py`), without a MATLAB licence. Speed,
measured 2026-09-18 on one 4p16p0 case (Octave 11.3, Windows, one BLAS thread): with the
optional compiled kernels built, Octave took about 1.5 times SiCoPR's run time (451 s
against 299 s); SiCoPR has changed since and the ratio has not been re-measured. See
[`octave/README.md`](octave/README.md).

**Why a Python port, when Octave already runs the Reference Code for free?** The two do
different jobs, and this repository ships both. Octave runs the Reference Code itself,
which makes it the right oracle: SiCoPR is verified against it. A Python port is useful
where COM has to live inside other software: called from Python scripts, batch sweeps
and CI, with every function traceable to the MATLAB lines it mirrors and testable on its
own. The Octave route rests on work by members of the COM community, credited in
[`NOTICE`](NOTICE) and below.

## How do I know it's right?

- **Function level.** All 146 value-bearing functions (of the 157 translated) are checked
  against the *executed* Reference Code (COM Octave), not against a reading of it.
  [`docs/VERIFICATION.md`](docs/VERIFICATION.md) is the contract, and
  `python com_functions/verification/report.py` answers "are there any opens?".
- **Against MATLAB, 4p15p0.** 208 reference cases (26 IEEE 802.3dj CR/KR channels ×
  4 package configurations × with/without crosstalk), last re-run **2026-09-23**:
  sampling phase (itick) and every equalizer selection identical on 208 / 208, COM within
  **4.6e-14 dB** (median 9.3e-15 dB), FOM within **3.4e-11 dB**. The reference is
  `com_ieee8023_4p15p0_adaptive_local_search.m`, the build the reference results came from.
- **Against COM Octave, 4p16p0.** 1368 cases (171 distinct channels), **2026-09-26**:
  COM within **5.3e-14 dB**, and itick, Tx FFE and CTLE gain identical on all 1368.

Those differences are double-precision arithmetic noise, not agreement to a tolerance.
The correlation inputs are IEEE contributions and are not redistributed here, and MATLAB
result values are never published here; only agreement statistics are. So the 208-case
figures cannot be reproduced from this repository alone: that needs the channels from
the IEEE 802.3dj public area and the MATLAB reference results. What you can reproduce
here is the worked example (above), against both SiCoPR's and COM Octave's results, and
any case of your own against the Reference Code under Octave (`tools/octave_compare.py`). Method, history
and per-case detail: [`MATLAB_Correlation_Review.md`](MATLAB_Correlation_Review.md) and
[`docs/FIX_SUMMARY.md`](docs/FIX_SUMMARY.md).

One known engine-to-engine difference: `DER_DFE` and `DER_MLSE` can differ by up to about
4.5% between any two engines, Octave and MATLAB included, because the reference reads the
CDF exactly on a bin edge; COM is unaffected.

### How it was built

The port was developed with AI coding assistance. That is why the verification above is
built the way it is: nothing rests on a reading of the code, by a person or a model. Each
Reference Code function was translated on its own and checked against that function
executed under GNU Octave on the same inputs; the assembled engine was then checked stage
by stage against Octave and case by case against the MATLAB reference results. Where the
two languages differ in a way a reading would miss (`std`, `round`, `interp1`,
`transpose` and others), the rule is recorded in
[`com_functions/verification/builtins.md`](com_functions/verification/builtins.md).
[`docs/VERIFICATION.md`](docs/VERIFICATION.md) describes the method.

## Known limitations

- **End-to-end coverage follows the corpora.** The correlation covers what the corpus
  configurations switch on; the shipped example alone runs Tx FFE, CTLE, DFE, Rx FFE
  (MMSE), MLSE and crosstalk. Features no corpus configuration enables are verified
  function by function only; [`docs/FEATURE_STATUS.md`](docs/FEATURE_STATUS.md) is the
  feature inventory (a dated snapshot; its §D lists the candidates).
- **Time-domain input (`TDMODE`)** is not fully wired; supply S-parameters.
- **`FFE_OPT_METHOD = 'WIENER-HOPF'`** is intentionally non-functional: its helper is
  undefined in the MATLAB reference itself. Use `'MMSE'`.
- **`FAST_NOISE_CONV`** is a speed approximation; the default exact path is recommended
  for reported results.
- **GUI file pickers** from the MATLAB tool are not ported; pass files on the command line.
- **Results produced before August 2026 are not comparable** with current output: engine
  fixes changed COM materially. Regenerate rather than compare with archived numbers.
- **No runtime comparison with MATLAB is offered**: the timings that exist were taken on
  different machines.

## What the maintainer commits to

SiCoPR is maintained by Todd Bermensolo, and the commitment is deliberately narrow:
**consistency and correlation with the IEEE 802.3 COM Reference Code.** The Reference
Code is the one COM code base: reviewed by the COM ad hoc and released on the IEEE COM
Git site. SiCoPR is an extra implementation outside it. It is not a second standard,
not IEEE collateral, and not a place to change COM. **For compliance and anything
normative, the IEEE 802.3 COM Reference Code is the authority; SiCoPR reproduces it but
does not replace it.**

- **SiCoPR follows official releases.** The default is the latest official Reference
  Code release. An older build stays selectable only while published correlation
  evidence depends on it. Today that is the 4p15p0 build with the adaptive local
  search, which produced the 208-case reference results. Drafts and development
  branches are not followed.
- **Each new release goes through the same steps:**
  1. Diff it against the previous release.
  2. Map the differences onto the functions that need re-checking.
  3. Rerun the per-function tests and the benchmark cases.
  4. Publish the correlation tables.

  The 4p15p0 to 4p16p0 step was done this way ([`docs/VERSIONS.md`](docs/VERSIONS.md)).
- **Between releases the engine changes for two reasons only:**
  - a **port defect**, meaning SiCoPR disagrees with the release it emulates;
  - a **speed-up proven to change no result**, judged by `tools/equivalence_check.py`
    ([`CONTRIBUTING.md`](CONTRIBUTING.md), *A change made for speed*).
- **A change to COM itself goes to the COM ad hoc and the Reference Code first.** That
  covers a new feature, a different method, or a correction to the Reference Code.
  SiCoPR re-correlates once the change is released. A defect found in the Reference
  Code is reproduced here, not corrected, and reported to the ad hoc.
- **Supporting tools are outside the commitment.** These are the configuration editor
  (`gui/`), the R reports and dashboard (`R/`), the study tools and the search
  instrumentation. None of them changes a COM result, they are provided as they are,
  and they are not re-verified with each release. The Octave release files under
  `octave/` are inside the commitment: they are regenerated for each release, because
  they are the executable reference the verification runs against.

## Issues and tests

Defect reports are welcome as [GitHub issues](https://github.com/tbermen/SiCoPR/issues),
especially any case where SiCoPR disagrees with the Reference Code. **Pull requests are
not accepted**: the maintainer makes every change, so every change goes through the same
verification. Feature requests and proposed changes to the COM method belong with the
COM ad hoc. [`CONTRIBUTING.md`](CONTRIBUTING.md) says what makes a report actionable,
and records how the code is worked on: the verification rules and the detail of the
test suite.

`sicopr.py` is generated from `com_functions/fn/<name>/py_impl.py` by
`assemble_sicopr.py`: edit the source and re-assemble, never the engine. The gate is

```powershell
powershell -ExecutionPolicy Bypass -File tests/run_all.ps1
```

Do not run bare `pytest tests`: it collects almost nothing and still exits 0. On a fresh
clone, tests that need the (unshipped) correlation data skip and say so.

## Licence, citation, acknowledgements

BSD-3-Clause ([`LICENSE`](LICENSE)). The MATLAB reference this is ported from is
BSD-3-Clause, *Copyright 2025 802-COM Authors*; its sources are redistributed unmodified
under `matlab/`, notices intact, and [`NOTICE`](NOTICE) records their versions and
checksums. The COM configuration workbooks carry the same licence in their
`License Notice` sheet.

**Not an IEEE product.** This project is not endorsed by, affiliated with, or approved by
IEEE, the IEEE 802.3 working group or the 802-COM Authors; it is an independent port of
the published reference code. "IEEE 802.3" appears here only to identify the standard the
reference implements.

To cite the software, use [`CITATION.cff`](CITATION.cff); when the subject is the COM
method itself, cite IEEE 802.3 and the reference code. The method, the reference code and
the configuration workbooks are the work of the IEEE 802.3 COM authors, to whom this port
owes everything it computes.

**Presented** to the IEEE 802.3 COM Open Source Project Ad Hoc on 2026-09-29, with
co-author Hansel D'Silva (Amphenol): *COM: how open-source ports match the Reference
Code* ([slides](https://www.ieee802.org/3/ad_hoc/COM/public/telecon/260929/bermensolo_COM_01_260929.pdf)).
The MATLAB reference results for the 208-case and 1368-case comparisons, and the
adaptive local search in the 4p15p0 build, are Hansel D'Silva's. COM Octave support came
from Rich Mellitz and Adam Gregory, and review and direction from Howard Heck. Their
help is acknowledged with thanks; it implies no endorsement by them or by IEEE.
