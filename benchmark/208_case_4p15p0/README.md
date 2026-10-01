# The 208-case benchmark, with COM Octave results

The case set behind the correlation result in the top-level README, published so
anyone can run it: 26 IEEE 802.3dj CR and KR channels x 4 package configurations x
with and without crosstalk = 208 cases, on the COM 4p15p0 build with the adaptive
local search.

The results here are **COM Octave results**: the IEEE 802.3 COM Reference Code run
under GNU Octave. They are not the MATLAB reference results, which are not published
in this repository. They travel with the benchmark so that a reader without a MATLAB
licence still has a reference to compare against.

| file | what it is |
|---|---|
| `cases_com_octave.csv` | one row per case: case id, condition, package case, sweep row, configuration workbook, IEEE contribution, THRU file, aggressor counts, and the COM Octave results (COM, COM_orig, delta_COM, FOM, itick, ERL, VEC, VEO, ICN, FOM_ILD, DER, pass, Tx FFE taps, CTLE, DFE taps, floating tap locations) |
| `channels.csv` | every channel file the cases use (164: 26 THRU plus their aggressors), its role, the IEEE contribution it comes from, the contribution's download URL, and the file's SHA-256 so you can confirm you have the same data |
| `configs/` | the eight COM configuration workbooks (Cases 1 to 4, each plain and with the Tx FFE swept), BSD-3-Clause by their own `License Notice` sheet |

The channel files are IEEE 802.3dj contributions, public but not redistributed here.
Download each contribution's zip from the URL in `channels.csv` (six zips, all in the
IEEE 802.3dj public tools area, https://www.ieee802.org/3/dj/public/tools/index.html)
and check the files against the SHA-256 listed.

## Provenance of the COM Octave results

| | |
|---|---|
| Reference Code release | `com_ieee8023_4p15p0_adaptive_local_search.m` (in `matlab/`; NOTICE records its SHA-256) |
| Octave | GNU Octave 11.3.0, Windows 11 |
| Octave route | the `Octave_compat` branch of the COM code repository at revision `7c321fe` (`src/`), with two overrides that restore mainline behaviour: `conv_fct_noPDFx` (restores `p.x`) and `read_Nport_touchstone` (filters NaN lines). This repository's own Octave release files (`octave/`) were generated later from the same release by a fuller patch set; they have not been re-run on all 208 cases. See `octave/README.md` |
| run | 2026-09-04, all 208 cases, no failures |
| measured agreement with the MATLAB reference results | COM within 5.06e-14 dB (median 9.77e-15 dB); FOM within 3.49e-11 dB; sampling phase (itick), equalizer selection and pass/fail at 3 dB identical on 208 of 208 |

COM spans -4.82 to +6.68 dB across the set, and 123 of the 208 cases fail the 3 dB
threshold, so the agreement is not confined to easy cases.

## Running it

Use the 4p15p0 build, which is what these results were produced with:

```powershell
python -m sicopr configs/<workbook>.xlsx <thru.s4p> --fext <f1.s4p ...> --next <n1.s4p ...> --matlab-version 4p15p0
```

Each row of `cases_com_octave.csv` names the workbook, the THRU file and how many
aggressors it has; the aggressors are the FEXT and NEXT files listed beside that THRU
in `channels.csv`. To compare at full precision, use `tools/_sicopr_case.py` (every
result field to JSON) or `tools/octave_compare.py`, not `results.csv`, which carries
about five significant digits as the reference's report does.

Agreement to look for: COM to about 1e-13 dB and identical sampling phase and
equalizer. `DER_DFE` and `DER_MLSE` can differ by a few percent between any two
engines, because the reference reads the CDF exactly on a bin edge; COM is unaffected.

---

The COM Octave results and this description: Copyright 2026 Todd Bermensolo,
BSD-3-Clause. The configuration workbooks: Copyright 802-COM Authors, BSD-3-Clause
(their `License Notice` sheet).
