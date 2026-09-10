# octave/

The IEEE 802.3 COM reference code, made to run under GNU Octave, and the
evidence behind it.

## The two files

| file | derived from | how |
|---|---|---|
| `com_ieee8023_4p15p0_octave_compat.m` | `matlab/com_ieee8023_4p15p0_adaptive_local_search.m` | `make_octave_compat.py` |
| `com_ieee8023_4p16p0_octave_compat.m` | `matlab/com_ieee8023_4p16p0.m` | `make_octave_compat.py` |

Until 2026-09-10 these were byte-identical to the MATLAB releases: files whose
name promised Octave compatibility and whose content had none. They are now
**generated** from `matlab/` by a small, named patch set, and committed so a
reader needs no build step, the same arrangement as `sicopr.py`. Each carries a
provenance block after its licence header naming its source and the source's
SHA-256. `tests/test_octave_compat.py` fails if a committed file is not what the
generator produces, or if Octave cannot parse it.

The 4p15p0 file exists so that Octave can be compared exactly with the MATLAB
reference results. Those were produced with **4p15p0 plus the adaptive local
search**, the build in `matlab/com_ieee8023_4p15p0_adaptive_local_search.m`,
which is also the build SiCoPR emulates (`VERSION.json`, `primary_reference`).
So that is the 4p15p0 source here, not the bare release, whose legacy local
search would agree with neither. The 4p16p0 file is the current release, which
adopted the adaptive search into the mainline. Both take the same patch set; the
two sources are identical in every function it touches.

```
python octave/make_octave_compat.py            # regenerate both
python octave/make_octave_compat.py --check    # what the test runs
```

## What the patch set is, and why each item is there

Every item was found by running the official release under Octave 11.3 in the
2026-09 three-way study, each costing a full run to discover.

| change | kind | why |
|---|---|---|
| `verLessThan('matlab', ...)` guarded by `exist('OCTAVE_VERSION','builtin')` | line edit in the main function | Octave resolves the first argument against installed packages and errors on every run |
| `read_Nport_touchstone` replaced | `patches/` | the mainline reader finds frequency lines by counting NaN per row of a 9-wide `textscan`, which needs MATLAB line-record semantics; the replacement reads a flat `%f` stream, **filters NaN before reshaping**, and reshapes by count. Without the filter, Octave emits one NaN per blank line and 12 of the 164 corpus files lose their frequency axis |
| `MMSE`: `Rn = real(Rn)` after the `ifft` | line edit | Octave `ifft` always returns complex; MATLAB returns real for a conjugate-symmetric input. The complex residue reaches the Toeplitz solve, every FOM candidate is rejected, and the equalizer search finds no solution. `MMSE` is a local subfunction, so no path shim can reach it: this is the item that forced the patch into the file |
| `CDF_ev` replaced | `patches/` | the mainline does `find` over an axis that grows every MLSE iteration; the `lookup` form is what keeps an Octave run at minutes rather than hours. **This is the whole performance story**: the release file under Octave was measured 30 to 60 times slower than the branch, and this one function is why |
| `MLSE_U1_c_178A`: `real()` on the `CDF_ev` arguments | line edit | as the branch has it; a no-op once `MMSE` is real |
| `COM_CommandLine_Parse` replaced | `patches/` | sets `OP.OCTAVE`, detected automatically here (the branch requires an `'Octave'` keyword, which still works) |
| `read_ParamConfigFile`: `.csv` via `csvread4com` | line edit + `patches/csvread4com.m` | no `xlsread` under Octave |
| `writecsv_transposed` replaced | `patches/` | Octave has no `writecell` |

Every replaced body is the version the three-way study ran on 208 cases against
the MATLAB reference to 5e-14 dB, taken from Rich Mellitz's `Octave_compat`
branch `src/` tree (plus our NaN filter, which the branch lacked and which was
reported upstream). **Nothing in the set changes a number under MATLAB**: each
edit is a no-op there, which is what makes the result a reference and not a
fork.

The generator also normalises function terminators: Octave requires that a file
close every function with `end` or none, the release closes none, and the
branch's stand-alone files are inconsistent. Getting this wrong is a parse error
reported at the last line of a 12,000-line file.

## Running a case

Octave cannot read the `.xlsx` configuration. Convert it once:

```
python tools/xlsx_to_com_mat.py config.xlsx -o config.mat --set RESULT_DIR=out/ --set SAVE_FIGURES=0 --set DISPLAY_WINDOW=0
octave-cli --no-gui --no-window-system --eval "addpath('octave'); r = com_ieee8023_4p15p0_octave_compat('config.mat', 0, 0, 'thru.s4p'); save('-v7','r.mat','r')"
```

or let `tools/octave_compare.py` do both sides and the comparison:

```
python tools/octave_compare.py config.xlsx thru.s4p --fext a.s4p b.s4p --next c.s4p --version 4p15p0
```

Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` when running several
Octave cases at once: COM under Octave is loop-bound, and one BLAS thread per
process is what lets N processes use N cores.

## Measured

One 208-corpus case (the sender's acceptance channel, KR package A case 1,
without crosstalk), on this machine, 2026-09-10:

| | COM_dB | wall |
|---|---|---|
| MATLAB reference (4p15p0, sender's machine) | 3.96618396709908 | unknown |
| SiCoPR, `python -m sicopr` | 3.9661839670990786 | 44 s |
| Octave 11.3, `..._4p16p0_octave_compat.m` | 3.96618396709909 | 147 s |

Octave at about 3.3 times SiCoPR's time, the ratio the three-way study measured
on the branch. The subset comparison across channel families and configurations
is the `octave-sicopr-correlation` study in the `si-studies` repository.

## shims/ and shims_B/

Kept as evidence, not used by anything.

`shims/` were the path overrides for running the official release file under
Octave before the fixes lived in the file. `shims_B/` were the two overrides the
three-way study applied to the branch `src/` tree; both are now inside the
patch set (`read_Nport_touchstone` verbatim; `conv_fct_noPDFx` is unnecessary
because the release calls `conv_fct`, which already builds the PDF axis the
branch's variant dropped).

Octave `addpath` **prepends**, so a shim directory must be added **last** to take
precedence. Getting that backwards runs the shadowed copy while appearing to
have applied the override. This is one reason the fixes are in the file now.

## Licence

The two release files, everything in `patches/` and `shims_B/` derive from the
COM reference code, `Copyright 2025 802-COM Authors`, SPDX `BSD-3-Clause`,
headers intact. The generator, the changes and `shims/` are
`Copyright 2026 Todd Bermensolo` under the same licence.
