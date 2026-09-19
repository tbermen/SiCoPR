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
| `read_Nport_touchstone` replaced | `patches/` | Octave's `textscan` can **stop part way through a touchstone file, silently**, whether it reads a file handle or the file's text as a string. On the 2026-09-15 corpus, 102 of 1650 files came back short and 36 of them below 67 GHz; a short THRU passes every check and puts COM tens of dB out. The replacement reads the file's text whole (`fileread`) and parses its tokens (`str2double`), with no `textscan` at all. That also retires the earlier blank-line problem (one NaN per blank line, 12 of 164 files losing their frequency axis), since blank lines are only whitespace between tokens |
| `MMSE`: `Rn = real(Rn)` after the `ifft` | line edit | Octave `ifft` always returns complex; MATLAB returns real for a conjugate-symmetric input. The complex residue reaches the Toeplitz solve, every FOM candidate is rejected, and the equalizer search finds no solution. `MMSE` is a local subfunction, so no path shim can reach it: this is the item that forced the patch into the file |
| `CDF_ev` replaced | `patches/` | the mainline does `find` over an axis that grows every MLSE iteration; the `lookup` form is what keeps an Octave run at minutes rather than hours. **This is the whole performance story**: the release file under Octave was measured 30 to 60 times slower than the branch, and this one function is why |
| `MLSE_U1_c_178A`: `real()` on the `CDF_ev` arguments | line edit | as the branch has it; a no-op once `MMSE` is real |
| `COM_CommandLine_Parse` replaced | `patches/` | sets `OP.OCTAVE`, detected automatically here (the branch requires an `'Octave'` keyword, which still works) |
| `read_ParamConfigFile`: `.csv` via `csvread4com` | line edit + `patches/csvread4com.m` | no `xlsread` under Octave |
| `writecsv_transposed` replaced | `patches/` | Octave has no `writecell` |

The `CDF_ev`, `COM_CommandLine_Parse` and `writecsv_transposed` bodies are the
versions the three-way study ran on 208 cases against the MATLAB reference to
5e-14 dB, taken from Rich Mellitz's `Octave_compat` branch `src/` tree. The
reader is ours (2026-09-16); the branch's version, with the NaN filter we
reported upstream, still uses `textscan`. **Nothing in this table changes a
number under MATLAB**: each edit is a no-op there, which is what makes the
result a reference and not a fork.

## Speed, with every result unchanged

Octave has no JIT, so its time goes to small calls made very often. The items
below remove that overhead and nothing else. The rule for admitting one is
that **no result may change, not even in the last bit**: a before/after run on
four cases (two 4p15p0, two 4p16p0 with crosstalk) compares every field of the
result struct byte for byte, and only `config_file` and `rtmin`, which name the
run, may differ. An item that would move a bit was dropped: computing one
`H'*H` for the whole search and indexing it per candidate was 1.8 times
faster on the search, and changed its FOMs in the last bits on 135 of 138 tap
sets.

| change | kind | what it saves |
|---|---|---|
| `read_Nport_touchstone`: `str2double` on the tokens | in the replaced reader | `sscanf` over a rejoined copy took 5.8 s to parse one 8001-point 4-port file, against 1.4 s; 1.8x on reading across all 1814 files of both corpora, identical S-parameters on every one |
| `MMSE_FOM`: `any(b ~= blim)` for `isequal(b, blim)`, and likewise for `w` | line edits | `isequal` is an m-file, 34 µs a call, run per candidate |
| `FFE`: each tap added in place as two blocks | line edit | `circshift` is an m-file and built a shifted copy per tap |
| `get_pdf_from_sampled_signal`: `Init_PDF_Fast` and `conv_fct` inlined, bins filled in one assignment when distinct | line edit | two calls and a struct per ISI sample |
| `FOM_rxffe_floating_taps` replaced | `patches/` | `MMSE_FOM`'s search-mode work inlined, everything that does not change per candidate computed once |

These are checked under Octave, where the files run. They are not checked
under MATLAB, which is not on this machine; the correctness items above are
the ones that must be no-ops there.

## Optional compiled kernels (`accel/`)

`accel/com_octave_accel.cc` runs three of the hottest loops compiled: the
floating-tap search's candidate loop, the ISI distribution loop and FFE's tap
loop. The release files work without it. When `com_octave_accel.oct` sits
beside them, `com_octave_accel_on` (an added function) sends those loops to it
instead.

```
python octave/accel/build_accel.py        # -> octave/com_octave_accel.oct
COM_OCTAVE_ACCEL=0                        # environment: run without it
```

The kernels return **exactly** what the interpreted code returns, byte for
byte. The constraint shapes the design:

- every operation is the one the interpreter performs, in the same order,
  through the same liboctave calls (`xgemm` with the interpreter's transpose
  flags, `Matrix::solve` as `A\b` calls it, `octave::range`, `octave::convn`);
- it is built with `-ffp-contract=off`, because a fused `a*b+c` rounds once
  where the interpreter rounds twice;
- the ISI distribution's convolution runs as a few shifted adds only on a step
  where every product is exact. `conv2` rounds some of its multiply-adds once
  (fused, inside BLAS) and some twice, which makes no difference while the
  products are exact and a real one once a distribution's far tail sinks into
  the subnormal range, as a few hundred 4-level steps do. On such a step the
  kernel calls `convn`, as `conv2` does.

`tests/test_octave_compat.py` runs the release's own functions with the
kernels off and on and compares every output, including every candidate FOM
of the search and a distribution whose tail goes subnormal. The check was
proven by planting five last-bit defects (a reassociated product, two
summation orders reversed, a build with fused multiply-add allowed, and the
convolution shortcut without its exactness check), and it fails on each. It
uses full-precision inputs:
Octave's `randn('seed', ...)` selects an old generator whose values are all
single precision, and the product of two of those is exact, which hides a
fused multiply-add completely. The first version of the check had exactly that
blind spot.

Measured 2026-09-18, four cases run side by side, every result field
byte-identical with the kernels off and on: the kernels alone are 1.36 to
1.92 times faster, and with the interpreted items above the release is 2.5 to
3 times faster than it was before any speed item (1368-case `wo_C1_R001`:
1418 s to 561 s). On that case Octave now takes 1.5 times
SiCoPR's time (451 s against 299 s, one `tools/octave_compare.py` run).

The `.oct` is built for one Octave version and platform, so it is not
committed. The release files ask it for its version string and ignore a
build that does not match. On Windows, Octave holds a loaded `.oct` open, so
rebuild with no Octave session using it.

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

## What is in this directory, and what left it

| | |
|---|---|
| `com_ieee8023_*_octave_compat.m` | the two generated files, committed |
| `make_octave_compat.py` | the generator |
| `patches/` | **the source.** The generator reads these; without them nothing can be regenerated and `--check` cannot run. Not spare parts |
| `accel/` | the optional compiled kernels' source and `build_accel.py` |
| `com_octave_accel.oct` | built by `build_accel.py`, never committed |
| `README.md` | this file |

`shims/` and `shims_B/` **moved out on 2026-09-10**, to
`studies/octave-three-way/harness/`. They were path overrides: files placed
earlier on Octave's search path so they shadowed the reference code's own
versions. That was the only way to change a monolithic release file from
outside, and it is why the third blocker ended the attempt, since `MMSE` is a
local subfunction that nothing on the path can shadow. Generating the file
removed the need for the technique entirely.

They still have one job, which is why they were moved rather than deleted: the
`octave-three-way` study's two arms are defined by them, and that study's
2026-09-04 release is the ad hoc deliverable. They now live beside the harness
that is their only caller.

One rule survives them, because it is about Octave rather than about shims:
`addpath` **prepends**, so a directory added last ends up searched first.
Getting that backwards runs the copy you meant to shadow while the log says the
override was applied.

## patches/

Do not delete this. The relationship is the one `sicopr.py` has with
`com_functions/fn/*/py_impl.py`: the generated file is committed for
convenience, and the sources are the code. `tests/test_octave_compat.py` runs
`make_octave_compat.py --check`, which regenerates from `patches/` and compares.

## Licence

The two generated files and everything in `patches/` derive from the COM
reference code, `Copyright 2025 802-COM Authors`, SPDX `BSD-3-Clause`, headers
intact. The generator and the changes it applies are
`Copyright 2026 Todd Bermensolo` under the same licence.
