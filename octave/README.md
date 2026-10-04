# octave/

The IEEE 802.3 COM Reference Code, made to run under GNU Octave, and the
evidence behind it.

**To run a case**, see [Running a case](#running-a-case): convert the workbook
to a `.mat`, then call the release file. Nothing here needs building.

**To make it faster**, build the optional compiled kernels once
(`python octave/accel/build_accel.py`). Every later run finds them and is about
1.4 to 1.9 times faster than the same files run interpreted (measured
2026-09-18), with the same command and the same results, bit for bit. See [Optional compiled kernels](#optional-compiled-kernels).

**To understand what was changed and why**, read on: the patch set is the point
of this directory, and every item in it is a no-op under MATLAB.

## The three files

| file | derived from | how |
|---|---|---|
| `com_ieee8023_4p15p0_octave_compat.m` | `matlab/com_ieee8023_4p15p0_adaptive_local_search.m` | `make_octave_compat.py` |
| `com_ieee8023_4p16p0_octave_compat.m` | `matlab/com_ieee8023_4p16p0.m` | `make_octave_compat.py` |
| `com_ieee8023_4p17p0_octave_compat.m` | `matlab/com_ieee8023_4p17p0.m` | `make_octave_compat.py` |

They are **generated** from `matlab/` by a small, named patch set, and committed so a
reader needs no build step, the same arrangement as `sicopr.py`. (The optional
compiled kernels below are the one thing that is built, and the files run
without them.) Each carries a
provenance block after its licence header naming its source and the source's
SHA-256. `tests/test_octave_compat.py` fails if a committed file is not what the
generator produces, or if Octave cannot parse it.

The 4p15p0 file exists so that Octave can be compared exactly with the MATLAB
reference results. Those were produced with **4p15p0 plus the adaptive local
search**, the build in `matlab/com_ieee8023_4p15p0_adaptive_local_search.m`,
which is also the build SiCoPR emulates (`VERSION.json`, `primary_reference`).
So that is the 4p15p0 source here, not the bare release, whose legacy local
search would agree with neither. The 4p16p0 file is the release that adopted the
adaptive search into the mainline, and the 4p17p0 file is the current release.
All three take the same patch set, with one exception: **4p17p0 does not take the
`FOM_rxffe_floating_taps` speed patch.** That patch inlines `MMSE_FOM`'s Gram matrix
as 4p16p0 forms it, `H(:,sel)'*H(:,sel)`; 4p17p0 builds it by lag from
`H(:,1)'*H`, so on 4p17p0 the patch would compute the previous release's
arithmetic. The release's own search runs there, and the compiled search kernel,
reached only through the patch, stays off (`NOT_REPLACED` in the generator). The
cost: on the shipped example 4p17p0 under Octave took 789 s against SiCoPR's 374 s.

```
python octave/make_octave_compat.py            # regenerate both
python octave/make_octave_compat.py --check    # what the test runs
```

## What the patch set is, and why each item is there

Every item was found by running the official release under Octave 11.3.

| change | kind | why |
|---|---|---|
| `verLessThan('matlab', ...)` guarded by `exist('OCTAVE_VERSION','builtin')` | line edit in the main function | Octave resolves the first argument against installed packages and errors on every run |
| `read_Nport_touchstone` replaced | `patches/` | Two defects in Octave's `textscan`, either of which is reason enough. It can **stop part way through a touchstone file, silently**, whether it reads a file handle or the file's text as a string; and it **does not parse to the nearest double** — on twelve values from a real channel its `%f` was one ulp out on nine, where `str2double` matched all twelve, so even a file it read in full came back with last-bit-wrong S-parameters (~1e-12 through the FD metrics, ~1e-14 on COM). On the 2026-09-15 corpus, 102 of 1650 files came back short and 36 of them below 67 GHz; a short THRU passes every check and puts COM tens of dB out. The replacement reads the file's text whole (`fileread`) and parses its tokens (`str2double`), with no `textscan` at all. That also retires the earlier blank-line problem (one NaN per blank line, 12 of 164 files losing their frequency axis), since blank lines are only whitespace between tokens |
| `MMSE`: `Rn = real(Rn)` after the `ifft` | line edit | Octave `ifft` always returns complex; MATLAB returns real for a conjugate-symmetric input. The complex residue reaches the Toeplitz solve, every FOM candidate is rejected, and the equalizer search finds no solution. `MMSE` is a local subfunction, so no path shim can reach it: this is the item that forced the patch into the file |
| `CDF_ev` replaced | `patches/` | the mainline does `find` over an axis that grows every MLSE iteration; the `lookup` form is what keeps an Octave run at minutes rather than hours. **This is the whole performance story**: the release file under Octave was measured 30 to 60 times slower than the branch, and this one function is why |
| `MLSE_U1_c_178A`: `real()` on the `CDF_ev` arguments | line edit | as the branch has it; a no-op once `MMSE` is real |
| `COM_CommandLine_Parse` replaced | `patches/` | sets `OP.OCTAVE`, detected automatically here (the branch requires an `'Octave'` keyword, which still works) |
| `read_ParamConfigFile`: `.csv` via `csvread4com` | line edit + `patches/csvread4com.m` | no `xlsread` under Octave |
| `writecsv_transposed` replaced | `patches/` | Octave has no `writecell` |

The `CDF_ev`, `COM_CommandLine_Parse` and `writecsv_transposed` bodies are the
versions that were run on 208 cases against the MATLAB reference to 5e-14 dB,
taken from the `src/` tree of an `Octave_compat` branch of the COM Reference
Code. The touchstone reader is this project's own; the branch's version, with a
NaN filter reported upstream, still uses `textscan`. **Nothing in this table changes a
number under MATLAB**: each edit is a no-op there, which is what makes the
result a reference and not a fork.

The generator also normalises function terminators: Octave requires that a file
close every function with `end` or none, the release closes none, and the
branch's stand-alone files are inconsistent. Getting this wrong is a parse error
reported at the last line of a 12,000-line file.

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

These are checked under Octave, where the files run. They have not been checked
under MATLAB; the correctness items above are the ones that must be no-ops
there.

## Optional compiled kernels

**Nothing here happens unless you build it.** The release files are complete on
their own and run interpreted, as they always have. `accel/com_octave_accel.cc`
is C++ for three of their hottest loops — the floating-tap search's candidate
loop, the ISI distribution loop and FFE's tap loop — and building it puts
`com_octave_accel.oct` beside the `.m` files. From then on those three loops run
compiled **automatically**, on any run that has `octave/` on its path, with no
change to the command, the configuration or the results.

```
python octave/accel/build_accel.py     # once per machine -> octave/com_octave_accel.oct
```

Is it in use? Ask the kernel itself; it answers only if it is built, loadable
and the version the release files expect:

```
octave-cli --eval "addpath('octave'); disp(com_octave_accel('version'))"
```

A version string means the release files will use it. `error: 'com_octave_accel'
undefined` means they will not, and will run interpreted.

To run without it, set `COM_OCTAVE_ACCEL=0` in the environment before starting
Octave (the release decides once per session, on its first COM call). That is
the switch the equivalence check uses, and the one to reach for if a result ever
looks suspect: the same case with the kernels off must give the same numbers.

What you need to build it: `mkoctfile`, which comes with Octave, and a C++
compiler. The Windows Octave installer brings its own `g++`, so nothing else is
needed there; on Linux or macOS install the compiler the usual way. Built and
measured here on Octave 11.3, Windows.

The `.oct` is built for one Octave version and platform, so **it is not
committed and there is no prebuilt binary to download**. Rebuild it after
upgrading Octave: the release files check the version string, and ignore a
build that does not match or does not load, falling back to interpreted code
rather than failing. On Windows, Octave holds a loaded `.oct` open, so rebuild
with no Octave session using it.

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
uses full-precision inputs, which matters more than it sounds:
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

**The command is the same with or without the compiled kernels.** The
`addpath('octave')` above is what finds them, since `com_octave_accel.oct` is
built into that directory, so a run picks them up with no flag once they are
built and ignores them when they are not. `COM_OCTAVE_ACCEL=0` in the
environment runs interpreted whatever is built. Same for `octave_compare.py`
and any harness that puts `octave/` on the path.

Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` when running several
Octave cases at once: COM under Octave is loop-bound, and one BLAS thread per
process is what lets N processes use N cores. Memory, not CPU, is the limit on
how many fit: about 0.3 GB for a case without crosstalk and up to 1.2 GB with
it.

## Measured

**Agreement.** One 208-corpus case (KR package A case 1, without crosstalk),
2026-09-10. Both agree with the
MATLAB reference result for this case to within 1e-14 dB:

| | COM_dB |
|---|---|
| SiCoPR, `python -m sicopr` | 3.9661839670990786 |
| Octave 11.3, `..._4p16p0_octave_compat.m` | 3.96618396709909 |

Since then, on a 1368-case 4p16p0 workload (171 channels, four configurations,
with and without crosstalk), **Octave and SiCoPR ran every case and agree within
5.3e-14 dB on all 1368**, with the same sampling phase on every one
(re-run 2026-09-26 on SiCoPR `df78b9c`). The earlier subset comparison across channel families is held
privately with the channels it needs and is not published.

**Speed**, per case, same machine, one BLAS thread each:

| | Octave against SiCoPR | measured |
|---|---|---|
| **interpreted, speed items in** | **2.2x to 2.4x** | 821 s / 786 s against 375 s / 335 s, the shipped example without / with crosstalk, 2026-10-01 |
| **with the compiled kernels** | **1.2x to 1.4x** | 457 s / 454 s against the same, 2026-10-01 |
| earlier: before the speed items (2026-09-10) | about 3.3x | 147 s against 44 s, 208-corpus case |
| earlier: with the compiled kernels (2026-09-18) | about 1.5x | 451 s against 299 s, one 1368-case 4p16p0 case (`wo_C1_R001`) |

The 2026-10-01 rows are current: one case at a time on the same machine, every
engine giving the same COM to within 2e-14 dB. The earlier rows were taken before
SiCoPR's September convolution changes and are kept as history.

## What is in this directory, and what left it

| | |
|---|---|
| `com_ieee8023_*_octave_compat.m` | the two generated files, committed |
| `make_octave_compat.py` | the generator |
| `patches/` | **the source.** The generator reads these; without them nothing can be regenerated and `--check` cannot run. Not spare parts |
| `accel/` | the optional compiled kernels: `com_octave_accel.cc` and `build_accel.py`. Nothing here runs until you build it |
| `com_octave_accel.oct` | what that build produces, beside the `.m` files where a run finds it. Per machine, never committed, safe to delete |
| `README.md` | this file |

Earlier path overrides (files placed earlier on Octave's search path to shadow
the Reference Code's own versions) are not part of this repository. They could
not reach `MMSE`, a local subfunction, and generating the file removed the need
for them.

One rule about Octave is worth keeping if you ever shadow a function this way:
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
Reference Code, `Copyright 2025 802-COM Authors`, SPDX `BSD-3-Clause`, headers
intact. The generator and the changes it applies are
`Copyright 2026 Todd Bermensolo` under the same licence.

`accel/com_octave_accel.cc` translates three of the release's own loops into
C++, so it is derived work as well and carries both notices.
`patches/com_octave_accel_on.m`, which decides whether the kernels are used, is
original and carries only ours. `NOTICE` records all of this.
