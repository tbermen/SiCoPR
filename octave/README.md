# octave/

What it takes to run the IEEE 802.3 COM reference code under GNU Octave, and
the evidence behind it.

## The two release files

| file | md5 | identical to |
|---|---|---|
| `com_ieee8023_4p15p0_octave_compat.m` | `1f2006a2e5bf4bba45d505ece9db016b` | `matlab/com_ieee8023_4p15p0.m` |
| `com_ieee8023_4p16p0_beta1_octave_compat.m` | `15bfd011eda92fabafea3fdab2cfc85f` | `matlab/com_ieee8023_4p16p0.m` |

These were downloaded from the `release/` folder of the `Octave_compat` branch
of the COM reference repository, and they are **byte-identical** to the
mainline releases already in `matlab/`. They are kept, duplication and all,
because that identity is the finding: `git diff main..Octave_compat -- release/`
is empty, so the branch's `release/` folder carries **none** of the Octave
compatibility work. Anyone downloading from there expecting Octave support will
not get it.

The compatibility work lives in the branch's `src/` folder instead, behind an
`OP.OCTAVE` flag set by passing `'Octave'` as the first argument.

Verified 2026-09-02 by direct comparison. Neither file contains any Octave-aware
code: no `OCTAVE_VERSION`, no `isoctave`, no guarded `exist` check.

## shims/

Overrides for running the **mainline release file** under Octave. Retired: that
route was abandoned after a third incompatibility and a roughly 30x runtime
penalty. Kept as evidence of what the official release needs.

| shim | why |
|---|---|
| `verLessThan.m` | line 92 calls `verLessThan('matlab','7.4.1')` unguarded on every run. Octave resolves the first argument against installed packages and errors. Returns a constant, so it cannot move a number. |
| `textscan.m` | the touchstone reader finds frequency lines by counting NaN per row of a 9-wide `textscan`, which needs MATLAB line-record semantics. Octave streams numbers continuously. Validated bit-exact against an independent parse on all 164 s4p files in the corpus before retirement. |

A third blocker had no shim and ended that route: Octave `ifft` always returns
complex where MATLAB returns real for a conjugate-symmetric input. The residue
reaches the MMSE solve and the equalizer search finds no solution at all. `MMSE`
is a local subfunction of the monolithic release file, so it cannot be shadowed
from the path, and the only override point would have been `ifft` itself.

## shims_B/

Active overrides for the branch `src/` tree, which is the route that works. Both
restore behaviour the mainline code already has; neither adds any.

| shim | why |
|---|---|
| `conv_fct_noPDFx.m` | the branch version skips building `p.x` for speed, but `CDF_ev` indexes that axis, so from the second MLSE loop iteration the index comes from a stale, shorter axis. Restores the one commented-out line `conv_fct` already has. Without it, DER_MLSE came out 9.4e-14 instead of 4.6e-08 and COM read 6.153 dB where the reference is 3.496 dB. |
| `read_Nport_touchstone.m` | the branch reader reshapes a flat value stream without filtering NaN. Octave `textscan` emits one NaN per blank line, so any touchstone file whose frequency blocks are separated by blank lines loses its frequency axis. 12 of 164 files in the corpus are written that way and every case using them failed. |

Both carry the upstream BSD-3-Clause header and copyright, and both were
reported upstream.

Octave `addpath` **prepends**, so a shim directory must be added **last** to take
precedence. Getting that backwards runs the shadowed copy while appearing to
have applied the override.

## Result

With the branch `src/`, these two shims and a `.mat` configuration, COM under
Octave 11.3.0 reproduced a MATLAB 4.15 reference across 208 cases to within
5.1e-14 dB, with matching equalizer selection, matching sampling phase and
identical pass/fail on every case.

The full procedure is in the Octave usage document produced by the
`octave-three-way` study.

## Licence

The two release files and the `shims_B/` overrides derive from the COM reference
code, `Copyright 2025 802-COM Authors`, SPDX `BSD-3-Clause`. Headers are intact.
`shims/` is original work under the same licence.
