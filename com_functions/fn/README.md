# `com_functions/fn/` — the per-function translation

Copyright 2025 802-COM Authors (upstream MATLAB)
Copyright 2026 Todd Bermensolo (Python translation)
SPDX-License-Identifier: BSD-3-Clause

One directory per translated function. **Everything under here is BSD-3-Clause**;
the full text is in [`../../LICENSE`](../../LICENSE) and the upstream provenance,
including SHA-256 checksums, is in [`../../NOTICE`](../../NOTICE).

This file exists because the individual files below carry no header of their own,
and someone who opens one — or lifts one out of the tree — should still be able
to tell what it is and what licence it is under.

## What is in each directory

| file | what it is | licence |
|---|---|---|
| `py_impl.py` | the Python translation of one MATLAB function. **This is the editable source**; `sicopr.py` is generated from it. | BSD-3-Clause, derivative of the upstream |
| `matlab_source.m` | a **verbatim excerpt of the upstream MATLAB**, kept beside its translation so a reviewer can compare them without opening the 12,000-line reference. 49 directories have one. | BSD-3-Clause, *Copyright 2025 802-COM Authors* — literal upstream material |
| `test_verify.py` | the per-function test | BSD-3-Clause |

## Why the files carry no per-file header

`assemble_sicopr.py` inlines each `py_impl.py` into the generated engine
**including its leading comments**. A licence header added to each of the 157
files would therefore be reproduced 157 times inside `sicopr.py`, which already
carries one authoritative header. The header lives here instead, once, covering
the whole directory.

`matlab_source.m` files are excerpts rather than complete files, so they do not
carry the upstream's own header either. They are unmodified upstream text and are
covered by the notice at the top of this file. The complete upstream files, with
their own licence notices intact, are in [`../../matlab/`](../../matlab).

## Correspondence to the upstream

[`../registry.json`](../registry.json) records, for all 157 functions, the
upstream line range each was translated from:

```json
{"name": "com_ieee8023_", "matlab_lines": [1, 904], "lines": 904, ...}
```

Most `py_impl.py` files repeat that range in their own header comment.

For **inlined helper copies** the record is
[`../inlined_copies.json`](../inlined_copies.json), which the assembler writes as it mints
the mangled names. A `py_impl.py` may define a private helper `_foo`; the assembler renames
it to `_<module>__foo` so copies in different modules cannot collide. Each entry names the
copy, the helper it came from, the function it was inlined into, and the upstream line range
of both:

```json
{"copy": "_get_PSDs__H_interp", "helper": "H_interp", "inlined_into": "get_PSDs",
 "inlined_into_matlab_lines": [6380, 6654],
 "canonical": "H_interp", "canonical_matlab_lines": [2180, 2198]}
```

280 copies are recorded; 177 of them duplicate a function that also exists as a canonical
top-level, and those are the ones `tests/test_inlined_copies.py` compares behaviourally.
**A fix to a `py_impl.py` reaches the canonical copy only** — the manifest is how you find
the others.

## Editing

Edit `py_impl.py`, run its `test_verify.py`, then re-run `assemble_sicopr.py`.
Never edit `sicopr.py` — see [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md).
