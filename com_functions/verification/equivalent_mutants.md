# Equivalent mutants

Some mutations provably change nothing. `ddof=1` on a single-element array is
the canonical case: both normalisations divide by the same thing when there is
one sample, so the mutant is not a defect and the test that "failed to catch"
it did nothing wrong.

These read as `survived` and are **not** test failures. They are listed here as
decided state, with the reason, and excluded by `EQUIVALENT` in
`tests/test_mutation_score.py`.

## Why this list is safe and a stored `discriminates` note is not

An entry here is **falsifiable**. If a mutant ever stops being equivalent, some
test starts catching it, and `test_mutation_score.py` reports the entry as
contradicted by `known_survivor_list_is_current`. The claim is re-checked on
every run against the code, exactly like an `xcheck` that starts passing.

A note saying "someone reintroduced the bug and watched the test fail" has no
such property: gut the test's assertions later and the note still says it
discriminates. That is why discrimination is computed and only equivalence is
written down.

## The rule for adding an entry here

An entry must argue that the mutant **cannot** change the result, from the code,
not that the change is small or unlikely to matter. "The test does not happen to
cover this input" is a reason to strengthen the test, and belongs in
`KNOWN_SURVIVORS` as a real gap. It is not equivalence.

If you cannot write the argument in one sentence that a reader can check, it is
not an equivalent mutant.

## Entries

Keyed `operator:function:line` in `EQUIVALENT`, never `operator:function`. MMSE
has six `.copy()` sites and exactly one is redundant, so a function-level entry
would excuse the other five.

| operator | site | why the mutant cannot change the result |
|---|---|---|
| `drop_dot_copy` | `MMSE:371` | `wmin = -wmax.copy()`. Unary negation already allocates a new array, so `-wmax` is not a view of `wmax` whether or not `.copy()` is there. The later `wmin[...] = 1.0` cannot reach `wmax` either way |
| `drop_dot_copy` | `get_PSDs:436` | `hisi = h[samp_idx].copy()` where `samp_idx = np.arange(...)`. Indexing with an integer ARRAY is fancy indexing, which always returns a new array, never a view. The copy is redundant |
| `drop_dot_copy` | `get_pdf_full:128` | `residual_response = SBR.copy()` where `SBR = np.interp(...)` on the line above. `np.interp` allocates its result, and nothing else holds a reference to it, so writing `residual_response` cannot be observed |

Note how each argument turns on **provenance of the source**, not on what the
tests happen to cover. `get_pdf:142` looks identical to `get_pdf_full:128` and
is NOT equivalent, because there `SBR` is `np.asarray(chdata.eq_pulse_response)
.ravel()`, which returns a VIEW onto the caller's data. That pair is the reason
this file demands the argument in writing.
