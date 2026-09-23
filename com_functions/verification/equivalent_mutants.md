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

## The `H_ph_corr = H_ph.copy()` family

| operator | site | why the mutant cannot change the result |
|---|---|---|
| `drop_dot_copy` | `interp_Sparam:238` | `H_ph_corr = H_ph.copy()` in the `trend_and_shift_to_DC` branch. `H_ph` is last read on the line that computes the group delay, and the branch never reads it again: everything after uses `H_ph_corr`. The in-place loop corrupts a local nobody looks at |
| `drop_dot_copy` | `interp_Sparam:268` | Same shape in the `extrap_cubic_to_dc_linear_to_inf` branch, whose last read of `H_ph` is the group-delay line |
| `drop_dot_copy` | `interp_Sparam:302` | `H_ph_i = H_ph_cubic.copy()`; `H_ph_cubic` is not read after the copy |
| `drop_dot_copy` | `s21_to_impulse_DC:200` | Same family; its branch has already read `H_ph` for the group delay |
| `drop_dot_copy` | `s21_to_impulse_DC:233` | Same family; its branch has already read `H_ph` for the group delay |

These five were each read and checked by hand, and they are the reason
`triage_copies.py` is a triage AID rather than an authority. It is
flow-insensitive: it sees the read of `H_ph` in a sibling `elif` and reports
the site as load-bearing, when the two branches are mutually exclusive and the
read can never follow the write.

Making that analysis flow-sensitive is not worth it. The tool's job is to turn
104 sites into a handful worth reading, and it does: 104 down to 4. A human
settles those 4. A tool that tried to settle them itself would be a tool whose
verdicts nobody checks, which is the 2026-07 ledger with better tooling.

**These entries are all equivalence of the same kind, and it is worth naming:**
the copy protects a value whose only remaining consumer is the copy itself. If
a later change starts reading the source after the write, the mutant stops
being equivalent, a test starts catching it, and
`known_survivor_list_is_current` contradicts the entry. The claim is checked on
every run, which is the whole point of writing it here rather than in a
comment.

## Unreachable code

| operator | site | why the mutant cannot change the result |
|---|---|---|
| `solve_to_lstsq` | `force:352` | The non-square else-branch. `VV` is built as `zeros(num_taps, num_taps)` in both languages, so the branch cannot execute and nothing in it can change a result. Its SQUARE counterpart at line 341 is caught, by the singular-VV test added under the 2026-09-23 ruling |

This is equivalence of a different kind from the rest of the file: not "the
write cannot be observed" but "the code cannot run". It is the weaker claim of
the two, because it rests on a reading of how `VV` is built rather than on a
property of the values, so it is worth re-checking if `force` is ever
restructured.
