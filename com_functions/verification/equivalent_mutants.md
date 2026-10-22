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
| `drop_dot_copy` | `MMSE:260` | `wmin = -wmax.copy()`. Unary negation already allocates a new array, so `-wmax` is not a view of `wmax` whether or not `.copy()` is there. The later `wmin[...] = 1.0` cannot reach `wmax` either way |
| `drop_dot_copy` | `get_PSDs:473` | `hisi = h[samp_idx].copy()` where `samp_idx = np.arange(...)`. Indexing with an integer ARRAY is fancy indexing, which always returns a new array, never a view. The copy is redundant |
| `drop_dot_copy` | `get_pdf_full:165` | `residual_response = SBR.copy()` where `SBR = np.interp(...)` on the line above. `np.interp` allocates its result, and nothing else holds a reference to it, so writing `residual_response` cannot be observed |

Note how each argument turns on **provenance of the source**, not on what the
tests happen to cover. `get_pdf:142` looks identical to `get_pdf_full:165` and
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
| `mround_to_np_round` | `get_ACBW:58` | `Nwin = max(3, round(smooth_window_GHz/delta_f))` (4p17p0 L6859). Its only consumer is `get_CICP_fit_sweep`'s `Nwin` argument, which the reference uses for `smoothed_residual` and `rms_residual_smoothed`, both computed and never read; the port does not compute them. The value cannot reach an output, so neither can its rounding. `get_BW_from_CICP_residual` computes its own window, and its rounding is caught (2026-10-03) |
| `solve_to_lstsq` | `force:352` | The non-square else-branch. `VV` is built as `zeros(num_taps, num_taps)` in both languages, so the branch cannot execute and nothing in it can change a result. Its SQUARE counterpart at line 341 is caught, by the singular-VV test added under the 2026-09-23 ruling |

This is equivalence of a different kind from the rest of the file: not "the
write cannot be observed" but "the code cannot run". It is the weaker claim of
the two, because it rests on a reading of how `VV` is built rather than on a
property of the values, so it is worth re-checking if `force` is ever
restructured.

## The 2026-10-15 pass over the remaining `drop_dot_copy` survivors

Fifty-four sites, read one by one. Twenty became catches, through
`com_functions/verification/value_semantics.py` (no two arrays MATLAB would
hold as separate values may share memory; an input the reference treats as
read-only comes back unchanged) or a value test where the copy guards a
result. Six sat in fallback stubs that production never calls
(`COM_eye_width`, `get_TDR:71`); the same day the owner approved replacing
those stubs with imports of the canonical functions, and the six sites went
with them. The rest are below, each on the same footing
as the entries above: the argument is about what the code can do, never about
what a test happens to cover.

| operator | site | why the mutant cannot change the result |
|---|---|---|
| `drop_dot_copy` | `read_s4p_files:79`, `make_pkg:24`, `make_pkg:25`, `make_full_pkg:77`, `make_full_pkg:78`, `s21_pkg:56` | `s12pad = s21pad.copy()` and `s22pad = s11pad.copy()` in the pad model. The four pad arrays are only ever passed to `combines4p`, which reads its arguments and allocates its results, and then rebound; nothing writes them and none is returned |
| `drop_dot_copy` | `read_s4p_files:581` | Both copies of the S12/S21 swap in `_read_p4_s4params_inline`'s `nport == 2` branch. That reader is called only from the `.s4p` branch and takes `nport` from the extension, so `nport` is 4 and the branch cannot run |
| `drop_dot_copy` | `read_s4p_files:716` | The FIRST copy of the 2-port swap, `sp[1,0,:].copy()`: Python evaluates the whole right side before writing either slot, so the view it would be reads the old value. The second copy, on the next line, is not redundant and is caught |
| `drop_dot_copy` | `interp_Sparam:84`, `:85`, `:147`, `:148`, `:247`, `:248`; `s21_to_impulse_DC:86`, `:127`, `:207` | `fin_x = fin.copy()`, `H_mag_x = H_mag.copy()`, `ph_x = H_ph_corr.copy()`. Each is only read: by `np.concatenate`, which allocates, and by `_interp_extrap`, which writes only its own `np.interp` result. None is written or returned |
| `drop_dot_copy` | `OptFom_Create_Output:120` | `SRn = PR.copy()`; `SRn` is only rebound (`SRn = SRn + SPR` allocates) and compared, never written in place or returned |
| `drop_dot_copy` | `OptFom_Update_BEST_Post_Optimize:148` | `dfe_SBRcursors` is read once, in `dfe_SBRcursors - BEST.DFE_taps_mV`, which allocates before the slice assignment writes `BEST.sampled_sbr_postcursors`; nothing between writes that buffer (`dfe_clipper` writes only its own copy, which its test now guards) |
| `drop_dot_copy` | `OptFom_Compute_DFE:79` | `dfecursors_q` goes only to `dfe_clipper`, which writes only its own copy (guarded by its test), and is not returned |
| `drop_dot_copy` | `applyDFEbk:38` | `flt_curval` is last read in the `tap_coef` line, which allocates, before `hisi[sl]` is written |
| `drop_dot_copy` | `floatingDFE:19` | `hisi[tap_loc]` indexes with an integer array, which always allocates; the copy is redundant |
| `drop_dot_copy` | `auto_port_order:54` | `LowFreq_Matrix` is rebound by a subtraction, never written in place, so `Raw_LowFreq_Matrix` is the same with or without the copy |
| `drop_dot_copy` | `auto_port_order:106` | `Floc` is only rebound by slicing and read; nothing writes it and it is not returned |
| `drop_dot_copy` | `comb_fct:33`, `comb_fct:34` | `p1_y` and `p2_y` are never written in place (the padding writes a new array), and the output `p.y = p1_y + p2_y` allocates |
| `drop_dot_copy` | `SNDR_ref:93` | `PR_noFFE` is never written, `FFE` writes nothing, and nothing derived from it by slicing is returned: `results` holds fresh arrays and floats |
| `drop_dot_copy` | `read_ParamConfigFile:615` | The branch runs only when `param.bmax` is empty, so no element write through either name can be observed |
| `drop_dot_copy` | `get_TDR:257` | `PTDR.pulse_orig` is never read after it is set, and `PTDR.pulse` is only rebound (`PTDR.pulse = PTDR.pulse * fctrx` allocates) |
| `drop_dot_copy` | `Bathtub_Contribution_Wrapper:41`, `plot_bathtub_curves:43` | `cursors_r.y`'s copy. `cursors` is not read after `cursors_r` is made and `cursors_l.y` was copied before, so the in-place zeroing through the shared array reaches nothing that is read again. `cursors_l.y`'s copy is NOT redundant and is caught |
