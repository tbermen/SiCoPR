# FIX SUMMARY, com_functions/ gated fix pass

Driven by `FIX_PROMPT_com_conversion_v2.md`. Fixes edit
`com_functions/fn/<name>/py_impl.py` (never `com.py`, which is a generated build
artifact reassembled by `assemble_com.py`). One finding per gate.

## Status: GATE BATCH-2 ASSEMBLED_VERIFIED (com.py reassembled + integration-tested)

Batch 2 = B01-D2 (get_PSDs ADC 'slow' clip) + B06-D9 (get_TDR s2p RL) -
ASSEMBLED_VERIFIED. assemble_com.py: 157 functions, 5 stubs (baseline match).
Integration: B01-D2 (adc_clip_slow + ctle_signal_sigma) and B06-D9 (s2p_RL) audit
checks flipped FAIL->PASS; fn suite 874 pass / 0 fail; smoke + e2e exit 0.
**get_PSDs fully resolved** (D1/D2 fixed, D3 kicked back). One stale guard flagged
(get_TDR_s2p_RL_is_the_wrong_formula, asserted the old bug). Next: F6 get_pdf_full
(B08-D12), or the always-on bugs B16-D20 / B13-D18.

---

## Prior: GATE BATCH-1 ASSEMBLED_VERIFIED (com.py reassembled + integration-tested)

Revision 4p15p0, dependency order. Batch 1 = B03-D6/D7, B02-D4/D5, B01-D1 -
ASSEMBLED_VERIFIED. `assemble_com.py`: 157 functions, 5 stubs (baseline match).
Integration: the audit "matches-MATLAB" checks for all three flipped FAIL->PASS
(s21 causality, interp_Sparam phase+mag, get_PSDs S_jn LIMIT); com_functions/fn
suite 872 pass / 0 fail (865 baseline + 7 new); tests/test_smoke.py and
tests/test_end_to_end.py exit 0. B01-D3 KICKED_BACK (unreachable). B01-D2 deferred
to its own gate.

Two audit divergence-documentation guards are now stale/invalid and flagged for
replacement (NOT edited): `s21_grid_round_half_away_from_zero` (asserts Python's
builtin `round`, not com.py) and `interp_Sparam_phase_extrap_is_flat` (asserts the
old clamped phase). The real B03-D7/B02-D5 fixes are verified by the fn tests.

Next: B01-D2 (own gate) then F4 get_TDR (B06-D9), or as you direct.

### Gate 0 findings

1. **MATLAB revision mismatch (Section 2 gate).** `registry.json` `matlab_lines`
   are indexed to **4p14p0**; the audit ledger and this pass cite **4p15p0** (the
   source-of-truth `.m` present, and the revision the audit re-derived against).
   Offset grows from +5 (~line 2585) to +130 (past line 7000) because 4p15p0
   added content. All 11 DIVERGENT functions are version-identical (4p14==4p15
   body, audit B13), so re-derivation is valid from either; 4p15p0 is chosen for
   citation consistency. `registry.json` will NOT be edited (dirs are name-keyed;
   the line base is metadata only). **Recommendation: confirm 4p15p0 authoritative.**

2. **Baseline build is not the prompt's literal "clean".** `python assemble_com.py`
   → "Assembled com.py from 157 functions. WARNING: 5 function(s) still contain
   NotImplementedError." The 5 are **deliberate stubs** for non-portable/undefined
   paths (`force` WIENER-HOPF [undefined in MATLAB], `get_s4p_files`/`get_TD_files`
   GUI file-picker, `read_package_parameters` cross-import, `RILN_TD` inlined
   non-zero-IL). All 5 functions' fn tests PASS. This is the recorded baseline
   reference (not a blocker; none are in the fix queue).

3. **git unavailable** → reversibility via `.bak` fallback (copy `py_impl.py` to
   `py_impl.<finding>.bak` before editing).

4. Test convention: `com_functions/fn/<name>/test_verify.py`, pytest, imports
   `py_impl` directly, `SimpleNamespace` inputs, analytic/invariant assertions.

### Fix queue (dependency-ordered, Section 5): 11 DIVERGENT functions

| Rank | Finding | Function | Stage | Trap | Audit recommendation (hypothesis) |
|---|---|---|---|---|---|
| 1 | B03-D6/D7 | s21_to_impulse_DC | FD->TD impulse (high fan-in) | index-base | fix causality window; half-away grid round |
| 2 | B02-D5/D4 | interp_Sparam | S-param interp (high fan-in) | interpolation/eps | linear phase extrap; eps=finfo.eps |
| 3 | B01-D1/D2/D3 | get_PSDs | noise PSD (high fan-in) | index/conditional | jitter +1; ADC-slow clip; stub defaults |
| 4 | B06-D9 | get_TDR | TDR/ERL | mldivide-vs-divide | RL=(s11-rho)/(1-rho*s11) |
| 5 | B06-D11 | TDR_ERL_Processing | TDR/ERL | cosmetic | file_names from first base |
| 6 | B08-D12 | get_pdf_full | C2M PDF | resample grid | match MATLAB grid |
| 7 | B13-D18 | get_xtlk_noise | crosstalk noise | argmax slice off-by-one | index_f2 = argmax+1 / len(f) |
| 8 | B12-D17/D16 | MMSE_FOM | optimizer MMSE | conditional recompute | blim recompute inside w!=wlim |
| 9 | B16-D20/B11-D15 | optimize_fom | optimizer driver | cursor off-by-one | drop the -1 at A_s/A_p/far/pre |
| 10 | B15-D19 | MLSE | post-processing | precedence | sigma_noise to numerator (diagnostic-only) |
| 11 | B05-D8 | s_for_c4 | cascade (UNUSED) | port reorder | apply snp2smp([1 3 2 4]) |

Cross-cutting (deferred, separate decision): **B10-D14** rounding (~50 bare-round
sites, measure-zero; half-away sweep vs accept).

**Recommended first fix:** rank 1, **s21_to_impulse_DC (B03-D6)** - upstream-most,
highest fan-in, so downstream fixes build on corrected TD impulse. (If you prefer
impact-ordering instead of dependency-ordering, the highest always-on COM impact
is B16-D20 optimize_fom and B13-D18 get_xtlk_noise.)

## Fixed (UNIT_VERIFIED, pending reassembly)

| Finding | Function | MATLAB (4p15p0) | Edit | Unit result |
|---|---|---|---|---|
| B03-D6 + B03-D7 | s21_to_impulse_DC | 11232, 11260-11261, 11267 | causality window `[:start_ind]`->`[:start_ind+1]`, `[half:]`->`[half-1:]`, signed `max`; fout `round`->half-away `_mround` | 9 PASS (2 new + 7 pre-existing); 38/38 neighbor tests PASS |
| B02-D4 + B02-D5 | interp_Sparam (+ inlined copy in s21_to_impulse_DC) | 8091, 8125, 8185 | `|S|` floor + HF threshold use machine `eps` (added `eps_mag`; log guards left at `tiny`); phase clamp -> linear end-slope extrapolation | 20 PASS (4 new + F1 + pre-existing); 53 neighbor tests PASS |
| B01-D1 | get_PSDs | 6676-6677 | LIMIT_JITTER early/late sampling centered at cursor: `cursor_i+M*k`->`cursor_i-1+M*k`, `cursor_i+2+M*k`->`cursor_i+1+M*k` | 8 PASS (1 new + 7 pre-existing); 44 neighbor tests PASS |
| B01-D2 | get_PSDs | 6716-6725 | ADC 'slow' clip: inlined get_pdf_from_sampled_signal/conv_fct/CDF_inv_ev; adc_clip = P_qc quantile of signal+noise PDF; set ctle_signal_sigma (else sum(abs) unchanged) | 9 PASS (1 new + 8 pre-existing); 40 neighbor tests PASS |
| B06-D9 | get_TDR | 7080-7081 | s2p RL: `interim/(s11-rho)` -> `(s11-rho)/interim` (MATLAB left-division; interim cancels -> `(s11-rho)/(1-rho*s11)`); s4p path untouched | 7 PASS (1 new + 6 pre-existing); 41 neighbor tests PASS |

Reach: propagates to mainline via the shared `s21_to_impulse_DC` (`COM_FD_to_TD`,
`get_TDR`, `get_RAW_FIR`, `calculate_delay_CausalityEnforcement`, `TD_FD_fillin`,
`RILN_TD`, `get_ILN_cmp_td` all call it). Backup: `py_impl.B03-D6-D7.bak`.

## Kicked back / version-delta / surfaced-new
- **B01-D3 (get_PSDs injection stub defaults) — KICKED_BACK** (human-approved).
  The `_S_RN_fn`/`_S_IN_fn`/`_H_interp_fn` stub defaults do diverge, but the
  assembled pipeline always injects the real functions via `_run_com`, so no
  reachable `com.py` result is affected. It cannot be fixed in `get_PSDs`'s
  `py_impl` (no cross-`py_impl` sibling imports; the stubs exist for standalone
  testability). Recorded as a latent API-robustness caveat, not a conversion bug.
- **B01-D2 (get_PSDs ADC 'slow' clip)** — deferred to its own later gate
  (substantial port; conditional on `N_qb!=0 & clip_method=='slow'`).
- Observation (not a finding): `RILN_TD` and `get_ILN_cmp_td` contain a **dead**
  inlined `_s21_to_impulse_DC` helper (raises for non-zero IL); it is unused - their
  real path calls the shared `s21_to_impulse_DC` - so the fix reaches them and no
  separate fix is required.
