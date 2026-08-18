# COM Python Port — Feature & Verification Status

> **Status as of 2026-07-01 — feature inventory only; the verification claims below
> are superseded by [`../MATLAB_Correlation_Review.md`](../MATLAB_Correlation_Review.md)
> (2026-08-17). Unit-test count is now 876, and end-to-end parity is established
> against 208 MATLAB reference cases rather than the single bundled config.**
>
> **Reference: `com_ieee8023_4p15p0.m` + Hansel D'silva's
> `com_ieee8023_4p15p0_adaptive_local_search.m` branch.**
> Source-of-truth = `assemble_com.py` + `com_functions/fn/*/py_impl.py` (edit py_impl,
> run its `test_verify.py`, re-run `assemble_com.py`).

## TL;DR
**Feature-complete for the standard FD-based COM flow, and past the stub stage.** Every
MATLAB 4p15p0 function has a Python implementation (157 registry functions; `LFSR` is
inlined inside `PRBS13Q`). **876 unit tests pass (0 fail)**; the bundled 802.3ck C2M
config runs end to end. (It produced Case 1 = 3.5664 dB, Case 2 = 3.0184 dB when this
was written; the August 2026 engine fixes moved those to 3.4694 and 2.9331 dB.) The items that were "true stubs" or "unwired dispatch" in
the previous revision of this doc (RxFFE, floating taps, causality, Rx quantization,
TD-ILN, COM pie plot) are **now implemented and wired** — see §A/§B.

**The remaining work is verification, not implementation:**
1. **No bit-for-bit MATLAB numeric parity** yet (no MATLAB reference run was available).
   All verification to date is translation fidelity + unit tests + internal consistency +
   e2e stability. This is the single biggest open item — see §E.
2. **A large implemented-but-unexercised feature surface** (crosstalk, MLSE, RxFFE,
   floating DFE, ERL, FD ICN/ILD, calibration, modal masks, quantization) — code + unit
   tests exist, but no config/channel in-repo triggers them end-to-end — see §D.

---

## A. Former "true stubs" — NOW IMPLEMENTED + UNIT-TESTED
| # | Function | Feature | Gated by | Status |
|---|---|---|---|---|
| A1 | `OptFom_Compute_RxFFE` | Rx FFE optimization dispatch | `OP.RxFFE` | ✅ impl + test; dispatched from `optimize_fom` |
| A2 | `FOM_rxffe_floating_taps` | Rx FFE floating taps | `RXFFE_FLOAT_CTL` | ✅ impl + test |
| A3 | `calculate_delay_CausalityEnforcement` | Causality enforcement delay | `OP.ENFORCE_CAUSALITY` | ✅ impl + test |

## B. Former "wiring gaps" — NOW WIRED (or intentional non-gaps)
| # | Site | Status |
|---|---|---|
| B1 | `Apply_EQ` Rx-FFE branch | ✅ WIRED — calls real `force(eq_pulse, param, OP, t_s, fom_result.RxFFE)` (py_impl L112). *(Header docstring saying "force() not yet implemented" is stale.)* |
| B2 | `Create_Noise_PDF` N_qb branch | ✅ WIRED — calls `adjust_Rx_noise_for_quantization` when `param.N_qb != 0` (py_impl L267-270). |
| B3 | `MMSE` floating sub-branch | ✅ `MMSE` (218 lines) + `FOM_rxffe_floating_taps` implemented + tested. |
| B4 | `force` WIENER-HOPF sub-path | ⛔ **Intentional non-gap.** Raises `NotImplementedError` — `WIENER_HOPF_MMSE` is *undefined in the reference MATLAB itself*. Use `FFE_OPT_METHOD='MMSE'`. |
| B5 | `get_ILN_cmp_td` TD-ILN | ✅ WIRED — calls the real `s21_to_impulse_DC` for non-zero sdd21 (py_impl L180-192). *(Its "raises NotImplementedError" docstring is stale.)* |
| B6 | `RILN_TD` s21→impulse (non-zero IL) | ⚠️ **Only remaining incomplete code.** `_RILN_TD__s21_to_impulse_DC` still raises for non-zero IL (interp_Sparam not wired into this copy). **But `RILN_TD`/`get_RILN_cmp_td` are translated-but-unwired** — the live `COMPUTE_RILN` path in `FD_Processing` uses `capture_RIL_RILN`, so this raise is unreachable from a normal CLI run. Fix = point it at the real top-level `s21_to_impulse_DC` (as `get_ILN_cmp_td` already does), *if* that RILN route is ever needed. |
| B7 | `plot_pie_com` | ✅ Implemented + test; wired into `Bathtub_Contribution_Wrapper`. |
| B8 | `get_pdf_from_sampled_signal` FAST path | ⏸ `FAST_NOISE_CONV` speed *approximation* only (default exact path recommended). Perf, not correctness. |

## C. NON-GAPS (intentional — not needed for file-driven runs)
- `get_TD_files` / `get_s4p_files` GUI file-selection branches — raise only on empty
  `file_list`; a `file_list` is always provided on the CLI.
- `read_p4_s4params` — the `NotImplementedError` text is in a **comment**; the function is implemented.
- `read_package_parameters` — the "raise NotImplementedError" text is in a **stale comment**;
  the body is a working translation.

## D. IMPLEMENTED BUT UNEXERCISED (the verification surface)
Code + unit tests exist, but no in-repo config/channel triggers these end-to-end, so their
integration behavior and numeric correctness are unconfirmed. Each needs a config/channel
that exercises it **plus** a MATLAB reference comparison (§E).

| Feature | Function(s) | How to exercise |
|---|---|---|
| **Crosstalk (FEXT/NEXT), ICN** | `get_xtlk_noise`, FD_Processing | Run with aggressor `.s4p` files (`--fext`/`--next`). ✅ Now **runs end-to-end** (dj KR config + akinwale_3dj aggressors; 2 bugs fixed 2026-07-02, see below). Numeric parity still unverified. |
| **FD ICN / ILD / FOM_ILD** | `FD_Processing` | Config with `GET_FD=1` (bundled config → `FOM_ILD=[]`) |
| **MLSE** | `MLSE_U1_c_178A` | Config with `OP.MLSE=1` (exercised by the dj crosstalk run above) |
| **Rx FFE (+ floating taps)** | `OptFom_Compute_RxFFE`, `MMSE`, `force` | Config with `OP.RxFFE=1` + `FFE_OPT_METHOD='MMSE'`. ✅ Ran end-to-end in the dj crosstalk run (`RxFFE_with_MMSE` path). Numeric parity still unverified. |
| **Floating DFE** | `floatingDFE` | Config with `Floating_DFE=1` |
| **ERL / TDR** | `TDR_ERL_Processing`, `get_TDR` | Validate ERL against MATLAB (runs; unverified) |
| **RX_CALIBRATION / PSDRXCAL** | calibration loop, `get_PSDs`, `S_RN`/`S_IN` | Config with those flags |
| **Rx quantization noise** | `adjust_Rx_noise_for_quantization` | Config with `N_qb != 0` |
| **TD-ILN / RILN** | `get_ILN_cmp_td`, `capture_RIL_RILN` | Config with `COMPUTE_TDILN=1` / `COMPUTE_RILN=1` |
| **Common-mode / modal masks** | `plot_modal` | Config with `CM_MASK_REPORT=1` |
| **TDMODE (time-domain input)** | `get_TD_files`, `PRBS13Q` | `.s4p`-less TD waveform input (path not fully wired) |

## E. Cross-cutting: numeric parity (highest-value next step)
Without a MATLAB run of the same config, all verification is internal. To lock parity:
1. Run the bundled config in the MATLAB reference; export its per-case results.
2. Diff mechanically against the Python `results.csv` (same columns). `--export-mat` already
   emits the FD/TD intermediates (`H_*`, `h_*`, `pulse_*`, PDFs, metrics) to localize any gap.
3. Repeat per feature in §D with a config/channel that toggles the relevant flag.

---

## Implementation Plan (what's left)
**Phase 1 — Numeric parity (do first).** MATLAB golden run + `results.csv` diff on the
bundled config → converts "self-consistent" into "verified." Unblocks everything in §D.

**Phase 2 — Exercise the §D surface.** One config/channel per feature flag, diff vs MATLAB:
crosstalk (needs aggressor `.s4p`), MLSE, RxFFE, floating DFE, GET_FD ICN/ILD, ERL,
calibration, modal masks, N_qb, TD-ILN/RILN.

**Phase 3 — Loose ends.** B6 `RILN_TD` interp_Sparam corner (only if that RILN route is
needed); TDMODE input path; B8 `FAST_NOISE_CONV` perf path. B4 Wiener-Hopf only if the
MATLAB reference ever defines `WIENER_HOPF_MMSE`.

## Historical note
The initial (2026-06-07) audit logged 7 candidate bugs (BUG-01…07) in `audit_state.json`.
All 7 were re-checked against the 4p15p0 source on 2026-07-01 and found **unfounded** — the
translation matches the reference. See `audit_state.json` for the per-function disposition.

**2026-07-02 — crosstalk + RxFFE/MMSE path exercised for the first time; two real bugs found
and fixed:**
1. `OptFom_Compute_TXFFE` — `pulse_struc` was pre-sized to a 1-element list, but MATLAB struct
   arrays auto-grow when `pulse_struc(ii).field` is assigned. With `RxFFE_with_MMSE` and
   crosstalk (`ich = num_s4p_files > 1`) this raised `IndexError`. Fixed by extending
   `pulse_struc` to `ich` entries. Regression test: `test_multichannel_grows_pulse_struc`.
2. CLI print loop (`assemble_com.py` HEADER) did `enumerate(_results)`, but `com_ieee8023_`
   unwraps a single-package-case run to a bare struct (faithful to MATLAB
   `if length(results)==1, results = results{1}`). Single-case configs (e.g. this dj config,
   `pkg_len_select=[1]`) therefore crashed with `TypeError: SimpleNamespace not iterable`.
   Fixed by normalizing `_results` to a list before the per-case print. *(This also affected
   thru-only on any single-case config, not just crosstalk.)*

Both fixed; the dj KR config + akinwale_3dj aggressors now runs end-to-end and prints COM /
VEO / VEC / ICN. This is a crash/soundness fix — numeric parity vs MATLAB is still unverified.
