# AUDIT FINDINGS, sicopr.py vs com_ieee8023_4p15p0.m

> **SUPERSEDED IN PART (2026-08-17).** This document records the static
> code-reading audit. It has since been overtaken by the 208-case MATLAB
> correlation — see [`../MATLAB_Correlation_Review.md`](../MATLAB_Correlation_Review.md),
> which is authoritative where the two disagree.
>
> What changed: eight engine defects were fixed (commit `git log --grep="Fix 8 engine defects"`), including
> **B16-D20, which this document rated "medium" and which a later end-to-end
> trial wrongly appeared to refute** — it was compensating a second defect in the
> die-network path, and both had to be fixed together. B12-D17 and B06-D9 are
> also fixed. The largest defect of all is **not in this ledger**: a
> by-reference `OP` leak in `process_sxp` that biased FOM low on 95.7% of cases,
> which line-by-line reading never caught.
>
> Treat the EQUIVALENT verdicts below as evidence, not proof. Twelve divergences
> remain open and accepted; they are marked `xcheck` in `tests/` and print as
> `XFAIL`.

Status: AUDIT COVERAGE COMPLETE (batches B01-B21). The gate this once waited at never ran -- the pass was overtaken by the 208-case correlation, as the banner above says. Nothing here is outstanding work.
All 157 MATLAB functions + 2 cross-cutting scans are now classified: 0
NOT_YET_AUDITED. B01-B12 covered the risk-ordered work queue (section 3, groups
G1-G10) with bespoke multi-angle tests; B13 resolved the 4p14p0->4p15p0 version
question for the whole file (101 byte-identical bodies; auto-port-order +
cursor_tap already in sicopr.py); B14-B21 completed the lower-risk remainder.

146 functions EQUIVALENT, 11 DIVERGENT functions (get_PSDs, interp_Sparam,
s21_to_impulse_DC, s_for_c4, get_TDR, TDR_ERL_Processing, get_pdf_full, MMSE_FOM,
get_xtlk_noise, MLSE, optimize_fom); the rounding cross-cutting scan is
DIVERGENT-low (B10-D14) and the reshape/order scan is EQUIVALENT.

Evidence level per function (honest disclosure): the risk-ordered functions
(B01-B12) and the numerical hotspots got bespoke tests and/or deep line-cited
reads. The lower-risk tail was closed with, per function, one of: a deep read; a
line-cited delegation argument (thin wrappers over already-audited primitives); a
version-neutral + green-fn-test method-verdict (8 numerical-tail functions -
MLSE_U1_c_178A, capture_RIL_RILN, RILN_TD, Output_Arg_Fill, Burst_Probability_Calc,
Bathtub_Contribution_Wrapper, parameter_size_adjustment, Bread_Crumb_Chdata_Reduction
- NOT exhaustively line-verified; residual translation risk disclosed); or a
line-cited pass-through (cosmetic/output + config/IO). The `audit_method` field in
audit_state.json records which applies to each. The whole file is version-neutral
(4p14==4p15 or 4p15-present) and every function has a green com_functions/fn test
(865 pass / 0 fail), but note that green fn tests did NOT catch B13-D18, B15-D19,
or B16-D20 - spot-checks/deep-reads did.

Divergences by severity: medium - B16-D20 (optimize_fom cursor/peak off-by-one in
A_s/A_p/ISI windows), B12-D17 (MMSE_FOM DFE-clip blim), B06-D9 (get_TDR s2p),
B03-D6 (s21 causality window), B02-D5/D4 (interp_Sparam), B01-D1/D2/D3 (get_PSDs);
low/edge - B13-D18 (get_xtlk_noise crosstalk band edge), B15-D19 (MLSE Gaussian
diagnostic only, no COM impact), B03-D7 (s21 grid round), B05-D8 (s_for_c4
unused), B08-D12 (get_pdf_full C2M), B10-D14 (rounding), B12-D16 (MMSE_FOM
subset-gate edge), B11-D15 (optimize_fom round).

B13 established version-neutrality for the remainder (101 byte-identical 4p14==4p15
bodies; auto-port-order + cursor_tap already in sicopr.py) and every function has a
green com_functions/fn test (865 pass / 0 fail). CAVEAT: a green fn test does NOT
prove full translation equivalence - spot-checks found real bugs in
"version-identical, fn-test-green" functions (get_xtlk_noise B13-D18; MLSE
B15-D19), so the remaining 67 still get genuine per-function reads (numerical
ones) or line-cited pass-through (mechanical ones).

Evidence:
tests/test_noise_units.py (42 checks, 37/5), tests/test_freq_grid.py (27 checks,
25/2), tests/test_impulse_spectrum.py (18 checks, 16/2), tests/test_conv_dfe.py
(21 checks, 21/0), tests/test_sparam_cascade.py (18 checks, 17/1),
tests/test_tline_tdr_erl.py (18 checks, 16/2), tests/test_pdf_pipeline.py (23
checks, 23/0), tests/test_noise_pdf_composite.py (18 checks, 17/1),
tests/test_cursor_indexing.py (19 checks, 19/0),
tests/test_rounding_reshape.py (14 checks, 13/1),
tests/test_optimizer_fom.py (33 checks, 32/1),
tests/test_optimizer_mmse.py (26 checks, 25/1), and
tests/test_xtlk_noise.py (5 checks, 3/2); every FAIL is deliberate divergence
documentation (results.csv, 2026-07-09/10/11). The com_functions/fn per-function
suite (865/0) backs the assembled sicopr.py bodies.

The historical double-DFE-subtraction concern is now closed: get_pdf applies the
DFE cancellation exactly once (B07, test-pinned) and get_PSDs S_isi applies it
once (B01, test-pinned).

## Top-risk open items

### B16-D20. optimize_fom reads one sample before the cursor/peak (A_s, A_p, ISI windows)

- MATLAB: com_ieee8023_4p15p0.m lines 8788-8801. `cursor_i` is 1-based (from
  `cursor_sample_index`), so `cursor = sbr(cursor_i)`, `A_p = sbr(sbr_peak_i)`,
  and the far/precursor colon windows read the true cursor/peak with no offset.
- Python: sicopr.py `optimize_fom` subtracts 1 at every one of these sites -
  `cursor = sbr[THIS.cursor_i - 1]` (12722), `A_p = sbr[sbr_peak_i - 1]` (12723),
  `far_start = cursor_i - T_O + M*(ndfe+1) - 1` (12731), `pre_start = cursor_i - M
  - 1` (12735) - but `OptFom_Find_Sample_Point` / `cursor_sample_index` already
  return 0-based indices (the convention used by the main pipeline `get_PSDs`
  (B01) and `get_pdf` (B07), and by `OptFom_Compute_DFE`, which correctly uses
  `sbr[cursor_i]`). The `-1` therefore double-subtracts.
- Consequence: `A_s` (main signal; COM/FOM = 20log10(A_s/noise)), `A_p`, and the
  far/precursor ISI windows are all taken ONE SAMPLE BEFORE the true cursor/peak,
  while the DFE (`OptFom_Compute_DFE`) is anchored to the correct sample - an
  internal inconsistency within each optimisation iteration. This biases the FOM
  and hence which EQ setting is selected, plus the reported `A_s`/`A_p`. Magnitude
  is a one-sample pulse droop at the cursor (small where the pulse is flat at the
  cursor, larger on a steep flank). Confirmed by tests/test_optimize_cursor.py:
  `OptFom_Find_Sample_Point` returns the 0-based peak (sbr[peak]=0.800=max) yet
  optimize_fom's `A_p = sbr[peak-1] = 0.784`.
- This SUPERSEDES the B11 EQUIVALENT verdict for optimize_fom. The B11 far/
  precursor tests were standalone formula reproductions that assumed a 1-based
  cursor_i, so they did not exercise the real 0-based return of
  cursor_sample_index. (Lesson: verify the index BASE end-to-end, not just the
  slice formula.)
- Recommended fix (not applied): remove the `-1` at sicopr.py 12722/12723/12731/
  12735 so A_s/A_p and the ISI windows use `sbr[cursor_i]`/`sbr[sbr_peak_i]`
  (0-based), matching OptFom_Compute_DFE and the main pipeline.

### B13-D18. get_xtlk_noise crosstalk ICN sum omits one frequency bin at fb

- MATLAB: com_ieee8023_4p15p0.m line 7976,
  `index_f2 = find(chdata(1).faxis(:) > param.fb, 1, 'first')` is a 1-based
  index, and lines 8051/8054 sum `... PWF(1:index_f2) ...`, which INCLUDES the
  first frequency bin above `fb`. The empty case (no bin above `fb`) sets
  `index_f2 = length(faxis)` (all bins).
- Python: sicopr.py line 11553 (`get_xtlk_noise`) sets
  `index_f2 = argmax(f > fb)` (0-based) and slices `PWF[:index_f2]`, which
  EXCLUDES the first bin above `fb`; the empty case uses `len(f)-1`, omitting the
  last bin.
- Consequence: the crosstalk ICN power sums `MDFEXT_ICN` / `MDNEXT_ICN`
  (eq 93A-46/47) are short by one frequency bin at the `fb` band edge, so
  crosstalk noise is slightly UNDER-estimated and COM slightly OPTIMISTIC
  whenever FEXT/NEXT aggressors are present. Reproduced on a synthetic single
  FEXT aggressor: sicopr.py sigma_FEXT 0.363781 vs MATLAB-correct 0.363824, with the
  gap equal exactly to the omitted first-bin term; empty case 0.360099 vs
  0.360942. Small per-run magnitude but systematic and in the optimistic
  direction.
- Evidence: FAIL xtlk_MDFEXT_ICN_matches_matlab_upper_bound and FAIL
  xtlk_MDFEXT_ICN_empty_case_matches_matlab in tests/test_xtlk_noise.py, with
  PASS xtlk_gap_equals_first_bin_above_fb confirming the mechanism.
- Recommended fix (not applied): `index_f2 = argmax(f > fb) + 1` (and
  `len(f)` for the empty case) so the sum includes the first bin above `fb`.

### B15-D19. MLSE Gaussian equivalent-SNR arg divides where MATLAB multiplies (diagnostic-only)

- MATLAB: com_ieee8023_4p15p0.m line 2311, inside the Gaussian equivalent-SNR
  step, `qfunc((1-2*alpha)*main/(L-1)*sigma_noise)`. MATLAB `*`/`/` are equal
  precedence and left-associative, so `sigma_noise` ends up in the NUMERATOR:
  `(1-2*alpha)*main*sigma_noise/(L-1)`. (The commented reference at line 2295 uses
  `/sigma_noise`, so 2311 looks like a MATLAB typo, but the audit reproduces MATLAB.)
- Python: sicopr.py line 2749, `_MLSE__qfunc((1-2*alpha)*A_peak/((L-1)*sigma_noise))`
  puts `sigma_noise` in the DENOMINATOR - differing from MATLAB by a factor
  `sigma_noise^2`.
- Consequence: DIAGNOSTIC ONLY. It changes the reported
  `MLSE_results.SNR_DFE_eqivalent_Gaussian` and `delta_com_Gaussian`, but NOT any
  COM value: MATLAB sets both `COM_Gaussian` and `COM_CDF` to `new_com_CDF` (the
  CDF path, lines 2345-2346) and sicopr.py does the same (2778-2779). The CDF path,
  the `j=1:200` DER_MLSE sum, and the CDF convergence loop all match.
- Evidence: FAIL mlse_gaussian_qfunc_arg_matches_matlab (MATLAB arg 2.667e-4 vs
  sicopr.py 6.667e-1, ratio = sigma^2) and PASS mlse_COM_Gaussian_equals_COM_CDF in
  tests/test_mlse.py.
- Recommended fix (not applied): to reproduce MATLAB exactly, multiply by
  `sigma_noise` at sicopr.py 2749; or leave as-is since it is diagnostic-only.

### B12-D17. MMSE_FOM recomputes the DFE tap limit unconditionally (DFE-clip path)

- MATLAB: com_ieee8023_4p15p0.m lines 2683-2692. The final DFE recompute
  `b = Hb*wlim; blim = min(bmax, max(bmin, b))` runs ONLY inside
  `if ~isequal(w, wlim)` (i.e., only when the RxFFE taps were clipped and wlim
  renormalised). If the RxFFE taps are within limits (w == wlim), MATLAB keeps
  the `blim` from the earlier DFE clip (line 2670 / 2687), i.e.
  `clip(original DFE taps)`.
- Python: sicopr.py lines 3367-3369 (top-level MMSE_FOM) and 3119-3121
  (hoisted `_MMSE__MMSE_FOM`) run `b_upd = Hb @ wlim; blim = clip(b_upd)`
  UNCONDITIONALLY whenever `Nb > 0`, outside the `if not allclose(w, wlim)` block.
- Consequence: when the DFE taps are clipped (the first re-solve at 2671-2674
  fires) but the RxFFE taps are not (block 2 quiet, w == wlim), MATLAB returns
  `blim = clip(original DFE)` while Python returns `blim = clip(Hb @ w_resolved)`.
  These differ, changing the returned DFE tap coefficients (used by MLSD, eq
  178A-39), `sigma_e`, and the MMSE `FOM`. Reproduced on a synthetic solve at
  `bmax = 0.0289`: Python `blim = [0.0289, 0.0274]` vs MATLAB `[0.0289, 0.0289]`,
  FOM 7.9898 vs 7.9895 dB (the FOM gap is small in this synthetic case but the
  returned DFE taps differ materially). Active only on the RxFFE-with-MMSE path
  (`OP.RxFFE` and `OP.FFE_OPT_METHOD == 'MMSE'`).
- Evidence: FAIL mmse_fom_D17_blim_matches_matlab in tests/test_optimizer_mmse.py,
  with PASS mmse_fom_D17_scenario_found (a DFE-clip-only bound exists) and PASS
  mmse_fom_D17_python_uses_Hb_times_w (Python's blim == clip(Hb*w_resolved)). The
  no-clip path is verified to match an independent MATLAB reference to 1e-9.
- Recommended fix (not applied): move `b = Hb @ wlim; blim = clip(b)` inside the
  `if not np.allclose(w, wlim):` branch at sicopr.py 3361/3114, matching MATLAB.

### B06-D9. get_TDR s2p reflection renormalisation inverted (mldivide mistranslated)

- MATLAB: com_ieee8023_4p15p0.m line 7080,
  `RL(i) = interim \ (s11 - rho) / (1 - rho*s11) * interim`. The `\` is
  left-division, so `interim \ X` = `X/interim` and the `interim` factors
  cancel, leaving the standard bilinear renormalisation
  `RL = (s11 - rho) / (1 - rho*s11)`.
- Python: sicopr.py line 10304 uses `/`:
  `interim / (s11 - rho) / (1 - rho*s11) * interim`
  = `interim^2 / ((s11 - rho) * (1 - rho*s11))`, the reciprocal structure.
- Consequence: for a matched reference (rho=0) MATLAB returns `s11` while Python
  returns `1/s11`; on the test s2p this produced a reflection magnitude of 5.76
  (a reflection coefficient above 1 is non-physical). This corrupts ERL computed
  from `.s2p` differential-return-loss inputs. The common `.s4p` 4-port path uses
  a different, correctly transcribed formula and is unaffected.
- Evidence: FAIL get_TDR_s2p_RL_matches_matlab in tests/test_tline_tdr_erl.py,
  with a companion PASS confirming the output matches the diagnosed wrong formula
  and a PASS showing the s4p matched line gives avgZport = 2*ZT.
- Recommended fix (not applied): `RL[i] = (s11_i - rho) / (1 - rho*s11_i)`.

### B03-D6. s21_to_impulse_DC causality window off by one (impulse response error)

- MATLAB: com_ieee8023_4p15p0.m lines 11256, 11260-11261. The
  Alternating-Projections causality step zeros `impulse_response(1:a(1))`
  (inclusive of the first above-threshold sample) and
  `impulse_response(floor(L/2):end)`.
- Python: sicopr.py line 17040 sets `start_ind = candidates[0]` (0-based) and zeros
  `impulse_response[:start_ind]`, one fewer leading sample; line 17045 zeros
  `impulse_response[half:]` with `half = L//2`, omitting the 0-based
  `floor(L/2)-1` sample MATLAB zeros. The Python causal window keeps two boundary
  samples that MATLAB removes.
- Consequence: measured 1.7 percent relative error in the enforced impulse
  response and a 0.78 dB gap in `causality_correction_dB` (Python -29.69 vs
  MATLAB -28.91) on a rippled test channel. `causality_correction_dB` is
  reported in every run, including the default `ENFORCE_CAUSALITY=0`; the
  impulse response that feeds all downstream COM math diverges whenever
  `ENFORCE_CAUSALITY=1`. A secondary difference: py 17050 uses
  `max(abs(ir))` in the convergence ratio where MATLAB line 11267 uses the
  signed `max(ir)` (they agree while the peak is positive, which is the norm).
- Evidence: FAIL s21_causality_window_indexing_matches_matlab in
  tests/test_impulse_spectrum.py. The default (ENFORCE off) path is verified to
  equal the plain Hermitian ifft (PASS s21_default_path_unaffected_by_apm).
- Recommended fix (not applied): zero `impulse_response[:candidates[0]+1]`, zero
  the second half from `[half-1:]`, and divide the error by the signed max.

### B03-D7. s21_to_impulse_DC fout grid uses banker's rounding

- MATLAB: line 11232 builds the grid with `round(fmax/freq_step)`
  (half-away-from-zero).
- Python: line 17009 uses the builtin `round()` (round-half-to-even).
- Consequence: at a half-integer ratio (for example 200.5) Python yields 200 and
  MATLAB 201, changing the grid-point count and therefore the entire impulse
  response length and content. Low probability but a real rounding trap.
- Evidence: FAIL s21_grid_round_half_away_from_zero.
- Recommended fix (not applied): use a half-away-from-zero round.

### B02-D5. interp_Sparam phase held flat where MATLAB extrapolates it

- MATLAB: com_ieee8023_4p15p0.m line 8185,
  `H_ph_i = interp1(fin, H_ph, fout, 'linear', 'extrap')` extrapolates phase
  linearly beyond the source band.
- Python: sicopr.py line 11855 uses `np.interp(fout, fin, H_ph, left=H_ph[0],
  right=H_ph[-1])`, which CLAMPS the phase to the endpoint value.
- Consequence: for the default phase method
  `extrap_cubic_to_dc_linear_to_inf` on a DC-referenced grid (fin[0]==0), the
  code falls through to this clamped base, so phase beyond fin[-1] is frozen
  instead of continuing the group-delay slope. The measured error is 62.8 rad
  (2*pi*10) at 30 GHz for a 1 ns delay extended from 20 to 30 GHz. Active only
  when the analysis grid extends past the touchstone fmax AND ZERO_PAD is off
  (default off). For fin[0] != 0 the default method builds its own linear HF
  tail (py 11932-11938) and does not diverge; `trend_and_shift_to_DC` also
  handles it. The default magnitude method `linear_trend_to_DC` pre-extends the
  interp grid and is safe (verified PASS).
- Evidence: FAIL interp_Sparam_phase_extrap_matches_matlab in tests/test_freq_grid.py.
- Recommended fix (not applied): replace the clamped base phase interp with
  genuine linear extrapolation (first/last two-point slope) for fout < fin[0]
  and fout > fin[-1].

### B02-D4. interp_Sparam magnitude floor uses the wrong constant

- MATLAB: lines 8091 and 8125 use `eps` = 2.2204e-16 as the magnitude floor
  (`H_mag(H_mag<eps)=eps`) and as the `hf_trend_val<eps` threshold.
- Python: line 11756 sets `eps = np.finfo(float).tiny` = 2.2251e-308, so the
  floor and the threshold effectively never fire.
- Consequence: any |S| below 2.2e-16 (loss > 314 dB, e.g. synthesized nulls) is
  left un-floored, changing log-based magnitude fits (`trend_to_DC`) for those
  bins. Low severity for realistic channels. Note MATLAB's explicit `realmin`
  at line 8127 IS correctly mapped to `np.finfo.tiny` at py 11793; only `eps`
  was confused with `realmin`.
- Evidence: FAIL interp_Sparam_mag_floor_matlab_eps in tests/test_freq_grid.py.
- Recommended fix (not applied): `eps = np.finfo(float).eps`.

## Prior top-risk items (from B01, get_PSDs)

### B01-D1. get_PSDs jitter-noise sampling off by one sample (conditional path)

- MATLAB: com_ieee8023_4p15p0.m lines 6676-6677. With
  OP.LIMIT_JITTER_CONTRIB_TO_DFE_SPAN enabled, the early/late cursor samples
  are h(cursor_i-1+M*(-1:ndfe)) and h(cursor_i+1+M*(-1:ndfe)), one sample
  either side of the cursor.
- Python: sicopr.py lines 9673-9674 use idx_early = cursor_i + M*k and
  idx_late = cursor_i + 2 + M*k (0-based), which is one sample late on both.
- Consequence: the jitter sensitivity vector h_J is the pulse slope evaluated
  at cursor+1 sample instead of at the cursor, so S_jn and S_rj_jn (and the
  COM jitter penalty) are computed from the wrong slope whenever the flag is
  on. The flag is experimental and defaults to 0 in both codebases, so default
  runs are unaffected. Same-function non-LIMIT branch is correct (test PASS).
- Evidence: FAIL get_PSDs_S_jn_LIMIT_matches_matlab in tests/test_noise_units.py.
- Recommended fix (not applied): idx_early = cursor_i - 1 + M*k,
  idx_late = cursor_i + 1 + M*k. Also note the Python-only bounds masks can
  silently misalign early/late pairs where MATLAB would error.

### B01-D2. get_PSDs ADC 'slow' clip method not ported (conditional path)

- MATLAB: lines 6716-6725. When param.N_qb is nonzero and param.clip_method is
  'slow', adc_clip is the P_qc quantile of the signal-plus-noise PDF:
  build the sampled-signal PDF, convolve with a Gaussian noise PDF, and take
  -CDF_inv_ev(param.P_qc, ...). Also sets result.ctle_signal_sigma (line 6725).
- Python: lines 9721-9730 substitute a heuristic max(abs(pulse)) + 3*sigma and
  never set ctle_signal_sigma.
- Consequence: quantization noise S_qn is wrong whenever the slow clip method
  is selected. On the B01 synthetic channel the clip levels were 1.0071
  (Python) vs 1.0887 (MATLAB transcription), about 8 percent, which shifts
  sigma_Q quadratically. Downstream consumers of ctle_signal_sigma would hit
  an AttributeError.
- Evidence: FAIL get_PSDs_adc_clip_slow_matches_matlab and
  FAIL get_PSDs_slow_sets_ctle_signal_sigma. Caveat: the oracle uses sicopr.py's
  own get_pdf_from_sampled_signal/conv_fct/CDF_inv_ev as MATLAB stand-ins;
  those are audited in B07/B08.
- Recommended fix (not applied): port MATLAB 6717-6725 verbatim using the
  already-ported PDF helpers, and set result.ctle_signal_sigma.

### B01-D3. get_PSDs default dependency stubs diverge from MATLAB (API trap)

- MATLAB: lines 6551, 6565, 6567 call the real S_RN, H_interp, S_IN.
- Python: get_PSDs takes _S_RN_fn/_S_IN_fn/_H_interp_fn parameters whose
  defaults (lines 9443-9456) are stubs: a flat eta_0 PSD with no 1/2 factor
  and no CTLE or Bessel-Thomson shaping, an all-zeros S_IN, and a
  magnitude-only linear interp.
- Consequence: every in-repo call site injects the real functions (sicopr.py
  167-170, 195, 215, and 4622), so mainline results are unaffected. But any
  direct call to get_PSDs mirroring MATLAB usage silently produces a wrong
  S_rn and a zero S_in instead of failing.
- Evidence: FAIL get_PSDs_stub_SRN_agrees_with_S_RN and
  FAIL get_PSDs_stub_SIN_agrees_with_S_IN.
- Recommended fix (not applied): make S_RN, S_IN, H_interp the parameter
  defaults and delete the stubs.

## Batch B13 (2026-07-11): coverage completion - version delta + spot-checks

This pass addresses the 108 functions still NOT_YET_AUDITED after the
risk-ordered queue (G1-G10). All are lower-risk by construction. Rather than
rubber-stamp them, B13 established two independent facts and then spot-checked.

**1. The 4p14p0 -> 4p15p0 version question is resolved for all 108.** A
normalized (comment- and whitespace-insensitive) function-body diff of the two
MATLAB files shows:
- 101 functions have byte-identical bodies in 4p14p0 and 4p15p0.
- 4 base functions differ, all for one coherent 4p15 feature plus one fix, and
  sicopr.py already carries the 4p15 behavior (verified in source):
  `com_ieee8023_`, `read_p4_s4params`, `read_Nport_touchstone` implement the
  automatic port-order detection (`snpPortsOrder -> []`, `auto_port_order`
  inlined and called when the order is empty, `output_args.port_order`);
  `adjust_Rx_noise_for_quantization` uses `cursor_tap = ffe_pre_tap_len + 1`.
- 3 (`auto_port_order`, `append_csv_row`, `compute_hard_cap`) are identical
  (BOM-only diff) to their 4p15 / adaptive-variant source.

So auditing against 4p15p0 is, for these, the same as auditing against the
4p14p0 basis the port was written from.

**2. Every function has green executable per-function coverage.** sicopr.py is
machine-assembled (assemble_sicopr.py) verbatim from com_functions/fn/*/py_impl.py,
and the com_functions/fn test suite is green (865 passed / 0 failed), so those
tests exercise the exact bodies that end up in sicopr.py.

**3. Spot-check - and why the remainder is not blanket-cleared.** A green fn test
only covers what it exercises. Spot-checking get_xtlk_noise (version-identical,
fn-test-green) surfaced B13-D18 (above), a real off-by-one in the crosstalk ICN
band edge. This is the same lesson as B01/B03/B12: translation bugs hide in
untested paths. Therefore the 94 still-open functions keep the NOT_YET_AUDITED
status for their Python translation; each now carries a `version_delta` and
`fn_test` annotation in audit_state.json so a future batch is a focused
translation review, not a version+translation audit.

Functions genuinely resolved in B13 (14): EQUIVALENT - `com_ieee8023_`,
`read_p4_s4params`, `read_Nport_touchstone`, `adjust_Rx_noise_for_quantization`
(4p15 feature confirmed present), `auto_port_order`, `append_csv_row`,
`compute_hard_cap` (variant-identical), and the six filter leaves `FD_CTLE`,
`Butterworth_Filter`, `Bessel_Thomson_Filter`, `Raised_Cosine_Filter`,
`Tukey_Window`, `bessel` (cross-checked to 1e-12 in B11). DIVERGENT -
`get_xtlk_noise` (B13-D18).

Recommended next steps at Gate 13: (a) continue focused translation-review
batches over the 94 remaining (config parsing, S-parameter file IO, the OptFom_*
orchestration helpers, RILN/ILN metrics, plotting, and the top-level driver), or
(b) open a separate gated fix pass for D1-D18, or (c) accept the current coverage
with the documented caveat.

## Batch B12 verdicts (2026-07-11): G10 optimizer MMSE + floating taps

| Function | MATLAB | Python | Verdict | Traps | Tests |
|---|---|---|---|---|---|
| Full_Grid_Matrix | 2124-2184 | 2569-2593 | EQUIVALENT | 4 | 3 PASS |
| FOM_rxffe_floating_taps | 2068-2113 | 2487-2525 | EQUIVALENT | 1,10,14 | 2 PASS |
| MMSE | 2485-2584 | 3133-3257 | EQUIVALENT | 1,4,19 | seam PASS |
| MMSE_FOM | 2585-2697 | 3286-3385 | DIVERGENT (B12-D17) | 1,8,19 | 8 PASS / 1 FAIL |

Batch total: 26 checks, 1 FAIL (the FAIL is the deliberate B12-D17 documentation).
This completes the risk-ordered work queue (G1-G10).

Key points confirmed:

- **Full_Grid_Matrix** reproduces the MATLAB `repmat`/`reshape(B',[numel 1])`
  cartesian construction, with the LAST column varying fastest, identical to
  `itertools.product` order (verified against the MATLAB docstring example and a
  3-variable product). Python returns a list-of-lists rather than a numeric/cell
  matrix, but the row order and values match; the order is built with explicit
  repeat/tile loops, so there is no NumPy reshape-order trap.

- **FOM_rxffe_floating_taps** greedy bank-by-bank selection is faithful: the
  1-based candidate locations, the `argmax` first-max tie-break (matching MATLAB
  `max`), the chosen-bank position removal, and the overlapping-start-value
  removal all match (Python removes both in one pass over the original `valid`,
  which is equivalent because the bad-tap values `< start_tap` are disjoint from
  the removed positions' values `>= start_tap`). The `h[isi_start:isi_end]` first
  slice relies on `MMSE` passing a 0-based `isi_start`, which it does
  (`isi_start = dh+1` == MATLAB `dh+2`). Driven with an injected mock `MMSE_FOM`,
  it selects the expected banks. Python adds a defensive `if not valid: break`
  (MATLAB would error on an empty `valid` via `max([])`); benign and more robust.

- **MMSE** RxFFE driver: the `samp_idx` starting phase (`cursor_i % M` 0-based ==
  MATLAB `mod(cursor_i-1,M)+1` 1-based), `dh` (np.where == MATLAB `find`-1),
  `isi_start = dh+1`, and the `C`-realignment column (`idx+RxFFE_cmx` 0-based ==
  MATLAB `idx+RxFFE_cmx+1` 1-based) are all faithful 1-based to 0-based
  translations. The `toeplitz` builds and the trim optimisation match, and
  `wmax`/`wmin` (with the centre tap 1.0) match. Benign notes: `Rn =
  real(ifft(S_n))*fb` takes the real part (equal to fp for a symmetric PSD, whose
  autocorrelation is real); the `C`-zeroing end index uses a `min(Nw_out, ...)`
  clamp (inert when `ffe_pre_tap_len <= RxFFE_cmx`); and `Craw = w/w[dw]` is
  guarded against a zero main tap. `MMSE` calls the hoisted `_MMSE__MMSE_FOM`,
  which is line-identical to the top-level `MMSE_FOM` (verified), so it inherits
  B12-D17.

- **MMSE_FOM** block-matrix solve (Schur-complement speedup path, the DFE clip
  and re-solve, the RxFFE clip and normalisation, and the `sigma_e`/`FOM`
  formulas) is validated against an independent re-implementation of the MATLAB
  on a no-clip input (sigma_e/FOM/w/blim agree to 1e-9), and the FOM sign physics
  holds (lower noise raises FOM). The one divergence is B12-D17 (above). A second,
  unreachable edge is noted below.

### B12-D16. MMSE_FOM H/Rnn subset gate differs (unreachable edge)

- MATLAB: lines 2608/2613 subset `H` and `Rnn` to the used tap columns when
  `param.N_bg ~= 0` (the floating-tap config flag).
- Python: sicopr.py line 3312 gates the same subsetting on `len(idx) > 0`.
- Consequence: identical in the normal case (`N_bg != 0` with a non-empty float
  search) and in the no-float case (`N_bg == 0`, empty idx). They differ only when
  `N_bg != 0` but the float search returns an empty `idx` (a degenerate config
  where `N_bmax - RxFFE_cpx < N_bf`, so no valid tap bank exists): MATLAB still
  subsets `H` to `[1:Nfix]`, Python leaves all columns, changing `w`. Not reached
  in valid configurations. Recorded for completeness; low priority.

## Batch B11 verdicts (2026-07-11): G10 optimizer FOM core

| Function | MATLAB | Python | Verdict | Traps | Tests |
|---|---|---|---|---|---|
| OptFom_Calc_Hr | 2879-2885 | 3861-3866 | EQUIVALENT | 13 | 5 PASS |
| optimize_fom | 8547-8923 | 12470-12841 | EQUIVALENT (B11-D15 low) | 1,2,3,10,16 | 17 PASS / 1 FAIL |
| OptFom_Adaptive_Local_Search | variant 2739-2992 | 3611-3720 | EQUIVALENT | 3,10,16 | 11 PASS |

Batch total: 33 checks, 1 FAIL (the FAIL is the deliberate B11-D15 documentation).
No new DIVERGENT function; the three optimizer functions are EQUIVALENT.

Key points confirmed:

- **OptFom_Calc_Hr** is a thin element-wise combiner `H_r = H_bw.*H_bt.*H_RCos`
  (order-agnostic). Its hoisted `_OptFom_Calc_Hr__butterworth/bessel_thomson/
  raised_cosine` copies reproduce the top-level `Butterworth_Filter`,
  `Bessel_Thomson_Filter`, and `Raised_Cosine_Filter` to 1e-12 (cross-path
  identity test). `_BW_POLY` = MATLAB `[1 2.613126 3.414214 2.613126 1]`; the
  Bessel poly matches MATLAB `bessel(n)` (line 5058); the Tukey window matches
  `Tukey_Window` (line 4682) for monotonic f. Unit DC gain and the Butterworth
  3 dB point (`1/sqrt(2)` with MATLAB's truncated coeffs) hold.

- **optimize_fom** is EQUIVALENT on the base-config COM path. Every 1-based to
  0-based cursor/slice seam is faithful: `THIS.cursor_i` is kept 1-based and
  decremented only at index sites (py 12722/12723); the far-cursor slice
  (py 12731-12733) and the reversed pre-cursor slice (py 12735-12740) reproduce
  the MATLAB colons at 8797 and 8800-8801 (verified on synthetic sbr including
  the empty-precursor boundary at `cursor_i<=samples_per_ui`); the sbr
  zero-extend length count is 1-based-consistent (py 12751); and the
  post-optimize grids `f=arange(1e8,100e9+1e8,1e8)` (1000 points) and
  `t=arange(N)*ui/spui` match the MATLAB colons at 8905/8908. Control flow
  (continue/break, the do_C2M `loop_count`/break, and the EQ-failed branch)
  matches. The `NonZeroLSMethod==1` branch is a config-gated adaptive-search
  SUPERSET (from the variant); the default/legacy path calls
  `OptFom_Local_Search`, matching base MATLAB 8726. `OptFom_Itick_LocalSearch`
  is guarded by `if LOCAL_SEARCH>0` in Python, which is equivalent because the
  MATLAB function returns `skip_it=0` when `LocalSearch_Value=0` (base 3611-3622).
  All added instrumentation (progress prints, `SWEEP_LOG_CSV` trajectory,
  `FOM_history`, `result.FOM_TRACKER`/`sweep_dims`) has no effect on COM. The
  defensive clamps (`far_start>=0`, `abs(A_s)`, the `sigma>0` guard, the qual
  min-clamp) are no-ops on valid inputs.

- **OptFom_Adaptive_Local_Search** is a faithful port of the ADAPTIVE VARIANT
  file `com_ieee8023_4p15p0_adaptive_local_search.m` 2739-2992 (the correct
  source of truth per the audit prompt's cross-check instruction; it does not
  exist in base 4p15p0). It is a config-gated (`NonZeroLSMethod==1`) search
  SPEEDUP skip predicate and does not touch the physics/noise/FOM math. Called
  directly, its decisions match the variant: exact-match evaluates, `|CTLE
  idx - BEST.ctle|>2` skips, the hard cap `max(min_radius, round(1.2*LSV))`
  skips, the L1/L2 rule skips outside limits, `adaptive_radius` stays at or above
  `min_radius=1`, and the persistent state (`_ALS_STATE`) re-inits on
  `iter_count==1` and via `reset_state()` (matching MATLAB `persistent` +
  `iter_count==1`). `_mround` is half-away (B10). A latent, unreachable
  difference is noted below.

### B11-D15. optimize_fom triple_transit_time uses banker's rounding (low)

- MATLAB: com_ieee8023_4p15p0.m line 8749,
  `triple_transit_time = round(sbr_peak_i*2/param.samples_per_ui)+20`
  (half-away-from-zero).
- Python: sicopr.py line 12689 uses the builtin `round()` (round-half-to-even).
- Consequence: at an exact half-integer `2*sbr_peak_i/samples_per_ui` (e.g.
  peak=5, spui=4 gives 2.5) Python yields 22 where MATLAB yields 23, shifting the
  `min_number_of_UI_in_response` lower bound by one UI. `min_number_of_UI_in_response`
  is only ever raised, never lowered, so the effect is at most a 1-UI change in a
  response-length floor, and only at a measure-zero input. Same class as B10-D14.
- Evidence: FAIL optfom_triple_transit_uses_matlab_half_away in
  tests/test_optimizer_fom.py, with a companion PASS confirming the value is the
  banker's result.
- Recommended fix (not applied): use a half-away round at py 12689.

### B11 latent note. OptFom_Adaptive_Local_Search empty-tap guard placement (unreachable)

- Variant: the empty-tap `return` sits inside `if log_yes_1_no_0==1` (line 2859),
  so with logging OFF (the default) MATLAB falls through to the `[lp;vga]`
  distance computation.
- Python: sicopr.py line 3680 returns `skip=False` unconditionally on empty tap
  vectors.
- Consequence: none. Tap vectors are always at least `[1]` (OptFom_Build_TXFFE
  sets `FULL_tx_index_vector=1` when there are no TXFFE taps), so the empty
  branch is never entered in either implementation. Recorded for completeness;
  no fix warranted.

## Batch B10 verdicts (2026-07-10): cross-cutting scans

| Scan | Verdict | Traps | Tests | Result |
|---|---|---|---|---|
| G8 rounding (mround vs np.round) | DIVERGENT (low) | 3, 8 | 10 checks | 9 PASS, 1 FAIL (D14) |
| G9 reshape/ravel/flatten order | EQUIVALENT | 9 | 4 checks | PASS |

### B10-D14. Bare banker's rounding at ~50 non-critical sites (low, systematic)

- MATLAB `round()` is half-away-from-zero; Python's builtin `round` and
  `np.round` are banker's (round-half-to-even).
- The port provides 6 `_mround` helpers (py 3578, 7514, 8684, 13635, 14928,
  15302) that correctly implement half-away, and uses them exactly where an
  off-by-one matters: the port-order quarter/three-quarter slice indices
  (auto_port_order, read_Nport_touchstone, read_p4_s4params, read_s4p_files) that
  select an S-parameter sub-band, and the adaptive-search radius
  (compute_hard_cap / OptFom_Adaptive_Local_Search).
- The remaining ~50 rounding sites use a bare `round`/`np.round` with banker's
  semantics: PDF bin snapping (d_cpdf, Init_PDF_Fast, get_pdf_from_sampled_signal
  and their many copies), grid-length counts (s21_to_impulse_DC, already tracked
  as B03-D7), normal_dist Min (py 12412), `nui = round(len/M)` (py 10916, 11284),
  and combine_pdf shift (py 8020).
- Consequence: these differ from MATLAB only for exact half-integer inputs.
- Demonstrated at normal_dist: with `2*nsigma*sigma/binsize = 2.5` the port gives
  Min = -2 (banker's) vs MATLAB -3 (half-away).
- Evidence: FAIL normal_dist_Min_uses_matlab_half_away in
  tests/test_rounding_reshape.py (all 6 `_mround` helpers pass the half-away
  classic cases).

> **CORRECTION 2026-08-22 — this verdict was wrong, and `nui` was a live defect.**
>
> The reasoning above ended "…which are measure-zero for continuous
> signal/frequency data", and on that basis the fix was left unapplied. That is
> sound for a *continuous* input and false for `nui = round(len/M)`:
> `len(residual_response)` and `M` are both **integers**, so the quotient is a
> rational that lands on `.5` exactly whenever the response length is an odd
> multiple of `M/2`. Not measure-zero — a discrete quantity that really does hit
> the tie.
>
> It did, on 4 of the 208 reference case-instances. Instrumenting a full run
> found exactly **one of 98,145** `round()` calls sitting on a tie
> (`2360.500000`: MATLAB 2361, Python 2360). One row is lost from the `vs`
> sampling matrix, one ISI sample is dropped from the residual-ISI PDF, and
> `sgm_isi` comes out low. It was **the last COM miss**: fixing it took the
> correlation from COM 207/208 to **208/208**, max \|ΔCOM\|
> 0.0076 dB → 3.3e-14.
>
> Fixed in `get_pdf` and `get_pdf_full` via a local `_mround`. See
> `MATLAB_Correlation_Review.md` §4.3 and `docs/FIX_SUMMARY.md` #16.
>
> **The lesson generalises beyond this site.** "Measure-zero for continuous data"
> is only an argument when the input is continuous. Several of the ~50 remaining
> bare-`round` sites take integer ratios or bin counts, and each of those needs
> the argument re-made rather than inherited. The other sites listed above have
> **not** been re-examined under this correction.
>
> This is the second EQUIVALENT/negligible verdict in this document overturned by
> the corpus; the first was the `process_sxp` reference leak. Treat the ledger as
> evidence, not proof.

### G9 reshape/order: clean

The single genuinely order-sensitive reshape - the get_PSDs fold
`sum(reshape(psd, num_ui, M).')` - uses `order='F'` (py 9484) and is verified
column-major (a C-order fold would give the wrong answer). The get_pdf and
get_pdf_full vs-matrix reshapes use numpy C-order, which equals MATLAB's
`reshape(v, cols, rows)` followed by transpose (verified). There are no
`.flatten()` calls; the ~28 other reshapes are 1-D to 2-D vector coercion
(order-agnostic), already-2D no-ops, or row-major-by-construction Touchstone
data; and the ~398 `ravel()` calls are `np.asarray(x).ravel()` 1-D coercion. No
divergence found in the largest single trap surface of the port.

## Batch B09 verdicts (2026-07-10)

No divergences. All four EQUIVALENT with passing tests.

| Function | Verdict | Trap categories | Tests | Result |
|---|---|---|---|---|
| cursor_sample_index (ML 5537-5611, py 8130-8192) | EQUIVALENT | 1, 2, 10 | 5 checks | PASS |
| get_center_of_UI (ML 7450-7460, py 10596-10606) | EQUIVALENT | 1 | 5 checks | PASS |
| COM_eye_width (ML 1367-1556, py 1607-1879) | EQUIVALENT | 1, 2, 10 | 6 checks | PASS |
| vma (ML 11479-11507, py 17664-17706) | EQUIVALENT | 1, 6 | 3 checks | PASS |

Verified equivalences worth recording (all test-pinned):

- cursor_sample_index reproduces the Muller-Mueller criterion with consistent
  1-based to 0-based conversions at every seam: the peak index, the zero-crossing
  detection (diff of sign rising through 0.01*peak), the MM metric window, and
  the final cursor. The no-zero-crossing flag fires correctly, and the
  first-precursor-small physics holds.
- get_center_of_UI has three copies, and each returns the index base its caller
  expects: the top-level returns argmin (0-based), the get_pdf_full copy returns
  M//2+1 (1-based, matching get_pdf_full's 1-based usage), and the COM_eye_width
  copy returns M//2 (0-based). This independently confirms the B08-D13
  withdrawal: get_pdf_full uses a 1-based half_UI correctly, so the only cause of
  the get_pdf_full divergence is the resample grid (B08-D12).
- COM_eye_width's eye-contour assembly, half_UI-centered eye height, and
  find_eye_width call are correct; the eye width is nonnegative and bounded by
  1 UI, the eye height is positive for an open eye and shrinks under heavier
  noise, and the widest opening sits at the center phase. Two caveats: on the
  real C2M path it calls the divergent get_pdf_full (inherits B08-D12), and
  py 1752-1775 adds a Python-only timing_bathtub side-channel (plotting only,
  outside the audited return values).
- vma computes VMA = P_3 - P_0 from PRBS13Q symbol runs with consistent 0-based
  index formulas; P_3 > P_0 and VMA is bounded by the bit-stream swing.

## Batch B08 verdicts (2026-07-09)

One divergence (get_pdf_full, C2M-only). The composite noise pipeline that
standard COM uses is EQUIVALENT.

| Function | Verdict | Trap categories | Tests | Result |
|---|---|---|---|---|
| Create_Noise_PDF (ML 1557-1685, py 1995-2125) | EQUIVALENT | 6, 15, 20 | 6 checks | PASS |
| combine_pdf_same_voltage_axis (ML 5418-5458, py 8009-8046) | EQUIVALENT | 2, 18 | 3 checks | PASS |
| comb_fct (ML 5390-5415, py 7960-7984) | EQUIVALENT | 2, 18 | 2 checks | PASS |
| conv_fct_MeanNotZero (ML 5521-5536, py 8114-8124) | EQUIVALENT | 6 | 2 checks | PASS |
| get_pdf_from_sampled_signal (ML 7603-7649, py 11037-11085) | EQUIVALENT | 3, 6 | 4 checks | PASS |
| get_pdf_full (ML 7650-7793, py 11197-11348) | DIVERGENT | 2, 7, 4 | 1 check | FAIL (D12) |

### B08-D12. get_pdf_full resample grid adds extra points (C2M only)

get_pdf_full is used only by optimize_fom_for_C2M (ML 1391), so this affects the
chip-to-module eye path, not standard COM.

- sicopr.py builds the resampled time axis with
  `np.arange(0, floor(x*samp_UI)+2)/samp_UI` on each side (py 11218-11222), which
  adds one extra point beyond the data range per side compared to MATLAB's colon
  `0:-1/samp_UI:min` / `0:1/samp_UI:max` (ML 7665-7667). np.interp clamps the
  extra points to the pulse endpoints. On a ratio-2 upsample the resampled pulse
  was length 641 vs MATLAB 639, and the cursor index t_s shifted 192 -> 193.
- That one-sample shift propagates to the sub-phase `mod(t_s, samp_UI)` and to
  the cursor-centering circshift (off by 1). Consequence: 2 of the 32
  phase-column PDFs returned by get_pdf_full diverge from a MATLAB-faithful
  oracle (30/32 match). The DFE cancellation window and the
  column-major-reshape-plus-transpose (equivalent to numpy C-order reshape) are
  correct; only the grid length is wrong.
- Evidence: FAIL get_pdf_full_matches_matlab_oracle in
  tests/test_noise_pdf_composite.py.
- Recommended fix (not applied): use `+1` instead of `+2` in the arange bounds;
  the t_s, phase, and circshift then realign automatically.

Correction (2026-07-10): an earlier draft of this section listed a second finding
B08-D13 claiming get_pdf_full's cursor-centering shift mixed a 0-based half_UI
with a 1-based phase. That was a diagnostic error: get_pdf_full uses
`_get_pdf_full__get_center_of_UI = M//2 + 1` (1-based, matching MATLAB), so the
shift is computed consistently in one base. B08-D13 is withdrawn; D12 is the sole
cause of the get_pdf_full divergence.

Verified equivalences worth recording (all test-pinned):

- Create_Noise_PDF's full convolution chain (gaussian noise, ne, p_DD, sci, cci)
  conserves probability mass, yields a monotone CDF ending at ~1, and computes
  sigma_TX and sigma_G by the 93A formulas.
- get_pdf_from_sampled_signal conserves mass, gives a zero-mean PDF for symmetric
  input, and its output variance equals the sum of the per-term variances (the
  hallmark of an independent-ISI convolution).
- conv_fct_MeanNotZero is byte-for-byte identical to conv_fct; comb_fct and
  combine_pdf_same_voltage_axis align and sum PDFs correctly.

## Batch B07 verdicts (2026-07-09)

No divergences. All six EQUIVALENT with passing tests.

| Function | Verdict | Trap categories | Tests | Result |
|---|---|---|---|---|
| normal_dist (ML 8540-8545, py 12400-12421) | EQUIVALENT | 3, 18 | 4 checks | PASS |
| pdf_to_cdf (ML 8983-8994, py 12973-12989) | EQUIVALENT | 6 | 3 checks | PASS |
| conv_fct (ML 5503-5517, py 8085-8095) | EQUIVALENT | 6 | 4 checks | PASS |
| d_cpdf (ML 5612-5662, py 8220-8269) | EQUIVALENT | 3, 18 | 4 checks | PASS |
| Init_PDF_Fast (ML 2204-2270, py 2649-2666) | EQUIVALENT | 3, 18 | 2 checks | PASS |
| get_pdf (ML 7507-7602, py 10864-10962) | EQUIVALENT | 1, 2, 4, 10 | 6 checks | PASS |

Verified equivalences worth recording (all test-pinned):

- **The DFE is applied exactly once in get_pdf.** The cursor + DFE cancellation
  subtracts the effective cancellation samples once over the window
  [t_s - M/2, t_s + (1/2 + ndfe)M - 1]: an in-bound postcursor is fully
  cancelled (residual 0) and a clipped postcursor leaves exactly tap - clip
  (a double subtraction would leave tap - 2*clip). Together with the
  already-verified get_PSDs S_isi path (B01), this closes the historical
  double-subtraction concern.
- get_pdf's vs-matrix and phase-column selection reconcile MATLAB's 1-based
  column p with Python's 0-based column p-1 at the same physical samples, so the
  THRU cursor phase and the MMSE crosstalk ixphase seam (0-based from get_PSDs)
  both index the correct sub-phase.
- normal_dist, d_cpdf, Init_PDF_Fast, and conv_fct all conserve probability mass
  and preserve the bin axis; convolution is associative; normal_dist is
  symmetric, centered, and has the right sigma.
- pdf_to_cdf produces a monotone-increasing bottom-eye CDF (yB) and
  monotone-decreasing top-eye CDF (yT) with small tails, matching MATLAB's
  element-wise min.

Minor note (no divergence): normal_dist and Init_PDF_Fast use Python's
banker's-rounding `round`/`np.round` where MATLAB rounds half-away; this only
changes a bin index when the rounded argument is exactly a half-integer, which
does not occur for realistic sigma/binsize values.

## Batch B06 verdicts (2026-07-09)

| Function | Verdict | Trap categories | Tests | Result |
|---|---|---|---|---|
| synth_tline (ML 11434-11458, py 17475-17524) | EQUIVALENT | 13, 14 | 5 checks | PASS |
| R_series2 (ML 4359-4364, py 6348-6367) | EQUIVALENT | 9 | shared | PASS |
| r_parrelell2 (ML 9520-9524, py 13577-13597) | EQUIVALENT | 9 | shared | PASS |
| get_TDR (ML 7034-7345, py 10231-10487) | DIVERGENT | 3, 5, 13 | 8 checks | 6 PASS, 2 FAIL (D9; D11) |
| TDR_ERL_Processing (ML 4503-4597, py 6632-6725) | DIVERGENT | 1, 15 | 3 checks | 2 PASS, 1 FAIL (D11) |

Secondary findings (lower severity):

- **B06-D10 (low, non-default).** In get_TDR's ERL phase search, when the config
  `ERL_FOM` (`RL_norm_test`) is 0, MATLAB (lines 7316-7320) assigns `best_erl`
  every iteration so it ends up as the last phase's ERL, while Python (lines
  10463-10468) assigns it only inside the worst-case branch. `ERL_FOM` defaults
  to 1 ("Do not change"), and in that default path both codes recompute
  `best_erl` from `best_ki` after the loop and agree. Divergence confined to the
  non-default mode.
- **B06-D11 (low, cosmetic).** TDR_ERL_Processing builds the ERL_ONLY
  `file_names` label from all channel bases (py 6721-6722) rather than only the
  first channel (ML 4592 `str2csv({chdata(1).base})`). Report string only, no
  numeric effect.

Verified equivalences worth recording (all test-pinned):

- synth_tline reproduces the 802.3 93A tline model including the complex
  propagation constant (sqrt/log branch) and the f=0 fix that keeps the log(0)
  NaN from leaking; it is reciprocal, symmetric, passive, and collapses to an
  ideal through at d=0.
- get_TDR's 4-port reflection kernel (the long rational renormalisation) matches
  MATLAB, a matched network gives zero reflection, and a matched line yields
  avgZport = 2*ZT end to end through the impulse, step, and pulse responses.
- TDR_ERL_Processing copies all TDR/ERL fields correctly and computes
  min_ERL = min(ERL11, ERL22) with the TDR_W_TXPKG and empty-ERL22 branches
  intact.
- R_series2 and r_parrelell2 are the correct series/shunt resistor 2-ports
  (passive, reciprocal, right open/through limits).

## Batch B05 verdicts (2026-07-09)

One divergence, in dead code (s_for_c4). The five cascade functions that the
pipeline actually uses are all EQUIVALENT.

| Function | Verdict | Trap categories | Tests | Result |
|---|---|---|---|---|
| combines4p (ML 5459-5502, py 8052-8064) | EQUIVALENT | 8, 9 | 3 checks | PASS |
| stot (ML 11418-11424, py 17398-17428) | EQUIVALENT | 8, 13 | 2 checks | PASS |
| ttos (ML 11460-11466, py 17543-17573) | EQUIVALENT | 8, 13 | 2 checks | PASS |
| make_full_pkg (ML 8296-8488, py 12088-12264) | EQUIVALENT | 1, 2, 9, 15 | 7 checks | PASS |
| s_for_c2 (ML 11289-11295, py 17098-17122) | EQUIVALENT | 9 | 3 checks | PASS |
| s_for_c4 (ML 11297-11301, py 17149-17173) | DIVERGENT | 9, 15 | 2 checks | 1 PASS, 1 FAIL (D8) |

### B05-D8. s_for_c4 skips the snp2smp([1 3 2 4]) port reorder (unused function)

- MATLAB: line 11301 applies `snp2smp(S.Parameters, zref, [1 3 2 4])`, reordering
  the ports of the block-diagonal 4-port.
- Python: line 17167 returns the block-diagonal unchanged and its docstring
  wrongly claims the reorder is the identity. A pure permutation check shows
  `[1 3 2 4]` moves the off-diagonal capacitor terms (the block-diagonal groups
  ports as {1,2}/{3,4}; the reorder regroups them as {1,3}/{2,4}).
- Consequence: none in the current pipeline. Neither `s_for_c2` nor `s_for_c4`
  has any caller in the codebase, so this is dead code. The underlying
  `s2_to_s4` and `snp2smp` are external RF-Toolbox functions absent from the
  source, so the exact MATLAB result cannot be fully reconstructed here.
- Evidence: FAIL s_for_c4_applies_port_reorder in tests/test_sparam_cascade.py
  (with a companion PASS proving the reorder is non-trivial).
- Recommended fix (not applied): apply the `[0,2,1,3]` permutation to the
  block-diagonal, or delete the function if confirmed dead.

Verified equivalences worth recording (all test-pinned):

- `stot`/`ttos` round-trip in both directions and match the Mavaddat T-parameter
  convention, with `delta` computed from the original divisor before the eps
  guard and the correct `eps` = 2.2e-16.
- `combines4p` equals the T-matrix cascade `ttos(stot(A) @ stot(B))`, confirming
  a consistent T convention and multiply order; cascading with a matched through
  returns the original network, and reciprocity is preserved.
- `s_for_c2` is the lossless shunt capacitor (|S11|^2+|S21|^2 = 1, through at
  DC).
- `make_full_pkg` selects the right TX vs RX parameter indices, applies the
  dc/cd common-mode scaling (Z0/2, Cpad*2, ...), and cascades the mele=4
  multi-block package identically to a manual combines4p chain, with a
  reciprocal and passive result.

## Batch B04 verdicts (2026-07-09)

No divergences. All five EQUIVALENT with passing tests.

| Function | Verdict | Trap categories | Tests | Result |
|---|---|---|---|---|
| FFE (ML 2031-2047, py 2417-2428) | EQUIVALENT | 1, 2, 6 | 4 checks | PASS |
| FFE_Fast (ML 2054-2067, py 2449-2456) | EQUIVALENT | 6, 9 | 3 checks | PASS |
| Fract_T_FFE (ML 2114-2123, py 2544-2549) | EQUIVALENT | 6 | 3 checks | PASS |
| TD_CTLE (ML 4598-4613, py 6748-6764) | EQUIVALENT | 6, 13 | 6 checks | PASS |
| dfe_clipper (ML 5663-5675, py 8290-8316) | EQUIVALENT | 4, 9 | 5 checks | PASS |

Verified equivalences worth recording (all test-pinned):

- MATLAB `circshift(V',[ishift,0])` maps to `np.roll(V, ishift)` in the same
  direction, and the tap-shift index remap `(i-1-cmx)*spui` (1-based) to
  `(i-cmx)*spui` (0-based) is correct in FFE and Fract_T_FFE. FFE's impulse tap
  placement (precursors wrap to the tail, postcursors move later) is verified
  independently of the transcription.
- FFE_Fast with pre-shifted columns reproduces FFE exactly (cross-path check).
- TD_CTLE's bilinear-transform IIR uses `np.poly` (which matches MATLAB `poly`:
  monic, highest-degree first) and `scipy.lfilter` (same direct-form as MATLAB
  `filter`). The discrete DC gain equals the analog CTLE DC gain
  10^(kacdc_dB/20) to 1e-9, an analytic invariant that confirms the whole
  pole/zero/gain construction, and the poles are inside the unit circle.
- dfe_clipper's element-wise clip matches, with both masks referencing the
  original input, and it keeps DFE taps within [bmin*cursor, bmax*cursor].

Note on the historical double-DFE-subtraction concern: none of these five
functions subtract the DFE (FFE/Fract_T_FFE are feed-forward tap sums, TD_CTLE is
an IIR, dfe_clipper is the clip primitive). The DFE subtraction lives in get_pdf
and get_PSDs; the get_PSDs S_isi path was already verified in B01, and get_pdf is
scheduled for B07-B09.

## Batch B03 verdicts (2026-07-09)

| Function | Verdict | Trap categories | Tests | Result |
|---|---|---|---|---|
| s21_to_impulse_DC (ML 11218-11288, py 16992-17075) | DIVERGENT | 1, 2, 3, 5 | 13 checks | 11 PASS, 2 FAIL (D6, D7) |
| COM_FD_to_TD (ML 1206-1366, py 1346-1482) | EQUIVALENT | 1, 6, 20 | 5 checks | PASS |

Verified equivalences worth recording (all test-pinned):

- The mainline impulse path in s21_to_impulse_DC matches MATLAB to 1e-11: the
  Hermitian extension forces DC and Nyquist real and mirrors the interior as
  conjugates (length 2N), the real ifft uses the same 1/N scaling, t_base =
  (0:L-1)/(df*L), and the `find(...,1,'last')` truncation maps correctly. This
  is the default output because ENFORCE_CAUSALITY is off by default, so the APM
  off-by-one (B03-D6) does not touch the returned voltage in default runs.
- Parseval energy consistency holds between IL_symmetric and the impulse
  response, and a causal delay channel produces a peak at the expected sample
  with negligible pre-cursor energy.
- The all-zero-IL branch uses MATLAB `eps` = 2.2e-16 correctly (py 17013), in
  contrast to the interp_Sparam floor bug B02-D4.
- COM_FD_to_TD is a faithful driver: the Gaussian Tx filter, the
  `filter(ones(1,M),1,ir)` running-sum pulse, the amplitude scaling, and the
  SCMR P_signal sampling phase/stride all match MATLAB; it delegates numerics to
  s21_to_impulse_DC and adds no divergence of its own.

## Batch B02 verdicts (2026-07-09)

| Function | Verdict | Trap categories | Tests | Result |
|---|---|---|---|---|
| FD_Processing (ML 1689-2030, py 2177-2394) | EQUIVALENT | 1, 2, 7, 20 | 16 checks | PASS |
| interp_Sparam (ML 8080-8295, py 11739-11971) | DIVERGENT | 3, 7, 13, 20 | 13 checks | 11 PASS, 2 FAIL (D4, D5) |
| read_s4p_files (ML 10729-10909, py 16060-16203) | EQUIVALENT | 1, 2 | 5 checks | PASS |

Verified equivalences worth recording (all test-pinned):

- FD_Processing grid index windows reproduce MATLAB `find(f>=x,1,'first')` and
  `find(f<=x,1,'last')` exactly, including exact-on-grid, between-point, and
  no-match clamp cases, and the integration step `delta_f = faxis(11)-faxis(10)`
  maps to the 0-based `[10]-[9]`.
- FD_Processing P_signal, FOM_ILD, MDFEXT_ICN, ICN, and the Nyquist-loss
  `interp1` all match MATLAB transcription to 1e-9; the bin exactly at f2 is
  inclusive in both.
- read_s4p_files carries the native Touchstone grid through un-resampled
  (GHz to Hz), and the crosstalk axis-consistency guard raises on a mismatched
  grid, matching MATLAB.
- interp_Sparam's default magnitude path (`linear_trend_to_DC`) pre-extends the
  interpolation grid to [0, fout[-1]], so its use of `np.interp` (which clamps)
  agrees with MATLAB `interp1(...,'extrap')` in the interior; the divergence is
  confined to phase extrapolation and the eps floor.

## Batch B01 verdicts (2026-07-09)

| Function | Verdict | Trap categories | Tests | Result |
|---|---|---|---|---|
| S_RN (ML 4479-4502, py 6594-6610) | EQUIVALENT | 8, 20 | 3 checks | PASS |
| N_s (ML 2698-2738, py 3391-3415) | EQUIVALENT | 1, 2, 18 | 8 checks | PASS |
| get_sigma_noise (ML 7950-7973, py 11489-11521) | EQUIVALENT | 1, 2, 10 | 4 checks | PASS |
| get_sigma_eta_ACCM_noise (ML 7926-7948, py 11447-11483) | EQUIVALENT | 1, 2, 9, 20 | 4 checks | PASS |
| get_PSDs (ML 6512-6784, py 9494-9789) | DIVERGENT | 1, 4, 5, 15, 20 | 23 checks | 18 PASS, 5 FAIL (D1-D3) |

Verified equivalences worth recording (all test-pinned):

- The eta_0 unit convention (V^2/GHz) and its /1e9 conversion at MATLAB 7936
  and 6553 are correct in Python (11457, 9558): sigma_N1 equals
  sqrt(eta_0 * bandwidth_GHz) in closed form.
- MATLAB 7973 computes sigma_HP without a square root (a quirk); Python
  preserves the quirk at 11520.
- The PSD fold sum(reshape(psd, num_ui, M).') uses order='F' semantics
  correctly (py 9484) and conserves total noise power.
- Python's H_rxffe loop indexes taps from 0 instead of MATLAB's 1 (py 9541 vs
  ML 6538); the difference is a pure one-UI linear phase that cancels in
  |H_rxffe|^2, so all downstream uses are identical.
- result.iphase is 0-based in Python vs 1-based in MATLAB; the selected
  crosstalk phase and hrn are identical, and the only consumer (get_pdf)
  documents its ixphase argument as 0-based. Watch this seam again when
  auditing get_pdf (B07-B09).

## Context and caveats

- sicopr.py header states it was assembled from com_ieee8023_4p14p0.m; this audit judges it against 4p15p0 per the audit prompt, so 4p14p0-to-4p15p0 deltas will surface as findings.
- compare_to_matlab.py no longer exists (only a stale .pyc); Tier 5 npz seam-dump convention must be recreated when first needed.
- Prior 7-bug hypothesis audit (completed 2026-07-01) preserved in audit_state_prior_bughunt_20260701.json; its per-function results are carried in prior_audit fields but do NOT exempt those functions from this audit's per-function test requirement.

## Coverage map (Tier 0)

157 MATLAB functions (154 in the base 4p15p0 file, 3 only in the adaptive local-search variant). All ported, none UNPORTED. All 157 carry a provenance banner citing 4p14p0 line numbers, so line cites below are re-derived against 4p15p0.

| MATLAB function | MATLAB lines (4p15p0) | Python counterpart | py lines | Ported | Provenance cite | Copies | Prior audit |
|---|---|---|---|---|---|---|---|
| com_ieee8023_ | 1-909 | com_ieee8023_ | 261-665 | yes | yes | 0 |  |
| Apply_EQ | 910-981 | Apply_EQ | 697-782 | yes | yes | 0 |  |
| Bathtub_Contribution_Wrapper | 982-1032 | Bathtub_Contribution_Wrapper | 867-941 | yes | yes | 0 |  |
| Bessel_Thomson_Filter | 1033-1042 | Bessel_Thomson_Filter | 953-965 | yes | yes | 7 |  |
| Bread_Crumb_Chdata_Reduction | 1043-1087 | Bread_Crumb_Chdata_Reduction | 966-1005 | yes | yes | 0 |  |
| Burst_Probability_Calc | 1088-1137 | Burst_Probability_Calc | 1085-1137 | yes | yes | 1 |  |
| Butterworth_Filter | 1138-1143 | Butterworth_Filter | 1138-1160 | yes | yes | 5 |  |
| CDF_ev | 1144-1146 | CDF_ev | 1161-1182 | yes | yes | 2 |  |
| CDF_inv_ev | 1147-1153 | CDF_inv_ev | 1183-1212 | yes | yes | 3 |  |
| COM_CommandLine_Parse | 1154-1205 | COM_CommandLine_Parse | 1221-1297 | yes | yes | 0 |  |
| COM_FD_to_TD | 1206-1366 | COM_FD_to_TD | 1346-1517 | yes | yes | 0 | BUG-01 unfounded (2026-07-01 re-check) |
| COM_eye_width | 1367-1556 | COM_eye_width | 1607-1879 | yes | yes | 0 |  |
| Create_Noise_PDF | 1557-1685 | Create_Noise_PDF | 1995-2142 | yes | yes | 0 | BUG-06 unfounded (2026-07-01 re-check) |
| FD_CTLE | 1686-1688 | FD_CTLE | 2143-2159 | yes | yes | 2 |  |
| FD_Processing | 1689-2030 | FD_Processing | 2177-2416 | yes | yes | 0 |  |
| FFE | 2031-2053 | FFE | 2417-2448 | yes | yes | 5 |  |
| FFE_Fast | 2054-2067 | FFE_Fast | 2449-2486 | yes | yes | 1 |  |
| FOM_rxffe_floating_taps | 2068-2113 | FOM_rxffe_floating_taps | 2487-2543 | yes | yes | 0 |  |
| Fract_T_FFE | 2114-2123 | Fract_T_FFE | 2544-2568 | yes | yes | 0 |  |
| Full_Grid_Matrix | 2124-2184 | Full_Grid_Matrix | 2569-2598 | yes | yes | 1 |  |
| H_interp | 2185-2203 | H_interp | 2599-2648 | yes | yes | 1 |  |
| Init_PDF_Fast | 2204-2270 | Init_PDF_Fast | 2649-2688 | yes | yes | 10 |  |
| MLSE | 2271-2352 | MLSE | 2713-2807 | yes | yes | 0 |  |
| MLSE_U1_c_178A | 2353-2484 | MLSE_U1_c_178A | 2884-3038 | yes | yes | 0 |  |
| MMSE | 2485-2584 | MMSE | 3133-3285 | yes | yes | 0 |  |
| MMSE_FOM | 2585-2697 | MMSE_FOM | 3286-3390 | yes | yes | 1 |  |
| N_s | 2698-2738 | N_s | 3391-3435 | yes | yes | 1 | BUG-05 unfounded (2026-07-01 re-check) |
| OptFom_Build_TXFFE | 2739-2821 | OptFom_Build_TXFFE | 3459-3577 | yes | yes | 0 |  |
| OptFom_Calc_FOM | 2822-2878 | OptFom_Calc_FOM | 3742-3817 | yes | yes | 0 |  |
| OptFom_Calc_Hr | 2879-2885 | OptFom_Calc_Hr | 3861-3888 | yes | yes | 2 |  |
| OptFom_Calc_Noise | 2886-3014 | OptFom_Calc_Noise | 3889-4031 | yes | yes | 0 |  |
| OptFom_Calc_Noise_XC | 3015-3038 | OptFom_Calc_Noise_XC | 4032-4087 | yes | yes | 0 |  |
| OptFom_Calculate_Settings | 3039-3148 | OptFom_Calculate_Settings | 4139-4265 | yes | yes | 0 |  |
| OptFom_Compute_CTLE | 3149-3223 | OptFom_Compute_CTLE | 4291-4390 | yes | yes | 0 |  |
| OptFom_Compute_DFE | 3224-3309 | OptFom_Compute_DFE | 4512-4611 | yes | yes | 0 |  |
| OptFom_Compute_RxFFE | 3310-3374 | OptFom_Compute_RxFFE | 4612-4656 | yes | yes | 0 |  |
| OptFom_Compute_TXFFE | 3375-3422 | OptFom_Compute_TXFFE | 4674-4744 | yes | yes | 0 |  |
| OptFom_Create_Output | 3423-3509 | OptFom_Create_Output | 4745-4824 | yes | yes | 0 |  |
| OptFom_FD_or_TD_Fields | 3510-3519 | OptFom_FD_or_TD_Fields | 4825-4834 | yes | yes | 1 |  |
| OptFom_Find_Sample_Point | 3520-3542 | OptFom_Find_Sample_Point | 4879-4912 | yes | yes | 0 |  |
| OptFom_Initialize_Loop_Struct | 3543-3584 | OptFom_Initialize_Loop_Struct | 4913-4950 | yes | yes | 0 |  |
| OptFom_Itick_BoxSearch | 3585-3610 | OptFom_Itick_BoxSearch | 4951-4984 | yes | yes | 0 |  |
| OptFom_Itick_LocalSearch | 3611-3622 | OptFom_Itick_LocalSearch | 4985-5014 | yes | yes | 0 |  |
| OptFom_Local_Search | 3623-3678 | OptFom_Local_Search | 5015-5057 | yes | yes | 0 |  |
| OptFom_Plot_Best_Results | 3679-3789 | OptFom_Plot_Best_Results | 5058-5104 | yes | yes | 0 |  |
| OptFom_Set_Best_Itick | 3790-3808 | OptFom_Set_Best_Itick | 5105-5123 | yes | yes | 0 |  |
| OptFom_Setup_Sampler_Sweep | 3809-3847 | OptFom_Setup_Sampler_Sweep | 5124-5173 | yes | yes | 0 |  |
| OptFom_Update_BEST_Post_Optimize | 3848-3895 | OptFom_Update_BEST_Post_Optimize | 5235-5332 | yes | yes | 0 |  |
| OptFom_Update_Best_Settings_EQ_Failed | 3896-3939 | OptFom_Update_Best_Settings_EQ_Failed | 5346-5408 | yes | yes | 0 |  |
| OptFom_Update_Best_Setttings | 3940-3980 | OptFom_Update_Best_Setttings | 5422-5492 | yes | yes | 0 |  |
| Output_Arg_Fill | 3981-4178 | Output_Arg_Fill | 5685-5894 | yes | yes | 0 |  |
| PRBS13Q | 4179-4217 | PRBS13Q | 5950-6001 | yes | yes | 2 |  |
| pam | 4218-4230 | pam | 6002-6043 | yes | yes | 3 |  |
| RILN_TD | 4231-4310 | RILN_TD | 6211-6298 | yes | yes | 0 |  |
| RXFFE_Illegal | 4311-4358 | RXFFE_Illegal | 6299-6347 | yes | yes | 0 |  |
| R_series2 | 4359-4366 | R_series2 | 6348-6384 | yes | yes | 2 |  |
| Raised_Cosine_Filter | 4367-4372 | Raised_Cosine_Filter | 6399-6408 | yes | yes | 0 |  |
| SL | 4373-4409 | SL | 6437-6468 | yes | yes | 1 |  |
| SNDR_ref | 4410-4447 | SNDR_ref | 6483-6545 | yes | yes | 0 |  |
| S_IN | 4448-4478 | S_IN | 6569-6593 | yes | yes | 1 |  |
| S_RN | 4479-4502 | S_RN | 6594-6631 | yes | yes | 1 |  |
| TDR_ERL_Processing | 4503-4597 | TDR_ERL_Processing | 6632-6747 | yes | yes | 0 |  |
| TD_CTLE | 4598-4614 | TD_CTLE | 6748-6784 | yes | yes | 2 |  |
| TD_FD_fillin | 4615-4681 | TD_FD_fillin | 6814-6901 | yes | yes | 0 |  |
| Tukey_Window | 4682-4701 | Tukey_Window | 6902-6940 | yes | yes | 3 |  |
| Tx_FFE_Filter | 4702-4746 | Tx_FFE_Filter | 6941-6973 | yes | yes | 0 |  |
| Write_CSV | 4747-4772 | Write_CSV | 6992-7017 | yes | yes | 0 |  |
| add_brd | 4773-4825 | add_brd | 7049-7113 | yes | yes | 1 |  |
| add_brdorig | 4826-4847 | add_brdorig | 7141-7187 | yes | yes | 0 |  |
| add_pkg_with_die | 4848-4862 | add_pkg_with_die | 7239-7288 | yes | yes | 1 |  |
| adjust_Rx_noise_for_quantization | 4863-4902 | adjust_Rx_noise_for_quantization | 7386-7432 | yes | yes | 0 |  |
| applyDFEbk | 4903-4931 | applyDFEbk | 7433-7473 | yes | yes | 0 |  |
| auto_port_order | 4932-5057 | auto_port_order | 7519-7645 | yes | yes | 3 |  |
| bessel | 5058-5064 | bessel | 7474-7513 | yes | yes | 4 |  |
| calculate_delay_CausalityEnforcement | 5065-5218 | calculate_delay_CausalityEnforcement | 7646-7755 | yes | yes | 1 |  |
| capture_RIL_RILN | 5219-5377 | capture_RIL_RILN | 7756-7869 | yes | yes | 0 |  |
| cdf_to_ber_contour | 5378-5389 | cdf_to_ber_contour | 7916-7959 | yes | yes | 1 |  |
| comb_fct | 5390-5417 | comb_fct | 7960-8008 | yes | yes | 0 |  |
| combine_pdf_same_voltage_axis | 5418-5458 | combine_pdf_same_voltage_axis | 8009-8051 | yes | yes | 1 |  |
| combines4p | 5459-5502 | combines4p | 8052-8084 | yes | yes | 8 |  |
| conv_fct | 5503-5520 | conv_fct | 8085-8113 | yes | yes | 14 |  |
| conv_fct_MeanNotZero | 5521-5536 | conv_fct_MeanNotZero | 8114-8129 | yes | yes | 1 |  |
| cursor_sample_index | 5537-5611 | cursor_sample_index | 8130-8219 | yes | yes | 1 |  |
| d_cpdf | 5612-5662 | d_cpdf | 8220-8289 | yes | yes | 13 |  |
| dfe_clipper | 5663-5676 | dfe_clipper | 8290-8336 | yes | yes | 4 |  |
| end_display_control | 5677-5854 | end_display_control | 8337-8479 | yes | yes | 0 |  |
| find_eye_width | 5855-5917 | find_eye_width | 8487-8547 | yes | yes | 1 |  |
| findbankloc | 5918-6069 | findbankloc | 8548-8660 | yes | yes | 4 |  |
| floatingDFE | 6070-6104 | floatingDFE | 8793-8827 | yes | yes | 1 |  |
| floating_taps_1sttest | 6105-6221 | floating_taps_1sttest | 8838-8963 | yes | yes | 0 |  |
| force | 6222-6385 | force | 8993-9148 | yes | yes | 0 |  |
| get_ILN | 6386-6401 | get_ILN | 9149-9202 | yes | yes | 0 |  |
| get_ILN_cmp_td | 6402-6511 | get_ILN_cmp_td | 9300-9442 | yes | yes | 0 |  |
| get_PSDs | 6512-6784 | get_PSDs | 9494-9794 | yes | yes | 0 | BUG-03 unfounded (2026-07-01 re-check) |
| get_PulseR | 6785-6815 | get_PulseR | 9795-9821 | yes | yes | 1 |  |
| get_RAW_FIR | 6816-6823 | get_RAW_FIR | 9822-9858 | yes | yes | 1 |  |
| get_RILN_cmp_td | 6824-7005 | get_RILN_cmp_td | 9915-10100 | yes | yes | 0 |  |
| get_StepR | 7006-7033 | get_StepR | 10101-10152 | yes | yes | 1 |  |
| get_TDR | 7034-7350 | get_TDR | 10231-10509 | yes | yes | 1 |  |
| get_TD_files | 7351-7449 | get_TD_files | 10510-10595 | yes | yes | 0 |  |
| get_center_of_UI | 7450-7460 | get_center_of_UI | 10596-10613 | yes | yes | 2 |  |
| get_cm_noise | 7461-7506 | get_cm_noise | 10700-10778 | yes | yes | 1 |  |
| get_pdf | 7507-7602 | get_pdf | 10864-10967 | yes | yes | 1 |  |
| get_pdf_from_sampled_signal | 7603-7649 | get_pdf_from_sampled_signal | 11037-11116 | yes | yes | 11 |  |
| get_pdf_full | 7650-7793 | get_pdf_full | 11197-11365 | yes | yes | 1 |  |
| get_s4p_files | 7794-7925 | get_s4p_files | 11366-11446 | yes | yes | 0 |  |
| get_sigma_eta_ACCM_noise | 7926-7949 | get_sigma_eta_ACCM_noise | 11447-11488 | yes | yes | 0 | BUG-04 unfounded (2026-07-01 re-check) |
| get_sigma_noise | 7950-7973 | get_sigma_noise | 11489-11552 | yes | yes | 1 |  |
| get_xtlk_noise | 7974-8068 | get_xtlk_noise | 11553-11676 | yes | yes | 0 |  |
| hrem | 8069-8079 | hrem | 11677-11724 | yes | yes | 1 |  |
| interp_Sparam | 8080-8295 | interp_Sparam | 11739-11997 | yes | yes | 1 |  |
| make_full_pkg | 8296-8488 | make_full_pkg | 12088-12269 | yes | yes | 2 |  |
| make_pkg | 8489-8535 | make_pkg | 12297-12369 | yes | yes | 3 |  |
| missingParameter | 8536-8539 | missingParameter | 12370-12399 | yes | yes | 0 |  |
| normal_dist | 8540-8546 | normal_dist | 12400-12446 | yes | yes | 5 |  |
| optimize_fom | 8547-8923 | optimize_fom | 12470-12858 | yes | yes | 0 | BUG-07 unfounded (2026-07-01 re-check) |
| parameter_size_adjustment | 8924-8975 | parameter_size_adjustment | 12859-12937 | yes | yes | 0 |  |
| pdf2sgm | 8976-8982 | pdf2sgm | 12938-12972 | yes | yes | 1 |  |
| pdf_to_cdf | 8983-8994 | pdf_to_cdf | 12973-13007 | yes | yes | 1 |  |
| plot_bathtub_curves | 8995-9043 | plot_bathtub_curves | 13048-13108 | yes | yes | 1 |  |
| plot_modal | 9044-9140 | plot_modal | 13138-13191 | yes | yes | 1 |  |
| plot_pie_com | 9141-9222 | plot_pie_com | 13192-13287 | yes | yes | 0 |  |
| process_sxp | 9223-9519 | process_sxp | 13377-13576 | yes | yes | 0 |  |
| r_parrelell2 | 9520-9530 | r_parrelell2 | 13577-13602 | yes | yes | 2 |  |
| rangelimit | 9531-9543 | rangelimit | 13603-13634 | yes | yes | 2 |  |
| read_Nport_touchstone | 9544-9704 | read_Nport_touchstone | 13698-13834 | yes | yes | 2 |  |
| read_PR_files | 9705-9736 | read_PR_files | 13835-13896 | yes | yes | 0 |  |
| read_ParamConfigFile | 9737-10394 | read_ParamConfigFile | 14151-14762 | yes | yes | 0 |  |
| read_p2_s2params | 10395-10526 | read_p2_s2params | 14865-14927 | yes | yes | 1 |  |
| read_p4_s4params | 10527-10659 | read_p4_s4params | 15094-15199 | yes | yes | 1 |  |
| read_package_parameters | 10660-10728 | read_package_parameters | 15213-15301 | yes | yes | 0 |  |
| read_s4p_files | 10729-10909 | read_s4p_files | 16060-16225 | yes | yes | 0 |  |
| readdataSnPx | 10910-11026 | readdataSnPx | 16226-16348 | yes | yes | 0 |  |
| recolor_plots | 11027-11041 | recolor_plots | 16349-16389 | yes | yes | 0 |  |
| s21_pkg | 11042-11217 | s21_pkg | 16626-16836 | yes | yes | 1 |  |
| s21_to_impulse_DC | 11218-11288 | s21_to_impulse_DC | 16992-17097 | yes | yes | 5 | BUG-02 unfounded (2026-07-01 re-check) |
| s_for_c2 | 11289-11296 | s_for_c2 | 17098-17148 | yes | yes | 0 |  |
| s_for_c4 | 11297-11306 | s_for_c4 | 17149-17178 | yes | yes | 0 |  |
| save_cmd_line | 11307-11316 | save_cmd_line | 17179-17197 | yes | yes | 0 |  |
| savefigs | 11317-11357 | savefigs | 17198-17248 | yes | yes | 0 |  |
| savefigs_png | 11358-11398 | savefigs_png | 17249-17307 | yes | yes | 0 |  |
| scaleCDF | 11399-11409 | scaleCDF | 17323-17353 | yes | yes | 1 |  |
| scalePDF | 11410-11417 | scalePDF | 17354-17397 | yes | yes | 1 |  |
| stot | 11418-11425 | stot | 17398-17448 | yes | yes | 0 |  |
| str2csv | 11426-11433 | str2csv | 17449-17474 | yes | yes | 1 |  |
| synth_tline | 11434-11459 | synth_tline | 17475-17542 | yes | yes | 6 |  |
| ttos | 11460-11467 | ttos | 17543-17593 | yes | yes | 0 |  |
| varargin_extractor | 11468-11478 | varargin_extractor | 17594-17608 | yes | yes | 0 |  |
| vma | 11479-11507 | vma | 17664-17727 | yes | yes | 1 |  |
| vref_intersect | 11508-11525 | vref_intersect | 17728-17755 | yes | yes | 1 |  |
| writecsv_transposed | 11526-11549 | writecsv_transposed | 17773-17788 | yes | yes | 0 |  |
| xls_parameter | 11550-11635 | xls_parameter | 17789-17827 | yes | yes | 0 |  |
| xls_parameter_txffe | 11636-11657 | xls_parameter_txffe | 17828-17867 | yes | yes | 0 |  |
| zzz_list_of_changes | 11658-11884 | zzz_list_of_changes | 17868-17929 | yes | yes | 0 |  |
| OptFom_Adaptive_Local_Search | adaptive: 2739-2992 | OptFom_Adaptive_Local_Search | 3611-3722 | yes | yes | 0 |  |
| append_csv_row | adaptive: 5157-5187 | append_csv_row | 7870-7915 | yes | yes | 1 |  |
| compute_hard_cap | adaptive: 5788-5794 | compute_hard_cap | 8689-8698 | yes | yes | 1 |  |

## Work queue (risk-ordered, one batch per run)

- **B01** (G1 unit conversions in noise integrals): get_PSDs, get_sigma_eta_ACCM_noise, get_sigma_noise, N_s, S_RN
- **B02** (G2 frequency grid construction): FD_Processing, interp_Sparam, read_s4p_files
- **B03** (G3 impulse/spectrum handling (prime suspects)): s21_to_impulse_DC, COM_FD_to_TD
- **B04** (G4 convolution and DFE): FFE, FFE_Fast, Fract_T_FFE, TD_CTLE, dfe_clipper
- **B05** (G5 S-parameter interpolation and cascade): combines4p, stot, ttos, make_full_pkg, s_for_c2, s_for_c4
- **B06** (G5 S-parameter synthesis + TDR/ERL): synth_tline, get_TDR, TDR_ERL_Processing, R_series2, r_parrelell2
- **B07** (G6 PDF/CDF noise pipeline, leaf level): d_cpdf, conv_fct, Init_PDF_Fast, normal_dist, get_pdf, pdf_to_cdf
- **B08** (G6 PDF/CDF noise pipeline, composite): Create_Noise_PDF, combine_pdf_same_voltage_axis, comb_fct, conv_fct_MeanNotZero, get_pdf_full, get_pdf_from_sampled_signal
- **B09** (G7 cursor and sample indexing): cursor_sample_index, get_center_of_UI, COM_eye_width, vma
- **B10** (G8 rounding + G9 reshape/flatten order, cross-cutting scans): __CROSS_CUTTING_SCAN__mround_vs_np_round, __CROSS_CUTTING_SCAN__reshape_ravel_flatten_order [DONE]
- **B11** (G10 optimizer loops, FOM core): optimize_fom, OptFom_Calc_Hr, OptFom_Adaptive_Local_Search [DONE 2026-07-11]
- **B12** (G10 optimizer loops, MMSE + floating taps): MMSE, MMSE_FOM, FOM_rxffe_floating_taps, Full_Grid_Matrix [DONE 2026-07-11]
- **B13** (coverage completion: 4p14->4p15 delta resolution + spot-checks): com_ieee8023_, read_p4_s4params, read_Nport_touchstone, adjust_Rx_noise_for_quantization, auto_port_order, append_csv_row, compute_hard_cap, FD_CTLE, Butterworth_Filter, Bessel_Thomson_Filter, Raised_Cosine_Filter, Tukey_Window, bessel, get_xtlk_noise [DONE 2026-07-11]

The risk-ordered queue (G1-G10) is complete and B13 resolved the 4p14->4p15
version question for the whole file. 94 lower-risk functions remain
NOT_YET_AUDITED for their Python translation (version-neutral, fn-test-green).
Suggested continuation batches (each a focused translation review):
- **B14** S-parameter file IO: read_Nport_touchstone done; read_p2_s2params, readdataSnPx, process_sxp, get_s4p_files, get_TD_files, read_PR_files, read_package_parameters
- **B15** config parsing: read_ParamConfigFile, xls_parameter, xls_parameter_txffe, COM_CommandLine_Parse
- **B16-B18** OptFom_* orchestration helpers (Build_TXFFE, Calculate_Settings, Compute_CTLE/TXFFE/RxFFE/DFE, Calc_FOM/Noise/Noise_XC, Create_Output, Setup_Sampler_Sweep, Update_Best_*, Find_Sample_Point, Initialize_Loop_Struct, Set_Best_Itick, Itick_BoxSearch, FD_or_TD_Fields)
- **B19** RILN/ILN/RIL metrics: force, get_RILN_cmp_td, RILN_TD, get_ILN_cmp_td, get_ILN, capture_RIL_RILN, TD_FD_fillin, get_PulseR, get_StepR, get_RAW_FIR
- **B20** noise/MLSE: get_cm_noise, SNDR_ref, S_IN, MLSE, MLSE_U1_c_178A, applyDFEbk
- **B21** DFE/tap/eye/causality: floatingDFE, floating_taps_1sttest, findbankloc, RXFFE_Illegal, find_eye_width, calculate_delay_CausalityEnforcement, Burst_Probability_Calc, Bathtub_Contribution_Wrapper
- **B22** package/board + misc: make_pkg, add_brd, add_brdorig, add_pkg_with_die, SL, Tx_FFE_Filter, parameter_size_adjustment, Bread_Crumb_Chdata_Reduction, Apply_EQ, H_interp, PRBS13Q, rangelimit, s21_pkg, save_cmd_line
- **B23** output/CSV/plot: Output_Arg_Fill, Write_CSV, writecsv_transposed, plot_modal, plot_pie_com, plot_bathtub_curves, end_display_control, savefigs, savefigs_png
- **B24** small leaves + changelog: CDF_ev, CDF_inv_ev, missingParameter, pam, pdf2sgm, scaleCDF, scalePDF, str2csv, recolor_plots, varargin_extractor, vref_intersect, hrem, cdf_to_ber_contour, zzz_list_of_changes, get_center_of_UI, and the top-level `com_ieee8023_` control-flow deep read

A human decision at Gate 13 is needed to (a) continue these batches, (b) open a
separate gated fix pass for the recorded divergences (D1-D18), or (c) accept the
current coverage with the documented caveat.

## Python-only helpers (no MATLAB counterpart, provenance review only)

`_COM_CommandLine_Parse__try_float`, `_FD_Processing__W`, `_FD_Processing__find_idx_ge`, `_FD_Processing__find_idx_le`, `_MLSE_U1_c_178A__scale_pdf`, `_MLSE__qfunc`, `_MLSE__qfuncinv`, `_OptFom_Adaptive_Local_Search__mround`, `_OptFom_Calc_Hr__bessel_poly`, `_OptFom_Calc_Hr__bessel_thomson`, `_OptFom_Calc_Hr__butterworth`, `_OptFom_Calc_Hr__raised_cosine`, `_OptFom_Calc_Hr__tukey_window`, `_OptFom_Calculate_Settings__bessel_poly`, `_OptFom_Calculate_Settings__bessel_thomson`, `_OptFom_Calculate_Settings__butterworth`, `_OptFom_Calculate_Settings__raised_cosine`, `_OptFom_Calculate_Settings__tukey_window`, `_OptFom_Update_BEST_Post_Optimize__bessel_poly`, `_OptFom_Update_BEST_Post_Optimize__tukey_window`, `_Output_Arg_Fill__erfc_inv_approx`, `_Output_Arg_Fill__lfsr`, `_Output_Arg_Fill__strfind_int`, `_PRBS13Q__lfsr`, `_RILN_TD__bessel_poly`, `_Raised_Cosine_Filter__tukey_window`, `_Write_CSV__item_to_str`, `_auto_port_order__mround`, `_com_ieee8023___save_case_outputs`, `_compute_hard_cap__mround`, `_ensure_array`, `_findbankloc__make_badV`, `_floatingDFE__bv`, `_get_PSDs__fold_psd`, `_get_PSDs__to_double_sided`, `_get_TDR__TDR_RL`, `_ns`, `_optimize_fom__sweep_flush`, `_optimize_fom__sweep_record`, `_plot_modal__RLcc178`, `_plot_modal__RLcc_mask`, `_plot_modal__RLdc_mask`, `_plot_modal__dB`, `_process_sxp__build_S_struct`, `_read_Nport_touchstone__mround`, `_read_ParamConfigFile__eval_matlab_value`, `_read_ParamConfigFile__isempty`, `_read_ParamConfigFile__isnan`, `_read_ParamConfigFile__load_csv`, `_read_ParamConfigFile__load_excel`, `_read_ParamConfigFile__load_parameter_sheet`, `_read_ParamConfigFile__parse_cell`, `_read_ParamConfigFile__parse_matlab_matrix`, `_read_ParamConfigFile__parse_matlab_scalar_or_range`, `_read_ParamConfigFile__read_pkg_params`, `_read_ParamConfigFile__xls_param`, `_read_ParamConfigFile__xls_param_txffe`, `_read_p4_s4params__mround`, `_read_package_parameters__xls_param`, `_read_s4p_files__mround`, `_run_com`, `_scaleCDF__scale_pdf`, `_vma__lfsr`, `_vma__strfind_int`, `_writecsv_transposed__val_to_str`, `reset_state`
