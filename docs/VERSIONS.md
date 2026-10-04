# MATLAB versions

SiCoPR emulates three releases of the IEEE 802.3 COM Reference Code. This page says
which is the default, how to choose, which evidence belongs to which, and what
differs between them.

## Which release, and how to choose

| release | how to select | source it follows |
|---|---|---|
| **`4p17p0`** (default) | nothing, or `--matlab-version 4p17p0` | `matlab/com_ieee8023_4p17p0.m`, the current IEEE release (4.17.0, 2026-09-23) |
| `4p16p0` | `--matlab-version 4p16p0` | `matlab/com_ieee8023_4p16p0.m` |
| `4p15p0` | `--matlab-version 4p15p0` | `matlab/com_ieee8023_4p15p0_adaptive_local_search.m`: 4p15p0 with the adaptive local search backported, the build the 208-case reference results were produced with |

[`VERSION.json`](../VERSION.json) is the source of truth for this table: the default
release, the supported releases and the reference file for each. Two tests check
that the engine's default matches it.

```powershell
python -m sicopr <config.xlsx> <thru.s4p> --matlab-version 4p15p0     # per run
```

From Python, `sicopr.COM_MATLAB_VERSION = '4p15p0'` does the same. A `COM Version`
keyword in the configuration workbook wins over both.

[`VERSION.json`](../VERSION.json) is the single record of the default and the supported
releases; `assemble_sicopr.py` writes it into `sicopr.py`'s header and the command line's
choices. Every reference file's origin, the tagged release at
https://opensource.ieee.org/802-com/com_code it is byte-identical to, is in
[`../NOTICE`](../NOTICE). `matlab/` also keeps
`com_ieee8023_4p14p0.m`, the original translation source, for history; it is no longer
emulated.

## Which evidence is which version

| corpus | version | compared against | result |
|---|---|---|---|
| 208 cases, 26 IEEE 802.3dj CR/KR channels × 4 package configurations × with/without crosstalk | **4p15p0** | MATLAB reference results | last re-run 2026-09-23: itick and every equalizer selection identical on 208 / 208, COM within 4.6e-14 dB |
| 208 cases, the same corpus | **4p17p0** | COM Octave | 2026-10-04: COM within 5.2e-14 dB (median 8.9e-15), FOM 6.0e-12 dB, itick, CTLE gain, Tx FFE and ERL identical on 208 / 208 ([`../benchmark/208_case_4p17p0/`](../benchmark/208_case_4p17p0/README.md)); against the same corpus's 4p15p0 COM Octave results, COM within 5.0e-14 dB and itick identical on all 208 |
| 1368 cases, 171 distinct channels | **4p16p0** | COM Octave (the Reference Code run under Octave, [`../octave/README.md`](../octave/README.md)) | 2026-09-26: COM within 5.3e-14 dB, itick, Tx FFE and CTLE gain identical on all 1368 |

Reproducing the 208-case result therefore takes `--matlab-version 4p15p0`. Detail is in
[`../MATLAB_Correlation_Review.md`](../MATLAB_Correlation_Review.md).

The 208-case corpus has also been run in 4p16p0 mode: 210 of 213 output columns are
identical on all 208 cases, and no COM, FOM, VEO, VEC, itick or ERL value moves. That
measurement predates the September 2026 oracle fixes and has not been re-run since.

## What 4p17p0 changes

4p17p0 is a small delta from 4p16p0: 149 function bodies unchanged, 7 changed, 4 added,
none removed. Full detail, with the reference behaviour reproduced as it stands:
[`MATLAB_4p17p0_CHANGES.md`](MATLAB_4p17p0_CHANGES.md).

| change | measured effect |
|---|---|
| `MMSE_FOM` builds its Gram matrix by lag from `H(:,1)'*H`, formed once in `MMSE` | equal to `H'*H` on every `H` `MMSE` builds, so results move by summation order only (~1e-14 dB) |
| apparent channel bandwidth: `get_ACBW` and three helpers, reported as `ACBW_GHz` | **off by default**; a workbook turns it on with `ACBW = 1` (threshold `T_dev`) |
| `get_ILN` returns its fit coefficients | no caller reads them |
| a `msgbox` in `MLSE_U1_c_178A` commented out | none in the port, which never had one |

Every 4p16p0 change carries into 4p17p0 (the `Clip Method` default, the pulse and step
scaling, the guards), since 4p17p0 left those functions unchanged.

## What 4p16p0 changes

4p16p0 is a small delta from 4p15p0: 146 of 152 function bodies are unchanged, 6 changed,
3 added, none removed. The three additions are `OptFom_Adaptive_Local_Search`,
`compute_hard_cap` and `append_csv_row`: the adaptive local search, adopted into the
released mainline rather than living in a branch.

| change | measured effect |
|---|---|
| pulse/step now scaled by channel amplitude `A` | `peak_uneq_pulse_mV`, `steady_state_voltage_mV` × A on all 208 cases. **COM, FOM, VEO, VEC, itick, ERL untouched** |
| `Clip Method` default `Fast` → `Slow` | **COM +0.007 dB, FOM +0.22 dB**, but only for configs that omit the keyword. All 208 reference configs set it |
| `min_radius` 1 → 2 in adaptive search | bit-identical answer, **4.3× the candidate evaluations, 2.5× the runtime** |
| four new step responses | additive fields |
| common-mode / TDR degenerate guards | never fired on any input tested |
| `OptFom_Create_Output`, `get_PSDs` edits | numerically neutral |

Inside the port these changes follow the selected version; the `Clip Method` default,
for example, is `Slow` under 4p16p0 and `Fast` under 4p15p0 when the workbook omits the
keyword. The exception is the adaptive-search radius floor, which needs no switch: the
4p15p0 reference results behave as the 4p16p0 mainline rule (`1` for a single Tx FFE
candidate, `2` otherwise), so the port applies that rule on both paths
([`MIN_RADIUS_ASSUMPTION.md`](MIN_RADIUS_ASSUMPTION.md)).

Full detail: [`MATLAB_4p16p0_CHANGES.md`](MATLAB_4p16p0_CHANGES.md) (the upstream diff) and
[`MATLAB_4p16p0_IMPACT.md`](MATLAB_4p16p0_IMPACT.md) (the measured effect). They were
produced with the version differ, which ships, plus a local sweep-comparison step that
does not:

```powershell
python tools/matlab_version_diff.py matlab/com_ieee8023_4p15p0.m matlab/com_ieee8023_4p16p0.m
python tools/matlab_version_diff.py --self-check    # validates the differ itself
```

For a future release, the same differ lists which `py_impl.py` files to re-check.

## What 4p15p0 added over 4p14p0

The port was first translated from 4p14p0. On top of that base it incorporates, from
4p15p0 and the adaptive-local-search branch:

- **Automatic port-order detection** (`auto_port_order`). When the config's `Port Order` is
  empty, `0`, contains `NaN`, or is not a single 4-element vector (e.g. a per-package
  `[1 2 3 4; 1 3 2 4]` matrix), the differential port order is detected from the
  S-parameters. Validated on the 802.3dj KR channels (resolves the standard `[1 3 2 4]`).
- **Adaptive local search** (`OptFom_Adaptive_Local_Search`), selected by the config
  keyword **`Non-zero Local Search Method`** (`param.NonZeroLSMethod`). It keeps a
  persistent search radius that grows and shrinks with recent FOM improvement, and prunes
  candidates by a weighted L1/L2 tap-space distance from the current best. It carries
  about ten hand-tuned constants (shrink factors, weights, the L2/L1 ratio, the CTLE
  window) whose provenance is not documented in the source.
- **Minor 4p15p0 fixes**: `get_PSDs` crosstalk pad length,
  `adjust_Rx_noise_for_quantization` cursor-tap off-by-one, port-order threading through
  the Touchstone readers.
- **Two pre-existing gaps**, fixed while exercising the 802.3dj config: named/complex
  packages (`param.PKG`) accessed by dict subscript instead of attribute, and a 2-row
  `Port Order` matrix now routing to auto-detection.
