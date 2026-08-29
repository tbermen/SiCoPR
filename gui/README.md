# SiCoPR config editor

A local web app for reading, editing and creating COM configuration workbooks
from a schematic view of the channel.

```
python gui/app.py
```

It serves on `http://127.0.0.1:8765` and opens a browser. Use
`--no-browser` to suppress that, and `SICOPR_GUI_PORT` to change the port.

Stdlib only — nothing to install beyond what the engine already needs.

## What it does

- Draws the channel as blocks: Tx die → Tx package → channel → Rx package →
  Rx die, with the FEXT/NEXT aggressors and the
  Tx FFE → CTLE → Rx FFE → DFE → detector chain.
- The config picker browses **any** directory in the repo that holds
  configuration workbooks, not a fixed list. **New…** creates one from a
  template, optionally resetting every setting to the default the workbook
  itself declares.
- Clicking a block lists the settings that block owns, with the units, the
  sheet's own annotation (`[TX RX]`, `[min:step:max]`), the description from
  the `keywords_*` sheet, the default, the MATLAB variable name, and the cell
  address.
- **`[TX RX]` settings appear under both ends of the link.** `C_d`, `L_s`,
  `C_b` and `PKG_NAME` are one cell holding both halves; the Tx and Rx blocks
  each edit their own half and the halves are recombined on save, in the shape
  the original literal had.
- Edits are tracked, shown on the schematic, and written to a **new** workbook.
- **Run…** picks the input channels, shows the exact command that will be run,
  executes `sicopr.py`, and streams its terminal output live.
- **Results** browses the run directories: per case, the headline numbers
  (COM, VEO, VEC, ERL, FOM, itick, the chosen EQ point) read from
  `results.csv`, and the generated figures grouped by the seven pipeline
  stages. Click a figure to enlarge it. **Refresh** reloads after a run.

## The workflow, without leaving the page

1. Pick a config and edit it on the schematic; **Save as…** a new workbook.
2. **Run…** — choose the channel directory, then the THRU (victim) and any
   FEXT/NEXT aggressors. Roles are pre-filled from the filenames, but every
   file is offered in every role: a filename is a hint, and mistaking a NEXT
   aggressor for the victim would produce a confidently wrong answer.
3. The **netlist** panel shows the resolved command before anything runs:

   ```
   python sicopr.py tests/1_.../config_..._Case1.xlsx akinwale_.../..._thru1.s4p --fext ...
   ```

   If the selection is not runnable, every problem is listed at once rather
   than just the first.
4. **Run sicopr.py** streams stdout and stderr into the output window. **Stop**
   terminates it.
5. **Results → Refresh** loads what the run wrote.

Only one run at a time — a COM run is CPU-heavy and a browser that could stack
them up is a way to fall over, not a feature. The command is assembled as an
argv list from validated repo-relative paths and never goes through a shell.

## "New from defaults" is narrower than it sounds

The `keywords_*` sheet documents a default **per keyword**. That set is not a
validated configuration, and applying it wholesale produces a workbook the
engine refuses to read:

- `PKG_NAME` declares `-`, a placeholder, giving `Package Block "-" not found`.
- `C_d` declares `4e-05 4e-05 9e-05 …` — unbracketed — where the config holds
  `[0.4e-4  0.9e-4  1.1e-4 ; …]`. The bare form reaches the engine as a string
  and the first arithmetic raises `can't multiply sequence by non-int`.
  Measured by bisection: of 47 otherwise-compatible defaults, that one alone
  broke the whole config.
- `g_DC` declares `-52`, which is a stray formula in the keywords sheet, against
  a `[-20:1:0]` sweep.

So a default is applied only when it is the same **shape** as the value it
replaces (number for number, bracketed literal for bracketed literal), and the
result is then handed to the real `read_ParamConfigFile` before you get it. A
config that fails that check is deleted rather than returned. On the stock
config this resets 43 settings and skips 26.

**A re-run may overwrite.** Configs commonly set a date-templated
`RESULT_DIR` (`.\results\CAKR_{date}\`), so two runs on the same day land in
the same directory and the second replaces the first. Change `RESULT_DIR` in
the Run-and-output block if you want to keep both.

## The three things it is careful about

**A sweep is not a number.** A tap whose value cell holds `0` is a fixed
setting; the same tap holding `[ -0.34:.02:0]` is a ~198-point sweep. The units
column contains a range template either way, so the two look alike at a glance.
The editor tags sweeps, preserves the distinction on write, and never coerces
one form into the other. Getting this wrong once cost this project a long
investigation into "missing" Tx FFE search space.

**Nine value cells are spreadsheet formulas.** `f_v`, `f_f`, `f_n`, `TR_TDR`,
`N_tail_start`, `f_z`, `f_p1`, `f_p2` and `f_HP_PZ` are computed from `f_b` and
friends. The engine reads *cached* values, and openpyxl can preserve the formula
or the cached value but not both — so saving through openpyxl would either
flatten every formula or, worse, leave the engine reading `None`. The writer
edits the sheet XML directly instead, changing only the targeted cells.

Because nothing here evaluates formulas, changing a cell a formula depends on
leaves the dependent value stale. That is reported as a `STALE:` warning rather
than silently written; open the file in Excel to recalculate.

**`.START` blocks are separate namespaces.** Column A can carry
`.START <name>` / `.END` markers delimiting package presets, and the engine
parses the main region as everything *before* the first `.START`. That is why
`A_v`, `C_p` and `R_d` each appear four times in a stock config without being
ambiguous. Package keywords are addressed as `BLOCK/name`.

## What is shown, and what is not

A **setting** is one of the 249 keywords `sicopr.py` actually looks up,
extracted from the engine source rather than kept by hand here. The sheet also
holds cells the engine never reads, and they are handled by where they belong
rather than by being swept into one block:

- Units columns and `[min:step:max]` templates are **already carried on the
  setting they annotate**, as its units and note, so they are not repeated as
  phantom fields. An annotation is folded away only when its text provably
  appears on a real field — 19 of 24 in the stock config.
- Reference values that annotate nothing (`T_t`, `T_ft`, `T_nt` — transition
  times read by neither the port nor the MATLAB original) stay as fields in the
  block they belong to, marked *reference only* and not editable.

Between the two, nothing in the workbook disappears; `test_gui_server.py`
asserts it.

## Tests

```
python tests/test_config_roundtrip.py     # the writer  (18 checks)
python tests/test_gui_server.py           # the HTTP layer (71 checks)
python tests/test_gui_static.py           # page assets (7 checks)
```

Both run as part of `tests/run_all.ps1`.

The round-trip test is the gate the rest stands on: for every config in the
repo it reads, rewrites and hands **both** files to the real
`read_ParamConfigFile`, requiring field-for-field identical `param` and `OP`.
Comparing the workbooks to each other would only prove the writer is
self-consistent.

Both suites are mutation-verified — each check has been shown to fail when the
defect it describes is reintroduced.

## Files

| file | role |
| --- | --- |
| `config_io.py` | read/write workbooks; no UI, no engine imports |
| `schematic.py` | which keyword belongs to which block (presentation only) |
| `runner.py` | spawns `sicopr.py` and buffers its output; one run at a time |
| `app.py` | the HTTP server and JSON API |
| `static/` | the page |

`test_gui_static.py` **parses** `app.js` — with `node --check` if a JS engine is
installed, otherwise with `esprima` (pure Python, listed in `requirements.txt`
as test-only). This is not optional rigour: a syntax error in `app.js` breaks
the entire page while every server-side test still passes, because the server is
not the thing that is wrong. That happened once — literal newlines inside a
quoted string left the editor with empty pickers and dead buttons — and the
delimiter-balance check that preceded the parser did not see it. If neither
parser is available the test says so loudly rather than passing quietly.
