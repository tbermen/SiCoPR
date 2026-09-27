# examples/

One channel, one configuration, four runs: COM computed with and without
crosstalk, by both engines in this repository. It exists so that a reader can
get a number out of this tool on their first evening, and then check that number
against one somebody else already got.

```powershell
# 1. fetch the channel (see akinwale_CR_22dB_VendorX/CHANNEL.md — one zip from ieee802.org)
# 2. run it, and have the run check itself
python examples/run_example.py --channels <where you unpacked the zip>
```

That runs both engines in both conditions, compares what they produce against
the values shipped beside the configuration, and prints anything that differs.
`--engine sicopr` or `--engine octave` runs one side; `--condition no_crosstalk`
runs the quick one. The Octave side is skipped if `octave-cli` is not on PATH.

Expect about six minutes per SiCoPR run on one core, longer under Octave.
Nothing is written into the repository.

## What is here, and what is not

| | |
|---|---|
| `akinwale_CR_22dB_VendorX/` | the configuration workbook, the expected results beside it, and `CHANNEL.md` naming the channel files |
| `run_example.py` | fetches nothing; runs the case and checks the result |

**The channel files are not here.** They are an IEEE 802.3 contribution: public,
and not ours to redistribute. `CHANNEL.md` says which contribution, where to
download it, which six files to use, and gives the SHA-256 of each, so you can
confirm you are running on exactly what produced the expected values. Everything
else — the workbook, which carries its own BSD-3-Clause notice, and the results —
ships here.

## Running it by hand instead

The script is a convenience, not a layer. The same two runs, typed out:

```powershell
# SiCoPR, no crosstalk
python -m sicopr examples/akinwale_CR_22dB_VendorX/config_com_dj_200G_CAKR_178_PKGA_06_2_2025__Case1.xlsx ^
       <channels>/Tx_PCB_4dB_OSFP_22dB_OSFP_4dB_PCB_Rx_TP0_TP5_VendorX_thru1.s4p ^
       --matlab-version 4p16p0

# COM Octave, same case: convert the workbook once, then call the release file
python tools/xlsx_to_com_mat.py examples/akinwale_CR_22dB_VendorX/config_..._Case1.xlsx -o config.mat ^
       --set RESULT_DIR=out/ --set SAVE_FIGURES=0 --set DISPLAY_WINDOW=0
octave-cli --no-gui --no-window-system --eval ^
  "addpath('octave'); r = com_ieee8023_4p16p0_octave_compat('config.mat', 0, 0, '<channels>/..._thru1.s4p'); save('-v7','r.mat','r')"
```

`--matlab-version 4p16p0` is the engine's default and is spelled out so the
command still means the same thing if the default ever changes (README §8). Add the
five aggressors with `--fext` and `--next` for the crosstalk case; the file roles
are in `CHANNEL.md`.

## Why this channel

It is one of the 1368 cases in the correlation corpus, so its numbers are not
special-cased for the example: the same configuration and channel appear in the
benchmark both engines were run over. It also teaches something. Without
crosstalk it makes 3.51 dB, just above the 3 dB threshold. With its five
aggressors it makes 2.90 dB, and does not. A channel that passed comfortably
either way would be a worse example.

[`akinwale_CR_22dB_VendorX/EXPECTED.md`](akinwale_CR_22dB_VendorX/EXPECTED.md)
has both engines' numbers side by side, and what they agree to.
