# What this example should produce

Generated from the runs behind the 1368-case benchmark, on the configuration workbook in this directory. Two engines, two conditions.

**Provenance of the COM Octave results.** The Reference Code release file `matlab/com_ieee8023_4p16p0.m`, made to run under GNU Octave 11.3.0 as `octave/com_ieee8023_4p16p0_octave_compat.m` by `octave/make_octave_compat.py` (SiCoPR revision `a130aca`; the 2026-09-20 runs before it differ only in a patch comment), run 2026-09-20. Measured against the MATLAB reference results for this case: COM within 7.1e-15 dB without crosstalk and 8.9e-16 dB with it, FOM within 4.8e-12 dB, the same sampling phase. They are COM Octave results, not the MATLAB reference results, which are not published here.

| quantity | SiCoPR, no crosstalk | COM Octave, no crosstalk | SiCoPR, with crosstalk | COM Octave, with crosstalk |
|---|---|---|---|---|
| `COM_dB` | 3.507002840250303 | 3.507002840250293 | 2.8958986848732575 | 2.8958986848732593 |
| `FOM` | 12.69786661399972 | 12.697866613999125 | 12.094255868167654 | 12.094255868167828 |
| `itick` | -6 | -6.0 | -6 | -6.0 |
| `ERL` | 10.463367429754786 | 10.463367429754785 | 10.463367429754786 | 10.463367429754785 |
| `VEC_dB` | 9.572145874933105 | 9.572145874933126 | 10.948376227366536 | 10.94837622736653 |
| `VEO_mV` | 2.389899105043428 | 2.389899105043924 | 1.7305214778934301 | 1.7305214778931666 |
| `ICN_mV` | 0.0 | 0.0 | 1.0173928373572119 | 1.0173928373572099 |
| `CTLE_DC_gain_dB` | -20.0 | -20.0 | -20.0 | -20.0 |
| `TXLE_taps` | [ 0.   -0.02  0.98  0.  ] | [0 -0.02 0.98 0] | [0. 0. 1. 0.] | [0 0 1 0] |
| `DFE_taps` | 0.7956462712960669 | 0.7956462712958841 | 0.8420108271485084 | 0.8420108271486482 |

The two engines share no code: SiCoPR is a Python port, COM Octave is the reference MATLAB file made to run under Octave. On this channel they differ by 1.0e-14 dB without crosstalk and 1.8e-15 dB with it, which is double-precision arithmetic noise.

**What the numbers say about the channel.** Without crosstalk it makes 3.51 dB, 0.51 dB above the 3 dB threshold. Add the 5 aggressors and it drops to 2.90 dB, which does not meet it. Same channel, same configuration: the aggressors cost 0.61 dB, and turn a pass into a fail. That is the example working as an example — a comfortable pass would teach less.

**Roughly how long.** On one modern desktop core, SiCoPR took 360 s without crosstalk and 336 s with it (measured 2026-09-26). COM Octave takes longer, more so without its compiled kernels built; `octave/README.md` has its timings.

Every column both engines report is in the `expected_*.csv` files beside this one.

