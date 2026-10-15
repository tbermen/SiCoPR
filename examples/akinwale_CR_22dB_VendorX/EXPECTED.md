# What this example should produce

Generated on the configuration workbook in this directory, one of the 1368 cases in the correlation benchmark. Two engines, two conditions, both on the 4p17p0 release.

**Provenance of the COM Octave results.** The Reference Code release file `matlab/com_ieee8023_4p17p0.m`, made to run under GNU Octave 11.3.0 as `octave/com_ieee8023_4p17p0_octave_compat.m` by `octave/make_octave_compat.py` (SiCoPR revision `bf44414`, compiled kernels on), run 2026-10-14. They are COM Octave results, not MATLAB reference results. Under 4p16p0 the same case was also measured against the MATLAB reference results: COM within 7.1e-15 dB without crosstalk and 8.9e-16 dB with it, the same sampling phase. 4p17p0 changes this calculation only in summation order, and both engines moved by less than 1e-14 dB.

| quantity | SiCoPR, no crosstalk | COM Octave, no crosstalk | SiCoPR, with crosstalk | COM Octave, with crosstalk |
|---|---|---|---|---|
| `COM_dB` | 3.5070028402503017 | 3.5070028402502937 | 2.895898684873251 | 2.895898684873241 |
| `FOM` | 12.697866613999523 | 12.697866613999423 | 12.09425586816679 | 12.094255868167135 |
| `itick` | -6 | -6 | -6 | -6 |
| `ERL` | 10.463367429754785 | 10.463367429754785 | 10.463367429754785 | 10.463367429754785 |
| `VEC_dB` | 9.572145874933112 | 9.572145874933126 | 10.94837622736655 | 10.948376227366577 |
| `VEO_mV` | 2.389899105043246 | 2.389899105043912 | 1.730521477892738 | 1.7305214778931457 |
| `ICN_mV` | 0.0 | 0.0 | 1.0173928373572119 | 1.0173928373572099 |
| `CTLE_DC_gain_dB` | -20.0 | -20.0 | -20.0 | -20.0 |
| `TXLE_taps` | [0 -0.02 0.98 0] | [0 -0.02 0.98 0] | [0 0 1 0] | [0 0 1 0] |
| `DFE_taps` | 0.7956462712961311 | 0.7956462712958905 | 0.8420108271488755 | 0.8420108271486514 |

The two engines share no code: SiCoPR is a Python port, COM Octave is the reference MATLAB file made to run under Octave. On this channel they differ by 8.0e-15 dB without crosstalk and 9.8e-15 dB with it, which is double-precision arithmetic noise.

**What the numbers say about the channel.** Without crosstalk it makes 3.51 dB, 0.51 dB above the 3 dB threshold. Add the 5 aggressors and it drops to 2.90 dB, which does not meet it. Same channel, same configuration: the aggressors cost 0.61 dB, and turn a pass into a fail. That is the example working as an example — a comfortable pass would teach less.

**Roughly how long.** On one modern desktop core (measured 2026-10-14), SiCoPR took 354 s without crosstalk and 318 s with it; COM Octave with its compiled kernels built took 305 s and 333 s. Without the kernels COM Octave takes longer; `octave/README.md` has its timings.

Every column both engines report is in the `expected_*.csv` files beside this one.
