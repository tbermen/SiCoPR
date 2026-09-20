# What this example should produce

Generated from the runs behind the 1368-case benchmark, on the configuration workbook in this directory. Two engines, two conditions.

| quantity | SiCoPR, no crosstalk | COM Octave, no crosstalk | SiCoPR, with crosstalk | COM Octave, with crosstalk |
|---|---|---|---|---|
| `COM_dB` | 3.507002840250303 | 3.5070028402502986 | 2.8958986848732575 | 2.895898684873259 |
| `FOM` | 12.69786661399972 | 12.697866613999324 | 12.094255868167654 | 12.09425586816722 |
| `itick` | -6 | -6.0 | -6 | -6.0 |
| `ERL` | 10.463367429754786 | 10.463367429754785 | 10.463367429754786 | 10.463367429754785 |
| `VEC_dB` | 9.572145874933105 | 9.572145874933117 | 10.948376227366536 | 10.948376227366534 |
| `VEO_mV` | 2.389899105043428 | 2.3898991050439378 | 1.7305214778934301 | 1.730521477893189 |
| `ICN_mV` | 0.0 | 0.0 | 1.0173928373572119 | 1.01739283735721 |
| `CTLE_DC_gain_dB` | -20.0 | -20.0 | -20.0 | -20.0 |
| `TXLE_taps` | [ 0.   -0.02  0.98  0.  ] | [0 -0.02 0.98 0] | [0. 0. 1. 0.] | [0 0 1 0] |
| `DFE_taps` | 0.7956462712960669 | 0.7956462712958845 | 0.8420108271485084 | 0.8420108271486361 |

The two engines share no code: SiCoPR is a Python port, COM Octave is the reference MATLAB file made to run under Octave. On this channel they differ by 4.4e-15 dB without crosstalk and 1.3e-15 dB with it, which is double-precision arithmetic noise.

**What the numbers say about the channel.** Without crosstalk it makes 3.51 dB, 0.51 dB above the 3 dB threshold. Add the 5 aggressors and it drops to 2.90 dB, which does not meet it. Same channel, same configuration: the aggressors cost 0.61 dB, and turn a pass into a fail. That is the example working as an example — a comfortable pass would teach less.

**Roughly how long.** On one modern desktop core, SiCoPR took 318 s without crosstalk and 479 s with it. COM Octave takes about 1.5 times that with its compiled kernels built, about 2.5 times without.

Every column both engines report is in the `expected_*.csv` files beside this one.

