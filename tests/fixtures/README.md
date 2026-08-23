# tests/fixtures — 802.3ck reference pair (not committed)

This directory is intentionally empty in the repository. The Stage-4 checkpoint
tests and the Stage-5 end-to-end test need an 802.3ck reference channel and its
config, and those two files are not ours to redistribute:

| file | what it is |
|---|---|
| `ieee8023ck_reference.xlsx` | COM config for the 802.3ck compliant host channel, 53.125 GBaud PAM-4 |
| `ieee8023ck_compliant_host_channel.s4p` | the matching Touchstone channel |

Without them, **11 tests skip**. That is the expected default, not a failure.

## Running them

Drop both files here, or point the tests at wherever you keep them:

```bash
COM_TEST_FIXTURES=/path/to/your/fixtures python -m pytest tests -q
```

```powershell
$env:COM_TEST_FIXTURES = "D:\com_fixtures"; python -m pytest tests -q
```

## What these tests are, and are not

They are a **single-channel smoke check** — the pipeline runs end to end and the
intermediate variables land in sensible ranges. `EXPECTED_COM_DB` in
`test_end_to_end.py` is deliberately `None`: no MATLAB run exists for this
particular 802.3ck pair, and filling it in from com.py's own output would make
the assertion circular.

Numeric parity with MATLAB is established elsewhere, and much more thoroughly —
208 reference cases via `tools/matlab_compare.py`. With each condition on the
settings its own reference used, FOM and COM are bit-exact on 208/208 and max
|ΔCOM| is 3.3e-14 dB; on the configs exactly as supplied it is 198/208 and 0.185 dB. See [`MATLAB_Correlation_Review.md`](../../MATLAB_Correlation_Review.md).
Those cases use the 802.3dj channel set (`akinwale_3dj_01_2310/`, 210 MB, also
not committed — see `.gitignore`).
