"""`python -m sicopr --exit-on-fail` exits 3 when a package case fails its COM
threshold; without the flag a completed run exits 0, pass or fail.

Runs the CLI block of sicopr.py itself (the `if __name__ == '__main__':` body)
with the engine call replaced by a stub, so the exit-code logic is checked in
seconds without channel data. Added 2026-10-01 (release audit P1-7): CI and
batch scripts had no way to see a failing channel in the exit status.

Run: python tests/test_cli_exit_code.py
"""
# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

import io
import os
import sys
import textwrap
from contextlib import redirect_stdout
from types import SimpleNamespace

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, finish  # noqa: E402
import sicopr  # noqa: E402

src = open(sicopr.__file__, encoding='utf-8').read()
marker = "if __name__ == '__main__':"
check('cli_block_found', marker in src, 'no __main__ block in sicopr.py')
body = textwrap.dedent(src.split(marker, 1)[1])


def run_cli(com_db, flags):
    ns = dict(vars(sicopr))
    ns['_run_com'] = lambda *a, **k: SimpleNamespace(COM_dB=com_db, VEO_mV=1.0, VEC_dB=1.0,
                                                     FOM_ILD=0.1, ICN_mV=0.0, pass_threshold=3.0)
    old = sys.argv
    sys.argv = ['sicopr', 'cfg.xlsx', 'thru.s4p'] + flags
    try:
        with redirect_stdout(io.StringIO()):
            exec(compile(body, 'sicopr_cli', 'exec'), ns)
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    finally:
        sys.argv = old


check('fail_without_flag_exits_0', run_cli(2.5, []) == 0)
check('fail_with_flag_exits_3', run_cli(2.5, ['--exit-on-fail']) == 3,
      'a failing case with --exit-on-fail must exit 3 (2 is argparse usage error)')
check('pass_with_flag_exits_0', run_cli(3.5, ['--exit-on-fail']) == 0)
finish()
