"""Which lines of the assembled engine a run executes.

    python com_functions/verification/reach.py OUT.json -- <sicopr arguments>

Runs `python -m sicopr <arguments>` in this process under sys.monitoring and
writes the 1-based line numbers of sicopr.py that executed at least once. Each
line reports once and is then switched off, so the run costs about what it
costs untraced.

mutations.py uses this to tell a mutant the checkpoint cases never EXECUTE from
one they execute and fail to notice: the first is a gap in the case set, the
second a gap in what the harness saves or how tightly it compares.

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import json
import os
import runpy
import sys


def main():
    out = sys.argv[1]
    if sys.argv[2] != '--':
        raise SystemExit('usage: reach.py OUT.json -- <sicopr arguments>')
    sys.argv = ['sicopr'] + sys.argv[3:]
    import importlib.util
    target = os.path.normcase(os.path.abspath(
        importlib.util.find_spec('sicopr').origin))
    hit = set()
    mon = sys.monitoring
    tool = mon.PROFILER_ID
    mon.use_tool_id(tool, 'sicopr-reach')

    def on_line(code, line):
        if os.path.normcase(code.co_filename) == target:
            hit.add(line)
        return mon.DISABLE

    mon.register_callback(tool, mon.events.LINE, on_line)
    mon.set_events(tool, mon.events.LINE)
    try:
        runpy.run_module('sicopr', run_name='__main__', alter_sys=True)
    except SystemExit:
        pass
    finally:
        mon.set_events(tool, 0)
        mon.free_tool_id(tool)
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump({'file': target, 'lines': sorted(hit)}, fh)


if __name__ == '__main__':
    main()
