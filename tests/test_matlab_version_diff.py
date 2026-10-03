"""tools/matlab_version_diff.py must see a function whose declaration is continued
with '...'.

Found 2026-10-03 at the 4p17p0 intake: the differ reported one new function where
there were four (get_ACBW and three helpers declared as
`function [a,b] = ...` with the name on the next line). An unrecognised
declaration does not just go missing: its body is merged into the function above,
so a change in it is reported against the wrong function. The same shape had hidden
s21_to_impulse_DC in every release since 4p14p0.

Run: python tests/test_matlab_version_diff.py
"""
# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import tempfile

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_root, 'tools'))

from audit_check import check, finish  # noqa: E402
import matlab_version_diff as mvd  # noqa: E402

SRC = """function y = first(x)
y = x + 1;
function [a, b] = ...
    second(x, y) % continued declaration
a = x;
b = y;
function third(z)
disp(z)
"""
with tempfile.NamedTemporaryFile('w', suffix='.m', delete=False, encoding='utf-8') as f:
    f.write(SRC)
fns = mvd.parse_functions(f.name)
os.unlink(f.name)

check('continued_declaration_is_found', 'second' in fns, sorted(fns))
check('body_above_does_not_swallow_it', fns['first']['end'] == 2,
      'first ends at %s' % fns['first']['end'])
check('continued_function_body_bounds', 'second' in fns and (fns['second']['start'], fns['second']['end']) == (3, 6),
      fns.get('second', {}).get('start'))
check('following_declaration_still_found', 'third' in fns, sorted(fns))

# the real releases
M = os.path.join(_root, 'matlab')
p16 = mvd.parse_functions(os.path.join(M, 'com_ieee8023_4p16p0.m'))
check('s21_to_impulse_DC_found_in_4p16p0', 's21_to_impulse_DC' in p16)
p17_path = os.path.join(M, 'com_ieee8023_4p17p0.m')
if os.path.exists(p17_path):
    p17 = mvd.parse_functions(p17_path)
    for name in ('get_ACBW', 'get_BW_from_CICP_residual', 'get_CICP_fit_residual', 'get_CICP_fit_sweep'):
        check('%s_found_in_4p17p0' % name, name in p17)

finish()
