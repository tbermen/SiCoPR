"""Guard against the MATLAB-by-value / Python-by-reference defect class.

MATLAB passes structs BY VALUE. A MATLAB function that writes `OP.foo = x` and
does not return OP changes nothing for its caller. The Python port passes the
same structs BY REFERENCE, so the identical line leaks into every later stage.

This was the most damaging defect class in the port: FIVE of the eight engine
defects found by the 208-case MATLAB correlation were instances of it. The
worst, in `process_sxp`, let a TDR-only impulse-response truncation threshold
(1e-5 instead of the configured 1e-3) escape into the rest of the run and biased
FOM LOW on 95.7% of the reference cases -- while all 876 unit tests stayed
green, because each tests one function in isolation and the damage is done to
the CALLER.

The check: find every `<param>.<field> = ...` where <param> is an argument the
function never returns. That is the process_sxp signature exactly. A site counts
as GUARDED when the function first rebinds the parameter to a copy
(`OP = SimpleNamespace(**vars(OP))`) -- the remedy, now used by process_sxp and
get_RILN_cmp_td.

This does not claim every unguarded site is a bug; several are the top-level
driver legitimately owning state it was handed. It pins the KNOWN set so that a
MATLAB update introducing a new one fails here, rather than being found months
later by another 208-case correlation.

Run: python tests/test_reference_leaks.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import ast
import glob
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_check import check, finish  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sites present and reviewed as of 2026-08-18. Each was checked against the
# MATLAB signature: none of these MATLAB functions return param/OP, so every
# entry is a genuine by-reference divergence -- listed because it is either
# unreachable today or the caller genuinely owns the struct.
KNOWN = {
    ('com_ieee8023_', 'com_ieee8023_', 'param'):
        'top-level driver; owns the config it was handed',
    ('com_ieee8023_', 'com_ieee8023_', 'OP'):
        'top-level driver; owns the option block it was handed',
    ('optimize_fom', 'optimize_fom', 'param'):
        'ML returns only result; mutations persist across package cases',
    ('optimize_fom', 'optimize_fom', 'OP'):
        'ML returns only result; mutations persist across package cases',
    ('get_TDR', 'get_TDR', 'OP'):
        'ML 7034 returns only TDR_results; contained because process_sxp copies OP',
    ('force', 'force', 'param'):
        'ML returns [Vfiltered, Cmod, idx]; current_ffegain is read back by the caller',
    ('SNDR_ref', 'SNDR_ref', 'param'):
        'ML returns only results',
    ('get_cm_noise', 'get_cm_noise', 'OP'):
        'ML returns only results',
    ('end_display_control', 'end_display_control', 'param'):
        'ML returns only msg; display bookkeeping only',
    ('OptFom_Compute_RxFFE', 'OptFom_Compute_RxFFE', 'OP'):
        'ML returns [sbr, THIS, skip_it]',
    ('read_s4p_files', '_read_p2_s2params_inline', 'param'):
        'ML read_p2_s2params returns no param; flim set identically by the caller',
    ('read_s4p_files', '_read_p4_s4params_inline', 'param'):
        'ML read_p4_s4params returns no param; flim set identically by the caller',
}

COPY_MARKERS = ('SimpleNamespace(**vars(', 'copy.copy(', 'deepcopy(', '.copy()')


def _returned(fn):
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and n.value is not None:
            vals = (n.value.elts
                    if isinstance(n.value, (ast.Tuple, ast.List)) else [n.value])
            out |= {v.id for v in vals if isinstance(v, ast.Name)}
    return out


def _guarded(fn):
    """Parameters rebound to a fresh copy before being written."""
    g = set()
    for n in ast.walk(fn):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Call)):
            src = ast.unparse(n.value)
            if any(m in src for m in COPY_MARKERS) and n.targets[0].id in src:
                g.add(n.targets[0].id)
    return g


def scan():
    """Return (unguarded, guarded) lists of (dir, def, param, fields)."""
    unguarded, guarded = [], []
    pattern = os.path.join(ROOT, 'com_functions', 'fn', '*', 'py_impl.py')
    for path in sorted(glob.glob(pattern)):
        d = os.path.basename(os.path.dirname(path))
        try:
            tree = ast.parse(io.open(path, encoding='utf-8').read())
        except SyntaxError:
            continue
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            params = ({a.arg for a in fn.args.args}
                      | {a.arg for a in fn.args.kwonlyargs})
            rets, safe = _returned(fn), _guarded(fn)
            mutated = {}
            for n in ast.walk(fn):
                if not isinstance(n, ast.Assign):
                    continue
                for t in n.targets:
                    if (isinstance(t, ast.Attribute)
                            and isinstance(t.value, ast.Name)
                            and t.value.id in params):
                        mutated.setdefault(t.value.id, set()).add(t.attr)
            for p, fields in mutated.items():
                if p in rets:
                    continue
                rec = (d, fn.name, p, sorted(fields))
                (guarded if p in safe else unguarded).append(rec)
    return unguarded, guarded


unguarded, guarded = scan()
seen = {(d, f, p) for d, f, p, _ in unguarded}

new = sorted(seen - set(KNOWN))
check("no_new_by_reference_leaks",
      not new,
      "NEW mutate-but-never-returned site(s): %s. MATLAB passes structs by "
      "value, so if the MATLAB original does not return this struct, the write "
      "must not escape. Fix by copying first (OP = SimpleNamespace(**vars(OP)), "
      "as process_sxp does), or add it to KNOWN with the MATLAB signature that "
      "justifies it." % (new,))

gone = sorted(set(KNOWN) - seen)
check("known_leak_list_is_current",
      not gone,
      "these sites no longer leak (fixed or removed): %s -- drop them from "
      "KNOWN so the list keeps describing the code" % (gone,))

# The remedy must stay where it was applied.
gset = {(d, f, p) for d, f, p, _ in guarded}
for _d, _f, _p in (('process_sxp', 'process_sxp', 'OP'),
                   ('get_RILN_cmp_td', 'get_RILN_cmp_td', 'OP')):
    check("copy_guard_retained__%s" % _d,
          (_d, _f, _p) in gset,
          "%s no longer copies %s before writing to it -- this is the exact "
          "defect that biased FOM low on 95.7%% of the 208 reference cases"
          % (_d, _p))

# process_sxp's specific payload: the TDR threshold must not escape.
_src = io.open(os.path.join(ROOT, 'com_functions', 'fn', 'process_sxp',
                            'py_impl.py'), encoding='utf-8').read()
_fn = [n for n in ast.walk(ast.parse(_src))
       if isinstance(n, ast.FunctionDef) and n.name == 'process_sxp'][0]
_copy_line = min((n.lineno for n in ast.walk(_fn)
                  if isinstance(n, ast.Assign) and len(n.targets) == 1
                  and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == 'OP'
                  and isinstance(n.value, ast.Call)
                  and 'vars(OP)' in ast.unparse(n.value)), default=None)
_thresh_line = min((n.lineno for n in ast.walk(_fn)
                    if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Attribute)
                            and t.attr == 'impulse_response_truncation_threshold'
                            for t in n.targets)), default=None)
check("process_sxp_copies_OP_before_setting_truncation_threshold",
      _copy_line is not None and _thresh_line is not None
      and _copy_line < _thresh_line,
      "the OP copy (line %s) must precede the truncation-threshold write "
      "(line %s); MATLAB 4p15p0 L9311 says it is 'Only for TDR not returned "
      "out of process_sxp function'" % (_copy_line, _thresh_line))

print("\nscanned: %d unguarded site(s), %d guarded" % (len(unguarded), len(guarded)))
finish()
