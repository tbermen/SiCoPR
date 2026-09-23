"""Layer 4d — writes to a caller's struct before an early return.

A sibling of `tests/test_reference_leaks.py`, for the shape that guard cannot
see.

`test_reference_leaks` finds `<param>.<field> = ...` where the function never
returns that param -- the `process_sxp` signature. Ledger #10b is a different
shape and slipped past it: `OptFom_Calc_Noise` DOES hand `THIS` back, so the
existing rule is satisfied, but it wrote `h_J`, `sigma_TX`, `ISI_N`, `sigma_N`
and `total_noise_rms` into `THIS` *before returning early* on the abort path.

In MATLAB those five are still locals at that point and are only committed to
`THIS` at the end of the function (ML 2932-2946), so an early return leaves the
caller's struct untouched. Python passes by reference, so the caller saw the
ABORTED tick's values where MATLAB sees the last successfully scored tick's.

**The rule.** If a function can return early, it must not have written to a
parameter's attributes on the path to that return. Commit to the caller's struct
once, at the end -- which is what MATLAB does for free.

The rule is over-inclusive by design: a function that legitimately fills a struct
it was handed and then returns early is flagged too. Those go in REVIEWED with
the reason, so the class cannot grow silently.

    python tests/test_abort_path_leaks.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import ast
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from audit_check import check, finish  # noqa: E402

FN = os.path.join(_ROOT, 'com_functions', 'fn')

# key: "<dir>:<function>"  ->  why an early return after writing is safe here.
#
# Five of these are already pinned by tests/test_reference_leaks.py with their
# own reasons, because the parameter is one the function never returns. They are
# repeated here only so this lint's set is complete; the reasoning lives there
# and is not restated.
REVIEWED = {
    # --- accumulators: filling the struct IS the function's job, and it is
    #     returned, so an early return commits exactly what was computed.
    'FD_Processing:FD_Processing':
        'output_args is the output accumulator and is returned',
    'TDR_ERL_Processing:TDR_ERL_Processing':
        'output_args is the output accumulator and is returned',
    'read_ParamConfigFile:read_ParamConfigFile':
        'param/OP ARE the outputs; the whole function exists to fill them',
    'com_ieee8023_:com_ieee8023_':
        'top-level driver; it owns param and OP rather than borrowing them',
    # Neither read_p2_s2params nor read_p4_s4params has a private _rangelimit
    # any more. Both copies carried a by-reference write of param.flim back
    # into the CALLER's object, where MATLAB passes param by value, and both
    # were replaced with the shared function on 2026-09-22. Nothing left here
    # to allow.

    # --- already covered by test_reference_leaks.py (param never returned)
    'optimize_fom:optimize_fom': 'see test_reference_leaks KNOWN set',
    'get_TDR:get_TDR': 'see test_reference_leaks KNOWN set',
    'SNDR_ref:SNDR_ref': 'see test_reference_leaks KNOWN set',
    'OptFom_Compute_RxFFE:OptFom_Compute_RxFFE': 'see test_reference_leaks KNOWN set',

    # --- reviewed here
    'OptFom_Itick_BoxSearch:OptFom_Itick_BoxSearch':
        'BEST.cluster is the box-search state the caller deliberately carries '
        'between ticks; ML keeps it in BEST too, so the write is the point',
}


def _early_returns(fnode):
    """Return nodes that are not the function's final statement."""
    last = fnode.body[-1] if fnode.body else None
    out = []
    for node in ast.walk(fnode):
        if isinstance(node, ast.Return) and node is not last:
            out.append(node)
    return out


def _param_names(fnode):
    a = fnode.args
    names = [x.arg for x in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    # `self` is not a COM struct; and single-letter numeric args are not either,
    # but keeping them costs only a REVIEWED entry if they ever matter.
    return set(n for n in names if n != 'self')


def _param_attr_writes(fnode, params):
    """(lineno, 'param.attr') for each assignment into a parameter's attribute."""
    out = []
    for node in ast.walk(fnode):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) \
                    and t.value.id in params:
                out.append((t.lineno, '%s.%s' % (t.value.id, t.attr)))
    return out


def _rebinds_to_copy(fnode, params):
    """Does the function shadow a param with its own copy? Then it cannot leak."""
    copied = set()
    for node in ast.walk(fnode):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in params:
            src = ast.dump(node.value)
            if 'SimpleNamespace' in src or 'copy' in src or 'deepcopy' in src:
                copied.add(node.targets[0].id)
    return copied


flagged = []
scanned = 0
for d in sorted(os.listdir(FN)):
    p = os.path.join(FN, d, 'py_impl.py')
    if not os.path.isfile(p):
        continue
    src = io.open(p, encoding='utf-8').read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for fnode in [n for n in ast.walk(tree)
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        scanned += 1
        params = _param_names(fnode)
        if not params:
            continue
        early = _early_returns(fnode)
        if not early:
            continue
        safe = _rebinds_to_copy(fnode, params)
        writes = [(ln, nm) for ln, nm in _param_attr_writes(fnode, params)
                  if nm.split('.')[0] not in safe]
        if not writes:
            continue
        first_write = min(ln for ln, _ in writes)
        # only an early return that can be reached AFTER a write matters
        after = [r for r in early if r.lineno > first_write]
        if not after:
            continue
        earliest_return = min(r.lineno for r in after)
        # Report ONLY the writes that precede that return. Listing every write in
        # the function -- including ones after it -- made the first version of
        # this output name fields that are not on the path at all, which is a
        # good way to send someone chasing a non-issue.
        names = sorted({nm for ln, nm in writes if ln < earliest_return})
        if not names:
            continue
        flagged.append(('%s:%s' % (d, fnode.name), first_write,
                        earliest_return, names))

check("functions_were_scanned", scanned > 100,
      "expected to walk the whole engine; only %d functions parsed" % scanned)

unreviewed = ['%s writes %s at line %d, then returns early at line %d'
              % (k, nms, w, r) for k, w, r, nms in flagged if k not in REVIEWED]

check("no_writes_to_a_caller_struct_before_an_early_return",
      not unreviewed,
      "these write into a parameter's attributes and can then return early. "
      "MATLAB's early return leaves the caller's struct untouched because those "
      "values are still locals; Python's does not. Ledger #10b is exactly this. "
      "Move the writes to a single commit at the end, rebind the parameter to a "
      "copy, or add a REVIEWED entry saying why the early return cannot be "
      "reached after the write:\n     " + "\n     ".join(unreviewed))

stale = sorted(set(REVIEWED) - {k for k, _, _, _ in flagged})
check("no_stale_entries_in_the_reviewed_allowlist", not stale,
      "REVIEWED names functions that no longer match the pattern: %s" % stale)

print("\n%d function(s) scanned; %d write to a caller struct before an early "
      "return; %d reviewed." % (scanned, len(flagged), len(REVIEWED)))
for k, w, r, nms in flagged:
    print("   %-46s write line %-5d early return %-5d %s" % (k, w, r, nms))

finish()
