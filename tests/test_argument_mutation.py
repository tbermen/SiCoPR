"""In-place mutation of an argument the function never returns.

Same defect class as tests/test_reference_leaks.py, which is the companion to
this file: MATLAB passes by value, so a MATLAB function CANNOT change its
caller's variable, while the identical Python line writes straight through the
reference. That class produced five of the eight engine defects found by the
208-case MATLAB correlation.

test_reference_leaks.py matches one syntactic form -- `param.field = x` where
`param` is a plain name. That is the shape process_sxp had. It is not the only
shape the divergence takes, and the others are invisible to it:

    subscript      h[i] = 0                  numpy element write
    aug-subscript  h[i] += 1
    augassign      x += 1                    in-place for ndarray, rebind for a scalar
    nested-attr    param.pkg.C = x           base is an Attribute, not a Name
    method         a.sort(), a.fill()        in-place numpy/list methods
    np func        np.place(a, ...), out=a

A site is NOT reported when the parameter is returned, or when it is rebound to
a fresh object before the write -- `chdata = list(chdata)` and
`OP = SimpleNamespace(**vars(OP))` are the remedy, so applying the remedy is
how a site leaves this list.

As with the companion test, this does not claim every listed site is a bug.
It pins the accepted set so a MATLAB update that introduces a new one fails
here, rather than being found months later by another correlation run.

Run: python tests/test_argument_mutation.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import ast
import glob
import io
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_check import check, finish  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Methods that mutate the receiver in place (numpy arrays, lists, dicts, sets).
MUT_METHODS = {
    'sort', 'fill', 'resize', 'put', 'itemset', 'partition', 'byteswap',
    'setfield', 'append', 'extend', 'insert', 'remove', 'pop', 'clear',
    'update', 'setdefault', 'popitem', 'add', 'discard',
}
# numpy functions whose FIRST argument is written in place.
MUT_NPFUNCS = {'place', 'put', 'putmask', 'copyto', 'fill_diagonal',
               'put_along_axis'}

# Accepted sites, each with the reason it is not a defect. Anything not here
# fails the run.
KNOWN = {
    ('COM_CommandLine_Parse', '_pop', 'args'):
        'args is a fresh local list, built at COM_CommandLine_Parse L81 as '
        'list(varargin), so args.pop(0) cannot reach the caller. This is the '
        'varargin_extractor idiom, where consuming the leading argument is the '
        'whole point of the helper.',
    ('COM_eye_width', 'COM_eye_width', 'chdata'):
        'timing_bathtub is a SiCoPR-only plotting side-channel with no MATLAB '
        'counterpart (grep: absent from com_ieee8023_4p16p0.m). com_plots.py '
        'and com_mat_export.py read it off chdata[0]; the driver calls '
        'COM_eye_width once at the end specifically to populate it. No '
        'reported value depends on it.',
}


def _base_name(node):
    """Walk Subscript/Attribute down to the root Name, or None."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _returned(fn):
    """Names appearing in a return, by their root -- `return chdata[0]` counts."""
    out = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and n.value is not None:
            vals = (n.value.elts
                    if isinstance(n.value, (ast.Tuple, ast.List)) else [n.value])
            for v in vals:
                b = _base_name(v)
                if b:
                    out.add(b)
    return out


def _rebound(fn):
    """{name: first line where it is bound to a different object}.

    An AugAssign is deliberately NOT a rebind: `x += 1` on an ndarray mutates
    in place, which is the very thing being looked for.
    """
    out = {}
    for n in ast.walk(fn):
        tgts = []
        if isinstance(n, ast.Assign):
            tgts = n.targets
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            tgts = [n.target]
        elif isinstance(n, ast.For):
            tgts = [n.target]
        for t in tgts:
            if isinstance(t, ast.Name):
                out[t.id] = min(out.get(t.id, n.lineno), n.lineno)
    return out


def _scan_fn(d, fn):
    params = ({a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
              | ({fn.args.vararg.arg} if fn.args.vararg else set())) - {'self'}
    if not params:
        return []
    rets, rb = _returned(fn), _rebound(fn)
    hits = defaultdict(list)

    def note(name, kind, line, text):
        if name not in params or name in rets:
            return
        if name in rb and rb[name] < line:
            return                       # bound to a fresh object first
        hits[name].append((kind, line, text[:70]))

    for n in ast.walk(fn):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                b = _base_name(t)
                if b is None:
                    continue
                if isinstance(t, ast.Subscript):
                    note(b, 'subscript', n.lineno, ast.unparse(n))
                elif (isinstance(t, ast.Attribute)
                      and not isinstance(t.value, ast.Name)):
                    # plain `param.field = x` belongs to test_reference_leaks.py;
                    # this is the nested form it cannot see.
                    note(b, 'nested-attr', n.lineno, ast.unparse(n))
        elif isinstance(n, ast.AugAssign):
            b = _base_name(n.target)
            if b is not None:
                note(b, 'augassign' if isinstance(n.target, ast.Name)
                     else 'aug-subscript', n.lineno, ast.unparse(n))
        elif isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                if f.attr in MUT_METHODS:
                    b = _base_name(f.value)
                    if b is not None:
                        note(b, 'method:' + f.attr, n.lineno, ast.unparse(n))
                if (f.attr in MUT_NPFUNCS and isinstance(f.value, ast.Name)
                        and f.value.id == 'np' and n.args):
                    b = _base_name(n.args[0])
                    if b is not None:
                        note(b, 'np.' + f.attr, n.lineno, ast.unparse(n))
            for kw in n.keywords:
                if kw.arg == 'out':
                    b = _base_name(kw.value)
                    if b is not None:
                        note(b, 'out=', n.lineno, ast.unparse(n))
    return [(d, fn.name, p, v) for p, v in sorted(hits.items())]


def scan():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'com_functions', 'fn',
                                              '*', 'py_impl.py'))):
        d = os.path.basename(os.path.dirname(path))
        try:
            tree = ast.parse(io.open(path, encoding='utf-8').read())
        except SyntaxError:
            continue
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            rows += _scan_fn(d, fn)
    return rows


rows = scan()
seen = {(d, f, p) for d, f, p, _ in rows}
evidence = {(d, f, p): ev for d, f, p, ev in rows}

new = sorted(seen - set(KNOWN))
check('no_new_in_place_argument_mutation',
      not new,
      'argument(s) mutated in place but never returned: %s. MATLAB passes by '
      'value, so if the MATLAB original does not return this argument the write '
      'must not escape. Rebind to a fresh object first (chdata = list(chdata); '
      'chdata[0] = SimpleNamespace(**vars(chdata[0])), as OptFom_Calc_FOM '
      'does), or add it to KNOWN with the MATLAB signature that justifies it. '
      'Evidence: %s'
      % (new, {k: evidence[k][:3] for k in new}))

gone = sorted(set(KNOWN) - seen)
check('accepted_mutation_list_is_current',
      not gone,
      'these sites no longer mutate their argument: %s -- drop them from KNOWN '
      'so the list keeps describing the code' % (gone,))

# The remedy must stay where it was applied. OptFom_Calc_FOM writes
# chdata(1).eq_pulse_response, a field get_PSDs reads back (MATLAB L547).
_src = io.open(os.path.join(ROOT, 'com_functions', 'fn', 'OptFom_Calc_FOM',
                            'py_impl.py'), encoding='utf-8').read()
_fn = [n for n in ast.walk(ast.parse(_src))
       if isinstance(n, ast.FunctionDef) and n.name == 'OptFom_Calc_FOM'][0]
_copy_line = min((n.lineno for n in ast.walk(_fn)
                  if isinstance(n, ast.Assign) and len(n.targets) == 1
                  and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id == 'chdata'), default=None)
_write_line = min((n.lineno for n in ast.walk(_fn)
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Attribute)
                           and t.attr == 'eq_pulse_response' for t in n.targets)),
                  default=None)
check('OptFom_Calc_FOM_copies_chdata_before_writing_eq_pulse_response',
      _copy_line is not None and _write_line is not None
      and _copy_line < _write_line,
      'the chdata copy (line %s) must precede the eq_pulse_response write '
      '(line %s). MATLAB L3097 returns only [FOM, skip_loop], so the write is '
      'local there; get_PSDs reads chdata(1).eq_pulse_response back at L547'
      % (_copy_line, _write_line))

by_kind = defaultdict(int)
for _d, _f, _p, ev in rows:
    for k, _l, _t in ev:
        by_kind[k.split(':')[0]] += 1
print('\nscanned %d py_impl files: %d mutating site(s)%s'
      % (len(glob.glob(os.path.join(ROOT, 'com_functions', 'fn', '*',
                                    'py_impl.py'))),
         len(rows),
         (' -- ' + ', '.join('%s %d' % kv for kv in sorted(by_kind.items())))
         if by_kind else ''))
finish()
