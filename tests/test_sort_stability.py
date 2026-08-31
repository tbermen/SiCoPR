"""Layer 4b — sort stability.

MATLAB's `sort` is **stable**: equal elements keep their original relative
order. NumPy's `argsort` defaults to quicksort, which is **not**. Wherever a
tie occurs and the resulting permutation is applied to a companion array, the
two languages can pair values differently and the answer diverges.

Ledger #10 records two such sites, found by the corpus rather than by any test,
and the coverage table has carried "10a unstable sort -- needs the argsort lint"
as an open item ever since. This is that lint.

**The rule.** `np.argsort` must pass `kind='stable'` UNLESS its permutation is
applied only to the array it was computed from -- in which case it is just
`np.sort` and ties are indistinguishable by construction.

`np.sort` itself is never flagged. Sorting values without carrying a companion
cannot expose a tie: equal elements are interchangeable, so stability is not
observable.

Each site the rule flags is either fixed or listed in REVIEWED below with the
reason it is safe. A site that is neither fails this test, so the class cannot
grow silently.

    python tests/test_sort_stability.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import ast
import io
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from audit_check import check, finish  # noqa: E402

FN = os.path.join(_ROOT, 'com_functions', 'fn')

# Sites whose permutation IS applied to a companion array, reviewed one by one.
# Key is "<function dir>:<line of the argsort>". The reason must say why a tie
# cannot change the answer -- "looks fine" is not a reason.
# Every argsort in the engine now passes kind='stable', so nothing needs
# excusing and this is empty. It is kept because the mechanism matters: a future
# site that cannot be made stable belongs here WITH the reason a tie cannot be
# observed, and the staleness check below stops the list outliving its entries.
#
# It was not always empty. When this lint was first written, 28 of 39 calls were
# unstable and 27 of those co-sorted a companion array. The first instinct was to
# allowlist them on the grounds that ties are rare -- so that was measured
# instead: instrumenting np.argsort over one real case found ties in **81% of
# calls**, 10,624 tied adjacent pairs. "Ties are rare" was simply false, and the
# sites were made stable rather than excused.
REVIEWED = {}


def _argsort_calls(tree):
    """Yield (lineno, node) for every np.argsort / x.argsort call."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = None
        if isinstance(f, ast.Attribute):
            name = f.attr
        if name != 'argsort':
            continue
        yield node


def _has_stable(node):
    for kw in node.keywords:
        if kw.arg == 'kind' and isinstance(kw.value, ast.Constant) \
                and kw.value.value == 'stable':
            return True
    return False


def _sorted_expr_name(node):
    """Name of the array being argsorted, if it is a simple name."""
    if node.args:
        a = node.args[0]
        if isinstance(a, ast.Name):
            return a.id
        # np.argsort(-x) / np.argsort(np.abs(x)) -- unwrap one level
        if isinstance(a, ast.UnaryOp) and isinstance(a.operand, ast.Name):
            return a.operand.id
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return node.func.value.id
    return None


unstable = []      # (fn_dir, lineno, source line)
total = 0
for d in sorted(os.listdir(FN)):
    p = os.path.join(FN, d, 'py_impl.py')
    if not os.path.isfile(p):
        continue
    src = io.open(p, encoding='utf-8').read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    lines = src.split('\n')
    for node in _argsort_calls(tree):
        total += 1
        if _has_stable(node):
            continue
        unstable.append((d, node.lineno, lines[node.lineno - 1].strip()))

check("engine_contains_sort_calls_to_check", total > 0,
      "found no argsort calls at all; the lint is looking in the wrong place")

# Which of the unstable ones apply their permutation to a companion array?
# Heuristic, deliberately over-inclusive: if the two lines after the argsort
# index anything other than the sorted array itself, treat it as a companion.
risky = []
for d, ln, text in unstable:
    p = os.path.join(FN, d, 'py_impl.py')
    lines = io.open(p, encoding='utf-8').read().split('\n')
    m = re.search(r'argsort\(\s*(?:-\s*)?(?:np\.abs\()?\s*([A-Za-z_]\w*)', text)
    base = m.group(1) if m else None
    window = ' '.join(lines[ln:ln + 3])
    idx_names = set(re.findall(r'([A-Za-z_]\w*)\s*\[\s*(?:si|sort_idx|idx|order|val_sort)\b',
                               window))
    companions = {n for n in idx_names if n != base}
    if companions:
        risky.append((d, ln, sorted(companions), text[:70]))

# Anything risky must be reviewed. Match on directory + enclosing helper name
# so the key survives line-number drift.
unreviewed = []
for d, ln, comps, text in risky:
    p = os.path.join(FN, d, 'py_impl.py')
    src_lines = io.open(p, encoding='utf-8').read().split('\n')
    func = '?'
    for i in range(ln - 1, -1, -1):
        mm = re.match(r'\s*def\s+(\w+)', src_lines[i])
        if mm:
            func = mm.group(1)
            break
    key = '%s:%s' % (d, func)
    if key not in REVIEWED:
        unreviewed.append('%s line %d  companions=%s  %s' % (key, ln, comps, text))

check("every_unstable_argsort_with_a_companion_is_reviewed",
      not unreviewed,
      "these argsort calls sort one array and apply the permutation to another, "
      "without kind='stable'. MATLAB's sort is stable, so a tie can pair values "
      "differently and change the answer -- ledger #10 is exactly this. Add "
      "kind='stable', or add an entry to REVIEWED saying why a tie cannot be "
      "observed:\n     " + "\n     ".join(unreviewed))

# Guard the guard: every REVIEWED key must still correspond to a real site, so
# the allowlist cannot rot into a list of sites that no longer exist.
present = set()
for d, ln, comps, text in risky:
    p = os.path.join(FN, d, 'py_impl.py')
    src_lines = io.open(p, encoding='utf-8').read().split('\n')
    for i in range(ln - 1, -1, -1):
        mm = re.match(r'\s*def\s+(\w+)', src_lines[i])
        if mm:
            present.add('%s:%s' % (d, mm.group(1)))
            break
stale = sorted(set(REVIEWED) - present)
check("no_stale_entries_in_the_reviewed_allowlist",
      not stale,
      "REVIEWED lists sites that no longer exist or no longer carry a companion "
      "sort: %s. Remove them so the allowlist keeps meaning something." % stale)

print("\n%d argsort call(s); %d without kind='stable'; %d of those carry a "
      "companion array; %d reviewed."
      % (total, len(unstable), len(risky), len(REVIEWED)))
for d, ln, comps, text in risky:
    print("   %-32s line %-5d companions %s" % (d, ln, comps))

finish()
