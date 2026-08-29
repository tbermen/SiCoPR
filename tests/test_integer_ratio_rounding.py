"""Layer 4c — rounding a ratio of integers.

MATLAB's `round()` is half-away-from-zero. Python's builtin `round()` and
`np.round()` are half-to-even (banker's). They differ only at an exact `.5`,
which is why the conversion audit dismissed every bare-`round` site as
"measure-zero for continuous data".

**That reasoning is correct for continuous inputs and false for a ratio of two
integers.** `a / b` with `a` and `b` both integral is a rational that lands on
`.5` exactly whenever `a` is an odd multiple of `b/2` -- not measure-zero, a
discrete quantity that really does hit the tie.

Ledger #16 is that defect: `nui = round(len(residual_response) / M)` sat on a
tie in exactly one of 98,145 `round()` calls in a single run, dropped one ISI
sample, and cost 4 of 208 cases their bit-exactness. `AUDIT_FINDINGS.md` carries
a correction saying the original verdict was wrong.

This is the generalisation the correction called for and the status report listed
as unfinished: find every `round()` whose argument is an integer ratio, and
require it to use the half-away helper.

**What is deliberately NOT flagged.** `round(voltage / binsize)` really is
continuous, and the audit's original reasoning holds for it. `round(i + j)` on
integers has no fractional part at all. Flagging those would bury the real
signal.

    python tests/test_integer_ratio_rounding.py
"""
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

# Names that are integer-valued by construction in this engine. Deliberately
# short: every entry is a count, an index or a sample rate, and adding a name
# here widens what the lint flags, so it should be justified.
INT_NAMES = {
    'M', 'samples_per_ui', 'spui', 'ndfe', 'N_b', 'N_bmax', 'N_bg', 'N_bf',
    'nport', 'NumPorts', 'levels', 'npts', 'n', 'N', 'nui', 'L',
    'ffe_pre_tap_len', 'ffe_post_tap_len', 'RxFFE_cmx', 'cursor_i', 't_s',
    'sbr_peak_i', 'tap_bk', 'num_ui', 'fb',
}

# Sites where the ratio is integral but a tie provably cannot occur, or where
# the result is not sensitive to it. Each needs a reason, not an assertion.
REVIEWED = {}


def _is_integral(node):
    """Conservatively: is this expression integer-valued by construction?"""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int)
    if isinstance(node, ast.Call):
        f = node.func
        nm = f.id if isinstance(f, ast.Name) else getattr(f, 'attr', None)
        return nm in ('int', 'len')
    if isinstance(node, ast.Name):
        return node.id in INT_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in INT_NAMES
    if isinstance(node, ast.BinOp):
        # int op int stays integral for +, -, *; division is the case we hunt
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            return _is_integral(node.left) and _is_integral(node.right)
    if isinstance(node, ast.UnaryOp):
        return _is_integral(node.operand)
    return False


def _integer_ratio(node):
    """Find an int/int division anywhere inside this expression."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Div):
            if _is_integral(sub.left) and _is_integral(sub.right):
                return sub
    return None


def _round_calls(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id == 'round':
            yield node, 'round'
        elif isinstance(f, ast.Attribute) and f.attr == 'round' \
                and getattr(f.value, 'id', '') == 'np':
            yield node, 'np.round'


flagged = []
total_round = 0
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
    for node, kind in _round_calls(tree):
        total_round += 1
        if not node.args:
            continue
        ratio = _integer_ratio(node.args[0])
        if ratio is None:
            continue
        # enclosing function, so the key survives line drift
        func = '?'
        for i in range(node.lineno - 1, -1, -1):
            import re as _re
            mm = _re.match(r'\s*def\s+(\w+)', lines[i])
            if mm:
                func = mm.group(1)
                break
        flagged.append(('%s:%s' % (d, func), node.lineno,
                        lines[node.lineno - 1].strip()[:100], kind))

check("engine_contains_round_calls_to_check", total_round > 0,
      "found no round() calls; the lint is looking in the wrong place")

unreviewed = ['%s line %d  [%s]  %s' % (k, ln, kind, text)
              for k, ln, text, kind in flagged if k not in REVIEWED]

check("no_bankers_rounding_on_an_integer_ratio",
      not unreviewed,
      "these round a ratio of two INTEGERS using banker's rounding, where MATLAB "
      "rounds half away from zero. The quotient can land on .5 exactly, so this "
      "is the shape of ledger #16 -- not the measure-zero case the audit "
      "dismissed. Use the half-away helper (_mround), or add a REVIEWED entry "
      "saying why a tie cannot occur here:\n     "
      + "\n     ".join(unreviewed))

stale = sorted(set(REVIEWED) - {k for k, _, _, _ in flagged})
check("no_stale_entries_in_the_reviewed_allowlist", not stale,
      "REVIEWED names sites that no longer round an integer ratio: %s" % stale)

# The half-away helper must actually be half-away, or everything above is moot.
def _mround(x):
    import math
    return int(math.floor(float(x) + 0.5)) if x >= 0 else int(math.ceil(float(x) - 0.5))


ties = [(0.5, 1), (1.5, 2), (2.5, 3), (-0.5, -1), (-1.5, -2), (2360.5, 2361)]
bad = [(v, _mround(v), want) for v, want in ties if _mround(v) != want]
check("half_away_helper_rounds_ties_away_from_zero", not bad,
      "the reference behaviour this lint steers code toward is itself wrong: %s"
      % bad)
bankers = [(v, round(v)) for v, want in ties if round(v) != want]
check("builtin_round_really_does_differ_at_ties", len(bankers) >= 3,
      "builtin round() agrees with half-away on these ties, so the whole "
      "distinction has gone away and this lint guards nothing: %s" % bankers)

print("\n%d round()/np.round() call(s) scanned; %d round an integer ratio; "
      "%d reviewed." % (total_round, len(flagged), len(REVIEWED)))
for k, ln, text, kind in flagged:
    print("   %-40s line %-5d %s" % (k, ln, text[:60]))

finish()
