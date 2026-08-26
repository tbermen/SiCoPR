"""Index-base conformance: declare each index field's base, then prove every use obeys it.

Four of the fourteen fixes in docs/FIX_SUMMARY.md were index-base errors, and
they share a shape that makes them nearly undetectable by ordinary testing: a
1-based value used as a 0-based subscript reads the WRONG ELEMENT while the
array is long enough, and only raises IndexError when the index happens to land
on the last entry. `BEST.ctle` carried that error from the initial commit until
a sweep put the winning CTLE last -- months of green test runs in between.

So the check cannot be "does it crash" or "does the comment say 0-based" (the
comment on BEST.ctle said 0-based and was wrong). It has to be: **the base is
declared once, and every use is proved against the declaration.**

Three rules, run over the assembled com.py:

  A  a 1-based field reaching a subscript without conversion
  B  an explicit `- 1` applied to a field declared 0-based
  C  an index-shaped struct attribute that is not declared at all

Rule C is what makes this scale. A new feature cannot quietly introduce an
undeclared index: the run fails until someone writes down which base it uses and
cites the MATLAB line it mirrors.

Measured recall when this was built, against the historical versions in git:

  the version before the 8-defect fix   flags t_s in Apply_EQ and cursor_i in optimize_fom  -> defects #11, #7
  the version before the ADC-clip fix   flags t_s, plus ctle and G_high_pass                -> defects #11, #14
  the version before the BEST.ctle fix   flags ctle and G_high_pass in post-optimize         -> defect  #14
  HEAD        clean

Three of the four known index defects, statically, with no false positives on
current code. The miss is #4 (get_TDR tfstart), which is not a base error at all
-- the index is valid, it is applied to the wrong array frame. That class needs
the sentinel-array probe, not this.

Why static and not a runtime type: this runs over com.py, the assembled
artifact, so it covers all inlined copies of a function at once (a fix reaching
only one of several copies is a known hazard here), and it sees code paths no
test executes -- which is exactly where BEST.ctle hid.

Run: python tests/test_index_base.py
"""
import ast
import io
import os
import re
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, xcheck, finish  # noqa: E402

# Optional path argument so the rules can be replayed against a historical
# com.py -- that is how the recall figures above were measured, and how a future
# change to the rules can be re-validated against known defects:
#     git show <commit>:com.py > old.py && python tests/test_index_base.py old.py
# Find the commit with e.g. git log --grep="ADC-clip sampling phase".
COM_PY = (sys.argv[1] if len(sys.argv) > 1
          else os.path.join(os.path.dirname(_here), 'com.py'))

# Structs whose attributes carry values between functions. These are the seams
# where a base convention gets misread; inside one function the author knows.
STRUCTS = {'BEST', 'THIS', 'result', 'fom_result', 'SETTINGS'}

# ---------------------------------------------------------------------------
# THE REGISTRY.  base 1 or 0 for real indices; NOT_INDEX for values that merely
# look like indices by name. Every entry cites the evidence for its base, so a
# reviewer can check the declaration itself rather than trusting it.
# ---------------------------------------------------------------------------
ONE, ZERO, NOT_INDEX, UNRESOLVED = 1, 0, 'value', 'unresolved'

REGISTRY = {
    # --- 1-based, mirroring MATLAB loop counters ---------------------------
    'ctle_index':   (ONE,  "optimize_fom: THIS.ctle_index = ctle_index + 1  '1-based to match MATLAB'"),
    'ctle':         (ONE,  "OptFom_Update_Best_Setttings: BEST.ctle = THIS.ctle_index"),
    'g_LP_index':   (ONE,  "optimize_fom: THIS.g_LP_index = g_LP_index + 1  '1-based'"),
    'G_high_pass':  (ONE,  "OptFom_Update_Best_Setttings: BEST.G_high_pass = THIS.g_LP_index"),
    'best_G_high_pass': (ONE, "result.best_G_high_pass = BEST.G_high_pass; consumers convert (com.py 507, 754, 2243)"),

    # --- 0-based, Python-native -------------------------------------------
    'cursor_i':     (ZERO, "OptFom_Update_BEST_Post_Optimize: cursor_i = int(BEST.cursor_i)  # 0-based"),
    't_s':          (ZERO, "fom_result.t_s is BEST.cursor_i; see docs/COM_PDF_RESIDUAL.md"),
    'start_max_idx': (ZERO, "com.py 4754: max(0, ...) -- a Python array position"),
    'end_max_idx':   (ZERO, "com.py 4755: min(len(uneq_data) - 1, ...) -- a Python array position"),
    'DFE_taps_i':    (ZERO, "com.py 5268: MATLAB cursor_i+(1:ndfe)*M is 1-based; "
                            "Python builds cursor_i+arange(1,ndfe+1)*M, 0-based"),

    # --- not indices: named like one, but never subscript anything ---------
    # Sampling-phase offsets. Signed (roughly -24..+24) and added to a cursor,
    # never used to subscript, so no base applies.
    'itick':                  (NOT_INDEX, "sampling-phase offset, signed"),
    'itick_in_cluster':       (NOT_INDEX, "sampling-phase offset, signed"),
    'negative_itick_in_loop': (NOT_INDEX, "sampling-phase offset, signed"),
    'positive_itick_in_loop': (NOT_INDEX, "sampling-phase offset, signed"),
    'negative_itick_FOM':     (NOT_INDEX, "FOM value, not an index"),
    'positive_itick_FOM':     (NOT_INDEX, "FOM value, not an index"),
    'SNR_ISI':                (NOT_INDEX, "SNR value, not an index"),
    # Tap-value vectors, compared elementwise by the local search.
    'txffe_index':      (NOT_INDEX, "Tx FFE tap VALUE vector, compared not subscripted"),
    'tx_index_vector':  (NOT_INDEX, "Tx FFE tap VALUE vector, compared not subscripted"),
    # Amplitudes and time vectors that merely match the name pattern.
    'cursor':                    (NOT_INDEX, "pulse amplitude at the cursor"),
    'precursors':                (NOT_INDEX, "amplitudes"),
    'far_cursors':               (NOT_INDEX, "amplitudes"),
    'excess_dfe_cursors':        (NOT_INDEX, "amplitudes"),
    'sampled_sbr_precursors_t':  (NOT_INDEX, "time vector, seconds"),
    'sampled_sbr_postcursors_t': (NOT_INDEX, "time vector, seconds"),

    'floating_tap_locations': (ONE,
                               "ML 3559/2576: MATLAB stores this 1-based, and both "
                               "consumers (ML 4133 time vector, ML 4143 DFE_taps_mV "
                               "lookup) read it that way. MMSE and force already "
                               "returned 1-based; OptFom_Compute_DFE now normalises "
                               "floatingDFE's 0-based output to match."),
}

# Attribute names that look like an index and therefore must be declared.
IDXISH = re.compile(r'(?:^|_)(idx|index|indices|loc|locations|pos|cursor_i|tick)(?:$|_)|_i$')


# Wrappers that pass an index through unchanged. Without these the check misses
# the common `np.asarray(BEST.field, dtype=int)` form -- which is exactly how the
# floating_tap_locations subscript is written, so the first version of this file
# reported clean on a use it should have flagged.
_PASSTHRU_FN = {'int', 'float'}
_PASSTHRU_METH = {'asarray', 'array', 'atleast_1d', 'atleast_2d', 'ravel',
                  'flatten', 'astype', 'squeeze', 'copy'}


def _strip_int(n):
    """Peel value-preserving wrappers until a bare expression remains."""
    for _ in range(8):                      # bounded; these never nest deeply
        if isinstance(n, ast.Call):
            fname = getattr(n.func, 'id', None)
            if fname in _PASSTHRU_FN and n.args:
                n = n.args[0]
                continue
            mname = getattr(n.func, 'attr', None)
            if mname in _PASSTHRU_METH:
                if n.args:                  # np.asarray(x) / np.array(x)
                    n = n.args[0]
                    continue
                if isinstance(n.func, ast.Attribute):   # x.ravel() / x.astype()
                    n = n.func.value
                    continue
        break
    return n


def _field(node):
    """The struct field this expression reads, if any."""
    n = _strip_int(node)
    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
       and n.value.id in STRUCTS:
        return n.attr
    return None


def _minus_one(node):
    """(inner, True) if node is `<expr> - 1`, else (node, False)."""
    n = _strip_int(node)
    if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub) \
       and isinstance(n.right, ast.Constant) and n.right.value == 1:
        return _strip_int(n.left), True
    return n, False


def scan(path):
    """Return (violations_A, violations_B, undeclared)."""
    tree = ast.parse(io.open(path, encoding='utf-8').read())
    vio_a, vio_b, undeclared = [], [], {}

    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        # local name -> (field, already_converted)
        alias = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
               and isinstance(node.targets[0], ast.Name):
                inner, conv = _minus_one(node.value)
                f = _field(inner)
                if f:
                    alias[node.targets[0].id] = (f, conv)

        # Rule B: an explicit -1 on a field declared 0-based
        for node in ast.walk(fn):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub) \
               and isinstance(node.right, ast.Constant) and node.right.value == 1:
                f = _field(node.left)
                if f and REGISTRY.get(f, (None,))[0] == ZERO:
                    vio_b.append((f, fn.name, node.lineno))

        # Rule A: a 1-based value reaching a subscript unconverted
        for node in ast.walk(fn):
            if not isinstance(node, ast.Subscript):
                continue
            s = node.slice
            cands = []
            if isinstance(s, ast.Name) and s.id in alias:
                cands.append(alias[s.id])
            f2 = _field(s)
            if f2:
                cands.append((f2, False))
            for f, conv in cands:
                base = REGISTRY.get(f, (None,))[0]
                if conv:
                    continue
                if base in (ONE, UNRESOLVED):
                    vio_a.append((f, fn.name, node.lineno))

        # Rule C: index-shaped struct attributes that are not declared
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
               and node.value.id in STRUCTS and IDXISH.search(node.attr) \
               and node.attr not in REGISTRY:
                undeclared.setdefault(node.attr, fn.name)

    dedup = lambda xs: sorted({x for x in xs})            # noqa: E731
    return dedup(vio_a), dedup(vio_b), undeclared


def _fmt(vs):
    return '; '.join('%s in %s (com.py:%d)' % v for v in vs[:6]) or 'none'


def main():
    if not os.path.isfile(COM_PY):
        check('index_base_com_py_present', False,
              'com.py not found -- run python assemble_com.py first')
        return finish()

    a, b, undeclared = scan(COM_PY)

    a_real = a

    check('rule_A_no_1based_index_used_raw', not a_real,
          'a field declared 1-based reaches a subscript without conversion: %s. '
          'This reads the WRONG element while the array is long enough and only '
          'raises IndexError when the index lands on the last entry -- convert '
          'with int(x) - 1, or correct the declaration in REGISTRY.' % _fmt(a_real))

    check('rule_B_no_minus_one_on_0based', not b,
          'a field declared 0-based has an explicit -1 applied: %s. Either the '
          'value is really 1-based (fix REGISTRY) or the -1 is the defect -- '
          'this is the shape of the ADC-clip off-by-one, docs/COM_PDF_RESIDUAL.md'
          % _fmt(b))

    check('rule_C_every_index_field_declares_its_base', not undeclared,
          'index-shaped struct attributes with no declared base: %s. Add each to '
          'REGISTRY with the base and the evidence for it, or mark it NOT_INDEX '
          'if the name only looks like an index. This is the rule that keeps the '
          'check honest as new features land.'
          % ', '.join('%s (%s)' % (k, v) for k, v in sorted(undeclared.items())[:8]))

    return finish()


if __name__ == '__main__':
    sys.exit(main())
