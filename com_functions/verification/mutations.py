"""The defect-class mutation catalogue, and the engine that runs it.

A test that passes on broken code is not evidence. This repository has shipped
two such checks (2026-09-04) that looked like proof. The remedy is NOT to record
that someone once reintroduced a bug and watched a test fail: that is a judgment
written once and never re-evaluated, which is the same defect as the 2026-07
ledger's stored `verdict` field. Discrimination belongs in derived state, so it
is recomputed here on every run.

For each operator this applies the mutation to a `py_impl.py`, runs THAT
function's own test, and records whether the test failed. A test that still
passes did not discriminate.

Generic mutation operators (flip `+` to `-`) are mostly noise. **Every operator
here is a defect that actually happened in this port.** See CATALOGUE below;
each entry names the real defect. A fix that produces no new operator is a
process failure, so the catalogue only grows, and it grows toward the defects
this port actually produces rather than toward a textbook list.

## Three things that would make this lie, and what is done about each

1. **Mutating a comment or a docstring.** All five `'extrap'` occurrences in the
   tree are in comments. Substituting there changes no behaviour, the test
   passes, and the operator reports a sleeping test that is wide awake. So sites
   are found only in CODE tokens: `tokenize` builds a mask of every COMMENT and
   STRING span and matches overlapping it are discarded.

2. **A mutation that does not parse.** A `SyntaxError` fails the test, which
   scores as a catch and certifies the test on evidence of nothing. So every
   mutant is `ast.parse`d; one that does not parse is a CATALOGUE BUG and is
   reported as such, never as a catch.

3. **A test that was already failing.** Then the mutant "fails" too and scores
   as caught. So the clean test must pass before the mutant is even built.

## Granularity

One site at a time, not one file at a time. A file with five `.copy()` calls
whose test guards only one would, mutated wholesale, score as discriminating
while four sites stay unguarded.

## Caching

Keyed by the content hash of the implementation, of the bound test, AND of the
operator's own definition. Change any of the three and the pair re-runs. This is
a memo over derived state, not stored judgment: nothing is believed because
someone wrote it down, only because the inputs that produced it are unchanged.

The operator fingerprint is not optional. Without it the key is id + function +
site INDEX, so narrowing a pattern re-serves the old rows under the same key
while index 0 now means a different site. That happened, and it reported a
scalar config flag the new pattern does not match, with its old line number.

    python com_functions/verification/mutations.py --list
    python com_functions/verification/mutations.py --run [--only ID] [--no-cache]
    python com_functions/verification/mutations.py --run --checkpoint [--ckpt-cases ...]

## The checkpoint binding

`--checkpoint` takes every mutant its own function test MISSED and puts it in
front of tests/test_octave_checkpoints.py, which compares the whole pipeline
against COM Octave on real cases. Running the engine per mutant per case is
hours, so each mutant first earns its run:

  * the mutant is assembled and diffed against the clean engine, which names
    the engine lines it changes (none: `not_in_engine`);
  * each checkpoint case was run once under reach.py, which records the lines
    it executes; a mutant whose lines no case executes is `unreached`, a gap in
    the CASE SET, and needs no run;
  * a reached mutant runs the harness on the cases that execute it, from its
    own copy of sicopr.py, several at once. A FAIL row the clean engine does
    not have is `caught`. Otherwise its saved checkpoints are compared with the
    clean engine's: identical is `survived_no_change` (the state it corrupts is
    not saved, or it is equivalent on these cases), different is
    `survived_within_tolerance` (the harness saw it and let it through).

The three survivor classes are the harness's own gap list.

`tests/test_mutation_score.py` is the gate that gets run by `tests\run_all.ps1`.
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import ast
import concurrent.futures as _cf
import difflib
import importlib.util
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tokenize

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
FN = os.path.join(_ROOT, 'com_functions', 'fn')
CACHE = os.path.join(_HERE, '.mutation_cache.json')
EQUIVALENT_MD = os.path.join(_HERE, 'equivalent_mutants.md')


class Op(object):
    """One mutation operator: a real defect from this port's history.

    `pattern` is searched in code tokens only. `replace` is a re.sub template.
    `defect` is what actually went wrong, and `evidence` points at where that is
    recorded, so an operator can never become a rule nobody can trace.
    """

    def __init__(self, id, pattern, replace, defect, evidence, skip=None,
                 guard=None, require=None):
        self.id = id
        self.pattern = re.compile(pattern)
        self.replace = replace
        self.defect = defect
        self.evidence = evidence
        # a regex that, if it matches the whole LINE, disqualifies the site
        self.skip = re.compile(skip) if skip else None
        # a regex the LINE must contain for the site to count. This is how an
        # operator is kept honest about the defect it represents: see
        # ne_zero_to_gt_zero, where the real defect was an ARRAY comparison and
        # a bare textual match dragged in 55 scalar config flags that cannot
        # express it.
        self.require = re.compile(require) if require else None
        # Identity of the operator's DEFINITION, for the result cache.
        #
        # Without this the cache key is op id + function + site INDEX, and
        # narrowing a pattern silently re-serves the old rows: index 0 comes to
        # mean a different site while the key stays the same. Narrowing
        # ne_zero_to_gt_zero from 63 sites to 4 duly reported a scalar config
        # flag that the new pattern does not even match, complete with its old
        # line number. Same family as the stale-bytecode bug, and the same
        # lesson: cached derived state must be keyed by EVERYTHING that
        # produced it.
        self.fingerprint = hashlib.sha256(
            repr((id, pattern, replace, skip, require)).encode('utf-8')
        ).hexdigest()[:12]
        # (audit script, check name): a REPO-WIDE lint that forbids the
        # mutated source form outright, so the defect cannot land even where
        # the function's own test is blind to it.
        #
        # This distinction matters a great deal for where effort goes. 40 of
        # the mmax/mmin mutants survive their bound test, which reads as 40
        # urgent gaps; but `maxmin_every_site_uses_matlab_nan_semantics` scans
        # the assembled engine and rejects a bare np.max anywhere, so none of
        # them could ever reach a release. Without this field the gate would
        # aim work at the best-defended code in the repository.
        #
        # The claim is VERIFIED, never trusted: verify_guards() mutates a real
        # site, re-assembles, runs the named check and requires it to fail. A
        # guard that stopped working would otherwise be a stored judgment, the
        # exact defect this framework exists to prevent.
        self.guard = guard


# Every entry is a defect that happened here. Do not add a textbook operator.
CATALOGUE = [
    Op('std_ddof',
       r'np\.std\(([^()]*), ddof=1\)', r'np.std(\1)',
       'MATLAB/Octave std normalises by N-1, numpy defaults to N. Sat inside a '
       'membership threshold in interp_Sparam on the branch every shipped '
       'workbook selects, so no unit test and no corpus channel reached it.',
       'commit d5bff6c; com-208-corpus-dc-blind-spot',
       guard=('tests/test_matlab_semantics.py', 'std_every_site_passes_ddof')),

    Op('mround_to_np_round',
       r'\b_mround\(', 'np.round(',
       'MATLAB round() is half AWAY from zero; np.round is half to EVEN. '
       '41 sites.',
       'docs/AUDIT_FINDINGS.md',
       skip=r'\s*def\s+_mround',
       guard=('tests/test_matlab_semantics.py', 'round_every_site_uses_matlab_semantics')),

    Op('mmax_to_np_max',
       r'\b_mmax\(', 'np.max(',
       'MATLAB max() skips NaN; np.max propagates it. 48 sites across the '
       'engine.',
       'docs/AUDIT_FINDINGS.md',
       skip=r'\s*def\s+_mmax',
       guard=('tests/test_matlab_semantics.py', 'maxmin_every_site_uses_matlab_nan_semantics')),

    Op('mmin_to_np_min',
       r'\b_mmin\(', 'np.min(',
       'MATLAB min() skips NaN; np.min propagates it.',
       'docs/AUDIT_FINDINGS.md',
       skip=r'\s*def\s+_mmin',
       guard=('tests/test_matlab_semantics.py', 'maxmin_every_site_uses_matlab_nan_semantics')),

    Op('mlength_to_len',
       r'\b_length\(', 'len(',
       "MATLAB length() is the LONGEST dimension, not the first. Broke "
       'Butterworth_Filter and Bessel_Thomson_Filter on a scalar f.',
       'docs/AUDIT_FINDINGS.md',
       skip=r'\s*def\s+_length'),

    Op('drop_copy_guard',
       r'\b(\w+) = list\(\1\)', r'\1 = \1',
       'MATLAB passes structs BY VALUE. Dropping the guard lets a callee mutate '
       "the caller's struct: 5 of the 8 defects found in 2026-08 were this "
       'class, including the process_sxp OP leak and the OptFom_Calc_FOM leak '
       'of a LOSING EQ candidate.',
       'com-matlab-correlation-complete'),

    Op('drop_dot_copy',
       r'\.copy\(\)', '',
       'Same by-value class as drop_copy_guard, via numpy. stot/ttos mutated '
       "the caller's array.",
       'com-matlab-correlation-complete'),

    # NARROWED 2026-09-22, and the narrowing is the point.
    #
    # The real defect is `support = np.where(pdf_y != 0)` in d_cpdf, where
    # pdf_y is an ARRAY of probabilities that can hold NaN. A bare textual
    # `!= 0` matched 63 sites, 58 of which survived and read as gaps. They were
    # not: almost all are `if param.N_qb != 0:` and its kin, scalar config
    # enable-flags where a count cannot go negative, so `> 0` and `!= 0` cannot
    # differ and no test could ever tell them apart.
    #
    # That is the "generic operators are mostly noise" failure, committed by
    # over-generalising ONE real array defect into every occurrence of a common
    # token. An operator that reports 58 unreachable gaps does not measure the
    # suite, it buries the 8 findings that matter. So the site must be an array
    # context: the comparison feeds a numpy reduction or a mask index.
    Op('ne_zero_to_gt_zero',
       r'!= 0\b', '> 0',
       'd_cpdf counted NaN as support: np.where(pdf_y != 0) on a probability '
       'array. The two differ exactly on negative and NaN entries, which a '
       'scalar config flag cannot be, so only array comparisons carry the '
       'defect.',
       'docs/AUDIT_FINDINGS.md',
       require=r'np\.(?:where|any|all|sum|nonzero|flatnonzero|count_nonzero)\('
               r'|\[[^\]]*!= 0'),

    # A mutation must be a BEHAVIOURAL difference, not a runtime error. An
    # earlier draft substituted a nonexistent numpy attribute: that raises
    # AttributeError, the test fails, and the operator scores a catch having
    # proved only that the code runs. lstsq agrees with solve on every
    # non-singular system and differs exactly on the singular one, which is the
    # defect.
    Op('solve_to_lstsq',
       r'np\.linalg\.solve\(([^,()]+), ([^,()]+)\)',
       r'np.linalg.lstsq(\1, \2, rcond=None)[0]',
       'force() on a singular VV. solve raises where lstsq silently returns a '
       'minimum-norm answer, which is also the Octave-vs-MATLAB backslash '
       'divergence repaired in commit 4c73cba.',
       'commit 4c73cba; octave_deviations.md'),

    Op('interp_drop_nan_bounds',
       r', left=np\.nan, right=np\.nan', '',
       'MATLAB interp1 returns NaN OUTSIDE the data range; np.interp clamps to '
       'the end values instead. Fixed in scalePDF and MLSE_U1_c_178A.',
       'commit 04c6371'),

    # Dropping the transpose is the defect whatever the shift expression is, so
    # match only the part that carries it. Requiring `axis=0` and a paren-free
    # shift found zero sites while three were sitting in the tree.
    Op('roll_drop_transpose',
       r'np\.roll\((\w+)\.T,', r'np.roll(\1,',
       'FFE on a 2-D input rolled the wrong axis.',
       'docs/AUDIT_FINDINGS.md'),

    # The one exception to the rule above, added 2026-09-23 because the
    # verification prompt asked for it: the documented MATLAB-vs-numpy
    # difference the catalogue had no operator for. No occurrence is recorded
    # in this port's history, and the Phase 0 census of complex comparisons
    # found none, so the evidence is the builtins row and the two sites.
    Op('ctranspose_to_transpose',
       r'\.conj\(\)\.T\b|\.T\.conj\(\)', '.T',
       "MATLAB ' is the CONJUGATE transpose and .' the plain one; numpy .T is "
       "the plain one. Interchangeable only on real data. The ILN fits apply "
       "it to fmbg, which is built from the complex sdd21, in the normal "
       "equations alpha = inv(fmbg'*fmbg)*fmbg'*LGw (ML 6740, 6755).",
       "com_functions/verification/builtins.md, row `transpose`; "
       "get_ILN_cmp_td and get_RILN_cmp_td (commit 5276a8e); Phase 0 complex "
       "census 2026-09-23: no hit"),
]


def code_mask(text):
    """Character offsets that sit inside a COMMENT or STRING token.

    This is the guard that keeps the catalogue honest. Without it the
    `'extrap'` operator reports five sleeping tests, because all five of its
    occurrences in this tree are in comments and mutating one changes nothing.
    """
    masked = set()
    lines = text.splitlines(True)
    starts = []
    off = 0
    for ln in lines:
        starts.append(off)
        off += len(ln)

    def pos(row, col):
        return starts[row - 1] + col if 0 < row <= len(starts) else None

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Cannot tokenise: mask everything so no site is offered. Silence is
        # correct here; inventing sites in a file we cannot read is not.
        return set(range(len(text)))
    for tok in toks:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        a, b = pos(*tok.start), pos(*tok.end)
        if a is not None and b is not None:
            masked.update(range(a, b))
    return masked


def sites(op, text):
    """(start, end, mutated_text) for each applicable CODE site, in order."""
    masked = code_mask(text)
    out = []
    for m in op.pattern.finditer(text):
        if any(i in masked for i in range(m.start(), m.end())):
            continue
        line_start = text.rfind('\n', 0, m.start()) + 1
        line_end = text.find('\n', m.end())
        line = text[line_start:line_end if line_end >= 0 else len(text)]
        if op.skip and op.skip.match(line):
            continue
        if op.require and not op.require.search(line):
            continue
        mutated = text[:m.start()] + m.expand(op.replace) + text[m.end():]
        out.append((m.start(), m.end(), mutated))
    return out


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def read_bytes(path):
    """Exact bytes, plus the decoded text.

    Byte-exact on purpose. An earlier version read in TEXT mode, which collapses
    this tree's CRLF endings to LF in memory, and restored with newline='',
    which writes them back as LF. The content was right and `git diff` was
    empty, but 24 files had their line endings rewritten, and the restore check
    could not see it because it compared the LF string against the LF file it
    had just written. Round-tripping the original bytes makes the restore
    verifiable against something the tool did not itself produce.
    """
    with open(path, 'rb') as f:
        raw = f.read()
    return raw, raw.decode('utf-8')


def write_bytes(path, raw):
    with open(path, 'wb') as f:
        f.write(raw)


def run_test(fn_dir):
    """Run one function's bound test. True if it PASSED.

    Bytecode caching has to be defeated here, and it is not optional.

    Python validates a .pyc by (mtime, size). EVERY mutant in this catalogue
    removes or replaces a few bytes, so consecutive mutants of the same file
    can have the SAME size, and consecutive runs fall in the same mtime second.
    The result is that pytest silently imports the PREVIOUS mutant's bytecode.

    That is not theoretical: it scored OptFom_Compute_DFE's three .copy() sites
    as survived/survived/caught when the truth is survived/caught/caught. Every
    site was being judged against the one before it, which makes the whole run
    worthless while looking perfectly healthy.

    So the cache is removed before each run and writing is disabled.
    """
    for base in (os.path.join(FN, fn_dir), FN, _ROOT):
        pyc = os.path.join(base, '__pycache__')
        if os.path.isdir(pyc):
            shutil.rmtree(pyc, ignore_errors=True)
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    r = subprocess.run(
        [sys.executable, '-B', '-m', 'pytest', os.path.join(FN, fn_dir), '-q',
         '-x', '--no-header', '-p', 'no:cacheprovider'],
        cwd=_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    return r.returncode == 0


def line_of(text, off):
    return text.count('\n', 0, off) + 1


def _assemble():
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return subprocess.run(
        [sys.executable, '-B', os.path.join(_ROOT, 'assemble_sicopr.py')],
        cwd=_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env).returncode == 0


def _named_check_fails(script, check_name):
    """Run an audit script and report whether that one check FAILED."""
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    r = subprocess.run([sys.executable, '-B', os.path.join(_ROOT, script)],
                       cwd=_ROOT, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, env=env)
    out = r.stdout.decode('utf-8', 'replace')
    return ('FAIL %s' % check_name) in out


def verify_guards():
    """Prove each operator's `guard` really does reject the mutated form.

    A guard is the reason a whole class of survivors is NOT urgent, so an
    unverified guard would quietly redirect every future effort away from real
    gaps. It is therefore demonstrated, not declared: mutate one real site,
    re-assemble the engine, run the named check, and require it to fail.

    Returns {operator id: True if its guard was demonstrated}.
    """
    out = {}
    for op in CATALOGUE:
        if not op.guard:
            continue
        script, check_name = op.guard
        work = [w for w in collect(op.id)]
        if not work:
            out[op.id] = False
            continue
        _o, fn_dir, _i, _off, mutated, orig_raw = work[0]
        impl = os.path.join(FN, fn_dir, 'py_impl.py')
        try:
            write_bytes(impl, mutated.encode('utf-8'))
            out[op.id] = _assemble() and _named_check_fails(script, check_name)
        finally:
            write_bytes(impl, orig_raw)
            with open(impl, 'rb') as f:
                if f.read() != orig_raw:
                    raise SystemExit(
                        'mutations.py: FAILED TO RESTORE %s during guard '
                        'verification. Restore it from git.' % impl)
            _assemble()
    return out


def collect(only=None):
    """Every (op, fn_dir, site) in the tree. Pure inspection, no mutation."""
    work = []
    for fn_dir in sorted(os.listdir(FN)):
        impl = os.path.join(FN, fn_dir, 'py_impl.py')
        test = os.path.join(FN, fn_dir, 'test_verify.py')
        if not (os.path.isfile(impl) and os.path.isfile(test)):
            continue
        raw, text = read_bytes(impl)
        for op in CATALOGUE:
            if only and op.id != only:
                continue
            for i, (a, b, mutated) in enumerate(sites(op, text)):
                work.append((op, fn_dir, i, a, mutated, raw))
    return work


def evaluate(only=None, use_cache=True, progress=True):
    """Every applicable mutant, evaluated. Returns the result rows.

    This is the importable entry point; `tests/test_mutation_score.py` is the
    gate that turns these rows into pass or fail. Nothing here decides anything,
    it only computes.
    """
    work = collect(only)
    cache = {}
    if os.path.isfile(CACHE) and use_cache:
        try:
            cache = json.load(open(CACHE, encoding='utf-8'))
        except ValueError:
            cache = {}

    clean_pass = {}
    rows = []
    if progress:
        print('%d mutants to evaluate\n' % len(work))
    for n, (op, fn_dir, idx, off, mutated, orig_raw) in enumerate(work, 1):
        orig = orig_raw.decode('utf-8')
        impl = os.path.join(FN, fn_dir, 'py_impl.py')
        test = os.path.join(FN, fn_dir, 'test_verify.py')
        key = '%s|%s|%s|%d|%s|%s' % (op.id, op.fingerprint, fn_dir, idx,
                                     _sha(impl), _sha(test))

        if key in cache and use_cache:
            rows.append(dict(cache[key], cached=True))
            continue

        # 3. a test that was already failing would score every mutant as caught
        if fn_dir not in clean_pass:
            clean_pass[fn_dir] = run_test(fn_dir)
        if not clean_pass[fn_dir]:
            row = {'op': op.id, 'fn': fn_dir, 'line': line_of(orig, off),
                   'outcome': 'test_already_failing'}
            rows.append(row)
            cache[key] = row
            continue

        # 2. a mutant that does not parse is a CATALOGUE BUG, never a catch
        try:
            ast.parse(mutated)
        except SyntaxError:
            row = {'op': op.id, 'fn': fn_dir, 'line': line_of(orig, off),
                   'outcome': 'catalogue_bug_mutant_does_not_parse'}
            rows.append(row)
            cache[key] = row
            continue

        try:
            write_bytes(impl, mutated.encode('utf-8'))
            caught = not run_test(fn_dir)
        finally:
            write_bytes(impl, orig_raw)
            # Compare the bytes on disk against the bytes read BEFORE any
            # mutation. Hashing the string we just wrote proves only that the
            # write succeeded, which is how the CRLF rewrite went unnoticed.
            with open(impl, 'rb') as _f:
                if _f.read() != orig_raw:
                    raise SystemExit(
                        'mutations.py: FAILED TO RESTORE %s byte-for-byte. The '
                        'working tree is now mutated. Restore it from git '
                        'before doing anything else.' % impl)

        row = {'op': op.id, 'fn': fn_dir, 'line': line_of(orig, off),
               'outcome': 'caught' if caught else 'survived'}
        rows.append(row)
        cache[key] = row
        if progress:
            sys.stdout.write('\r  %d/%d  %-30s %s        '
                             % (n, len(work), fn_dir, row['outcome']))
            sys.stdout.flush()
    if progress:
        print()

    if use_cache:
        with io.open(CACHE, 'w', encoding='utf-8') as f:
            f.write(unicode_json(cache))
    return rows


# ------------------------------------------------ the checkpoint binding
CKPT_TEST = os.path.join(_ROOT, 'tests', 'test_octave_checkpoints.py')
REACH = os.path.join(_HERE, 'reach.py')


def _ckpt_module():
    spec = importlib.util.spec_from_file_location('_ckpt', CKPT_TEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _engine_lines_changed(clean, mutant):
    """1-based lines of the CLEAN engine that the mutant alters."""
    a, b = clean.splitlines(), mutant.splitlines()
    if len(a) == len(b):
        return {i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y}
    out = set()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag != 'equal':
            out.update(range(i1 + 1, max(i2, i1 + 1) + 1))
    return out


def _sicopr_args(meta):
    cmd = [meta['config'], meta['thru']]
    if meta.get('fext'):
        cmd += ['--fext'] + list(meta['fext'])
    if meta.get('next'):
        cmd += ['--next'] + list(meta['next'])
    return cmd + ['--matlab-version', os.environ.get('COM_CHECKPOINT_VERSION',
                                                     '4p15p0')]


def _harness(case, report, pythonpath, reuse=False):
    env = dict(os.environ, COM_CHECKPOINT_REPORT=report,
               COM_CHECKPOINT_CASES=case, COM_CHECKPOINT_JOBS='1',
               PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join([pythonpath, _ROOT]))
    env.pop('COM_CHECKPOINT_REUSE', None)
    if reuse:
        env['COM_CHECKPOINT_REUSE'] = '1'
    r = subprocess.run([sys.executable, '-B', CKPT_TEST], cwd=_ROOT, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    rows = []
    fp = os.path.join(report, case, 'fields.csv')
    if os.path.isfile(fp):
        import csv
        with open(fp, encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
    return rows, r.stdout.decode('utf-8', 'replace')


def _fails(rows):
    return {(r['checkpoint'], r['field']) for r in rows if r['status'] == 'FAIL'}


def _tree_diff(ck, a, b, path=''):
    """-> list of (path, max_rel) where two from_python trees differ."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append((path + '.' + k, float('inf')))
            else:
                out += _tree_diff(ck, a[k], b[k], path + '.' + k)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [(path, float('inf'))]
        for i, (x, y) in enumerate(zip(a, b)):
            out += _tree_diff(ck, x, y, '%s[%d]' % (path, i))
        return out
    import numpy as np
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            x, y = np.asarray(a), np.asarray(b)
            if x.shape != y.shape:
                return [(path, float('inf'))]
            if x.dtype.kind in 'biufc' and y.dtype.kind in 'biufc':
                if np.array_equal(x, y, equal_nan=True):
                    return []
                d = np.abs(x.astype(complex) - y.astype(complex))
                d = d[np.isfinite(d)]
                pk = float(np.max(np.abs(x[np.isfinite(x)]))) if np.isfinite(x).any() else 1.0
                return [(path, float(np.max(d)) / (pk or 1.0) if d.size else float('inf'))]
            return [] if np.array_equal(x, y) else [(path, float('inf'))]
        except (TypeError, ValueError):
            return [] if repr(a) == repr(b) else [(path, float('inf'))]
    return [] if (a == b or (a != a and b != b)) else [(path, float('inf'))]


def _saved_state_diff(ck_mod, clean_ck, mut_ck):
    out = []
    for f in sorted(os.listdir(clean_ck)):
        if not f.endswith('.pkl'):
            continue
        m = os.path.join(mut_ck, f)
        if not os.path.isfile(m):
            out.append((f[:-4], float('inf')))
            continue
        a, _ = ck_mod._load_python(os.path.join(clean_ck, f))
        b, _ = ck_mod._load_python(m)
        for p, r in _tree_diff(ck_mod, a, b):
            p = p.lstrip('.')
            # where and when the run happened, not what it computed: the
            # harness's own ENVIRONMENT set, and the export timestamps
            if ck_mod._canon(p) in ck_mod.ENVIRONMENT or p.startswith('OP.export_'):
                continue
            out.append((f[:-4] + ':' + p, r))
    return out


def default_ckpt_cases(manifest):
    """The harness's own quick subset -- the first woXtalk case per package
    configuration -- plus the first wXtalk case, the only way into the
    crosstalk branches."""
    ids = sorted(manifest['cases'])
    seen, out = set(), []
    for cid in ids:
        m = re.match(r'woXtalk_T(\d)_', cid)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(cid)
    out += [c for c in ids if c.startswith('wXtalk_')][:1]
    return out


def evaluate_checkpoint(work, rows, cases, workdir, jobs=4, progress=True):
    """Put every function-test survivor in front of the checkpoint harness."""
    ck = _ckpt_module()
    if not ck.GOLD or not os.path.isfile(os.path.join(ck.GOLD, 'manifest.json')):
        raise SystemExit('--checkpoint needs COM_OCTAVE_CHECKPOINTS (the goldens)')
    manifest = json.load(open(os.path.join(ck.GOLD, 'manifest.json'),
                              encoding='utf-8'))
    cases = cases or default_ckpt_cases(manifest)
    engine = os.path.join(_ROOT, 'sicopr.py')
    with open(engine, 'rb') as fh:
        clean_raw = fh.read()
    clean = clean_raw.decode('utf-8')
    base = os.path.join(workdir, 'clean')

    # 1. the clean engine, once per case: executed lines, checkpoints, fields
    reach, clean_rows = {}, {}
    for cid in cases:
        out = os.path.join(base, cid)
        ckd = os.path.join(out, 'checkpoints')
        rj = os.path.join(out, 'reach.json')
        if not os.path.isfile(rj):
            os.makedirs(ckd, exist_ok=True)
            if progress:
                print('clean engine, %s: tracing executed lines' % cid)
            env = dict(os.environ, COM_CHECKPOINT_DIR=os.path.abspath(ckd),
                       PYTHONPATH=_ROOT, PYTHONDONTWRITEBYTECODE='1')
            subprocess.run([sys.executable, '-B', REACH, rj, '--'] +
                           _sicopr_args(manifest['cases'][cid]), cwd=out,
                           env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        reach[cid] = set(json.load(open(rj, encoding='utf-8'))['lines'])
        clean_rows[cid] = _harness(cid, base, _ROOT, reuse=True)[0]

    # 2. every survivor, assembled into its own engine; the tree restored
    todo, results = [], []
    try:
        for (op, fn_dir, idx, off, mutated, orig_raw), row in zip(work, rows):
            if row['outcome'] != 'survived':
                continue
            impl = os.path.join(FN, fn_dir, 'py_impl.py')
            try:
                write_bytes(impl, mutated.encode('utf-8'))
                ok = _assemble()
                with open(engine, 'rb') as fh:
                    mut_engine = fh.read().decode('utf-8')
            finally:
                write_bytes(impl, orig_raw)
                with open(impl, 'rb') as fh:
                    if fh.read() != orig_raw:
                        raise SystemExit('mutations.py: FAILED TO RESTORE %s'
                                         % impl)
            res = dict(row, checkpoint=None, cases=[], detail='')
            results.append(res)
            if not ok:
                res['checkpoint'] = 'catalogue_bug_does_not_assemble'
                continue
            lines = _engine_lines_changed(clean, mut_engine)
            if not lines:
                res['checkpoint'] = 'not_in_engine'
                continue
            hit = [c for c in cases if lines & reach[c]]
            res['engine_lines'] = sorted(lines)[:5]
            if not hit:
                res['checkpoint'] = 'unreached'
                continue
            d = os.path.join(workdir, 'm%03d' % len(results))
            os.makedirs(d, exist_ok=True)
            with io.open(os.path.join(d, 'sicopr.py'), 'w', encoding='utf-8',
                         newline='') as fh:
                fh.write(mut_engine)
            todo.append((res, d, hit))
    finally:
        _assemble()
        with open(engine, 'rb') as fh:
            if fh.read() != clean_raw:
                raise SystemExit('mutations.py: the re-assembled engine is not '
                                 'byte-identical to the clean one. Check git.')

    if progress:
        n = {}
        for r in results:
            n[r['checkpoint']] = n.get(r['checkpoint'], 0) + 1
        print('%d function-test survivors: %d to run against the harness, %s'
              % (len(results), len(todo),
                 ', '.join('%s %d' % kv for kv in sorted(n.items(), key=str)
                           if kv[0])))

    # 3. the reached ones against the harness, several at once
    def one(item):
        res, d, hit = item
        for cid in hit:
            fields, _log = _harness(cid, os.path.join(d, 'report'), d)
            clean_fields = clean_rows[cid]
            new = sorted(_fails(fields) - _fails(clean_fields))
            res['cases'].append(cid)
            if not fields or new:
                res['checkpoint'] = 'caught'
                res['detail'] = ('%s %s' % new[0]) if new else 'no fields.csv'
                return res
        diffs = []
        for cid in hit:
            diffs += _saved_state_diff(
                ck, os.path.join(base, cid, 'checkpoints'),
                os.path.join(d, 'report', cid, 'checkpoints'))
        if not diffs:
            res['checkpoint'] = 'survived_no_change'
        else:
            res['checkpoint'] = 'survived_within_tolerance'
            p, r = max(diffs, key=lambda t: t[1])
            res['detail'] = '%d saved field(s) moved; largest %s rel %.1e' % (
                len(diffs), p, r)
        return res

    with _cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        for k, res in enumerate(ex.map(one, todo), 1):
            if progress:
                print('  %3d/%d  %-22s %-30s line %-5d %s'
                      % (k, len(todo), res['op'], res['fn'], res['line'],
                         res['checkpoint']))
    return results


def report_checkpoint(results):
    by = {}
    for r in results:
        by.setdefault(r['checkpoint'], []).append(r)
    print('\nfunction-test survivors put in front of the checkpoint harness: %d'
          % len(results))
    for k in ('caught', 'survived_within_tolerance', 'survived_no_change',
              'unreached', 'not_in_engine', 'catalogue_bug_does_not_assemble'):
        if by.get(k):
            print('  %-34s %d' % (k, len(by[k])))
    for k, head in (('survived_within_tolerance',
                     'the harness saw a change and let it through'),
                    ('survived_no_change',
                     'nothing the harness saves changed on the cases run'),
                    ('unreached', 'no checkpoint case executes the line')):
        if by.get(k):
            print('\n%s -- %s:' % (k.upper(), head))
            for r in sorted(by[k], key=lambda r: (r['op'], r['fn'], r['line'])):
                print('   %-24s %-34s line %-5d %s'
                      % (r['op'], r['fn'], r['line'], r.get('detail', '')))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--list', action='store_true',
                    help='show the catalogue and its site counts, run nothing')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--only', help='a single operator id')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--json', help='write the result rows here')
    ap.add_argument('--checkpoint', action='store_true',
                    help='also run every function-test survivor against the '
                         'Octave checkpoint harness (needs the goldens)')
    ap.add_argument('--ckpt-cases', nargs='*',
                    help='checkpoint case ids (default: the harness quick '
                         'subset plus the first wXtalk case)')
    ap.add_argument('--ckpt-jobs', type=int, default=4)
    ap.add_argument('--ckpt-dir', help='working directory for the mutant '
                    'engines and their reports (outside the repository)')
    args = ap.parse_args()

    if args.list or not args.run:
        work = collect(args.only)
        print('%d operators, %d applicable code sites\n'
              % (len(CATALOGUE), len(work)))
        for op in CATALOGUE:
            n = sum(1 for w in work if w[0].id == op.id)
            files = len({w[1] for w in work if w[0].id == op.id})
            print('  %-24s %4d sites in %3d functions' % (op.id, n, files))
            summary = op.defect.replace('\n', ' ')
            print('      %s' % (summary[:76] + '...' if len(summary) > 79
                                else summary))
        return 0

    rows = evaluate(args.only, use_cache=not args.no_cache)
    report(rows)
    if args.checkpoint:
        if not args.ckpt_dir:
            raise SystemExit('--checkpoint needs --ckpt-dir')
        rows = evaluate_checkpoint(collect(args.only), rows, args.ckpt_cases,
                                   args.ckpt_dir, jobs=args.ckpt_jobs)
        report_checkpoint(rows)
    if args.json:
        with io.open(args.json, 'w', encoding='utf-8') as f:
            f.write(unicode_json(rows))
    return 0


def unicode_json(obj):
    return json.dumps(obj, indent=1, sort_keys=True)


def report(rows):
    by = {}
    for r in rows:
        by.setdefault(r['outcome'], []).append(r)
    print('\n%-42s %d' % ('mutants evaluated', len(rows)))
    for k in ('caught', 'survived', 'test_already_failing',
              'catalogue_bug_mutant_does_not_parse'):
        if by.get(k):
            print('%-42s %d' % ('  ' + k, len(by[k])))
    surv = by.get('survived', [])
    if surv:
        print('\nSURVIVED -- the bound test did not notice this defect class:')
        for r in sorted(surv, key=lambda r: (r['op'], r['fn'])):
            print('   %-24s %-34s line %d' % (r['op'], r['fn'], r['line']))
    bug = by.get('catalogue_bug_mutant_does_not_parse', [])
    if bug:
        print('\nCATALOGUE BUGS -- these mutants do not parse, so they prove '
              'nothing. Fix the operator; do NOT read them as catches:')
        for r in bug:
            print('   %-24s %-34s line %d' % (r['op'], r['fn'], r['line']))


if __name__ == '__main__':
    sys.exit(main())
