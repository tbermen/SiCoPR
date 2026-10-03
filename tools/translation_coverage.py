"""How much of the MATLAB reference is verified against the reference?

Line coverage answers the wrong question for a port. It measures how much of
the Python ran, which moves when Python is refactored and says nothing about
whether the answer matches MATLAB. It also falls when a correct helper is
added, which is how a real improvement can look like a regression.

The question that matters is per FUNCTION: for each function in
matlab/com_ieee8023_<ver>.m, is there a Python translation, does it have a
test, and does that test check its VALUES against the executed reference?

Functions are split by how hard they are to reach:

  leaf       calls no other translated function. These are the syntax
             translations -- the places a library default differs, where
             `std` became `np.std`. They are cheap to drive directly, so
             anything less than a value check here is a gap with no excuse.
  composite  calls other translated functions. Harder to set up, and a
             failure may belong to a callee.

Run:
    python tools/translation_coverage.py             # summary
    python tools/translation_coverage.py --gaps      # the untested ones
    python tools/translation_coverage.py --csv out.csv
    python tools/translation_coverage.py --census DIR  # what the grade cannot see

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import argparse
import csv
import io
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
FN = os.path.join(_ROOT, 'com_functions', 'fn')

def _default_ref():
    """The release the engine emulates by default, from VERSION.json: the coverage
    is measured against what a plain run follows. (Was hardcoded to 4p16p0, which
    would have left 4p17p0's four new functions out of the count.)"""
    import json
    with open(os.path.join(_ROOT, 'VERSION.json'), encoding='utf-8') as fh:
        v = json.load(fh)
    return os.path.join(_ROOT, v['references'][v['default_matlab_version']]['file'])


DEFAULT_REF = _default_ref()

# a test that quotes the reference, rather than a reading of it
ORACLE_MARKS = ('COM Octave', 'octave_oracle', 'COM_Octave')
# An assertion that pins a VALUE rather than a shape or a type.
#
# Floats and array comparisons are obvious. Integer and string equality counts
# too: several functions return a decision -- a skip flag, a chosen index, a
# field name -- and `assert skip_it == 1` pins that as surely as a float does.
# What does not count is equality against a length or a shape, which is
# structural however it is written.
VALUE = re.compile(r'\d+\.\d|\de-\d|\de\+\d|approx|allclose|isclose'
                   r'|assert_array|assert_almost|== \[|== _OCT|_OCT\w*\['
                   r"|[=!]= *-?\d|[=!]= *'|[=!]= *\"|is True|is False")
STRUCTURAL = re.compile(r'len\(|\.shape|\.size|\.ndim|isinstance|hasattr'
                        r'|in vars\(|\.dtype')
# things MATLAB provides that are not translated functions
BUILTIN = set('''if else elseif end for while switch case otherwise function return
 zeros ones length size abs real imag sum max min exp sqrt log log10 find isempty
 error warning sprintf fprintf disp numel reshape repmat squeeze permute sort
 round floor ceil fix mod rem linspace logspace cumsum mean median std var
 interp1 fft ifft conv filter toeplitz eye diag inv pinv norm det rank trace
 transpose fliplr flipud circshift cat horzcat vertcat double single logical
 int32 char cell struct fieldnames isfield isnumeric ischar iscell isstruct
 strcmp strcmpi strrep strsplit strtrim upper lower num2str str2num str2double
 nan inf isnan isinf any all xor not and or cellfun arrayfun structfun deal
 nargin nargout varargin varargout figure plot hold grid title xlabel ylabel
 legend axis subplot close drawnow print saveas set get gca gcf text line
 unwrap angle conj polyfit polyval trapz cumtrapz histc hist rand randn
 tic toc clock datestr now pause input keyboard assert isequal unique
 ismember setdiff intersect union fliplr triu tril kron'''.split())


def matlab_functions(path):
    """{name: body} for every function defined in the reference."""
    src = io.open(path, encoding='latin-1').read()
    starts = []
    for m in re.finditer(r'\nfunction\b', src):
        head = src[m.start():src.find('\n', m.start() + 10) + 1]
        cont = head
        j = m.start()
        while cont.rstrip().endswith('...'):
            j = src.find('\n', j + 1)
            nxt = src[j + 1:src.find('\n', j + 1) + 1]
            cont = nxt
            head += nxt
        nm = re.search(r'[\[\s=]([A-Za-z_]\w*)\s*\(', head.split('function', 1)[1])
        if nm:
            starts.append((m.start() + 1, nm.group(1)))
    out = {}
    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] - 1 if i + 1 < len(starts) else len(src)
        out[name] = src[pos:end]
    return out


def callees(body, known):
    """Translated functions this body calls."""
    names = set(re.findall(r'(?<![\w.])([A-Za-z_]\w*)\s*\(', body))
    return {n for n in names if n in known and n not in BUILTIN}


def inlined_definitions():
    """Functions the assembler inlines rather than giving their own directory.

    LFSR has no fn/ directory because PRBS13Q carries it as _lfsr. Counting
    that as untranslated is a false negative, so every py_impl is scanned for
    a matching def, allowing the leading underscore and a case change.

    Returns {name_lower: host directory}: an inlined helper can still be
    tested directly, and the host is where such a test lives.
    """
    hosts = {}
    for d in sorted(os.listdir(FN)):
        p = os.path.join(FN, d, 'py_impl.py')
        if not os.path.isfile(p):
            continue
        src = io.open(p, encoding='utf-8', errors='replace').read()
        for m in re.findall(r'(?m)^def _?(\w+)\s*\(', src):
            hosts.setdefault(m.lower(), d)
    return hosts


def direct_tests(host_dir, helper):
    """Source of the host's test functions that call `helper` themselves.

    An inlined helper is credited only with the assertions of tests that
    actually drive it. Handing it the host's whole test file would give LFSR
    every check PRBS13Q has, which is the opposite of measuring it.
    """
    import ast
    path = os.path.join(FN, host_dir, 'test_verify.py')
    if not os.path.isfile(path):
        return ''
    src = io.open(path, encoding='utf-8', errors='replace').read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return ''
    lines = src.split('\n')
    wanted = {helper.lower(), '_' + helper.lower()}
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith('test_'):
            continue
        called = {n.func.id.lower() for n in ast.walk(node)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        if called & wanted:
            out.append('\n'.join(lines[node.lineno - 1:node.end_lineno]))
    return '\n'.join(out)


def survey(ref=DEFAULT_REF):
    ml = matlab_functions(ref)
    known = set(ml)
    inlined = inlined_definitions()
    rows = []
    for name in sorted(ml):
        d = os.path.join(FN, name)
        impl = os.path.join(d, 'py_impl.py')
        test = os.path.join(d, 'test_verify.py')
        deps = callees(ml[name], known) - {name}
        host = inlined.get(name.lower())
        r = {'function': name,
             'kind': 'leaf' if not deps else 'composite',
             'callees': len(deps),
             'translated': os.path.isfile(impl) or host is not None,
             'inlined_only': (not os.path.isfile(impl)) and host is not None,
             'has_test': os.path.isfile(test),
             'oracle': False, 'value_checks': 0, 'shape_checks': 0, 'checks': 0}
        # An inlined helper has no directory of its own, so its tests are the
        # host's tests that call it by name -- and only those.
        src = None
        if r['inlined_only']:
            src = direct_tests(host, name)
            r['has_test'] = bool(src.strip())
            if r['has_test']:
                host_src = io.open(os.path.join(FN, host, 'test_verify.py'),
                                   encoding='utf-8', errors='replace').read()
                # the oracle marker is a comment ABOVE the block, so it is
                # looked for in the host file, not inside the function bodies
                src = src + '\n' + '\n'.join(
                    L for L in host_src.split('\n') if L.lstrip().startswith('#'))
        elif r['has_test']:
            src = io.open(test, encoding='utf-8', errors='replace').read()
        if r['has_test']:
            checks = re.findall(r'assert [^\n]*|pytest\.raises\([^\n]*'
                                r'|np\.testing\.[^\n]*', src)
            r['checks'] = len(checks)
            r['value_checks'] = sum(1 for c in checks
                                    if VALUE.search(c) and not STRUCTURAL.search(c))
            # Shape is a SEPARATE question from value, not a weaker version of
            # it. A test can pin every number and still never say how many of
            # them there are, or what orientation they come back in -- and
            # MATLAB distinguishes a row from a column where numpy's 1-D array
            # does not. pdf_to_cdf is the worked example: MATLAB returns y as a
            # column while yB, yT and x stay rows.
            r['shape_checks'] = sum(1 for c in checks if STRUCTURAL.search(c))
            r['oracle'] = any(k in src for k in ORACLE_MARKS)
        rows.append(r)
    return rows


def grade(r):
    if not r['translated']:
        return 'not translated'
    if r['function'] in NON_NUMERIC:
        return 'no numeric result'
    # An inlined helper with no test of its own is still a gap. One the host
    # drives by name is not, so it grades like any other function.
    if r.get('inlined_only') and not r['has_test']:
        return 'inlined in caller'
    if not r['has_test'] or not r['checks']:
        return 'no test'
    # The oracle marker is a comment. It says where the numbers came from, not
    # that any assertion uses them, so it only upgrades a test that already
    # pins a value -- otherwise stripping the assertions from an
    # oracle-documented test would leave the grade untouched.
    if r['value_checks'] and r['oracle']:
        return 'oracle'
    if r['value_checks']:
        return 'values pinned'
    return 'shape only'


# Functions that carry no numeric result, so "pins no value" is the right
# state for them and not a gap. Kept explicit and short: anything added here
# stops being counted, so it needs a reason that survives reading.
NON_NUMERIC = {
    'plot_modal': 'draws a figure',
    'plot_pie_com': 'draws a figure',
    'plot_bathtub_curves': 'draws a figure',
    'recolor_plots': 'restyles existing axes',
    'savefigs': 'writes figure files',
    'savefigs_png': 'writes figure files',
    'save_cmd_line': 'writes the command line to a file',
    'missingParameter': 'raises; verified with pytest.raises',
    'end_display_control': 'formats and prints the end-of-run summary',
    'OptFom_Plot_Best_Results': 'draws the best-result figures',
}

ORDER = ['oracle', 'values pinned', 'shape only', 'no numeric result',
         'inlined in caller', 'no test', 'not translated']



# ======================================================================= census
# Phase 0 of dev/prompts/PROMPT_SiCoPR_oracle_testing_2026_09_23.md: what the
# grade above cannot see. grade() calls a file 'oracle' when the marker appears
# anywhere in it and it pins at least one value. It cannot tell
#   - a block that EXECUTED the reference from one that evaluated an Octave
#     expression written for the test (the prompt's PSEUDO-ORACLE),
#   - a test that invokes Octave when it runs from one that pins literals,
#   - how loose each tolerance is, whether a complex comparison looks at the
#     phase, or which tests can silently skip.
# `--census DIR` answers those five questions and writes one CSV per question.
# It reads source only; it runs nothing.

import ast           # noqa: E402
import tokenize      # noqa: E402

MARK = 'COM Octave'
TESTS = os.path.join(_ROOT, 'tests')

# a block names the route by which the reference was executed
_ROUTE = re.compile(r'octave_oracle|oo\.(?:call|extract)\(|\bverbatim\b|'
                    r'_octave_compat|compat file|executing the reference|'
                    r"(?:COM )?Octave's own|running the reference|"
                    r'octave/patches/', re.I)
# Builtins that INSPECT a value rather than compute one. "size(H_low_xc) is
# 1x1" reports the shape of the function's own output, so a block citing one
# is attributing a result to the function under test, not evaluating Octave on
# its own. Three of the first five PSEUDO flags were exactly this.
_INSPECT = {'size', 'numel', 'length', 'isempty', 'ndims', 'iscolumn',
            'isrow', 'isvector', 'isscalar', 'class', 'isreal', 'isnan',
            'isinf', 'any', 'all'}
# Control words are in BUILTIN because the MATLAB call graph must skip them,
# but in prose "The 2-D case (what ...)" is not a call.
_KEYWORDS = {'if', 'else', 'elseif', 'end', 'for', 'while', 'switch', 'case',
             'otherwise', 'function', 'return'}
# F(args) followed by a stated result
_HEAD = re.compile(r'(?<![\w.])([A-Za-z_]\w*)\s*\((?:[^()]|\([^()]*\))*\)\s*'
                   r'(?:->|=>|gives|returns|is\b|==|=\s)')
_CALLFORM = re.compile(r'(?<![\w.])([A-Za-z_]\w*)\s*\(')


def _test_files(include_tests_dir=False):
    out = sorted(os.path.join(FN, d, 'test_verify.py') for d in os.listdir(FN)
                 if os.path.isfile(os.path.join(FN, d, 'test_verify.py')))
    if include_tests_dir:
        out += sorted(os.path.join(TESTS, f) for f in os.listdir(TESTS)
                      if f.startswith('test_') and f.endswith('.py'))
    return out


def _rel(path):
    return os.path.relpath(path, _ROOT).replace('\\', '/')


def _read(path):
    return io.open(path, encoding='utf-8', errors='replace').read()


def _comment_blocks(src):
    """Runs of comment tokens on consecutive lines: (start, end, text)."""
    out, run, start, prev = [], [], None, None
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type != tokenize.COMMENT:
                continue
            ln = t.start[0]
            if prev is not None and ln == prev + 1:
                run.append(t.string)
            else:
                if run:
                    out.append((start, prev, '\n'.join(run)))
                run, start = [t.string], ln
            prev = ln
    except tokenize.TokenError:
        pass
    if run:
        out.append((start, prev, '\n'.join(run)))
    return out


def _docstrings(tree):
    """(owner, start, end, text) for the module, each function and class."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) \
                    and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                c = b[0].value
                out.append((getattr(node, 'name', '<module>'), c.lineno,
                            c.end_lineno, c.value))
    return out


_REF_NAMES = None
_REF_BODIES = {}


def _ref_names():
    global _REF_NAMES
    if _REF_NAMES is None:
        names = set(matlab_functions(DEFAULT_REF))
        try:
            sys.path.insert(0, _HERE)
            import octave_oracle as _oo
            names |= set(_oo.function_names())
        except Exception:                                   # noqa: BLE001
            pass
        _REF_NAMES = names
    return _REF_NAMES


def _ref_body(fn):
    """The reference body of `fn`, whitespace removed, for verbatim tests."""
    if fn not in _REF_BODIES:
        body = matlab_functions(DEFAULT_REF).get(fn, '')
        _REF_BODIES[fn] = re.sub(r'\s+', '', body)
    return _REF_BODIES[fn]


def classify_block(text, fn):
    """(class, evidence) for one 'COM Octave' provenance block.

    ORACLE-RUN    the block names a reference function being called, or the
                  route (octave_oracle, a verbatim extract, the compat file)
    ORACLE-ATTR   the block states inputs and a result, or a behaviour, of the
                  function under test, without naming the route
    ORACLE-SLICE  the block cites a builtin expression that appears verbatim
                  in the reference body of the function under test, with its
                  operands bound by hand. Counts as ORACLE by the prompt's
                  definition, but the step from that expression to the
                  function's output is a reading, so it is kept apart
    PSEUDO        the block cites Octave that is not in the reference body
    MARKER        the marker alone, tagging a pinned literal
    """
    body = text.replace(MARK, ' ')
    if not re.sub(r'[#\s\-=:,]+', '', body):
        return 'MARKER', 'tags a pinned literal'
    refs = _ref_names()
    named = sorted({c for c in _CALLFORM.findall(body) if c in refs})
    # 'COM Octave (FD_Processing, 4p16p0): ...' names the function it ran
    named += [c for c in re.findall(re.escape(MARK) + r"\s*\(\s*(\w+)", text)
              if c in refs and c not in named]
    if named:
        return 'ORACLE-RUN', 'calls %s' % ','.join(named[:3])
    m = _ROUTE.search(body)
    if m:
        return 'ORACLE-RUN', 'route: %s' % m.group(0)
    heads = [h for h in _HEAD.findall(body)
             if h not in refs and h not in _KEYWORDS]
    builtin_heads = [h for h in heads if h in BUILTIN and h not in _INSPECT]
    if builtin_heads:
        rb = _ref_body(fn)
        inbody = [h for h in builtin_heads if (h + '(') in rb]
        if inbody:
            return 'ORACLE-SLICE', '%s( is in the reference body of %s' % (
                inbody[0], fn)
        return 'PSEUDO', '%s( is not in the reference body of %s' % (
            builtin_heads[0], fn)
    return 'ORACLE-ATTR', 'inputs/behaviour of %s' % fn


def census_oracle_blocks():
    """One row per provenance block in the per-function tests."""
    rows = []
    for p in _test_files():
        fn = os.path.basename(os.path.dirname(p))
        src = _read(p)
        route_in_file = bool(_ROUTE.search(src))
        blocks = [(s, e, t, 'comment') for s, e, t in _comment_blocks(src)]
        blocks += [(s, e, t, 'docstring:' + o)
                   for o, s, e, t in _docstrings(ast.parse(src))]
        for s, e, t, kind in blocks:
            if MARK not in t:
                continue
            cls, ev = classify_block(t, fn)
            rows.append({'file': _rel(p), 'line': s, 'function': fn,
                         'kind': kind, 'class': cls, 'evidence': ev,
                         'route_documented_in_file': route_in_file,
                         'text': ' '.join(t.split())[:200]})
    return rows


_OCT_NAME = re.compile(r'\b_?OCT(?:_\w+)?\b|\b_OCT\w*|\b_TRACE_\w*|'
                       r'\b_\w*_OCT\b|\b_\w*_OCT_\w*|\bORACLE\w*\b')
# a section header that governs the tests below it
_ORACLE_HEADER = re.compile(r'COM Octave|\boracle\b|executed.reference|'
                            r'against the reference', re.I)


def census_tests_backed():
    """Per test function: does an oracle block govern it?

    A test is oracle-backed if its docstring is a provenance block, if the
    nearest multi-line comment block above it carries the marker, if its body
    or its parametrize decorator reads a pinned `_OCT*` / `_TRACE_*` table, or
    if its body carries the marker on a literal. Everything else is the
    prompt's NEITHER: a property, a shape, or a value from a reading.
    """
    rows = []
    for p in _test_files():
        fn = os.path.basename(os.path.dirname(p))
        src = _read(p)
        lines = src.split('\n')
        tree = ast.parse(src)
        headers = [(s, e, t) for s, e, t in _comment_blocks(src) if e > s]
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) \
                    or not node.name.startswith('test_'):
                continue
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            seg = '\n'.join(lines[start - 1:node.end_lineno])
            doc = ast.get_docstring(node) or ''
            above = [h for h in headers if h[1] < start]
            gov = above[-1] if above else None
            why = ''
            if MARK in doc:
                why = 'docstring'
            elif gov and _ORACLE_HEADER.search(gov[2]):
                why = 'section header line %d' % gov[0]
            elif _OCT_NAME.search(seg):
                why = 'reads %s' % _OCT_NAME.search(seg).group(0)
            elif MARK in seg:
                why = 'marker in body'
            rows.append({'file': _rel(p), 'line': node.lineno,
                         'function': fn, 'test': node.name,
                         'oracle_backed': bool(why), 'why': why})
    return rows


def _call_name(node):
    f = node.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return '.'.join(reversed(parts))


def _enclosing(tree):
    """{id(node): name of the innermost enclosing function, or '<module>'}."""
    owner = {}

    def walk(node, name):
        for ch in ast.iter_child_nodes(node):
            nm = ch.name if isinstance(ch, (ast.FunctionDef,
                                            ast.AsyncFunctionDef)) else name
            owner[id(ch)] = nm
            walk(ch, nm)
    walk(tree, '<module>')
    return owner


_RUNS_OCTAVE = re.compile(r'(?:^|\.)(?:call|extract|_find_octave|find_octave)$')


def census_octave_at_test_time():
    """Per file referencing octave_oracle: does anything run Octave when the
    tests run, or is the name only quoted in a comment or docstring?"""
    rows = []
    for p in _test_files(include_tests_dir=True):
        src = _read(p)
        if 'octave' not in src.lower():
            continue
        tree = ast.parse(src)
        owner = _enclosing(tree)
        imports = any(
            (isinstance(n, ast.Import) and any('octave' in a.name for a in n.names))
            or (isinstance(n, ast.ImportFrom) and n.module
                and 'octave' in n.module)
            for n in ast.walk(tree))
        sites = []
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            nm = _call_name(n)
            hit = bool(_RUNS_OCTAVE.search(nm)) and (
                'oo' in nm.split('.')[0] or 'octave' in nm.lower()
                or nm.split('.')[0] in ('octave_oracle',))
            if not hit and nm.split('.')[-1] in ('run', 'Popen', 'check_output',
                                                  'call') \
                    and nm.startswith('subprocess'):
                hit = 'octave' in ast.get_source_segment(src, n).lower()
            if hit:
                sites.append('%s@%d in %s' % (nm, n.lineno,
                                              owner.get(id(n), '<module>')))
        in_code_ref = 'octave_oracle' in re.sub(r'#.*', '', src)
        rows.append({'file': _rel(p),
                     'references_octave_oracle': 'octave_oracle' in src,
                     'imports_it': imports,
                     'invokes_octave_when_run': bool(sites),
                     'sites': '; '.join(sites)[:300],
                     'mention_outside_comments': in_code_ref})
    return rows


# defaults are part of the tolerance: a bare np.allclose is rtol 1e-5
_TOL_FUNCS = {
    'assert_allclose': ({'rtol': 1e-7, 'atol': 0.0}, ('rtol', 'atol')),
    'allclose': ({'rtol': 1e-5, 'atol': 1e-8}, ('rtol', 'atol')),
    'isclose': ({'rtol': 1e-5, 'atol': 1e-8}, ('rtol', 'atol')),
    'approx': ({'rel': 1e-6, 'abs': 1e-12}, ('rel', 'abs')),
    'assert_almost_equal': ({'decimal': 7}, ('decimal',)),
    'assert_array_almost_equal': ({'decimal': 6}, ('decimal',)),
}
_REL_NAME = re.compile(r'\brel\w*|\bratio|/\s*|\*\s*max\(|\bworst_rel', re.I)
_DIFF_NAME = re.compile(r'abs\(|\brel\w*|\berr\w*|\bworst\w*|\bdiff\w*|'
                        r'\bdelta\w*|\bd\b|\bdev\w*', re.I)


def _num(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _num(node.operand)
        return None if v is None else -v
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        a, b = _num(node.left), _num(node.right)
        return None if a is None or b is None else a ** b
    return None


def _implied_rel(left, tol):
    """tol / |expected| when the comparison is abs(x - LITERAL)."""
    for sub_ in ast.walk(left):
        if isinstance(sub_, ast.BinOp) and isinstance(sub_.op, ast.Sub):
            for side in (sub_.right, sub_.left):
                v = _num(side)
                if v is None and isinstance(side, ast.BinOp):
                    a, b = _num(side.left), _num(side.right)
                    if a is not None and b is not None:
                        v = {ast.Mult: a * b, ast.Add: a + b}.get(
                            type(side.op))
                if v:
                    return tol / abs(v)
    return ''


def _justification(lines, lineno, end_lineno):
    """A comment on the same line(s), or the comment line(s) just above."""
    got = []
    for ln in range(lineno, end_lineno + 1):
        s = lines[ln - 1]
        if '#' in s:
            c = s[s.index('#') + 1:].strip()
            if c and c != MARK:
                got.append(c)
    k = lineno - 2
    above = []
    while k >= 0 and lines[k].lstrip().startswith('#'):
        above.insert(0, lines[k].strip('# ').strip())
        k -= 1
    got += [a for a in above[-3:] if a and a != MARK]
    return ' | '.join(got)[:160]


def census_vacuous():
    """Assertions that cannot fail: `assert X or True`, `assert True`.

    grade() counts these as value checks, so each one is coverage that is not
    there. MMSE/test_verify.py:102 was found reading the tolerance census.
    """
    rows = []
    for p in _test_files(include_tests_dir=True):
        src = _read(p)
        for n in ast.walk(ast.parse(src)):
            if not isinstance(n, ast.Assert):
                continue
            t = n.test
            dead = isinstance(t, ast.Constant) and bool(t.value)
            if isinstance(t, ast.BoolOp) and isinstance(t.op, ast.Or):
                dead = dead or any(isinstance(v, ast.Constant) and bool(v.value)
                                   for v in t.values)
            if dead:
                rows.append({'file': _rel(p), 'line': n.lineno,
                             'source': ' '.join((ast.get_source_segment(src, n)
                                                 or '').split())[:160]})
    return rows


def census_tolerance():
    """Every comparison looser than rtol=1e-6 or atol=1e-9, defaults included."""
    rows = []
    for p in _test_files(include_tests_dir=True):
        src = _read(p)
        lines = src.split('\n')
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                short = _call_name(n).split('.')[-1]
                if short not in _TOL_FUNCS:
                    continue
                defaults, keys = _TOL_FUNCS[short]
                eff = dict(defaults)
                explicit = set()
                for kw in n.keywords:
                    if kw.arg in keys:
                        v = _num(kw.value)
                        if v is not None:
                            eff[kw.arg] = v
                            explicit.add(kw.arg)
                if short.endswith('almost_equal'):
                    rel, ab = 0.0, 1.5 * 10 ** (-eff['decimal'])
                elif short == 'approx':
                    rel, ab = eff['rel'], eff['abs']
                else:
                    rel, ab = eff['rtol'], eff['atol']
                if rel > 1e-6 or ab > 1e-9:
                    rows.append({
                        'file': _rel(p), 'line': n.lineno, 'form': short,
                        'rtol': rel, 'atol': ab,
                        'from_default': ','.join(sorted(set(keys) - explicit))
                                        or '',
                        'implied_rel': '',
                        'justification': _justification(lines, n.lineno,
                                                         n.end_lineno)})
            elif isinstance(n, ast.Compare) and len(n.ops) == 1 \
                    and isinstance(n.ops[0], (ast.Lt, ast.LtE)):
                seg_l = ast.get_source_segment(src, n.left) or ''
                rhs = n.comparators[0]
                v = _num(rhs)
                scaled = False
                if v is None and isinstance(rhs, ast.BinOp) \
                        and isinstance(rhs.op, ast.Mult):
                    v = _num(rhs.left) if _num(rhs.left) is not None \
                        else _num(rhs.right)
                    scaled = v is not None
                if v is None or v <= 0 or not _DIFF_NAME.search(seg_l):
                    continue
                relative = scaled or bool(_REL_NAME.search(seg_l))
                if (relative and v > 1e-6) or (not relative and v > 1e-9):
                    rows.append({
                        'file': _rel(p), 'line': n.lineno,
                        'form': 'compare %s' % ('relative' if relative
                                                else 'absolute'),
                        'rtol': v if relative else 0.0,
                        'atol': 0.0 if relative else v, 'from_default': '',
                        'implied_rel': '' if relative
                                       else _implied_rel(n.left, v),
                        'justification': _justification(lines, n.lineno,
                                                         n.end_lineno)})
    return rows


_COMPLEXY = re.compile(r'\b\d+(?:\.\d+)?j\b|complex\(|dtype=complex|'
                       r'\.imag\b|np\.conj|1j\b')


def _is_mag(node):
    """abs()/np.abs() of something that is not a difference."""
    if not isinstance(node, ast.Call):
        return False
    nm = _call_name(node)
    if nm.split('.')[-1] not in ('abs', 'absolute'):
        return False
    if not node.args:
        return False
    a = node.args[0]
    return not (isinstance(a, ast.BinOp) and isinstance(a.op, ast.Sub))


def _direct_operands(node):
    """The expressions an assertion actually compares, not their scales.

    `abs(got - want) <= 1e-11 * max(abs(want), 1e-300)` compares a DIFFERENCE;
    the abs() inside the scale is not an operand. Looking at the whole subtree
    flagged that form as magnitude-only, which it is not.
    """
    if isinstance(node, ast.Assert):
        t = node.test
        if isinstance(t, ast.Compare):
            return [t.left] + list(t.comparators)
        if isinstance(t, ast.Call) and _call_name(t).split('.')[-1] in (
                'allclose', 'isclose', 'array_equal', 'array_equiv'):
            return list(t.args[:2])
        return []
    return list(node.args[:2])


def census_complex():
    """Comparisons in files that handle complex data: full, or magnitude only.

    FULL       the assertion carries a complex literal, complex(), .imag, or
               an angle alongside the magnitude
    MAGNITUDE  a compared operand IS abs(X) (not abs of a difference), with
               nothing looking at the phase
    REAL       .real without .imag
    A MAGNITUDE or REAL row is `companioned` when the same test function also
    makes a FULL comparison, so the phase is checked elsewhere in that test.
    """
    rows = []
    for p in _test_files(include_tests_dir=True):
        src = _read(p)
        if not _COMPLEXY.search(src):
            continue
        tree = ast.parse(src)
        owner = _enclosing(tree)
        found = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Assert):
                pass
            elif isinstance(n, ast.Call) and _call_name(n).split('.')[-1] in (
                    'assert_allclose', 'assert_array_equal', 'assert_equal',
                    'assert_array_almost_equal', 'assert_almost_equal'):
                pass
            else:
                continue
            seg = ast.get_source_segment(src, n) or ''
            phase = bool(re.search(r'angle|unwrap|phase', seg))
            literal = bool(re.search(r'\d+(?:\.\d+)?j\b|complex\(', seg))
            has_imag, has_real = '.imag' in seg or 'np.imag' in seg, \
                '.real' in seg or 'np.real' in seg
            mag = any(_is_mag(op) for op in _direct_operands(n))
            if literal or has_imag or (mag and phase):
                cls = 'FULL'
            elif mag:
                cls = 'MAGNITUDE'
            elif has_real:
                cls = 'REAL'
            else:
                continue
            found.append((owner.get(id(n), '<module>'), n.lineno, cls,
                          ' '.join(seg.split())[:160]))
        full_in = {o for o, _l, c, _s in found if c == 'FULL'}
        for o, ln, cls, seg in found:
            rows.append({'file': _rel(p), 'line': ln, 'test': o, 'class': cls,
                         'companioned': cls != 'FULL' and o in full_in,
                         'source': seg})
    return rows


# An instruction: an environment variable (it has an underscore, so 'MATLAB'
# is not one) or a verb telling the reader what to do. Case-sensitive on the
# variable, so the first draft's re.I, which let every capitalised word count,
# is gone.
_ENABLE = re.compile(r'\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b|'
                     r'(?i:\bset\b|\binstall\b|\bpoint\b|\benable\b|'
                     r'\bexport\b|\bpip\b|\brun\b|\bsee\b)')


def census_skip():
    """Every way a test can skip, with its condition and whether it says how
    to make it run."""
    rows = []
    for p in _test_files(include_tests_dir=True):
        src = _read(p)
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                nm = _call_name(n)
                short = nm.split('.')[-1]
                if short not in ('skip', 'skipif', 'importorskip', 'skipTest'):
                    continue
                if short in ('skip', 'skipif') and 'pytest' not in nm \
                        and 'mark' not in nm:
                    continue
                seg = ' '.join((ast.get_source_segment(src, n) or '').split())
                reason = ''
                for kw in n.keywords:
                    if kw.arg == 'reason':
                        reason = ' '.join((ast.get_source_segment(src, kw.value)
                                           or '').split())
                if not reason and n.args:
                    last = n.args[-1]
                    if isinstance(last, (ast.Constant, ast.JoinedStr,
                                         ast.BinOp)):
                        reason = ' '.join((ast.get_source_segment(src, last)
                                           or '').split())
                if short == 'importorskip':
                    how = 'implicit: install the package'
                else:
                    how = 'yes' if _ENABLE.search(reason) else 'no'
                rows.append({'file': _rel(p), 'line': n.lineno, 'form': short,
                             'condition': seg[:160],
                             'says_how_to_enable': how,
                             'reason': reason[:160]})
        for i, ln in enumerate(src.split('\n'), 1):
            if re.search(r'''print\(\s*f?["']SKIP''', ln):
                # a SKIP message often runs to several print lines; the
                # stage-oracle one names its variable on the fifth
                block = ' '.join(src.split('\n')[i - 1:i + 8])
                rows.append({'file': _rel(p), 'line': i, 'form': 'print SKIP',
                             'condition': ' '.join(ln.split())[:160],
                             'says_how_to_enable': 'yes' if _ENABLE.search(block)
                                                   else 'no',
                             'reason': ' '.join(block.split())[:160]})
    return rows


def _write(rows, path):
    if not rows:
        io.open(path, 'w', encoding='utf-8').write('')
        return
    with io.open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def census_main(outdir):
    from collections import Counter
    os.makedirs(outdir, exist_ok=True)

    blocks = census_oracle_blocks()
    _write(blocks, os.path.join(outdir, 'census_oracle_blocks.csv'))
    c = Counter(r['class'] for r in blocks)
    print('1. PROVENANCE BLOCKS in %d per-function test files: %d'
          % (len({r['file'] for r in blocks}), len(blocks)))
    for k in ('ORACLE-RUN', 'ORACLE-ATTR', 'ORACLE-SLICE', 'PSEUDO', 'MARKER'):
        print('     %-13s %4d' % (k, c.get(k, 0)))
    attr = [r for r in blocks if r['class'] == 'ORACLE-ATTR']
    print('     ORACLE-ATTR in a file that documents the route: %d of %d'
          % (sum(r['route_documented_in_file'] for r in attr), len(attr)))
    for r in blocks:
        if r['class'] == 'PSEUDO':
            print('     PSEUDO %s:%d  %s' % (r['file'], r['line'], r['evidence']))

    tests = census_tests_backed()
    _write(tests, os.path.join(outdir, 'census_tests.csv'))
    nb = sum(r['oracle_backed'] for r in tests)
    print('\n   TEST FUNCTIONS in those files: %d; governed by an oracle block '
          '%d, not %d' % (len(tests), nb, len(tests) - nb))

    oc = census_octave_at_test_time()
    _write(oc, os.path.join(outdir, 'census_octave_runtime.csv'))
    ref = [r for r in oc if r['references_octave_oracle']]
    inv = [r for r in oc if r['invokes_octave_when_run']]
    print('\n2. FILES REFERENCING octave_oracle: %d; importing it %d; '
          'INVOKING OCTAVE WHEN RUN %d'
          % (len(ref), sum(r['imports_it'] for r in ref), len(inv)))
    for r in inv:
        print('     %s  %s' % (r['file'], r['sites'][:110]))

    tol = census_tolerance()
    _write(tol, os.path.join(outdir, 'census_tolerance.csv'))
    tight_in_units = [r for r in tol if r['implied_rel'] != ''
                      and float(r['implied_rel']) <= 1e-6]
    print('\n3. COMPARISONS LOOSER THAN rtol=1e-6 / atol=1e-9: %d, '
          '%d of them with a written justification, %d loose only by DEFAULT, '
          '%d absolute bounds that are <= 1e-6 RELATIVE to their own literal'
          % (len(tol), sum(1 for r in tol if r['justification']),
             sum(1 for r in tol if r['from_default']), len(tight_in_units)))
    vac = census_vacuous()
    _write(vac, os.path.join(outdir, 'census_vacuous.csv'))
    print('   ASSERTIONS THAT CANNOT FAIL: %d' % len(vac))
    for r in vac:
        print('     %s:%d  %s' % (r['file'], r['line'], r['source'][:90]))

    cx = census_complex()
    _write(cx, os.path.join(outdir, 'census_complex.csv'))
    cc = Counter(r['class'] for r in cx)
    lone = [r for r in cx if r['class'] != 'FULL' and not r['companioned']]
    print('\n4. COMPLEX COMPARISONS: FULL %d, MAGNITUDE-ONLY %d, REAL-ONLY %d; '
          'of the last two, %d with no FULL comparison in the same test'
          % (cc.get('FULL', 0), cc.get('MAGNITUDE', 0), cc.get('REAL', 0),
             len(lone)))

    sk = census_skip()
    _write(sk, os.path.join(outdir, 'census_skip.csv'))
    hc = Counter(r['says_how_to_enable'] for r in sk)
    print('\n5. SKIP SITES: %d; say how to enable: yes %d, no %d, '
          'implicit (a package to install) %d'
          % (len(sk), hc.get('yes', 0), hc.get('no', 0),
             hc.get('implicit: install the package', 0)))
    for var in ('COM_STAGE_ORACLES', 'COM_TEST_FIXTURES',
                'COM_OCTAVE_CHECKPOINTS'):
        print('   %-24s %s' % (var, 'set' if os.environ.get(var)
                                 else 'NOT SET in this environment'))
    print('\n-> %s' % outdir)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default=DEFAULT_REF)
    ap.add_argument('--gaps', action='store_true',
                    help='list the functions that are not value-checked')
    ap.add_argument('--csv', help='write the full table here')
    ap.add_argument('--census', metavar='DIR',
                    help='write the five Phase 0 censuses here and '
                         'stop: provenance, Octave at test time, '
                         'tolerance, complex, skip')
    a = ap.parse_args(argv)
    if a.census:
        return census_main(a.census)

    rows = survey(a.ref)
    for r in rows:
        r['grade'] = grade(r)

    print('MATLAB reference: %s' % os.path.basename(a.ref))
    print('%d functions defined, %d translated\n'
          % (len(rows), sum(1 for r in rows if r['translated'])))

    hdr = '%-16s %7s %9s %8s' % ('', 'leaf', 'composite', 'all')
    print(hdr)
    print('-' * len(hdr))
    for g in ORDER:
        leaf = sum(1 for r in rows if r['grade'] == g and r['kind'] == 'leaf')
        comp = sum(1 for r in rows if r['grade'] == g and r['kind'] == 'composite')
        if leaf or comp:
            print('%-16s %7d %9d %8d' % (g, leaf, comp, leaf + comp))
    n_leaf = sum(1 for r in rows if r['kind'] == 'leaf')
    n_comp = len(rows) - n_leaf
    print('-' * len(hdr))
    print('%-16s %7d %9d %8d' % ('total', n_leaf, n_comp, len(rows)))

    # The denominator that matters is the functions that produce a number.
    bearing = [r for r in rows if r['grade'] != 'no numeric result']
    checked = [r for r in bearing if r['grade'] in ('oracle', 'values pinned')]
    b_leaf = [r for r in bearing if r['kind'] == 'leaf']
    c_leaf = [r for r in checked if r['kind'] == 'leaf']
    n_oracle = sum(1 for r in bearing if r['grade'] == 'oracle')
    print('\n%d of %d functions carry no numeric result: plots, file writes, '
          'an error raiser' % (len(rows) - len(bearing), len(rows)))
    print('verified against a value : %d of %d value-bearing (%.0f%%)'
          % (len(checked), len(bearing), 100.0 * len(checked) / len(bearing)))
    print('  of the leaf functions  : %d of %d (%.0f%%)'
          % (len(c_leaf), len(b_leaf), 100.0 * len(c_leaf) / len(b_leaf)))
    print('  against the reference  : %d of %d (%.0f%%)'
          % (n_oracle, len(bearing), 100.0 * n_oracle / len(bearing)))

    # Shape is tracked separately: pinning every number says nothing about how
    # many there are or which way round they come back.
    n_shape = sum(1 for r in bearing if r['shape_checks'])
    n_both = sum(1 for r in bearing if r['shape_checks'] and r['value_checks'])
    print('shape also checked       : %d of %d (%.0f%%)'
          % (n_shape, len(bearing), 100.0 * n_shape / len(bearing)))
    print('  value AND shape        : %d of %d (%.0f%%)'
          % (n_both, len(bearing), 100.0 * n_both / len(bearing)))

    if a.gaps:
        for kind in ('leaf', 'composite'):
            bad = [r for r in rows
                   if r['kind'] == kind and r['grade'] in ('shape only', 'no test',
                                                           'not translated')]
            print('\n%s functions not value-checked (%d):' % (kind, len(bad)))
            for r in bad:
                print('   %-38s %-14s %d checks, %d callees'
                      % (r['function'], r['grade'], r['checks'], r['callees']))

    if a.csv:
        with open(a.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('\nfull table -> %s' % a.csv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
