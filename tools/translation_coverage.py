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
DEFAULT_REF = os.path.join(_ROOT, 'matlab', 'com_ieee8023_4p16p0.m')

# a test that quotes the reference, rather than a reading of it
ORACLE_MARKS = ('COM Octave', 'octave_oracle', 'COM_Octave')
# an assertion that pins a value rather than a shape or a type
VALUE = re.compile(r'\d+\.\d|\de-\d|\de\+\d|approx|allclose|isclose'
                   r'|assert_array|assert_almost|== \[|== _OCT|_OCT\w*\[')
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
    """
    names = set()
    for d in os.listdir(FN):
        p = os.path.join(FN, d, 'py_impl.py')
        if not os.path.isfile(p):
            continue
        src = io.open(p, encoding='utf-8', errors='replace').read()
        names |= {m.lower()
                  for m in re.findall(r'(?m)^def _?(\w+)\s*\(', src)}
    return names


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
        r = {'function': name,
             'kind': 'leaf' if not deps else 'composite',
             'callees': len(deps),
             'translated': os.path.isfile(impl) or name.lower() in inlined,
             'inlined_only': (not os.path.isfile(impl)) and name.lower() in inlined,
             'has_test': os.path.isfile(test),
             'oracle': False, 'value_checks': 0, 'checks': 0}
        if r['has_test']:
            src = io.open(test, encoding='utf-8', errors='replace').read()
            checks = re.findall(r'assert [^\n]*|pytest\.raises\([^\n]*'
                                r'|np\.testing\.[^\n]*', src)
            r['checks'] = len(checks)
            r['value_checks'] = sum(1 for c in checks if VALUE.search(c))
            r['oracle'] = any(k in src for k in ORACLE_MARKS)
        rows.append(r)
    return rows


def grade(r):
    if not r['translated']:
        return 'not translated'
    if r['function'] in NON_NUMERIC:
        return 'no numeric result'
    if r.get('inlined_only'):
        return 'inlined in caller'
    if not r['has_test'] or not r['checks']:
        return 'no test'
    if r['oracle']:
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
}

ORDER = ['oracle', 'values pinned', 'shape only', 'no numeric result',
         'inlined in caller', 'no test', 'not translated']


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default=DEFAULT_REF)
    ap.add_argument('--gaps', action='store_true',
                    help='list the functions that are not value-checked')
    ap.add_argument('--csv', help='write the full table here')
    a = ap.parse_args(argv)

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
