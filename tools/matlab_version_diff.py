"""Turn a new MATLAB COM release into a concrete Python worklist.

When Hansel publishes the next com_ieee8023_*.m, the question is not "what
changed in MATLAB" but "which py_impl.py files must I now re-check". This
answers that directly: it diffs the two .m files function by function, then
maps each changed MATLAB function onto the Python source that translates it.

It also uses something the port already carries: most py_impl.py files have a
comment naming the MATLAB lines they were translated from ("MATLAB lines:
2580-2692"). When a MATLAB edit shifts line numbers, those citations go stale
silently -- every later reviewer then reads the wrong side of the reference.
This reports which citations no longer point where they used to.

Nothing here needs anything from the MATLAB author beyond the .m file itself.

    python tools/matlab_version_diff.py OLD.m NEW.m
    python tools/matlab_version_diff.py --self-check
    python tools/matlab_version_diff.py OLD.m NEW.m --json out.json

Exit status is 0 always: this is a report, not a gate.
"""
import argparse
import difflib
import io
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FUNC_RE = re.compile(r'^\s*function\s+(?:\[(?P<outs>[^\]]*)\]|(?P<out>[\w.]+))?'
                     r'\s*=?\s*(?P<name>\w+)\s*\((?P<args>[^)]*)\)')
# "MATLAB lines: 2580-2692" / "MATLAB line 4592" / "ML 7976"
CITE_RE = re.compile(r'(?:MATLAB|ML)\s*(?:source\s*)?lines?\s*[:#]?\s*'
                     r'(\d{3,5})\s*(?:[-–]\s*(\d{3,5}))?', re.I)


def parse_functions(path):
    """-> {name: {'start','end','body','sig'}} for one .m file."""
    lines = io.open(path, encoding='utf-8', errors='replace').read().splitlines()
    marks = []
    for i, line in enumerate(lines):
        m = FUNC_RE.match(line)
        if m:
            marks.append((i, m.group('name'), line.strip()))
    out = {}
    for k, (i, name, sig) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
        out[name] = {'start': i + 1, 'end': end, 'sig': sig,
                     'body': lines[i:end]}
    return out


def normalise(body):
    """Ignore comment-only lines, trailing comments and blank space."""
    out = []
    for raw in body:
        s = raw.split('%')[0].rstrip()
        if s.strip():
            out.append(' '.join(s.split()))
    return out


def py_impl_path(name):
    p = os.path.join(_ROOT, 'com_functions', 'fn', name, 'py_impl.py')
    return p if os.path.exists(p) else None


def citations(name):
    """MATLAB line ranges cited by a function's py_impl.py."""
    p = py_impl_path(name)
    if not p:
        return []
    txt = io.open(p, encoding='utf-8', errors='replace').read()
    out = []
    for m in CITE_RE.finditer(txt):
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        out.append((lo, hi))
    return out


def compare(old_path, new_path):
    old, new = parse_functions(old_path), parse_functions(new_path)
    names = set(old) | set(new)
    rep = {'old': os.path.basename(old_path), 'new': os.path.basename(new_path),
           'added': [], 'removed': [], 'changed': [], 'identical': [],
           'stale_citations': []}

    for name in sorted(names):
        if name not in old:
            rep['added'].append({'name': name,
                                 'ported': bool(py_impl_path(name)),
                                 'lines': (new[name]['start'], new[name]['end'])})
            continue
        if name not in new:
            rep['removed'].append({'name': name,
                                   'ported': bool(py_impl_path(name))})
            continue

        a, b = normalise(old[name]['body']), normalise(new[name]['body'])
        if a == b:
            rep['identical'].append(name)
            continue

        diff = list(difflib.unified_diff(a, b, 'old/' + name, 'new/' + name,
                                         lineterm='', n=0))
        adds = sum(1 for d in diff[2:] if d.startswith('+'))
        dels = sum(1 for d in diff[2:] if d.startswith('-'))
        rep['changed'].append({
            'name': name,
            'py_impl': (os.path.relpath(py_impl_path(name), _ROOT)
                        if py_impl_path(name) else None),
            'added_lines': adds, 'removed_lines': dels,
            'old_lines': (old[name]['start'], old[name]['end']),
            'new_lines': (new[name]['start'], new[name]['end']),
            'diff': diff[:40],
        })

    # Citation drift: a py_impl citing lines that no longer bound its function.
    for name in sorted(names & set(new)):
        if not py_impl_path(name):
            continue
        lo_n, hi_n = new[name]['start'], new[name]['end']
        for lo, hi in citations(name):
            # Only judge citations that plausibly refer to this function.
            if name in old:
                lo_o, hi_o = old[name]['start'], old[name]['end']
                inside_old = lo_o <= lo <= hi_o
            else:
                inside_old = False
            if inside_old and not (lo_n <= lo <= hi_n):
                entry = {'name': name,
                         'py_impl': os.path.relpath(py_impl_path(name), _ROOT),
                         'cited': [lo, hi], 'now_at': [lo_n, hi_n]}
                if entry not in rep['stale_citations']:
                    rep['stale_citations'].append(entry)
    return rep


def render(rep, show_diff=False):
    print('MATLAB version diff: %s -> %s' % (rep['old'], rep['new']))
    print('=' * 72)
    print('  identical bodies : %d' % len(rep['identical']))
    print('  CHANGED          : %d' % len(rep['changed']))
    print('  added in new     : %d' % len(rep['added']))
    print('  removed in new   : %d' % len(rep['removed']))
    print('  stale line refs  : %d' % len(rep['stale_citations']))

    ported = [c for c in rep['changed'] if c['py_impl']]
    unported = [c for c in rep['changed'] if not c['py_impl']]

    if ported:
        print('\nPYTHON FILES TO RE-CHECK (%d)' % len(ported))
        print('-' * 72)
        for c in sorted(ported, key=lambda x: -(x['added_lines']
                                                + x['removed_lines'])):
            print('  %-34s %+4d/-%-4d  ML %d-%d  ->  %s'
                  % (c['name'], c['added_lines'], c['removed_lines'],
                     c['new_lines'][0], c['new_lines'][1], c['py_impl']))
            if show_diff:
                for d in c['diff'][2:12]:
                    print('        %s' % d)

    if unported:
        print('\nCHANGED BUT NOT PORTED AS A FUNCTION (%d)' % len(unported))
        print('-' * 72)
        print('  ' + ', '.join(c['name'] for c in unported))
        print('  (inlined into a caller, or not translated -- check by hand)')

    if rep['added']:
        print('\nNEW MATLAB FUNCTIONS (%d)' % len(rep['added']))
        print('-' * 72)
        for a in rep['added']:
            print('  %-34s %s' % (a['name'],
                                  'already ported' if a['ported']
                                  else 'NO py_impl -- needs translation'))

    if rep['removed']:
        print('\nREMOVED FROM MATLAB (%d)' % len(rep['removed']))
        print('-' * 72)
        for r in rep['removed']:
            print('  %-34s %s' % (r['name'],
                                  'py_impl still present' if r['ported'] else ''))

    if rep['stale_citations']:
        print('\nSTALE "MATLAB lines:" CITATIONS (%d)' % len(rep['stale_citations']))
        print('-' * 72)
        for s in rep['stale_citations'][:30]:
            print('  %-30s cites %s, now at %s  (%s)'
                  % (s['name'], s['cited'], s['now_at'], s['py_impl']))


def self_check():
    """Validate against a known result: 4p14p0 -> 4p15p0.

    The conversion audit (docs/AUDIT_FINDINGS.md, batch B13) established that
    101 function bodies are byte-identical between these two versions. If this
    tool does not land near that, its parsing is wrong.
    """
    old = os.path.join(_ROOT, 'matlab', 'com_ieee8023_4p14p0.m')
    new = os.path.join(_ROOT, 'matlab', 'com_ieee8023_4p15p0.m')
    if not (os.path.exists(old) and os.path.exists(new)):
        print('self-check needs both matlab/com_ieee8023_4p14p0.m and 4p15p0.m')
        return
    rep = compare(old, new)
    render(rep)
    n = len(rep['identical'])
    print('\nSELF-CHECK: %d identical bodies; audit B13 reported 101 '
          'byte-identical 4p14==4p15 bodies.' % n)
    print('  %s' % ('consistent (this tool ignores comment-only differences, '
                    'so >= 101 is expected)' if n >= 101
                    else 'BELOW the audited figure -- parsing is suspect'))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('old', nargs='?', help='previous com_ieee8023_*.m')
    ap.add_argument('new', nargs='?', help='new com_ieee8023_*.m')
    ap.add_argument('--self-check', action='store_true',
                    help='validate the tool against 4p14p0 -> 4p15p0')
    ap.add_argument('--show-diff', action='store_true',
                    help='print the first lines of each function diff')
    ap.add_argument('--json', metavar='PATH', help='also write the report as JSON')
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return 0
    if not (args.old and args.new):
        ap.error('give OLD.m and NEW.m, or --self-check')

    rep = compare(args.old, args.new)
    render(rep, show_diff=args.show_diff)
    if args.json:
        with io.open(args.json, 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=1)
        print('\nJSON report -> %s' % args.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
