"""The answer to "are there any opens?", computed rather than remembered.

`dev/prompts/SiCoPR_VERIFICATION_OBJECTIVE.md` sets the objective: reach a state
where that question is answered by running this, and it prints zero. Completion
is never asserted from judgment. The report IS the answer, and it derives that
answer from executing the suite.

## Why this does not read tests/results.csv

`tests/audit_check.py` appends every check to `tests/results.csv`, which by now
holds 45,491 rows going back to 2026-09-04, with no column saying which script
a check came from. Reading "the last result for each check name" out of that
would mean a check whose script was deleted, or whose name was changed, keeps
reporting PASS forever, from a run that happened weeks ago.

That is the 2026-07 ledger exactly: a stored verdict nobody re-evaluates. So
the log is treated as a log. This runs the gates and reads what they say now.

The cost is about 90 seconds. That is the price of an answer that is true when
it is printed, and it is worth paying for the one command whose whole job is to
be trusted.

## Determinism

Same tree, same output. Nothing here prints a wall-clock time or a duration,
because `--approve` stores the report next to the git sha it was computed from,
and the promise is that anyone can check out that sha, run this, and get the
same text back. A report that cannot reproduce at its own sha is itself a
finding.

## What this does NOT measure

Stated plainly, because a report that quietly implies more coverage than it has
is worse than no report. This aggregates the gates that exist. It does not
implement the inventory row set from the contract, per-row `oracle_id`s, Octave
exposure sets, or waiver approval. Those remain unbuilt, and the closing
section says so on every run rather than leaving a reader to assume.

    python com_functions/verification/report.py
    python com_functions/verification/report.py --approve
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import datetime
import glob
import io
import json
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
TESTS = os.path.join(_ROOT, 'tests')
APPROVED = os.path.join(_HERE, 'approved')

# The four pytest modules in tests/ are not audit scripts; run_all.ps1 drives
# them with pytest. Everything else there that imports audit_check is a gate.
PYTEST_MODULES = {'test_smoke.py', 'test_checkpoints.py', 'test_end_to_end.py',
                  'test_export_columns.py'}

# `xcheck` prints `XFAIL <name> (known divergence): <reason>`, so the suffix
# between the name and the colon is not optional decoration: a pattern that
# demands the colon straight after the name silently drops every accepted
# divergence, and the report then says there are none. It said exactly that on
# its first run, with six xcheck calls sitting in the suite.
RESULT = re.compile(
    r'^(PASS|FAIL|XFAIL|XPASS) (\S+)(?: \(known divergence\))?(?:: (.*))?$')


def audit_scripts():
    out = []
    for p in sorted(glob.glob(os.path.join(TESTS, 'test_*.py'))):
        if os.path.basename(p) in PYTEST_MODULES:
            continue
        src = io.open(p, encoding='utf-8', errors='replace').read()
        if 'audit_check' in src:
            out.append(p)
    return out


def run_gate(path):
    """Run one audit script. Returns (checks, stdout, crashed)."""
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    r = subprocess.run([sys.executable, '-B', path], cwd=_ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       env=env)
    out = r.stdout.decode('utf-8', 'replace')
    checks = []
    for line in out.splitlines():
        m = RESULT.match(line.strip())
        if m:
            checks.append((m.group(1), m.group(2), (m.group(3) or '').strip()))
    # A script that exits non-zero having reported no FAIL/XPASS has crashed,
    # which must never read as "no opens".
    bad = [c for c in checks if c[0] in ('FAIL', 'XPASS')]
    crashed = r.returncode != 0 and not bad
    return checks, out, crashed


def git(*args):
    try:
        r = subprocess.run(['git'] + list(args), cwd=_ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return r.stdout.decode('utf-8', 'replace').strip()
    except OSError:
        return ''


def anchor():
    """(sha, filename) of the newest approved report, or (None, None)."""
    if not os.path.isdir(APPROVED):
        return None, None
    files = sorted(glob.glob(os.path.join(APPROVED, '*.txt')))
    if not files:
        return None, None
    newest = files[-1]
    m = re.search(r'-([0-9a-f]{7,40})\.txt$', os.path.basename(newest))
    return (m.group(1) if m else None), os.path.basename(newest)


def untrailered_commits(sha):
    """py_impl commits since the anchor that carry no `Finding:` trailer.

    The contract's one new convention, and the only thing that catches a defect
    fixed without ever being written down. Inactive until an approved report
    exists, because "since" has no meaning before then.
    """
    if not sha:
        return None
    log = git('log', '--format=%H%x1f%s%x1f%b%x1e',
              '%s..HEAD' % sha, '--', 'com_functions/fn')
    if not log:
        return []
    out = []
    for rec in log.split('\x1e'):
        rec = rec.strip()
        if not rec:
            continue
        parts = rec.split('\x1f')
        if len(parts) < 3:
            continue
        h, subject, body = parts[0], parts[1], parts[2]
        if not re.search(r'(?m)^Finding:', body):
            out.append('%s %s' % (h[:9], subject[:64]))
    return out


def grab(text, pattern, cast=int):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--approve', action='store_true',
                    help='store this report as the anchor for every "since"')
    ap.add_argument('--json', help='also write the machine-readable rows here')
    args = ap.parse_args()

    scripts = audit_scripts()
    all_checks, texts, crashed = [], {}, []
    for p in scripts:
        checks, out, bad = run_gate(p)
        name = os.path.basename(p)
        texts[name] = out
        if bad:
            crashed.append(name)
        for kind, check_name, reason in checks:
            all_checks.append((name, kind, check_name, reason))

    fails = [c for c in all_checks if c[1] == 'FAIL']
    xpasses = [c for c in all_checks if c[1] == 'XPASS']
    xfails = [c for c in all_checks if c[1] == 'XFAIL']

    cov = texts.get('test_translation_coverage.py', '')
    mut = texts.get('test_mutation_score.py', '')
    opt = texts.get('test_option_coverage.py', '')
    inl = texts.get('test_inlined_copies.py', '')

    sha = git('rev-parse', 'HEAD')[:9]
    # The anchor directory is this tool's OUTPUT, not an input to it. Counting
    # it would mean --approve dirties the tree by writing the very file whose
    # promise is that the tree was clean, and the next run would then report
    # itself as unreproducible.
    dirty = bool([ln for ln in git('status', '--porcelain').splitlines()
                  if 'verification/approved/' not in ln.replace('\\', '/')])
    anchor_sha, anchor_file = anchor()
    untrailered = untrailered_commits(anchor_sha)

    L = []
    add = L.append
    add('SiCoPR verification report')
    add('=' * 66)
    add('tree            %s%s' % (sha, '  DIRTY, so this is not reproducible'
                                  if dirty else ''))
    add('anchor          %s' % (anchor_file or
                                'none yet: every "since" spans the whole history'))
    add('')

    add('GATES  (is the inventory complete?)')
    add('  %d audit scripts, %d checks' % (len(scripts), len(all_checks)))
    add('  %d failed, %d unexpected passes, %d accepted divergences'
        % (len(fails), len(xpasses), len(xfails)))
    if crashed:
        add('  %d SCRIPT(S) CRASHED without reporting a failure: %s'
            % (len(crashed), ', '.join(crashed)))
    add('')

    add('COVERAGE  (against the reference, not against a reading of it)')
    for label, pat in (
            ('functions translated      ',
             r'(\d+) functions in the reference, (?:\d+) translated'),
            ('checked against Octave    ',
             r'(\d+) checked against the executed reference')):
        v = grab(cov, pat)
        if v is not None:
            add('  %s %s' % (label, v))
    u = grab(opt, r'(\d+) uncovered')
    if u is not None:
        add('  option branches uncovered  %d' % u)
    add('')

    add('DISCRIMINATION  (are the tests awake?)')
    tot = grab(mut, r'(\d+) mutants, \d+ caught')
    got = grab(mut, r'\d+ mutants, (\d+) caught')
    eq = grab(mut, r'(\d+) equivalent-mutant entries')
    guarded = grab(mut, r'(\d+) survive with a repo-wide lint')
    real = grab(mut, r'(\d+) survive with NO other gate')
    if tot is not None:
        add('  %d mutants, %d caught' % (tot, got))
        add('  %d survivors a repo-wide lint still forbids' % (guarded or 0))
        add('  %d survivors NOTHING blocks          <-- the real gaps'
            % (real or 0))
        add('  %d equivalent mutants argued in equivalent_mutants.md' % (eq or 0))
    add('')

    opens = []
    for script, _kind, name, reason in fails:
        opens.append('FAIL   %-34s %s' % (name, reason[:100]))
    for script, _kind, name, reason in xpasses:
        opens.append('XPASS  %-34s an accepted divergence stopped being real'
                     % name)
    for name in crashed:
        opens.append('CRASH  %-34s exited non-zero reporting no failure' % name)
    if untrailered:
        for c in untrailered:
            opens.append('NOTE   py_impl commit with no Finding: trailer: %s' % c)

    if opens:
        add('OPENS  (%d)' % len(opens))
        for o in opens:
            add('  ' + o)
    else:
        add('OPENS  none')
    add('')

    add('NOT MEASURED BY THIS REPORT')
    add('  The contract also asks for an inventory row set, a per-row')
    add('  oracle_id, Octave exposure sets and waiver approval. None of those')
    add('  are built. "No opens" below means no open GATE, which is a weaker')
    add('  claim than the contract intends, and is stated here every run so')
    add('  that nobody has to remember the difference.')
    add('  The 208-case and 1368-case corpus runs are separate and are not')
    add('  part of this.')

    text = '\n'.join(L)
    print(text)

    if args.json:
        io.open(args.json, 'w', encoding='utf-8').write(json.dumps(
            {'sha': sha, 'dirty': dirty,
             'checks': [{'script': s, 'kind': k, 'name': n, 'reason': r}
                        for s, k, n, r in all_checks],
             'opens': opens}, indent=1, sort_keys=True))

    if args.approve:
        if dirty:
            raise SystemExit(
                '\nreport.py: refusing to approve a DIRTY tree. The anchor '
                'promises that checking out its sha and re-running reproduces '
                'the text, and uncommitted changes break that promise.')
        if opens:
            raise SystemExit(
                '\nreport.py: refusing to approve a report with %d open '
                'item(s). An anchor is a statement that the tree was clean at '
                'that sha.' % len(opens))
        if not os.path.isdir(APPROVED):
            os.makedirs(APPROVED)
        day = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
        name = '%s-%s.txt' % (day, sha)
        path = os.path.join(APPROVED, name)
        # Self-reference: the report says which anchor it was computed against,
        # and approving it makes it the anchor. Stored verbatim, the first
        # report would say "none yet" and every re-run at its own sha would
        # differ on that one line, so the tool would fail its own
        # reproducibility promise immediately. After approval this report IS
        # the anchor, so that is what the stored copy records.
        stored = re.sub(r'(?m)^anchor +.*$', 'anchor          %s' % name, text,
                        count=1)
        io.open(path, 'w', encoding='utf-8', newline='\n').write(stored + '\n')
        print('\napproved: %s' % os.path.relpath(path, _ROOT))

    return 1 if opens else 0


if __name__ == '__main__':
    sys.exit(main())
