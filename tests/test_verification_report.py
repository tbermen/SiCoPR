"""The verification report's own moving parts.

report.py is what answers "are there any opens?", so a defect in it is a
defect in the answer. Two of its functions decide what the rest of the report
means and neither had a test:

  anchor() picks WHICH approved report the run is measured against, and
  untrailered_commits() walks the commits since that anchor.

anchor() sorted filenames. The date prefix settles it while no two approvals
share a day; 2026-09-23 has four, `e82669f2d` sorts after `8a69c29c4`, and the
anchor walked backwards to the first of the four -- so "since the anchor"
re-opened work that had already been approved and the reproducibility
self-check read the wrong stored copy.

Run: python tests/test_verification_report.py
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause

import io
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from audit_check import check, finish                       # noqa: E402
from com_functions.verification import report                # noqa: E402


# The order that matters, and the order a filename sort gives, are opposite:
# committed newest last, but `e...` sorts after `8...`.
_FAKE = [
    ('2026-09-23-223c48fec.txt', '223c48fec', 1000),
    ('2026-09-23-e82669f2d.txt', 'e82669f2d', 2000),
    ('2026-09-23-600b427d2.txt', '600b427d2', 3000),
    ('2026-09-23-8a69c29c4.txt', '8a69c29c4', 4000),   # newest by commit time
]


def _with_fake_approvals(fn):
    """Run fn() with APPROVED pointed at a temp dir and git() stubbed."""
    d = tempfile.mkdtemp(prefix='anchor_')
    real_approved, real_git = report.APPROVED, report.git
    try:
        for name, _sha, _ts in _FAKE:
            io.open(os.path.join(d, name), 'w', encoding='utf-8').write('x\n')
        stamps = {sha: ts for _n, sha, ts in _FAKE}

        def fake_git(*args):
            if args[:3] == ('show', '-s', '--format=%ct'):
                return str(stamps.get(args[3], ''))
            return real_git(*args)

        report.APPROVED, report.git = d, fake_git
        return fn()
    finally:
        report.APPROVED, report.git = real_approved, real_git
        shutil.rmtree(d, ignore_errors=True)


_sha, _file = _with_fake_approvals(report.anchor)

check('anchor_is_the_newest_by_commit_time_not_by_filename',
      _file == '2026-09-23-8a69c29c4.txt',
      'four approvals share the date 2026-09-23 and the newest by commit time '
      'is 8a69c29c4, but anchor() returned %r. A filename sort gives '
      'e82669f2d, which is the second-oldest of the four.' % _file)

check('anchor_returns_the_sha_from_the_name_it_chose',
      _sha == '8a69c29c4',
      'anchor() returned sha %r for file %r; the two have to agree or the '
      '"since the anchor" window is measured from a different commit than the '
      'report names.' % (_sha, _file))


def _unresolvable():
    """A sha git cannot resolve must not make the function return nothing."""
    d = tempfile.mkdtemp(prefix='anchor_bad_')
    real_approved, real_git = report.APPROVED, report.git
    try:
        io.open(os.path.join(d, '2026-01-01-deadbeef.txt'), 'w',
                encoding='utf-8').write('x\n')
        report.APPROVED = d
        report.git = lambda *a: ''          # git resolves nothing
        return report.anchor()
    finally:
        report.APPROVED, report.git = real_approved, real_git
        shutil.rmtree(d, ignore_errors=True)


_bad_sha, _bad_file = _unresolvable()

check('an_unresolvable_sha_falls_back_to_the_name',
      _bad_file == '2026-01-01-deadbeef.txt' and _bad_sha == 'deadbeef',
      'a file whose sha git cannot resolve came back as %r/%r. The fallback '
      'exists so this never returns nothing where the old sort returned '
      'something.' % (_bad_sha, _bad_file))


def _empty():
    d = tempfile.mkdtemp(prefix='anchor_empty_')
    real = report.APPROVED
    try:
        report.APPROVED = d
        return report.anchor()
    finally:
        report.APPROVED = real
        shutil.rmtree(d, ignore_errors=True)


check('no_approvals_means_no_anchor',
      _empty() == (None, None),
      'an empty approved directory must give (None, None); untrailered_commits '
      'treats that as "no window yet" rather than "everything".')

check('missing_approved_directory_means_no_anchor',
      (lambda: (setattr(report, 'APPROVED', os.path.join(_ROOT, 'nope_')),
                report.anchor())[1])() == (None, None),
      'a missing approved directory must give (None, None)')
report.APPROVED = os.path.join(
    _ROOT, 'com_functions', 'verification', 'approved')

# The live directory, not a fixture: the report this repo actually prints has
# to name the approval that was committed last.
_live_sha, _live_file = report.anchor()
_head_ok = True
_why = ''
if _live_file:
    _ts = report.git('show', '-s', '--format=%ct', _live_sha)
    _head_ok = bool(_ts)
    _why = ('the live anchor %s names sha %s, which this clone cannot resolve'
            % (_live_file, _live_sha))

check('the_live_anchor_names_a_sha_this_clone_has',
      _head_ok,
      _why or 'no approvals yet, so nothing to resolve')

finish()
