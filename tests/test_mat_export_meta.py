"""The --export-mat `meta` struct must be readable by the R dashboard.

Found 2026-10-01 in the release audit: every export since 2026-09-05 failed in
R.matlab::readMat ("invalid substring arguments"). The one failing variable was
`meta`, whose `git_commit` was an empty string: com_mat_export asked git for the
commit in the WORKING directory, and runs launched outside the repository (the
GUI's --run-dir) got nothing back. R.matlab cannot read a zero-length char array,
so one empty field made the whole file unreadable. `meta.com_version` was also a
stale constant (4p14p0) rather than the release the run emulated.

Run: python tests/test_mat_export_meta.py
"""
# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from audit_check import check, finish  # noqa: E402
import com_mat_export as cme  # noqa: E402
from scipy.io import loadmat, savemat  # noqa: E402

# 1. the commit, asked from a directory that is not a repository
outside = tempfile.mkdtemp(prefix='sicopr_meta_')
cwd = os.getcwd()
os.chdir(outside)
try:
    sha = cme._git_commit()
finally:
    os.chdir(cwd)
check('git_commit_is_never_empty_outside_the_repository', bool(sha),
      'from %s _git_commit() returned %r' % (outside, sha))

# 2. the meta struct itself
meta_fn = getattr(cme, '_meta', None)
check('exporter_builds_meta_in_one_place', meta_fn is not None,
      'com_mat_export has no _meta(OP, param, case_i)')
if meta_fn is not None:
    m = meta_fn(SimpleNamespace(), SimpleNamespace(matlab_version='4p16p0'), 1)
    check('com_version_names_the_emulated_release', m['com_version'] == 'com_ieee8023_4p16p0',
          'com_version = %r' % m['com_version'])
    empty = [k for k, v in m.items() if isinstance(v, str) and not v]
    check('no_empty_text_field_in_meta', not empty, 'empty: %s' % empty)
    check('no_empty_list_in_meta', all(len(v) for v in m.values() if isinstance(v, list)),
          'an empty list is written as an empty cell array')

    # 3. round trip: scipy always; R.matlab when Rscript is installed
    d = {}
    cme._add(d, 'meta', m)
    p = os.path.join(outside, 'meta.mat')
    savemat(p, d, do_compression=True, oned_as='column')
    back = loadmat(p, squeeze_me=True)['meta']
    check('meta_round_trips_through_scipy', str(back['com_version']) == m['com_version'])

    rs = shutil.which('Rscript') or shutil.which('Rscript.exe')
    if not rs:
        hits = sorted(glob.glob(r'C:\Program Files\R\R-*\bin\Rscript.exe'))
        rs = hits[-1] if hits else None
    ok_r = None
    if rs:
        code = ('ok <- tryCatch({suppressWarnings(library(R.matlab)); readMat("%s"); "ok"},'
                ' error=function(e) conditionMessage(e)); cat(ok)' % p.replace('\\', '/'))
        r = subprocess.run([rs, '-e', code], capture_output=True, text=True, timeout=120)
        out = (r.stdout or '').strip()
        if 'there is no package' in (r.stderr or '') or 'R.matlab' in out and 'no package' in out:
            print('SKIP meta_reads_in_R: R.matlab not installed')
        else:
            ok_r = out.endswith('ok')
            check('meta_reads_in_R_matlab', ok_r, 'readMat said: %s' % out[-200:])
    else:
        print('SKIP meta_reads_in_R: Rscript not found')

finish()
