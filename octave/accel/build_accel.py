"""build_accel.py -- build the optional compiled kernels for the Octave release files.

    python octave/accel/build_accel.py              # -> octave/com_octave_accel.oct
    python octave/accel/build_accel.py --octave PATH

The release files (octave/com_ieee8023_<ver>_octave_compat.m) run on their own.
With com_octave_accel.oct beside them, three hot loops run compiled instead and
return exactly what the interpreted code returns; see com_octave_accel.cc for how
that is kept true, and tests/test_octave_compat.py for the check. Set the
environment variable COM_OCTAVE_ACCEL=0 to run without it.

The build uses mkoctfile, which ships with Octave (and on Windows brings its own
g++). One flag matters for correctness: -ffp-contract=off, so the compiler never
fuses a*b+c into one rounding where the interpreter rounds twice. The .oct is
built for one Octave version and platform and is not committed; rebuild after
upgrading Octave. The release files check the kernel's version string and ignore a
stale build.

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OCT_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(OCT_DIR), 'tools'))
from xlsx_to_com_mat import find_octave  # noqa: E402

CXXFLAGS = '-g -O2 -ffp-contract=off'


def fwd(p):
    return os.path.abspath(p).replace('\\', '/')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--octave', help='path to octave-cli (default: PATH, then the stock Windows install)')
    a = ap.parse_args(argv)
    octave = a.octave or find_octave()
    if not octave:
        sys.exit('octave-cli not found; give --octave')
    src = os.path.join(HERE, 'com_octave_accel.cc')
    out = os.path.join(OCT_DIR, 'com_octave_accel.oct')
    # mkoctfile run from inside Octave, so it is the one that matches the interpreter
    ev = ("setenv('CXXFLAGS', '%s'); [s, o] = system('mkoctfile -o \"%s\" \"%s\"'); disp(o); "
          "if s, exit(s), end; addpath('%s'); disp(com_octave_accel('version')); exit(0);"
          % (CXXFLAGS, fwd(out), fwd(src), fwd(OCT_DIR)))
    q = subprocess.run([octave, '--no-gui', '--no-window-system', '--eval', ev],
                       capture_output=True, text=True, cwd=HERE, errors='replace')
    print((q.stdout + q.stderr).strip())
    if q.returncode != 0 or not os.path.isfile(out):
        sys.exit('build failed')
    for leftover in ('com_octave_accel.o',):
        p = os.path.join(HERE, leftover)
        if os.path.isfile(p):
            os.remove(p)
    print('-> %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
