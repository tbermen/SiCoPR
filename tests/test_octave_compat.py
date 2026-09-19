"""The Octave-capable release files: generated, licensed, and parseable.

octave/com_ieee8023_<ver>_octave_compat.m used to be byte-identical copies of the
MATLAB releases -- files whose name promised Octave compatibility and whose
content had none. They are now derived from matlab/ by
octave/make_octave_compat.py with a small named patch set, and committed so a
reader needs no build step. These checks keep that arrangement honest:

  * the committed files are exactly what the generator produces (no hand edits);
  * every patch body carries the upstream licence header;
  * each compat file differs from its matlab/ source in the places the patch
    set names, so the name is no longer a lie;
  * when octave-cli is installed, Octave parses both files. Parsing a
    12,000-line function file is what caught the "inconsistent function
    endings" and the stray-`end` defects, neither of which any text check saw.

An end-to-end Octave-versus-SiCoPR run takes minutes and needs a channel that
is not in the repository, so it is opt-in: set COM_OCTAVE_CASE to
"<config.xlsx>;<thru.s4p>" and the last check runs both engines and requires
COM_dB to agree within 1e-9 dB.

Run: python tests/test_octave_compat.py
"""
# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

import glob
import hashlib
import io
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, 'tools'))

from audit_check import check, finish            # noqa: E402

OCT = os.path.join(_ROOT, 'octave')
GEN = os.path.join(OCT, 'make_octave_compat.py')
sys.path.insert(0, OCT)
from make_octave_compat import VERSIONS as FILES   # noqa: E402  source -> output, per version

# --------------------------------------------------- generated, not edited
p = subprocess.run([sys.executable, GEN, '--check'], capture_output=True, text=True, cwd=_ROOT)
check("octave_compat_files_are_exactly_what_the_generator_produces",
      p.returncode == 0,
      "make_octave_compat.py --check disagrees with the committed files; edit "
      "the patches and regenerate rather than the output:\n%s" % (p.stdout + p.stderr).strip())

# ------------------------------------------------------------- licences
patches = sorted(glob.glob(os.path.join(OCT, 'patches', '*.m')))
check("every_patch_body_carries_the_upstream_licence",
      patches and all('BSD-3-Clause' in io.open(f, encoding='utf-8').read() for f in patches),
      "patch files without an SPDX BSD-3-Clause header: %s"
      % [os.path.basename(f) for f in patches
         if 'BSD-3-Clause' not in io.open(f, encoding='utf-8').read()])

# --------------------------------------------------- the name is now true
for ver, (src_rel, dst_rel) in FILES.items():
    src = io.open(os.path.join(_ROOT, src_rel), encoding='latin-1').read()
    dst = io.open(os.path.join(_ROOT, dst_rel), encoding='latin-1').read()
    check("%s_compat_file_differs_from_the_matlab_release" % ver,
          hashlib.sha256(src.encode('latin-1')).hexdigest()
          != hashlib.sha256(dst.encode('latin-1')).hexdigest(),
          "%s is byte-identical to %s: the file promises Octave compatibility and "
          "carries none" % (dst_rel, src_rel))
    # The reader replaced the blank-line NaN shim on 2026-09-16: Octave's
    # textscan can stop part way through a touchstone file, silently, from a
    # file handle or a string, and parsing the file's text whole subsumes both
    # problems. str2double replaced sscanf there on 2026-09-18, for speed; the
    # isequal and circshift markers are the two other speed substitutions.
    markers = ["Rn=real(Rn)", "lookup(PDF.x", "OCTAVE_VERSION", "csvread4com(paramFile)",
               "raw = str2double(tokens)", "any(b ~= blim)", "any(w ~= wlim)",
               "Vt([n_V-s_V+1:n_V, 1:n_V-s_V])", "pdf_y = conv2(pdf_y, q)",
               "R = Hs'*Hs+RnnS(cols,cols)", "OCTAVE-CAPABLE DERIVATIVE"]
    missing = [m for m in markers if m not in dst]
    check("%s_compat_file_carries_every_named_change" % ver, not missing,
          "%s lacks: %s" % (dst_rel, missing))
    check("%s_compat_file_keeps_the_release_function_count_plus_one" % ver,
          dst.count('\nfunction') + dst.startswith('function')
          == src.count('\nfunction') + src.startswith('function') + 1,
          "expected the release's functions plus csvread4com; got %d vs %d"
          % (dst.count('\nfunction') + dst.startswith('function'),
             src.count('\nfunction') + src.startswith('function')))

# -------------------------------------------------------- octave parses
from xlsx_to_com_mat import find_octave       # noqa: E402
octave = find_octave()
if not octave:
    print('\nSKIP: octave-cli not found; the parse and end-to-end checks did not run.')
else:
    for ver, (_src, dst_rel) in FILES.items():
        entry = os.path.splitext(os.path.basename(dst_rel))[0]
        # nargin() forces a parse of the function file without running it
        ev = "addpath('%s'); printf('NARGIN %%d\\n', nargin('%s'));" % (
            OCT.replace('\\', '/'), entry)
        q = subprocess.run([octave, '--no-gui', '--no-window-system', '--eval', ev],
                           capture_output=True, text=True, timeout=300, errors='replace')
        ok = q.returncode == 0 and 'NARGIN -1' in q.stdout and 'error:' not in (q.stdout + q.stderr)
        check("octave_parses_the_%s_compat_file" % ver, ok,
              "Octave %s on %s:\n%s" % ('failed' if q.returncode else 'reported',
                                       dst_rel, (q.stdout + q.stderr).strip()[-600:]))

    # ------------------------------------------ the reader reads the whole file
    # Octave's textscan can stop part way through a touchstone file, silently,
    # whether it reads a file handle or the text as a string: on the 2026-09-15
    # corpus it returned 2180 of 8001 points for one THRU and 928 of 8001 for
    # the FEXT beside it. A short crosstalk file trips the caller's point-count
    # check; a short THRU passes every check and puts COM tens of dB out (9
    # cases came back at -12 to -23.5 dB). The reader does not use textscan.
    # This builds a file large enough to provoke that failure, in the layout the
    # affected files use: blocks of four lines separated by a blank line.
    npts = 8001
    tmp = tempfile.mkdtemp(prefix='sicopr_s4p_')
    s4p = os.path.join(tmp, 'synthetic.s4p')
    with io.open(s4p, 'w', encoding='ascii', newline='') as fh:
        fh.write('!synthetic, %d points\r\n# HZ S MA R 50.000000 \r\n' % npts)
        for k in range(npts):
            pair = '\t'.join('%.9g\t%.9g' % (0.5 - 1e-6 * k, (k % 360) - 180.0)
                             for _ in range(4))
            fh.write('%d.\t%s\t\r\n' % (k * 10000000, pair))
            for _ in range(3):
                fh.write('\t%s\t\r\n' % pair)
            fh.write('\t\r\n')
    ev = ("addpath('%s'); [sch, fax] = read_Nport_touchstone('%s', [1 2 3 4], 50); "
          "printf('POINTS %%d\\n', numel(fax));"
          % (os.path.join(OCT, 'patches').replace('\\', '/'), s4p.replace('\\', '/')))
    q = subprocess.run([octave, '--no-gui', '--no-window-system', '--eval', ev],
                       capture_output=True, text=True, timeout=600, errors='replace')
    check("octave_reader_reads_every_point_of_a_large_touchstone_file",
          ('POINTS %d' % npts) in q.stdout,
          "expected %d points from %s; Octave said:\n%s"
          % (npts, s4p, (q.stdout + q.stderr).strip()[-600:]))

    case = os.environ.get('COM_OCTAVE_CASE', '')
    if ';' in case:
        cfg, thru = case.split(';', 1)
        from octave_compare import run_case                 # noqa: E402
        out = tempfile.mkdtemp(prefix='sicopr_octave_')
        row = run_case({'id': 'case', 'config': cfg, 'thru': thru}, '4p15p0', out, octave)
        check("octave_and_sicopr_agree_on_COM_for_the_given_case",
              'd_COM_dB' in row and abs(row['d_COM_dB']) < 1e-9,
              "row was %r" % {k: v for k, v in row.items() if 'COM' in k or 'error' in k})
        print('\nend-to-end: Octave %.1f s, SiCoPR %.1f s, evidence under %s'
              % (row.get('octave_wall_s', -1), row.get('sicopr_wall_s', -1), out))
    else:
        print('\nnote: set COM_OCTAVE_CASE="<config.xlsx>;<thru.s4p>" to run both '
              'engines on a case (minutes).')

finish()
