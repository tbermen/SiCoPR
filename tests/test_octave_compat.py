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
    endings" and the stray-`end` defects, neither of which any text check saw;
  * when the optional compiled kernels are built (octave/accel/), they return
    exactly what the interpreted code returns, byte for byte.

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
import importlib.util as _ilu

_HERE = os.path.dirname(os.path.abspath(__file__))
# The generator owns the list of shims it adds; read it from there so this
# test cannot disagree with the thing it is testing.
_gen_spec = _ilu.spec_from_file_location(
    '_mkoct', os.path.join(os.path.dirname(_HERE), 'octave',
                           'make_octave_compat.py'))
_gen = _ilu.module_from_spec(_gen_spec)
_gen_spec.loader.exec_module(_gen)
_ADDED = list(_gen.ADDED)
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, 'tools'))

from audit_check import check, finish            # noqa: E402

OCT = os.path.join(_ROOT, 'octave')
GEN = os.path.join(OCT, 'make_octave_compat.py')
sys.path.insert(0, OCT)
from make_octave_compat import VERSIONS as FILES   # noqa: E402  source -> output, per version
from make_octave_compat import NOT_REPLACED as _NOT_REPLACED   # noqa: E402

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
    # isequal and circshift markers are the two other speed substitutions. The
    # com_octave_accel calls are the three sites the optional kernels take over.
    markers = ["Rn=real(Rn)", "lookup(PDF.x", "OCTAVE_VERSION", "csvread4com(paramFile)",
               "raw = str2double(tokens)", "any(b ~= blim)", "any(w ~= wlim)",
               "V0(s_V+1:n_V)=Vt(1:n_V-s_V)*C(i)+V0(s_V+1:n_V)", "pdf_y = conv2(pdf_y, q)",
               "q(bp) = prob", "com_octave_accel('pdf_build'",
               "com_octave_accel('ffe'", "OCTAVE-CAPABLE DERIVATIVE"]
    # The floating-tap search patch and its kernel, except where a release does
    # not take it (4p17p0: the patch reproduces 4p16p0's HH arithmetic). There
    # the release's own search, which hands MMSE_FOM HH_unique_values, must run.
    if 'FOM_rxffe_floating_taps' in _NOT_REPLACED.get(ver, ()):
        markers += ["sigma_X2,new_idx,HH_unique_values);"]
        stale = [m for m in ("R = Hs'*Hs+RnnS(cols,cols)", "com_octave_accel('floating_fom'")
                 if m in dst]
        check("%s_compat_file_runs_the_release_floating_tap_search" % ver, not stale,
              "%s carries the pre-4p17p0 search patch: %s" % (dst_rel, stale))
    else:
        markers += ["R = Hs'*Hs+RnnS(cols,cols)", "fom_num/sigma_e",
                    "com_octave_accel('floating_fom'"]
    missing = [m for m in markers if m not in dst]
    check("%s_compat_file_carries_every_named_change" % ver, not missing,
          "%s lacks: %s" % (dst_rel, missing))
    # Derived from the generator's ADDED list, not hardcoded. This said "plus
    # two" and broke the moment a third shim was added (erfcinv, 2026-09-22),
    # which is a check measuring its own staleness rather than the file. The
    # property that matters is that the compat file adds EXACTLY the named
    # shims and loses none of the release's own functions.
    _n_dst = dst.count('\nfunction') + dst.startswith('function')
    _n_src = src.count('\nfunction') + src.startswith('function')
    check("%s_compat_file_adds_exactly_the_named_shims" % ver,
          _n_dst == _n_src + len(_ADDED),
          "expected the release's %d functions plus the %d added shims (%s); "
          "got %d. A mismatch means a shim was added without being named in "
          "ADDED, or a release function went missing."
          % (_n_src, len(_ADDED), ', '.join(_ADDED), _n_dst))

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

    # ------------------------------ the compiled kernels change nothing
    # octave/accel/com_octave_accel.cc, when built, runs three hot loops of the
    # release files compiled. It must return exactly what the interpreted code
    # returns. This pulls the GENERATED functions out of the release file, runs
    # them on seeded synthetic inputs in two Octave processes, kernels off
    # (COM_OCTAVE_ACCEL=0) and on, and compares every output byte for byte, so
    # signed zeros count. The inputs cover 4-level signalling (sparse adds in
    # the kernel), 3-level (the kernel calls convn), zero taps, negative shifts
    # and colliding distribution bins, and a distribution long enough that its
    # tail goes subnormal, where products stop being exact and conv2's own
    # rounding (fused in places) decides the bins. The search is compared on every
    # candidate's FOM, not only on the taps it picks: a last-bit change rarely
    # moves the maximum. The seed is 'state', not 'seed': 'seed' selects Octave's
    # old generator, whose values are all single precision, and the product of
    # two of those is exact, which hides a fused multiply-add completely.
    # ---------------------------- a kernel that cannot be loaded is not fatal
    # The kernels are optional and per machine, so the case to survive is a
    # com_octave_accel.oct left over from another Octave version or another
    # platform. Loading one is an error, not a false, so com_octave_accel_on
    # catches it and the release runs its own interpreted code. Without that
    # catch a run dies with "opening the library ... failed".
    stale = tempfile.mkdtemp(prefix='sicopr_staleoct_')
    io.open(os.path.join(stale, 'com_octave_accel.oct'), 'wb').write(b'not a loadable oct file')
    ev = ("addpath('%s'); addpath('%s'); printf('ON %%d\\n', com_octave_accel_on());"
          % (os.path.join(OCT, 'patches').replace('\\', '/'), stale.replace('\\', '/')))
    q = subprocess.run([octave, '--no-gui', '--no-window-system', '--eval', ev],
                       capture_output=True, text=True, timeout=300, errors='replace')
    check("an_unloadable_compiled_kernel_is_ignored_not_fatal",
          q.returncode == 0 and 'ON 0' in q.stdout,
          "expected the release to fall back to interpreted code; Octave said:\n%s"
          % (q.stdout + q.stderr).strip()[-400:])

    accel = os.path.join(OCT, 'com_octave_accel.oct')
    if not os.path.isfile(accel):
        print('\nnote: octave/com_octave_accel.oct not built; the kernel checks did not run '
              '(python octave/accel/build_accel.py).')
    else:
        import re
        import numpy as np
        import scipy.io
        rel = io.open(os.path.join(_ROOT, FILES['4p16p0'][1]), encoding='latin-1').read().split('\n')
        starts = [i for i, l in enumerate(rel) if re.match(r'^function\b', l)]
        wanted = ('FOM_rxffe_floating_taps', 'get_pdf_from_sampled_signal', 'FFE',
                  'com_octave_accel_on', 'd_cpdf', 'normal_dist')
        kdir = tempfile.mkdtemp(prefix='sicopr_accel_')
        for k, i in enumerate(starts):
            m = re.search(r'=\s*([A-Za-z_]\w*)\s*\(', rel[i]) or re.match(r'^function\s+([A-Za-z_]\w*)', rel[i])
            if m and m.group(1) in wanted:
                j = starts[k + 1] if k + 1 < len(starts) else len(rel)
                body = '\n'.join(rel[i:j]) + '\n'
                if m.group(1) == 'FOM_rxffe_floating_taps':
                    first = body.index('\n') + 1
                    hook = '    [~,best_FOM_idx]=max(best_FOM);\n'
                    assert body.count(hook) == 1, 'the search has changed; update this test'
                    body = (body[:first] + 'global accel_fom accel_by_kernel\n' + body[first:]).replace(
                        hook, '    accel_fom{end+1} = best_FOM; accel_by_kernel(end+1) = accel_done;\n' + hook)
                io.open(os.path.join(kdir, m.group(1) + '.m'), 'w', encoding='latin-1').write(body)
        driver = r"""
function accel_driver(out)
  rand('state', 11); randn('state', 11);   % 'seed' would give single-precision values
  % FFE: several pulses and tap sets, zeros and a negative shift included
  for t = 1:6
    V = randn(1, 800 + 700*t); V(1:7:end) = 0; V(3:11:end) = -0;
    C = randn(1, 2 + 4*t); C(2) = 0;
    ffe{t} = FFE(C, min(t, numel(C)-1), 8*t, V);
  end
  ffe{7} = FFE(zeros(1, 5), 2, 32, randn(1, 100));
  % the ISI distribution: 4 and 3 levels, values tiny enough to collide bins
  for t = 1:8
    L = 4 - (t > 6);
    v = randn(200 + 50*t, 1) .* 10 .^ (-6 + 4*rand(200 + 50*t, 1));
    p = get_pdf_from_sampled_signal(v, L, 1e-5 * (1 + t/10));
    pdf_y{t} = p.y; pdf_min(t) = p.Min; pdf_x{t} = p.x;
  end
  % a long ISI vector: the far tail sinks below realmin, where products stop
  % being exact and conv2's own rounding, fused in places, decides the bins
  v = (2 + abs(randn(700, 1))) * 1e-5 .* sign(randn(700, 1));
  p = get_pdf_from_sampled_signal(v, 4, 1e-5);
  pdf_y{9} = p.y; pdf_min(9) = p.Min; pdf_x{9} = p.x;
  subnormal_bins = sum(p.y > 0 & p.y < realmin);
  % the floating-tap search on a synthetic but well-posed system
  global accel_fom accel_by_kernel
  accel_fom = {}; accel_by_kernel = [];
  param = struct('RxFFE_cmx', 6, 'RxFFE_cpx', 8, 'N_bmax', 80, 'N_bf', 4, 'N_bg', 2, ...
                 'R_LM', 0.95, 'levels', 4);
  for t = 1:2
    H = randn(1800, 87) * 0.02; H(706, :) = H(706, :) + 1;   % the cursor row, d+1
    h = randn(1, 400) * 0.05;
    Rnn = eye(87) * 1e-4;
    idx{t} = FOM_rxffe_floating_taps(param, h, H, t, Rnn, 6, 705, ones(87,1)*0.8, -ones(87,1)*0.8, ...
                                     -0.5, 0.5, 15/27, 1, 200);
  end
  used = com_octave_accel_on();
  save('-v7', out, 'ffe', 'pdf_y', 'pdf_min', 'pdf_x', 'idx', 'accel_fom', 'accel_by_kernel', 'subnormal_bins', 'used');
end
"""
        io.open(os.path.join(kdir, 'accel_driver.m'), 'w', encoding='latin-1').write(driver)
        res = {}
        for mode in ('0', '1'):
            out = os.path.join(kdir, 'out_%s.mat' % mode)
            ev = ("addpath('%s'); addpath('%s'); accel_driver('%s');"
                  % (kdir.replace('\\', '/'), OCT.replace('\\', '/'), out.replace('\\', '/')))
            q = subprocess.run([octave, '--no-gui', '--no-window-system', '--eval', ev],
                               capture_output=True, text=True, timeout=900, errors='replace',
                               env=dict(os.environ, COM_OCTAVE_ACCEL=mode))
            res[mode] = scipy.io.loadmat(out) if os.path.isfile(out) else None
            if res[mode] is None:
                print((q.stdout + q.stderr)[-800:])

        def flat(d):
            vals = []
            for k in ('ffe', 'pdf_y', 'pdf_min', 'pdf_x', 'idx', 'accel_fom'):
                v = d[k]
                if v.dtype == object:
                    vals += [np.asarray(x, dtype=float) for x in v.ravel()]
                else:
                    vals.append(np.asarray(v, dtype=float))
            return vals

        ok = res['0'] is not None and res['1'] is not None
        if ok:
            a, b = flat(res['0']), flat(res['1'])
            ok = (len(a) == len(b) and all(x.shape == y.shape and x.tobytes() == y.tobytes()
                                           for x, y in zip(a, b)))
        if ok:
            by_kernel = [np.asarray(res[m]['accel_by_kernel']).ravel() for m in ('0', '1')]
            ok = by_kernel[1].size > 0 and by_kernel[1].all() and not by_kernel[0].any()
            # the long-vector case must still reach the subnormal range, or it tests nothing
            ok = ok and int(np.asarray(res['0']['subnormal_bins']).ravel()[0]) > 0
        check("compiled_kernels_return_what_the_interpreted_code_returns",
              ok and bool(res['1']['used']) and not bool(res['0']['used']),
              "kernels off/on disagree, or were not used when on (used: off=%s on=%s); see %s"
              % (res['0'] and bool(res['0']['used']), res['1'] and bool(res['1']['used']), kdir))

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
