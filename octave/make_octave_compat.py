"""make_octave_compat.py -- derive the Octave-capable release files from matlab/.

    python octave/make_octave_compat.py            # regenerate both files
    python octave/make_octave_compat.py --check    # exit 1 if the committed files differ

The official COM release files cannot run under GNU Octave. Three things stop
them, each found the hard way in the 2026-09 three-way study: `verLessThan` is
called unguarded, the Touchstone reader depends on MATLAB `textscan` starting a
new record at every line break, and `ifft` returns complex where MATLAB returns
real for a conjugate-symmetric input, which poisons the MMSE solve inside a
local subfunction that no path shim can reach. A fourth item is speed: the
mainline `CDF_ev` does a `find` over an axis that grows every MLSE iteration,
which under Octave costs 30 to 60 times the run time of the `lookup` form.

Rich Mellitz's `Octave_compat` branch fixes all of this in its `src/` tree, but
its `release/` folder is byte-identical to mainline, so nobody downloading a
release gets the fixes. This script carries them into the release files instead,
as a small, named set of changes applied to `matlab/com_ieee8023_<ver>.m`:

  replaced functions (octave/patches/<name>.m, whole subfunction swapped)
    CDF_ev                  lookup() when available; the speed fix. CORRECTED
                            2026-09-22: lookup(x,v)+1 is NOT find(x>=v,1) --
                            lookup returns the LAST i with x(i)<=v, so the +1
                            answered one bin high on an exact grid hit and ran
                            off the end above the axis. It disagreed with find
                            on 32.35% of 27200 probes; the current form on none
    COM_CommandLine_Parse   OP.OCTAVE, detected or forced with 'Octave'
    read_Nport_touchstone   whole file tokenized and parsed with str2double;
                            Octave's textscan can stop part way through a
                            file, silently
    writecsv_transposed     fprintf, since Octave has no writecell
    H_interp                interp1(...,'pchip') given 'extrap'. MATLAB
                            extrapolates for pchip; Octave returns NA without
                            the flag, so out-of-band points came back NA and
                            propagated into H_new. No-op inside the range
    OptFom_Calc_Noise_XC    ifft(X,n,'symmetric') spelled out. Octave has no
                            such flag; the operation is to pad to n, keep the
                            first n/2+1 entries and mirror by conjugate
                            symmetry, which is written explicitly here. NOT the
                            same as real(ifft([X zeros])), which is the ifft of
                            the Hermitian PART and disagrees by a non-constant
                            ratio
  added functions
    csvread4com             a .csv config reader without xlsread
    com_octave_accel_on     whether the optional compiled kernels are in use
    erfcinv                 shadows Octave's, which is ~1.1e-9 relative out in
                            the tail where COM lives (specBER 1e-5 to 1e-9);
                            MATLAB's is ~1e-15. Starts from erfinv(1-y) and
                            refines with Newton on erfc, which is accurate in
                            both languages, so the result does not depend on
                            the starting accuracy. Bit-identical to
                            scipy.special.erfcinv on every value tested
  line substitutions
    main                    verLessThan guarded by the Octave test
    MMSE                    Rn = real(Rn) after the ifft
    MLSE_U1_c_178A          real() on the CDF_ev arguments
    read_ParamConfigFile    .csv configs read by csvread4com
    MMSE_FOM, FFE           speed only: an elementwise test for isequal, and
                            each FFE tap added in place instead of through a
                            shifted copy; same results
    get_pdf_from_sampled_signal  speed only: its loop's two helpers inlined
  replaced for speed
    FOM_rxffe_floating_taps MMSE_FOM's search-mode work inlined, invariants
                            hoisted; every candidate FOM bit-identical
  optional compiled kernels (octave/accel/, built with build_accel.py)
    the search's candidate loop, the ISI distribution loop and FFE's tap loop
    run compiled when com_octave_accel.oct is present; each returns exactly
    what the interpreted code returns, and COM_OCTAVE_ACCEL=0 turns them off

CDF_ev, COM_CommandLine_Parse and writecsv_transposed are the versions the
three-way study ran on 208 cases against the MATLAB reference to 5e-14 dB. The
reader is ours, because the branch's still uses textscan; so is the
floating-tap search, which is a speed rewrite of the release's own. Nothing here
changes a number under MATLAB: each edit is a no-op there, which is what makes
the result a reference and not a fork. The speed items are checked under Octave,
by running whole cases before and after and comparing every field of the result.

THE RULE FOR ADDING A PATCH HERE
-------------------------------
MATLAB is the reference. Octave is a proxy, and where the two disagree the
Octave side is what gets corrected, so that it keeps earning the right to be
used as an oracle for the Python port. A known divergence left in place does
not merely sit there: it silently caps the accuracy of every unit test written
against this file, and nothing downstream can be pinned tighter than the
divergence.

So when a difference is found:

  1. Determine what MATLAB does. Read the documentation for the exact
     semantics, not the shape of the call.
  2. Express that in Octave. "Octave has no such function or flag" is almost
     never the end of it. ifft(X,n,'symmetric') looked unreachable and is four
     lines of explicit mirroring; erfcinv looked like a reimplementation job
     and is a Newton step on a function both languages get right. Reach for a
     documented hole only after trying to spell the operation out.
  3. Where the shim can be checked against something independent, check it.
     erfcinv is bit-identical to scipy; the ifft construction was measured
     against the alternative reading and they differ by a non-constant ratio.
  4. Record the measurement here, with numbers, not an adjective.
  5. Take the divergence to the COM ad hoc. A portability defect in this port
     is ours; a defect in the reference is theirs, and the two need separating.

A patch that changes a number under MATLAB is not allowed. Each edit is either
a no-op there or a correction of something Octave does differently, which is
what makes the result a reference implementation rather than a fork.

The generated files are committed, like sicopr.py, so a reader needs no build
step. `--check` is the test that they were not edited by hand.

Copyright 2026 Todd Bermensolo
SPDX-License-Identifier: BSD-3-Clause
"""
import hashlib
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATCHES = os.path.join(HERE, 'patches')

# The 4p15p0 source is the adaptive-local-search build, not the bare release.
# It is the build the 208-case MATLAB reference results were produced with and
# the build SiCoPR emulates (VERSION.json, primary_reference); the bare 4p15p0
# release has only the legacy local search, so an Octave run of it would differ
# from both of the things this file exists to be compared against.
VERSIONS = {
    '4p15p0': ('matlab/com_ieee8023_4p15p0_adaptive_local_search.m',
               'octave/com_ieee8023_4p15p0_octave_compat.m'),
    '4p16p0': ('matlab/com_ieee8023_4p16p0.m',
               'octave/com_ieee8023_4p16p0_octave_compat.m'),
}

REPLACED = ['CDF_ev', 'COM_CommandLine_Parse', 'read_Nport_touchstone',
            'writecsv_transposed', 'FOM_rxffe_floating_taps', 'H_interp',
            'OptFom_Calc_Noise_XC']
ADDED = ['csvread4com', 'com_octave_accel_on', 'erfcinv']

# (label, old, new, expected count). Exact text; a miss is an error, never a
# silent skip, because a substitution that no longer matches means the release
# changed under the patch.
SUBSTITUTIONS = [
    ('main: guard verLessThan under Octave',
     "if verLessThan('matlab', '7.4.1')\n",
     "if ~exist('OCTAVE_VERSION', 'builtin') && verLessThan('matlab', '7.4.1') "
     "% OCTAVE: verLessThan resolves its first argument against packages there\n",
     1),
    ('MMSE: real part after ifft',
     "Rn=ifft(S_n)*fb;\n",
     "Rn=ifft(S_n)*fb;\n"
     "Rn=real(Rn); % OCTAVE: ifft returns complex for a conjugate-symmetric input; "
     "MATLAB returns real. No-op under MATLAB.\n",
     1),
    ('read_ParamConfigFile: .csv configs via csvread4com',
     "[na1, na2, parameter] = xlsread(paramFile);\n",
     "[parameter] = csvread4com(paramFile); % OCTAVE: no xlsread\n",
     2),
    # Speed, 2026-09-18. Each gives results identical to the line it replaces,
    # under Octave and MATLAB; measured in runs/report_docs/Octave_Speedup_Options_2026_09_18.md.
    # isequal costs 34 us a call under Octave against 2.9 us for the elementwise
    # test, and MMSE_FOM makes these two checks about 3 million times in one
    # Tx-FFE-swept case: 7 percent of the run. The operands are always same-size
    # numeric vectors (blim and wlim are clipped copies of b and w), for which
    # ~isequal and any(~=) agree, NaN included.
    ('MMSE_FOM: elementwise test instead of isequal (speed)',
     "if (Nb > 0) && ~isequal(b, blim)\n",
     "if (Nb > 0) && any(b ~= blim) % OCTAVE speed: isequal is 13x slower here; same result\n",
     1),
    ('MMSE_FOM: elementwise test instead of isequal (speed)',
     "if ~isequal(w, wlim)\n",
     "if any(w ~= wlim) % OCTAVE speed: isequal is 13x slower here; same result\n",
     1),
    # FFE makes one full-length circular shift per nonzero tap: 7 percent of a
    # Tx-FFE-swept case. circshift is an m-file under Octave, and the shifted
    # copy is not needed at all: a circular shift by s is two contiguous blocks,
    # so each tap is added in place, element k of the shifted pulse being
    # Vt(k-s) for k > s and Vt(n-s+k) for k <= s. Every element sees the same
    # x*C(i)+V0 as before, in the same tap order; the first add is still onto a
    # zero, so signed zeros come out the same, and all-zero taps still return
    # the scalar 0. 1.2 to 2.2x on this function (2026-09-18, second pass;
    # the first pass replaced circshift with an index expression, 1.1x).
    ('FFE: taps added in place as two blocks, no shifted copy (speed)',
     "V0=0;\n"
     "if iscolumn(V); V=V.';end\n"
     "for i=1:length(C)\n"
     "    if C(i)~=0\n"
     "        ishift=(i-1-cmx)*spui;\n"
     "        V0=circshift(V',[ishift,0])*C(i)+V0;\n"
     "    end\n"
     "end\n",
     "V0=0;\n"
     "if iscolumn(V); V=V.';end\n"
     "Vt=V'; n_V=numel(Vt); % OCTAVE speed: taps added in place, no shifted copy; same result\n"
     "if n_V > 0 && isreal(V) && isreal(C) && com_octave_accel_on()   % compiled, same values\n"
     "    V0=com_octave_accel('ffe', C, cmx, spui, V);\n"
     "else\n"
     "for i=1:length(C)\n"
     "    if C(i)~=0\n"
     "        ishift=(i-1-cmx)*spui;\n"
     "        s_V=mod(ishift,n_V);\n"
     "        if isscalar(V0); V0=zeros(n_V,1)+V0; end\n"
     "        V0(s_V+1:n_V)=Vt(1:n_V-s_V)*C(i)+V0(s_V+1:n_V);\n"
     "        V0(1:s_V)=Vt(n_V-s_V+1:n_V)*C(i)+V0(1:s_V);\n"
     "    end\n"
     "end\n"
     "end\n",
     1),
    # The ISI distribution build: two function calls and two struct copies per
    # ISI term, 1.7 million times in a Tx-FFE-swept case. Inlined, carrying only
    # the running y and Min; the axis is built once at the end by conv_fct's own
    # formula. conv_fct never reads the running axis, so the ones it built were
    # never used -- but every consumer of the RESULT gets a fresh one, which is
    # the point defect #1 of the three-way study turned on (a branch dropped
    # conv_fct's axis line and CDF_ev read a stale axis). 1.6 to 2.3x on this
    # function, output identical field for field on captured real inputs.
    ('get_pdf_from_sampled_signal: Init_PDF_Fast and conv_fct inlined (speed)',
     "empty_pdf=pdf;\n"
     "for k = 1:length(input_vector)\n"
     "    %     pdfn=d_cpdf(BinSize, abs(input_vector(k))*values, prob);\n"
     "    pdfn=Init_PDF_Fast(empty_pdf, abs(input_vector(k))*values, prob);\n"
     "    pdf=conv_fct(pdf, pdfn);\n"
     "end\n",
     "% OCTAVE speed: Init_PDF_Fast and conv_fct inlined; same result field for field.\n"
     "% Only y and Min are carried through the loop. The axis is built once, at the\n"
     "% end, exactly as conv_fct builds it: never drop that line (see defect #1).\n"
     "if length(input_vector) > 0 && com_octave_accel_on()          % compiled, same values\n"
     "    [pdf_y, pdf_Min] = com_octave_accel('pdf_build', input_vector, values, prob, BinSize, pdf.y, pdf.Min);\n"
     "else\n"
     "pdf_y = pdf.y;\n"
     "pdf_Min = pdf.Min;\n"
     "for k = 1:length(input_vector)\n"
     "    rv = round((abs(input_vector(k))*values)/BinSize);    % Init_PDF_Fast\n"
     "    q = zeros(1, numel(BinSize*rv(1):BinSize:BinSize*rv(end)));\n"
     "    bp = rv-rv(1)+1;\n"
     "    if all(diff(bp) > 0)                                  % distinct bins: no sums\n"
     "        q(bp) = prob;\n"
     "    else\n"
     "        q(bp(1)) = prob(1);\n"
     "        for m = 2:L\n"
     "            q(bp(m)) = q(bp(m))+prob(m);\n"
     "        end\n"
     "    end\n"
     "    pdf_Min = round(pdf_Min+rv(1));                       % conv_fct\n"
     "    pdf_y = conv2(pdf_y, q);\n"
     "end\n"
     "end\n"
     "if length(input_vector) > 0\n"
     "    pdf.Min = pdf_Min;\n"
     "    pdf.y = pdf_y;\n"
     "    pMax = pdf.Min+length(pdf.y)-1;\n"
     "    pdf.x = (pdf.Min*pdf.BinSize:pdf.BinSize:pMax*pdf.BinSize);\n"
     "end\n",
     1),
]

# Regex substitutions for the two MLSE lines, whose whitespace is not worth
# pinning: wrap the CDF_ev argument in real().
REGEX_SUBSTITUTIONS = [
    ('MLSE_U1_c_178A: real() on CDF_ev arguments',
     re.compile(r"CDF_ev\(\s*(A_s\s*\*\s*\(u_(?:j|trunc)[^,]*?\^\(1/2\))\s*,"),
     r"CDF_ev( real(\1),",
     2),
]


def sha256(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def function_spans(lines):
    """name -> (start, end) line indices of every top-level subfunction."""
    starts = [i for i, l in enumerate(lines) if re.match(r'^function\b', l)]
    spans = {}
    for k, i in enumerate(starts):
        j = starts[k + 1] if k + 1 < len(starts) else len(lines)
        hdr, q = lines[i], i
        while hdr.rstrip().endswith('...') and q + 1 < j:
            q += 1
            hdr = hdr.rstrip()[:-3] + lines[q]
        m = (re.search(r'=\s*([A-Za-z_]\w*)\s*\(', hdr)
             or re.match(r'^function\s+([A-Za-z_]\w*)', hdr))
        spans[m.group(1)] = (i, j)
    return spans


def read_patch(name, release_ends_functions):
    """A patch body, with its function-terminating `end` matched to the release.

    Octave requires that within one file either every function is closed with
    `end` or none is. The branch's stand-alone files are inconsistent about it,
    and a mismatch is a parse error reported at the last line of a 12,000-line
    file. So the terminator is normalised here rather than left to the patch.
    """
    with io.open(os.path.join(PATCHES, name + '.m'), encoding='utf-8') as fh:
        text = fh.read().replace('\r\n', '\n')
    if 'BSD-3-Clause' not in text:
        raise SystemExit('patch %s carries no licence header' % name)
    lines = text.rstrip('\n').split('\n')
    while lines and not lines[-1].strip():
        lines.pop()
    has_end = function_is_closed(lines)
    if release_ends_functions and not has_end:
        lines.append('end')
    elif not release_ends_functions and has_end:
        if lines[-1].strip() != 'end':
            raise SystemExit('patch %s: balance says the function is closed but '
                             'its last line is %r' % (name, lines[-1]))
        lines.pop()
    return lines


_BLOCK_OPEN = {'if', 'for', 'parfor', 'while', 'switch', 'try', 'unwind_protect', 'do'}
_TOKENS = re.compile(r"[\[\]\(\)\{\}]|\b(?:if|for|parfor|while|switch|try|"
                     r"unwind_protect|do|until|function|end\w*)\b")


def _code_only(line):
    """A line with strings and the trailing comment removed, so that `end`
    inside 'text' or after % is not counted."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if c in '%#':
            break
        if c == '"' or (c == "'" and (not out or not out[-1].strip()
                                      or out[-1].strip()[-1] not in "\\w)]}'\".")
                        and not re.search(r"[\w)\]}'\"]$", ''.join(out).rstrip())):
            q, i = c, i + 1
            while i < n:
                if line[i] == q and not (i + 1 < n and line[i + 1] == q):
                    break
                i += 2 if line[i] == q else 1
            out.append(' ')
            i += 1
            continue
        out.append(c)
        i += 1
    return ''.join(out)


def function_is_closed(lines):
    """Do the `end`s outnumber the blocks they close? Then one closes the
    function. `end` inside brackets is an index, not a terminator."""
    depth = opens = ends = 0
    for raw in lines:
        for m in _TOKENS.finditer(_code_only(raw)):
            t = m.group(0)
            if t in '([{':
                depth += 1
            elif t in ')]}':
                depth = max(0, depth - 1)
            elif depth == 0:
                if t in _BLOCK_OPEN:
                    opens += 1
                elif t == 'until':
                    ends += 1          # do ... until closes a do block
                elif t.startswith('end'):
                    ends += 1
    return ends > opens


def ends_functions(lines, spans):
    """Does the release close its functions with `end`? Judged by the last
    code line before each function header after the first."""
    votes = 0
    for name, (i, _j) in spans.items():
        if i == 0:
            continue
        k = i - 1
        while k > 0 and (not lines[k].strip() or lines[k].strip().startswith('%')):
            k -= 1
        votes += lines[k].strip() == 'end'
    return votes > (len(spans) - 1) / 2


def provenance(ver, src_rel, src_sha):
    return [
        '%% ---------------------------------------------------------------------',
        '%% OCTAVE-CAPABLE DERIVATIVE. Generated by octave/make_octave_compat.py',
        '%%%% from %s (sha256 %s).' % (src_rel, src_sha),
        '%% Do not edit by hand; edit the patches and regenerate.',
        '%%',
        '%% Changes from the release, each computing what the line it replaces',
        '%% computes, and each a no-op under MATLAB:',
        '%%   to run     CDF_ev (lookup, not find on a growing axis),',
        '%%             COM_CommandLine_Parse (OP.OCTAVE), read_Nport_touchstone',
        '%%             (the file read whole, no textscan), writecsv_transposed,',
        '%%             csvread4com (added); verLessThan guard; MMSE Rn=real(Rn);',
        '%%             MLSE real() on CDF_ev args; .csv config via csvread4com',
        '%%   for speed  FOM_rxffe_floating_taps (the search inlined, invariants',
        '%%             hoisted); get_pdf_from_sampled_signal (its two helpers',
        '%%             inlined); FFE (taps added in place); MMSE_FOM (elementwise',
        '%%             tests for isequal). Results are bit-identical: checked by',
        '%%             running whole cases before and after and comparing every',
        '%%             field of the result.',
        '%% Optional, and not required to run this file: if com_octave_accel.oct',
        '%% (octave/accel/, built by build_accel.py) is on the path, three hot loops',
        '%% run compiled and return the same bits, about 1.4 to 1.9 times faster.',
        '%% com_octave_accel_on (added) decides; COM_OCTAVE_ACCEL=0 turns it off.',
        '%% Configs: .mat (parameter cell array, see tools/xlsx_to_com_mat.py) or .csv.',
        '%%%% Version %s%s. Copyright 2025 802-COM Authors; changes Copyright 2026'
        % (ver, ' with adaptive local search' if 'adaptive' in src_rel else ''),
        '%% Todd Bermensolo. SPDX-License-Identifier: BSD-3-Clause',
        '%% ---------------------------------------------------------------------',
    ]


def build(ver):
    src_rel, dst_rel = VERSIONS[ver]
    src = os.path.join(ROOT, src_rel)
    # latin-1 maps every byte to one code point, so the release's stray cp1252
    # characters (a trademark sign in the header) survive the round trip
    with io.open(src, encoding='latin-1') as fh:
        text = fh.read()
    nl = '\r\n' if '\r\n' in text else '\n'
    text = text.replace('\r\n', '\n')

    for label, old, new, count in SUBSTITUTIONS:
        n = text.count(old)
        if n != count:
            raise SystemExit('%s: expected %d match(es) in %s, found %d'
                             % (label, count, src_rel, n))
        text = text.replace(old, new)
    for label, rx, repl, count in REGEX_SUBSTITUTIONS:
        text, n = rx.subn(repl, text)
        if n != count:
            raise SystemExit('%s: expected %d match(es) in %s, found %d'
                             % (label, count, src_rel, n))

    lines = text.split('\n')
    spans = function_spans(lines)
    closed = ends_functions(lines, spans)
    # replace from the bottom up so earlier spans stay valid
    for name in sorted(REPLACED, key=lambda n: -spans[n][0]):
        if name not in spans:
            raise SystemExit('%s: function %s not found' % (src_rel, name))
        i, j = spans[name]
        lines[i:j] = read_patch(name, closed) + ['']
    for name in ADDED:
        if name in spans:
            raise SystemExit('%s: %s already exists in the release' % (src_rel, name))
        lines += [''] + read_patch(name, closed) + ['']

    # provenance block right after the SPDX line of the file header
    k = next(i for i, l in enumerate(lines) if 'SPDX-License-Identifier' in l)
    lines[k + 1:k + 1] = provenance(ver, src_rel, sha256(src))
    return nl.join(lines), dst_rel


def main(argv):
    check = '--check' in argv
    bad = 0
    for ver in VERSIONS:
        out, dst_rel = build(ver)
        dst = os.path.join(ROOT, dst_rel)
        if check:
            have = io.open(dst, encoding='latin-1', newline='').read() if os.path.exists(dst) else None
            ok = have == out
            print('%-8s %s %s' % (ver, 'OK      ' if ok else 'DIFFERS ', dst_rel))
            bad += not ok
        else:
            with io.open(dst, 'w', encoding='latin-1', newline='') as fh:
                fh.write(out)
            print('%-8s wrote %s (%d lines)' % (ver, dst_rel, out.count('\n') + 1))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
