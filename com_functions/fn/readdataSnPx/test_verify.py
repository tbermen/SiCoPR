"""Tests for readdataSnPx (MATLAB lines ~10768).

MATLAB GROUND TRUTH:
  2-port Touchstone RI file → result.cs[1,0,:] swapped from file order.
  Frequency scaling: GHz in file → Hz in output.
  MA format: mag*exp(j*angle_deg*pi/180).
  DB format: 10^(db/20) * exp(j*angle_deg*pi/180).
"""
import os
import tempfile
import numpy as np
import pytest
from com_functions.fn.readdataSnPx.py_impl import readdataSnPx


def _write_s2p_ri(freq_ghz, s_data):
    """Write minimal RI-format s2p file. s_data shape: (nport, nport, nfreq)."""
    nport = s_data.shape[0]
    nfreq = len(freq_ghz)
    lines = ['! test\n', f'# GHz S RI R 50\n']
    for k in range(nfreq):
        row = [str(freq_ghz[k])]
        for i in range(nport):
            for j in range(nport):
                row.append(f'{s_data[i,j,k].real:.6f}')
                row.append(f'{s_data[i,j,k].imag:.6f}')
        lines.append(' '.join(row) + '\n')
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.s2p', delete=False)
    tmp.writelines(lines)
    tmp.close()
    return tmp.name


def test_freq_scaling_ghz():
    freq_ghz = np.array([1.0, 2.0, 3.0])
    s = np.zeros((2, 2, 3), dtype=complex)
    f = _write_s2p_ri(freq_ghz, s)
    try:
        r = readdataSnPx(f, 2)
        np.testing.assert_allclose(r.freq, freq_ghz * 1e9)
    finally:
        os.unlink(f)


def test_2port_swap():
    freq_ghz = np.array([1.0])
    s = np.zeros((2, 2, 1), dtype=complex)
    s[0, 1, 0] = 0.3 + 0.1j  # S12
    s[1, 0, 0] = 0.5 + 0.2j  # S21 — will be swapped with S12
    f = _write_s2p_ri(freq_ghz, s)
    try:
        r = readdataSnPx(f, 2)
        # After swap: result.cs[0,1] should be what was originally S21=s[1,0]
        assert abs(r.cs[0, 1, 0] - (0.5 + 0.2j)) < 1e-5
        assert abs(r.cs[1, 0, 0] - (0.3 + 0.1j)) < 1e-5
    finally:
        os.unlink(f)


def test_output_shape():
    freq_ghz = np.linspace(1, 30, 10)
    nport = 4
    s = np.random.default_rng(0).standard_normal((nport, nport, 10)) + 1j * np.random.default_rng(1).standard_normal((nport, nport, 10))
    lines = ['! test\n', '# GHz S RI R 50\n']
    for k in range(10):
        row = [str(freq_ghz[k])]
        for i in range(nport):
            for j in range(nport):
                row.append(f'{s[i,j,k].real:.6f}')
                row.append(f'{s[i,j,k].imag:.6f}')
        lines.append(' '.join(row) + '\n')
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.s4p', delete=False)
    tmp.writelines(lines)
    tmp.close()
    try:
        r = readdataSnPx(tmp.name, nport)
        assert r.cs.shape == (nport, nport, 10)
    finally:
        os.unlink(tmp.name)


def test_ri_values():
    freq_ghz = np.array([5.0])
    s = np.zeros((2, 2, 1), dtype=complex)
    s[0, 0, 0] = 0.1 + 0.2j
    f = _write_s2p_ri(freq_ghz, s)
    try:
        r = readdataSnPx(f, 2)
        assert abs(r.cs[0, 0, 0] - (0.1 + 0.2j)) < 1e-5
    finally:
        os.unlink(f)


def _write_s2p_ma(freq_ghz, mags, angles_deg):
    lines = ['! test\n', '# GHz S MA R 50\n']
    nport = 2
    for k in range(len(freq_ghz)):
        row = [str(freq_ghz[k])]
        for i in range(nport):
            for j in range(nport):
                row.append(f'{mags[i,j,k]:.6f}')
                row.append(f'{angles_deg[i,j,k]:.6f}')
        lines.append(' '.join(row) + '\n')
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.s2p', delete=False)
    tmp.writelines(lines)
    tmp.close()
    return tmp.name


def test_ma_format():
    freq_ghz = np.array([2.0])
    mags = np.ones((2, 2, 1)) * 0.5
    angles = np.zeros((2, 2, 1))
    angles[0, 0, 0] = 90.0
    f = _write_s2p_ma(freq_ghz, mags, angles)
    try:
        r = readdataSnPx(f, 2)
        expected = 0.5 * np.exp(1j * np.pi / 2)
        assert abs(r.cs[0, 0, 0] - expected) < 1e-5
    finally:
        os.unlink(f)


# ============================================================
# COM Octave 4p16p0 oracle pins.
# readdataSnPx extracted verbatim from
# octave/com_ieee8023_4p16p0_octave_compat.m (byte-identical to
# matlab/com_ieee8023_4p16p0.m) and run under Octave on the very files these
# tests write, so the reference read the same bytes.
#
# Shape note, not a divergence: Octave drops a trailing singleton dimension,
# so a single-frequency cs comes back 2x2 there and (2,2,1) here.
# ============================================================
_BODY2 = ('1.0 0.100000 0.200000 0.300000 0.400000 '
          '0.500000 0.600000 0.700000 0.800000\n'
          '2.0 0.110000 0.210000 0.310000 0.410000 '
          '0.510000 0.610000 0.710000 0.810000\n')

# cs for _BODY2 read as a 2-port, AFTER the S12/S21 swap, flattened in
# numpy C order: [ (0,0,f0) (0,0,f1) (0,1,f0) (0,1,f1) (1,0,..) (1,1,..) ]
_OCT_CS2 = [0.1 + 0.2j, 0.11 + 0.21j,
            0.5 + 0.6j, 0.51 + 0.61j,
            0.3 + 0.4j, 0.31 + 0.41j,
            0.7 + 0.8j, 0.71 + 0.81j]


def _tmp(text, suffix='.s2p'):
    t = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False,
                                    newline='\n')
    t.write(text)
    t.close()
    return t.name


def _read(text, nport=2, suffix='.s2p'):
    f = _tmp(text, suffix)
    try:
        return readdataSnPx(f, nport)
    finally:
        os.unlink(f)


def _raises(text, nport=2, suffix='.s2p'):
    f = _tmp(text, suffix)
    try:
        with pytest.raises(ValueError):
            readdataSnPx(f, nport)
    finally:
        os.unlink(f)


def test_octave_canonical_2port_values_and_swap():
    """COM Octave: RI values with S12 and S21 exchanged per the Touchstone
    spec, and GHz scaled to Hz."""
    r = _read('! test\n# GHz S RI R 50\n' + _BODY2)
    np.testing.assert_allclose(r.freq, [1e9, 2e9], rtol=1e-15)
    np.testing.assert_allclose(r.cs.ravel(), _OCT_CS2, rtol=1e-14)


def test_octave_frequency_units():
    """COM Octave: only hz, khz, mhz and ghz scale."""
    for header, expect in (('# Hz S RI R 50', [1.0, 2.0]),
                           ('# kHz S RI R 50', [1e3, 2e3]),
                           ('# MHz S RI R 50', [1e6, 2e6]),
                           ('# GHz S RI R 50', [1e9, 2e9]),
                           ('# ghz S RI R 50', [1e9, 2e9])):
        r = _read(header + '\n' + _BODY2)
        np.testing.assert_allclose(r.freq, expect, rtol=1e-15), header


def test_octave_unknown_unit_leaves_the_frequencies_alone():
    """COM Octave: `switch lower(units)` has no otherwise, so '# S RI R 50'
    and '# THz S RI R 50' both give freq = [1 2].  Defaulting to GHz scaled
    those files by a billion."""
    for header in ('# S RI R 50', '# THz S RI R 50'):
        r = _read(header + '\n' + _BODY2)
        np.testing.assert_allclose(r.freq, [1.0, 2.0], rtol=1e-15), header


def test_octave_ma_and_db_formats():
    """COM Octave: MA is mag*exp(1i*ang*pi/180), DB is 10^(db/20) at the same
    angle."""
    r = _read('# GHz S MA R 50\n1.0 0.5 90 0.6 45 0.7 -30 0.8 180\n')
    np.testing.assert_allclose(
        r.cs.ravel(),
        [3.061616997868383e-17 + 0.5j,
         0.60621778264910708 - 0.34999999999999992j,
         0.42426406871192851 + 0.42426406871192851j,
         -0.80000000000000004 + 9.7971743931788262e-17j], rtol=1e-13,
        atol=1e-16)
    r = _read('# GHz S DB R 50\n1.0 -6 90 -3 45 -1 -30 0 180\n')
    np.testing.assert_allclose(
        r.cs.ravel(),
        [3.0688867071757785e-17 + 0.50118723362727224j,
         0.77184595357053676 - 0.44562546906687273j,
         0.50059326485045341 + 0.50059326485045341j,
         -1 + 1.2246467991473532e-16j], rtol=1e-13, atol=1e-16)


def test_octave_option_line_S_is_case_sensitive():
    """COM Octave: `p = find(A=='S')` matches only an upper-case S, so a fully
    lower-case option line leaves the format empty and the reference stops
    with 'readdataSnP: Unknown data format'.  Upper-casing the tokens read the
    file happily."""
    _raises('# ghz s ri r 50\n' + _BODY2)
    # an upper-case S with lower-case units is fine
    r = _read('# ghz S RI R 50\n' + _BODY2)
    np.testing.assert_allclose(r.freq, [1e9, 2e9], rtol=1e-15)


def test_octave_option_line_without_a_format_is_refused():
    """COM Octave: format = A(p+1:p+2), so '# GHz S' errors with
    'A(7): out of bound 5'.  Defaulting to MA answered instead."""
    _raises('# GHz S\n' + _BODY2)


def test_octave_unknown_format_is_refused():
    """COM Octave: 'readdataSnP: Unknown data format'."""
    _raises('# GHz S XY R 50\n' + _BODY2)


def test_octave_truncated_record_is_refused():
    """COM Octave: the missing pair makes cs(ni,nj,nk) = [] and the read stops
    with '=: nonconformant arguments'.  Breaking out of the loop left zeros in
    the unread entries and reported them as data."""
    _raises('# GHz S RI R 50\n'
            '1.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8\n'
            '2.0 0.11 0.21\n')


def test_octave_file_with_no_data_is_refused():
    """COM Octave: freq and cs are never assigned, so result.cs = cs errors
    with 'cs undefined'.  Empty arrays were returned instead."""
    _raises('! only a header\n# GHz S RI R 50\n')


def test_octave_stray_token_discards_the_rest_of_its_line():
    """COM Octave: between records fscanf('%f') fails, fscanf('%s') eats the
    token and fgetl() throws away the rest of THAT LINE, so the 2 GHz record
    behind 'JUNK' is lost entirely.  A flat token stream kept it."""
    r = _read('# GHz S RI R 50\n'
              '1.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8\n'
              'JUNK 2.0 0.11 0.21 0.31 0.41 0.51 0.61 0.71 0.81\n'
              '3.0 0.12 0.22 0.32 0.42 0.52 0.62 0.72 0.82\n')
    np.testing.assert_allclose(r.freq, [1e9, 3e9], rtol=1e-15)
    np.testing.assert_allclose(
        r.cs.ravel(),
        [0.1 + 0.2j, 0.12 + 0.22j, 0.5 + 0.6j, 0.52 + 0.62j,
         0.3 + 0.4j, 0.32 + 0.42j, 0.7 + 0.8j, 0.72 + 0.82j], rtol=1e-14)


def test_octave_comments_and_wrapped_records():
    """COM Octave: a trailing '!' comment, a whole comment line between
    records, and a record wrapped over two lines all read the same."""
    want_f, want_cs = [1e9, 2e9], _OCT_CS2
    for text in (
            '# GHz S RI R 50\n'
            '1.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 ! trailing\n'
            '2.0 0.11 0.21 0.31 0.41 0.51 0.61 0.71 0.81\n',
            '# GHz S RI R 50\n'
            '1.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8\n'
            '! a comment\n'
            '2.0 0.11 0.21 0.31 0.41 0.51 0.61 0.71 0.81\n',
            '# GHz S RI R 50\n'
            '1.0 0.1 0.2 0.3 0.4\n   0.5 0.6 0.7 0.8\n'
            '2.0 0.11 0.21 0.31 0.41\n   0.51 0.61 0.71 0.81\n'):
        r = _read(text)
        np.testing.assert_allclose(r.freq, want_f, rtol=1e-15)
        np.testing.assert_allclose(r.cs.ravel(), want_cs, rtol=1e-14)


def test_octave_one_port_is_not_swapped():
    """COM Octave: the S12/S21 exchange only happens when nport == 2."""
    r = _read('# GHz S RI R 50\n1.0 0.1 0.2\n2.0 0.11 0.21\n', nport=1,
              suffix='.s1p')
    np.testing.assert_allclose(r.cs.ravel(), [0.1 + 0.2j, 0.11 + 0.21j],
                               rtol=1e-14)


def test_octave_four_port_ordering():
    """COM Octave: cs(i,j,k) is filled row by row from the file and 4-port
    data is NOT swapped."""
    rows = []
    for k, fv in enumerate((1.0, 2.0)):
        row = [str(fv)]
        for i in range(4):
            for j in range(4):
                row += ['%.4f' % (0.01 * (4 * i + j) + k),
                        '%.4f' % (-0.01 * (4 * i + j) - k)]
        rows.append(' '.join(row))
    r = _read('# GHz S RI R 50\n' + '\n'.join(rows) + '\n', nport=4,
              suffix='.s4p')
    assert r.cs.shape == (4, 4, 2)
    np.testing.assert_allclose(
        r.cs.ravel()[:8],
        [0 + 0j, 1 - 1j, 0.01 - 0.01j, 1.01 - 1.01j,
         0.02 - 0.02j, 1.02 - 1.02j, 0.03 - 0.03j, 1.03 - 1.03j], rtol=1e-14,
        atol=1e-16)


def test_missing_option_line_is_refused():
    """The reference DOES NOT TERMINATE here, so there is no answer to match:
    `while ~strcmp(str(1),'#')` keeps calling fgetl, which returns the number
    -1 at end of file, and -1 is neither '#' nor empty, so the n>1000 escape
    inside `if isempty(str)` is never reached.  COM Octave was killed after
    120 s.  Refusing is the only honest behaviour; this test carries no
    oracle value because none exists."""
    _raises('! nothing here\n1.0 2.0\n')
