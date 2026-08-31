"""Which config keyword belongs to which block of the channel schematic.

The GUI draws a signal path -- Tx die, Tx package, channel, Rx package, Rx die,
with FEXT/NEXT aggressors and the TxFFE -> CTLE -> RxFFE -> DFE equalisation
chain -- and clicking a block shows the settings that block owns.

Two rules this file exists to enforce:

  * **Nothing is hidden.** Every keyword in a config lands in exactly one block,
    and anything not claimed by name falls into OTHER. A GUI that silently drops
    a setting it does not recognise is worse than no GUI, because the user
    reasonably assumes what they see is everything.
  * **The grouping is presentation only.** It never changes which cell a value
    is written to; that is `config_io`'s job and is driven by the workbook.

`tests/test_config_roundtrip.py` checks the partition property against the real
configs, so a new keyword in a future config shows up as a test failure rather
than as a setting nobody can find.
"""
# Copyright 2025 802-COM Authors (upstream MATLAB reference)
# Copyright 2026 Todd Bermensolo (Python port)
# SPDX-License-Identifier: BSD-3-Clause


# Block order is the drawing order along the signal path.
# (id, label, row, keywords) -- `row` picks the lane in the schematic:
#   'signal'  the main left-to-right channel
#   'agg'     aggressors feeding the victim
#   'eq'      the receiver equalisation chain
#   'ctl'     analysis and run control, drawn as a side panel
BLOCKS = [
    ('tx_die', 'Tx die', 'signal', [
        'A_v', 'R_d', 'C_d', 'L_s', 'SNR_TX', 'T_r', 'T_t', 'FORCE_TR',
    ]),
    ('tx_pkg', 'Tx package', 'signal', [
        'z_p (TX)', 'z_bp (TX)', 'C_p', 'C_b', 'z_p select', 'PKG_NAME',
        'package_Z_c', 'package_tl_gamma0_a1_a2', 'package_tl_tau',
        'TDR_W_TXPKG',
    ]),
    ('channel', 'Channel', 'signal', [
        'f_b', 'f_min', 'f_1', 'f_2', 'Delta_f', 'flim', 'f_v', 'f_f', 'f_n',
        'f_r', 'Include PCB', 'board_Z_c', 'board_tl_gamma0_a1_a2',
        'board_tl_tau', 'Port Order', 'PMD_type', 'C2C', 'zero_pad', 'UI',
        'fixture delay time',
    ]),
    ('rx_pkg', 'Rx package', 'signal', [
        'z_p (RX)', 'z_bp (RX)',
    ]),
    ('rx_die', 'Rx die', 'signal', [
        'R_0', 'Z_t', 'C_0', 'C_1',
    ]),

    # A_ft / A_nt are the FEXT / NEXT aggressor amplitudes used for ICN -- they
    # belong with their aggressors, not with the Rx die where an earlier pass
    # of this file put them.
    ('fext', 'FEXT aggressors', 'agg', [
        'A_fe', 'A_ft', 'z_p (FEXT)', 'z_bp (FEXT)', 'T_ft',
    ]),
    ('next', 'NEXT aggressors', 'agg', [
        'A_ne', 'A_nt', 'z_p (NEXT)', 'z_bp (NEXT)', 'T_nt',
    ]),

    ('txffe', 'Tx FFE', 'eq', [
        'c(0)', 'c(-1)', 'c(-2)', 'c(-3)', 'c(-4)',
        'c(1)', 'c(2)', 'c(3)',
    ]),
    ('ctle', 'CTLE', 'eq', [
        'g_DC', 'f_z', 'f_p1', 'f_p2', 'g_DC_HP', 'f_HP_PZ',
    ]),
    ('rxffe', 'Rx FFE', 'eq', [
        'ffe_pre_tap_len', 'ffe_post_tap_len', 'ffe_pre_tap1_max',
        'ffe_post_tap1_max', 'ffe_tapn_max', 'N_f', 'num_ui_RXFF_noise',
        'max FFE value for floating taps',
    ]),
    ('dfe', 'DFE', 'eq', [
        'N_b', 'b_max(1)', 'b_max(2..N_b)', 'b_min(1)', 'b_min(2..N_b)',
        'N_bg', 'N_bf', 'N_bx', 'bmaxg', 'N_tail_start',
        'taps per group', 'UI span for floating taps',
    ]),
    ('detect', 'Detector and noise', 'eq', [
        'DER_0', 'sigma_RJ', 'A_DD', 'eta_0', 'R_LM', 'L', 'M', 'N',
        'Sigma BBN step', 'MLSE', 'Clip Method', 'N_qb', 'P_qc', 'S',
    ]),

    ('search', 'Optimiser', 'ctl', [
        'Local Search', 'Non-zero Local Search Method', 'TS_SRCH_MODE',
        'sample_adjustment', 'ts_anchor', 'Overwrite Minimum Radius',
    ]),
    ('erl', 'ERL / TDR', 'ctl', [
        'ERL', 'ERL_ONLY', 'ERL Pass threshold', 'TDR', 'TR_TDR',
        'TDR_Butterworth', 'Tukey_Window', 'N_bx_TDR', 'rho_x', 'beta_x',
    ]),
    ('run', 'Run and output', 'ctl', [
        'COM Pass threshold', 'RESULT_DIR', 'RUNTAG', 'SAVE_FIGURES',
        'DIAGNOSTICS', 'DISPLAY_WINDOW', 'CSV_REPORT', 'SAVE_CONFIG2MAT',
        'COM_CONTRIBUTION', 'RX_CALIBRATION', 'EW', 'BREAD_CRUMBS',
    ]),
]

OTHER = ('other', 'Other settings', 'ctl')

# Several die and package settings are a single cell holding BOTH ends of the
# link -- `C_d`, `L_s`, `C_b` and `PKG_NAME` are marked `[TX RX]` in the sheet's
# own annotation column. Listing them only under the Tx block is what left the
# Rx package showing one setting against the Tx package's five.
#
# So a `[TX RX]` keyword listed in a Tx-side block is mirrored into the paired
# Rx-side block, each side showing and editing its own half of the value. The
# cell written is still the one cell the engine reads.
MIRROR = {'tx_die': 'rx_die', 'tx_pkg': 'rx_pkg'}


def assign(keyword_names):
    """Partition `keyword_names` across the blocks.

    Returns a list of {id, label, row, keywords} in drawing order. Blocks with
    no keywords in this config are still returned (empty) so the schematic keeps
    its shape; the OTHER block collects everything unclaimed.
    """
    remaining = list(keyword_names)
    lookup = {k.lower(): k for k in remaining}
    out = []
    claimed = set()

    for bid, label, row, names in BLOCKS:
        mine = []
        for n in names:
            actual = lookup.get(n.lower())
            if actual is not None and actual not in claimed:
                mine.append(actual)
                claimed.add(actual)
        out.append({'id': bid, 'label': label, 'row': row, 'keywords': mine})

    leftover = [k for k in remaining if k not in claimed]
    out.append({'id': OTHER[0], 'label': OTHER[1], 'row': OTHER[2],
                'keywords': leftover})
    return out


def duplicate_names():
    """Keywords listed under more than one block -- a spec error in this file,
    not in any config. `assign` would silently give the keyword to whichever
    block came first, so it is worth failing on."""
    seen, dup = set(), set()
    for _, _, _, names in BLOCKS:
        for n in names:
            key = n.lower()
            if key in seen:
                dup.add(n)
            seen.add(key)
    return sorted(dup)
