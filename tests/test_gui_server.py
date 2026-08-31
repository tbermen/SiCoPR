"""GUI — the HTTP layer, exercised over a real socket.

`test_config_roundtrip.py` gates the writer. This gates everything between the
writer and the browser: the endpoints, the JSON shape the page depends on, the
partition property the schematic relies on, and the guards on the save path.

The server is started on a real port and driven with urllib, rather than by
calling the handler functions directly, because half of what can go wrong here
is in the plumbing -- a 500 that returns HTML, a path that escapes the repo, a
value that will not serialise.

    python tests/test_gui_server.py
"""
import glob
import json
import os
import shutil
import sys
import threading
import time
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import re as _re                                   # noqa: E402
from audit_check import check, finish              # noqa: E402
from gui import app as gui_app                     # noqa: E402
from gui import schematic                          # noqa: E402
from gui.config_io import read_config, is_setting, engine_keywords  # noqa: E402

srv = ThreadingHTTPServer(('127.0.0.1', 0), gui_app.Handler)
PORT = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = 'http://127.0.0.1:%d' % PORT


def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=20) as r:
            return r.status, r.read(), r.headers.get('Content-Type', '')
    except urllib.error.HTTPError as e:
        # an error response is a response; the page has to be able to read it
        return e.code, e.read(), e.headers.get('Content-Type', '')


def get_json(path):
    return json.loads(get(path)[1])


def post_json(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


try:
    # -------------------------------------------------------------- static
    code, body, ctype = get('/')
    check("the_page_is_served", code == 200 and b'<svg' in body,
          "GET / returned %s (%s) and %s an inline svg"
          % (code, ctype, 'did' if b'<svg' in body else 'did not'))
    for asset in ('/static/app.js', '/static/style.css'):
        code, body, ctype = get(asset)
        check("asset_is_served_%s" % asset.split('.')[-1],
              code == 200 and len(body) > 200,
              "GET %s returned %s with %d bytes" % (asset, code, len(body)))

    # ------------------------------------------------------------- configs
    configs = get_json('/api/configs')
    check("the_configs_endpoint_answers", isinstance(configs, list),
          "/api/configs returned %r" % (configs,))

    if not configs:
        # Configuration workbooks are IEEE contributions and are not shipped
        # with the repository. A fresh clone has none, and the contract CI
        # enforces is that this SKIPS rather than fails -- otherwise every
        # public clone is red for want of data it cannot have. The static
        # checks above still ran; everything below needs a real config.
        print('\nSKIP: no configuration workbooks present, so the config, '
              'netlist, run and results checks cannot run. See README '
              'section 1 for where they come from.')
        finish()

    rel = configs[0]['path']
    payload = get_json('/api/config?path=' + rel)

    check("config_payload_has_the_shape_the_page_expects",
          all(k in payload for k in ('blocks', 'packages', 'counts', 'name')),
          "keys were %s" % sorted(payload))

    # every cell must carry the fields app.js reads, or the UI renders blanks
    needed = {'name', 'value', 'units', 'coord', 'is_formula', 'formula',
              'is_sweep', 'default', 'info', 'matlab'}
    cells = [c for b in payload['blocks'] for c in b['cells']]
    missing = {k for c in cells for k in needed - set(c)}
    check("every_cell_carries_the_fields_the_page_reads", not missing,
          "cells are missing %s, which the page would render as blanks"
          % sorted(missing))

    # The header line reads CFG.counts.<key>. Renaming a key server-side while
    # the page still reads the old one renders "undefined" -- quiet, and no
    # other check here would notice. (It happened: `decoration` became
    # `annotations`.)
    js_src = get('/static/app.js')[1].decode('utf-8')
    m = _re.search(r'const c = CFG\.counts;(.{0,400})', js_src, _re.S)
    used = set(_re.findall(r'\bc\.([A-Za-z_][A-Za-z0-9_]*)', m.group(1))) if m else set()
    check("the_page_reads_count_keys_that_exist",
          m and used and used <= set(payload['counts']),
          "app.js reads CFG.counts.%s but the payload has %s"
          % (sorted(used - set(payload['counts'])), sorted(payload['counts'])))

    check("the_payload_is_json_serialisable_without_loss",
          all(c['value'] is None or isinstance(c['value'], (bool, int, float, str))
              for c in cells),
          "a cell value survived as a non-primitive: %s"
          % [c['name'] for c in cells
             if not (c['value'] is None
                     or isinstance(c['value'], (bool, int, float, str)))][:5])

    # ------------------------------------- the partition property, on real data
    doc = read_config(os.path.join(_ROOT, rel))
    shown = [c['name'] for c in cells]
    keys = [c['key'] for c in cells]

    # A `[TX RX]` cell appears under both ends of the link deliberately, so the
    # uniqueness invariant is on the FIELD KEY (`C_d#tx`, `C_d#rx`), not on the
    # keyword. Anything else appearing twice is still a bug.
    check("no_field_is_shown_twice",
          len(keys) == len(set(keys)),
          "these field keys appear under more than one block: %s"
          % sorted({k for k in keys if keys.count(k) > 1}))
    dupes = {n for n in shown if shown.count(n) > 1}
    halved = {c['name'] for c in cells if c['half']}
    check("only_tx_rx_pairs_appear_under_two_blocks",
          dupes <= halved,
          "these keywords appear twice without being [TX RX] pairs: %s"
          % sorted(dupes - halved))

    # Nothing may vanish. A keyword is either an editable field, or an
    # annotation that was folded away *because its text is already visible on a
    # real field* as that field's value, units or note. Folding anything else
    # would be hiding a cell the user can see in Excel.
    visible_text = set()
    for c in cells:
        for x in (c['value'], c['units'], c['note']):
            if x is not None:
                visible_text.add(str(x).strip())
    HEADERS = {'parameter', 'setting', 'units', 'information', 'info'}
    lost = [n for n in doc.keywords
            if n not in set(shown)
            and n.strip() not in visible_text
            and n.strip().lower() not in HEADERS]
    check("nothing_in_the_workbook_disappears",
          not lost,
          "these cells are in the sheet, are not shown as a field, and their "
          "text appears nowhere on any field either -- so they are simply "
          "gone: %s" % sorted(lost)[:10])
    check("the_page_invents_nothing",
          not (set(shown) - set(doc.keywords)),
          "the page shows fields that are not in the workbook: %s"
          % sorted(set(shown) - set(doc.keywords))[:8])

    # The halves must reconstruct the cell the engine actually reads.
    for c in cells:
        if not c['half']:
            continue
        check("half_field_%s_carries_its_own_value" % c['key'].replace('#', '_'),
              c['half_value'] is not None and str(c['half_value']).strip() != '',
              "%s is shown as one half of %r but has no half value; the field "
              "would render empty" % (c['key'], c['value']))
    check("schematic_blocks_do_not_claim_a_keyword_twice",
          not schematic.duplicate_names(),
          "gui/schematic.py lists these under two blocks: %s"
          % schematic.duplicate_names())

    # the engine-keyword extraction is the basis of the settings/decoration
    # split; if the regex ever stops matching, everything becomes "decoration"
    kws = engine_keywords()
    check("engine_keyword_extraction_still_works", len(kws) > 150,
          "only %d keywords were extracted from sicopr.py. The settings/"
          "decoration split is built on this, so a broken regex would quietly "
          "mark real settings as unread decoration." % len(kws))
    for probe in ('f_b', 'DER_0', 'N_b', 'g_DC'):
        check("engine_keywords_contains_%s" % probe, is_setting(probe),
              "%r is a real COM setting and the extraction missed it" % probe)
    check("engine_keywords_excludes_sheet_decoration",
          not is_setting('nH') and not is_setting('T_ft'),
          "'nH' is a units cell and 'T_ft' is a reference value read by neither "
          "the port nor MATLAB; treating them as settings would put junk in the "
          "editor")

    # ----------------------------------------------------------- save guards
    # Assert on WHICH error, not merely that there was one. Checking only for
    # `'error' in r` passes whenever any guard fires -- and if a previous run
    # left the escaped file on disk, the *overwrite* guard fires and hides a
    # completely missing path check. That happened during mutation testing: the
    # guard was removed, a file really was written outside the repo, and the
    # test still went green on the following run.
    escaped = os.path.abspath(os.path.join(_ROOT, '..', 'escape.xlsx'))
    existed_before = os.path.exists(escaped)
    code, r = post_json('/api/save',
                        {'src': rel, 'dst': '../escape.xlsx', 'changes': {}})
    check("save_refuses_a_destination_outside_the_repo",
          'outside the repo' in str(r.get('error', '')),
          "writing to '../escape.xlsx' returned %r; the refusal has to be about "
          "the path leaving the repo, not about anything else that happened to "
          "fail" % r)
    check("nothing_was_written_outside_the_repo",
          os.path.exists(escaped) == existed_before,
          "the server created %s. A path parameter that can walk out of the "
          "repo turns this convenience server into a write primitive for "
          "anything else on the machine." % escaped)

    code, r = post_json('/api/save',
                        {'src': rel, 'dst': 'x.txt', 'changes': {}})
    check("save_refuses_a_non_xlsx_destination",
          'xlsx' in str(r.get('error', '')),
          "writing to 'x.txt' returned %r" % r)

    code, r = post_json('/api/save',
                        {'src': rel, 'dst': rel, 'changes': {}})
    check("save_refuses_to_overwrite_without_being_told_to",
          'error' in r and r.get('exists'),
          "saving over the source config returned %r -- a GUI that silently "
          "overwrites the input is how a reference config gets lost" % r)

    code, r = post_json('/api/save',
                        {'src': rel, 'dst': 'gui_tmp_out/x.xlsx',
                         'changes': {'not_a_real_keyword': 1}})
    check("save_rejects_an_unknown_keyword", 'error' in r,
          "an unknown keyword was accepted: %r -- silently dropping it would "
          "produce a config that does not contain the edit the user made" % r)

    # ------------------------------------------------------- a real save
    out_dir = os.path.join(_ROOT, 'gui_tmp_out')
    code, r = post_json('/api/save',
                        {'src': rel, 'dst': 'gui_tmp_out/edited.xlsx',
                         'changes': {'DER_0': 1e-5}})
    check("a_valid_save_succeeds", r.get('ok'),
          "saving one change returned %r" % r)
    if r.get('ok'):
        written = os.path.join(_ROOT, r['dst'])
        check("the_saved_file_exists", os.path.isfile(written),
              "server reported success but %s is not there" % r['dst'])
        back = read_config(written)
        check("the_saved_change_is_in_the_file",
              float(back.keywords['DER_0'].value) == 1e-5,
              "DER_0 was saved as 1e-5 and reads back as %r"
              % back.keywords['DER_0'].value)
        # and the rest of the workbook is untouched
        moved = [k for k in doc.keywords
                 if k != 'DER_0' and k in back.keywords
                 and str(back.keywords[k].value) != str(doc.keywords[k].value)]
        check("saving_one_setting_changes_only_that_setting", not moved,
              "these also changed: %s" % moved[:10])

    # ------------------------------------------------------------ results
    runs = get_json('/api/results')
    check("results_endpoint_returns_a_list", isinstance(runs, list),
          "/api/results returned %r" % (runs,))

    # A run's timestamp must reflect the newest file it contains, not the
    # directory's own mtime. On Windows a directory's mtime does not move when
    # a file inside case_NN/ is rewritten, so a re-run into an existing
    # RESULT_DIR (configs commonly template it by date) would sort BELOW runs
    # days older -- the opposite of what "refresh for the latest" needs.
    stale = []
    for r in runs:
        d = os.path.join(_ROOT, r['path'])
        newest = 0
        for sub, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(('.csv', '.png')):
                    try:
                        newest = max(newest, os.path.getmtime(os.path.join(sub, f)))
                    except OSError:
                        pass
        if newest and r['mtime'] + 1 < newest:
            stale.append((r['path'], r['mtime'], newest))
    check("a_runs_timestamp_reflects_its_newest_output",
          not stale,
          "these report a timestamp older than a file they contain, so a "
          "just-finished re-run would not sort to the top: %s" % stale[:4])

    check("the_results_container_is_not_listed_as_a_run",
          not any(r['path'].rstrip('/') == 'results' for r in runs),
          "the bare 'results' directory is listed as if it were a run; it "
          "aggregates every run inside it and its figure count is the sum of "
          "all of them")

    with_cases = [r for r in runs if r.get('cases')]
    if not with_cases:
        # generated output is not tracked, so a fresh clone legitimately has
        # none. Say so rather than passing silently on an empty corpus.
        check("results_rows_have_the_fields_the_page_reads",
              all({'path', 'cases', 'png', 'mtime'} <= set(r) for r in runs),
              "rows were %r" % (runs[:2],))
        print('\nno result directories with cases present; '
              'the run/figure endpoints were not exercised.')
    else:
        run = get_json('/api/run?path=' + with_cases[0]['path'])
        check("a_run_payload_lists_its_cases",
              run.get('cases'),
              "/api/run for %s returned %r" % (with_cases[0]['path'], run))

        # The run picker counts `case_NN` directories; this payload must agree.
        # It used to treat EVERY subdirectory as a case, so the R dashboard's
        # `lib/` asset folder became a phantom case reporting "no results.csv"
        # and "no figures" -- both false of the run -- and the two counts
        # disagreed.
        check("the_case_count_matches_the_run_listing",
              len(run['cases']) == with_cases[0]['cases']
              or with_cases[0]['cases'] == 0,
              "the picker says %d case(s) and the payload says %d: %s"
              % (with_cases[0]['cases'], len(run['cases']),
                 [c['name'] for c in run['cases']]))
        check("only_case_directories_are_cases",
              all(c['name'].lower().startswith('case_')
                  or c['name'] == os.path.basename(with_cases[0]['path'])
                  for c in run['cases']),
              "these are reported as cases but are not case_NN directories: %s"
              % [c['name'] for c in run['cases']
                 if not c['name'].lower().startswith('case_')])

        # A directory holding exports from more than one configuration has had
        # its figures overwritten; that has to be reported, not discovered.
        multi = None
        for r in runs:
            rp = get_json('/api/run?path=' + urllib.parse.quote(r['path']))
            if len(rp.get('configs_here') or []) > 1:
                multi = rp
                break
        if multi:
            check("a_directory_with_two_configs_says_figures_were_overwritten",
                  multi.get('overwritten'),
                  "%s holds exports from %s but reports no warning; its PNG "
                  "figures belong to whichever ran last"
                  % (multi['path'], multi['configs_here']))

        case = run['cases'][0]
        check("a_case_carries_metrics_or_says_why_not",
              'metrics' in case and 'has_csv' in case,
              "case payload was %r" % case)
        # Every group must carry figures and be labelled. Runs predating the
        # s<N>_ figure naming land in the '?' group rather than being dropped,
        # so a digit id is not required -- only that nothing is unlabelled or
        # empty.
        check("stage_figures_are_grouped_and_labelled",
              all((s['id'].isdigit() or s['id'] == '?')
                  and s['label'] and s['figures'] for s in case['stages']),
              "stages were %r"
              % [(s['id'], s['label'], len(s['figures']))
                 for s in case['stages']])

        figs = [f for s in case['stages'] for f in s['figures']]
        if figs:
            code, body, ctype = get('/figure?path=' + figs[0])
            check("a_figure_is_served_as_a_png",
                  code == 200 and body[:8] == b'\x89PNG\r\n\x1a\n',
                  "GET /figure returned %s (%s) and %r"
                  % (code, ctype, body[:8]))

        # The figure endpoint reads files by path, so it needs the same guard
        # as the save path -- and a type restriction, or it becomes a way to
        # read any file in the repo.
        code, body, _ = get('/figure?path=' + 'README.md')
        check("the_figure_endpoint_refuses_a_non_png",
              code == 404,
              "GET /figure?path=README.md returned %s with %d bytes; this "
              "endpoint exists to show run output, not to read the repo"
              % (code, len(body)))
        # Probe with a png that REALLY EXISTS outside the repo. Asking for a
        # made-up path proves nothing: it 404s because the file is absent,
        # whether or not the guard is there. (That false pass was real -- the
        # first version of this check survived deleting the guard.)
        outside = tempfile.mkdtemp(prefix='sicopr_outside_')
        try:
            bait = os.path.join(outside, 'secret.png')
            with open(bait, 'wb') as f:
                f.write(b'\x89PNG\r\n\x1a\n' + b'\0' * 64)
            rel_bait = os.path.relpath(bait, _ROOT).replace('\\', '/')
            check("the_bait_file_is_reachable_by_relative_path",
                  os.path.isfile(os.path.join(_ROOT, rel_bait)),
                  "the probe path %r does not resolve to the bait file, so the "
                  "next check would pass vacuously" % rel_bait)
            code, body, _ = get('/figure?path=' + urllib.parse.quote(rel_bait))
            check("the_figure_endpoint_refuses_a_path_outside_the_repo",
                  code == 404,
                  "a real .png outside the repo was served (%s, %d bytes). The "
                  "path guard is the only thing stopping this endpoint from "
                  "reading files anywhere on the machine." % (code, len(body)))
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    # -------------------------------------------------------- new config
    dirs_cfg = get_json('/api/config_dirs')
    check("config_directories_are_listed",
          isinstance(dirs_cfg, list) and dirs_cfg,
          "/api/config_dirs returned %r" % (dirs_cfg,))
    check("config_directory_listing_finds_more_than_the_fixed_globs",
          len(dirs_cfg) >= 1 and all('dir' in d and 'count' in d for d in dirs_cfg),
          "rows were %r" % (dirs_cfg[:2],))

    in_dir = get_json('/api/configs?dir='
                      + urllib.parse.quote(dirs_cfg[0]['dir']))
    check("configs_can_be_listed_per_directory",
          isinstance(in_dir, list) and in_dir,
          "listing %s returned %r" % (dirs_cfg[0]['dir'], in_dir))

    # A plain copy must be runnable; that is the whole point of "new from a
    # template" rather than "new from nothing".
    code, r = post_json('/api/new_config',
                        {'template': rel, 'dst': 'gui_tmp_out/copy.xlsx',
                         'reset_defaults': False})
    check("a_new_config_can_be_made_from_a_template", r.get('ok'),
          "creating a plain copy returned %r" % r)

    # Reset-to-defaults must ALSO be runnable. The keywords sheet documents a
    # default per keyword; that set is not a validated configuration, and
    # applying it wholesale produced a workbook the engine refused with
    # `Package Block "-" not found` and then with a TypeError from `C_d`.
    code, r2 = post_json('/api/new_config',
                         {'template': rel, 'dst': 'gui_tmp_out/reset.xlsx',
                          'reset_defaults': True})
    check("a_config_reset_to_declared_defaults_is_still_readable",
          r2.get('ok'),
          "resetting to the workbook's own declared defaults produced a config "
          "the engine cannot read: %r" % r2)
    if r2.get('ok'):
        check("resetting_defaults_actually_changes_something",
              r2.get('reset', 0) > 0,
              "reset_defaults=True changed %r settings" % r2.get('reset'))
        check("settings_without_a_usable_default_are_reported",
              r2.get('no_default', 0) > 0,
              "no settings were reported as lacking a usable default, which "
              "would mean the placeholder/shape filtering silently did nothing")

    code, r3 = post_json('/api/new_config',
                         {'template': rel, 'dst': 'gui_tmp_out/copy.xlsx',
                          'reset_defaults': False})
    check("new_config_refuses_to_overwrite", 'error' in r3,
          "creating over an existing file returned %r" % r3)

    code, r4 = post_json('/api/new_config',
                         {'template': rel, 'dst': '../outside_new.xlsx'})
    check("new_config_refuses_a_path_outside_the_repo", 'error' in r4,
          "creating outside the repo returned %r" % r4)

    # ------------------------------------------------------------- netlist
    dirs = get_json('/api/channel_dirs')
    check("channel_directories_are_listed", isinstance(dirs, list),
          "/api/channel_dirs returned %r" % (dirs,))

    if not dirs:
        # channel S-parameters are not redistributed with the repo, so a fresh
        # clone legitimately has none. Say so; do not pass silently.
        print('\nno .s4p files present; the netlist and run endpoints were '
              'not exercised.')
    else:
        chans = get_json('/api/channels?dir=' + urllib.parse.quote(dirs[0]['dir']))
        check("channels_are_listed_with_a_role", chans and all(
            c['role'] in ('thru', 'fext', 'next', 'unknown') for c in chans),
            "channels came back as %r" % (chans[:2],))

        thru = next((c for c in chans if c['role'] == 'thru'), chans[0])
        fexts = [c['path'] for c in chans if c['role'] == 'fext'][:2]

        code, r = post_json('/api/netlist', {
            'config': rel, 'thru': thru['path'], 'fext': fexts})
        check("a_complete_selection_produces_a_runnable_command",
              r.get('ok') and 'sicopr.py' in r.get('command', ''),
              "netlist returned %r" % r)
        check("the_command_names_every_selected_input",
              all(os.path.basename(p) in r['command']
                  for p in [rel, thru['path']] + fexts),
              "the command %r does not mention every input; a netlist that "
              "silently drops an aggressor would give a quietly wrong answer"
              % r.get('command'))

        code, r = post_json('/api/netlist', {'config': rel, 'thru': ''})
        check("a_missing_thru_is_reported_not_guessed",
              not r.get('ok') and any('THRU' in p for p in r.get('problems', [])),
              "netlist without a THRU returned %r" % r)

        # As with the figure endpoint: probe with a file that REALLY EXISTS
        # outside the repo. A made-up path is rejected for being absent, so it
        # cannot tell a working containment check from a missing one.
        outside_ch = tempfile.mkdtemp(prefix='sicopr_outside_ch_')
        try:
            bait = os.path.join(outside_ch, 'outside.s4p')
            with open(bait, 'w') as f:
                f.write('! not a real touchstone file\n')
            rel_bait = os.path.relpath(bait, _ROOT).replace('\\', '/')
            check("the_channel_bait_file_is_reachable_by_relative_path",
                  os.path.isfile(os.path.join(_ROOT, rel_bait)),
                  "the probe path %r does not resolve; the next check would "
                  "pass vacuously" % rel_bait)
            code, r = post_json('/api/netlist',
                                {'config': rel, 'thru': rel_bait})
            check("the_netlist_refuses_a_channel_outside_the_repo",
                  not r.get('ok')
                  and any('outside' in p for p in r.get('problems', [])),
                  "netlist accepted a real .s4p outside the repo: %r" % r)
        finally:
            shutil.rmtree(outside_ch, ignore_errors=True)

        code, r = post_json('/api/netlist',
                            {'config': rel, 'thru': 'README.md'})
        check("the_netlist_refuses_a_non_s4p_channel",
              not r.get('ok'), "netlist accepted README.md as a channel: %r" % r)

        # -------------------------------------------------- S-parameters
        sp = get_json('/api/sparam?path=' + urllib.parse.quote(thru['path']))
        check("a_touchstone_file_can_be_read_for_plotting",
              sp.get('f_ghz') and sp.get('il_db'),
              "/api/sparam returned %r" % {k: v for k, v in sp.items()
                                           if not isinstance(v, list)})
        check("every_sparam_trace_has_one_point_per_frequency",
              len({len(sp['f_ghz']), len(sp['il_db']),
                   len(sp['rl1_db']), len(sp['rl2_db'])}) == 1,
              "trace lengths differ (f=%d il=%d rl1=%d rl2=%d); the chart would "
              "pair a value with the wrong frequency"
              % (len(sp['f_ghz']), len(sp['il_db']),
                 len(sp['rl1_db']), len(sp['rl2_db'])))
        check("the_sparam_response_is_decimated_for_the_browser",
              sp['decimated_to'] <= 1000 < sp['n_points'],
              "a %d-point file came back as %d points; sending every point of "
              "several overlaid files is how the chart gets slow"
              % (sp['n_points'], sp['decimated_to']))
        check("insertion_loss_is_negative_and_falls_with_frequency",
              sp['il_db'][0] > sp['il_db'][-1] and sp['il_db'][-1] < 0,
              "SDD21 starts at %.2f dB and ends at %.2f dB, which is not the "
              "shape of a channel -- the mixed-mode conversion or the port "
              "order is wrong" % (sp['il_db'][0], sp['il_db'][-1]))
        check("the_engine_resolved_the_port_order",
              len(sp.get('ports') or []) == 4,
              "ports came back as %r; the plot would be of the wrong pairs"
              % (sp.get('ports'),))

        code, body, _ = get('/api/sparam?path=' + urllib.parse.quote(rel))
        check("the_sparam_endpoint_refuses_a_non_touchstone_file",
              b'error' in body,
              "asking for S-parameters of an .xlsx returned %r" % body[:120])

        # ------------------------------------------------------ execution
        # Run something real but instant, rather than a COM run: this proves
        # the spawn, the streaming and the exit reporting, which is what the
        # GUI layer owns. Whether sicopr.py itself is correct is what the rest
        # of the suite is for.
        import gui.runner as _runner
        probe = _runner.start(
            [sys.executable, '-c',
             'import sys;print("hello from the child");sys.exit(3)'],
            _ROOT, label='selftest')
        for _ in range(100):
            if not probe.running:
                break
            time.sleep(0.05)
        st = get_json('/api/exec/status?since=0')
        text = '\n'.join(st.get('lines', []))
        check("a_started_process_streams_its_output",
              'hello from the child' in text,
              "the child printed to stdout and the status endpoint returned %r"
              % text)
        check("the_exit_code_is_reported", st.get('returncode') == 3,
              "the child exited 3; status reported %r" % st.get('returncode'))
        check("a_finished_run_is_not_reported_as_running",
              st.get('running') is False,
              "status said running=%r after the child exited" % st.get('running'))

        since = st['next']
        st2 = get_json('/api/exec/status?since=' + str(since))
        check("polling_from_the_last_offset_returns_nothing_new",
              st2.get('lines') == [],
              "re-polling from offset %d returned %r; the page would print the "
              "same lines again" % (since, st2.get('lines')))

        # ------------------------------------------- long, loud runs
        #
        # A full-grid search is long AND chatty. Two things have to hold or the
        # UI looks frozen while the engine is fine: one response must not carry
        # the whole buffer, and draining it in chunks must not lose a line.
        loud = _runner.start(
            [sys.executable, '-c',
             'for i in range(9000): print("line %d" % i)'],
            _ROOT, label='loadtest')
        for _ in range(400):
            if not loud.running:
                break
            time.sleep(0.05)

        first = get_json('/api/exec/status?since=0')
        check("one_status_response_is_capped",
              0 < len(first['lines']) <= _runner._MAX_CHUNK,
              "a single response carried %d lines. Handing the browser the "
              "whole buffer at once, repeatedly, is what makes a long run lock "
              "the tab." % len(first['lines']))
        check("a_capped_response_says_how_far_behind_it_is",
              first.get('behind', 0) > 0,
              "the response was capped but reports behind=%r, so the client "
              "has no way to know it should keep draining"
              % first.get('behind'))

        got, since2, polls = [], 0, 0
        while polls < 60:
            s = get_json('/api/exec/status?since=%d' % since2)
            got += s['lines']
            polls += 1
            if s['next'] == since2:
                break
            since2 = s['next']
        nums = [int(l.split()[1]) for l in got if l.startswith('line ')]
        check("draining_in_chunks_loses_no_output",
              nums and nums == list(range(nums[0], nums[0] + len(nums))),
              "the lines received across %d polls are not contiguous; the "
              "offset accounting drops output. got %d numbered lines, "
              "first=%r last=%r"
              % (polls, len(nums), nums[:1], nums[-1:]))
        check("draining_terminates",
              polls < 60,
              "the client polled 60 times without the offset settling, so it "
              "would poll forever")

        # a second run must not be able to start on top of a live one
        slow = _runner.start([sys.executable, '-c',
                              'import time;time.sleep(30)'], _ROOT)
        try:
            code, r = post_json('/api/exec/start',
                                {'config': rel, 'thru': thru['path']})
            check("a_second_run_is_refused_while_one_is_live",
                  'error' in r,
                  "starting a second run returned %r. COM runs are CPU-heavy; "
                  "letting the browser stack them up is how the machine falls "
                  "over." % r)
        finally:
            slow.stop()

    # --------------------------------------------------- tabs and layout
    #
    # Each operation owns a tab. The point of the split is that the Run tab
    # cannot quietly change which config is being run -- that belongs to
    # Config -- so the structural check is that the picker lives in one place.
    page = get('/')[1].decode('utf-8')
    tabs = _re.findall(r'data-view="(\w+)"', page)
    views = _re.findall(r'<section id="(\w+)View"', page)
    check("the_page_has_the_five_tabs",
          tabs == ['config', 'sparam', 'run', 'results', 'dynamic'],
          "tabs are %s" % tabs)
    check("every_tab_has_a_view", set(tabs) <= set(views),
          "tabs %s but views %s" % (tabs, views))

    def section(name):
        i = page.index('id="%sView"' % name)
        rest = [page.index('id="%sView"' % v) for v in views
                if page.index('id="%sView"' % v) > i]
        return page[i:min(rest) if rest else len(page)]

    check("the_config_picker_lives_in_the_config_tab",
          'id="configPick"' in section('config'),
          "the configuration picker is not in the Config tab")
    check("the_run_tab_does_not_pick_a_config",
          'id="configPick"' not in section('run')
          and 'id="dirPick"' not in section('run'),
          "the Run tab contains a config picker. Separating the tabs is "
          "pointless if the run can still change what it is running from "
          "inside itself.")
    check("the_run_tab_says_which_config_it_will_use",
          'id="nlConfig"' in section('run'),
          "the Run tab does not show the selected configuration, so the user "
          "cannot tell what is about to run")

    # ------------------------------------------------- dynamic results
    dyn_runs = [r for r in runs
                if glob.glob(os.path.join(_ROOT, r['path'], '**', '*.mat'),
                             recursive=True)]
    if not dyn_runs:
        print('\nno .mat exports present; the dynamic-results endpoints were '
              'not exercised.')
    else:
        d = get_json('/api/dynamic?path='
                     + urllib.parse.quote(dyn_runs[0]['path']))
        check("dynamic_results_list_the_mat_exports",
              d.get('cases'),
              "/api/dynamic for %s returned %r" % (dyn_runs[0]['path'], d))
        check("each_export_says_whether_a_dashboard_exists",
              all('report' in c and 'mat' in c for c in d['cases']),
              "case rows were %r" % d['cases'][:1])

        built = [c for c in d['cases'] if c['report']]
        if built:
            rpt = built[0]['report']
            mapped = '/rpt/' + '/'.join(urllib.parse.quote(p)
                                        for p in rpt.split('/'))
            code, body, ctype = get(mapped)
            check("a_built_dashboard_is_served_as_html",
                  code == 200 and b'<' in body[:200] and 'html' in ctype,
                  "GET %s returned %s (%s), %d bytes"
                  % (mapped, code, ctype, len(body)))

            # `htmltools::save_html` links Plotly and friends from a sibling
            # lib/ with RELATIVE paths -- it does not inline them. Serving only
            # the .html gave a dashboard with all its text and none of its
            # graphs. Every asset the page references must resolve, or the
            # plots are blank and nothing else here would notice.
            import posixpath
            page = body.decode('utf-8', 'replace')
            refs = sorted({m for m in _re.findall(
                r'(?:src|href)="([^"]+)"', page)
                if not m.startswith(('data:', '#', 'http'))})
            check("the_dashboard_references_relative_assets", refs,
                  "the page references no relative assets, so this check "
                  "proves nothing about whether they would load")
            base = posixpath.dirname(mapped)
            missing = []
            for ref in refs:
                au = posixpath.normpath(
                    posixpath.join(base, urllib.parse.quote(ref)))
                c2, b2, t2 = get(au)
                if c2 != 200 or not b2:
                    missing.append(ref)
                elif ref.endswith('.js') and 'javascript' not in t2:
                    missing.append('%s (served as %s)' % (ref, t2))
            check("every_asset_the_dashboard_needs_is_served",
                  not missing,
                  "these are referenced by the dashboard and do not load, so "
                  "its graphs render blank: %s" % missing)

            # The asset route must not become a way to read the repo's source.
            for bad in ('gui/static/app.js', 'gui/app.py', 'VERSION.json'):
                c3, _, _ = get('/rpt/' + urllib.parse.quote(bad))
                check("the_asset_route_refuses_%s"
                      % bad.replace('/', '_').replace('.', '_'),
                      c3 == 404,
                      "GET /rpt/%s returned %s; assets are only servable when "
                      "they live under a directory containing a dashboard"
                      % (bad, c3))

        # /report reads files by path, so it needs the same containment and
        # type restrictions as /figure.
        for bad in ('README.md', 'gui/app.py'):
            code, _, _ = get('/rpt/' + urllib.parse.quote(bad))
            check("the_report_route_refuses_%s" % bad.replace('/', '_')
                  .replace('.', '_'),
                  code == 404,
                  "GET /rpt/%s returned %s; this route exists to show a "
                  "generated dashboard, not to serve the repo" % (bad, code))

        # a render must not be blocked by, or block, a COM run
        held = _runner.start([sys.executable, '-c', 'import time;time.sleep(20)'],
                             _ROOT, slot='com')
        try:
            other = _runner.start(
                [sys.executable, '-c', 'print("render slot")'],
                _ROOT, slot='render')
            check("a_dashboard_build_does_not_collide_with_a_com_run",
                  other is not None,
                  "starting a render while a COM run is live was refused; the "
                  "two are different kinds of work and must not share a slot")
            for _ in range(100):
                if not other.running:
                    break
                time.sleep(0.05)
            st = get_json('/api/exec/status?slot=render&since=0')
            check("the_render_slot_reports_separately",
                  'render slot' in '\n'.join(st.get('lines', [])),
                  "the render slot's status returned %r" % st.get('lines'))
            st_com = get_json('/api/exec/status?slot=com&since=0')
            check("the_two_slots_do_not_share_output",
                  'render slot' not in '\n'.join(st_com.get('lines', [])),
                  "the COM slot is showing the render job's output")
        finally:
            held.stop()

    # ------------------------------------------ the page and the script agree
    #
    # `$('#thing')` on an id the HTML does not define returns null, and the
    # failure shows up as a dead button rather than an error. Nothing else in
    # this suite would notice: the server is perfectly happy. This caught a real
    # one while the results view was being built (`#resultsDlg` was renamed to
    # `#figDlg` and a handler kept pointing at the old id).
    html = get('/')[1].decode('utf-8')
    js = get('/static/app.js')[1].decode('utf-8')
    html_ids = set(_re.findall(r'id="([^"]+)"', html))
    js_ids = set(_re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", js))
    dangling = sorted(js_ids - html_ids)
    check("every_element_the_script_looks_up_exists_in_the_page",
          not dangling,
          "app.js calls $('#x') for ids the page does not define: %s. Those "
          "return null and fail silently in the browser." % dangling)

    css = get('/static/style.css')[1].decode('utf-8')
    styled = set(_re.findall(r'#([A-Za-z][A-Za-z0-9_-]*)\s*[{,:]', css))
    check("the_stylesheet_does_not_target_removed_elements",
          not (styled - html_ids),
          "style.css styles ids that no longer exist: %s"
          % sorted(styled - html_ids))

    # ------------------------------------------------------------- errors
    code, body, ctype = get('/api/config?path=nope.xlsx')
    check("a_missing_config_is_a_json_404",
          code == 404 or b'error' in body,
          "requesting a missing config returned %s / %r" % (code, body[:80]))
    check("errors_are_json_not_html", 'json' in ctype.lower(),
          "the page parses every response as JSON; this one was %r" % ctype)

    print("\nserved %d config(s); %d cells across %d blocks; %d engine keywords."
          % (len(configs), len(cells), len(payload['blocks']), len(kws)))
finally:
    srv.shutdown()
    shutil.rmtree(os.path.join(_ROOT, 'gui_tmp_out'), ignore_errors=True)

finish()
