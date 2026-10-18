"""GUI — static sanity checks on the page assets.

**`app.js` is parsed, not eyeballed.** A syntax error there breaks the entire
page while every server-side test still passes: the endpoints keep returning
200, the ids all still line up, and the UI simply does nothing. That is not
hypothetical -- an editing slip put literal newlines inside a single-quoted
string, and the result was a config editor with empty pickers and dead buttons
that no other test in this repo could see.

Parser, in order of preference:

  1. `node --check` if node/deno/bun is installed
  2. the `esprima` package (pure Python, dev-only: `pip install esprima`)
  3. a delimiter-balance fallback, which is NOT a parser and misses exactly the
     defect described above -- it is a last resort, and it says so

Install esprima if this ever reports that it fell through to the fallback.

    python tests/test_gui_static.py
"""
# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

import io
import os
import re
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from audit_check import check, finish  # noqa: E402

STATIC = os.path.join(_ROOT, 'gui', 'static')
CSS = os.path.join(STATIC, 'style.css')
HTML = os.path.join(STATIC, 'index.html')
JS = os.path.join(STATIC, 'app.js')


def strip_js(src):
    """Remove strings, template literals and comments.

    Regex literals are not handled: this codebase writes them only inside
    `.test(...)` / `.replace(...)` calls where they are balanced anyway, and a
    half-parser that tried to guess regex-vs-division would report noise.
    """
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ''
        if c == '/' and nxt == '/':
            i = src.find('\n', i)
            if i < 0:
                break
            continue
        if c == '/' and nxt == '*':
            j = src.find('*/', i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in '"\'`':
            quote, i = c, i + 1
            while i < n:
                if src[i] == '\\':
                    i += 2
                    continue
                if src[i] == quote:
                    i += 1
                    break
                i += 1
            out.append('""')
            continue
        out.append(c)
        i += 1
    return ''.join(out)


src = io.open(JS, encoding='utf-8').read()
check("app_js_is_not_empty", len(src) > 1000,
      "gui/static/app.js is %d bytes" % len(src))

# Prefer a real parser whenever the machine has one.
engine = next((e for e in ('node', 'deno', 'bun') if shutil.which(e)), None)
try:
    import esprima
except ImportError:
    esprima = None

parsed = False
if engine == 'node':
    p = subprocess.run([engine, '--check', JS], capture_output=True, text=True)
    check("app_js_parses", p.returncode == 0,
          "node --check failed:\n%s" % (p.stderr or p.stdout))
    parsed = True
elif esprima is not None:
    err = ''
    try:
        esprima.parseScript(src)
    except Exception as e:                                # noqa: BLE001
        ln = getattr(e, 'lineNumber', None)
        ctx = ''
        if ln:
            lines = src.split('\n')
            ctx = '\n     '.join(
                '%s%5d| %s' % ('>>' if i == ln - 1 else '  ', i + 1, lines[i])
                for i in range(max(0, ln - 3), min(len(lines), ln + 2)))
        err = '%s\n     %s' % (e, ctx)
    check("app_js_parses", not err,
          "app.js is not valid JavaScript, so the page will not run at all "
          "and no server-side test can tell:\n     %s" % err)
    parsed = True

if not parsed:
    body = strip_js(src)
    stack, bad = [], []
    pairs = {')': '(', ']': '[', '}': '{'}
    for lineno, line in enumerate(body.split('\n'), 1):
        for ch in line:
            if ch in '([{':
                stack.append((ch, lineno))
            elif ch in ')]}':
                if not stack:
                    bad.append('line %d: stray %r' % (lineno, ch))
                elif stack[-1][0] != pairs[ch]:
                    bad.append('line %d: %r closes %r opened on line %d'
                               % (lineno, ch, stack[-1][0], stack[-1][1]))
                    stack.pop()
                else:
                    stack.pop()
    if stack:
        bad.append('unclosed %r from line %d' % (stack[-1][0], stack[-1][1]))
    check("app_js_delimiters_balance", not bad,
          "app.js has unbalanced delimiters, which means the page will not "
          "run at all. NOTE: this is a delimiter check, not a parser -- no "
          "node/deno/bun is installed here.\n     " + "\n     ".join(bad[:6]))

    # An unterminated string swallows the rest of the file, which usually shows
    # up as a wildly wrong stripped length rather than as an imbalance.
    check("app_js_has_no_runaway_string",
          body.count('""') < src.count('"') + src.count("'") + src.count('`'),
          "the string-stripping pass produced an implausible result; a quote "
          "is probably unterminated")

# ---------------------------------------------------------- freeze guards
#
# Three shapes in the status loop each present to the user as "the GUI froze"
# while the run itself is perfectly healthy. All three were real.
if esprima is not None:
    tree = esprima.parseScript(src, {'range': True})

    def find_fn(node, name, out=None):
        out = [] if out is None else out
        if isinstance(node, list):
            for n in node:
                find_fn(n, name, out)
            return out
        if not hasattr(node, 'type'):
            return out
        if getattr(node, 'type', '') in ('FunctionDeclaration',) \
                and getattr(getattr(node, 'id', None), 'name', '') == name:
            out.append(node)
        for key in dir(node):
            if key.startswith('_') or key in ('type', 'range'):
                continue
            try:
                v = getattr(node, key)
            except Exception:                          # noqa: BLE001
                continue
            if isinstance(v, list) or hasattr(v, 'type'):
                find_fn(v, name, out)
        return out

    fns = find_fn(tree.body, 'pollStatus')
    check("pollStatus_exists", fns, "no pollStatus function found in app.js")
    if fns:
        a, b = fns[0].range
        body = src[a:b]
        # 1. a failed poll must retry, not end the loop
        catch_at = body.find('catch')
        nxt = body.find('if (st.idle', catch_at)
        catch_body = body[catch_at:nxt if nxt > 0 else len(body)]
        check("a_failed_status_poll_reschedules_itself",
              'setTimeout' in catch_body,
              "pollStatus's catch block does not schedule another poll. One "
              "transient failure then leaves the status window dead for the "
              "rest of the run while sicopr.py keeps going -- which is "
              "indistinguishable from a freeze.")

    # 1b. finishing is an event, not a state.
    #
    # The completion branch used to fire on every poll that saw a finished run,
    # and entering the Run tab polls -- so returning to that tab relaunched the
    # R dashboard build and threw the user straight back out to the dynamic
    # tab, with no way to get back. The branch must be guarded by which run has
    # already been announced.
    if fns:
        check("run_completion_fires_once_per_run",
              'RUN_DONE' in body and 'RUN_DONE = st.started' in body,
              "pollStatus does not guard its completion branch by run "
              "identity, so every poll of an already-finished run re-triggers "
              "it -- including the poll that happens when the Run tab is "
              "reopened.")
        auto = body.find('autoBuildDynamic')
        guard = body.find('RUN_DONE === st.started')
        check("the_guard_precedes_the_completion_action",
              guard != -1 and auto != -1 and guard < auto,
              "the fire-once guard does not come before autoBuildDynamic(), "
              "so it cannot prevent the re-trigger")

    # 2. the terminal must be bounded
    check("the_terminal_output_is_bounded",
          'TERM_MAX' in src and 'TERM.slice' in src,
          "app.js does not trim the terminal. An unbounded <pre> on a long, "
          "chatty run grows into megabytes and every update re-lays it out.")

    # 3. and must not be built by repeated string concatenation.
    #
    # Checked against ANY `textContent +=`, not the one spelling this bug
    # happened to have. The first version of this check looked for
    # `#term').textContent +=` literally and passed happily when the same
    # defect was reintroduced through a local variable.
    def code_lines(text):
        """(lineno, code) with comments removed, line numbers preserved.

        Needed because this very file's explanatory comment quotes the defect
        it forbids, and a naive scan flagged the comment as the bug.
        """
        out, in_block = [], False
        for n, line in enumerate(text.split('\n'), 1):
            buf, i = [], 0
            while i < len(line):
                two = line[i:i + 2]
                if in_block:
                    if two == '*/':
                        in_block = False
                        i += 2
                    else:
                        i += 1
                    continue
                if two == '/*':
                    in_block = True
                    i += 2
                    continue
                if two == '//':
                    break
                buf.append(line[i])
                i += 1
            out.append((n, ''.join(buf)))
        return out

    concat = [n for n, l in code_lines(src)
              if 'textContent +=' in l or 'innerHTML +=' in l]
    check("no_dom_node_is_grown_by_concatenation",
          not concat,
          "app.js appends to a DOM node with `+=` at line(s) %s. That re-reads "
          "and re-renders the whole node on every update -- O(n) per poll, and "
          "on a long chatty run the tab locks up." % concat)

# 'use strict' at the top means a stray assignment to an undeclared name is a
# runtime error rather than a silent global. The directive must be the first
# *statement*; leading comments (the licence header) do not displace it, so this
# asks the parser rather than matching raw text.
if esprima is not None:
    body = tree.body if not isinstance(tree, dict) else tree['body']
    first = body[0] if body else None
    directive = getattr(first, 'directive', None) if first is not None else None
    check("app_js_is_strict_mode", directive == 'use strict',
          "the first statement of app.js is not the 'use strict' directive "
          "(got %r)" % (directive,))
else:
    check("app_js_is_strict_mode", src.lstrip().startswith("'use strict'"),
          "app.js does not start with 'use strict'")

# Every id the stylesheet targets and every id the script looks up must exist.
# (The script/page agreement is also checked in test_gui_server.py against the
# served copies; this catches it without starting a server.)
html = io.open(os.path.join(STATIC, 'index.html'), encoding='utf-8').read()
css = io.open(os.path.join(STATIC, 'style.css'), encoding='utf-8').read()
html_ids = set(re.findall(r'id="([^"]+)"', html))
js_ids = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", src))
check("script_ids_exist_in_the_page", not (js_ids - html_ids),
      "app.js looks up ids the page does not define: %s"
      % sorted(js_ids - html_ids))
css_ids = set(re.findall(r'#([A-Za-z][A-Za-z0-9_-]*)\s*[{,: ]', css))
check("stylesheet_ids_exist_in_the_page", not (css_ids - html_ids),
      "style.css targets ids the page does not define: %s"
      % sorted(css_ids - html_ids))

# Tags the panel emits must have styling, or they render as unstyled text.
tags = set(re.findall(r"tag\('([a-z]+)'", src))
styled = set(re.findall(r'\.tag\.([a-z]+)', css))
check("every_tag_style_exists", not (tags - styled),
      "app.js emits tag classes with no style rule: %s" % sorted(tags - styled))

how = ('node --check' if engine == 'node'
       else 'esprima' if esprima is not None
       else 'DELIMITER FALLBACK (not a parser)')
print("\napp.js %d bytes, checked with %s; %d ids used, %d defined; tags: %s"
      % (len(src), how, len(js_ids), len(html_ids), sorted(tags)))
if not parsed:
    print("   WARNING: app.js was NOT parsed. Install a JS engine, or "
          "`pip install esprima` -- a syntax error here is invisible to every "
          "other test in this repo.")

# ---- the hidden attribute actually hides -------------------------------
#
# `el.hidden = true` relies on the browser's `[hidden] { display: none }`, which
# is a UA rule and loses to ANY author rule that sets `display` on the element.
# So a class with `display: flex` silently makes `hidden` do nothing, and the
# element stays on screen while the JS believes it closed it.
#
# That shipped: the config browse dialog had `.modalback { display: flex }` and
# no `[hidden]` rule, so neither Cancel nor "Use this directory" appeared to
# work. Nothing else here could see it -- the ids lined up, the JS parsed, the
# endpoints answered.
_html = io.open(HTML, encoding='utf-8').read()
_css = io.open(CSS, encoding='utf-8').read()

_classes = set()
for _tag, _attrs in re.findall(r'<(\w+)([^>]*\bhidden\b[^>]*)>', _html):
    _m = re.search(r'class="([^"]+)"', _attrs)
    if _m:
        _classes.update(_m.group(1).split())

_broken = []
for _c in sorted(_classes):
    _sets_display = re.search(
        r'\.' + re.escape(_c) + r'\s*(?:,[^{]*)?\{[^}]*?display:\s*[a-z-]+', _css)
    _has_hide = re.search(r'\.' + re.escape(_c) + r'\[hidden\]', _css)
    if _sets_display and not _has_hide:
        _broken.append(_c)

check("every_hidden_class_can_actually_be_hidden",
      not _broken,
      "these classes are used with the hidden attribute AND set display, so "
      "hidden does nothing for them; each needs its own [hidden] rule: %s"
      % ', '.join('.' + c for c in _broken))


# ---- dialog text is readable -------------------------------------------
#
# The same dialog shipped with hardcoded greys: #8b95a1 on white is about
# 2.9:1, below the 4.5:1 that normal text needs, and a hardcoded white panel is
# wrong in dark mode. The stylesheet already carries variables for both.
_modal_block = _css[_css.find('.modalback'):] if '.modalback' in _css else ''
_hardcoded = re.findall(r'(?:color|background)\s*:\s*(#[0-9a-fA-F]{3,6})',
                        _modal_block)
check("the_browse_dialog_takes_its_colours_from_the_theme",
      not _hardcoded,
      "hardcoded colours in the browse dialog (%s); use var(--ink), "
      "var(--muted), var(--panel), var(--line) so it follows the theme and "
      "stays legible" % ', '.join(sorted(set(_hardcoded))))


# ---- the dashboard URL keeps its directory ------------------------------
#
# /rpt/ is path-mapped so that a dashboard's relative lib/... links resolve to
# its own directory. That only works if the separators are real URL segments.
# A run outside the repo (say D:\runs) reaches the page as an absolute Windows path, and
# '...'.split('/') on one of those returns a SINGLE segment, so the base for
# every relative link collapses to '/rpt/'. The dashboard still loads, so
# nothing looks broken: the text renders, Plotly 404s, and no chart is drawn.
# Found on 2026-09-14, after the runs tree moved outside the repository.
_rpt = re.search(r"""['"]/rpt/['"]\s*\+\s*(.+?);""", src, re.S)
check("the_dashboard_url_is_split_into_path_segments",
      _rpt and '.split(' in _rpt.group(1),
      "the /rpt/ URL is not built by splitting the path into segments; "
      "relative lib/... links inside the dashboard will not resolve")
check("the_dashboard_url_normalises_windows_separators",
      _rpt and re.search(r"replace\(\s*/\\\\+/g\s*,\s*['\"]/['\"]\s*\)",
                         _rpt.group(1)),
      "the /rpt/ URL is built without turning backslashes into '/' first, so "
      "an absolute Windows report path becomes one URL segment and every "
      "relative lib/... link 404s: the dashboard renders with no charts")

# A file name taken with split('/') keeps the whole of a Windows path. In the
# static results that made every figure caption a full path and, worse, made
# the "not produced by this run" list name all 17 figures while 15 or 17 were
# on screen (release audit, 2026-10-01).
check("file_names_are_split_on_either_separator",
      not re.search(r"split\(\s*['\"]/['\"]\s*\)\s*\.pop\(\)", src),
      "app.js takes a file name with split('/').pop(); a Windows path has "
      "backslashes, so the 'name' is the whole path")

# The unsaved-changes bar sets display:flex, which beats the browser's
# [hidden] rule: the bar stayed on screen at 0 changes, over the page.
check("dirty_bar_hides_when_hidden",
      re.search(r"#dirtyBar\[hidden\]\s*\{[^}]*display\s*:\s*none", css),
      "#dirtyBar sets display:flex and has no [hidden] rule, so "
      "el.hidden = true leaves it on screen")


# ---- panels: the slot and the event contract ----------------------------
#
# With no panel installed the page must look as it did before panels: the slot
# starts hidden, and only loadPanels() shows it, after /api/panels lists one.
check("the_panel_slot_is_hidden_by_default",
      re.search(r'<aside id="panelSlot"[^>]*\bhidden\b', html) is not None,
      "index.html does not start the panel slot hidden; with no panel it "
      "would take a column of the page")
check("the_panel_slot_has_its_own_hidden_rule",
      re.search(r"\.panelslot\[hidden\]\s*\{[^}]*display\s*:\s*none", css),
      ".panelslot sets display and has no [hidden] rule, so the empty slot "
      "would stay on screen")

# The events are the whole of the contract on the page. A name the script
# sends or handles that the README does not list is one no panel author can
# know about; a renamed one silently breaks every panel.
_readme = io.open(os.path.join(_ROOT, 'gui', 'README.md'), encoding='utf-8').read()
_sent = set(re.findall(r"\bemit\('([a-z-]+)'", src))
_heard = set(re.findall(r"addEventListener\('sicopr:([a-z-]+)'", src))
_documented = set(re.findall(r'`sicopr:([a-z-]+)`', _readme))
check("the_page_sends_and_handles_panel_events", _sent and _heard,
      "found no sicopr: events in app.js (sent %r, handled %r)" % (_sent, _heard))
check("every_panel_event_is_in_the_readme",
      not ((_sent | _heard) - _documented),
      "app.js uses these events, which gui/README.md does not list: %s"
      % sorted((_sent | _heard) - _documented))
check("the_readme_lists_no_event_the_page_lacks",
      not (_documented - (_sent | _heard)),
      "gui/README.md lists these events, which app.js neither sends nor "
      "handles: %s" % sorted(_documented - (_sent | _heard)))


# ---- panels: the channel files chosen for a run (sicopr:channels) -------
#
# A panel hears which workbook is shown (sicopr:config) but had no way to know
# which THRU, FEXT and NEXT files are picked in Run, so "run what is selected"
# typed in a panel could not mean anything. The event is sent once at start
# and whenever that selection changes, whoever changed it: a pick, or the
# lists being filled again (a directory, the filter, the first THRU taken by
# default).
def _fn_body(name):
    """The text of a top-level function of app.js: from its header to the
    closing brace in column 0."""
    m = re.search(r'^(?:async )?function %s\(' % re.escape(name), src, re.M)
    if not m:
        return ''
    end = re.search(r'^\}', src[m.start():], re.M)
    return src[m.start():m.start() + end.end()] if end else ''


_ec = _fn_body('emitChannels')
check("the_channels_event_carries_thru_fext_next_and_cause",
      re.search(r"emit\('channels'", _ec)
      and re.search(r"thru:\s*\$\('#nlThru'\)\.value", _ec)
      and re.search(r"fext:\s*selected\(\$\('#nlFext'\)\)", _ec)
      and re.search(r"next:\s*selected\(\$\('#nlNext'\)\)", _ec)
      and re.search(r"\{\s*cause\s*\}", _ec),
      "emitChannels() does not send sicopr:channels with the THRU, the FEXT "
      "and NEXT lists as the Run tab holds them, and the cause")
check("the_channels_event_is_sent_once_at_start",
      re.search(r"emitChannels\('boot'\)", _fn_body('boot')),
      "boot() does not send sicopr:channels, so a panel that starts with the "
      "page does not know the selection until something changes")
check("the_channels_event_is_sent_when_the_lists_are_filled",
      re.search(r"emitChannels\(want \? 'panel' : 'list'\)", _fn_body('fillChannelSelects')),
      "fillChannelSelects() changes what is selected (a directory, the filter, "
      "the first THRU by default) without sending sicopr:channels")
_picks = re.search(
    r"for \(const id of \[([^\]]*)\]\) \{\s*\$\(id\)\.addEventListener\("
    r"'change', \(\) => emitChannels\('user'\)\);", src)
check("the_channels_event_is_sent_on_every_pick",
      _picks and set(re.findall(r"'(#\w+)'", _picks.group(1)))
      == {'#nlThru', '#nlFext', '#nlNext', '#nlMat', '#nlEye'},
      "a change of #nlThru, #nlFext, #nlNext or of the two option boxes "
      "(#nlMat, #nlEye) does not send sicopr:channels "
      "(listeners found for %r)"
      % (_picks and re.findall(r"'(#\w+)'", _picks.group(1)),))
check("an_unchanged_selection_is_not_sent_again",
      re.search(r"cause !== 'boot' && now === CHANNELS_SENT\) return", _ec)
      and re.search(r"CHANNELS_SENT = now", _ec),
      "emitChannels() sends the same selection again; the filter checkbox and "
      "a refilled list would then look like a change to a panel")
# With no panel installed the event has no listener. It must then cost
# nothing: no request, no change on the page.
check("the_channels_event_makes_no_request_and_changes_nothing",
      _ec and not re.search(r"\bapi\(|\bfetch\(|textContent|\.hidden\b|"
                            r"\.value\s*=[^=]", _ec),
      "emitChannels() does more than dispatch an event, so the page with no "
      "panel installed is no longer what it was")
_row = re.search(r'^\| `sicopr:channels` \|([^|]*)\|([^\n]*)$', _readme, re.M)
check("the_readme_gives_the_channels_event_its_detail_and_causes",
      _row and all(w in _row.group(1) for w in ('thru', 'fext', 'next', 'cause'))
      and all(('`%s`' % c) in _row.group(2) for c in ('boot', 'user', 'list')),
      "gui/README.md does not list sicopr:channels with {thru, fext, next, "
      "cause} and the three causes")



# ---- panels: the Run tab assembled by a panel (sicopr:set-run, sicopr:run-request)
#
# A panel that drives the engine its own way can put the setup it will run in
# the Run tab: the files, the two options, its own command and a note. The Run
# tab's controls stay live, each change goes back in sicopr:channels (now with
# the options), and while it is assembled Run hands the run to the panel. With
# no panel, or one that never assembles, nothing of this is reached.
check("the_channels_event_carries_the_two_options",
      re.search(r"options:\s*\{\s*export_mat:\s*\$\('#nlMat'\)\.checked,\s*"
                r"eye_under_mlse:\s*\$\('#nlEye'\)\.checked", _ec),
      "emitChannels() does not send the Output dynamic results and "
      "--eye-under-mlse boxes as options {export_mat, eye_under_mlse}")
_sr = _fn_body('setRun')
check("set_run_selects_the_files_and_sets_the_options",
      re.search(r"\$\('#nlMat'\)\.checked = !!\(d\.options \|\| \{\}\)\.export_mat", _sr)
      and re.search(r"\$\('#nlEye'\)\.checked = !!\(d\.options \|\| \{\}\)\.eye_under_mlse", _sr)
      and re.search(r"fillChannelSelects\(await api\('/api/channels\?dir='[^;]*"
                    r"\{ thru: d\.thru, fext: d\.fext \|\| \[\], next: d\.next \|\| \[\] \}\)",
                    _sr, re.S),
      "setRun() does not select the panel's THRU, FEXT and NEXT files and set "
      "the two option boxes from sicopr:set-run")
check("set_run_is_handled",
      re.search(r"addEventListener\('sicopr:set-run', \(e\) => \{\s*setRun\(e\.detail\)", src),
      "the page does not hand sicopr:set-run to setRun()")
check("only_set_run_assembles_the_run_tab",
      re.findall(r"\bASSEMBLED\s*=[^=]", src) == ['ASSEMBLED = ', 'ASSEMBLED = ']
      and re.search(r"^let ASSEMBLED = null;", src, re.M)
      and re.search(r"ASSEMBLED = d && typeof d\.command === 'string' \? d : null;", _sr),
      "ASSEMBLED is set somewhere other than its declaration and setRun(), so "
      "the Run tab could be assembled with no panel having asked")
_rn = _fn_body('refreshNetlist')
check("an_assembled_run_tab_shows_the_panels_command",
      re.search(r"\$\('#nlCmd'\)\.textContent = ASSEMBLED \? ASSEMBLED\.command\s*"
                r": \(r\.command \|\| '\(not runnable yet\)'\);", _rn),
      "refreshNetlist() shows the editor's own command while a panel has the "
      "Run tab assembled, or no longer shows it when none has")
check("the_note_of_set_run_replaces_the_overlay_run_note",
      re.search(r"ASSEMBLED \? \(ASSEMBLED\.note \|\| ''\)\s*: \(OVERLAY \? \(OVERLAY\.run_note \|\| ''\) : ''\)",
                _fn_body('showRunNote'))
      and 'showRunNote();' in _sr,
      "the line under Configuration does not show the note of sicopr:set-run")
check("set_run_changes_are_sent_with_cause_panel",
      re.search(r"emitChannels\('panel'\)", _sr)
      and re.search(r"emitChannels\(want \? 'panel' : 'list'\)", _fn_body('fillChannelSelects')),
      "what sicopr:set-run selected is not sent back with cause panel, so a "
      "panel could not tell its own change from the user's")
_ex = re.search(r"\$\('#btnExec'\)\.addEventListener\('click', async \(\) => \{(.*?)\n\}\);",
                src, re.S)
_ex = _ex.group(1) if _ex else ''
check("run_hands_an_assembled_run_tab_to_the_panel",
      re.search(r"if \(ASSEMBLED && !emit\('run-request', \{\s*thru: \$\('#nlThru'\)\.value, "
                r"fext: selected\(\$\('#nlFext'\)\),\s*next: selected\(\$\('#nlNext'\)\),\s*"
                r"options: \{ export_mat: \$\('#nlMat'\)\.checked,\s*"
                r"eye_under_mlse: \$\('#nlEye'\)\.checked \},\s*\}, true\)\) return;", _ex),
      "Run sicopr.py does not send sicopr:run-request (cancelable, with the "
      "files and options) first when the Run tab is assembled")
check("run_runs_as_before_when_not_taken",
      _ex.find("emit('run-request'") < _ex.find("api('/api/exec/start'")
      and _ex.count('return;') >= 1
      and re.search(r"return window\.dispatchEvent\(new CustomEvent\('sicopr:' \+ name,\s*"
                    r"\{ detail, cancelable: !!cancelable \}\)\);", _fn_body('emit')),
      "Run does not fall through to /api/exec/start when no panel takes the "
      "run, or emit() does not report whether a cancelable event was taken")
check("show_can_ask_for_the_dashboard_to_be_built",
      re.search(r"WANT_BUILD = d\.build && d\.run \? d\.run : null;", src)
      and re.search(r"if \(WANT_BUILD && samePath\(path, WANT_BUILD\)\) \{\s*WANT_BUILD = null;\s*"
                    r"if \(!first && d\.rscript\)", _fn_body('dynLoad')),
      "sicopr:show with build does not build the run's dashboard in Results, "
      "dynamic when it has none")
_rows = {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(
    r'^\| `sicopr:([a-z-]+)` \|([^|]*)\|([^\n]*)$', _readme, re.M)}
check("the_readme_gives_set_run_run_request_and_the_new_fields",
      all(w in _rows.get('set-run', ('', ''))[0] for w in
          ('config', 'thru', 'fext', 'next', 'export_mat', 'eye_under_mlse', 'command', 'note'))
      and all(w in _rows.get('run-request', ('', ''))[0] for w in ('thru', 'fext', 'next', 'options'))
      and 'preventDefault()' in _rows.get('run-request', ('', ''))[1]
      and 'options' in _rows.get('channels', ('', ''))[0] and '`panel`' in _rows.get('channels', ('', ''))[1]
      and 'build' in _rows.get('show', ('', ''))[0],
      "gui/README.md does not give sicopr:set-run, sicopr:run-request, the "
      "options and the panel cause of sicopr:channels, and build of sicopr:show")
check("the_assembled_command_style_needs_no_hidden_rule",
      re.search(r"\.cmd\.assembled\s*\{[^}]*border-left", css)
      and not re.search(r"\.cmd\.assembled\s*\{[^}]*display", css),
      ".cmd.assembled sets display, which would need its own [hidden] rule")


# ---- panels: a panel's run in the Run tab's Output (sicopr:output) -------
#
# A panel that runs the engine its own way shows that run where a run of the
# Run tab is shown: the Output terminal (bounded, follow output), the status
# line and Stop. Stop then asks the panel, because the editor's runner is not
# the one running. With no panel nothing sends the event, PANEL_OUT stays
# null, and the Run tab is exactly what it was.
_po = _fn_body('panelOutput')
check("output_is_handled",
      re.search(r"addEventListener\('sicopr:output', \(e\) => panelOutput\(e\.detail\)\)", src),
      "the page does not hand sicopr:output to panelOutput()")
check("output_shows_a_panel_run_in_the_terminal",
      re.search(r"if \(!PANEL_OUT \|\| PANEL_OUT\.job !== job\) \{\s*clearTimeout\(POLL\);[^}]*"
                r"termReset\(''\);\s*\}", _po)
      and re.search(r"termAppend\(\(d\.lines \|\| \[\]\)\.map\(String\)\);", _po)
      and re.search(r"\$\('#execState'\)\.textContent = String\(d\.status \|\| ''\);", _po),
      "panelOutput() does not clear the Output for a new job, append its lines "
      "to the bounded terminal, and show its status in the status line")
check("output_keeps_stop_live_until_done",
      re.search(r"\$\('#btnKill'\)\.disabled = PANEL_OUT\.done;", _po)
      and re.search(r"if \(PANEL_OUT\.done\) refreshNetlist\(\);\s*else \$\('#btnExec'\)\.disabled = true;",
                    _po),
      "panelOutput() does not keep Stop live (and Run off) while the panel's "
      "run is under way, or does not give Run back when it is done")
_kill = re.search(r"\$\('#btnKill'\)\.addEventListener\('click', \(\) => \{(.*?)\n\}\);", src, re.S)
_kill = _kill.group(1) if _kill else ''
check("stop_asks_the_panel_for_its_run",
      re.search(r"if \(PANEL_OUT && !PANEL_OUT\.done\) \{\s*emit\('stop-request', \{ job: PANEL_OUT\.job \}\);"
                r"\s*return;\s*\}", _kill)
      and _kill.find("emit('stop-request'") < _kill.find("api('/api/exec/stop'"),
      "Stop does not send sicopr:stop-request while the Output shows a panel's "
      "run, or still asks the editor's own runner first")
check("only_output_sets_the_panel_run",
      re.findall(r"\bPANEL_OUT\s*=[^=]", src) == ['PANEL_OUT = ', 'PANEL_OUT = ', 'PANEL_OUT = ']
      and re.search(r"^let PANEL_OUT = null;", src, re.M)
      and re.search(r"PANEL_OUT = \{ job, done: !!d\.done \};", _po)
      and re.search(r"PANEL_OUT = null;\s*// the Output is this run's now\s*termReset\(''\);", _ex),
      "PANEL_OUT is set somewhere other than its declaration, panelOutput() "
      "and a run of the Run tab, so the Output could change hands with no "
      "panel having asked")
check("the_run_tab_does_not_poll_over_a_panel_run",
      re.search(r"if \(!PANEL_OUT\) pollStatus\(\);", _fn_body('runEnter')),
      "runEnter() polls the editor's runner while the Output shows a panel's "
      "run, and its 'idle' would overwrite the panel's status and Stop")
_sh = re.search(r"addEventListener\('sicopr:show', \(e\) => \{(.*?)\n\}\);", src, re.S)
_sh = _sh.group(1) if _sh else ''
check("show_can_leave_an_edit_in_config_alone",
      re.search(r"if \(d\.unless_editing && VIEW === 'config' && \(Object\.keys\(EDITS\)\.length\s*"
                r"\|\| \$\('#configView'\)\.contains\(document\.activeElement\)\)\) return;", _sh)
      and _sh.find('unless_editing') < _sh.find('show(d.view)'),
      "sicopr:show with unless_editing switches tab while the user is editing "
      "in Config")
check("the_readme_gives_output_stop_request_and_unless_editing",
      all(w in _rows.get('output', ('', ''))[0] for w in ('job', 'lines', 'status', 'done'))
      and 'job' in _rows.get('stop-request', ('', ''))[0]
      and 'unless_editing' in _rows.get('show', ('', ''))[0]
      and 'sicopr:output' in _readme.split('## Tests')[0].split('With no panel installed nothing sends')[-1],
      "gui/README.md does not give sicopr:output, sicopr:stop-request and "
      "unless_editing of sicopr:show, and that with no panel nothing sends "
      "sicopr:output")


finish()
