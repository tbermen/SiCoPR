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

# 'use strict' at the top means a stray assignment to an undeclared name is a
# runtime error rather than a silent global.
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

finish()
