/* Copyright 2026 Todd Bermensolo
 * SPDX-License-Identifier: BSD-3-Clause */
'use strict';
/* SiCoPR config editor.
 *
 * The one rule that matters here: a value the workbook stores as a STRING is a
 * MATLAB sweep range, and a value it stores as a NUMBER is a fixed setting.
 * The editor keeps them apart and never coerces one into the other, because
 * quietly turning "[-0.34:.02:0]" into -0.34 shrinks a ~198-point Tx FFE search
 * to a single point and the run still looks perfectly healthy.
 */

const $ = (s) => document.querySelector(s);

let CFG = null;        // current /api/config payload
let SEL = null;        // selected block id
let EDITS = {};        // keyword -> new value (typed, not raw text)

/* ---------------------------------------------------------------- helpers */

function toast(msg, bad) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.toggle('bad', !!bad);
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, bad ? 7000 : 3200);
}

async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({ error: 'bad response from server' }));
  if (j && j.error) throw new Error(j.error);
  return j;
}

/* A cell's value keeps the type the workbook gave it. Text that parses cleanly
 * as a number becomes a number; anything else stays a string. `1e-4` counts as
 * a number, `[0 0]` and `[-0.34:.02:0]` do not. */
function typed(text) {
  const s = String(text).trim();
  if (s === '') return '';
  if (/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(s)) return Number(s);
  return s;
}

function sameValue(a, b) {
  if (typeof a === 'number' && typeof b === 'number') return a === b;
  return String(a) === String(b);
}

function fmt(v) {
  return v === null || v === undefined ? '' : String(v);
}

function dirtyBlocks() {
  const out = new Set();
  if (!CFG) return out;
  for (const b of CFG.blocks) {
    if (b.cells.some((c) => c.key in EDITS)) out.add(b.id);
  }
  return out;
}

/* -------------------------------------------------------------- schematic */

const LANE_ORDER = ['signal', 'agg', 'eq', 'ctl'];
const LANE_LABEL = {
  signal: 'signal path', agg: 'aggressors',
  eq: 'receiver equalisation', ctl: 'analysis and run control',
};

/* The viewBox is computed from the lanes that actually have blocks, rather
 * than fixed. A hard-coded height cropped the bottom lane the moment the
 * block set changed, and an SVG that silently hides a row of controls is a
 * bad way to find out. */
function drawSchematic() {
  const svg = $('#svg');
  svg.textContent = '';
  const NS = 'http://www.w3.org/2000/svg';
  const el = (n, a, txt) => {
    const e = document.createElementNS(NS, n);
    for (const k in a) e.setAttribute(k, a[k]);
    if (txt !== undefined) e.textContent = txt;
    return e;
  };

  const lanes = {};
  for (const b of CFG.blocks) (lanes[b.row] = lanes[b.row] || []).push(b);

  const dirty = dirtyBlocks();
  const W = 1000, PAD = 24, BH = 58, LANE_GAP = 46, TOP = 34;

  const used = LANE_ORDER.filter((r) => (lanes[r] || []).length);
  const laneY = {};
  let y0 = TOP;
  for (const r of used) {
    laneY[r] = y0;
    y0 += BH + LANE_GAP;
  }
  const H = y0 - LANE_GAP + PAD;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  for (const row of used) {
    const list = lanes[row];
    const y = laneY[row];
    svg.appendChild(el('text', { x: PAD, y: y - 12, class: 'lane' },
      LANE_LABEL[row]));

    const n = list.length;
    const gap = 14;
    const bw = Math.min(168, (W - PAD * 2 - gap * (n - 1)) / n);
    const bh = BH;
    const total = bw * n + gap * (n - 1);
    let x = (W - total) / 2;

    list.forEach((b, i) => {
      const g = el('g', {
        class: 'blk' + (b.id === SEL ? ' sel' : '') +
          (b.cells.length ? '' : ' empty') +
          (dirty.has(b.id) ? ' dirty' : ''),
        tabindex: '0', role: 'button',
      });
      g.appendChild(el('rect', { x, y, width: bw, height: bh }));
      // label, wrapped to two lines if needed
      const words = b.label.split(' ');
      const l1 = words.length > 2 ? words.slice(0, 2).join(' ') : b.label;
      const l2 = words.length > 2 ? words.slice(2).join(' ') : '';
      g.appendChild(el('text', { x: x + bw / 2, y: y + (l2 ? 21 : 26) }, l1));
      if (l2) g.appendChild(el('text', { x: x + bw / 2, y: y + 35 }, l2));
      g.appendChild(el('text', {
        x: x + bw / 2, y: y + bh - 8, class: 'n',
      }, b.cells.length + (b.cells.length === 1 ? ' setting' : ' settings')));
      g.addEventListener('click', () => select(b.id));
      g.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(b.id); }
      });
      svg.appendChild(g);

      // wire to the next block in the same lane
      if (i < n - 1) {
        const x2 = x + bw;
        svg.appendChild(el('path', {
          class: 'wire',
          d: `M${x2} ${y + bh / 2} H${x2 + gap}`,
        }));
      }
      b._x = x; b._y = y; b._w = bw; b._h = bh;
      x += bw + gap;
    });
  }

  // aggressors couple into the channel; EQ hangs off the Rx die
  const ch = CFG.blocks.find((b) => b.id === 'channel');
  const rx = CFG.blocks.find((b) => b.id === 'rx_die');
  const fx = CFG.blocks.find((b) => b.id === 'fext');
  const eq0 = (lanes.eq || [])[0];
  const mid = (b) => b._x + b._w / 2;
  if (ch && fx) {
    svg.appendChild(el('path', {
      class: 'wire dash',
      d: `M${mid(fx)} ${fx._y} V${ch._y + ch._h + 12} H${mid(ch)} V${ch._y + ch._h}`,
    }));
  }
  if (rx && eq0) {
    svg.appendChild(el('path', {
      class: 'wire',
      d: `M${mid(rx)} ${rx._y + rx._h} V${eq0._y - 14} H${mid(eq0)} V${eq0._y}`,
    }));
  }
}

/* ----------------------------------------------------------------- panel */

function select(id) {
  SEL = id;
  drawSchematic();
  renderPanel();
}

function renderPanel() {
  const b = CFG.blocks.find((x) => x.id === SEL);
  const host = $('#cells');
  host.textContent = '';
  if (!b) return;

  $('#panelTitle').textContent = b.label;
  const nref = b.cells.filter((c) => !c.engine_read).length;
  $('#panelSub').textContent =
    b.cells.length + ' field' + (b.cells.length === 1 ? '' : 's')
    + ' from ' + CFG.name
    + (nref ? `  ·  ${nref} reference-only (not read by the engine)` : '');

  for (const c of b.cells) {
    // A [TX RX] cell is one spreadsheet cell shown under both ends of the
    // link; each field edits its own half and the server recombines on save.
    const base = c.half ? c.half_value : c.value;

    const wrap = document.createElement('div');
    wrap.className = 'cell';

    const top = document.createElement('div');
    top.className = 'top';
    const nm = document.createElement('span');
    nm.className = 'nm';
    nm.textContent = c.name + (c.half ? '  ' + c.half.toUpperCase() : '');
    top.appendChild(nm);
    if (c.units) {
      const u = document.createElement('span');
      u.className = 'un';
      u.textContent = c.units;
      top.appendChild(u);
    }
    if (c.is_sweep) top.appendChild(tag('sweep', 'sweep range'));
    if (c.is_formula) top.appendChild(tag('formula', 'formula'));
    if (c.half) top.appendChild(tag('pair', '[TX RX] pair'));
    if (!c.engine_read) top.appendChild(tag('ref', 'reference only'));
    if (c.key in EDITS) top.appendChild(tag('dirty', 'changed'));
    wrap.appendChild(top);

    const inp = document.createElement('input');
    inp.type = 'text';
    inp.spellcheck = false;
    inp.value = fmt(c.key in EDITS ? EDITS[c.key] : base);
    // A cell the engine never reads is context, not configuration; editing it
    // would change the workbook without changing the run.
    inp.disabled = !c.engine_read;
    inp.addEventListener('input', () => {
      const v = typed(inp.value);
      if (sameValue(v, base)) delete EDITS[c.key];
      else EDITS[c.key] = v;
      wrap.classList.toggle('changed', c.key in EDITS);
      refreshDirty();
      drawSchematic();
    });
    wrap.classList.toggle('changed', c.key in EDITS);
    wrap.appendChild(inp);

    // The workbook's own annotation column: units templates like
    // `[min:step:max]` and pair markers like `[TX RX]`. This is the text that
    // used to sit in a separate "not read by the engine" block, where it told
    // the reader nothing because it was away from the field it describes.
    const bits = [];
    if (c.note) bits.push(c.note);
    if (c.info) bits.push(c.info);
    if (c.half) bits.push('one half of ' + c.name + ' = ' + fmt(c.value));
    if (!c.engine_read) bits.push('not read by the engine — reference value');
    if (c.is_formula) bits.push('spreadsheet formula ' + c.formula
      + ' — replacing it writes a fixed number');
    if (c.default !== null && c.default !== undefined && c.default !== '')
      bits.push('default ' + c.default);
    if (bits.length) {
      const i = document.createElement('div');
      i.className = 'info';
      i.textContent = bits.join(' · ');
      wrap.appendChild(i);
    }
    const co = document.createElement('div');
    co.className = 'coord';
    co.textContent = c.coord + (c.matlab ? '  ·  ' + c.matlab : '');
    wrap.appendChild(co);

    host.appendChild(wrap);
  }
}

function tag(cls, text) {
  const s = document.createElement('span');
  s.className = 'tag ' + cls;
  s.textContent = text;
  return s;
}

function refreshDirty() {
  const n = Object.keys(EDITS).length;
  $('#dirtyBar').hidden = n === 0;
  $('#dirtyCount').textContent = n + ' unsaved change' + (n === 1 ? '' : 's');
  $('#btnSave').disabled = n === 0;
}

/* ------------------------------------------------------------------ load */

async function loadConfig(path) {
  CFG = await api('/api/config?path=' + encodeURIComponent(path));
  EDITS = {};
  SEL = SEL && CFG.blocks.some((b) => b.id === SEL) ? SEL : 'channel';
  const c = CFG.counts;
  $('#counts').textContent =
    `${c.settings} settings · ${c.annotations} annotations `
    + `(${c.annotations_folded} shown in context) · `
    + `${Object.keys(CFG.packages).length} package blocks`;
  refreshDirty();
  drawSchematic();
  renderPanel();
}

async function fillConfigList(dir) {
  const list = await api('/api/configs' + (dir ? '?dir=' + encodeURIComponent(dir) : ''));
  const pick = $('#configPick');
  pick.textContent = '';
  for (const c of list) {
    const o = document.createElement('option');
    o.value = c.path;
    o.textContent = c.name;
    pick.appendChild(o);
  }
  return list;
}

/* ---- directory browser -------------------------------------------------
 *
 * A page cannot open a native folder picker for a path on the SERVER, and the
 * server is where everything is read. So the browsing is server-side: ask what
 * is under a path, show it, descend. The counts beside each directory -- of
 * configs, .s4p files and run directories -- are the point: they say where to
 * go without opening anything.
 *
 * One dialog serves every tab. What differs is what happens on "Use": which
 * picker gets refilled, or, for the Run tab's working directory, which server
 * setting changes. Opening a directory trusts it for the rest of the session
 * and for every tab at once. Nothing is written to disk; restarting the
 * editor forgets it.
 */
let BR_AT = '';
let BR_FOR = 'config';   // config | channel | results | rundir

const BR_TITLES = {
  config: 'Choose a directory holding configuration workbooks',
  channel: 'Choose a directory holding Touchstone (.s4p) channels',
  results: 'Choose a directory holding run output',
  rundir: 'Choose the working directory for runs',
};

function fillDirPick(dirs, dsel) {
  dsel.textContent = '';
  for (const d of dirs) {
    const o = document.createElement('option');
    o.value = d.dir;
    o.textContent = `${d.dir}  (${d.count})`;
    dsel.appendChild(o);
  }
}

function brOpen(purpose) {
  BR_FOR = purpose;
  $('#brTitle').textContent = BR_TITLES[purpose];
  $('#browseBack').hidden = false;
  brShow('').catch((e) => toast(e.message, true));
}

function brCounts(e) {
  const bits = [];
  if (e.configs) bits.push(`${e.configs} config${e.configs === 1 ? '' : 's'}`);
  if (e.s4p) bits.push(`${e.s4p} .s4p`);
  if (e.runs) bits.push(`${e.runs} run${e.runs === 1 ? '' : 's'}`);
  return bits.join(' · ');
}

async function brShow(path) {
  const d = await api(`/api/browse?path=${encodeURIComponent(path || '')}`);
  const err = $('#brErr');
  err.hidden = !d.error;
  err.textContent = d.error || '';
  if (d.error) return;
  BR_AT = d.path;
  const here = brCounts(d);
  $('#brPath').textContent = d.places
    ? 'Start from one of these'
    : `${d.path}    (${here || 'nothing of interest directly'} here)`;
  $('#brUp').disabled = !d.parent;
  // Any real directory may be opened: what is under it is searched
  // recursively, so a parent that shows nothing "here" can still be the
  // right choice.
  $('#brUse').disabled = d.places;
  const ul = $('#brList');
  ul.textContent = '';
  if (!d.dirs.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = 'no subdirectories';
    ul.appendChild(li);
    return;
  }
  for (const e of d.dirs) {
    const li = document.createElement('li');
    const n = document.createElement('span');
    n.className = 'n';
    n.textContent = e.name;
    li.appendChild(n);
    const counts = brCounts(e);
    if (counts) {
      const c = document.createElement('span');
      c.className = 'c';
      c.textContent = counts;
      li.appendChild(c);
    }
    li.addEventListener('click', () => brShow(e.path)
      .catch((x) => toast(x.message, true)));
    ul.appendChild(li);
  }
}

function brClose() { $('#browseBack').hidden = true; }

/* Choosing is what grants access; browsing to it did not. What happens next
 * depends on which tab asked. */
async function brUse() {
  const chosen = BR_AT;
  brClose();
  if (BR_FOR === 'rundir') {
    const r = await api(`/api/run_dir?path=${encodeURIComponent(chosen)}`);
    toast('runs will execute in ' + r.dir);
    await refreshNetlist();
    return;
  }
  await api(`/api/open?path=${encodeURIComponent(chosen)}`);
  if (BR_FOR === 'channel') {
    const dirs = await fillChannelDirs();
    // prefer a directory under what was just opened, else keep the first
    const under = dirs.find((d) => d.dir.startsWith(chosen));
    if (!under) { toast('no .s4p files under that directory', true); return; }
    for (const id of ['#spDir', '#nlDir']) $(id).value = under.dir;
    await spLoadDir(under.dir);
    fillChannelSelects(await api('/api/channels?dir='
      + encodeURIComponent(under.dir)));
    return;
  }
  if (BR_FOR === 'results') {
    resultsEnter();
    dynEnter();
    return;
  }
  const dsel = $('#dirPick');
  fillDirPick(await api('/api/config_dirs'), dsel);
  const match = [...dsel.options].find((o) => o.value === chosen);
  if (!match) {
    const o = document.createElement('option');
    o.value = chosen;
    o.textContent = chosen;
    dsel.appendChild(o);
  }
  dsel.value = chosen;
  const l = await fillConfigList(chosen);
  if (l.length) await loadConfig(l[0].path);
  else toast('no configuration workbooks in that directory', true);
}

async function boot() {
  const dirs = await api('/api/config_dirs');
  const dsel = $('#dirPick');
  fillDirPick(dirs, dsel);

  $('#btnBrowse').addEventListener('click', () => brOpen('config'));
  $('#spBrowse').addEventListener('click', () => brOpen('channel'));
  $('#nlBrowse').addEventListener('click', () => brOpen('channel'));
  $('#nlCwdBrowse').addEventListener('click', () => brOpen('rundir'));
  $('#btnResBrowse').addEventListener('click', () => brOpen('results'));
  $('#btnDynBrowse').addEventListener('click', () => brOpen('results'));
  $('#brUp').addEventListener('click', () => {
    const parent = BR_AT ? BR_AT.replace(/[\\/][^\\/]+[\\/]?$/, '') : '';
    brShow(parent && parent !== BR_AT ? parent : '')
      .catch((e) => toast(e.message, true));
  });
  $('#brCancel').addEventListener('click', brClose);
  $('#brUse').addEventListener('click', () => brUse()
    .catch((e) => toast(e.message, true)));
  $('#browseBack').addEventListener('click', (ev) => {
    if (ev.target === $('#browseBack')) brClose();
  });
  dsel.addEventListener('change', async () => {
    if (Object.keys(EDITS).length
      && !confirm('Discard unsaved changes and change directory?')) {
      dsel.value = CFG.path.split('/').slice(0, -1).join('/');
      return;
    }
    const l = await fillConfigList(dsel.value);
    if (l.length) await loadConfig(l[0].path);
  });

  const list = await fillConfigList(dirs.length ? dirs[0].dir : null);
  const pick = $('#configPick');
  pick.addEventListener('change', () => {
    if (Object.keys(EDITS).length
      && !confirm('Discard unsaved changes and load another config?')) {
      pick.value = CFG.path;
      return;
    }
    loadConfig(pick.value).catch((e) => toast(e.message, true));
  });
  if (list.length) await loadConfig(list[0].path);
  else toast('no configuration workbooks found \u2014 use Browse to point at '
             + 'a directory that has some', true);
}

/* ------------------------------------------------------------ new config */

$('#btnNew').addEventListener('click', async () => {
  const sel = $('#newTemplate');
  sel.textContent = '';
  for (const o of $('#configPick').options) {
    const t = document.createElement('option');
    t.value = o.value;
    t.textContent = o.textContent;
    sel.appendChild(t);
  }
  if (CFG) sel.value = CFG.path;
  const dir = CFG ? CFG.path.split('/').slice(0, -1).join('/') : '';
  $('#newPath').value = (dir ? dir + '/' : '') + 'config_new.xlsx';
  $('#newDlg').showModal();
});

$('#newGo').addEventListener('click', (e) => {
  e.preventDefault();
  api('/api/new_config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      template: $('#newTemplate').value,
      dst: $('#newPath').value.trim(),
      reset_defaults: $('#newReset').checked,
    }),
  }).then(async (r) => {
    $('#newDlg').close();
    toast(`created ${r.dst} — ${r.reset} setting(s) reset, `
      + `${r.no_default} had no declared default`);
    await fillConfigList($('#dirPick').value);
    const pick = $('#configPick');
    if (!Array.from(pick.options).some((o) => o.value === r.dst)) {
      const o = document.createElement('option');
      o.value = r.dst;
      o.textContent = r.dst.split('/').pop();
      pick.appendChild(o);
    }
    pick.value = r.dst;
    await loadConfig(r.dst);
    if (r.warnings && r.warnings.length) {
      alert('Created, with warnings:\n\n' + r.warnings.join('\n\n'));
    }
  }).catch((err) => toast(err.message, true));
});

/* ------------------------------------------------------------------ save */

$('#btnSave').addEventListener('click', () => {
  const base = CFG.path.replace(/\.xlsx$/i, '');
  $('#savePath').value = base + '_edited.xlsx';
  const host = $('#saveChanges');
  host.textContent = '';
  for (const k of Object.keys(EDITS).sort()) {
    const cell = CFG.blocks.flatMap((b) => b.cells).find((c) => c.key === k);
    const before = cell ? (cell.half ? cell.half_value : cell.value) : '';
    const d = document.createElement('div');
    d.className = 'ch';
    const b1 = document.createElement('b');
    b1.textContent = k;
    d.appendChild(b1);
    d.appendChild(document.createTextNode(
      `${fmt(before)}  →  ${fmt(EDITS[k])}`));
    host.appendChild(d);
  }
  $('#saveDlg').showModal();
});

$('#saveGo').addEventListener('click', (e) => {
  e.preventDefault();
  const dst = $('#savePath').value.trim();
  if (!dst) { toast('give the new config a path', true); return; }
  api('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      src: CFG.path, dst, changes: EDITS,
      overwrite: $('#saveOverwrite').checked,
    }),
  }).then((r) => {
    $('#saveDlg').close();
    toast(`wrote ${r.dst} (${r.changed} change${r.changed === 1 ? '' : 's'})`);
    if (r.warnings && r.warnings.length) {
      alert('Written, with warnings:\n\n' + r.warnings.join('\n\n'));
    }
    return api('/api/configs').then((list) => {
      const pick = $('#configPick');
      if (!Array.from(pick.options).some((o) => o.value === r.dst)) {
        const o = document.createElement('option');
        o.value = r.dst;
        o.textContent = r.dst.split('/').pop();
        pick.appendChild(o);
      }
      pick.value = r.dst;
      return loadConfig(r.dst);
    });
  }).catch((err) => toast(err.message, true));
});

$('#btnRevert').addEventListener('click', () => {
  EDITS = {};
  refreshDirty();
  drawSchematic();
  renderPanel();
});

/* ------------------------------------------------------------ navigation */

/* One tab is visible at a time. Each operation owns its own tab: the config is
 * chosen and edited in Config only, so the Run tab is about running and cannot
 * quietly change what is being run. */
const VIEWS = {
  config: '#configView',
  sparam: '#sparamView',
  run: '#runView',
  results: '#resultsView',
  dynamic: '#dynamicView',
};
let VIEW = 'config';

function show(which) {
  if (!VIEWS[which]) return;
  VIEW = which;
  for (const [name, sel] of Object.entries(VIEWS)) {
    $(sel).hidden = name !== which;
  }
  for (const b of document.querySelectorAll('#tabs .tab')) {
    const on = b.dataset.view === which;
    b.classList.toggle('on', on);
    b.setAttribute('aria-selected', on ? 'true' : 'false');
  }
  if (which === 'sparam') spEnter();
  if (which === 'run') runEnter();
  if (which === 'results') resultsEnter();
  if (which === 'dynamic') dynEnter();
}

/* The full figure set com_plots can produce, so a run that made fewer can say
 * which are absent instead of leaving the reader to wonder. */
const EXPECTED_FIGS = [
  's1_filters.png', 's1_insertion_loss.png', 's1_return_loss.png',
  's2_erl_summary.png', 's2_tdr_impedance.png',
  's3_sbr_full.png', 's3_sbr_zoom.png',
  's4_fom_vs_phase.png',
  's5_eq_taps.png', 's5_eq_vs_uneq_sbr.png', 's5_fom_convergence.png',
  's6_contribution_pie.png', 's6_noise_terms.png', 's6_pdfs.png',
  's7_eye_contour.png', 's7_timing_bathtub.png', 's7_voltage_bathtub.png',
];

const showEditor = () => show('config');
const showResults = () => show('results');

for (const b of document.querySelectorAll('#tabs .tab')) {
  b.addEventListener('click', () => show(b.dataset.view));
}
$('#btnToConfig').addEventListener('click', () => show('config'));

/* ------------------------------------------------------------ run: netlist */

let POLL = null;         // status poll timer
let NEXT_LINE = 0;       // how many output lines we have consumed

function selected(sel) {
  return Array.from(sel.selectedOptions).map((o) => o.value);
}

function netlistBody() {
  return {
    config: CFG ? CFG.path : '',
    thru: $('#nlThru').value,
    fext: selected($('#nlFext')),
    next: selected($('#nlNext')),
    export_mat: $('#nlMat').checked,
    eye_under_mlse: $('#nlEye').checked,
    matlab_version: $('#nlVer').value,
  };
}

async function refreshNetlist() {
  const r = await api('/api/netlist', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(netlistBody()),
  }).catch((e) => ({ ok: false, problems: [e.message], command: '' }));

  $('#nlCmd').textContent = r.command || '(not runnable yet)';
  if (r.cwd) $('#nlCwd').textContent = r.cwd;
  const host = $('#nlProblems');
  host.textContent = '';
  for (const p of r.problems || []) {
    const d = document.createElement('div');
    d.className = 'problem';
    d.textContent = p;
    host.appendChild(d);
  }
  // Never enable Run while something is already going: the server refuses a
  // second run anyway, and a button that looks live but is not is worse.
  $('#btnExec').disabled = !r.ok || $('#btnKill').disabled === false;
}

let CHANNELS = [];

/* With the filter on, each role lists only the files whose name matches it --
 * 56 files in one directory is unusable otherwise. The filter is a checkbox
 * and not a rule: a filename is a hint, so overriding it stays possible, and
 * a file shown under a role it was not named for is labelled. */
function fillChannelSelects(files) {
  if (files) CHANNELS = files;
  const thru = $('#nlThru');
  const fx = $('#nlFext');
  const nx = $('#nlNext');
  const keepThru = thru.value;
  const keepFx = new Set(selected(fx));
  const keepNx = new Set(selected(nx));
  for (const sel of [thru, fx, nx]) sel.textContent = '';

  const blank = document.createElement('option');
  blank.value = '';
  blank.textContent = '— none —';
  thru.appendChild(blank);

  const onlyMatching = $('#nlFilter').checked;
  for (const [sel, role, keep] of [[thru, 'thru', null],
                                   [fx, 'fext', keepFx],
                                   [nx, 'next', keepNx]]) {
    let shown = 0;
    for (const f of CHANNELS) {
      const matches = f.role === role;
      // never hide something already selected, or the netlist would silently
      // stop matching what the page shows
      const isKept = keep ? keep.has(f.path) : f.path === keepThru;
      if (onlyMatching && !matches && !isKept) continue;
      const o = document.createElement('option');
      o.value = f.path;
      o.textContent = f.name + (matches ? '' : `   (${f.role})`);
      if (isKept) o.selected = true;
      sel.appendChild(o);
      shown++;
    }
    if (!shown && sel !== thru) {
      const o = document.createElement('option');
      o.disabled = true;
      o.textContent = `no ${role.toUpperCase()} files in this directory`;
      sel.appendChild(o);
    }
  }
  if (!thru.value) {
    const first = CHANNELS.find((f) => f.role === 'thru');
    if (first) thru.value = first.path;
  }
  refreshNetlist();
}

/* Both channel pickers -- S-parameters and Run -- list the same directories,
 * so one call fills both. Channels are not in the repository, so an empty
 * list is the normal first state, and the placeholder says what to do. */
const NO_S4P = 'no .s4p files in any opened directory — use Browse…';
let CHANNEL_DIRS = null;   // null until first fetched

async function fillChannelDirs() {
  const dirs = await api('/api/channel_dirs');
  CHANNEL_DIRS = dirs;
  for (const id of ['#spDir', '#nlDir']) {
    const dsel = $(id);
    const keep = dsel.value;
    dsel.textContent = '';
    if (!dirs.length) {
      const o = document.createElement('option');
      o.value = '';
      o.textContent = NO_S4P;
      dsel.appendChild(o);
    }
    for (const d of dirs) {
      const o = document.createElement('option');
      o.value = d.dir;
      o.textContent = `${d.dir}  (${d.count})`;
      dsel.appendChild(o);
    }
    if (dirs.some((d) => d.dir === keep)) dsel.value = keep;
  }
  return dirs;
}

async function runEnter() {
  try {
    $('#nlConfig').textContent = CFG ? CFG.path : '(none)';
    if (CHANNEL_DIRS === null) {
      const dirs = await fillChannelDirs();
      if (dirs.length) fillChannelSelects(await api('/api/channels?dir='
        + encodeURIComponent($('#nlDir').value)));
    }
    refreshNetlist();
    pollStatus();          // pick up a run already in progress
  } catch (e) { toast(e.message, true); }
}

$('#nlDir').addEventListener('change', async () => {
  try {
    if (!$('#nlDir').value) return;
    fillChannelSelects(await api('/api/channels?dir='
      + encodeURIComponent($('#nlDir').value)));
  } catch (e) { toast(e.message, true); }
});

for (const id of ['#nlThru', '#nlFext', '#nlNext', '#nlMat', '#nlEye', '#nlVer']) {
  $(id).addEventListener('change', refreshNetlist);
}
$('#nlFilter').addEventListener('change', () => fillChannelSelects(null));

/* ------------------------------------------------------------ run: execute */

$('#btnExec').addEventListener('click', async () => {
  try {
    const r = await api('/api/exec/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(netlistBody()),
    });
    if (r.problems) { toast(r.problems.join('; '), true); return; }
    termReset('');
    NEXT_LINE = 0;
    POLL_FAILS = 0;
    RUN_DONE = null;
    $('#btnExec').disabled = true;
    $('#btnKill').disabled = false;
    pollStatus();
  } catch (e) { toast(e.message, true); }
});

$('#btnKill').addEventListener('click', () => {
  api('/api/exec/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
    .then((r) => toast(r.stopped ? 'stop requested' : 'nothing was running'))
    .catch((e) => toast(e.message, true));
});

/* The terminal keeps only the last TERM_MAX lines.
 *
 * The first version appended with `#term.textContent += ...` and never
 * trimmed. On a full-grid run — long AND loud — that means re-reading,
 * re-concatenating and re-laying-out a string that grows into megabytes, on
 * every poll, while also forcing a reflow to scroll. The tab locks up, and the
 * run looks frozen when it is fine. Keeping a bounded array and writing it once
 * makes the cost independent of how long the run has been going. */
const TERM_MAX = 4000;
let TERM = [];
let POLL_FAILS = 0;
let RUN_DONE = null;   // `started` of the run already announced

function termReset(msg) {
  TERM = msg ? [msg] : [];
  $('#term').textContent = TERM.join('\n');
}

function termAppend(lines) {
  if (!lines.length) return;
  TERM = TERM.concat(lines);
  let dropped = 0;
  if (TERM.length > TERM_MAX) {
    dropped = TERM.length - TERM_MAX;
    TERM = TERM.slice(dropped);
  }
  const el = $('#term');
  el.textContent = TERM.join('\n') + '\n';
  if ($('#follow').checked) el.scrollTop = el.scrollHeight;
  return dropped;
}

async function pollStatus() {
  clearTimeout(POLL);
  let st;
  try {
    st = await api('/api/exec/status?since=' + NEXT_LINE);
    POLL_FAILS = 0;
  } catch (e) {
    // A failed poll must never end the loop. The original version returned
    // here, so one hiccup left the status window dead for the rest of the run
    // while sicopr.py carried on — indistinguishable from a freeze.
    POLL_FAILS += 1;
    const wait = Math.min(30000, 900 * Math.pow(2, POLL_FAILS));
    $('#execState').textContent =
      `status unavailable (${POLL_FAILS}x): ${e.message} — retrying in `
      + `${Math.round(wait / 1000)}s`;
    POLL = setTimeout(pollStatus, wait);
    return;
  }

  if (st.idle) {
    // No run on the server. If we were watching one, it is over; otherwise
    // this is just an idle page. Either way, stop cleanly rather than spin.
    $('#execState').textContent = 'idle';
    $('#btnKill').disabled = true;
    refreshNetlist();
    return;
  }

  if (st.resync) {
    termReset('[gui] output truncated — showing the tail');
  }
  termAppend(st.lines);
  NEXT_LINE = st.next;

  const behind = st.behind
    ? `  ·  ${st.behind} line${st.behind === 1 ? '' : 's'} buffered` : '';
  // A quiet run is the other thing that looks like a freeze: a full-grid
  // search can spend minutes between progress lines. Say how long it has been
  // silent, so "nothing is happening" and "nothing is being printed" are
  // distinguishable.
  const quiet = st.running && st.quiet > 20
    ? `  ·  no output for ${Math.round(st.quiet)}s` : '';
  $('#execState').textContent = st.running
    ? `running — ${st.elapsed.toFixed(0)}s${behind}${quiet}`
    : `finished (exit ${st.returncode}) in ${st.elapsed.toFixed(1)}s`;
  $('#btnKill').disabled = !st.running;

  if (st.running || st.behind) {
    // drain a backlog quickly; idle along when caught up
    POLL = setTimeout(pollStatus, st.behind ? 120 : 900);
    return;
  }

  refreshNetlist();

  // Finishing is an EVENT, not a state. This used to fire whenever a poll saw
  // a finished run -- and runEnter() polls -- so coming back to the Run tab
  // re-launched the R build and threw the user out to the dynamic tab again,
  // with no way back. `started` identifies the run; each one is announced once.
  if (RUN_DONE === st.started) return;
  RUN_DONE = st.started;

  if (st.returncode === 0) {
    if ($('#nlMat').checked) {
      toast('run finished — building the dynamic dashboard');
      autoBuildDynamic();
    } else {
      toast('run finished — see Results · static');
    }
  } else {
    toast('run exited with code ' + st.returncode, true);
  }
}

/* Run output is not in the repository either, so an empty list means "nothing
 * opened yet" at least as often as "nothing has run". */
const NO_RUNS = 'No run directories in any opened directory. Use Browse… to '
  + 'open one holding runs, or run sicopr.py with SAVE_FIGURES / CSV_REPORT '
  + 'enabled in the config.';

function resultsEnter() {
  api('/api/results').then((rows) => {
    const pick = $('#runPick');
    const keepSel = pick.value;
    pick.textContent = '';
    if (!rows.length) {
      $('#runBody').textContent = NO_RUNS;
      return;
    }
    const keep = pick.value;
    for (const r of rows) {
      const o = document.createElement('option');
      o.value = r.path;
      o.textContent = `${r.path}  (${r.cases} case${r.cases === 1 ? '' : 's'}, `
        + `${r.png} figures)`;
      pick.appendChild(o);
    }
    pick.value = rows.some((r) => r.path === keep) ? keep : rows[0].path;
    return loadRun(pick.value);
  }).catch((e) => toast(e.message, true));
}

$('#runPick').addEventListener('change', () => {
  loadRun($('#runPick').value).catch((e) => toast(e.message, true));
});

/* Manual refresh: re-list the run directories and reload the current one, so a
 * run that just finished appears without reloading the page. Deliberately not
 * automatic — a poll that redrew figures under the reader would be worse. */
$('#btnRefresh').addEventListener('click', async () => {
  try {
    const keep = $('#runPick').value;
    const rows = await api('/api/results');
    const pick = $('#runPick');
    pick.textContent = '';
    for (const r of rows) {
      const o = document.createElement('option');
      o.value = r.path;
      o.textContent = `${r.path}  (${r.cases} case${r.cases === 1 ? '' : 's'}, `
        + `${r.png} figures)`;
      pick.appendChild(o);
    }
    if (!rows.length) { $('#runBody').textContent = NO_RUNS; return; }
    // stay where the reader was if it still exists, else show the newest
    pick.value = rows.some((r) => r.path === keep) ? keep : rows[0].path;
    await loadRun(pick.value);
    toast(pick.value === keep ? 'reloaded ' + keep : 'newest run: ' + pick.value);
  } catch (e) { toast(e.message, true); }
});

async function loadRun(path) {
  const run = await api('/api/run?path=' + encodeURIComponent(path));
  const host = $('#runBody');
  host.textContent = '';
  $('#runMeta').textContent =
    `${run.cases.length} case${run.cases.length === 1 ? '' : 's'}`;

  // Two configs run on the same day share a dated RESULT_DIR. Their .mat
  // exports both survive; the PNGs do not, because they are written under
  // fixed names. Saying so beats letting it be discovered.
  if (run.overwritten) {
    const w = document.createElement('div');
    w.className = 'problem';
    w.textContent = run.overwritten;
    host.appendChild(w);
  }

  for (const c of run.cases) {
    const sec = document.createElement('section');
    sec.className = 'case';

    const h = document.createElement('h3');
    h.textContent = c.name;
    sec.appendChild(h);

    if (c.metrics.length) {
      const grid = document.createElement('div');
      grid.className = 'metrics';
      for (const m of c.metrics) {
        const cell = document.createElement('div');
        cell.className = 'metric';
        const lab = document.createElement('div');
        lab.className = 'ml';
        lab.textContent = m.label;
        const val = document.createElement('div');
        val.className = 'mv';
        val.textContent = fmtNum(m.value) + (m.unit ? ' ' + m.unit : '');
        val.title = m.value;
        cell.appendChild(lab);
        cell.appendChild(val);
        grid.appendChild(cell);
      }
      sec.appendChild(grid);
    } else {
      const p = document.createElement('p');
      p.className = 'muted';
      p.textContent = c.has_csv
        ? 'results.csv has no recognised headline columns.'
        : 'no results.csv in this case directory.';
      sec.appendChild(p);
    }

    for (const st of c.stages) {
      const sh = document.createElement('h4');
      sh.textContent = 'Stage ' + st.id + ' — ' + st.label;
      sec.appendChild(sh);
      const strip = document.createElement('div');
      strip.className = 'figs';
      for (const f of st.figures) {
        const fig = document.createElement('figure');
        const img = document.createElement('img');
        img.loading = 'lazy';
        img.src = '/figure?path=' + encodeURIComponent(f);
        img.alt = f.split('/').pop();
        const cap = document.createElement('figcaption');
        cap.textContent = f.split('/').pop();
        fig.appendChild(img);
        fig.appendChild(cap);
        fig.addEventListener('click', () => {
          $('#figTitle').textContent = f.split('/').pop();
          $('#figImg').src = img.src;
          $('#figImg').alt = img.alt;
          $('#figDlg').showModal();
        });
        strip.appendChild(fig);
      }
      sec.appendChild(strip);
    }

    if (!c.stages.length) {
      const p = document.createElement('p');
      p.className = 'muted';
      p.textContent = 'no figures — the run had SAVE_FIGURES off.';
      sec.appendChild(p);
    } else {
      // Say which standard figures this run did not produce. The eye contour
      // and timing bathtub are gated on MLSE being off, so their absence is a
      // setting rather than a fault — "where is my eye diagram" is otherwise
      // a puzzle with no clue in the output.
      const have = new Set(c.stages.flatMap((st) => st.figures)
        .map((f) => f.split('/').pop()));
      const missing = EXPECTED_FIGS.filter((f) => !have.has(f));
      if (missing.length) {
        const p = document.createElement('p');
        p.className = 'missing';
        p.textContent = 'not produced by this run: ' + missing.join(', ')
          + (missing.some((m) => m.includes('eye') || m.includes('timing'))
            ? '  — the eye contour and timing bathtub are only drawn when '
              + 'MLSE is off, or with --eye-under-mlse.'
            : '');
        sec.appendChild(p);
      }
    }

    // Every column of results.csv, collapsed by default.
    if (c.all && c.all.length) {
      const det = document.createElement('details');
      det.className = 'allwrap';
      const sum = document.createElement('summary');
      sum.textContent = `All ${c.all.length} results.csv columns`;
      det.appendChild(sum);
      const wrap = document.createElement('div');
      wrap.className = 'alltable';
      const tbl = document.createElement('table');
      const thead = document.createElement('thead');
      const hr = document.createElement('tr');
      for (const h of ['column', 'value']) {
        const th = document.createElement('th');
        th.textContent = h;
        hr.appendChild(th);
      }
      thead.appendChild(hr);
      tbl.appendChild(thead);
      const tb = document.createElement('tbody');
      for (const col of c.all) {
        const tr = document.createElement('tr');
        const td1 = document.createElement('td');
        td1.textContent = col.name;
        const td2 = document.createElement('td');
        td2.textContent = col.value;
        tr.appendChild(td1);
        tr.appendChild(td2);
        tb.appendChild(tr);
      }
      tbl.appendChild(tb);
      wrap.appendChild(tbl);
      det.appendChild(wrap);
      sec.appendChild(det);
    }

    host.appendChild(sec);
  }
}

/* Numbers in results.csv are full double precision. Four significant figures is
 * what a person reads; the full value stays in the tooltip. */
function fmtNum(text) {
  const n = Number(text);
  if (!isFinite(n) || text.trim() === '') return text;
  if (Number.isInteger(n)) return String(n);
  if (Math.abs(n) >= 1e-3 && Math.abs(n) < 1e6) return n.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
  return n.toExponential(3);
}

window.addEventListener('beforeunload', (e) => {
  if (Object.keys(EDITS).length) { e.preventDefault(); e.returnValue = ''; }
});

show('config');
boot().catch((e) => toast(e.message, true));

/* ---------------------------------------------------------- S-parameters */

/* Colours cycle per file; IL is drawn solid, RL dashed and dimmer, so an
 * overlay of several channels stays readable without a legend lookup. */
const SP_COLOURS = ['#1e5fbf', '#c2410c', '#0d7a4a', '#6a3fb5',
                    '#b45309', '#0e7490', '#9d174d', '#4d7c0f'];
let SP_LOADED = [];      // [{payload, colour}]
let SP_ALL = [];         // every file in the current directory


async function spEnter() {
  try {
    if (CHANNEL_DIRS === null) await fillChannelDirs();
    if (!CHANNEL_DIRS.length) {
      toast(NO_S4P, true);
      return;
    }
    if (!SP_ALL.length) await spLoadDir($('#spDir').value);
  } catch (e) { toast(e.message, true); }
}

async function spLoadDir(dir) {
  SP_ALL = await api('/api/channels?dir=' + encodeURIComponent(dir));
  spFillFiles();
}

function spFillFiles() {
  const sel = $('#spFiles');
  const keep = new Set(selected(sel));
  sel.textContent = '';
  const thruOnly = $('#spFilter').checked;
  let shown = 0;
  for (const f of SP_ALL) {
    if (thruOnly && f.role !== 'thru' && !keep.has(f.path)) continue;
    const o = document.createElement('option');
    o.value = f.path;
    o.textContent = f.name + (f.role === 'thru' ? '' : `   (${f.role})`);
    if (keep.has(f.path)) o.selected = true;
    sel.appendChild(o);
    shown++;
  }
  $('#spMeta').textContent = `${shown} of ${SP_ALL.length} file(s)`;
  if (!sel.value && sel.options.length) sel.options[0].selected = true;
  spRefresh();
}

$('#spDir').addEventListener('change', () => {
  if (!$('#spDir').value) return;
  spLoadDir($('#spDir').value).catch((e) => toast(e.message, true));
});
$('#spFilter').addEventListener('change', spFillFiles);
$('#spFiles').addEventListener('change', spRefresh);
for (const id of ['#spIL', '#spRL', '#spLog']) {
  $(id).addEventListener('change', () => spDraw());
}

async function spRefresh() {
  const want = selected($('#spFiles'));
  if (!want.length) { SP_LOADED = []; spDraw(); return; }
  $('#spMeta').textContent = `reading ${want.length} file(s)...`;
  const out = [];
  for (let i = 0; i < want.length; i++) {
    try {
      // Sequential, not parallel: each file is a real Touchstone read through
      // the engine (~1.5 s cold), and firing twenty at once would stall the
      // single-threaded reader and the UI with it.
      const p = await api('/api/sparam?path=' + encodeURIComponent(want[i]));
      out.push({ p, colour: SP_COLOURS[i % SP_COLOURS.length] });
    } catch (e) {
      toast(`${want[i].split('/').pop()}: ${e.message}`, true);
    }
  }
  SP_LOADED = out;
  const pts = out.reduce((a, x) => a + x.p.n_points, 0);
  $('#spMeta').textContent =
    `${out.length} file(s) · ${pts} points · to ${
      Math.max(...out.map((x) => x.p.f_max_ghz)).toFixed(0)} GHz`;
  spDraw();
}

function spDraw() {
  const svg = $('#spChart');
  svg.textContent = '';
  const NS = 'http://www.w3.org/2000/svg';
  const el = (n, a, t) => {
    const e = document.createElementNS(NS, n);
    for (const k in a) e.setAttribute(k, a[k]);
    if (t !== undefined) e.textContent = t;
    return e;
  };
  const legend = $('#spLegend');
  legend.textContent = '';

  const W = 900, H = 520, L = 62, R = 16, T = 16, B = 44;
  if (!SP_LOADED.length) {
    svg.appendChild(el('text', {
      x: W / 2, y: H / 2, 'text-anchor': 'middle', class: 'spempty',
    }, 'select one or more Touchstone files'));
    return;
  }

  const showIL = $('#spIL').checked;
  const showRL = $('#spRL').checked;
  const logx = $('#spLog').checked;

  // extents
  let fmin = Infinity, fmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const { p } of SP_LOADED) {
    for (let i = 0; i < p.f_ghz.length; i++) {
      const f = p.f_ghz[i];
      if (logx && f <= 0) continue;
      if (f < fmin) fmin = f;
      if (f > fmax) fmax = f;
    }
    const series = [];
    if (showIL) series.push(p.il_db);
    if (showRL) series.push(p.rl1_db, p.rl2_db);
    for (const s of series) {
      for (const v of s) {
        if (!isFinite(v)) continue;
        if (v < ymin) ymin = v;
        if (v > ymax) ymax = v;
      }
    }
  }
  if (!isFinite(fmin) || !isFinite(ymin)) return;
  // a floor keeps a -140 dB tail from flattening the useful range
  ymin = Math.max(ymin, -80);
  ymax = Math.min(ymax + 2, 5);

  const fx = (f) => {
    const a = logx ? Math.log10(Math.max(f, fmin)) : f;
    const lo = logx ? Math.log10(fmin) : fmin;
    const hi = logx ? Math.log10(fmax) : fmax;
    return L + (a - lo) / (hi - lo) * (W - L - R);
  };
  const fy = (v) => T + (ymax - v) / (ymax - ymin) * (H - T - B);

  // grid
  const yticks = [];
  const stepY = (ymax - ymin) > 60 ? 20 : 10;
  for (let v = Math.ceil(ymax / stepY) * stepY; v >= ymin; v -= stepY) yticks.push(v);
  for (const v of yticks) {
    svg.appendChild(el('line', {
      x1: L, x2: W - R, y1: fy(v), y2: fy(v), class: 'spgrid',
    }));
    svg.appendChild(el('text', {
      x: L - 8, y: fy(v) + 4, 'text-anchor': 'end', class: 'spaxis',
    }, String(v)));
  }
  const xticks = logx
    ? [0.01, 0.1, 1, 10, 100].filter((f) => f >= fmin && f <= fmax)
    : Array.from({ length: 6 }, (_, i) => fmin + (fmax - fmin) * i / 5);
  for (const f of xticks) {
    svg.appendChild(el('line', {
      x1: fx(f), x2: fx(f), y1: T, y2: H - B, class: 'spgrid',
    }));
    svg.appendChild(el('text', {
      x: fx(f), y: H - B + 18, 'text-anchor': 'middle', class: 'spaxis',
    }, f >= 1 ? String(Math.round(f)) : String(f)));
  }
  svg.appendChild(el('text', {
    x: (L + W - R) / 2, y: H - 8, 'text-anchor': 'middle', class: 'spaxis',
  }, 'frequency (GHz)'));
  svg.appendChild(el('text', {
    x: 14, y: (T + H - B) / 2, class: 'spaxis',
    transform: `rotate(-90 14 ${(T + H - B) / 2})`, 'text-anchor': 'middle',
  }, 'dB'));

  const path = (xs, ys) => {
    let d = '', pen = false;
    for (let i = 0; i < xs.length; i++) {
      const f = xs[i], v = ys[i];
      if ((logx && f <= 0) || !isFinite(v)) { pen = false; continue; }
      const X = fx(f).toFixed(1), Y = fy(Math.max(ymin, Math.min(ymax, v))).toFixed(1);
      d += (pen ? 'L' : 'M') + X + ' ' + Y + ' ';
      pen = true;
    }
    return d;
  };

  for (const { p, colour } of SP_LOADED) {
    if (showIL) {
      svg.appendChild(el('path', {
        d: path(p.f_ghz, p.il_db), stroke: colour, class: 'spline',
      }));
    }
    if (showRL) {
      for (const s of [p.rl1_db, p.rl2_db]) {
        svg.appendChild(el('path', {
          d: path(p.f_ghz, s), stroke: colour, class: 'spline rl',
        }));
      }
    }
    const item = document.createElement('span');
    item.className = 'spitem';
    const sw = document.createElement('i');
    sw.style.background = colour;
    item.appendChild(sw);
    item.appendChild(document.createTextNode(
      `${p.name}  ·  ports ${p.ports.join(' ')}  ·  ${p.n_points} pts`));
    legend.appendChild(item);
  }
}


/* ------------------------------------------------------ dynamic results */

/* The static tab shows the PNGs every run writes. This one shows the
 * interactive R dashboard, which exists only when the run was given
 * --export-mat: `Rscript R/com_analysis.R <case>.mat` turns the export into a
 * self-contained HTML page, served here in an iframe. */
let DYN_POLL = null;
let DYN_SINCE = 0;

async function dynEnter() {
  try {
    const rows = await api('/api/results');
    const pick = $('#dynPick');
    const keep = pick.value;
    pick.textContent = '';
    for (const r of rows) {
      const o = document.createElement('option');
      o.value = r.path;
      o.textContent = r.path;
      pick.appendChild(o);
    }
    if (!rows.length) {
      $('#dynList').textContent = NO_RUNS;
      return;
    }
    pick.value = rows.some((r) => r.path === keep) ? keep : rows[0].path;
    await dynLoad(pick.value);
  } catch (e) { toast(e.message, true); }
}

$('#dynPick').addEventListener('change', () => {
  dynLoad($('#dynPick').value).catch((e) => toast(e.message, true));
});
$('#btnDynRefresh').addEventListener('click', () => {
  dynLoad($('#dynPick').value).catch((e) => toast(e.message, true));
});

async function dynLoad(path) {
  const d = await api('/api/dynamic?path=' + encodeURIComponent(path));
  const host = $('#dynList');
  host.textContent = '';
  $('#dynMeta').textContent = d.cases.length
    ? `${d.cases.length} export(s)`
    : 'no .mat exports in this run';

  if (!d.rscript) {
    const w = document.createElement('div');
    w.className = 'problem';
    w.textContent = 'Rscript was not found, so dashboards cannot be built here. '
      + 'Existing ones still display.';
    host.appendChild(w);
  }

  if (!d.cases.length) {
    const p = document.createElement('p');
    p.className = 'muted';
    p.textContent = 'This run has no .mat export. Tick "Output dynamic results" '
      + 'on the Run tab and run again.';
    host.appendChild(p);
    return;
  }

  for (const c of d.cases) {
    const row = document.createElement('div');
    row.className = 'dyncase';
    const nm = document.createElement('div');
    nm.className = 'dynname';
    nm.textContent = c.name;
    row.appendChild(nm);

    const bar = document.createElement('div');
    bar.className = 'dynbtns';
    if (c.report) {
      const open = document.createElement('button');
      open.className = 'ghost';
      open.textContent = 'Open';
      open.addEventListener('click', () => dynOpen(c));
      bar.appendChild(open);
    }
    const build = document.createElement('button');
    build.className = c.report ? 'ghost' : '';
    build.textContent = c.report ? 'Rebuild' : 'Build dashboard';
    build.disabled = !d.rscript;
    build.addEventListener('click', () => dynBuild(c));
    bar.appendChild(build);

    // A dashboard older than its export is stale: the run was repeated and
    // this page is showing the previous one.
    if (c.report && c.report_mtime < c.mat_mtime) {
      const s = document.createElement('span');
      s.className = 'tag ref';
      s.textContent = 'older than the export';
      bar.appendChild(s);
    }
    row.appendChild(bar);
    host.appendChild(row);
  }

  const first = d.cases.find((c) => c.report);
  if (first) dynOpen(first);
}

function dynOpen(c) {
  $('#dynTitle').textContent = c.name.replace(/\.mat$/, '');
  const fr = $('#dynFrame');
  // path-mapped, not a query parameter: the dashboard links Plotly and friends
  // with RELATIVE paths (lib/...), which only resolve if the page is served
  // from a URL that mirrors its directory.
  //
  // Normalise the separators FIRST, or that mirroring is lost. A run inside the
  // repo arrives as 'results/x/y_report.html' and splits into segments; a run
  // outside it (say D:\runs) arrives as an absolute Windows path, and splitting THAT on
  // '/' yields a single segment. The dashboard itself still loads, because the
  // server decodes the whole thing back to a path, so the failure is silent:
  // every relative lib/... link resolves against '/rpt/' instead of the run
  // directory, 404s, and the page renders all of its text with no Plotly loaded
  // and therefore no charts.
  fr.src = '/rpt/' + c.report.replace(/\\/g, '/')
    .split('/').map(encodeURIComponent).join('/');
  fr.hidden = false;
  $('#dynEmpty').hidden = true;
}

async function dynBuild(c) {
  try {
    const r = await api('/api/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mat: c.mat }),
    });
    toast('building: ' + r.command);
    $('#dynLog').hidden = false;
    $('#dynLog').textContent = '';
    DYN_SINCE = 0;
    dynPoll();
  } catch (e) { toast(e.message, true); }
}

async function dynPoll() {
  clearTimeout(DYN_POLL);
  let st;
  try {
    st = await api('/api/exec/status?slot=render&since=' + DYN_SINCE);
  } catch (e) {
    // same rule as the run poller: a failed poll must not end the loop
    $('#dynMeta').textContent = 'status unavailable: ' + e.message;
    DYN_POLL = setTimeout(dynPoll, 3000);
    return;
  }
  if (st.idle) return;
  if (st.lines.length) {
    const el = $('#dynLog');
    const keep = (el.textContent + st.lines.join('\n') + '\n').split('\n');
    el.textContent = keep.slice(-400).join('\n');
    el.scrollTop = el.scrollHeight;
  }
  DYN_SINCE = st.next;
  $('#dynMeta').textContent = st.running
    ? `building — ${st.elapsed.toFixed(0)}s`
    : `build finished (exit ${st.returncode}) in ${st.elapsed.toFixed(1)}s`;
  if (st.running || st.behind) {
    DYN_POLL = setTimeout(dynPoll, st.behind ? 150 : 1000);
  } else if (st.returncode === 0) {
    toast('dashboard built');
    dynLoad($('#dynPick').value).catch((e) => toast(e.message, true));
  } else {
    toast('R exited with code ' + st.returncode, true);
  }
}


/* After a run with "Output dynamic results", find the newest export and build
 * its dashboard, so the user does not have to go looking for it. */
async function autoBuildDynamic() {
  try {
    const rows = await api('/api/results');
    if (!rows.length) return;
    const d = await api('/api/dynamic?path=' + encodeURIComponent(rows[0].path));
    if (!d.cases.length) {
      toast('the run produced no .mat export', true);
      return;
    }
    if (!d.rscript) {
      toast('run finished, but Rscript was not found so no dashboard was built',
            true);
      return;
    }
    $('#dynPick').value = rows[0].path;
    show('dynamic');
    // newest export, not cases[0]: two configs writing into one dated
    // RESULT_DIR leave several .mat here, and the alphabetically first is not
    // the one that just ran.
    const newest = d.cases.slice().sort((a, b) => b.mat_mtime - a.mat_mtime)[0];
    await dynBuild(newest);
  } catch (e) { toast(e.message, true); }
}
