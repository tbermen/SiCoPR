"""Every pipeline stage must produce at least one figure.

Stage 2 (TDR/ERL) and stage 4 (Sampling) produced no per-run artifact at all
for months. Nothing said so: a directory of PNGs looks complete, and absence is
invisible in a listing. The correlation harness localises a disagreement to one
of seven stages, so a stage with no picture is a stage you can only debug from
a CSV column.

This is a static check -- it reads the _save() calls in com_plots.py rather than
running the engine -- so it costs nothing and works on a fresh clone with no
correlation data.

    python tests/test_stage_figures.py
"""
import io
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from audit_check import check, finish  # noqa: E402

PLOTS = os.path.join(_ROOT, 'com_plots.py')
src = io.open(PLOTS, encoding='utf-8').read()

# Figure names as written, e.g. _save(fig, outdir, "s2_tdr_impedance.png")
EMITTED = sorted(set(re.findall(r'_save\(\s*fig\s*,\s*outdir\s*,\s*"([^"]+)"', src)))

check("com_plots_emits_figures", len(EMITTED) >= 10,
      "expected the standard figure set; found %d _save() calls" % len(EMITTED))

# The stage list com_plots publishes, which is also what STAGE_INDEX.md renders.
import com_plots  # noqa: E402

STAGE_NUMS = [n for n, _name, _m, _w in com_plots.STAGES]

check("stage_list_is_the_seven_pipeline_stages",
      STAGE_NUMS == ['1', '2', '3', '4', '5', '6', '7'],
      "com_plots.STAGES should carry stages 1-7, got %s" % STAGE_NUMS)

# ---- the check this file exists for -------------------------------------
for num, name, _metrics, _what in com_plots.STAGES:
    figs = [f for f in EMITTED if f.startswith('s%s_' % num)]
    check("stage_%s_has_a_figure" % num, bool(figs),
          "stage %s (%s) emits no figure. A stage with no picture can only be "
          "debugged from a CSV column -- add one to com_plots.py, or if the "
          "stage genuinely cannot produce one, say so here explicitly."
          % (num, name))

# Every emitted figure should belong to a stage, so a new figure cannot land
# outside the index and go unnoticed.
orphans = [f for f in EMITTED
           if not any(f.startswith('s%s_' % n) for n in STAGE_NUMS)]
check("every_figure_maps_to_a_stage", not orphans,
      "figures with no stage prefix: %s. Name them s<stage>_<what>.png so they "
      "appear in STAGE_INDEX.md." % orphans)

# The index writer must list a stage even when it has nothing, otherwise the
# gap is invisible again -- which is the failure this whole file is about.
check("index_reports_stages_with_no_figure",
      '**no figure**' in src,
      "_write_stage_index must mark empty stages explicitly rather than "
      "omitting them")

# ---- the interactive report needs the same stages ------------------------
# The PNG set and the R dashboard cover the same seven stages, but from
# different data: the PNGs are drawn in-process from the live pipeline objects,
# the dashboard from the .mat export. A stage can therefore be covered in one
# and dark in the other -- which it was, until the exporter started carrying
# tdr / fom_vs_phase / noise_terms. Those plots degrade to a placeholder when
# their data is absent, so losing an export would not error the report, it would
# quietly stop showing the stage. That is the failure mode this file exists for.
EXPORT = io.open(os.path.join(_ROOT, 'com_mat_export.py'), encoding='utf-8').read()
DASH = io.open(os.path.join(_ROOT, 'R', 'com_analysis.R'), encoding='utf-8').read()

for group, stage in (('tdr', '2'), ('fom_vs_phase', '4'), ('noise_terms', '6')):
    check("mat_export_carries_%s" % group,
          '_add(d, "%s"' % group in EXPORT,
          'com_mat_export.py no longer exports "%s"; the stage %s plot in the R '
          'dashboard will silently render a placeholder instead of failing'
          % (group, stage))

for fn in ('plot_tdr_impedance', 'plot_erl', 'plot_fom_vs_phase',
           'plot_eq_taps', 'plot_noise_terms'):
    check("dashboard_has_%s" % fn,
          ('%s <- function' % fn) in DASH and ('%s(dat)' % fn) in DASH,
          'R/com_analysis.R must define %s AND call it from build_dashboard; '
          'defining it without wiring it in is how a plot goes missing quietly'
          % fn)

check("dashboard_groups_by_stage",
      all(('Stage %s' % n) in DASH for n in STAGE_NUMS),
      'build_dashboard should carry a heading for each of stages 1-7 so the '
      'report reads as stage evidence')

print("\n%d figures across %d stages:" % (len(EMITTED), len(STAGE_NUMS)))
for num, name, _m, _w in com_plots.STAGES:
    figs = [f for f in EMITTED if f.startswith('s%s_' % num)]
    print("   stage %s  %-16s %s" % (num, name, ', '.join(figs) or '(none)'))

finish()
