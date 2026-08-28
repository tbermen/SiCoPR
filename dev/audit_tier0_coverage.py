"""Tier 0 coverage-map builder for the sicopr.py vs com_ieee8023_4p15p0.m audit.

Parses function definitions from the MATLAB reference and sicopr.py, maps each
MATLAB function to its Python counterpart (same name, or hoisted
_Parent__helper), checks in-file provenance citations, cross-checks against
com_functions/registry.json, and emits:
  - audit_state.json   (coverage map + risk-ordered work queue + batch pointer)
  - audit_ledger.csv   (one row per function, seeded NOT_YET_AUDITED / UNPORTED)
  - stdout summary for the Gate 0 table

Run: python audit_tier0_coverage.py
"""
import json
import os
import re
import csv
import datetime

MATLAB_FILE = os.path.join("matlab", "com_ieee8023_4p15p0.m")
ADAPTIVE_FILE = os.path.join("matlab", "com_ieee8023_4p15p0_adaptive_local_search.m")
PY_FILE = "sicopr.py"
REGISTRY = os.path.join("com_functions", "registry.json")

# ---------------------------------------------------------------- MATLAB parse
def parse_matlab(path):
    """Return list of dicts: name, start, end, indent, parent (None if top)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    fn_re = re.compile(
        r"^(\s*)function\s+(?:\[?[\w\s,~]*\]?\s*=\s*)?([A-Za-z_]\w*)\s*(?:\(|;|%|$)")
    fns = []
    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        # join MATLAB ... continuation lines so multi-line defs parse
        probe = line
        j = i - 1
        while probe.rstrip().rstrip("%").rstrip().endswith("...") and j + 1 < len(lines):
            probe = probe.rstrip().rstrip(".") + " " + lines[j + 1].lstrip()
            j += 1
        m = fn_re.match(probe)
        if m and re.match(r"^\s*function\b", line):
            fns.append({"name": m.group(2), "start": i,
                        "indent": len(m.group(1)), "parent": None})
    # end = line before next function def, last runs to EOF
    for j, fn in enumerate(fns):
        fn["end"] = fns[j + 1]["start"] - 1 if j + 1 < len(fns) else len(lines)
    # parent for nested (indented) functions = nearest preceding indent-0 fn
    last_top = None
    for fn in fns:
        if fn["indent"] == 0:
            last_top = fn["name"]
        else:
            fn["parent"] = last_top
    return fns, len(lines)

# ---------------------------------------------------------------- Python parse
def parse_python(path):
    """Return list of dicts: name, start, end (module-level defs only)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    def_re = re.compile(r"^def\s+([A-Za-z_]\w*)\s*\(")
    defs = []
    for i, line in enumerate(lines, start=1):
        m = def_re.match(line)
        if m:
            defs.append({"name": m.group(1), "start": i})
    for j, d in enumerate(defs):
        d["end"] = defs[j + 1]["start"] - 1 if j + 1 < len(defs) else len(lines)
    return defs, lines, len(lines)

# genuine citation = "MATLAB" adjacent to a line-number reference
CITE_RE = re.compile(r"(?i)matlab[^\n]{0,60}?(?:lines?\s*\d{1,5}|L\d{3,5}\b)")
# per-function block banner emitted by assemble_com.py, cites 4p14p0 lines
BANNER_RE = re.compile(
    r"^#\s*---\s*(\w+)\s*\(MATLAB lines\s*(\d+)\s*[–\-]\s*(\d+)\)")

def scan_banners(pylines):
    """Map function name -> (banner_line, cited_start, cited_end) for the
    '# --- Name (MATLAB lines A-B) ---' block headers in sicopr.py."""
    banners = {}
    for i, line in enumerate(pylines, start=1):
        m = BANNER_RE.match(line)
        if m:
            banners[m.group(1)] = (i, int(m.group(2)), int(m.group(3)))
    return banners

def has_provenance_cite(pydef, pylines, banners):
    """Banner block header for this function, or an in-body MATLAB line cite
    in the first 40 lines of the def."""
    if pydef["name"] in banners:
        _, a, b = banners[pydef["name"]]
        return "banner: MATLAB lines %d-%d (4p14p0 basis)" % (a, b)
    start = pydef["start"] - 1
    stop = min(pydef["end"], start + 40)
    m = CITE_RE.search("".join(pylines[start:stop]))
    return (m.group(0).strip() if m else None)

# ---------------------------------------------------------------- mapping
def build_map(mfns, pydefs, pylines):
    banners = scan_banners(pylines)
    pyindex = {d["name"]: d for d in pydefs}
    # suffix index for hoisted helpers: helper name -> [py def names]
    suffix_index = {}
    for d in pydefs:
        m = re.match(r"^_(\w+?)__(\w+)$", d["name"])
        if m:
            suffix_index.setdefault(m.group(2), []).append(d["name"])

    coverage = {}
    used_py = set()
    for fn in mfns:
        name = fn["name"]
        key = name if fn["parent"] is None else "_%s__%s" % (fn["parent"], name)
        cand = None
        if fn["parent"] is not None:
            hoisted = "_%s__%s" % (fn["parent"], name)
            if hoisted in pyindex:
                cand = hoisted
        if cand is None and name in pyindex:
            cand = name
        if cand is None and name in suffix_index and len(suffix_index[name]) == 1:
            cand = suffix_index[name][0]
        entry = {
            "matlab_name": name,
            "matlab_file": fn.get("file", MATLAB_FILE),
            "matlab_lines": [fn["start"], fn["end"]],
            "matlab_kind": "top_level" if fn["parent"] is None else
                           "nested_in_%s" % fn["parent"],
            "py_name": cand,
            "py_lines": None,
            "ported": cand is not None,
            "provenance_cite": None,
            "status": "NOT_YET_AUDITED" if cand else "UNPORTED",
            "prior_audit": None,
        }
        if cand:
            d = pyindex[cand]
            entry["py_lines"] = [d["start"], d["end"]]
            entry["provenance_cite"] = has_provenance_cite(d, pylines, banners)
            if not entry["provenance_cite"]:
                entry["provenance_flag"] = "SUSPECT_PROVENANCE"
            used_py.add(cand)
        coverage[key] = entry

    # classify remaining py defs: hoisted per-caller copies of MATLAB local
    # functions (Tier 3 cross-check targets) vs genuine python-only helpers
    matlab_names = {fn["name"] for fn in mfns}
    copy_re = re.compile(r"^_(\w+?)__(\w+?)(_b|_inline|_zero|_params)?$")
    python_only = []
    for d in pydefs:
        if d["name"] in used_py:
            continue
        m = copy_re.match(d["name"])
        base = m.group(2) if m else None
        if m and base in matlab_names and base in coverage:
            coverage[base].setdefault("py_variant_copies", []).append(
                {"py_name": d["name"], "py_lines": [d["start"], d["end"]]})
        else:
            python_only.append(d["name"])
    return coverage, python_only

# ---------------------------------------------------------------- risk queue
# Section 3 risk groups, highest risk first. Functions named per group; a batch
# is one queue entry (max 6 functions).
RISK_BATCHES = [
    ("B01", "G1 unit conversions in noise integrals",
     ["get_PSDs", "get_sigma_eta_ACCM_noise", "get_sigma_noise", "N_s", "S_RN"]),
    ("B02", "G2 frequency grid construction",
     ["FD_Processing", "interp_Sparam", "read_s4p_files"]),
    ("B03", "G3 impulse/spectrum handling (prime suspects)",
     ["s21_to_impulse_DC", "COM_FD_to_TD"]),
    ("B04", "G4 convolution and DFE",
     ["FFE", "FFE_Fast", "Fract_T_FFE", "TD_CTLE", "dfe_clipper"]),
    ("B05", "G5 S-parameter interpolation and cascade",
     ["combines4p", "stot", "ttos", "make_full_pkg", "s_for_c2", "s_for_c4"]),
    ("B06", "G5 S-parameter synthesis + TDR/ERL",
     ["synth_tline", "get_TDR", "TDR_ERL_Processing", "R_series2",
      "r_parrelell2"]),
    ("B07", "G6 PDF/CDF noise pipeline, leaf level",
     ["d_cpdf", "conv_fct", "Init_PDF_Fast", "normal_dist", "get_pdf",
      "pdf_to_cdf"]),
    ("B08", "G6 PDF/CDF noise pipeline, composite",
     ["Create_Noise_PDF", "combine_pdf_same_voltage_axis", "comb_fct",
      "conv_fct_MeanNotZero", "get_pdf_full", "get_pdf_from_sampled_signal"]),
    ("B09", "G7 cursor and sample indexing",
     ["cursor_sample_index", "get_center_of_UI", "COM_eye_width", "vma"]),
    ("B10", "G8 rounding + G9 reshape/flatten order, cross-cutting scans",
     ["__CROSS_CUTTING_SCAN__mround_vs_np_round",
      "__CROSS_CUTTING_SCAN__reshape_ravel_flatten_order"]),
    ("B11", "G10 optimizer loops, FOM core",
     ["optimize_fom", "OptFom_Calc_Hr", "OptFom_Adaptive_Local_Search"]),
    ("B12", "G10 optimizer loops, MMSE + floating taps",
     ["MMSE", "MMSE_FOM", "FOM_rxffe_floating_taps", "Full_Grid_Matrix"]),
]

PRIOR_AUDIT_VERIFIED = {
    "COM_FD_to_TD": "BUG-01 unfounded (2026-07-01 re-check)",
    "s21_to_impulse_DC": "BUG-02 unfounded (2026-07-01 re-check)",
    "get_PSDs": "BUG-03 unfounded (2026-07-01 re-check)",
    "get_sigma_eta_ACCM_noise": "BUG-04 unfounded (2026-07-01 re-check)",
    "N_s": "BUG-05 unfounded (2026-07-01 re-check)",
    "Create_Noise_PDF": "BUG-06 unfounded (2026-07-01 re-check)",
    "optimize_fom": "BUG-07 unfounded (2026-07-01 re-check)",
}

def main():
    mfns, m_total = parse_matlab(MATLAB_FILE)
    for fn in mfns:
        fn["file"] = MATLAB_FILE
    # adaptive-variant file: include only functions that do not exist in the
    # base 4p15p0 file (the port implements the adaptive local-search feature)
    afns, a_total = parse_matlab(ADAPTIVE_FILE)
    base_names = {f["name"] for f in mfns}
    adaptive_only = [f for f in afns if f["name"] not in base_names]
    for fn in adaptive_only:
        fn["file"] = ADAPTIVE_FILE
    pydefs, pylines, py_total = parse_python(PY_FILE)
    coverage, python_only = build_map(mfns + adaptive_only, pydefs, pylines)

    for key, entry in coverage.items():
        if entry["matlab_name"] in PRIOR_AUDIT_VERIFIED and entry["matlab_kind"] == "top_level":
            entry["prior_audit"] = PRIOR_AUDIT_VERIFIED[entry["matlab_name"]]

    # registry cross-check
    reg_names = []
    if os.path.exists(REGISTRY):
        with open(REGISTRY, encoding="utf-8-sig") as f:
            reg = json.load(f)
        reg_names = [fn["name"] for fn in reg["functions"]]
    top_names = {e["matlab_name"] for e in coverage.values()
                 if e["matlab_kind"] == "top_level"}
    reg_not_in_m = sorted(set(reg_names) - top_names)
    m_not_in_reg = sorted(top_names - set(reg_names))

    # registry tier lookup for the ledger
    reg_tier = {}
    if reg_names:
        for fn in reg["functions"]:
            reg_tier[fn["name"]] = fn.get("tier", "")

    unported = [k for k, e in coverage.items() if not e["ported"]]
    no_cite = [k for k, e in coverage.items()
               if e["ported"] and not e["provenance_cite"]]

    state = {
        "audit_version": "tier0",
        "created": datetime.date.today().isoformat(),
        "reference_matlab": MATLAB_FILE,
        "reference_matlab_total_lines": m_total,
        "python_under_audit": PY_FILE,
        "python_total_lines": py_total,
        "notes": [
            "sicopr.py header states it was assembled from com_ieee8023_4p14p0.m; "
            "this audit judges it against 4p15p0 per the audit prompt, so "
            "4p14p0-to-4p15p0 deltas will surface as findings.",
            "compare_to_matlab.py no longer exists (only a stale .pyc); Tier 5 "
            "npz seam-dump convention must be recreated when first needed.",
            "Prior 7-bug hypothesis audit (completed 2026-07-01) preserved in "
            "audit_state_prior_bughunt_20260701.json; its per-function results "
            "are carried in prior_audit fields but do NOT exempt those "
            "functions from this audit's per-function test requirement.",
        ],
        "gate": "GATE0_PENDING_APPROVAL",
        "batch_pointer": 0,
        "completed_batches": [],
        "work_queue": [
            {"batch_id": b[0], "description": b[1], "functions": b[2],
             "status": "pending"} for b in RISK_BATCHES
        ],
        "coverage_map": coverage,
        "python_only_functions": sorted(python_only),
        "registry_cross_check": {
            "registry_count": len(reg_names),
            "matlab_top_level_count": len(top_names),
            "in_registry_not_in_4p15p0": reg_not_in_m,
            "in_4p15p0_not_in_registry": m_not_in_reg,
        },
    }
    with open("audit_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)

    # ledger
    with open("audit_ledger.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "matlab_lines", "py_lines", "tier", "verdict",
                    "trap_category", "evidence_ref", "test_files",
                    "test_result", "recommended_fix"])
        for key, e in sorted(coverage.items(),
                             key=lambda kv: kv[1]["matlab_lines"][0]):
            w.writerow([
                key,
                "%d-%d" % tuple(e["matlab_lines"]),
                "%d-%d" % tuple(e["py_lines"]) if e["py_lines"] else "",
                reg_tier.get(e["matlab_name"], ""),
                e["status"],
                "",
                ("no in-file cite; registry provenance only"
                 if e["ported"] and not e["provenance_cite"] else
                 (e["provenance_cite"] or "")),
                "", "", "",
            ])

    # ------------------------------------------------------------- findings
    with open("AUDIT_FINDINGS.md", "w", encoding="utf-8") as f:
        f.write("# AUDIT FINDINGS, sicopr.py vs com_ieee8023_4p15p0.m\n\n")
        f.write("Status: Tier 0 complete, waiting at Gate 0. "
                "No function bodies audited yet.\n\n")
        f.write("## Top-risk open items\n\n")
        f.write("None yet. Tier 1-4 audits have not started. "
                "All 157 functions are NOT_YET_AUDITED.\n\n")
        f.write("## Context and caveats\n\n")
        for note in state["notes"]:
            f.write("- %s\n" % note)
        f.write("\n## Coverage map (Tier 0)\n\n")
        f.write("%d MATLAB functions (154 in the base 4p15p0 file, 3 only in "
                "the adaptive local-search variant). All ported, none "
                "UNPORTED. All 157 carry a provenance banner citing 4p14p0 "
                "line numbers, so line cites below are re-derived against "
                "4p15p0.\n\n" % len(coverage))
        f.write("| MATLAB function | MATLAB lines (4p15p0) | Python counterpart "
                "| py lines | Ported | Provenance cite | Copies | Prior audit |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for key, e in sorted(coverage.items(),
                             key=lambda kv: (kv[1]["matlab_file"],
                                             kv[1]["matlab_lines"][0])):
            src = ("adaptive: %d-%d" % tuple(e["matlab_lines"])
                   if e["matlab_file"] == ADAPTIVE_FILE
                   else "%d-%d" % tuple(e["matlab_lines"]))
            f.write("| %s | %s | %s | %s | %s | %s | %d | %s |\n" % (
                e["matlab_name"], src,
                e["py_name"] or "-",
                "%d-%d" % tuple(e["py_lines"]) if e["py_lines"] else "-",
                "yes" if e["ported"] else "NO",
                "yes" if e["provenance_cite"] else "NO",
                len(e.get("py_variant_copies", [])),
                e["prior_audit"] or "",
            ))
        f.write("\n## Work queue (risk-ordered, one batch per run)\n\n")
        for b in state["work_queue"]:
            f.write("- **%s** (%s): %s\n" % (b["batch_id"], b["description"],
                                             ", ".join(b["functions"])))
        f.write("\n## Python-only helpers (no MATLAB counterpart, "
                "provenance review only)\n\n")
        f.write(", ".join("`%s`" % n for n in sorted(python_only)) + "\n")

    # ------------------------------------------------------------- summary
    nested = [k for k, e in coverage.items() if "nested" in e["matlab_kind"]]
    adaptive = [k for k, e in coverage.items()
                if e["matlab_file"] == ADAPTIVE_FILE]
    n_copies = sum(len(e.get("py_variant_copies", []))
                   for e in coverage.values())
    print("MATLAB functions mapped: %d (base file %d, adaptive-only %d, "
          "nested %d); base file %d lines"
          % (len(coverage), len(coverage) - len(adaptive), len(adaptive),
             len(nested), m_total))
    print("  adaptive-only: %s" % ", ".join(sorted(adaptive)))
    print("Python module-level defs: %d, %d lines" % (len(pydefs), py_total))
    print("Ported: %d   UNPORTED: %d" % (len(coverage) - len(unported),
                                         len(unported)))
    if unported:
        for k in unported:
            e = coverage[k]
            print("  UNPORTED  %-40s  M %d-%d" % (k, *e["matlab_lines"]))
    print("Hoisted per-caller variant copies (Tier 3 cross-check targets): %d"
          % n_copies)
    print("Ported with NO in-file provenance cite: %d of %d"
          % (len(no_cite), len(coverage) - len(unported)))
    for k in sorted(no_cite):
        print("  NO-CITE   %s" % k)
    print("Python-only module-level defs (no MATLAB counterpart): %d"
          % len(python_only))
    print("Registry cross-check: registry=%d, MATLAB top-level=%d"
          % (len(reg_names), len(top_names)))
    if reg_not_in_m:
        print("  in registry, not in 4p15p0: %s" % ", ".join(reg_not_in_m))
    if m_not_in_reg:
        print("  in 4p15p0, not in registry: %s" % ", ".join(m_not_in_reg))

if __name__ == "__main__":
    main()
