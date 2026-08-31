# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

# sweep_compare.R -- interactive R/Plotly visualisation of the EQ-search method
# comparison (full grid / legacy local search / adaptive local search) from the
# per-candidate trajectory logs written by sweep_compare.py.
#
# Two halves, and the distinction matters:
#   FOM side  -- per-candidate trajectory logs (<method>_log.csv). Dense: one row
#                per TX-FFE candidate considered, for every method.
#   COM side  -- one scalar per run (summary.json), because optimize_fom runs with
#                OP.COMPUTE_COM = False and COM is evaluated once at the end. The
#                only way to see COM *across* operating points is fom_com_probe.py,
#                which recomputes it for the top-K FOM candidates; its output
#                (fom_com_probe.csv) drives the "FOM as a proxy for COM" section.
#
# Dependencies:  install.packages(c("plotly", "htmltools", "jsonlite"))
#
# Usage:
#   Rscript R/sweep_compare.R [sweep_results_dir] [out.html]
#
# Or interactively:
#   source("R/sweep_compare.R")
#   logs <- load_sweep("sweep_results")
#   build_sweep_dashboard("sweep_results")     # full interactive HTML report

suppressPackageStartupMessages({
  library(plotly)
  library(htmltools)
  library(jsonlite)
})

METHOD_COLORS <- c(full_grid = "#999999", legacy = "#1f77b4", adaptive = "#d62728")

# -- loading ------------------------------------------------------------------
parse_taps <- function(s) {
  s <- gsub("\\[|\\]|\"", "", s)
  lapply(strsplit(trimws(s), "\\s+"), as.numeric)
}

load_one <- function(path) {
  if (!file.exists(path)) return(NULL)
  d <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  if (nrow(d) == 0) return(NULL)
  taps <- parse_taps(d$tx_taps)
  ntap <- max(lengths(taps))
  tapm <- t(sapply(taps, function(x) { length(x) <- ntap; x }))
  colnames(tapm) <- paste0("t", seq_len(ntap) - 1)
  # best_itick / best_cursor_i were added to the log later -- intersect() so that
  # logs written before that still load (those columns simply go missing).
  keep <- intersect(c("method", "ctle_index", "lp_index", "candidate_FOM",
                      "best_FOM", "best_itick", "best_cursor_i",
                      "evaluated", "eval_count"), names(d))
  cbind(d[keep], as.data.frame(tapm))
}

load_sweep <- function(dir = "sweep_results") {
  methods <- c("full_grid", "legacy", "adaptive")
  out <- lapply(methods, function(m) load_one(file.path(dir, paste0(m, "_log.csv"))))
  names(out) <- methods
  out[!sapply(out, is.null)]
}

# summary.json -- the per-run COM scalars, winning EQ point, and speedups.
load_summary <- function(dir = "sweep_results") {
  p <- file.path(dir, "summary.json")
  if (!file.exists(p)) return(NULL)
  jsonlite::fromJSON(p, simplifyVector = FALSE)
}

# fom_com_probe.csv (+ its summary json) -- true COM for the top-K FOM candidates.
load_probe <- function(dir = "sweep_results") {
  p <- file.path(dir, "fom_com_probe.csv")
  if (!file.exists(p)) return(NULL)
  d <- read.csv(p, stringsAsFactors = FALSE, check.names = FALSE)
  if (nrow(d) == 0) return(NULL)
  s <- file.path(dir, "fom_com_probe_summary.json")
  list(d = d[order(d$fom_rank), ],
       s = if (file.exists(s)) jsonlite::fromJSON(s, simplifyVector = FALSE) else NULL)
}

# -- small html helpers -------------------------------------------------------
.fmt <- function(x, dig = 4) {
  if (is.null(x) || length(x) == 0) return("--")
  if (is.logical(x)) return(if (isTRUE(x)) "yes" else "no")
  if (is.numeric(x)) return(formatC(x, format = "f", digits = dig))
  as.character(x)
}

.html_table <- function(header, rows) {
  tags$table(
    style = paste("border-collapse:collapse;font-family:monospace;",
                  "font-size:13px;margin:8px 0"),
    tags$thead(tags$tr(lapply(header, function(h)
      tags$th(h, style = "border:1px solid #ccc;padding:4px 8px;background:#f0f0f0;text-align:left")))),
    tags$tbody(lapply(rows, function(r) tags$tr(lapply(r, function(c)
      tags$td(HTML(as.character(c)), style = "border:1px solid #ccc;padding:4px 8px"))))))
}

# -- plots --------------------------------------------------------------------
plot_convergence <- function(logs) {
  p <- plot_ly()
  for (nm in names(logs)) {
    d <- logs[[nm]]; d <- d[d$evaluated == 1, ]
    d <- d[order(d$eval_count), ]
    p <- add_lines(p, x = d$eval_count, y = d$best_FOM, name = nm,
                   line = list(color = METHOD_COLORS[[nm]]))
  }
  layout(p, title = "Convergence: best-so-far FOM vs candidates evaluated",
         xaxis = list(title = "candidates evaluated"),
         yaxis = list(title = "best FOM (dB)"))
}

# 2-D TX-FFE tap coverage, colored by FOM; one trace per method, hover = settings
plot_coverage <- function(logs) {
  fg <- if (!is.null(logs$full_grid)) logs$full_grid else logs[[1]]
  tapcols <- grep("^t[0-9]+$", names(fg), value = TRUE)
  v <- sapply(tapcols, function(c) var(fg[[c]], na.rm = TRUE))
  ij <- tapcols[order(v, decreasing = TRUE)][1:2]
  cmin <- min(fg$candidate_FOM[is.finite(fg$candidate_FOM)])
  cmax <- max(fg$candidate_FOM[is.finite(fg$candidate_FOM)])
  p <- plot_ly()
  for (nm in names(logs)) {
    d <- logs[[nm]]; ev <- d[d$evaluated == 1 & is.finite(d$candidate_FOM), ]
    hov <- sprintf("FOM=%.3f<br>ctle=%g lp=%g<br>%s=%g %s=%g",
                   ev$candidate_FOM, ev$ctle_index, ev$lp_index,
                   ij[1], ev[[ij[1]]], ij[2], ev[[ij[2]]])
    p <- add_markers(p, x = jitter(ev[[ij[1]]]), y = jitter(ev[[ij[2]]]),
                     name = nm, text = hov, hoverinfo = "text",
                     marker = list(color = ev$candidate_FOM, colorscale = "Viridis",
                                   cmin = cmin, cmax = cmax, size = 7,
                                   line = list(width = 0.5, color = "black"),
                                   showscale = (nm == names(logs)[1]),
                                   colorbar = list(title = "FOM (dB)")),
                     visible = if (nm == names(logs)[1]) TRUE else "legendonly")
  }
  layout(p, title = "EQ-settings vs FOM: TX-FFE tap-space coverage (toggle methods in legend)",
         xaxis = list(title = paste0("TX-FFE ", ij[1], " index")),
         yaxis = list(title = paste0("TX-FFE ", ij[2], " index")))
}

# parallel coordinates over all tap dims + CTLE + LP, colored by FOM
plot_parcoords <- function(logs, method = NULL) {
  if (is.null(method)) method <- if (!is.null(logs$adaptive)) "adaptive" else names(logs)[1]
  d <- logs[[method]]; d <- d[d$evaluated == 1 & is.finite(d$candidate_FOM), ]
  tapcols <- grep("^t[0-9]+$", names(d), value = TRUE)
  dims <- lapply(c(tapcols, "ctle_index", "lp_index"), function(c)
    list(label = c, values = d[[c]]))
  plot_ly(type = "parcoords",
          line = list(color = d$candidate_FOM, colorscale = "Viridis",
                      showscale = TRUE, colorbar = list(title = "FOM (dB)")),
          dimensions = dims) |>
    layout(title = paste0("EQ settings (parallel coordinates) colored by FOM — ", method))
}

# -- COM side -----------------------------------------------------------------
# Which sampling phase rode along with the winning FOM. Two methods can agree on
# every EQ index and still hand a different operating point to the COM evaluation,
# because BEST.itick is carried by the same strict `THIS.FOM > BEST.FOM` update.
# A step here that differs between methods is a COM difference waiting to happen.
plot_itick <- function(logs) {
  have <- names(logs)[sapply(logs, function(d) "best_itick" %in% names(d))]
  if (length(have) == 0) return(NULL)
  p <- plot_ly()
  for (nm in have) {
    d <- logs[[nm]]; d <- d[d$evaluated == 1, ]
    d <- d[order(d$eval_count), ]
    it <- suppressWarnings(as.numeric(d$best_itick))
    p <- add_lines(p, x = d$eval_count, y = it, name = nm,
                   line = list(color = METHOD_COLORS[[nm]], shape = "hv"))
  }
  layout(p, title = "Winning sample phase (BEST.itick) vs candidates evaluated",
         xaxis = list(title = "candidates evaluated"),
         yaxis = list(title = "BEST.itick"))
}

# THE COM plot: does maximising FOM actually maximise COM?
# Each point is a top-K FOM candidate whose true COM was recomputed by
# fom_com_probe.py. If FOM were a faithful proxy the points would rise
# monotonically left to right; the gap between the search's pick (FOM rank 1) and
# the best COM in the set is the regret -- what a FOM-driven search gives up.
plot_fom_vs_com <- function(probe) {
  if (is.null(probe)) return(NULL)
  d <- probe$d
  ok <- is.finite(d$COM_dB) & is.finite(d$FOM_dB)
  d <- d[ok, ]
  if (nrow(d) < 2) return(NULL)

  pick <- d[d$fom_rank == min(d$fom_rank), ][1, ]
  best <- d[which.max(d$COM_dB), ][1, ]
  regret <- best$COM_dB - pick$COM_dB

  hov <- sprintf("FOM rank %d<br>FOM=%.4f dB<br>COM=%.4f dB<br>ctle=%s lp=%s taps=%s",
                 d$fom_rank, d$FOM_dB, d$COM_dB, d$ctle_index, d$lp_index, d$tx_taps)

  p <- plot_ly() |>
    add_markers(x = d$FOM_dB, y = d$COM_dB, text = hov, hoverinfo = "text",
                name = "top-K candidates",
                marker = list(size = 9, color = d$fom_rank, colorscale = "Viridis",
                              reversescale = TRUE, showscale = TRUE,
                              colorbar = list(title = "FOM rank"),
                              line = list(width = 0.5, color = "black"))) |>
    add_markers(x = pick$FOM_dB, y = pick$COM_dB, name = "search picks this (FOM rank 1)",
                marker = list(size = 18, symbol = "circle-open",
                              line = list(width = 3, color = "#d62728"))) |>
    add_markers(x = best$FOM_dB, y = best$COM_dB, name = "best COM in top-K",
                marker = list(size = 18, symbol = "diamond-open",
                              line = list(width = 3, color = "#2ca02c")))

  layout(p,
         title = sprintf(paste0("FOM as a proxy for COM -- COM regret = %.4f dB",
                                "<br><sub>the search maximises FOM (x); COM (y) is ",
                                "what actually decides pass/fail</sub>"), regret),
         xaxis = list(title = "FOM (dB)"),
         yaxis = list(title = "true COM (dB), recomputed"),
         legend = list(orientation = "h", y = -0.2))
}

# The same data read as an ordering question: walk down the FOM ranking and watch
# what COM does. Flat/noisy => FOM ordering carries little COM information.
plot_com_vs_rank <- function(probe) {
  if (is.null(probe)) return(NULL)
  d <- probe$d
  d <- d[is.finite(d$COM_dB), ]
  if (nrow(d) < 2) return(NULL)
  running <- cummax(d$COM_dB)
  p <- plot_ly() |>
    add_trace(x = d$fom_rank, y = d$COM_dB, name = "COM at this FOM rank",
              type = "scatter", mode = "lines+markers",
              line = list(color = "#1f77b4"), marker = list(size = 6)) |>
    add_lines(x = d$fom_rank, y = running, name = "best COM seen so far",
              line = list(color = "#2ca02c", dash = "dot"))
  layout(p, title = "COM along the FOM ranking",
         xaxis = list(title = "FOM rank (1 = what the search picks)"),
         yaxis = list(title = "true COM (dB)"),
         legend = list(orientation = "h", y = -0.2))
}

# Per-method run summary: the COM scalars, the winning EQ point, and whether the
# pruned methods landed where the full grid did.
method_table <- function(summ) {
  if (is.null(summ) || is.null(summ$methods)) return(NULL)
  ms <- summ$methods
  rows <- lapply(names(ms), function(nm) {
    m <- ms[[nm]]
    eq <- m$best_EQ
    eqs <- if (is.null(eq)) "--" else sprintf(
      "ctle=%s lp=%s tk=%s taps=%s itick=%s",
      .fmt(eq$ctle_index), .fmt(eq$lp_index), .fmt(eq$txffe_index),
      .fmt(eq$tx_taps), .fmt(eq[["best_itick"]]))
    list(sprintf("<b>%s</b>", nm), .fmt(m$COM_dB), .fmt(m$best_FOM_dB, 4),
         .fmt(m$n_evaluated, 0), .fmt(m$pct_of_full_grid_evaluated, 1),
         sprintf("%sx", .fmt(m$speedup_vs_full_grid, 2)),
         .fmt(m[["same_EQ_as_full_grid"]]), .fmt(m[["dCOM_vs_full_grid"]], 6), eqs)
  })
  .html_table(c("method", "COM (dB)", "best FOM (dB)", "evaluated", "% grid",
                "speedup", "same EQ as full grid", "dCOM vs full grid",
                "winning EQ point"), rows)
}

probe_table <- function(probe) {
  if (is.null(probe) || is.null(probe$s)) return(NULL)
  s <- probe$s
  if (is.null(s[["spearman_rho_FOM_vs_COM"]])) {
    return(tags$p(sprintf("Probe ran but could not be summarised: %s",
                          .fmt(s[["note"]]))))
  }
  rows <- list(
    list("Spearman rho(FOM, COM)", .fmt(s$spearman_rho_FOM_vs_COM),
         "+1 = FOM ordering perfectly tracks COM ordering"),
    list("candidates probed", .fmt(s$n_probed, 0), "top-K by FOM"),
    list("FOM spread over top-K", sprintf("%s dB", .fmt(s$FOM_spread_topK_dB)),
         "how tightly the top of the FOM surface is packed"),
    list("COM spread over top-K", sprintf("%s dB", .fmt(s$COM_spread_topK_dB)),
         "how much COM actually moves across that same set"),
    list("COM at FOM argmax", sprintf("%s dB", .fmt(s$COM_at_FOM_argmax_dB)),
         "what the search returns"),
    list("best COM in top-K", sprintf("%s dB", .fmt(s$best_COM_in_topK_dB)),
         sprintf("found at FOM rank %s", .fmt(s$com_argmax_fom_rank, 0))),
    list("<b>COM regret</b>", sprintf("<b>%s dB</b>", .fmt(s$COM_regret_dB)),
         "COM given up by trusting the FOM ranking"))
  .html_table(c("quantity", "value", "meaning"), rows)
}

# -- dashboard ----------------------------------------------------------------
build_sweep_dashboard <- function(dir = "sweep_results",
                                  out_html = file.path(dir, "sweep_compare.html")) {
  logs <- load_sweep(dir)
  if (length(logs) == 0) stop("No *_log.csv found in ", dir, " (run sweep_compare.py first)")
  summ  <- load_summary(dir)
  probe <- load_probe(dir)
  parc  <- lapply(names(logs), function(m) plot_parcoords(logs, m))

  # Sections that need data the run may not have produced yet degrade to a note
  # rather than failing the whole report.
  com_section <- if (is.null(probe)) {
    tags$p(tags$em(paste("No fom_com_probe.csv in this directory -- run",
                         "fom_com_probe.py to populate the COM side.")))
  } else {
    tagList(probe_table(probe), plot_fom_vs_com(probe), plot_com_vs_rank(probe))
  }
  itick_plot <- plot_itick(logs)
  itick_section <- if (is.null(itick_plot)) {
    tags$p(tags$em(paste("Logs predate the best_itick column -- re-run",
                         "sweep_compare.py to populate the sample-phase trace.")))
  } else itick_plot

  hdr <- if (is.null(summ)) NULL else tags$p(sprintf(
    "config: %s | thru: %s | LOCAL_SEARCH=%s | max_ctle=%s | max_tap_vals=%s",
    .fmt(summ$config), .fmt(summ$thru), .fmt(summ$local_search, 0),
    .fmt(summ$max_ctle, 0), .fmt(summ$max_tap_vals, 0)))

  page <- tagList(
    tags$h1("EQ-search method comparison"),
    hdr,
    tags$p(sprintf("Methods: %s", paste(names(logs), collapse = ", "))),

    tags$h2("Run summary (COM, winning EQ point)"),
    if (is.null(summ)) tags$p(tags$em("No summary.json found.")) else method_table(summ),

    tags$h2("FOM as a proxy for COM"),
    tags$p(paste("COM is evaluated once per run, after the search. These panels are",
                 "the only view of how COM behaves across operating points, and they",
                 "bound what any FOM-driven pruning can cost.")),
    com_section,

    tags$h2("Convergence"), plot_convergence(logs),
    tags$h2("Winning sample phase"), itick_section,
    tags$h2("TX-FFE tap-space coverage"), plot_coverage(logs),
    tags$h2("EQ settings (parallel coordinates)"), tagList(parc)
  )
  save_html(page, out_html)
  message("Wrote ", out_html)
  invisible(out_html)
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  dir <- if (length(args) >= 1) args[1] else "sweep_results"
  out <- if (length(args) >= 2) args[2] else file.path(dir, "sweep_compare.html")
  build_sweep_dashboard(dir, out)
}
