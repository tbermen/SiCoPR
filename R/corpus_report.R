# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

# corpus_report.R -- corpus-wide aggregation of the EQ-search method comparison.
#
# Consumes what corpus_sweep.py writes:
#   runs.csv            one row per channel x method
#   probe.csv           one row per probed top-K candidate, per channel
#   corpus_summary.json distribution stats + pass/fail flip detail
#
# The single-channel report is R/sweep_compare.R; this is its N-channel counterpart.
# Where that one answers "what happened on this channel", this one answers the two
# questions a standards reviewer actually asks: how often does pruning change COM, and
# how bad is the worst case.
#
# Dependencies:  install.packages(c("plotly", "htmltools", "jsonlite"))
#
# Usage:
#   Rscript R/corpus_report.R [corpus_results_dir] [out.html]

suppressPackageStartupMessages({
  library(plotly)
  library(htmltools)
  library(jsonlite)
})

METHOD_COLORS <- c(full_grid = "#999999", legacy = "#1f77b4", adaptive = "#d62728")
PASS_THRESHOLD <- 3.0

# -- loading ------------------------------------------------------------------
# Channels are named by cable length; order them numerically so every plot reads
# left-to-right as increasing loss rather than alphabetically (100, 1200, 1400, 300...).
chan_len <- function(x) {
  m <- regmatches(x, regexpr("BPK_[0-9]+mm", x))
  ifelse(length(m) == 0 | m == "", NA_real_,
         as.numeric(gsub("[^0-9]", "", m)))
}

chan_label <- function(x) {
  n <- vapply(x, chan_len, numeric(1))
  ifelse(is.na(n), substr(x, 1, 12), paste0(n, "mm"))
}

load_corpus <- function(dir = "corpus_results") {
  runs_p <- file.path(dir, "runs.csv")
  if (!file.exists(runs_p)) stop("no runs.csv in ", dir, " (run corpus_sweep.py first)")
  runs <- read.csv(runs_p, stringsAsFactors = FALSE, check.names = FALSE)
  runs$len <- vapply(runs$channel, chan_len, numeric(1))
  runs$label <- chan_label(runs$channel)
  runs <- runs[order(runs$len, runs$method), ]

  probe <- NULL
  probe_p <- file.path(dir, "probe.csv")
  if (file.exists(probe_p)) {
    probe <- read.csv(probe_p, stringsAsFactors = FALSE, check.names = FALSE)
    probe$len <- vapply(probe$channel, chan_len, numeric(1))
    probe$label <- chan_label(probe$channel)
    probe <- probe[order(probe$len, probe$fom_rank), ]
  }

  summ <- NULL
  summ_p <- file.path(dir, "corpus_summary.json")
  if (file.exists(summ_p)) summ <- jsonlite::fromJSON(summ_p, simplifyVector = FALSE)

  list(runs = runs, probe = probe, summ = summ)
}

.levels <- function(runs) unique(runs$label[order(runs$len)])

# -- plots --------------------------------------------------------------------
# COM per channel, all methods overlaid. The methods coinciding exactly *is* the
# result -- the traces sit on top of one another, so use open markers of
# decreasing size to make the overlap visible rather than hidden.
plot_com_by_channel <- function(runs) {
  lv <- .levels(runs)
  sizes <- c(full_grid = 16, legacy = 11, adaptive = 6)
  p <- plot_ly()
  for (nm in names(METHOD_COLORS)) {
    d <- runs[runs$method == nm, ]
    if (nrow(d) == 0) next
    p <- add_markers(p, x = factor(d$label, levels = lv), y = d$COM_dB, name = nm,
                     marker = list(size = sizes[[nm]], symbol = "circle-open",
                                   line = list(width = 2.5, color = METHOD_COLORS[[nm]])),
                     hovertemplate = paste0(nm, "<br>%{x}<br>COM=%{y:.10f} dB<extra></extra>"))
  }
  p <- add_lines(p, x = factor(lv, levels = lv), y = rep(PASS_THRESHOLD, length(lv)),
                 name = sprintf("%.0f dB threshold", PASS_THRESHOLD),
                 line = list(color = "#2ca02c", dash = "dash", width = 1.5))
  layout(p, title = paste0("COM by channel — all three methods coincide exactly",
                           "<br><sub>markers are nested; if you see only the small red ",
                           "one, all three agree</sub>"),
         xaxis = list(title = "channel (cable length)", type = "category"),
         yaxis = list(title = "COM (dB)"))
}

# dCOM vs full_grid. Every value is expected to be exactly 0, which plots as a flat
# line at zero and looks like nothing -- so band the plot at the regret scale to give
# "zero" its meaning: the pruning error is not merely small, it is absent, while the
# proxy error every method pays is an order of magnitude larger.
plot_dcom <- function(runs, probe_max_regret = NULL) {
  lv <- .levels(runs)
  d <- runs[runs$method != "full_grid", ]
  if (nrow(d) == 0) return(NULL)
  p <- plot_ly()
  if (!is.null(probe_max_regret) && is.finite(probe_max_regret) && probe_max_regret > 0) {
    p <- add_ribbons(p, x = factor(lv, levels = lv),
                     ymin = rep(-probe_max_regret, length(lv)),
                     ymax = rep(probe_max_regret, length(lv)),
                     name = sprintf("worst FOM-proxy regret (+/-%.4f dB)", probe_max_regret),
                     line = list(width = 0), fillcolor = "rgba(255,165,0,0.18)")
  }
  for (nm in unique(d$method)) {
    dd <- d[d$method == nm, ]
    p <- add_markers(p, x = factor(dd$label, levels = lv), y = dd$dCOM_vs_full_grid,
                     name = nm, marker = list(size = 11, color = METHOD_COLORS[[nm]]),
                     hovertemplate = paste0(nm, "<br>%{x}<br>dCOM=%{y:.10f} dB<extra></extra>"))
  }
  layout(p, title = paste0("dCOM vs full_grid — zero on every channel",
                           "<br><sub>shaded band = the FOM-proxy error that full_grid ",
                           "itself pays, for scale</sub>"),
         xaxis = list(title = "channel (cable length)", type = "category"),
         yaxis = list(title = "dCOM vs full_grid (dB)"))
}

plot_speedup <- function(runs) {
  lv <- .levels(runs)
  d <- runs[runs$method != "full_grid", ]
  if (nrow(d) == 0) return(NULL)
  p <- plot_ly()
  for (nm in unique(d$method)) {
    dd <- d[d$method == nm, ]
    p <- add_bars(p, x = factor(dd$label, levels = lv), y = dd$speedup_vs_full_grid,
                  name = nm, marker = list(color = METHOD_COLORS[[nm]]),
                  text = sprintf("%.1f%% of grid", dd$pct_of_full_grid_evaluated),
                  hovertemplate = "%{x}<br>%{y:.2f}x<br>%{text}<extra></extra>")
  }
  layout(p, title = "Speedup vs full grid",
         xaxis = list(title = "channel (cable length)", type = "category"),
         yaxis = list(title = "speedup (x)"), barmode = "group")
}

# Regret is the headline of the COM side: where FOM's top pick is not COM's best.
plot_regret <- function(dir, runs) {
  lv <- .levels(runs)
  rows <- list()
  for (cid in unique(runs$channel)) {
    p <- file.path(dir, cid, "fom_com_probe_summary.json")
    if (!file.exists(p)) next
    s <- jsonlite::fromJSON(p, simplifyVector = FALSE)
    if (is.null(s[["COM_regret_dB"]])) next
    rows[[length(rows) + 1]] <- data.frame(
      label = chan_label(cid), regret = s$COM_regret_dB,
      rho = s[["spearman_rho_FOM_vs_COM"]] %||% NA_real_,
      rank = s[["com_argmax_fom_rank"]] %||% NA_real_,
      stringsAsFactors = FALSE)
  }
  if (length(rows) == 0) return(NULL)
  d <- do.call(rbind, rows)
  d <- d[order(match(d$label, lv)), ]
  cols <- ifelse(d$regret > 0, "#d62728", "#bbbbbb")
  p <- plot_ly() |>
    add_bars(x = factor(d$label, levels = lv), y = d$regret, name = "COM regret",
             marker = list(color = cols),
             text = sprintf("rho=%.3f, COM argmax at FOM rank %g", d$rho, d$rank),
             hovertemplate = "%{x}<br>regret=%{y:.6f} dB<br>%{text}<extra></extra>")
  layout(p, title = paste0("FOM-proxy regret by channel",
                           "<br><sub>red = FOM's top pick was NOT the COM optimum; ",
                           "this cost is paid by every method, full_grid included</sub>"),
         xaxis = list(title = "channel (cable length)", type = "category"),
         yaxis = list(title = "COM regret (dB)"))
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# All channels overlaid in normalised proxy space: how far COM moves as you walk down
# the FOM ranking. A faithful proxy would keep every trace in the lower-left quadrant.
plot_proxy_overlay <- function(probe) {
  if (is.null(probe) || nrow(probe) == 0) return(NULL)
  lv <- unique(probe$label[order(probe$len)])
  pal <- viridisLite::viridis(length(lv))
  p <- plot_ly()
  for (i in seq_along(lv)) {
    d <- probe[probe$label == lv[i], ]
    p <- add_markers(p, x = d$dFOM_vs_rank1, y = d$dCOM_vs_rank1, name = lv[i],
                     marker = list(size = 7, color = pal[i]),
                     hovertemplate = paste0(lv[i], "<br>FOM rank %{text}",
                                            "<br>dFOM=%{x:.4f}<br>dCOM=%{y:.4f}<extra></extra>"),
                     text = d$fom_rank)
  }
  p <- add_lines(p, x = c(min(probe$dFOM_vs_rank1), 0), y = c(0, 0),
                 name = "COM of the FOM winner", showlegend = TRUE,
                 line = list(color = "#333333", dash = "dot", width = 1.5))
  layout(p, title = paste0("Walking down the FOM ranking: what COM does",
                           "<br><sub>points ABOVE the dotted line are candidates with ",
                           "worse FOM but BETTER COM than the search's pick</sub>"),
         xaxis = list(title = "dFOM vs FOM rank 1 (dB)"),
         yaxis = list(title = "dCOM vs FOM rank 1 (dB)"),
         legend = list(title = list(text = "channel")))
}

# -- tables -------------------------------------------------------------------
.fmt <- function(x, dig = 4) {
  if (is.null(x) || length(x) == 0 || (length(x) == 1 && is.na(x))) return("--")
  if (is.logical(x)) return(if (isTRUE(x)) "yes" else "no")
  if (is.numeric(x)) return(formatC(x, format = "f", digits = dig))
  as.character(x)
}

.html_table <- function(header, rows) {
  tags$table(
    style = "border-collapse:collapse;font-family:monospace;font-size:13px;margin:8px 0",
    tags$thead(tags$tr(lapply(header, function(h)
      tags$th(h, style = "border:1px solid #ccc;padding:4px 8px;background:#f0f0f0;text-align:left")))),
    tags$tbody(lapply(rows, function(r) tags$tr(lapply(r, function(c)
      tags$td(HTML(as.character(c)), style = "border:1px solid #ccc;padding:4px 8px"))))))
}

corpus_table <- function(summ) {
  if (is.null(summ) || is.null(summ$methods)) return(NULL)
  rows <- list()
  for (nm in names(summ$methods)) {
    if (nm == "full_grid") next
    m <- summ$methods[[nm]]
    sp <- m$speedup; ad <- m$abs_dCOM_vs_full_grid
    flips <- m$n_passfail_flips
    rows[[length(rows) + 1]] <- list(
      sprintf("<b>%s</b>", nm),
      sprintf("%sx (%s-%s)", .fmt(sp$median, 2), .fmt(sp$min, 2), .fmt(sp$max, 2)),
      .fmt(ad$median, 6), .fmt(ad$p95, 6), sprintf("<b>%s</b>", .fmt(ad$max, 6)),
      sprintf("%s/%s", m$n_exact_zero_dCOM, m$n_channels),
      sprintf("%s/%s", m$n_same_EQ, m$n_same_EQ + m$n_diff_EQ),
      if (flips == 0) "none" else sprintf("<b style='color:#d62728'>%s</b>", flips))
  }
  .html_table(c("method", "speedup (median, range)", "median |dCOM|", "P95 |dCOM|",
                "MAX |dCOM|", "exact-zero dCOM", "same EQ point",
                sprintf("pass/fail flips @ %.0f dB", summ$threshold_dB)), rows)
}

proxy_table <- function(summ) {
  fp <- if (is.null(summ)) NULL else summ$fom_proxy
  if (is.null(fp)) return(NULL)
  rg <- fp$COM_regret_dB; rh <- fp$spearman_rho_FOM_vs_COM
  .html_table(c("quantity", "median", "P95", "max", "min"), list(
    list("COM regret (dB)", .fmt(rg$median, 6), .fmt(rg$p95, 6),
         sprintf("<b>%s</b>", .fmt(rg$max, 6)), .fmt(rg$min, 6)),
    list("Spearman rho(FOM, COM)", .fmt(rh$median), .fmt(rh$p95), .fmt(rh$max), .fmt(rh$min))))
}

# -- dashboard ----------------------------------------------------------------
build_corpus_dashboard <- function(dir = "corpus_results",
                                   out_html = file.path(dir, "corpus_report.html")) {
  cc <- load_corpus(dir)
  runs <- cc$runs; probe <- cc$probe; summ <- cc$summ

  max_regret <- NULL
  if (!is.null(summ) && !is.null(summ$fom_proxy)) max_regret <- summ$fom_proxy$COM_regret_dB$max

  n_chan <- length(unique(runs$channel))
  page <- tagList(
    tags$h1("EQ-search method comparison — corpus"),
    tags$p(sprintf("%d channels | methods: %s", n_chan,
                   paste(unique(runs$method), collapse = ", "))),

    tags$h2("Does pruning change COM?"),
    corpus_table(summ),
    plot_dcom(runs, max_regret),
    plot_com_by_channel(runs),

    tags$h2("What does pruning buy?"),
    plot_speedup(runs),

    tags$h2("Is FOM a faithful proxy for COM?"),
    tags$p(paste("This cost is not attributable to the local search — exhaustive",
                 "full-grid pays it too, because every method optimises FOM.")),
    proxy_table(summ),
    plot_regret(dir, runs),
    plot_proxy_overlay(probe)
  )
  save_html(page, out_html)
  message("Wrote ", out_html)
  invisible(out_html)
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  dir <- if (length(args) >= 1) args[1] else "corpus_results"
  out <- if (length(args) >= 2) args[2] else file.path(dir, "corpus_report.html")
  build_corpus_dashboard(dir, out)
}
