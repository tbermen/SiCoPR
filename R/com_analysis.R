# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

# com_analysis.R -- interactive R/Plotly visualisation of a COM engineering .mat
# snapshot produced by `python sicopr.py ... --export-mat`.
#
# Dependencies:  install.packages(c("R.matlab", "plotly", "htmltools"))
#
# Usage:
#   Rscript R/com_analysis.R <run>_case01.mat [out_report.html]
#
# Or interactively:
#   source("R/com_analysis.R")
#   dat <- load_com("results/.../<run>_case01.mat")
#   plot_channel_fd(dat)            # any single plot
#   build_dashboard("<run>_case01.mat")   # full HTML report


suppressPackageStartupMessages({
  library(R.matlab)
  library(plotly)
  library(htmltools)
})

# -- loading + small accessors ------------------------------------------------
load_com <- function(matfile) readMat(matfile)

# top-level variable accessor. R.matlab rewrites every '_' in a name to '.'
# (f_GHz -> f.GHz), so try the name as given and the dotted variant.
g0 <- function(dat, name) {
  x <- dat[[name]]
  if (is.null(x)) x <- dat[[gsub("_", ".", name)]]
  x
}
has0 <- function(dat, name) !is.null(g0(dat, name))

# Wrap a plot that may be absent. A stage plot returns NULL when its data is not
# in the .mat -- an export written before the stage plots existed, or a run with
# TDR disabled -- and a NULL child would otherwise abort the whole report. The
# note is deliberate: a silently missing stage is what this work set out to fix.
optional <- function(p, height = 700) {
  if (is.null(p))
    return(tags$p(style = "color:#888;font-style:italic;",
                  "(no data for this stage in the .mat export)"))
  tags$div(style = sprintf("height:%dpx;", height), p)
}

# top-level vector (drops the Nx1 column shape R.matlab returns)
v  <- function(dat, name) as.numeric(g0(dat, name))
cv <- function(dat, name) as.complex(g0(dat, name))
db <- function(H) 20 * log10(Mod(as.complex(H)) + 1e-300)

# pull a field out of a 1x1 struct, robust to R.matlab's list wrapping and to
# whether it keeps underscores or rewrites them to dots in field names.
gf <- function(s, name) {
  if (!is.null(dim(s)) && length(dim(s)) == 3) s <- s[, , 1]
  cand <- unique(c(name, gsub("_", ".", name), gsub(".", "_", name, fixed = TRUE)))
  val <- NULL
  for (nm in cand) {
    val <- tryCatch(s[[nm]], error = function(e) NULL)
    if (!is.null(val)) break
  }
  if (is.list(val) && length(val) == 1) val <- val[[1]]
  val
}

# -- helpers for the collapsible results / config-metadata tables -------------
# format a single field value (scalar, short vector, or string) for display
.fmt_val <- function(x) {
  if (is.null(x)) return("")
  if (is.list(x)) {
    if (length(x) == 1) x <- x[[1]]
    else return(paste(vapply(x, .fmt_val, character(1)), collapse = "; "))
  }
  if (is.character(x)) return(paste(x, collapse = ", "))
  if (is.complex(x)) x <- Mod(x)
  x <- suppressWarnings(as.numeric(x)); x <- x[!is.na(x)]
  if (length(x) == 0) return("")
  if (length(x) == 1) return(formatC(x, format = "g", digits = 6))
  if (length(x) > 10)
    return(sprintf("[%d values] %s ...", length(x),
                   paste(formatC(utils::head(x, 5), format = "g", digits = 4), collapse = ", ")))
  paste(formatC(x, format = "g", digits = 5), collapse = ", ")
}

# normalise an R.matlab struct node to a named list of fields, or NULL if it is
# not a struct. R.matlab returns a 1x1 struct as a dimensioned list-array (e.g.
# 180x1x1) whose field names appear only after dropping the trailing dims, so a
# plain names() test misses nested structs (param, ctle_settings, ...).
.fields <- function(x) {
  if (is.null(x)) return(NULL)
  if (!is.null(dim(x)) && length(dim(x)) == 3) x <- x[, , 1]
  if (is.list(x) && !is.null(names(x)) && all(nzchar(names(x)))) return(x)
  NULL
}

# flatten a (possibly nested) struct into a list of c(label, value) rows, so
# every value carries its parameter name (dotted for nested sub-structs).
.struct_rows <- function(s, prefix = "") {
  flds <- .fields(s)
  if (is.null(flds)) return(list())
  out <- list()
  for (nm in names(flds)) {
    val <- flds[[nm]]
    if (is.list(val) && length(val) == 1 && is.null(names(val))) val <- val[[1]]
    key <- if (nzchar(prefix)) paste0(prefix, ".", nm) else nm
    sub <- .fields(val)
    if (!is.null(sub)) {
      out <- c(out, .struct_rows(val, key))           # recurse into sub-struct
    } else {
      out <- c(out, list(c(key, .fmt_val(val))))
    }
  }
  out
}

# render label/value rows as a compact HTML table (pairs_per_row label:value pairs)
.kv_table <- function(rows, pairs_per_row = 2) {
  if (length(rows) == 0) return(tags$p("(none)"))
  cell <- function(r) list(
    tags$td(tags$b(r[1]), style = "padding:2px 8px; border-bottom:1px solid #eee; white-space:nowrap;"),
    tags$td(r[2],        style = "padding:2px 16px 2px 0; border-bottom:1px solid #eee;"))
  trs <- list(); i <- 1
  while (i <= length(rows)) {
    grp <- rows[i:min(i + pairs_per_row - 1, length(rows))]
    trs[[length(trs) + 1]] <- do.call(tags$tr, do.call(c, lapply(grp, cell)))
    i <- i + pairs_per_row
  }
  do.call(tags$table,
          c(list(style = "border-collapse:collapse; font-family:monospace; font-size:12px;"), trs))
}

# wrap content in a <details> block, collapsed by default
.collapsible <- function(title, body) {
  tags$details(
    tags$summary(title, style = "cursor:pointer; font-family:sans-serif; font-weight:bold; margin:14px 0 6px;"),
    body)
}

# -- Plot 1: channel frequency response (cumulative EQ chain) -----------------
plot_channel_fd <- function(dat) {
  fG <- v(dat, "f_GHz")
  p <- plot_ly(height = 700)
  p <- add_lines(p, x = fG, y = db(cv(dat, "H_channel")), name = "Channel only")
  p <- add_lines(p, x = fG, y = db(cv(dat, "H_tx")),      name = "Channel + Tx FFE")
  p <- add_lines(p, x = fG, y = db(cv(dat, "H_final")),   name = "Channel + Tx FFE + CTLE (final)")
  layout(p, title = "Channel frequency response",
         xaxis = list(
           title = "Frequency [GHz]",
           type = "log",
           showgrid = TRUE,
           minor = list(
             ticks = "inside",
             showgrid = TRUE
           )
         ),
         yaxis = list(title = "Magnitude [dB]",
                      range = c(-60, 10)
         ),
         hovermode = "x unified")
}

# -- Plot 2: CTLE response ----------------------------------------------------
# plot_ctle <- function(dat) {
#   fG <- v(dat, "f_GHz")
#   g  <- db(cv(dat, "H_ctle"))
#   plot_ly(x = fG, y = g, type = "scatter", mode = "lines", name = "CTLE",
#           hovertemplate = "f = %{x:.2f} GHz<br>gain = %{y:.2f} dB<extra></extra>") |>
#     layout(title = "CTLE response",
#            xaxis = list(title = "Frequency [GHz]"),
#            yaxis = list(title = "CTLE gain [dB]"))
# }

plot_ctle <- function(dat) {
  fG <- v(dat, "f_GHz")
  idx <- fG > 0
  fG <- fG[idx]

  cpar <- g0(dat, "ctle_parameters")
  settings <- sprintf("gdc=%.1f dB, fz=%.2f GHz, fp1=%.2f GHz, fp2=%.2f GHz",
                      as.numeric(gf(cpar, "gdc_dB")),
                      as.numeric(gf(cpar, "fz_Hz")) / 1e9,
                      as.numeric(gf(cpar, "fp1_Hz")) / 1e9,
                      as.numeric(gf(cpar, "fp2_Hz")) / 1e9)

  p <- plot_ly(height = 700)
  p <- add_lines(p, x = fG, y = db(cv(dat, "H_ctle"))[idx], name = "CTLE",
                 hovertemplate = "f = %{x:.3f} GHz<br>gain = %{y:.2f} dB<extra></extra>")
  if (has0(dat, "H_ctle_rx"))   # CTLE cascaded with the Rx bandwidth-limiting filter
    p <- add_lines(p, x = fG, y = db(cv(dat, "H_ctle_rx"))[idx], name = "CTLE + Rx filter",
                   hovertemplate = "f = %{x:.3f} GHz<br>gain = %{y:.2f} dB<extra></extra>")
  layout(
    p,
    title = paste0("CTLE response - ", settings),
    xaxis = list(
      title = "Frequency [GHz]",
      type = "log",
      showgrid = TRUE,
      minor = list(ticks = "inside", showgrid = TRUE)
    ),
    yaxis = list(title = "Gain [dB]")
  )
}

# -- Plot 3: impulse responses ------------------------------------------------
# plot_impulse <- function(dat) {
#   t <- v(dat, "t_ns")
#   p <- plot_ly()
#   p <- add_lines(p, x = t, y = v(dat, "h_channel"), name = "Channel")
#   if (has0(dat, "h_ctle")) p <- add_lines(p, x = t, y = v(dat, "h_ctle"), name = "Channel + CTLE")
#   if (has0(dat, "h_final")) p <- add_lines(p, x = t, y = v(dat, "h_final"), name = "Final (EQ)")
#   layout(p, title = "Impulse responses",
#          xaxis = list(title = "Time [ns]"),
#          yaxis = list(title = "Amplitude"))
# }

plot_impulse <- function(dat) {
  
  t  <- v(dat, "t_ns")
  hc <- v(dat, "h_channel")
  
  # Find peak location of channel impulse
  t_peak <- t[which.max(abs(hc))]
  
  p <- plot_ly(height = 700)
  p <- add_lines(p, x = t, y = hc, name = "Channel")
  
  if (has0(dat, "h_ctle"))     p <- add_lines(p, x = t, y = v(dat, "h_ctle"),  name = "Channel + CTLE")
  if (has0(dat, "h_ctle_ffe")) p <- add_lines(p, x = t, y = v(dat, "h_ctle"),  name = "Channel + CTLE + Tx FFE")
  if (has0(dat, "h_final"))    p <- add_lines(p, x = t, y = v(dat, "h_final"), name = "Final (EQ)")
  
  layout(
    p,
    title = "Impulse responses",
    xaxis = list(
      title = "Time [ns]",
      range = c(t_peak - 0.25, t_peak + 0.75)
    ),
    yaxis = list(title = "Amplitude")
  )
}

# -- Plot 4: pulse responses (single-bit) -------------------------------------
plot_pulse <- function(dat) {
  t  <- v(dat, "t_ns")
  hc <- v(dat, "pulse_channel")
  
  # Find peak location of channel impulse
  t_peak <- t[which.max(abs(hc))]
  
  cs <- as.integer(g0(dat, "cursor_sample_index"))  # 1-based
  M  <- as.integer(g0(dat, "samples_per_ui"))       # samples per UI -> 1 UI spacing
  p <- plot_ly(height = 700)
  p <- add_lines(p, x = t, y = v(dat, "pulse_channel"),   name = "Channel")

  if (has0(dat, "pulse_ctle"))     p <- add_lines(p, x = t, y = v(dat, "pulse_ctle"),      name = "Channel + CTLE")
  if (has0(dat, "pulse_ctle_ffe")) p <- add_lines(p, x = t, y = v(dat, "pulse_ctle_ffe"),  name = "Channel + CTLE + Tx FFE")
  if (has0(dat, "pulse_final"))    p <- add_lines(p, x = t, y = v(dat, "pulse_final"),     name = "Channel + CTLE + Tx FFE (EQ, pre-DFE)")

  # Channel + CTLE + Tx FFE + DFE: the DFE feedback for post-cursor tap n is held
  # constant over the whole nth UI symbol period (not just the sample instant), so
  # the cancellation dfe_taps[n] * pulse_final[cursor] is subtracted across every
  # sample of that UI. This produces the characteristic DFE staircase, with a
  # discontinuity at each UI boundary. COM never stores a DFE-applied waveform
  # (the DFE acts on the statistical PDF), so we derive it here from the genuine
  # equalized SBR (pulse_final), its cursor, and the selected DFE taps.
  b   <- as.numeric(g0(dat, "dfe_taps"))
  if (has0(dat, "pulse_final") && !is.na(cs) && !is.na(M) && M > 0 &&
      length(b) > 0 && any(b != 0)) {
    pdfe <- v(dat, "pulse_final")
    h0   <- pdfe[cs]
    # UI index of each sample relative to the cursor; tap n owns the full UI window
    # round((i - cs)/M) == n, giving non-overlapping M-sample steps.
    ui_idx <- round((seq_along(pdfe) - cs) / M)
    dfe_k <- integer(0)
    for (n in seq_along(b)) {
      sel <- which(ui_idx == n)
      if (length(sel) > 0) {
        pdfe[sel] <- pdfe[sel] - b[n] * h0
        k <- cs + n * M                       # sample instant for the hover marker
        if (k >= 1 && k <= length(pdfe)) dfe_k <- c(dfe_k, k)
      }
    }
    p <- add_lines(p, x = t, y = pdfe, name = "Channel + CTLE + Tx FFE + DFE",
                   line = list(color = "black"))
    if (length(dfe_k) > 0)
      p <- add_markers(p, x = t[dfe_k], y = pdfe[dfe_k],
                       name = "DFE-cancelled cursors", legendgroup = "dfe",
                       marker = list(color = "black", symbol = "circle-open", size = 7),
                       hovertemplate = "t = %{x:.3f} ns<br>post-DFE SBR = %{y:.4f}<extra></extra>")
  }

  # Cursor + pre/post-cursor markers: vertical dotted lines. The cursor spans the
  # full final-pulse range; pre/post-cursor lines start at the Channel SBR value
  # at that UI and drop below the x-axis (per request).
  pch  <- v(dat, "pulse_channel")
  pfin <- v(dat, "pulse_final")
  ylo  <- min(pfin)
  shapes <- list(); anns <- list()
  add_marker <- function(shapes, anns, k, y_top, color, label) {
    if (is.na(k) || k < 1 || k > length(t)) return(list(shapes, anns))
    shapes[[length(shapes) + 1]] <- list(type = "line", x0 = t[k], x1 = t[k],
                                         y0 = ylo, y1 = y_top,
                                         line = list(color = color, dash = "dot"))
    anns[[length(anns) + 1]] <- list(x = t[k], y = y_top, text = label,
                                     showarrow = TRUE, font = list(color = color))
    list(shapes, anns)
  }
  # 6/23/2026: Removing marker text because chart is too crowded and details are not necessary for experienced viewer.
  # if (!is.na(cs)) {
  #   r <- add_marker(shapes, anns, cs,      max(pfin),                 "red",    "cursor / sampling point"); shapes <- r[[1]]; anns <- r[[2]]
  #   r <- add_marker(shapes, anns, cs - M,  pch[max(cs - M, 1)],       "orange", "pre-cursor");              shapes <- r[[1]]; anns <- r[[2]]
  #   r <- add_marker(shapes, anns, cs + M,  pch[min(cs + M, length(pch))], "purple", "post-cursor");         shapes <- r[[1]]; anns <- r[[2]]
  # }

  # Sampling-point lollipop (toggleable via legend, ON by default): the channel
  # SBR sampled at the main cursor and at integer-UI offsets -- 10 pre-cursor
  # (left), the main cursor, and 100 post-cursor (right). Stems drop to y=0.
  if (!is.na(cs) && !is.na(M) && M > 0) {
    ks  <- seq(-10, 100)                       # UI offsets from the main cursor
    idx <- cs + ks * M
    keep <- idx >= 1 & idx <= length(pch)
    idx <- idx[keep]
    xs  <- t[idx]
    ys  <- pch[idx]                            # magnitude of channel SBR
    # one stem trace: x/y pairs separated by NA so each stem is its own segment
    stem_x <- as.vector(rbind(xs, xs, NA))
    stem_y <- as.vector(rbind(rep(0, length(ys)), ys, NA))
    p <- add_lines(p, x = stem_x, y = stem_y, name = "SBR sampling points",
                   legendgroup = "samp", # visible = "legendonly",
                   line = list(color = "black", width = 1), hoverinfo = "skip")
    p <- add_markers(p, x = xs, y = ys, name = "SBR sampling points",
                     # legendgroup = "samp", visible = "legendonly", showlegend = FALSE,
                     legendgroup = "samp", showlegend = FALSE,
                     marker = list(color = "black", size = 6),
                     hovertemplate = "t = %{x:.3f} ns<br>SBR = %{y:.4f}<extra></extra>")
  }

  layout(p, title = "Pulse responses (single-bit)",
         shapes = shapes, annotations = anns,
         xaxis = list(
           title = "Time [ns]",
           range = c(t_peak - 0.25, t_peak + 0.75)
         ),
         yaxis = list(title = "Amplitude"))
}

# -- why the eye / bathtub panels can be legitimately empty --------------------
# COM_eye_width is what produces the eye contour AND the timing bathtub, and both
# MATLAB (com_ieee8023_4p15p0.m L620) and this port gate it identically:
#     OP.RX_CALIBRATION == 0 && OP.EW == 1 && OP.MLSE == 0
# So a run with MLSE enabled has no eye data in EITHER tool -- it is a
# configuration consequence, not a missing export. Say so rather than showing an
# unexplained blank panel.
.eye_absent_note <- function(dat, what) {
  rf <- tryCatch(g0(dat, "results_full"), error = function(e) NULL)
  der <- tryCatch(as.numeric(gf(rf, "DER_MLSE")), error = function(e) NA)
  why <- if (length(der) > 0 && !is.na(der[1]))
    paste0("MLSE is enabled for this run (DER_MLSE = ", sprintf("%.3g", der[1]),
           "). MATLAB gates COM_eye_width on OP.MLSE == 0 (4p15p0 L620) and this ",
           "port follows it, so neither tool emits an eye by default. That is a ",
           "reporting choice, not a limitation: MLSE is applied afterwards ",
           "(L667), so the pre-MLSE (DFE-only) eye is well defined. To plot it, ",
           "re-run with  sicopr.EYE_PLOT_UNDER_MLSE = True  -- diagnostic only, it ",
           "changes no reported COM, VEC, VEO or EW value.")
  else
    paste0("No eye data in this .mat. COM_eye_width runs only when ",
           "RX_CALIBRATION = 0, EW = 1 and MLSE = 0.")
  plot_ly(height = 420) |>
    layout(title = paste0(what, " - not available for this run"),
           xaxis = list(visible = FALSE), yaxis = list(visible = FALSE),
           annotations = list(list(text = why, showarrow = FALSE,
                                   x = 0.5, y = 0.5, xref = "paper", yref = "paper",
                                   align = "center",
                                   font = list(size = 13, color = "#444"))))
}

# -- Plot 5: eye diagram (statistical BER contour at DER) ---------------------
plot_eye <- function(dat) {
  eye <- tryCatch(g0(dat, "eye"), error = function(e) NULL)
  ec <- tryCatch(as.matrix(gf(eye, "eye_contour")), error = function(e) NULL)
  if (is.null(ec) || length(ec) == 0) return(.eye_absent_note(dat, "Eye diagram"))
  phase <- as.numeric(gf(eye, "phase_UI"))
  p <- plot_ly(height = 1100)
  ncol_pairs <- ncol(ec) %/% 2
  for (i in seq_len(ncol_pairs)) {
    p <- add_lines(p, x = phase, y = ec[, 2 * i - 1] * 1000, name = paste0("eye", i, " upper"),
                   line = list(color = "green"))
    p <- add_lines(p, x = phase, y = ec[, 2 * i] * 1000, name = paste0("eye", i, " lower"),
                   line = list(color = "blue"))
  }
  layout(p, title = "Eye diagram (BER contour at DER)",
         xaxis = list(title = "Sample phase [UI from centre]", range = c(-0.2, 0.2)),
         yaxis = list(title = "Voltage [mV]"),
         legend = list(x = 1.06),
         hovermode = "closest")
}

# -- Plot 5b: timing bathtub (BER density vs sample phase) --------------------
# The bathtub is a distinct chart from the eye diagram: BER along the horizontal
# scan at each eye centre, on a log axis.
# -- Plot 5c: voltage bathtub (BER vs decision threshold, centre phase) -------
# One curve per eye, each centred on its own PAM level. Distinct from the timing
# bathtub: this sweeps the DECISION THRESHOLD at the optimum phase, where the
# timing bathtub sweeps PHASE at each eye's fixed threshold.
plot_voltage_bathtub <- function(dat) {
  eye <- tryCatch(g0(dat, "eye"), error = function(e) NULL)
  vb <- tryCatch(as.matrix(gf(eye, "vbt_ber")), error = function(e) NULL)
  vax <- tryCatch(as.numeric(gf(eye, "vbt_threshold_V")), error = function(e) NULL)
  if (is.null(vb) || length(vb) == 0 || is.null(vax) || length(vax) == 0)
    return(.eye_absent_note(dat, "Voltage bathtub"))
  if (nrow(vb) != length(vax) && ncol(vb) == length(vax)) vb <- t(vb)
  nm <- if (ncol(vb) == 3) c("lower", "central", "upper") else
    paste0("eye", seq_len(ncol(vb)))
  p <- plot_ly(height = 700)
  for (i in seq_len(ncol(vb)))
    p <- add_lines(p, x = vax * 1000, y = pmax(vb[, i], 1e-20),
                   name = paste(nm[i], "eye"),
                   hovertemplate = "threshold = %{x:.3f} mV<br>BER = %{y:.2e}<extra></extra>")
  layout(p, title = "Voltage bathtub (BER vs decision threshold, at the centre phase)",
         xaxis = list(title = "Decision threshold [mV]"),
         yaxis = list(title = "BER", type = "log", dtick = 1,
                      exponentformat = "power", showexponent = "all"),
         legend = list(x = 1.02), hovermode = "closest")
}

plot_bathtub <- function(dat) {
  eye <- tryCatch(g0(dat, "eye"), error = function(e) NULL)
  ber <- tryCatch(as.matrix(gf(eye, "ber_eyes")), error = function(e) NULL)
  if (is.null(ber) || length(ber) == 0)
    return(.eye_absent_note(dat, "Timing bathtub"))
  phase <- as.numeric(gf(eye, "phase_UI"))
  if (nrow(ber) != length(phase) && ncol(ber) == length(phase)) ber <- t(ber)
  p <- plot_ly(height = 700)
  for (i in seq_len(ncol(ber))) {
    p <- add_lines(p, x = phase, y = pmax(ber[, i], 1e-20), name = paste0("BER eye", i),
                   hovertemplate = "phase = %{x:.3f} UI<br>BER = %{y:.2e}<extra></extra>")
  }
  layout(p, title = "Timing bathtub (BER density vs sample phase)",
         xaxis = list(title = "Sample phase [UI from centre]", range = c(-0.2, 0.2)),
         yaxis = list(title = "BER", type = "log",
                      dtick = 1,                     # one tick per decade
                      exponentformat = "power",      # scientific labels: 10^-1, 10^-2, ...
                      showexponent = "all"),
         hovermode = "x unified")
}

# -- Plot 6: equalizer contribution (cumulative) ------------------------------
plot_eq_contribution <- function(dat) {
  fG <- v(dat, "f_GHz")
  p <- plot_ly(height = 700)
  p <- add_lines(p, x = fG, y = db(cv(dat, "H_channel")), name = "1. Channel")
  p <- add_lines(p, x = fG, y = db(cv(dat, "H_tx")),      name = "2. + Tx FFE")
  if (has0(dat, "H_ch_ctle"))
    p <- add_lines(p, x = fG, y = db(cv(dat, "H_ch_ctle")), name = "3. + CTLE (no Tx FFE)")
  p <- add_lines(p, x = fG, y = db(cv(dat, "H_final")), name = "4. Final response")
  layout(p, title = "Equalizer contribution (cumulative)",
         xaxis = list(
           title = "Frequency [GHz]",
           type = "log",
           showgrid = TRUE,
           minor = list(
             ticks = "inside",
             showgrid = TRUE
           )
         ),
         yaxis = list(title = "Magnitude [dB]",
                      range = c(-60, 10)
                      ), hovermode = "x unified")
}

# -- Plot 6b: all available CTLE curves (the selectable CTLE bank) ------------
plot_ctle_bank <- function(dat) {
  fG <- v(dat, "f_GHz")
  idx <- fG > 0
  fGp <- fG[idx]
  H <- g0(dat, "H_ctle_all")          # (n_ctle x n_freq) complex
  if (is.null(H)) return(plot_ly() |> layout(title = "CTLE bank (no H_ctle_all in .mat)"))
  H <- matrix(as.complex(H), nrow = dim(H)[1])
  gdc <- tryCatch(v(dat, "ctle_gdc_values"), error = function(e) seq_len(nrow(H)))
  sel <- tryCatch(as.integer(g0(dat, "ctle_index")), error = function(e) NA)
  p <- plot_ly(height = 700)
  for (i in seq_len(nrow(H))) {
    is_sel <- (!is.na(sel) && i == sel)
    p <- add_lines(p, x = fGp, y = db(H[i, ])[idx],
                   name = sprintf("gdc=%.1f dB%s", gdc[i], if (is_sel) " (selected)" else ""),
                   line = list(width = if (is_sel) 4 else 1))
  }
  layout(p, title = "CTLE bank - all available DC-gain settings (no high-pass stage)",
         xaxis = list(title = "Frequency [GHz]", type = "log",
                      showgrid = TRUE, minor = list(ticks = "inside", showgrid = TRUE)),
         yaxis = list(title = "Gain [dB]"), hovermode = "closest")
}

# -- Plot 7: COM dashboard (single HTML report) -------------------------------
# -- Stage 2: TDR impedance profile ------------------------------------------
# The reference line is 2 * Z_t, not Z_t: the exporter has already doubled it
# (Z_t is single-ended, the ZSR trace is differential). A first version of the
# PNG equivalent drew the undoubled value and put the band at 46 ohm under a
# trace at 97, which is worse than drawing no reference at all.
# Min/max envelope decimation. The TDR trace is ~224k points per port; plotly
# keeps every point as JSON, so plotting it raw produced a 95 MB report that was
# slow to open. Naive stride-sampling would drop the connector and via
# discontinuities, which are exactly the narrow spikes this plot exists to show,
# so each output bucket keeps its own min AND max. Spikes survive; the point
# count does not.
.envelope <- function(x, y, n_out = 3000) {
  n <- length(y)
  if (n <= n_out * 2) return(list(x = x, y = y))
  b <- ceiling(n / n_out)
  idx <- split(seq_len(n), ceiling(seq_len(n) / b))
  keep <- unlist(lapply(idx, function(i) {
    if (length(i) == 1) return(i)
    unique(c(i[which.min(y[i])], i[which.max(y[i])]))
  }), use.names = FALSE)
  keep <- sort(unique(keep))
  list(x = x[keep], y = y[keep])
}

plot_tdr_impedance <- function(dat) {
  td <- tryCatch(g0(dat, "tdr"), error = function(e) NULL)
  if (is.null(td)) return(NULL)
  p <- plot_ly(height = 700)
  drew <- FALSE
  for (nm in c("TDR11", "TDR22")) {
    t_ns <- tryCatch(as.numeric(gf(td, paste0(nm, "_t_ns"))), error = function(e) NULL)
    z <- tryCatch(as.numeric(gf(td, paste0(nm, "_Z_ohm"))), error = function(e) NULL)
    if (is.null(t_ns) || is.null(z) || length(t_ns) != length(z) || !length(z)) next
    az <- tryCatch(as.numeric(gf(td, paste0(nm, "_avgZ_ohm"))), error = function(e) NA)
    e <- .envelope(t_ns, z)
    p <- add_lines(p, x = e$x, y = e$y,
                   name = sprintf("%s  (avgZ %.2f ohm, %d of %d pts)",
                                  nm, az[1], length(e$y), length(z)))
    drew <- TRUE
  }
  if (!drew) return(NULL)
  zref <- tryCatch(as.numeric(gf(td, "Z_ref_ohm")), error = function(e) NA)
  shp <- list()
  if (length(zref) && is.finite(zref[1])) {
    shp <- list(
      list(type = "rect", xref = "paper", x0 = 0, x1 = 1, yref = "y",
           y0 = zref[1] * 0.9, y1 = zref[1] * 1.1,
           fillcolor = "rgba(128,128,128,0.12)", line = list(width = 0)),
      list(type = "line", xref = "paper", x0 = 0, x1 = 1, yref = "y",
           y0 = zref[1], y1 = zref[1],
           line = list(color = "grey", dash = "dash", width = 1)))
  }
  layout(p,
         title = if (length(zref) && is.finite(zref[1]))
           sprintf("Stage 2 - TDR impedance profile (band = 2 x Z_t = %.1f ohm +/-10%%)", zref[1])
         else "Stage 2 - TDR impedance profile",
         xaxis = list(title = "Time [ns]"),
         yaxis = list(title = "Impedance [ohm]"),
         shapes = shp, hovermode = "x unified")
}

# -- Stage 2: effective return loss ------------------------------------------
plot_erl <- function(dat) {
  td <- tryCatch(g0(dat, "tdr"), error = function(e) NULL)
  if (is.null(td)) return(NULL)
  nms <- c("ERL11", "ERL11_CD", "ERL11_DC", "ERL11_CC",
           "ERL22", "ERL22_CD", "ERL22_DC", "ERL22_CC")
  lab <- c(); val <- c()
  for (nm in nms) {
    x <- tryCatch(as.numeric(gf(td, nm)), error = function(e) NULL)
    if (!is.null(x) && length(x) && is.finite(x[1])) { lab <- c(lab, nm); val <- c(val, x[1]) }
  }
  if (!length(val)) return(NULL)
  p <- plot_ly(height = 500, x = lab, y = val, type = "bar",
               marker = list(color = "#1f77b4"),
               text = sprintf("%.2f dB", val), textposition = "outside",
               hovertemplate = "%{x}<br>%{y:.4f} dB<extra></extra>")
  layout(p, title = "Stage 2 - Effective return loss (higher is better)",
         xaxis = list(title = ""), yaxis = list(title = "ERL [dB]"))
}

# -- Stage 4: FOM against sampling phase -------------------------------------
# Answers how sharp the sampling optimum is, which the reported itick alone
# cannot. A flat top means the phase choice is uncritical.
plot_fom_vs_phase <- function(dat) {
  fp <- tryCatch(g0(dat, "fom_vs_phase"), error = function(e) NULL)
  if (is.null(fp)) return(NULL)
  it <- tryCatch(as.numeric(gf(fp, "itick")), error = function(e) NULL)
  fm <- tryCatch(as.numeric(gf(fp, "FOM_dB")), error = function(e) NULL)
  if (is.null(it) || is.null(fm) || length(it) != length(fm) || length(it) < 2)
    return(NULL)
  ok <- is.finite(fm)
  if (sum(ok) < 2) return(NULL)
  sel <- tryCatch(as.numeric(gf(fp, "selected_itick")), error = function(e) NA)
  span <- max(fm[ok]) - min(fm[ok])
  p <- plot_ly(height = 700)
  p <- add_trace(p, x = it[ok], y = fm[ok], type = "scatter", mode = "lines+markers",
                 name = "best FOM at this phase",
                 marker = list(size = 5), line = list(width = 1.5),
                 hovertemplate = "itick %{x}<br>FOM %{y:.6f} dB<extra></extra>")
  shp <- list()
  if (length(sel) && is.finite(sel[1]))
    shp <- list(list(type = "line", x0 = sel[1], x1 = sel[1], yref = "paper",
                     y0 = 0, y1 = 1, line = list(color = "red", width = 1.5)))
  layout(p,
         title = sprintf("Stage 4 - FOM vs sampling phase (selected itick %s, range %.3f dB)",
                         if (length(sel) && is.finite(sel[1])) as.character(round(sel[1])) else "n/a",
                         span),
         xaxis = list(title = "Sampling phase itick [samples from the raw cursor]"),
         yaxis = list(title = "FOM [dB]"),
         shapes = shp, hovermode = "x unified")
}

# -- Stage 5: the equalizer taps actually selected ---------------------------
# A COM value cannot say whether the setting that produced it is one the
# silicon can produce. These are the vectors to read for that.
plot_eq_taps <- function(dat) {
  mk <- function(nm, key) {
    x <- tryCatch(as.numeric(g0(dat, key)), error = function(e) numeric(0))
    if (!length(x)) return(NULL)
    list(name = nm, y = x, x = seq_along(x) - 1)
  }
  series <- Filter(Negate(is.null),
                   list(mk("Tx FFE", "ffe_taps"),
                        mk("Rx FFE", "rxffe_taps"),
                        mk("DFE", "dfe_taps")))
  if (!length(series)) return(NULL)
  p <- plot_ly(height = 700)
  for (s in series)
    p <- add_trace(p, x = s$x, y = s$y, type = "scatter", mode = "markers",
                   name = sprintf("%s (%d taps)", s$name, length(s$y)),
                   marker = list(size = 7),
                   hovertemplate = paste0(s$name, " tap %{x}<br>%{y:.6f}<extra></extra>"))
  layout(p, title = "Stage 5 - Selected equalizer taps",
         xaxis = list(title = "Tap index"),
         yaxis = list(title = "Tap value"), hovermode = "closest")
}

# -- Stage 6: the individual noise terms -------------------------------------
# sigma_total was the only noise scalar the report carried, and a total cannot
# say whether the transmitter or the channel dominates.
plot_noise_terms <- function(dat) {
  nt <- tryCatch(g0(dat, "noise_terms"), error = function(e) NULL)
  if (is.null(nt)) return(NULL)
  keys <- c("sigma_TX_mV", "sigma_G_mV", "sigma_N_mV", "sigma_rjit_mV",
            "sigma_Q_mV", "sigma_hp_mV", "cci_sigma_mV", "sci_sigma_mV")
  desc <- c(sigma_TX_mV = "transmitter (SNR_TX, R_LM)",
            sigma_G_mV = "jitter, via pulse slope",
            sigma_N_mV = "receiver referred (eta_0)",
            sigma_rjit_mV = "random jitter",
            sigma_Q_mV = "quantisation",
            sigma_hp_mV = "clause 162 broadband",
            cci_sigma_mV = "co-channel interference",
            sci_sigma_mV = "self-channel interference")
  lab <- c(); val <- c()
  for (k in keys) {
    x <- tryCatch(as.numeric(gf(nt, k)), error = function(e) NULL)
    if (!is.null(x) && length(x) && is.finite(x[1]) && x[1] > 0) {
      lab <- c(lab, sprintf("%s - %s", sub("_mV$", "", k), desc[[k]]))
      val <- c(val, x[1])
    }
  }
  if (!length(val)) return(NULL)
  o <- order(val)
  lab <- lab[o]; val <- val[o]
  extra <- c()
  for (k in c("sigma_before_clip_mV", "peak_clip_mV")) {
    x <- tryCatch(as.numeric(gf(nt, k)), error = function(e) NULL)
    if (!is.null(x) && length(x) && is.finite(x[1]))
      extra <- c(extra, sprintf("%s = %.4f mV", sub("_mV$", "", k), x[1]))
  }
  p <- plot_ly(height = 600, x = val, y = factor(lab, levels = lab), type = "bar",
               orientation = "h", marker = list(color = "#1f77b4"),
               text = sprintf("%.4f mV", val), textposition = "outside",
               hovertemplate = "%{y}<br>%{x:.6f} mV<extra></extra>")
  layout(p,
         title = paste0("Stage 6 - Noise terms, largest first",
                        if (length(extra)) paste0("  (", paste(extra, collapse = ";  "), ")") else ""),
         xaxis = list(title = "sigma [mV]"),
         yaxis = list(title = "", automargin = TRUE),
         margin = list(l = 260))
}


build_dashboard <- function(matfile, out_html = NULL) {
  dat <- load_com(matfile)
  if (is.null(out_html))
    out_html <- sub("\\.mat$", "_report.html", matfile)

  num <- function(name) {
    x <- tryCatch(as.numeric(g0(dat, name)), error = function(e) NA)
    if (length(x) == 0) NA else x[1]
  }
  cpar <- g0(dat, "ctle_parameters")
  ctle_txt <- sprintf("type=%s  gdc=%.1f dB  fz=%.2f GHz  fp1=%.2f GHz  fp2=%.2f GHz",
                      as.character(gf(cpar, "CTLE_type")),
                      as.numeric(gf(cpar, "gdc_dB")),
                      as.numeric(gf(cpar, "fz_Hz")) / 1e9,
                      as.numeric(gf(cpar, "fp1_Hz")) / 1e9,
                      as.numeric(gf(cpar, "fp2_Hz")) / 1e9)
  dfe_txt <- paste(sprintf("%.4f", as.numeric(g0(dat, "dfe_taps"))), collapse = ", ")

  # results_full carries the engine's own reported outputs -- the same fields the
  # MATLAB reference workbooks record. Read them for the summary rows below.
  rfull <- tryCatch(g0(dat, "results_full"), error = function(e) NULL)
  rnum <- function(name) {
    x <- tryCatch(as.numeric(gf(rfull, name)), error = function(e) NA)
    if (length(x) == 0) NA else x[1]
  }

  # Tx FFE: state how many candidates the search evaluated alongside the winning
  # taps. Without that, an all-zero result reads as "initial values were exported"
  # when it is in fact the no-equalisation corner winning a full sweep.
  n_txffe <- tryCatch(as.numeric(gf(gf(g0(dat, "run_summary"), "sweep"), "n_TXFFE")),
                      error = function(e) NA)
  ffe_txt <- paste(sprintf("%.4f", as.numeric(g0(dat, "ffe_taps"))), collapse = ", ")
  if (!is.na(n_txffe) && n_txffe > 1)
    ffe_txt <- sprintf("%s   (winner of %d Tx FFE candidates)", ffe_txt, round(n_txffe))

  # RxFFE: 87 taps is too many for a header row, so summarise and point at the
  # full vector, which is in the collapsible results table.
  rxffe <- tryCatch(as.numeric(g0(dat, "rxffe_taps")), error = function(e) numeric(0))
  rxffe_txt <- if (length(rxffe) == 0) "not used" else
    sprintf("%d taps, cursor %.4f at index %d, peak |tap| %.4f, gain %s",
            length(rxffe), rxffe[which.max(abs(rxffe))], which.max(abs(rxffe)),
            max(abs(rxffe)),
            if (is.na(rnum("RxFFEgain"))) "n/a" else sprintf("%.4g", rnum("RxFFEgain")))

  # MLSE: reported via its error contribution. A finite DER_MLSE means the MLSE
  # path ran; it also means COM_eye_width was skipped (see the eye/bathtub note).
  der_mlse <- rnum("DER_MLSE")
  mlse_txt <- if (is.na(der_mlse)) "not enabled" else
    sprintf("enabled - DER_MLSE = %.4g", der_mlse)

  case_idx <- tryCatch(as.integer(gf(g0(dat, "meta"), "case_index")), error = function(e) NA)
  if (length(case_idx) == 0 || is.na(case_idx)) case_idx <- 1L

  # Run summary: sweep dimensions, number of cases, wall-clock time.
  rs <- g0(dat, "run_summary")
  n_cases <- tryCatch(as.integer(gf(rs, "n_cases")), error = function(e) NA)
  # Per-case run time (time spent on this case alone); fall back to the older
  # cumulative field for .mat files produced before per-case timing existed.
  elapsed <- tryCatch(as.numeric(gf(rs, "elapsed_s_this_case")), error = function(e) NA)
  if (length(elapsed) == 0 || is.na(elapsed))
    elapsed <- tryCatch(as.numeric(gf(rs, "elapsed_s_through_this_case")), error = function(e) NA)
  sweep_desc <- tryCatch(as.character(gf(gf(rs, "sweep"), "description")), error = function(e) "")
  if (length(sweep_desc) == 0) sweep_desc <- ""
  fmt_dur <- function(s) if (is.na(s)) "n/a" else sprintf("%.1f s (%.2f min)", s, s / 60)

  header <- tags$div(
    style = "font-family: sans-serif; margin-bottom: 12px;",
    tags$h2(sprintf("COM report - case %d of %d", case_idx,
                    if (is.na(n_cases)) case_idx else n_cases)),
    tags$table(
      style = "border-collapse: collapse;",
      tags$tr(tags$td(tags$b("COM")),    tags$td(sprintf("%.4f dB", num("COM_dB")))),
      tags$tr(tags$td(tags$b("FOM (objective the EQ search maximises)")),
              tags$td(sprintf("%.4f dB", rnum("FOM")))),
      tags$tr(tags$td(tags$b("VEO")),    tags$td(sprintf("%.3f mV", num("VEO_mV")))),
      tags$tr(tags$td(tags$b("VEC")),    tags$td(sprintf("%.3f dB", num("VEC_dB")))),
      tags$tr(tags$td(tags$b("A_s")),    tags$td(sprintf("%.3f mV", num("A_s_mV")))),
      tags$tr(tags$td(tags$b("Baud rate")),
              tags$td(sprintf("%.6g GBd  (Nyquist %.6g GHz)",
                              rnum("baud_rate_GHz"), rnum("f_Nyquist_GHz")))),
      tags$tr(tags$td(tags$b("IL die-to-die at Fnq")),
              tags$td(sprintf("%.3f dB", rnum("IL_db_die_to_die_at_Fnq")))),
      tags$tr(tags$td(tags$b("ICN")),
              tags$td(sprintf("%.4f mV  (MDNEXT %.4f, MDFEXT %.4f)",
                              rnum("ICN_mV"), rnum("MDNEXT_ICN_92_46_mV"),
                              rnum("MDFEXT_ICN_92_47_mV")))),
      tags$tr(tags$td(tags$b("CTLE")),   tags$td(ctle_txt)),
      tags$tr(tags$td(tags$b("Tx FFE")), tags$td(ffe_txt)),
      tags$tr(tags$td(tags$b("Rx FFE")), tags$td(rxffe_txt)),
      tags$tr(tags$td(tags$b("DFE")),    tags$td(dfe_txt)),
      tags$tr(tags$td(tags$b("MLSE")),   tags$td(mlse_txt)),
      tags$tr(tags$td(tags$b("EQ sweep")), tags$td(sweep_desc)),
      tags$tr(tags$td(tags$b("Cases run")), tags$td(sprintf("%s", if (is.na(n_cases)) "n/a" else n_cases))),
      tags$tr(tags$td(tags$b("Run time (this case)")), tags$td(fmt_dur(elapsed)))
    )
  )

  # -- collapsible tables: full results, then configuration & metadata --------
  results_section <- .collapsible(
    "Full results (click to expand)",
    .kv_table(.struct_rows(g0(dat, "results_full")), pairs_per_row = 2))

  cfgmeta_rows <- c(.struct_rows(g0(dat, "meta")), .struct_rows(g0(dat, "config")))
  cfgmeta_section <- .collapsible(
    "Configuration & metadata (click to expand)",
    .kv_table(cfgmeta_rows, pairs_per_row = 2))

  page <- browsable(tagList(
    header,
    # Grouped by the seven pipeline stages the correlation harness localises a
    # disagreement to, and in that order, so the report reads as stage evidence.
    # A plot returns NULL when its data is absent (an older .mat, or a run with
    # TDR off); optional() drops it rather than erroring the whole report.
    tags$h3("Stage 1 - Channel / frequency domain"),
    tags$div(style="height:800px;", plot_channel_fd(dat)),
    tags$div(style="height:800px;", plot_ctle(dat)),
    tags$div(style="height:800px;", plot_ctle_bank(dat)),
    tags$h3("Stage 2 - TDR / ERL"),
    optional(plot_tdr_impedance(dat), 700),
    optional(plot_erl(dat), 500),
    tags$h3("Stage 3 - Pulse (time domain)"),
    tags$div(style="height:800px;", plot_impulse(dat)),
    tags$div(style="height:800px;", plot_pulse(dat)),
    tags$h3("Stage 4 - Sampling"),
    optional(plot_fom_vs_phase(dat), 700),
    tags$h3("Stage 5 - Equalization"),
    tags$div(style="height:800px;", plot_eq_contribution(dat)),
    optional(plot_eq_taps(dat), 700),
    tags$h3("Stage 6 - Noise"),
    optional(plot_noise_terms(dat), 600),
    tags$h3("Stage 7 - COM"),
    tags$div(style="height:1200px;", plot_eye(dat)),
    tags$div(style="height:800px;", plot_bathtub(dat)),
    tags$div(style="height:800px;", plot_voltage_bathtub(dat)),
    results_section,
    cfgmeta_section
  ))
  
  save_html(page, file = out_html)
  message("Wrote ", out_html)
  invisible(out_html)
}

# -- CLI entry ----------------------------------------------------------------
if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 1) {
    stop("Usage: Rscript R/com_analysis.R <run>_caseNN.mat [out_report.html]")
  }
  build_dashboard(args[1], if (length(args) >= 2) args[2] else NULL)
}
