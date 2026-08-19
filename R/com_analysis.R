# com_analysis.R -- interactive R/Plotly visualisation of a COM engineering .mat
# snapshot produced by `python com.py ... --export-mat`.
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
           "re-run with  com.EYE_PLOT_UNDER_MLSE = True  -- diagnostic only, it ",
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
    tags$div(style="height:800px;", plot_channel_fd(dat)),
    tags$div(style="height:800px;", plot_ctle(dat)),
    tags$div(style="height:800px;", plot_ctle_bank(dat)),
    tags$div(style="height:800px;", plot_impulse(dat)),
    tags$div(style="height:800px;", plot_pulse(dat)),
    tags$div(style="height:1200px;", plot_eye(dat)),
    tags$div(style="height:800px;", plot_bathtub(dat)),
    tags$div(style="height:800px;", plot_eq_contribution(dat)),
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
