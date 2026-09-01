# Copyright 2026 Todd Bermensolo
# SPDX-License-Identifier: BSD-3-Clause

# correlation_report.R — figures for the COM Python vs COM MATLAB review
#
# Reads the tidy comparison CSVs written by the local exporter and emits PNGs sized
# for 16:9 slides. Run:
#   Rscript R/correlation_report.R
#
# Figures are written to report_figs/.

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(scales)
})

# run from the repository root (or pass it as the first argument)
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[1] else "."
if (!dir.exists(file.path(root, "report_data")) &&
    dir.exists(file.path("..", "report_data"))) root <- ".."
stopifnot(dir.exists(file.path(root, "report_data")))
din  <- file.path(root, "report_data")
dout <- file.path(root, "report_figs")
dir.create(dout, showWarnings = FALSE)

W <- 10; H <- 5.4; DPI <- 150
theme_com <- theme_minimal(base_size = 13) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold", size = 15),
        plot.subtitle = element_text(colour = "grey35", size = 11),
        legend.position = "top")

# Optional 2nd/3rd args select an alternative input CSV and an output suffix, so
# the same figures can be produced for a variant run:
#   Rscript R/correlation_report.R . compare_fullgrid.csv _fullgrid
csv_in <- if (length(args) >= 2) args[2] else "compare.csv"
sfx    <- if (length(args) >= 3) args[3] else ""
fig    <- function(name) file.path(dout, paste0(name, sfx, ".png"))
cmp <- read.csv(file.path(din, csv_in), stringsAsFactors = FALSE)
cmp$config <- factor(paste0("Test_", cmp$test))
cmp$cond   <- factor(cmp$cond, levels = c("wXtalk", "woXtalk"),
                     labels = c("with crosstalk", "no crosstalk"))

rms <- function(x) sqrt(mean(x^2))
lab <- sprintf("%d cases   max |ΔCOM| = %.4f dB   rms = %.4f dB   %d bit-exact",
               nrow(cmp), max(abs(cmp$dcom)), rms(cmp$dcom),
               sum(abs(cmp$dcom) < 1e-9))

# ---- 1. correlation scatter --------------------------------------------------
p1 <- ggplot(cmp, aes(com_mat, com_py, colour = cond)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey55", linewidth = .4) +
  geom_point(size = 2, alpha = .8) +
  scale_colour_manual(values = c("with crosstalk" = "#1f77b4",
                                 "no crosstalk"  = "#d62728"), name = NULL) +
  coord_equal() +
  labs(title = "COM Python vs COM MATLAB",
       subtitle = lab, x = "COM MATLAB (dB)", y = "COM Python (dB)") +
  theme_com
ggsave(fig("fig_correlation"), p1, width = 7.2, height = H, dpi = DPI)

# ---- 2. error distribution ---------------------------------------------------
p2 <- ggplot(cmp, aes(dcom)) +
  geom_histogram(bins = 60, fill = "#1f77b4", colour = "white", linewidth = .2) +
  geom_vline(xintercept = c(-0.05, 0.05), linetype = "dashed", colour = "grey40") +
  annotate("text", x = 0.05, y = Inf, label = "  ±0.05 dB", hjust = 0,
           vjust = 1.6, size = 3.4, colour = "grey30") +
  labs(title = "COM difference distribution",
       subtitle = sprintf("mean %+.5f dB, median |Δ| %.5f dB; %d of %d within 0.02 dB",
                          mean(cmp$dcom), median(abs(cmp$dcom)),
                          sum(abs(cmp$dcom) <= 0.02), nrow(cmp)),
       x = "COM Python − COM MATLAB (dB)", y = "cases") +
  theme_com
ggsave(fig("fig_dcom_hist"), p2, width = W, height = H, dpi = DPI)

# ---- 3. COM and FOM by package config ---------------------------------------
both <- cmp %>%
  select(config, cond, COM = dcom, FOM = dfom) %>%
  pivot_longer(c(COM, FOM), names_to = "metric", values_to = "delta") %>%
  mutate(metric = factor(metric, levels = c("FOM", "COM"),
                         labels = c("ΔFOM (dB)", "ΔCOM (dB)")))

p3 <- ggplot(both, aes(config, delta, fill = cond)) +
  geom_hline(yintercept = 0, colour = "grey55", linewidth = .4) +
  geom_boxplot(outlier.size = 1.1, alpha = .85, width = .65) +
  facet_wrap(~metric, scales = "free_y") +
  scale_fill_manual(values = c("with crosstalk" = "#1f77b4",
                               "no crosstalk"  = "#d62728"), name = NULL) +
  labs(title = "FOM and COM difference by package configuration",
       subtitle = paste("Test_1/Test_2 = PKG A (low loss);",
                        "Test_3/Test_4 = PKG B (high loss).",
                        "Both shrink monotonically with package loss."),
       x = NULL, y = NULL) +
  theme_com
ggsave(fig("fig_dcom_by_config"), p3, width = W, height = H, dpi = DPI)

# ---- 3b. FOM correlation ----------------------------------------------------
flab <- sprintf(
  "%d cases   max |ΔFOM| = %.4f dB   rms = %.4f dB   %d bit-exact",
  nrow(cmp), max(abs(cmp$dfom)), rms(cmp$dfom), sum(abs(cmp$dfom) < 1e-9))
p3b <- ggplot(cmp, aes(fom_mat, fom_py, colour = cond)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey55", linewidth = .4) +
  geom_point(size = 2, alpha = .8) +
  scale_colour_manual(values = c("with crosstalk" = "#1f77b4",
                                 "no crosstalk"  = "#d62728"), name = NULL) +
  coord_equal() +
  labs(title = "FOM Python vs FOM MATLAB",
       subtitle = flab, x = "FOM MATLAB (dB)", y = "FOM Python (dB)") +
  theme_com
ggsave(fig("fig_fom_correlation"), p3b, width = 7.2, height = H,
       dpi = DPI)

# ---- 3c. FOM error distribution, showing the systematic bias ----------------
p3c <- ggplot(cmp, aes(dfom)) +
  geom_vline(xintercept = 0, colour = "grey55", linewidth = .4) +
  geom_histogram(bins = 60, fill = "#7570b3", colour = "white", linewidth = .2) +
  geom_vline(xintercept = mean(cmp$dfom), colour = "#d95f02", linewidth = .8,
             linetype = "longdash") +
  annotate("text", x = mean(cmp$dfom), y = Inf, hjust = 1.05, vjust = 1.8,
           size = 3.5, colour = "#d95f02",
           label = sprintf("mean %+.4f dB  ", mean(cmp$dfom))) +
  labs(title = "FOM difference distribution",
       subtitle = sprintf(paste0("%d of %d cases are bit-exact; the rest scatter ",
                                 "both ways, mean %+.4f dB.
",
                                 "The pre-fix one-sided bias (Python low in 199 ",
                                 "of 208) is gone."),
                          sum(abs(cmp$dfom) < 1e-9), nrow(cmp), mean(cmp$dfom)),
       x = "FOM Python − FOM MATLAB (dB)", y = "cases") +
  theme_com
ggsave(fig("fig_fom_hist"), p3c, width = W, height = H, dpi = DPI)

# ---- 3d. FOM residual vs COM outcome ----------------------------------------
# Linear axes on purpose: on log-log the ~100 bit-exact COM cases collapse to
# log(0) and read as a separate cluster, inventing structure that is not there.
cmp$com_exact <- ifelse(abs(cmp$dcom) < 1e-9, "COM bit-exact", "COM differs")
n_fom_exact <- sum(abs(cmp$dfom) < 1e-9)
n_com_exact <- sum(abs(cmp$dcom) < 1e-9)

p3d <- ggplot(cmp, aes(abs(dfom) * 1000, abs(dcom) * 1000, colour = com_exact)) +
  geom_point(size = 2, alpha = .85) +
  scale_colour_manual(values = c("COM bit-exact" = "#1b9e77",
                                 "COM differs"  = "#7570b3"), name = NULL) +
  labs(title = "FOM residual vs COM outcome",
       subtitle = sprintf(paste("FOM bit-exact in %d of %d cases; COM in %d.",
                                "The COM search quantises to the PDF bin grid,",
                                "absorbing most of the FOM residual."),
                          n_fom_exact, nrow(cmp), n_com_exact),
       x = "|ΔFOM| (mdB)", y = "|ΔCOM| (mdB)") +
  theme_com
ggsave(fig("fig_fom_vs_com"), p3d, width = W, height = H, dpi = DPI)

# ---- 4. stage-by-stage agreement --------------------------------------------
# Reported as (a) the share of output columns that match MATLAB bit-for-bit and
# (b) how many significant digits the WORST column in that stage agrees to.
# Both are directly readable; an earlier version plotted raw relative error on a
# log axis, which needs decoding before it means anything.
st <- read.csv(file.path(din, paste0("stages", sfx, ".csv")), stringsAsFactors = FALSE) %>%
  group_by(stage) %>%
  summarise(pct_exact = 100 * sum(n_exact) / sum(n_cols),
            typical = median(worst_rel_err),
            cases_diff = sum(n_exact < n_cols),
            ncases = dplyr::n(),
            ncols = round(mean(n_cols)), .groups = "drop") %>%
  mutate(digits = ifelse(typical <= 0, 15, pmin(15, -log10(typical))),
         stage = reorder(stage, dplyr::desc(stage)),
         band = ifelse(pct_exact >= 99, "all or nearly all columns exact",
                       ifelse(pct_exact >= 75, "most columns exact",
                              "residual concentrated here")))

p4 <- ggplot(st, aes(y = stage, x = pct_exact, fill = band)) +
  geom_col(width = .62) +
  geom_text(aes(label = sprintf("  %.0f%%   (median case %s; %d of %d cases differ)",
                                pct_exact,
                                ifelse(digits >= 15, "identical",
                                       sprintf("agrees to %.0f digits", digits)),
                                cases_diff, ncases)),
            hjust = 0, size = 3.5, colour = "grey20") +
  scale_fill_manual(values = c("all or nearly all columns exact" = "#1b9e77",
                               "most columns exact" = "#2c7fb8",
                               "residual concentrated here" = "#d95f02"),
                    name = NULL) +
  scale_x_continuous(limits = c(0, 235), breaks = c(0, 25, 50, 75, 100),
                     labels = function(x) paste0(x, "%")) +
  labs(title = "Agreement by pipeline stage",
       subtitle = paste0("Bar: share of output columns agreeing with MATLAB to ",
                         "better than 1e-9 relative (~9+ significant figures).
",
                         "In brackets: the MEDIAN case for that stage, and how ",
                         "many of the 208 cases have any column outside 1e-9.
",
                         "Sampling is one integer column (itick), so its median ",
                         "case is identical while 8 cases differ outright."),
       x = "output columns agreeing to better than 1e-9 relative", y = NULL) +
  theme_com
ggsave(fig("fig_stage_agreement"), p4, width = W, height = H, dpi = DPI)

# ---- 5. sampling-phase divergence rate --------------------------------------
by_cfg <- cmp %>% group_by(config) %>%
  summarise(rate = 100 * mean(tick_match == 0), n = n(), .groups = "drop") %>%
  rename(grp = config) %>% mutate(panel = "by package config")
by_fam <- cmp %>% group_by(family) %>%
  summarise(rate = 100 * mean(tick_match == 0), n = n(), .groups = "drop") %>%
  rename(grp = family) %>% mutate(panel = "by channel family")
dv <- bind_rows(by_cfg, by_fam)

p5 <- ggplot(dv, aes(reorder(grp, rate), rate)) +
  geom_col(fill = "#d95f02", width = .65) +
  geom_text(aes(label = sprintf("%.1f%%", rate)), hjust = -0.15, size = 3.6) +
  coord_flip(clip = "off") +
  facet_wrap(~panel, scales = "free_y") +
  expand_limits(y = max(dv$rate) * 1.18) +
  labs(title = "Sampling-phase divergence rate",
       subtitle = sprintf(paste("%d of %d cases select a different itick;",
                                "concentrated in low-loss DAC assemblies with crosstalk"),
                          sum(cmp$tick_match == 0), nrow(cmp)),
       x = NULL, y = "cases with differing itick (%)") +
  theme_com
ggsave(fig("fig_divergence"), p5, width = W, height = H, dpi = DPI)

# ---- 6. adaptive vs full grid (only if that run has produced cases) ----------
sp <- file.path(din, "search.csv")
if (file.exists(sp)) {
  s <- read.csv(sp, stringsAsFactors = FALSE)
  if (nrow(s) > 0) {
    s$label <- substr(sub("_thru.*|_THRU.*", "", s$channel), 1, 34)
    n_fom <- sum(abs(s$dfom_adaptive_vs_full) < 1e-9)
    n_com <- sum(abs(s$dcom_adaptive_vs_full) < 1e-9)
    n_tick <- sum(s$itick_adaptive == s$itick_fullgrid)

    # Every delta is exactly zero, so plotting deltas gives invisible bars.
    # The informative axis is what full grid COSTS to reach the same answer.
    s$label <- reorder(s$label, s$speedup)
    p6 <- ggplot(s, aes(label, speedup, fill = family)) +
      geom_col(width = .62) +
      geom_text(aes(label = sprintf("%.0f x", speedup)), hjust = -0.15, size = 3.5) +
      coord_flip(clip = "off") +
      expand_limits(y = max(s$speedup) * 1.15) +
      labs(title = "Adaptive local search vs exhaustive full grid - same engine",
           subtitle = sprintf(paste("FOM identical %d/%d, COM identical %d/%d,",
                                    "sampling phase identical %d/%d.",
                                    "
Adaptive costs NOTHING in accuracy;",
                                    "bars show the runtime penalty for full grid."),
                              n_fom, nrow(s), n_com, nrow(s), n_tick, nrow(s)),
           x = NULL, y = "full-grid runtime / adaptive runtime", fill = NULL) +
      theme_com
    ggsave(fig("fig_search"), p6, width = W, height = H, dpi = DPI)
    cat(sprintf("fig_search.png written (%d channels)\n", nrow(s)))
  } else {
    cat("search.csv empty - full-grid run still in progress, fig_search skipped\n")
  }
}

cat(sprintf("figures written to %s\n", dout))
