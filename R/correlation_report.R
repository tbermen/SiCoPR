# correlation_report.R — figures for the COM Python vs COM MATLAB review
#
# Reads the tidy CSVs written by tools/export_compare_csv.py and emits PNGs sized
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

cmp <- read.csv(file.path(din, "compare.csv"), stringsAsFactors = FALSE)
cmp$config <- factor(paste0("Test_", cmp$test))
cmp$cond   <- factor(cmp$cond, levels = c("wXtalk", "woXtalk"),
                     labels = c("with crosstalk", "no crosstalk"))

rms <- function(x) sqrt(mean(x^2))
lab <- sprintf("208 cases   max |ΔCOM| = %.4f dB   rms = %.4f dB   %d bit-exact",
               max(abs(cmp$dcom)), rms(cmp$dcom), sum(abs(cmp$dcom) < 1e-9))

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
ggsave(file.path(dout, "fig_correlation.png"), p1, width = 7.2, height = H, dpi = DPI)

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
ggsave(file.path(dout, "fig_dcom_hist.png"), p2, width = W, height = H, dpi = DPI)

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
ggsave(file.path(dout, "fig_dcom_by_config.png"), p3, width = W, height = H, dpi = DPI)

# ---- 3b. FOM correlation ----------------------------------------------------
nlow <- sum(cmp$dfom < 0)
flab <- sprintf(
  "%d cases   max |ΔFOM| = %.4f dB   rms = %.4f dB   COM Python lower in %d of %d",
  nrow(cmp), max(abs(cmp$dfom)), rms(cmp$dfom), nlow, nrow(cmp))
p3b <- ggplot(cmp, aes(fom_mat, fom_py, colour = cond)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey55", linewidth = .4) +
  geom_point(size = 2, alpha = .8) +
  scale_colour_manual(values = c("with crosstalk" = "#1f77b4",
                                 "no crosstalk"  = "#d62728"), name = NULL) +
  coord_equal() +
  labs(title = "FOM Python vs FOM MATLAB",
       subtitle = flab, x = "FOM MATLAB (dB)", y = "FOM Python (dB)") +
  theme_com
ggsave(file.path(dout, "fig_fom_correlation.png"), p3b, width = 7.2, height = H,
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
       subtitle = sprintf(paste("COM Python's FOM is lower in %d of %d cases —",
                                "a systematic bias, not scatter.",
                                "Adaptive search shrinks its radius on a",
                                "0.002 dB improvement threshold."),
                          nlow, nrow(cmp)),
       x = "FOM Python − FOM MATLAB (dB)", y = "cases") +
  theme_com
ggsave(file.path(dout, "fig_fom_hist.png"), p3c, width = W, height = H, dpi = DPI)

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
ggsave(file.path(dout, "fig_fom_vs_com.png"), p3d, width = W, height = H, dpi = DPI)

# ---- 4. stage-by-stage agreement --------------------------------------------
st <- read.csv(file.path(din, "stages.csv"), stringsAsFactors = FALSE) %>%
  group_by(stage) %>%
  summarise(typical = median(median_rel_err),
            worst   = median(worst_rel_err),
            pct_exact = 100 * sum(n_exact) / sum(n_cols), .groups = "drop") %>%
  mutate(typical = pmax(typical, 1e-16), worst = pmax(worst, 1e-16),
         stage = reorder(stage, dplyr::desc(stage)))

# lollipop, not geom_col: bars on a log axis are drawn from y=1 and read as if
# every stage reached 1e0.
p4 <- ggplot(st, aes(y = stage)) +
  geom_segment(aes(x = typical, xend = worst, yend = stage),
               colour = "grey75", linewidth = 2.4, lineend = "round") +
  geom_point(aes(x = typical, colour = "typical column"), size = 4.2) +
  geom_point(aes(x = worst, colour = "worst column"), size = 4.2) +
  geom_vline(xintercept = 1e-9, linetype = "dashed", colour = "grey40") +
  geom_text(aes(x = worst, label = sprintf("  %.0f%% of columns exact", pct_exact)),
            hjust = 0, size = 3.3, colour = "grey25", nudge_x = 0.35) +
  annotate("text", x = 1e-9, y = 0.62, label = "machine-exact  ", hjust = 1,
           size = 3.3, colour = "grey30") +
  scale_colour_manual(values = c("typical column" = "#2c7fb8",
                                 "worst column" = "#d95f02"), name = NULL) +
  scale_x_log10(labels = trans_format("log10", math_format(10^.x)),
                breaks = 10^seq(-16, 0, 2),
                limits = c(1e-16, 3e3)) +
  labs(title = "Agreement by pipeline stage",
       subtitle = paste("median across 208 cases; stages 1-4 are machine-exact,",
                        "the residual enters at equalization"),
       x = "relative error vs MATLAB", y = NULL) +
  theme_com
ggsave(file.path(dout, "fig_stage_agreement.png"), p4, width = W, height = H, dpi = DPI)

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
       subtitle = "9 of 208 cases select a different itick; concentrated in low-loss DAC assemblies with crosstalk",
       x = NULL, y = "cases with differing itick (%)") +
  theme_com
ggsave(file.path(dout, "fig_divergence.png"), p5, width = W, height = H, dpi = DPI)

# ---- 6. adaptive vs full grid (only if that run has produced cases) ----------
sp <- file.path(din, "search.csv")
if (file.exists(sp)) {
  s <- read.csv(sp, stringsAsFactors = FALSE)
  if (nrow(s) > 0) {
    s$label <- sub("_thru.*|_THRU.*", "", s$channel)
    long <- s %>%
      select(label, family, dcom_adaptive_vs_full, speedup) %>%
      mutate(label = reorder(label, dcom_adaptive_vs_full))
    p6 <- ggplot(long, aes(label, dcom_adaptive_vs_full, fill = family)) +
      geom_hline(yintercept = 0, colour = "grey55") +
      geom_col(width = .6) +
      coord_flip() +
      labs(title = "Adaptive local search vs full grid (same engine)",
           subtitle = sprintf("%d channels, Test_1 with crosstalk; median speedup %.1f×",
                              nrow(s), median(s$speedup, na.rm = TRUE)),
           x = NULL, y = "COM(adaptive) − COM(full grid)  (dB)") +
      theme_com
    ggsave(file.path(dout, "fig_search.png"), p6, width = W, height = H, dpi = DPI)
    cat(sprintf("fig_search.png written (%d channels)\n", nrow(s)))
  } else {
    cat("search.csv empty - full-grid run still in progress, fig_search skipped\n")
  }
}

cat(sprintf("figures written to %s\n", dout))
