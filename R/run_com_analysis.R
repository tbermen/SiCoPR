
# Remove all variables currently loaded into memory
rm(list=ls()) 

# Work from the repository root, wherever the clone happens to live.
# Rscript exposes the script path in the command line; RStudio does not, so fall
# back to rstudioapi and finally to the current directory.
repo_root <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) return(normalizePath(file.path(dirname(f), "..")))
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable())
    return(normalizePath(file.path(dirname(rstudioapi::getSourceEditorContext()$path), "..")))
  normalizePath(".")
})
setwd(repo_root)

source(file.path(repo_root, "R", "com_analysis.R"))
# dat <- load_com(file.choose())
# plot_channel_fd(dat)            # any single plot
# plot_ctle(dat)
# plot_eq_contribution(dat)
# plot_eye(dat)
# plot_pulse(dat)
build_dashboard(file.choose())   # full HTML report
