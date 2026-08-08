
# Remove all variables currently loaded into memory
rm(list=ls()) 

# Set working directory
# if (requireNamespace("rstudioapi", quietly=TRUE) && rstudioapi::isAvailable()) setwd(dirname(rstudioapi::getSourceEditorContext()$path))
setwd("C:/Users/tberm/Documents/Python_projects/COM_matlab_to_python_3/results")

source("../R/com_analysis.R")
# dat <- load_com(file.choose())
# plot_channel_fd(dat)            # any single plot
# plot_ctle(dat)
# plot_eq_contribution(dat)
# plot_eye(dat)
# plot_pulse(dat)
build_dashboard(file.choose())   # full HTML report
