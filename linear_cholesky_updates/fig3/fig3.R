#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(scales)
})

script_arg <- commandArgs(trailingOnly = FALSE)
script_file <- sub("^--file=", "", script_arg[grep("^--file=", script_arg)][1])
script_dir <- if (!is.na(script_file)) dirname(normalizePath(script_file)) else getwd()
result_dir <- file.path(script_dir, "results")
figure_dir <- normalizePath(file.path(script_dir, "..", "figures"), mustWork = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

read_result <- function(path) {
  read_csv(path, show_col_types = FALSE, progress = FALSE)
}

files <- list.files(result_dir, pattern = "\\.csv$", full.names = TRUE)
if (length(files) == 0) {
  stop("No CSV files found in ", result_dir)
}

raw <- bind_rows(lapply(files, read_result), .id = "source_id")
if (!"status" %in% names(raw)) {
  raw$status <- "completed"
}

completed <- raw %>%
  filter(status == "completed", !is.na(target_fdr)) %>%
  mutate(
    covariance = factor(covariance, levels = c("AR1", "ER", "block")),
    method_name = factor(method_name, levels = c("ME", "MVR", "SDP")),
    update_strategy = factor(
      update_strategy,
      levels = c("standard", "early", "parallel32"),
      labels = c("standard", "early stop", "parallel 32")
    )
  )

summary <- completed %>%
  group_by(covariance, method_name, update_strategy, target_fdr) %>%
  summarize(
    power = mean(power),
    fdr = mean(fdr),
    nruns = n(),
    .groups = "drop"
  )

write_csv(summary, file.path(script_dir, "fig3_summary.csv"))

status_summary <- raw %>%
  count(status, update_strategy, method_name, covariance, name = "n")
write_csv(status_summary, file.path(script_dir, "fig3_status_summary.csv"))

plot_data <- bind_rows(
  summary %>% transmute(covariance, method_name, update_strategy, target_fdr, metric = "Power", value = power, nruns),
  summary %>% transmute(covariance, method_name, update_strategy, target_fdr, metric = "Empirical FDR", value = fdr, nruns)
) %>%
  mutate(metric = factor(metric, levels = c("Power", "Empirical FDR")))

method_colors <- c("ME" = "#009E73", "MVR" = "#0072B2", "SDP" = "#D55E00")
strategy_linetypes <- c("standard" = "solid", "early stop" = "longdash", "parallel 32" = "dotted")

diag_data <- expand.grid(
  metric = factor("Empirical FDR", levels = c("Power", "Empirical FDR")),
  covariance = factor(c("AR1", "ER", "block"), levels = c("AR1", "ER", "block"))
)

p <- ggplot(plot_data, aes(x = target_fdr, y = value, color = method_name, linetype = update_strategy,
                           group = interaction(method_name, update_strategy))) +
  geom_segment(
    data = diag_data,
    aes(x = 0, y = 0, xend = 0.2, yend = 0.2),
    color = "grey55",
    linewidth = 0.45,
    linetype = "dashed",
    inherit.aes = FALSE
  ) +
  geom_line(linewidth = 0.7, na.rm = TRUE) +
  geom_point(size = 1.5, stroke = 0.2, na.rm = TRUE) +
  facet_grid(metric ~ covariance, scales = "free_y") +
  scale_x_continuous(breaks = seq(0, 0.2, by = 0.05), limits = c(0, 0.2)) +
  scale_y_continuous(labels = label_number(accuracy = 0.01), limits = c(0, NA)) +
  scale_color_manual(values = method_colors, drop = FALSE, name = "Method") +
  scale_linetype_manual(values = strategy_linetypes, drop = FALSE, name = "Update") +
  labs(x = "Target FDR", y = NULL) +
  theme_bw(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey70"),
    strip.text = element_text(face = "bold", size = 9),
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.key.width = unit(1.2, "cm"),
    plot.margin = margin(8, 10, 8, 8)
  )

ggsave(file.path(figure_dir, "fig3_power_fdr_R.pdf"), p, width = 9.5, height = 5.8, device = "pdf")

message("Wrote ", file.path(figure_dir, "fig3_power_fdr_R.pdf"))
