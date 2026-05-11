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
  filter(status == "completed", !is.na(elapsed_sec)) %>%
  mutate(
    covariance = factor(covariance, levels = c("AR1", "ER", "block")),
    method_name = factor(method_name, levels = c("ME", "MVR", "SDP")),
    update_strategy = factor(
      update_strategy,
      levels = c("standard", "early", "parallel2", "parallel4", "parallel8", "parallel16", "parallel32"),
      labels = c("standard", "early stop", "parallel 2", "parallel 4", "parallel 8", "parallel 16", "parallel 32")
    )
  )

summary <- completed %>%
  group_by(covariance, p, method_name, update_strategy, nworkers) %>%
  summarize(
    mean_sec = mean(elapsed_sec),
    sd_sec = sd(elapsed_sec),
    nruns = n(),
    .groups = "drop"
  )

write_csv(summary, file.path(script_dir, "fig2_summary.csv"))

status_summary <- raw %>%
  count(status, update_strategy, method_name, covariance, p, name = "n")
write_csv(status_summary, file.path(script_dir, "fig2_status_summary.csv"))

strategy_colors <- c(
  "standard" = "#222222",
  "early stop" = "#0072B2",
  "parallel 2" = "#E69F00",
  "parallel 4" = "#009E73",
  "parallel 8" = "#D55E00",
  "parallel 16" = "#CC79A7",
  "parallel 32" = "#6A3D9A"
)

line_data <- summary %>%
  group_by(covariance, method_name, update_strategy) %>%
  filter(n() > 1) %>%
  ungroup()

p <- ggplot(summary, aes(x = p, y = mean_sec, color = update_strategy, group = update_strategy)) +
  geom_line(data = line_data, linewidth = 0.7, na.rm = TRUE) +
  geom_point(size = 1.8, stroke = 0.2, na.rm = TRUE) +
  geom_text(
    aes(label = ifelse(nruns < 5, paste0("n=", nruns), "")),
    color = "grey25",
    size = 2.2,
    vjust = -0.8,
    show.legend = FALSE,
    na.rm = TRUE
  ) +
  facet_grid(method_name ~ covariance) +
  scale_x_log10(breaks = c(500, 1000, 2000, 5000, 10000, 20000), labels = comma) +
  scale_y_log10(labels = label_number(accuracy = 0.1)) +
  scale_color_manual(values = strategy_colors, drop = FALSE, name = NULL) +
  labs(x = "Number of variables (p)", y = "Mean solve time (seconds)") +
  theme_bw(base_size = 10) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey70"),
    strip.text = element_text(face = "bold", size = 9),
    legend.position = "bottom",
    legend.key.width = unit(1.2, "cm"),
    axis.text.x = element_text(angle = 35, hjust = 1),
    plot.margin = margin(8, 10, 8, 8)
  )

ggsave(file.path(figure_dir, "fig2_timing_R.pdf"), p, width = 9.5, height = 7.2, device = "pdf")

message("Wrote ", file.path(figure_dir, "fig2_timing_R.pdf"))
