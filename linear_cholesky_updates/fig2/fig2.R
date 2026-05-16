#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(grid)
  library(gtable)
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
    covariance = factor(covariance, levels = c("AR1", "ER", "block", "stress")),
    method_name = factor(method_name, levels = c("ME", "MVR", "SDP")),
    update_strategy = factor(
      update_strategy,
      levels = c("serial", "serial_robust", "local0", "buffer16", "buffer32", "buffer64"),
      labels = c("serial", "serial robust", "local b=0", "buffer 16", "buffer 32", "buffer 64")
    )
  ) %>%
  filter(!is.na(update_strategy), !is.na(covariance))

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
  "serial" = "#222222",
  "serial robust" = "#777777",
  "local b=0" = "#D55E00",
  "buffer 16" = "#E69F00",
  "buffer 32" = "#009E73",
  "buffer 64" = "#0072B2"
)

unlink(file.path(script_dir, "fig2_speedup_summary.csv"))
unlink(file.path(figure_dir, "fig2_speedup_R.pdf"))

break_low <- 20000
break_high <- 50000
break_gap <- 2500
high_time_compression <- 5
time_transform <- function(x) {
  case_when(
    x <= break_low ~ x,
    x < break_high ~ break_low + break_gap / 2,
    TRUE ~ break_low + break_gap + (x - break_high) / high_time_compression
  )
}
time_breaks <- c(0, 5000, 10000, 15000, 20000, 50000, 100000, 150000)
time_break_positions <- time_transform(time_breaks)

plot_data <- summary %>%
  mutate(mean_sec_plot = time_transform(mean_sec))

line_data <- plot_data %>%
  group_by(covariance, method_name, update_strategy) %>%
  filter(n() > 1) %>%
  ungroup()

p <- ggplot(plot_data, aes(x = p, y = mean_sec_plot, color = update_strategy, group = update_strategy)) +
  geom_line(data = line_data, linewidth = 0.7, na.rm = TRUE) +
  geom_point(size = 1.8, stroke = 0.2, na.rm = TRUE) +
  facet_grid(method_name ~ covariance, scales = "free_y") +
  scale_x_log10(
    breaks = c(500, 1000, 2000, 5000, 10000, 20000),
    labels = comma
  ) +
  scale_y_continuous(
    breaks = time_break_positions,
    labels = label_number(accuracy = 1)(time_breaks),
    limits = c(0, NA),
    expand = expansion(mult = c(0, 0.08))
  ) +
  scale_color_manual(values = strategy_colors, drop = FALSE, name = NULL) +
  coord_cartesian(xlim = c(430, 23500), clip = "off") +
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

add_axis_break_marks <- function(plot, data) {
  gt <- ggplotGrob(plot)
  break_y <- break_low + break_gap * 0.5
  row_max <- data %>%
    group_by(method_name) %>%
    summarize(max_y = max(mean_sec_plot, na.rm = TRUE), .groups = "drop") %>%
    mutate(y_npc = break_y / (max_y * 1.08))

  left_panels <- gt$layout %>%
    filter(grepl("^panel-[0-9]+-1$", name)) %>%
    mutate(panel_row = as.integer(sub("^panel-([0-9]+)-1$", "\\1", name))) %>%
    arrange(panel_row)

  for (i in seq_len(nrow(left_panels))) {
    y_npc <- row_max$y_npc[row_max$method_name == levels(data$method_name)[i]]
    y_npc <- max(0.08, min(0.92, y_npc))
    mark <- grobTree(
      segmentsGrob(
        x0 = unit(-0.012, "npc"), x1 = unit(0.022, "npc"),
        y0 = unit(y_npc - 0.010, "npc"), y1 = unit(y_npc + 0.010, "npc"),
        gp = gpar(col = "grey15", lwd = 1.0, lineend = "round")
      ),
      segmentsGrob(
        x0 = unit(-0.012, "npc"), x1 = unit(0.022, "npc"),
        y0 = unit(y_npc + 0.010, "npc"), y1 = unit(y_npc + 0.030, "npc"),
        gp = gpar(col = "grey15", lwd = 1.0, lineend = "round")
      )
    )
    gt <- gtable_add_grob(
      gt,
      mark,
      t = left_panels$t[i], l = left_panels$l[i],
      name = paste0("axis-break-", i),
      clip = "off"
    )
  }

  gt
}

figure_path <- file.path(figure_dir, "fig2_timing_R.pdf")
plot_grob <- add_axis_break_marks(p, plot_data)
pdf(figure_path, width = 9.5, height = 7.2)
grid.newpage()
grid.draw(plot_grob)
invisible(dev.off())

message("Wrote ", figure_path)
