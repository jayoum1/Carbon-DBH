# ==============================================================================
# DBH Previous Year Visualization Script
# ==============================================================================
# Creates visualizations with PrevDBH_cm as the y-variable (target)
# Useful for understanding the relationship between previous DBH and current DBH
# ==============================================================================

# Source package setup
source("R/setup_packages.R")

# Load required libraries
library(ggplot2)
library(dplyr)
library(readr)
library(patchwork)

# ==============================================================================
# Configuration
# ==============================================================================

DATA_PATH <- "Data/Processed Data/Carbon/all_plots_with_carbon_encoded.csv"
PLOTS_DIR <- "plots/dbh_prev"

# Create plots directory if it doesn't exist
if (!dir.exists(PLOTS_DIR)) {
  dir.create(PLOTS_DIR, recursive = TRUE)
  message("Created plots directory: ", PLOTS_DIR)
}

# ==============================================================================
# Load Data
# ==============================================================================

message("=")
message("Loading Data for PrevDBH Visualization")
message("=")

df <- read_csv(DATA_PATH, show_col_types = FALSE)

# Create PrevDBH_cm if it doesn't exist
if (!"PrevDBH_cm" %in% colnames(df)) {
  message("Creating PrevDBH_cm column...")
  df <- df %>%
    arrange(TreeID, Year) %>%
    group_by(TreeID) %>%
    mutate(PrevDBH_cm = lag(DBH_cm)) %>%
    ungroup()
}

# Filter valid data
df_valid <- df %>%
  filter(!is.na(PrevDBH_cm) & !is.na(DBH_cm))

message("✓ Loaded ", nrow(df), " rows")
message("✓ Valid rows (with PrevDBH_cm and DBH_cm): ", nrow(df_valid))

# Reconstruct plot names
plot_cols <- colnames(df_valid)[startsWith(colnames(df_valid), "Plot_")]
if (length(plot_cols) > 0) {
  df_valid$plot <- apply(df_valid[, plot_cols], 1, function(x) {
    if (any(x == 1)) {
      names(x)[x == 1][1] %>% str_replace("Plot_", "")
    } else {
      "Lower"
    }
  })
} else {
  df_valid$plot <- "Unknown"
}

# ==============================================================================
# Plot Functions
# ==============================================================================

plot_dbh_vs_prev_dbh <- function(df, save = TRUE) {
  # Scatter plot of DBH_cm vs PrevDBH_cm.
  p <- ggplot(df, aes(x = PrevDBH_cm, y = DBH_cm)) +
    geom_point(alpha = 0.3, size = 0.5) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
    geom_smooth(method = "lm", se = TRUE, color = "blue", alpha = 0.3) +
    labs(
      title = "Current DBH vs Previous DBH",
      subtitle = "Red line: y=x (no growth), Blue line: linear fit",
      x = "Previous DBH (cm)",
      y = "Current DBH (cm)"
    ) +
    theme_minimal() +
    theme(
      plot.background = element_rect(fill = "white", color = "white"),
      panel.background = element_rect(fill = "white", color = "white")
    )
  
  if (save) {
    filename <- file.path(PLOTS_DIR, "dbh_vs_prev_dbh.png")
    ggsave(filename, plot = p, width = 10, height = 8, dpi = 300, bg = "white")
    message("Saved: ", filename)
  }
  
  return(p)
}

plot_dbh_vs_prev_dbh_by_plot <- function(df, save = TRUE) {
  # Scatter plot of DBH_cm vs PrevDBH_cm faceted by plot.
  if (!"plot" %in% colnames(df)) {
    warning("Plot column not found. Skipping plot-specific visualization.")
    return(NULL)
  }
  
  p <- ggplot(df, aes(x = PrevDBH_cm, y = DBH_cm)) +
    geom_point(alpha = 0.3, size = 0.5) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
    geom_smooth(method = "lm", se = TRUE, color = "blue", alpha = 0.3) +
    facet_wrap(~ plot, scales = "free") +
    labs(
      title = "Current DBH vs Previous DBH by Plot",
      subtitle = "Red line: y=x (no growth), Blue line: linear fit",
      x = "Previous DBH (cm)",
      y = "Current DBH (cm)"
    ) +
    theme_minimal() +
    theme(
      plot.background = element_rect(fill = "white", color = "white"),
      panel.background = element_rect(fill = "white", color = "white")
    )
  
  if (save) {
    filename <- file.path(PLOTS_DIR, "dbh_vs_prev_dbh_by_plot.png")
    ggsave(filename, plot = p, width = 14, height = 8, dpi = 300, bg = "white")
    message("Saved: ", filename)
  }
  
  return(p)
}

plot_growth_vs_prev_dbh <- function(df, save = TRUE) {
  # Plot DBH growth (DBH_cm - PrevDBH_cm) vs PrevDBH_cm.
  df_growth <- df %>%
    mutate(growth = DBH_cm - PrevDBH_cm)
  
  p <- ggplot(df_growth, aes(x = PrevDBH_cm, y = growth)) +
    geom_point(alpha = 0.3, size = 0.5) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
    geom_smooth(method = "loess", se = TRUE, color = "blue", alpha = 0.3) +
    labs(
      title = "DBH Growth vs Previous DBH",
      subtitle = "Growth = Current DBH - Previous DBH",
      x = "Previous DBH (cm)",
      y = "DBH Growth (cm)"
    ) +
    theme_minimal() +
    theme(
      plot.background = element_rect(fill = "white", color = "white"),
      panel.background = element_rect(fill = "white", color = "white")
    )
  
  if (save) {
    filename <- file.path(PLOTS_DIR, "growth_vs_prev_dbh.png")
    ggsave(filename, plot = p, width = 10, height = 8, dpi = 300, bg = "white")
    message("Saved: ", filename)
  }
  
  return(p)
}

plot_growth_rate_vs_prev_dbh <- function(df, save = TRUE) {
  # Plot DBH growth rate vs PrevDBH_cm.
  df_growth <- df %>%
    mutate(
      growth = DBH_cm - PrevDBH_cm,
      growth_rate = ifelse(PrevDBH_cm > 0, growth / PrevDBH_cm, NA)
    ) %>%
    filter(!is.na(growth_rate))
  
  p <- ggplot(df_growth, aes(x = PrevDBH_cm, y = growth_rate)) +
    geom_point(alpha = 0.3, size = 0.5) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
    geom_smooth(method = "loess", se = TRUE, color = "blue", alpha = 0.3) +
    labs(
      title = "DBH Growth Rate vs Previous DBH",
      subtitle = "Growth Rate = (Current DBH - Previous DBH) / Previous DBH",
      x = "Previous DBH (cm)",
      y = "Growth Rate"
    ) +
    theme_minimal() +
    theme(
      plot.background = element_rect(fill = "white", color = "white"),
      panel.background = element_rect(fill = "white", color = "white")
    )
  
  if (save) {
    filename <- file.path(PLOTS_DIR, "growth_rate_vs_prev_dbh.png")
    ggsave(filename, plot = p, width = 10, height = 8, dpi = 300, bg = "white")
    message("Saved: ", filename)
  }
  
  return(p)
}

plot_prev_dbh_distribution <- function(df, save = TRUE) {
  # Histogram of PrevDBH_cm distribution.
  p <- ggplot(df, aes(x = PrevDBH_cm)) +
    geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7, color = "white") +
    labs(
      title = "Distribution of Previous DBH",
      x = "Previous DBH (cm)",
      y = "Frequency"
    ) +
    theme_minimal() +
    theme(
      plot.background = element_rect(fill = "white", color = "white"),
      panel.background = element_rect(fill = "white", color = "white")
    )
  
  if (save) {
    filename <- file.path(PLOTS_DIR, "prev_dbh_distribution.png")
    ggsave(filename, plot = p, width = 10, height = 6, dpi = 300, bg = "white")
    message("Saved: ", filename)
  }
  
  return(p)
}

plot_prev_dbh_by_plot <- function(df, save = TRUE) {
  # Boxplot of PrevDBH_cm by plot.
  if (!"plot" %in% colnames(df)) {
    warning("Plot column not found. Skipping plot-specific visualization.")
    return(NULL)
  }
  
  p <- ggplot(df, aes(x = plot, y = PrevDBH_cm, fill = plot)) +
    geom_boxplot(alpha = 0.7) +
    labs(
      title = "Previous DBH Distribution by Plot",
      x = "Plot",
      y = "Previous DBH (cm)"
    ) +
    theme_minimal() +
    theme(
      plot.background = element_rect(fill = "white", color = "white"),
      panel.background = element_rect(fill = "white", color = "white")
    ) +
    theme(legend.position = "none")
  
  if (save) {
    filename <- file.path(PLOTS_DIR, "prev_dbh_by_plot.png")
    ggsave(filename, plot = p, width = 10, height = 6, dpi = 300, bg = "white")
    message("Saved: ", filename)
  }
  
  return(p)
}

# ==============================================================================
# Generate Plots
# ==============================================================================

message("")
message("=")
message("Generating PrevDBH Visualizations")
message("=")

# Generate all plots
plot_dbh_vs_prev_dbh(df_valid)
plot_dbh_vs_prev_dbh_by_plot(df_valid)
plot_growth_vs_prev_dbh(df_valid)
plot_growth_rate_vs_prev_dbh(df_valid)
plot_prev_dbh_distribution(df_valid)
plot_prev_dbh_by_plot(df_valid)

message("")
message("=")
message("PrevDBH Visualization Complete")
message("=")
message("")
message("Plots saved to: ", PLOTS_DIR)
message("")

