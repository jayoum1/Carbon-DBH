# ==============================================================================
# DBH Plotting Script for Carbon DBH Project
# ==============================================================================
# Creates publication-quality plots for DBH analysis
# ==============================================================================

# Source package setup
source("R/setup_packages.R")

# Load required libraries explicitly
library(ggplot2)
library(dplyr)
library(readr)
library(patchwork)

# ==============================================================================
# Configuration
# ==============================================================================

# Data paths (relative to project root)
DATA_PATH <- "Data/Processed Data/Carbon/all_plots_with_carbon.csv"
PLOTS_DIR <- "plots"

# Create plots directory if it doesn't exist
if (!dir.exists(PLOTS_DIR)) {
  dir.create(PLOTS_DIR, recursive = TRUE)
  message("Created plots directory: ", PLOTS_DIR)
}

# ==============================================================================
# Load Data
# ==============================================================================

message("Loading data from: ", DATA_PATH)
df <- read_csv(DATA_PATH, show_col_types = FALSE)

# Standardize column names (handle case variations)
colnames(df) <- tolower(colnames(df))

# Ensure required columns exist (with flexible naming)
if (!"dbh_cm" %in% colnames(df) && "dbh" %in% colnames(df)) {
  # Convert DBH from inches to cm if needed
  df$dbh_cm <- df$dbh * 2.54
  message("Converted DBH from inches to cm")
}

# Check for growth rate column
if (!"growthrate" %in% colnames(df)) {
  if ("growth_rate" %in% colnames(df)) {
    df$growthrate <- df$growth_rate
  } else {
    message("Warning: GrowthRate column not found. Some plots may not work.")
  }
}

# Validate required columns exist
required_cols <- c("plot", "year")
missing_cols <- required_cols[!required_cols %in% colnames(df)]
if (length(missing_cols) > 0) {
  stop("Error: Missing required columns: ", paste(missing_cols, collapse = ", "))
}

message("Data loaded: ", nrow(df), " rows × ", ncol(df), " columns")
message("Plots: ", paste(unique(df$plot), collapse = ", "))
message("Years: ", min(df$year, na.rm = TRUE), " - ", max(df$year, na.rm = TRUE))

# ==============================================================================
# Plotting Functions
# ==============================================================================

plot_dbh_by_plot <- function(data = df, save = TRUE) {
  # Create histogram of DBH by plot, faceted by plot.
  # Parameters:
  #   data: data.frame with DBH_cm and Plot columns
  #   save: logical, whether to save the plot to file
  p <- ggplot(data, aes(x = dbh_cm)) +
    geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7, color = "white") +
    facet_wrap(~ plot, ncol = 3) +
    labs(
      title = "Distribution of DBH by Plot",
      subtitle = "Diameter at Breast Height (cm)",
      x = "DBH (cm)",
      y = "Frequency"
    ) +
    theme_minimal() +
    theme(
      plot.background = element_rect(fill = "white", color = "white"),
      panel.background = element_rect(fill = "white", color = "white")
    )
  
  if (save) {
    ggsave(
      filename = file.path(PLOTS_DIR, "dbh_histogram_by_plot.png"),
      plot = p,
      width = 12,
      height = 4,
      dpi = 300,
      bg = "white"
    )
    message("Saved: plots/dbh_histogram_by_plot.png")
  }
  
  return(p)
}

plot_growth_by_species <- function(data = df, save = TRUE) {
  # Create boxplot of growth by species, colored by plot.
  # Parameters:
  #   data: data.frame with GrowthRate, Species, and Plot columns
  #   save: logical, whether to save the plot to file
  # Filter out NA values and get top species by count
  growth_data <- data %>%
    filter(!is.na(growthrate), !is.na(species)) %>%
    mutate(species = as.factor(species))
  
  # Get top 10 species by count
  top_species <- growth_data %>%
    count(species, sort = TRUE) %>%
    head(10) %>%
    pull(species)
  
  growth_data <- growth_data %>%
    filter(species %in% top_species)
  
  p <- ggplot(growth_data, aes(x = reorder(species, growthrate, median, na.rm = TRUE), 
                                y = growthrate, 
                                fill = plot)) +
    geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
    coord_flip() +
    labs(
      title = "Growth Rate by Species (Top 10)",
      subtitle = "Colored by Plot",
      x = "Species",
      y = "Growth Rate",
      fill = "Plot"
    ) +
    theme_minimal() +
    theme(
      plot.background = element_rect(fill = "white", color = "white"),
      panel.background = element_rect(fill = "white", color = "white")
    ) +
    theme(legend.position = "bottom")
  
  if (save) {
    ggsave(
      filename = file.path(PLOTS_DIR, "growth_boxplot_by_species.png"),
      plot = p,
      width = 10,
      height = 8,
      dpi = 300,
      bg = "white"
    )
    message("Saved: plots/growth_boxplot_by_species.png")
  }
  
  return(p)
}

plot_dbh_timeseries <- function(data = df, save = TRUE) {
  # Create time series plot of average DBH per plot per year.
  # Parameters:
  #   data: data.frame with DBH_cm, Plot, and Year columns
  #   save: logical, whether to save the plot to file
  # Calculate mean DBH per plot per year
  ts_data <- data %>%
    filter(!is.na(dbh_cm), !is.na(year)) %>%
    group_by(plot, year) %>%
    summarise(
      mean_dbh = mean(dbh_cm, na.rm = TRUE),
      se_dbh = sd(dbh_cm, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  p <- ggplot(ts_data, aes(x = year, y = mean_dbh, color = plot, group = plot)) +
    geom_line(linewidth = 1.2, alpha = 0.8) +
    geom_point(size = 2.5, alpha = 0.8) +
    geom_errorbar(aes(ymin = mean_dbh - se_dbh, ymax = mean_dbh + se_dbh),
                  width = 0.3, alpha = 0.5) +
    labs(
      title = "Average DBH Over Time by Plot",
      subtitle = "Mean ± Standard Error",
      x = "Year",
      y = "Mean DBH (cm)",
      color = "Plot"
    ) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) +
    theme_minimal() +
    theme(
      plot.background = element_rect(fill = "white", color = "white"),
      panel.background = element_rect(fill = "white", color = "white")
    ) +
    theme(legend.position = "bottom")
  
  if (save) {
    ggsave(
      filename = file.path(PLOTS_DIR, "dbh_timeseries_by_plot.png"),
      plot = p,
      width = 10,
      height = 6,
      dpi = 300,
      bg = "white"
    )
    message("Saved: plots/dbh_timeseries_by_plot.png")
  }
  
  return(p)
}

plot_carbon_by_plot <- function(data = df, save = TRUE) {
  # Create time series plot of average carbon per plot per year.
  # Parameters:
  #   data: data.frame with Carbon, Plot, and Year columns
  #   save: logical, whether to save the plot to file
  if (!"carbon" %in% colnames(data)) {
    message("Warning: Carbon column not found. Skipping carbon plot.")
    return(NULL)
  }
  
  # Calculate mean carbon per plot per year
  carbon_data <- data %>%
    filter(!is.na(carbon), !is.na(year)) %>%
    group_by(plot, year) %>%
    summarise(
      mean_carbon = mean(carbon, na.rm = TRUE),
      se_carbon = sd(carbon, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )
  
  p <- ggplot(carbon_data, aes(x = year, y = mean_carbon, color = plot, group = plot)) +
    geom_line(linewidth = 1.2, alpha = 0.8) +
    geom_point(size = 2.5, alpha = 0.8) +
    geom_errorbar(aes(ymin = mean_carbon - se_carbon, ymax = mean_carbon + se_carbon),
                  width = 0.3, alpha = 0.5) +
    labs(
      title = "Average Carbon Storage Over Time by Plot",
      subtitle = "Mean ± Standard Error",
      x = "Year",
      y = "Mean Carbon (kg)",
      color = "Plot"
    ) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) +
    theme_minimal() +
    theme(
      plot.background = element_rect(fill = "white", color = "white"),
      panel.background = element_rect(fill = "white", color = "white")
    ) +
    theme(legend.position = "bottom")
  
  if (save) {
    ggsave(
      filename = file.path(PLOTS_DIR, "carbon_timeseries_by_plot.png"),
      plot = p,
      width = 10,
      height = 6,
      dpi = 300,
      bg = "white"
    )
    message("Saved: plots/carbon_timeseries_by_plot.png")
  }
  
  return(p)
}

# ==============================================================================
# Generate All Plots
# ==============================================================================

message("")
message("=")
message("Generating DBH plots...")
message("=")

p1 <- plot_dbh_by_plot()
p2 <- plot_growth_by_species()
p3 <- plot_dbh_timeseries()
p4 <- plot_carbon_by_plot()

message("")
message("=")
message("✓ All plots generated successfully!")
message("=")

