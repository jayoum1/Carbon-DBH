# ==============================================================================
# Model Diagnostics Plotting Script
# ==============================================================================
# Creates residual diagnostics plots for linear regression models
# ==============================================================================

# Source package setup
source("R/setup_packages.R")

# Load required libraries
library(ggplot2)
library(dplyr)
library(readr)

# ==============================================================================
# Configuration
# ==============================================================================

# Note: This script expects model results from Python analysis
# For now, we'll create diagnostics from the encoded dataset
# In the future, you can save model predictions/residuals to CSV

DATA_PATH <- "Data/Processed Data/Carbon/all_plots_with_carbon_encoded.csv"
DIAGNOSTICS_DIR <- "plots/model_diagnostics"

# Create diagnostics directory if it doesn't exist
if (!dir.exists(DIAGNOSTICS_DIR)) {
  dir.create(DIAGNOSTICS_DIR, recursive = TRUE)
  message("Created diagnostics directory: ", DIAGNOSTICS_DIR)
}

# ==============================================================================
# Helper Functions
# ==============================================================================

plot_residuals_vs_fitted <- function(fitted_values, residuals, plot_name = NULL, save = TRUE) {
  # Create residuals vs fitted values plot.
  # Parameters:
  #   fitted_values: numeric vector of fitted/predicted values
  #   residuals: numeric vector of residual values (observed - predicted)
  #   plot_name: optional character vector for faceting
  #   save: logical, whether to save the plot
  plot_data <- data.frame(
    fitted = fitted_values,
    residuals = residuals
  )
  
  if (!is.null(plot_name)) {
    plot_data$plot <- plot_name
  }
  
  p <- ggplot(plot_data, aes(x = fitted, y = residuals)) +
    geom_point(alpha = 0.5, size = 0.8) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
    geom_smooth(method = "loess", se = TRUE, color = "blue", alpha = 0.3) +
    labs(
      title = "Residuals vs Fitted Values",
      subtitle = "Check for heteroscedasticity and non-linear patterns",
      x = "Fitted Values",
      y = "Residuals"
    ) +
    theme_minimal()
  
  if (!is.null(plot_name)) {
    p <- p + facet_wrap(~ plot)
  }
  
  if (save) {
    if (!is.null(plot_name)) {
      # Handle vector plot_name: create separate files for each unique plot
      if (length(plot_name) > 1) {
        unique_plots <- unique(plot_name)
        for (plot_val in unique_plots) {
          filename <- file.path(DIAGNOSTICS_DIR, paste0("residuals_vs_fitted_", plot_val, ".png"))
          ggsave(
            filename = filename,
            plot = p,
            width = 10,
            height = 6,
            dpi = 300
          )
          message("Saved: ", filename)
        }
      } else {
        # Single plot_name value
        filename <- file.path(DIAGNOSTICS_DIR, paste0("residuals_vs_fitted_", plot_name, ".png"))
        ggsave(
          filename = filename,
          plot = p,
          width = 10,
          height = 6,
          dpi = 300
        )
        message("Saved: ", filename)
      }
    } else {
      filename <- file.path(DIAGNOSTICS_DIR, "residuals_vs_fitted.png")
      ggsave(
        filename = filename,
        plot = p,
        width = 10,
        height = 6,
        dpi = 300
      )
      message("Saved: ", filename)
    }
  }
  
  return(p)
}

plot_residual_histogram <- function(residuals, plot_name = NULL, save = TRUE) {
  # Create histogram of residuals.
  # Parameters:
  #   residuals: numeric vector of residual values
  #   plot_name: optional character vector for faceting
  #   save: logical, whether to save the plot
  plot_data <- data.frame(residuals = residuals)
  
  if (!is.null(plot_name)) {
    plot_data$plot <- plot_name
  }
  
  p <- ggplot(plot_data, aes(x = residuals)) +
    geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7, color = "white") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "red", linewidth = 1) +
    labs(
      title = "Distribution of Residuals",
      subtitle = "Check for normality",
      x = "Residuals",
      y = "Frequency"
    ) +
    theme_minimal()
  
  if (!is.null(plot_name)) {
    p <- p + facet_wrap(~ plot, scales = "free")
  }
  
  if (save) {
    if (!is.null(plot_name)) {
      # Handle vector plot_name: create separate files for each unique plot
      if (length(plot_name) > 1) {
        unique_plots <- unique(plot_name)
        for (plot_val in unique_plots) {
          filename <- file.path(DIAGNOSTICS_DIR, paste0("residuals_histogram_", plot_val, ".png"))
          ggsave(
            filename = filename,
            plot = p,
            width = 10,
            height = 6,
            dpi = 300
          )
          message("Saved: ", filename)
        }
      } else {
        # Single plot_name value
        filename <- file.path(DIAGNOSTICS_DIR, paste0("residuals_histogram_", plot_name, ".png"))
        ggsave(
          filename = filename,
          plot = p,
          width = 10,
          height = 6,
          dpi = 300
        )
        message("Saved: ", filename)
      }
    } else {
      filename <- file.path(DIAGNOSTICS_DIR, "residuals_histogram.png")
      ggsave(
        filename = filename,
        plot = p,
        width = 10,
        height = 6,
        dpi = 300
      )
      message("Saved: ", filename)
    }
  }
  
  return(p)
}

plot_residual_qq <- function(residuals, plot_name = NULL, save = TRUE) {
  # Create Q-Q plot of residuals.
  # Parameters:
  #   residuals: numeric vector of residual values
  #   plot_name: optional character vector for faceting
  #   save: logical, whether to save the plot
  plot_data <- data.frame(residuals = residuals)
  
  if (!is.null(plot_name)) {
    plot_data$plot <- plot_name
  }
  
  p <- ggplot(plot_data, aes(sample = residuals)) +
    stat_qq(alpha = 0.5) +
    stat_qq_line(color = "red", linetype = "dashed", linewidth = 1) +
    labs(
      title = "Q-Q Plot of Residuals",
      subtitle = "Check for normality (points should follow red line)",
      x = "Theoretical Quantiles",
      y = "Sample Quantiles"
    ) +
    theme_minimal()
  
  if (!is.null(plot_name)) {
    p <- p + facet_wrap(~ plot)
  }
  
  if (save) {
    if (!is.null(plot_name)) {
      # Handle vector plot_name: create separate files for each unique plot
      if (length(plot_name) > 1) {
        unique_plots <- unique(plot_name)
        for (plot_val in unique_plots) {
          filename <- file.path(DIAGNOSTICS_DIR, paste0("residuals_qq_", plot_val, ".png"))
          ggsave(
            filename = filename,
            plot = p,
            width = 10,
            height = 6,
            dpi = 300
          )
          message("Saved: ", filename)
        }
      } else {
        # Single plot_name value
        filename <- file.path(DIAGNOSTICS_DIR, paste0("residuals_qq_", plot_name, ".png"))
        ggsave(
          filename = filename,
          plot = p,
          width = 10,
          height = 6,
          dpi = 300
        )
        message("Saved: ", filename)
      }
    } else {
      filename <- file.path(DIAGNOSTICS_DIR, "residuals_qq.png")
      ggsave(
        filename = filename,
        plot = p,
        width = 10,
        height = 6,
        dpi = 300
      )
      message("Saved: ", filename)
    }
  }
  
  return(p)
}

# ==============================================================================
# Generate Diagnostics from Data
# ==============================================================================
# Note: This is a placeholder that demonstrates the structure.
# In practice, you would load model predictions/residuals from a CSV file
# saved by your Python modeling script.

message("=")
message("Model Diagnostics Script")
message("=")
message("")
message("Note: This script expects model results CSV.")
message("For now, creating example structure.")
message("")
message("To use this script:")
message("1. Save model predictions and residuals from Python to CSV")
message("2. Update DATA_PATH to point to that CSV")
message("3. Adjust column names in the code below")
message("")

# Example: If you have a CSV with columns: fitted, residuals, plot
# Uncomment and modify this section:

# results <- read_csv(DATA_PATH, show_col_types = FALSE)
# 
# # Generate diagnostics
# plot_residuals_vs_fitted(
#   fitted_values = results$fitted,
#   residuals = results$residuals,
#   plot_name = results$plot,
#   save = TRUE
# )
# 
# plot_residual_histogram(
#   residuals = results$residuals,
#   plot_name = results$plot,
#   save = TRUE
# )
# 
# plot_residual_qq(
#   residuals = results$residuals,
#   plot_name = results$plot,
#   save = TRUE
# )

message("=")
message("Diagnostics functions are ready to use!")
message("=")
message("")
message("Available functions:")
message("  - plot_residuals_vs_fitted(fitted_values, residuals, plot_name)")
message("  - plot_residual_histogram(residuals, plot_name)")
message("  - plot_residual_qq(residuals, plot_name)")
message("")

