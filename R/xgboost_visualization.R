# ==============================================================================
# XGBoost Model Visualization Script
# ==============================================================================
# Creates visualizations for XGBoost DBH growth model including:
# - SHAP value plots
# - Feature importance plots
# - Cross-validation results
# - Model performance metrics
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

MODELS_DIR <- "Models"
PLOTS_DIR <- "plots/xgboost_model"
CV_RESULTS_PATH <- file.path(MODELS_DIR, "dbh_growth_model_cv_results.csv")
SELECTED_FEATURES_PATH <- file.path(MODELS_DIR, "dbh_growth_model_selected_features.txt")

# Create plots directory if it doesn't exist
if (!dir.exists(PLOTS_DIR)) {
  dir.create(PLOTS_DIR, recursive = TRUE)
  message("Created plots directory: ", PLOTS_DIR)
}

# ==============================================================================
# Load Data
# ==============================================================================

message("=")
message("Loading XGBoost Model Results")
message("=")

# Load CV results
if (file.exists(CV_RESULTS_PATH)) {
  cv_results <- read_csv(CV_RESULTS_PATH, show_col_types = FALSE)
  message("✓ Loaded CV results")
} else {
  warning("CV results file not found: ", CV_RESULTS_PATH)
  cv_results <- NULL
}

# Load selected features
if (file.exists(SELECTED_FEATURES_PATH)) {
  selected_features <- read_lines(SELECTED_FEATURES_PATH)
  message("✓ Loaded selected features (", length(selected_features), " features)")
} else {
  warning("Selected features file not found: ", SELECTED_FEATURES_PATH)
  selected_features <- NULL
}

# ==============================================================================
# Plot Functions
# ==============================================================================

plot_cv_results <- function(cv_results, save = TRUE) {
  # Plot cross-validation results showing R² scores across folds.
  # Parameters:
  #   cv_results: DataFrame with columns 'fold', 'r2_score', 'rmse'
  #   save: logical, whether to save the plot
  if (is.null(cv_results)) {
    warning("CV results not available. Skipping CV plot.")
    return(NULL)
  }
  
  p <- ggplot(cv_results, aes(x = factor(fold), y = r2_score)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
    geom_hline(yintercept = mean(cv_results$r2_score), 
               linetype = "dashed", color = "red", linewidth = 1) +
    geom_text(aes(label = sprintf("%.4f", r2_score)), 
              vjust = -0.5, size = 3) +
    labs(
      title = "5-Fold Cross-Validation Results",
      subtitle = paste0("Mean R² = ", sprintf("%.4f", mean(cv_results$r2_score)), 
                        " (±", sprintf("%.4f", sd(cv_results$r2_score)), ")"),
      x = "Fold",
      y = "R² Score"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 12),
      plot.background = element_rect(fill = "white", color = "white"),
      panel.background = element_rect(fill = "white", color = "white")
    )
  
  if (save) {
    filename <- file.path(PLOTS_DIR, "cv_results.png")
    ggsave(filename, plot = p, width = 10, height = 6, dpi = 300, bg = "white")
    message("Saved: ", filename)
  }
  
  return(p)
}

plot_cv_rmse <- function(cv_results, save = TRUE) {
  # Plot RMSE across CV folds.
  if (is.null(cv_results)) {
    warning("CV results not available. Skipping RMSE plot.")
    return(NULL)
  }
  
  p <- ggplot(cv_results, aes(x = factor(fold), y = rmse)) +
    geom_bar(stat = "identity", fill = "coral", alpha = 0.7) +
    geom_hline(yintercept = mean(cv_results$rmse), 
               linetype = "dashed", color = "red", linewidth = 1) +
    geom_text(aes(label = sprintf("%.2f", rmse)), 
              vjust = -0.5, size = 3) +
    labs(
      title = "5-Fold Cross-Validation RMSE",
      subtitle = paste0("Mean RMSE = ", sprintf("%.2f", mean(cv_results$rmse)), " cm"),
      x = "Fold",
      y = "RMSE (cm)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 12),
      plot.background = element_rect(fill = "white", color = "white"),
      panel.background = element_rect(fill = "white", color = "white")
    )
  
  if (save) {
    filename <- file.path(PLOTS_DIR, "cv_rmse.png")
    ggsave(filename, plot = p, width = 10, height = 6, dpi = 300, bg = "white")
    message("Saved: ", filename)
  }
  
  return(p)
}

plot_feature_selection <- function(selected_features, save = TRUE) {
  # Visualize selected features from RFECV.
  if (is.null(selected_features) || length(selected_features) == 0) {
    warning("Selected features not available. Skipping feature selection plot.")
    return(NULL)
  }
  
  # Create a summary
  feature_df <- data.frame(
    feature = selected_features,
    selected = TRUE
  ) %>%
    mutate(
      feature_type = case_when(
        startsWith(feature, "Species_") ~ "Species",
        startsWith(feature, "Plot_") ~ "Plot",
        startsWith(feature, "Group_") ~ "Group",
        startsWith(feature, "GrowthType_") ~ "GrowthType",
        TRUE ~ "Other"
      )
    )
  
  # Count by type
  type_counts <- feature_df %>%
    count(feature_type) %>%
    arrange(desc(n))
  
  p <- ggplot(type_counts, aes(x = reorder(feature_type, n), y = n)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
    geom_text(aes(label = n), hjust = -0.2, size = 4) +
    coord_flip() +
    labs(
      title = "Selected Features by Type",
      subtitle = paste("Total features selected:", length(selected_features)),
      x = "Feature Type",
      y = "Number of Features"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold")
    )
  
  if (save) {
    filename <- file.path(PLOTS_DIR, "feature_selection.png")
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
message("Generating XGBoost Model Visualizations")
message("=")

# Plot CV results
if (!is.null(cv_results)) {
  plot_cv_results(cv_results)
  plot_cv_rmse(cv_results)
}

# Plot feature selection
plot_feature_selection(selected_features)

message("")
message("=")
message("XGBoost Visualization Complete")
message("=")
message("")
message("Plots saved to: ", PLOTS_DIR)
message("")

