# ==============================================================================
# Setup Packages for Carbon DBH R Analysis
# ==============================================================================
# This script ensures all required R packages are installed and loaded.
# Source this file at the beginning of other R scripts to guarantee
# the plotting environment is ready.
# ==============================================================================

# List of required packages
required_packages <- c(
  "tidyverse",    # Includes ggplot2, dplyr, readr, etc.
  "ggplot2",     # Plotting
  "readr",        # Reading CSV files
  "dplyr",        # Data manipulation
  "patchwork"     # Combining plots
)

# Function to check and install packages
check_and_install <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    message(paste("Installing", pkg, "..."))
    install.packages(pkg, dependencies = TRUE, repos = "https://cran.rstudio.com/")
    library(pkg, character.only = TRUE)
  } else {
    message(paste("✓", pkg, "is already installed"))
  }
}

# Install and load all required packages
message("=")
message("Setting up R packages for Carbon DBH analysis...")
message("=")

for (pkg in required_packages) {
  check_and_install(pkg)
}

message("")
message("=")
message("✓ All packages loaded successfully!")
message("=")
message("")

# Set default ggplot2 theme for publication-style plots
if (require(ggplot2, quietly = TRUE)) {
  theme_set(theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 10),
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold")
    ))
  message("✓ ggplot2 theme set to publication style")
}

