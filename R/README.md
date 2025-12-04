# R Scripts for Carbon DBH Analysis

This directory contains R scripts for data visualization and model diagnostics.

## Scripts

### `setup_packages.R`
Sets up the R environment by checking and installing required packages:
- tidyverse (includes ggplot2, dplyr, readr)
- patchwork (for combining plots)

**Usage:**
```r
source("R/setup_packages.R")
```

### `plots_dbh.R`
Creates publication-quality plots for DBH analysis.

**Functions:**
- `plot_dbh_by_plot()` - Histogram of DBH distribution by plot
- `plot_growth_by_species()` - Boxplot of growth rates by species
- `plot_dbh_timeseries()` - Time series of average DBH per plot
- `plot_carbon_by_plot()` - Time series of average carbon per plot

**Usage:**
```r
source("R/plots_dbh.R")
# All plots are generated automatically and saved to plots/
```

**Output:**
- `plots/dbh_histogram_by_plot.png`
- `plots/growth_boxplot_by_species.png`
- `plots/dbh_timeseries_by_plot.png`
- `plots/carbon_timeseries_by_plot.png`

### `model_diagnostics.R`
Creates residual diagnostics plots for regression models.

**Functions:**
- `plot_residuals_vs_fitted()` - Residuals vs fitted values
- `plot_residual_histogram()` - Distribution of residuals
- `plot_residual_qq()` - Q-Q plot for normality check

**Usage:**
```r
source("R/model_diagnostics.R")

# Example: After loading model results
results <- read_csv("path/to/model_results.csv")
plot_residuals_vs_fitted(results$fitted, results$residuals, results$plot)
plot_residual_histogram(results$residuals, results$plot)
plot_residual_qq(results$residuals, results$plot)
```

**Output:**
- `plots/model_diagnostics/residuals_vs_fitted.png`
- `plots/model_diagnostics/residuals_histogram.png`
- `plots/model_diagnostics/residuals_qq.png`

## Running Scripts

From the project root directory:

```bash
# Run DBH plots
Rscript R/plots_dbh.R

# Run model diagnostics (requires model results CSV)
Rscript R/model_diagnostics.R
```

Or from R/RStudio:

```r
setwd("/path/to/Carbon DBH")
source("R/plots_dbh.R")
```

## Requirements

All required packages are automatically installed by `setup_packages.R`:
- tidyverse >= 2.0.0
- ggplot2 >= 3.5.0
- readr >= 2.1.0
- dplyr >= 1.1.0
- patchwork >= 1.1.0

