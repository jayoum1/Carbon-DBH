# XGBoost DBH Growth Model

This document describes the XGBoost-based DBH growth model with RFECV feature selection and SHAP value analysis.

## Overview

The model predicts tree diameter growth from one year to the next using:
- **XGBoost Regressor** as the base model
- **5-fold Cross-Validation** for robust evaluation
- **RFECV (Recursive Feature Elimination with Cross-Validation)** for feature selection
- **SHAP values** for model interpretation

## Model Architecture

### Features
- **PrevDBH_cm**: Previous year's DBH (in cm) - most important feature
- **Species features**: One-hot encoded species (22 features)
- **Plot features**: One-hot encoded plot locations (2 features)
- **GapYears**: Years between measurements
- **Group features**: Tree group classification (softwood/hardwood)
- **GrowthType features**: Growth type indicators

### Training Process

1. **Data Preparation**
   - Load processed dataset (`all_plots_with_carbon_encoded.csv`)
   - Create `PrevDBH_cm` by shifting `DBH_cm` within each TreeID group
   - Filter rows with valid `PrevDBH_cm` and `DBH_cm`

2. **Feature Selection (RFECV)**
   - Use 5-fold CV to evaluate feature subsets
   - Recursively eliminate least important features
   - Select optimal number of features based on CV score

3. **Model Training**
   - Train XGBoost on selected features
   - Perform 5-fold CV for final evaluation
   - Compute SHAP values for model interpretation

4. **Model Evaluation**
   - R² score (coefficient of determination)
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - Performance by plot location

## Usage

### Training the Model

```bash
python3 src/models/dbh_growth_model.py
```

This will:
- Train the XGBoost model with RFECV
- Save the model to `Models/dbh_growth_model.pkl`
- Save selected features to `Models/dbh_growth_model_selected_features.txt`
- Save CV results to `Models/dbh_growth_model_cv_results.csv`
- Save SHAP values to `Models/dbh_growth_model_shap_values.npy`
- Generate SHAP summary plot and RFECV plot

### Making Predictions

```python
from src.models.dbh_growth_model import predict_dbh_next_year

# Predict next year's DBH
next_dbh = predict_dbh_next_year(
    prev_dbh_cm=25.0,      # Current DBH
    species='red oak',     # Species name
    plot='Upper',          # Plot location
    gap_years=1.0          # Years until next measurement
)
```

**Important**: The function interprets `prev_dbh_cm` as the current DBH and returns the predicted DBH for next year.

## Visualization

### R Scripts

#### XGBoost Model Visualization (`R/xgboost_visualization.R`)

Generates plots for:
- Cross-validation results (R² scores across folds)
- RMSE across CV folds
- Feature selection summary (by feature type)

**Usage:**
```bash
Rscript R/xgboost_visualization.R
```

**Output:** `plots/xgboost_model/`

#### PrevDBH Visualization (`R/dbh_prev_visualization.R`)

Creates visualizations with `PrevDBH_cm` as the y-variable:
- Scatter plot: Current DBH vs Previous DBH
- Growth vs Previous DBH
- Growth rate vs Previous DBH
- Distribution of Previous DBH
- Previous DBH by plot

**Usage:**
```bash
Rscript R/dbh_prev_visualization.R
```

**Output:** `plots/dbh_prev/`

## Model Files

After training, the following files are created:

- `Models/dbh_growth_model.pkl` - Trained XGBoost model
- `Models/dbh_growth_model_selected_features.txt` - Selected feature names
- `Models/dbh_growth_model_cv_results.csv` - Cross-validation results
- `Models/dbh_growth_model_shap_values.npy` - SHAP values array
- `Models/dbh_growth_model_shap_metadata.pkl` - SHAP explainer and test samples
- `Models/dbh_growth_model_shap_summary.png` - SHAP summary plot
- `Models/dbh_growth_model_rfecv.png` - RFECV feature selection plot

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Key packages:
- `xgboost>=2.0.0` - XGBoost gradient boosting
- `shap>=0.42.0` - SHAP value computation
- `scikit-learn>=1.2.0` - RFECV and CV utilities
- `pandas`, `numpy`, `matplotlib` - Data manipulation and visualization

## Performance Metrics

The model will display:
- **Test Set R²**: Coefficient of determination on test set
- **Test Set RMSE**: Root mean squared error (in cm)
- **Test Set MAE**: Mean absolute error (in cm)
- **CV R²**: Mean R² across 5 folds with standard deviation
- **Feature Importance**: Top 10 most important features

## SHAP Values

SHAP (SHapley Additive exPlanations) values provide:
- **Feature importance**: Which features contribute most to predictions
- **Feature effects**: How each feature affects predictions
- **Individual predictions**: Explanation for specific predictions

The SHAP summary plot shows:
- Feature importance (y-axis)
- Feature value distribution
- Impact on prediction (color: red increases, blue decreases)

## Notes

- The model uses `PrevDBH_cm` as the primary feature (typically >99% importance)
- RFECV helps identify the optimal feature subset
- SHAP values provide interpretability beyond simple feature importance
- The model is cached after first load for faster predictions

