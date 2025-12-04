# Carbon DBH Analysis Project

Data analysis project exploring carbon sequestration and tree growth in the Pomfret School forest.

## Overview

This project analyzes DBH (diameter at breast height) data for ~450 trees across 3 forest plots (upper, middle, lower) to:
- Predict DBH growth (year → year)
- Model carbon storage
- Forecast future forest structure over time

## Project Structure

```
Carbon DBH/
├── src/                          # Source code
│   ├── config.py                # Centralized paths & constants
│   ├── preprocessing/            # Data cleaning & transformation
│   │   ├── transform.py         # DBH transformation (wide → long)
│   │   ├── carbon_calc.py       # Carbon calculations
│   │   ├── species.py           # Species standardization
│   │   ├── growth.py            # Growth calculations
│   │   ├── outliers.py          # Outlier handling
│   │   └── encoding.py          # One-hot encoding
│   ├── forestry/                # Domain-specific modules
│   │   ├── species_classifier.py # Hardwood/softwood classification
│   │   ├── allometry.py         # DBH → biomass/carbon equations
│   │   └── valuation.py         # Timber value calculations (future)
│   ├── analysis/                 # Statistical analysis
│   │   └── modeling.py          # Linear regression modeling
│   ├── visualization/           # Plotting & visualization
│   │   ├── eda.py              # EDA plots
│   │   └── modeling.py         # Model diagnostic plots
│   └── crawling/                # External data (future)
│       └── timber_prices_crawler.py
├── Data/
│   ├── Raw Data/                # Original CSV files
│   └── Processed Data/          # Cleaned & transformed data
├── Graphs/                       # Generated visualizations
├── Models/                       # Saved model files
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jayoum1/Carbon-DBH.git
cd Carbon-DBH
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Preprocessing Pipeline

Run preprocessing scripts in order:

```bash
# 1. Transform raw data to long format
python3 src/preprocessing/transform.py

# 2. Add carbon calculations
python3 src/preprocessing/carbon_calc.py

# 3. Standardize species names
python3 src/preprocessing/species.py

# 4. Add CarbonGrowth column
python3 src/preprocessing/growth.py

# 5. Handle outliers (optional)
python3 src/preprocessing/outliers.py

# 6. One-hot encode for modeling
python3 src/preprocessing/encoding.py
```

### Analysis

```bash
# Run linear regression modeling
python3 src/analysis/modeling.py

# Generate EDA visualizations
python3 src/visualization/eda.py

# Generate modeling diagnostic plots
python3 src/visualization/modeling.py
```

### Data Quality Check

```bash
python3 check_data_quality.py
```

## Data

- **Raw Data**: 3 CSV files (Upper, Middle, Lower plots)
- **Processed Data**: Long-format datasets with growth rates and carbon calculations
- **Years**: 2015-2025
- **Trees**: ~450 trees across 3 plots

## Key Features

- **Modular Architecture**: Clean separation of preprocessing, analysis, and visualization
- **Forestry Domain Logic**: Reusable allometric equations and species classification
- **Centralized Configuration**: All paths and constants in `src/config.py`
- **Data Quality Checks**: Automated validation scripts
- **Species Standardization**: Handles naming inconsistencies automatically

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- scipy >= 1.10.0
- statsmodels >= 0.14.0
- matplotlib >= 3.6.0

## Results

Current model performance:
- **CarbonGrowthRate**: R² = 0.111, RMSE = 2.67
- **CarbonGrowth**: R² = 0.214, RMSE = 93.02

Key findings:
- Plot location significantly affects CarbonGrowthRate (p < 0.05)
- DBH is a strong predictor for absolute carbon growth
- Species effects are not statistically significant

## Future Work

- [ ] Advanced ML models (Random Forest, Gradient Boosting)
- [ ] Timber price crawler integration
- [ ] Web app/API for interactive analysis
- [ ] 3D visualization of forest structure
- [ ] Real-time external data integration

## License

MIT License - see LICENSE file for details

## Author

Jay Youm

