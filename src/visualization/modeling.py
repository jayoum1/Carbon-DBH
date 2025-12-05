import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

plt.rcParams["figure.figsize"] = (7, 5)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.edgecolor"] = "white"

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CARBON_ALL_PLOTS_ENCODED, GRAPHS_LINEAR_REGRESSION_DIR

# ------------------------------------------------------------
# 1. Paths and target selection
# ------------------------------------------------------------

DATA_PATH = str(CARBON_ALL_PLOTS_ENCODED)
SAVE_DIR  = str(GRAPHS_LINEAR_REGRESSION_DIR)

os.makedirs(SAVE_DIR, exist_ok=True)

def savefig(fig, name):
    out_path = os.path.join(SAVE_DIR, name)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='white')
    print("Saved:", out_path)

# Choose which target you want to visualize:
TARGET_COL = "CarbonGrowthRate"   # change to "CarbonGrowth" to switch

# ------------------------------------------------------------
# 2. Load and prepare data
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column {TARGET_COL} not found.")

# Drop rows with NaN in target
df_model = df[~df[TARGET_COL].isna()].copy()
print("Shape after dropping NaN target rows:", df_model.shape)

# Exclude non-feature columns
exclude_cols = [
    "TreeID",
    TARGET_COL,
    "Carbon",
    "CO2e",
    "PrevCarbon",
]

numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude_cols]

X = df_model[feature_cols]
y = df_model[TARGET_COL]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Fit model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)

# ------------------------------------------------------------
# 3. Plot y_true vs y_pred with 1:1 line
# ------------------------------------------------------------

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.3)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)  # 1:1 line

ax.set_xlabel("True values")
ax.set_ylabel("Predicted values")
ax.set_title(f"{TARGET_COL}: True vs Predicted (Linear Regression)")
plt.tight_layout()
savefig(fig, f"{TARGET_COL}_true_vs_pred.png")
plt.show()

# ------------------------------------------------------------
# 4. Plot target vs key individual predictors + simple trendline
# ------------------------------------------------------------

# Choose a few important / interpretable predictors to visualize
# Make sure these appear in feature_cols
candidate_features = [
    "DBH",        # tree size in inches
    "DBH_cm",     # same in cm
    "Year",
    "GapYears",   # effect of gap length on growth
]

# Keep only those that actually exist
plot_features = [f for f in candidate_features if f in feature_cols]

for feat in plot_features:
    x = df_model[feat]
    y_full = df_model[TARGET_COL]

    # Optional: trim extreme outliers visually
    mask = x.notna() & y_full.notna()
    x = x[mask]
    y_full = y_full[mask]

    # Quantile trim to avoid huge outliers wrecking the view
    low_x, high_x = x.quantile([0.01, 0.99])
    low_y, high_y = y_full.quantile([0.01, 0.99])
    keep = (x >= low_x) & (x <= high_x) & (y_full >= low_y) & (y_full <= high_y)

    x = x[keep]
    y_trim = y_full[keep]

    # Fit simple univariate linear trendline for visualization
    if len(x) > 2:
        slope, intercept = np.polyfit(x, y_trim, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
    else:
        x_line, y_line = None, None

    fig, ax = plt.subplots()
    ax.scatter(x, y_trim, alpha=0.2)
    if x_line is not None:
        ax.plot(x_line, y_line, 'r-', linewidth=2, label="Trendline")

    ax.set_xlabel(feat)
    ax.set_ylabel(TARGET_COL)
    ax.set_title(f"{TARGET_COL} vs {feat} (with simple trendline)")
    if x_line is not None:
        ax.legend()
    plt.tight_layout()
    savefig(fig, f"{TARGET_COL}_vs_{feat}.png")
    plt.show()

