# ============================================================
# EDA NOTEBOOK: DBH, CARBON, AND GROWTH – POMFRET FOREST
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["savefig.edgecolor"] = "white"

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CARBON_ALL_PLOTS, GRAPHS_EDA_DIR

# ------------------------------------------------------------
# 0. Paths: SET THESE
# ------------------------------------------------------------

DATA_PATH = str(CARBON_ALL_PLOTS)
SAVE_DIR  = str(GRAPHS_EDA_DIR)

os.makedirs(SAVE_DIR, exist_ok=True)

def savefig(fig, name):
    """Helper to save figures to the SAVE_DIR with a consistent style."""
    out_path = os.path.join(SAVE_DIR, name)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='white')
    print(f"Saved: {out_path}")

# ------------------------------------------------------------
# Helper: outlier trimming for visualization
# ------------------------------------------------------------

def trim_df_by_quantile(df, col, low_q=0.01, high_q=0.99):
    """
    Return a trimmed copy of df keeping only rows where `col`
    is between the chosen quantiles.
    """
    mask = df[col].notna()
    if mask.sum() == 0:
        return df.copy(), None

    low, high = df.loc[mask, col].quantile([low_q, high_q])
    trimmed = df[mask & (df[col] >= low) & (df[col] <= high)].copy()
    return trimmed, (low, high)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df['Year'] = df['Year'].astype(int)
df['TreeID'] = df['TreeID'].astype(str)

print("Data shape:", df.shape)
print(df.head())

print("\nColumns:", df.columns.tolist())
print("\nPlots:", df['Plot'].unique())
print("\nYears:", sorted(df['Year'].unique()))
print("\nExample species:", df['Species'].value_counts().head(10))

plots = sorted(df['Plot'].unique())

# ------------------------------------------------------------
# 2. DBH vs Year per Plot (sampled trees + mean per plot)
# ------------------------------------------------------------

max_trees_per_plot = 50

fig, ax = plt.subplots()

for plot_name, group in df.groupby('Plot'):
    tree_ids = group['TreeID'].unique()
    if len(tree_ids) == 0:
        continue
    sample_ids = np.random.choice(
        tree_ids,
        size=min(len(tree_ids), max_trees_per_plot),
        replace=False
    )
    sub = group[group['TreeID'].isin(sample_ids)]

    # Thin lines per tree
    for tid, gtree in sub.groupby('TreeID'):
        ax.plot(gtree['Year'], gtree['DBH'], alpha=0.2)

# Mean DBH per year per plot
for plot_name, group in df.groupby('Plot'):
    mean_dbh = group.groupby('Year')['DBH'].mean()
    ax.plot(mean_dbh.index, mean_dbh.values, label=f"{plot_name} mean", linewidth=2)

ax.set_xlabel("Year")
ax.set_ylabel("DBH (inches)")
ax.set_title("DBH vs Year (sampled trees + plot means)")
ax.legend()
plt.tight_layout()
savefig(fig, "01_DBH_vs_Year_by_Plot.png")
plt.show()

# ------------------------------------------------------------
# 3. Mean Carbon per Year by Plot
# ------------------------------------------------------------

fig, ax = plt.subplots()

for plot_name, group in df.groupby('Plot'):
    mean_carbon = group.groupby('Year')['Carbon'].mean()
    ax.plot(mean_carbon.index, mean_carbon.values, marker='o', label=plot_name)

ax.set_xlabel("Year")
ax.set_ylabel("Mean Carbon")
ax.set_title("Mean Carbon per Tree vs Year by Plot")
ax.legend()
plt.tight_layout()
savefig(fig, "02_Mean_Carbon_vs_Year_by_Plot.png")
plt.show()

# ------------------------------------------------------------
# 4. Distribution of CarbonGrowthRate (trimmed)
# ------------------------------------------------------------

growth_clean = df['CarbonGrowthRate'].dropna()
growth_df = df[~df['CarbonGrowthRate'].isna()].copy()
growth_df_trim, growth_bounds = trim_df_by_quantile(growth_df, 'CarbonGrowthRate',
                                                    low_q=0.01, high_q=0.99)
print("CarbonGrowthRate trim bounds:", growth_bounds)

fig, ax = plt.subplots()
ax.hist(growth_df_trim['CarbonGrowthRate'], bins=40)
ax.set_xlabel("Carbon Growth Rate (annualized)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Carbon Growth Rate (trimmed)")
plt.tight_layout()
savefig(fig, "03_Distribution_CarbonGrowthRate_trimmed.png")
plt.show()

# By plot (trimmed per-plot)
fig, axes = plt.subplots(1, len(plots), figsize=(5 * len(plots), 4), sharey=True)

if len(plots) == 1:
    axes = [axes]

for ax, plot_name in zip(axes, plots):
    sub = df[(df['Plot'] == plot_name) & (~df['CarbonGrowthRate'].isna())].copy()
    sub_trim, bounds = trim_df_by_quantile(sub, 'CarbonGrowthRate',
                                           low_q=0.01, high_q=0.99)
    ax.hist(sub_trim['CarbonGrowthRate'], bins=30)
    ax.set_title(f"{plot_name}\ntrim={bounds}")
    ax.set_xlabel("Carbon Growth Rate")

axes[0].set_ylabel("Count")
plt.suptitle("Distribution of Carbon Growth Rate by Plot (trimmed)")
plt.tight_layout()
savefig(fig, "04_Distribution_CarbonGrowthRate_by_Plot_trimmed.png")
plt.show()

# ------------------------------------------------------------
# 5. Mean CarbonGrowthRate vs Year by Plot (trimmed)
# ------------------------------------------------------------

growth_clean = df[~df['CarbonGrowthRate'].isna()].copy()
growth_clean_trim, _ = trim_df_by_quantile(growth_clean, 'CarbonGrowthRate',
                                           low_q=0.01, high_q=0.99)

fig, ax = plt.subplots()
for plot_name, group in growth_clean_trim.groupby('Plot'):
    mean_growth = group.groupby('Year')['CarbonGrowthRate'].mean()
    ax.plot(mean_growth.index, mean_growth.values, marker='o', label=plot_name)

ax.set_xlabel("Year")
ax.set_ylabel("Mean Carbon Growth Rate (trimmed)")
ax.set_title("Mean Carbon Growth Rate vs Year by Plot (trimmed)")
ax.legend()
plt.tight_layout()
savefig(fig, "05_Mean_CarbonGrowthRate_vs_Year_by_Plot_trimmed.png")
plt.show()

# ------------------------------------------------------------
# 6. DBH_cm vs Carbon (sanity check) - trim extreme DBH & Carbon
# ------------------------------------------------------------

df_trim_dbh, dbh_bounds = trim_df_by_quantile(df, 'DBH_cm', low_q=0.01, high_q=0.99)
df_trim_dbh, carbon_bounds = trim_df_by_quantile(df_trim_dbh, 'Carbon',
                                                 low_q=0.01, high_q=0.99)
print("DBH_cm trim bounds:", dbh_bounds)
print("Carbon trim bounds (for scatter):", carbon_bounds)

fig, ax = plt.subplots()
ax.scatter(df_trim_dbh['DBH_cm'], df_trim_dbh['Carbon'], alpha=0.2)
ax.set_xlabel("DBH (cm)")
ax.set_ylabel("Carbon")
ax.set_title("DBH (cm) vs Carbon (trimmed)")
plt.tight_layout()
savefig(fig, "06_DBHcm_vs_Carbon_trimmed.png")
plt.show()

# ------------------------------------------------------------
# 7. Boxplots by Species (Carbon & CarbonGrowthRate) – top N species
#    With special handling: drop the single highest red oak Carbon outlier
#    ONLY for the Carbon boxplot, so the graph is readable.
# ------------------------------------------------------------

top_n_species = 5
top_species = df['Species'].value_counts().head(top_n_species).index
df_top_species = df[df['Species'].isin(top_species)].copy()

# ---- Remove the single highest Carbon red oak point (for visualization only) ----
# Species names should already be lowercase after your cleaning script.
red_oak_mask = df_top_species['Species'] == 'red oak'

if red_oak_mask.any():
    # Index of the row with the max Carbon among red oaks
    idx_max_red_oak = df_top_species.loc[red_oak_mask, 'Carbon'].idxmax()
    print("Dropping extreme red oak Carbon outlier at index:", idx_max_red_oak,
          "with Carbon =", df_top_species.loc[idx_max_red_oak, 'Carbon'])

    # Create a copy *without* that one point for the Carbon boxplot
    df_top_species_no_ro_max = df_top_species.drop(index=idx_max_red_oak)
else:
    # If for some reason there's no 'red oak' in top_species, just use the original
    df_top_species_no_ro_max = df_top_species

species_order = list(top_species)

# ---- Carbon by Species (with highest red oak Carbon removed) ----
fig, ax = plt.subplots()
data_to_plot = [
    df_top_species_no_ro_max[df_top_species_no_ro_max['Species'] == sp]['Carbon'].dropna()
    for sp in species_order
]
ax.boxplot(data_to_plot, labels=species_order)
ax.set_xlabel("Species")
ax.set_ylabel("Carbon")
ax.set_title("Carbon by Species (Top Species, red oak max outlier removed)")
plt.xticks(rotation=45)
plt.tight_layout()
savefig(fig, "07_Boxplot_Carbon_by_Species_topN_redoak_max_removed.png")
plt.show()

# ---- CarbonGrowthRate by Species (trimmed, as before) ----
# (This part stays basically the same – no manual red oak removal here.)
df_top_species_trim, _ = trim_df_by_quantile(df_top_species, 'CarbonGrowthRate',
                                             low_q=0.01, high_q=0.99)

fig, ax = plt.subplots()
data_to_plot = [
    df_top_species_trim[df_top_species_trim['Species'] == sp]['CarbonGrowthRate'].dropna()
    for sp in species_order
]
ax.boxplot(data_to_plot, labels=species_order)
ax.set_xlabel("Species")
ax.set_ylabel("Carbon Growth Rate")
ax.set_title("Carbon Growth Rate by Species (Top Species, trimmed)")
plt.xticks(rotation=45)
plt.tight_layout()
savefig(fig, "08_Boxplot_CarbonGrowthRate_by_Species_topN_trimmed.png")
plt.show()

# ------------------------------------------------------------
# 8. CarbonGrowthRate vs DBH – with trimming + hard cap
# ------------------------------------------------------------

growth_clean = df[~df['CarbonGrowthRate'].isna()].copy()

# Quantile trim first (remove truly insane points)
growth_trim, bounds_growth = trim_df_by_quantile(growth_clean, 'CarbonGrowthRate',
                                                 low_q=0.01, high_q=0.99)
growth_trim, bounds_dbh = trim_df_by_quantile(growth_trim, 'DBH',
                                              low_q=0.01, high_q=0.99)

# Then apply a hard cap to CarbonGrowthRate for visualization
# (e.g., keep only between -0.5 and 0.5 per year)
hard_min, hard_max = -0.5, 0.5
growth_trim = growth_trim[
    (growth_trim['CarbonGrowthRate'] >= hard_min) &
    (growth_trim['CarbonGrowthRate'] <= hard_max)
].copy()

print("Growth trim bounds for scatter (quantile-based):", bounds_growth)
print("DBH trim bounds for scatter:", bounds_dbh)
print("Hard cap applied to CarbonGrowthRate:", (hard_min, hard_max))

fig, ax = plt.subplots()
ax.scatter(growth_trim['DBH'], growth_trim['CarbonGrowthRate'], alpha=0.2)
ax.set_xlabel("DBH (inches)")
ax.set_ylabel("Carbon Growth Rate")
ax.set_title("Carbon Growth Rate vs DBH (trimmed + capped)")
plt.tight_layout()
savefig(fig, "09_Scatter_CarbonGrowthRate_vs_DBH_trimmed_capped.png")
plt.show()

# By plot
fig, axes = plt.subplots(1, len(plots), figsize=(5 * len(plots), 4), sharey=True)
if len(plots) == 1:
    axes = [axes]

for ax, plot_name in zip(axes, plots):
    sub = growth_trim[growth_trim['Plot'] == plot_name]
    ax.scatter(sub['DBH'], sub['CarbonGrowthRate'], alpha=0.2)
    ax.set_title(plot_name)
    ax.set_xlabel("DBH (inches)")

axes[0].set_ylabel("Carbon Growth Rate")
plt.suptitle("Carbon Growth Rate vs DBH by Plot (trimmed + capped)")
plt.tight_layout()
savefig(fig, "10_Scatter_CarbonGrowthRate_vs_DBH_by_Plot_trimmed_capped.png")
plt.show()

# ------------------------------------------------------------
# 8b. CarbonGrowth vs DBH – absolute carbon gain per year
# ------------------------------------------------------------

if 'CarbonGrowth' not in df.columns:
    print("WARNING: 'CarbonGrowth' column not found. "
          "Run the CarbonGrowth creation script first.")
else:
    cg_clean = df[~df['CarbonGrowth'].isna()].copy()

    # Trim CarbonGrowth and DBH via quantiles
    cg_trim, cg_bounds = trim_df_by_quantile(cg_clean, 'CarbonGrowth',
                                             low_q=0.01, high_q=0.99)
    cg_trim, dbh_bounds_cg = trim_df_by_quantile(cg_trim, 'DBH',
                                                 low_q=0.01, high_q=0.99)

    print("CarbonGrowth trim bounds (quantile-based):", cg_bounds)
    print("DBH trim bounds for CarbonGrowth scatter:", dbh_bounds_cg)

    fig, ax = plt.subplots()
    ax.scatter(cg_trim['DBH'], cg_trim['CarbonGrowth'], alpha=0.2)
    ax.set_xlabel("DBH (inches)")
    ax.set_ylabel("Carbon Growth (absolute, per year)")
    ax.set_title("Carbon Growth vs DBH (trimmed)")
    plt.tight_layout()
    savefig(fig, "09b_Scatter_CarbonGrowth_vs_DBH_trimmed.png")
    plt.show()

    # By plot
    fig, axes = plt.subplots(1, len(plots), figsize=(5 * len(plots), 4), sharey=True)
    if len(plots) == 1:
        axes = [axes]

    for ax, plot_name in zip(axes, plots):
        sub = cg_trim[cg_trim['Plot'] == plot_name]
        ax.scatter(sub['DBH'], sub['CarbonGrowth'], alpha=0.2)
        ax.set_title(plot_name)
        ax.set_xlabel("DBH (inches)")

    axes[0].set_ylabel("Carbon Growth (absolute, per year)")
    plt.suptitle("Carbon Growth vs DBH by Plot (trimmed)")
    plt.tight_layout()
    savefig(fig, "09c_Scatter_CarbonGrowth_vs_DBH_by_Plot_trimmed.png")
    plt.show()


# ------------------------------------------------------------
# 9. CarbonGrowthRate vs Carbon (trimmed)
# ------------------------------------------------------------

growth_trim2, _ = trim_df_by_quantile(growth_clean, 'CarbonGrowthRate',
                                      low_q=0.01, high_q=0.99)
growth_trim2, _ = trim_df_by_quantile(growth_trim2, 'Carbon',
                                      low_q=0.01, high_q=0.99)

fig, ax = plt.subplots()
ax.scatter(growth_trim2['Carbon'], growth_trim2['CarbonGrowthRate'], alpha=0.2)
ax.set_xlabel("Carbon")
ax.set_ylabel("Carbon Growth Rate")
ax.set_title("Carbon Growth Rate vs Carbon (trimmed)")
plt.tight_layout()
savefig(fig, "11_Scatter_CarbonGrowthRate_vs_Carbon_trimmed.png")
plt.show()

# ------------------------------------------------------------
# 10. Total CO2e per Plot per Year
# ------------------------------------------------------------

grouped_co2 = df.groupby(['Plot', 'Year'])['CO2e'].sum().reset_index()

fig, ax = plt.subplots()
for plot_name, g in grouped_co2.groupby('Plot'):
    ax.plot(g['Year'], g['CO2e'], marker='o', label=plot_name)

ax.set_xlabel("Year")
ax.set_ylabel("Total CO2e (sum over trees)")
ax.set_title("Total CO2e per Plot per Year")
ax.legend()
plt.tight_layout()
savefig(fig, "12_Total_CO2e_per_Plot_per_Year.png")
plt.show()

# ------------------------------------------------------------
# 11. Missing Data Pattern Heatmap (TreeID x Year) for one plot
# ------------------------------------------------------------

plot_to_show = 'Upper'  # change if you want
df_plot = df[df['Plot'] == plot_to_show].copy()

presence = df_plot.pivot_table(
    index='TreeID',
    columns='Year',
    values='DBH',
    aggfunc=lambda x: 1,
    fill_value=0
)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(presence.values, aspect='auto')

ax.set_xlabel("Year")
ax.set_ylabel("TreeID (rows)")
ax.set_title(f"Measurement Presence Heatmap – {plot_to_show} Plot")

ax.set_xticks(range(len(presence.columns)))
ax.set_xticklabels(presence.columns, rotation=90, fontsize=8)
ax.set_yticks([])

cbar = plt.colorbar(im, ax=ax, label="Measurement present (1) or missing (0)")
plt.tight_layout()
savefig(fig, f"13_MissingData_Heatmap_{plot_to_show}.png")
plt.show()

plot_to_show = 'Middle'  # change if you want
df_plot = df[df['Plot'] == plot_to_show].copy()

presence = df_plot.pivot_table(
    index='TreeID',
    columns='Year',
    values='DBH',
    aggfunc=lambda x: 1,
    fill_value=0
)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(presence.values, aspect='auto')

ax.set_xlabel("Year")
ax.set_ylabel("TreeID (rows)")
ax.set_title(f"Measurement Presence Heatmap – {plot_to_show} Plot")

ax.set_xticks(range(len(presence.columns)))
ax.set_xticklabels(presence.columns, rotation=90, fontsize=8)
ax.set_yticks([])

cbar = plt.colorbar(im, ax=ax, label="Measurement present (1) or missing (0)")
plt.tight_layout()
savefig(fig, f"13_MissingData_Heatmap_{plot_to_show}.png")
plt.show()

plot_to_show = 'Lower'  # change if you want
df_plot = df[df['Plot'] == plot_to_show].copy()

presence = df_plot.pivot_table(
    index='TreeID',
    columns='Year',
    values='DBH',
    aggfunc=lambda x: 1,
    fill_value=0
)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(presence.values, aspect='auto')

ax.set_xlabel("Year")
ax.set_ylabel("TreeID (rows)")
ax.set_title(f"Measurement Presence Heatmap – {plot_to_show} Plot")

ax.set_xticks(range(len(presence.columns)))
ax.set_xticklabels(presence.columns, rotation=90, fontsize=8)
ax.set_yticks([])

cbar = plt.colorbar(im, ax=ax, label="Measurement present (1) or missing (0)")
plt.tight_layout()
savefig(fig, f"13_MissingData_Heatmap_{plot_to_show}.png")
plt.show()

# ------------------------------------------------------------
# 12. Multicollinearity heatmap (correlation matrix)
# ------------------------------------------------------------

# Choose the main continuous variables you care about
heatmap_vars = [
    'CarbonGrowthRate',
    'CarbonGrowth',
    'Carbon',
    'PrevCarbon',
    'DBH',
    'DBH_cm',
    'PrevDBH',
    'GapYears',
    'Year'
]

# Keep only those that actually exist in the dataframe
heatmap_vars = [c for c in heatmap_vars if c in df.columns]

corr = df[heatmap_vars].corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='coolwarm')

ax.set_xticks(np.arange(len(heatmap_vars)))
ax.set_yticks(np.arange(len(heatmap_vars)))
ax.set_xticklabels(heatmap_vars, rotation=45, ha='right')
ax.set_yticklabels(heatmap_vars)

fig.colorbar(im, ax=ax, label='Correlation')
ax.set_title("Correlation Heatmap (Multicollinearity)")
plt.tight_layout()
savefig(fig, "_Correlation_Heatmap.png")
plt.show()

# ============================================================
# END OF EDA SCRIPT
# ============================================================

