# ======= plot_sweep.py =======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

# -----------------------------------------------------------------------------
# 1) LOAD THE CSV PRODUCED BY master_main.py
# -----------------------------------------------------------------------------
df = pd.read_csv("sweep_results.csv")

# Assume SWEEP_REPLICATES is known (e.g. 10). If you used a
# different replicate count, replace 10.0 accordingly.
REPLICATES = 10.0
df["SURVIVAL_RATE"] = df["N_SURVIVED"] / REPLICATES

# Compute "division-to-meal" ratio
df["DIV_TO_MEAL_RATIO"] = df["MEAN_DIV_TIME"] / df["REFILL_INT"]

# Convert REFILL_INT → FREQ = 1 / REFILL_INT so the x-axis is frequency
df["FREQ"] = 1.0 / df["REFILL_INT"]

# -----------------------------------------------------------------------------
# 2) MAKE THREE PIVOT TABLES (INDEX = MEAL_AMOUNT, COLUMNS = FREQ)
# -----------------------------------------------------------------------------
# 2A) Survival-rate pivot
survival_pivot = df.pivot(
    index="MEAL_AMOUNT",
    columns="FREQ",
    values="SURVIVAL_RATE",
)
survival_pivot = survival_pivot.sort_index(ascending=True).sort_index(
    axis=1, ascending=True
)

# 2B) Mean-biomass pivot
biomass_pivot = df.pivot(
    index="MEAL_AMOUNT",
    columns="FREQ",
    values="MEAN_BIOMASS",
)
biomass_pivot = biomass_pivot.sort_index(ascending=True).sort_index(
    axis=1, ascending=True
)

# 2C) Division-to-meal ratio pivot
ratio_pivot = df.pivot(
    index="MEAL_AMOUNT",
    columns="FREQ",
    values="DIV_TO_MEAL_RATIO",
)
ratio_pivot = ratio_pivot.sort_index(ascending=True).sort_index(
    axis=1, ascending=True
)

# -----------------------------------------------------------------------------
# 3) PLOT ALL THREE HEATMAPS SIDE-BY-SIDE
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), sharey=True)

# 3A) Survival heatmap
sns.heatmap(
    survival_pivot,
    ax=axes[0],
    cmap="YlGnBu",
    vmin=0.0,
    vmax=1.0,
    cbar_kws={"label": "Survival Rate"},
    xticklabels=[f"{freq:.3f}" for freq in survival_pivot.columns],
    yticklabels=survival_pivot.index.astype(int),
)
axes[0].set_xlabel("Frequency (1 / Refill Interval)")
axes[0].set_ylabel("Meal Amount")
axes[0].set_title("Survival Rate")

# 3B) Biomass heatmap
sns.heatmap(
    biomass_pivot,
    ax=axes[1],
    cmap="viridis",
    cbar_kws={"label": "Mean Biomass"},
    xticklabels=[f"{freq:.3f}" for freq in biomass_pivot.columns],
    yticklabels=biomass_pivot.index.astype(int),
)
axes[1].set_xlabel("Frequency (1 / Refill Interval)")
axes[1].set_ylabel("")  # sharey ⇒ no y‐label duplicated
axes[1].set_title("Mean Biomass")

# 3C) Division‐to‐meal ratio heatmap (with log‐normalized colors)
min_nonzero = ratio_pivot.replace(0, np.nan).stack().min()
max_ratio   = ratio_pivot.stack().max()

sns.heatmap(
    ratio_pivot,
    ax=axes[2],
    cmap="rocket_r",
    norm=colors.LogNorm(vmin=max(min_nonzero, 1e-2), vmax=max_ratio),
    cbar_kws={"label": "Division Time ÷ Refill Interval (log scale)"},
    xticklabels=[f"{freq:.3f}" for freq in ratio_pivot.columns],
    yticklabels=ratio_pivot.index.astype(int),
)
axes[2].set_xlabel("Frequency (1 / Refill Interval)")
axes[2].set_ylabel("")  # sharey
axes[2].set_title("Avg Division Time ÷ Refill Interval (log‐norm)")

# Invert the y‐axis on all subplots so that larger Meal Amounts appear at top
for ax in axes:
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig("comparison_heatmaps.png", dpi=300)
print("Saved: comparison_heatmaps.png")
plt.close()
