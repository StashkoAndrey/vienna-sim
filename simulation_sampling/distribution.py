import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ───── Load and filter data ─────
df = pd.read_csv("phase_metrics.csv")
df = df[(df["avg_phage"] > 0) & (df["avg_phage"] < 1e36)]
df = df[df["phase_switch_count"] > 0]
df = df[df["avg_bacteria"] > 0]  # log-safe

# ───── Log transform both variables to reduce skew ─────
df["log_phage"] = np.log10(df["avg_phage"] + 1e-3)
df["log_bacteria"] = np.log10(df["avg_bacteria"] + 1e-3)

# ───── Use quantile binning on both log-transformed values ─────
df["bacteria_bin"] = pd.qcut(df["log_bacteria"], q=20, duplicates="drop")
df["phage_bin"] = pd.qcut(df["log_phage"], q=20, duplicates="drop")

# ───── Compute average phase switch count per bin pair ─────
pivot = df.pivot_table(
    index="phage_bin",
    columns="bacteria_bin",
    values="phase_switch_count",
    aggfunc="mean"
)

# ───── Plot heatmap ─────
plt.figure(figsize=(16, 10))
sns.heatmap(
    pivot,
    cmap="viridis",
    linewidths=0.2,
    linecolor="gray",
    cbar_kws={"label": "Avg Phase Switch Count"},
    square=False
)
plt.title("Heatmap: Phase Switching by log10(Bacteria) & log10(Phage) (Quantile Bins)")
plt.xlabel("log10(avg_bacteria) (quantile binned)")
plt.ylabel("log10(avg_phage) (quantile binned)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
