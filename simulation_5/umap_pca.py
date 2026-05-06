# ======= project_umap_pca_parallel.py =======
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP
import multiprocessing as mp
import os
import numpy as np

# ───────────── Setup ─────────────
metric_cols = [
    "phase_switch_count", "dwell_time_entropy", "spectral_entropy",
    "dominance_ratio", "avg_bacteria", "avg_phage",
    "survival_time", "zero_crossings"
]

df = pd.read_csv("phase_metrics.csv")
df_clean = df[metric_cols].replace([float("inf"), -float("inf")], pd.NA).dropna()
X_scaled = StandardScaler().fit_transform(df_clean)

# Защита от x < -1 и NaN
df["log_phase_switch_count"] = np.log1p(
    df["phase_switch_count"].clip(lower=-0.9999).fillna(0)
)


# ───────────── Projections ─────────────
umap_embed = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X_scaled)
pca_embed = PCA(n_components=2).fit_transform(X_scaled)

df.loc[df_clean.index, "umap_x"] = umap_embed[:, 0]
df.loc[df_clean.index, "umap_y"] = umap_embed[:, 1]
df.loc[df_clean.index, "pca_x"] = pca_embed[:, 0]
df.loc[df_clean.index, "pca_y"] = pca_embed[:, 1]
df.to_csv("phase_metrics_with_projections.csv", index=False)

os.makedirs("plots_umap_pca", exist_ok=True)

# ───────────── Parallel plotting task ─────────────
def plot_metric(metric):
    print(f"→ Starting plot for {metric}...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    norm = LogNorm() if (df[metric] > 0).all() else None

    sns.scatterplot(
        x="umap_x", y="umap_y", hue=metric,
        data=df, ax=axes[0], palette="viridis",
        hue_norm=norm, s=10, edgecolor=None, legend=False
    )
    axes[0].set_title(f"UMAP: {metric}")

    sns.scatterplot(
        x="pca_x", y="pca_y", hue=metric,
        data=df, ax=axes[1], palette="viridis",
        hue_norm=norm, s=10, edgecolor=None, legend=False
    )
    axes[1].set_title(f"PCA: {metric}")

    plt.tight_layout()
    fig.savefig(f"plots/{metric}_umap_pca.png")
    plt.close()
    print(f"✓ Saved plots/{metric}_umap_pca.png")

# ───────────── Run in parallel ─────────────
if __name__ == "__main__":
    print("Starting parallel plotting...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(plot_metric, metric_cols)
    print("All plots saved in 'plots/'")

# ───────────── Optional: One big combined grid ─────────────
def plot_big_grid():
    print("Creating combined grid plot...")
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    for i, metric in enumerate(metric_cols):
        norm = LogNorm() if (df[metric] > 0).all() else None
        sns.scatterplot(
            x="umap_x", y="umap_y", hue=metric,
            data=df, ax=axes[2*i], palette="viridis",
            hue_norm=norm, s=10, edgecolor=None, legend=False
        )
        axes[2*i].set_title(f"UMAP: {metric}")

        sns.scatterplot(
            x="pca_x", y="pca_y", hue=metric,
            data=df, ax=axes[2*i+1], palette="viridis",
            hue_norm=norm, s=10, edgecolor=None, legend=False
        )
        axes[2*i+1].set_title(f"PCA: {metric}")
    plt.tight_layout()
    fig.savefig("plots_umap_pca/all_metric_projections_grid.png")
    print("Combined grid saved as plots_umap_pca/all_metric_projections_grid.png")

plot_big_grid()
