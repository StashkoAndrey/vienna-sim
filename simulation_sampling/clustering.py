# Updated clustering.py: now clusters directly in UMAP space and removes reclustering
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from umap import UMAP
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import os
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

# ───── CONFIG ─────
logging.basicConfig(level=logging.INFO)
BEHAVIOR_FILE = "phase_metrics.csv"
PARAM_FILE = "parameters_2.csv"
OUT_FILE = "parameter_centroids_by_umap_cluster.csv"
PLOT_DIR = "umap_clustering_plots"
os.makedirs(PLOT_DIR, exist_ok=True)
VARIABILITY_THRESHOLD = 0.2  # STD / range

# ───── Load and Merge Data ─────
logging.info("Loading data...")
df_behavior = pd.read_csv(BEHAVIOR_FILE)
df_params = pd.read_csv(PARAM_FILE)
df = df_behavior.merge(df_params, left_on="sim_id", right_index=True).dropna()

# ───── Clustering Input ─────
metric_cols = [
    "phase_switch_count",
    "dominance_ratio", "avg_phage",
]

df_clean = df[metric_cols].replace([np.inf, -np.inf], np.nan).dropna()
X = StandardScaler().fit_transform(df_clean)
df = df.loc[df_clean.index]  # Align

# ───── UMAP for later visualization ─────
umap_embed = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)
df["umap_x"] = umap_embed[:, 0]
df["umap_y"] = umap_embed[:, 1]

# ───── Clustering directly in UMAP space ─────
logging.info("Running GMM and HDBSCAN clustering in UMAP space...")
df["cluster_gmm"] = GaussianMixture(n_components=6, random_state=42).fit_predict(df[["umap_x", "umap_y"]])
df["cluster_hdb"] = hdbscan.HDBSCAN(min_cluster_size=100).fit_predict(df[["umap_x", "umap_y"]])

# ───── Cluster Variability and Centroids ─────
def process_clusters(method):
    logging.info(f"Analyzing {method.upper()} clusters...")
    results, variability_records = [], []
    cluster_col = f"cluster_{method}"
    param_cols = df_params.columns

    for cluster_id in sorted(df[cluster_col].unique()):
        if cluster_id == -1:
            continue
        cluster_df = df[df[cluster_col] == cluster_id]
        variability = {}
        centroids = {}

        for param in param_cols:
            vals = cluster_df[param]
            std = np.std(vals)
            rng = np.max(vals) - np.min(vals)
            rel_std = std / rng if rng != 0 else 0
            variability[param] = rel_std
            centroids[param] = np.mean(vals)

        variability_records.append(pd.Series(variability, name=f"{method}_cluster_{cluster_id}"))

        results.append({
            "cluster_method": method,
            "cluster_id": f"{cluster_id}",
            "n_members": len(cluster_df),
            "score": len(cluster_df),
            **centroids
        })

    return results, pd.DataFrame(variability_records).T

# ───── Run for GMM and HDB ─────
gmm_results, gmm_var = process_clusters("gmm")
hdb_results, hdb_var = process_clusters("hdb")

# ───── Save Outputs ─────
df_out = pd.DataFrame(gmm_results + hdb_results)
df_out.to_csv(OUT_FILE, index=False)
gmm_var.to_csv("gmm_param_variability.csv")
hdb_var.to_csv("hdb_param_variability.csv")
logging.info(f"✓ Results saved to {OUT_FILE}")

# ───── Variability Plots ─────
def plot_variability(df_var, name):
    df_var.plot(kind="bar", figsize=(16, 6))
    plt.title(f"{name.upper()} Parameter Variability per Cluster")
    plt.ylabel("Relative Std (std/range)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}_param_variability.png")
    plt.close()

plot_variability(gmm_var, "gmm")
plot_variability(hdb_var, "hdb")

# ───── UMAP Filled Visualization ─────
def plot_umap_clusters(df, cluster_col):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="umap_x", y="umap_y",
        hue=cluster_col,
        palette="tab10",
        s=20,
        linewidth=0,
        alpha=0.9
    )
    plt.title(f"UMAP Projection Colored by {cluster_col}")
    plt.xlabel("umap_x")
    plt.ylabel("umap_y")
    plt.legend(title=cluster_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/umap_{cluster_col}_clusters.png", dpi=300)
    plt.close()

# ───── Plot Both Cluster Methods ─────
for method in ["gmm", "hdb"]:
    cluster_col = f"cluster_{method}"
    if cluster_col in df.columns:
        plot_umap_clusters(df, cluster_col)

# ───── Final Message ─────
logging.info("✓ All cluster visualizations and analysis complete.")