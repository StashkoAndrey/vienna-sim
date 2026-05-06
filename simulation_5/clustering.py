import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from umap import UMAP
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ───── CONFIG ─────
logging.basicConfig(level=logging.INFO)
BEHAVIOR_FILE = "phase_metrics.csv"
PARAM_FILE = "parameters_2.csv"
OUT_FILE = "parameter_centroids_by_cluster.csv"
VARIABILITY_THRESHOLD = 0.2  # STD / range
PLOT_DIR = "cluster_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ───── Load and Merge Data ─────
logging.info("Loading data...")
df_behavior = pd.read_csv(BEHAVIOR_FILE)
df_params = pd.read_csv(PARAM_FILE)
df = df_behavior.merge(df_params, left_on="sim_id", right_index=True).dropna()

# ───── Clustering Input ─────
metric_cols = [
    "phase_switch_count", "dwell_time_entropy", "spectral_entropy",
    "dominance_ratio", "avg_bacteria", "avg_phage",
    "survival_time", "zero_crossings"
]

df_clean = df[metric_cols].replace([np.inf, -np.inf], np.nan).dropna()
X = StandardScaler().fit_transform(df_clean)
df = df.loc[df_clean.index]  # Align

# ───── UMAP for later visualization ─────
umap_embed = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)
df["umap_x"] = umap_embed[:, 0]
df["umap_y"] = umap_embed[:, 1]

# ───── Clustering ─────
logging.info("Running GMM and HDBSCAN clustering...")
df["cluster_gmm"] = GaussianMixture(n_components=6, random_state=42).fit_predict(X)
df["cluster_hdb"] = hdbscan.HDBSCAN(min_cluster_size=50).fit_predict(X)

# ───── Analysis, Reclustering, Centroids ─────
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

        variable_params = [p for p, v in variability.items() if v > VARIABILITY_THRESHOLD]
        variability_records.append(pd.Series(variability, name=f"{method}_cluster_{cluster_id}"))

        # Reclustering (within 8D metric space)
        if variable_params:
            logging.info(f"Reclustering {method.upper()} cluster {cluster_id} (var params: {variable_params})")
            subset_metrics = StandardScaler().fit_transform(cluster_df[metric_cols])
            sub_labels = GaussianMixture(n_components=2, random_state=42).fit_predict(subset_metrics)
            df.loc[cluster_df.index, f"{cluster_col}_sub"] = sub_labels
        else:
            df.loc[cluster_df.index, f"{cluster_col}_sub"] = 0

        results.append({
            "cluster_method": method,
            "cluster_id": cluster_id,
            "n_members": len(cluster_df),
            "variable_params": ", ".join(variable_params),
            **centroids
        })

    return results, pd.DataFrame(variability_records).T

gmm_results, gmm_var = process_clusters("gmm")
hdb_results, hdb_var = process_clusters("hdb")

# ───── Save Outputs ─────
df_out = pd.DataFrame(gmm_results + hdb_results)
df_out.to_csv(OUT_FILE, index=False)
gmm_var.to_csv("gmm_param_variability.csv")
hdb_var.to_csv("hdb_param_variability.csv")
logging.info(f"✓ Results saved to {OUT_FILE}")

# ───── Parameter Variability Plots ─────
def plot_variability(df_var, name):
    df_var.plot(kind="bar", figsize=(16, 6))
    plt.title(f"{name.upper()} Parameter Variability per Cluster")
    plt.ylabel("Relative Std (std/range)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}_param_variability.png")
    plt.close()

plot_variability(gmm_var, "gmm")
plot_variability(hdb_var, "hdb")

# ───── UMAP Colored by Cluster ─────
for method in ["gmm", "hdb"]:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="umap_x", y="umap_y", hue=f"cluster_{method}", palette="tab10", s=20)
    plt.title(f"UMAP Projection Colored by {method.upper()} Clusters")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/umap_{method}_clusters.png")
    plt.close()

# ───── Parameter Space Pairplots ─────
plot_cols = df_params.columns[:6]  # adjust as needed
for method in ["gmm", "hdb"]:
    sns.pairplot(df, vars=plot_cols, hue=f"cluster_{method}", plot_kws={'alpha': 0.6})
    plt.suptitle(f"{method.upper()} Clusters in Parameter Space", y=1.02)
    plt.savefig(f"{PLOT_DIR}/pairplot_{method}.png")
    plt.close()

# ───── Individual Per-Cluster Variability Plots ─────
def plot_per_cluster_variability(df_var, method):
    for cluster_id in df_var.columns:
        plt.figure(figsize=(10, 4))
        df_var[cluster_id].sort_values().plot(kind='bar')
        plt.title(f"{method.upper()} Cluster {cluster_id} – Parameter Variability")
        plt.ylabel("Relative Std (std / range)")
        plt.tight_layout()
        filename = f"{PLOT_DIR}/{method}_cluster_{cluster_id}_variability.png"
        plt.savefig(filename)
        plt.close()
        logging.info(f"Saved variability plot → {filename}")

plot_per_cluster_variability(gmm_var, "gmm")
plot_per_cluster_variability(hdb_var, "hdb")


logging.info("✓ All cluster visualizations and analysis complete.")
