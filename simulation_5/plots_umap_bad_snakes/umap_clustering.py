# ======= umap_clustering.py =======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from umap import UMAP
import hdbscan

# ────── Load both CSVs ──────
df_params = pd.read_csv("parameters_2.csv")
df_summary = pd.read_csv("simulation_summary.csv")

# ────── Merge by number_of_simulation ──────
df_summary = df_summary.rename(columns={"number_of_simulation": "sim_id"})
df_summary["sim_id"] = df_summary["sim_id"].astype(int)
df_params["sim_id"] = df_params.index + 1  # Ensure alignment

df = pd.merge(df_params, df_summary, on="sim_id")

# ────── Extract behavioral outcomes ──────
outcome_cols = ["avg_phage", "avg_energy", "avg_bacteria", "survival_time"]
X = df[outcome_cols].replace([np.inf, -np.inf], np.nan).dropna()
clean_idx = X.index  # keep track of which rows are valid
X_scaled = StandardScaler().fit_transform(X)

# ────── UMAP projection ──────
embedding = UMAP(n_neighbors=45, min_dist=0.5, random_state=42).fit_transform(X_scaled)

# ────── Clustering 1: HDBSCAN ──────
hdb = hdbscan.HDBSCAN(min_cluster_size=5)
labels_hdb = hdb.fit_predict(embedding)

# ────── Clustering 2: GMM ──────
gmm = GaussianMixture(n_components=4, random_state=42)
labels_gmm = gmm.fit_predict(embedding)

# ────── Store results ──────
df_combined = df.loc[clean_idx].copy()
df_combined["umap_x"] = embedding[:, 0]
df_combined["umap_y"] = embedding[:, 1]
df_combined["cluster_hdb"] = labels_hdb
df_combined["cluster_gmm"] = labels_gmm
df_combined["log_avg_bacteria"] = np.log1p(df_combined["avg_bacteria"])


# ────── Save combined data ──────
df_combined.to_csv("combined_data.csv", index=False)
print("✓ Saved → combined_data.csv")

# ────── Plot: UMAP colored by avg_bacteria ──────
plt.figure(figsize=(8,6))
sns.scatterplot(
    x="umap_x", y="umap_y",
    hue="log_avg_bacteria",  # log-transformed
    data=df_combined,
    palette="viridis",
    s=50
)
plt.title("UMAP Projection Colored by avg_bacteria")
plt.tight_layout()
plt.savefig("umap_behavior_plot_bacteria.png")
plt.show()


import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# Extract outcome data (same cleaned version used for UMAP)
X = df[outcome_cols].replace([np.inf, -np.inf], np.nan).dropna()

# Compute Spearman correlation matrix
corr_spearman, p_values = stats.spearmanr(X)

# Convert to DataFrame for readability
corr_df = pd.DataFrame(corr_spearman, index=outcome_cols, columns=outcome_cols)

# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Spearman Correlation Matrix")
plt.tight_layout()
plt.show()
plt.savefig("spearman_correlation_matrix.png")


'''
# ────── Plot: UMAP colored by avg_phage ──────
plt.figure(figsize=(8,6))
sns.scatterplot(x="umap_x", y="umap_y", hue="avg_phage", data=df_combined,
                palette="viridis", s=50)
plt.title("UMAP Projection Colored by avg_phage")
plt.tight_layout()
plt.savefig("umap_behavior_plot_phage.png")
plt.show()

# ────── Plot: UMAP colored by avg_energy ──────
plt.figure(figsize=(8,6))
sns.scatterplot(x="umap_x", y="umap_y", hue="avg_energy", data=df_combined,
                palette="viridis", s=50)
plt.title("UMAP Projection Colored by avg_energy")
plt.tight_layout()
plt.savefig("umap_behavior_plot_energy.png")
plt.show()

# ────── Plot: UMAP colored by survival_time ──────
plt.figure(figsize=(8,6))
sns.scatterplot(x="umap_x", y="umap_y", hue="survival_time", data=df_combined,
                palette="viridis", s=50)
plt.title("UMAP Projection Colored by survival_time")
plt.tight_layout()
plt.savefig("umap_behavior_plot_survival.png")
plt.show()

# ────── Plot: UMAP colored by HDBSCAN cluster ──────
plt.figure(figsize=(8,6))
sns.scatterplot(x="umap_x", y="umap_y", hue="cluster_hdb", data=df_combined,
                palette="tab10", s=50)
plt.title("UMAP Projection Colored by HDBSCAN Cluster")
plt.tight_layout()
plt.savefig("umap_cluster_hdbscan.png")
plt.show()

# ────── Plot: UMAP colored by GMM cluster ──────
plt.figure(figsize=(8,6))
sns.scatterplot(x="umap_x", y="umap_y", hue="cluster_gmm", data=df_combined,
                palette="tab10", s=50)
plt.title("UMAP Projection Colored by GMM Cluster")
plt.tight_layout()
plt.savefig("umap_cluster_gmm.png")
plt.show()
'''