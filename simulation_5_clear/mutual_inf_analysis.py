import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

# Step 1: Load parameter sets
params_df = pd.read_csv("parameters_2.csv")
ts_folder = Path("C:/Users/reawe/Desktop/Vienna/simulation_5_clear/timeseries")
param_cols = [col for col in params_df.columns if col != "number_of_simulation"]

# Step 2: Load time series and compute behavioral metrics (averaging over replicates)
required_cols = ['bacteria', 'phage', 'energy', 'majA', 'majB', 'majC']

def extract_metrics(ts):
    return {
        "avg_bacteria": ts["bacteria"].mean(),
        "avg_phage": ts["phage"].mean(),
        "avg_energy": ts["energy"].mean(),
        "survival_time": ts[ts["bacteria"] > 1].shape[0],
        "switch_count_A": (ts["majA"].diff().fillna(0) != 0).sum(),
        "switch_count_B": (ts["majB"].diff().fillna(0) != 0).sum(),
        "switch_count_C": (ts["majC"].diff().fillna(0) != 0).sum(),
    }

metrics = []

for i in range(1001):
    pattern = ts_folder / f"sim_{i:04d}_rep_*.csv"
    replicate_files = glob.glob(str(pattern))

    if not replicate_files:
        print(f"Missing: {pattern}")
        dummy = pd.DataFrame(columns=required_cols)
        metrics.append({k: np.nan for k in extract_metrics(dummy)})
        continue

    replicate_metrics = []
    for file in replicate_files:
        ts = pd.read_csv(file)
        if all(col in ts.columns for col in required_cols):
            replicate_metrics.append(extract_metrics(ts))

    if replicate_metrics:
        df_reps = pd.DataFrame(replicate_metrics)
        metrics.append(df_reps.mean().to_dict())
    else:
        dummy = pd.DataFrame(columns=required_cols)
        metrics.append({k: np.nan for k in extract_metrics(dummy)})

metrics_df = pd.DataFrame(metrics).dropna()
params_df = params_df.loc[metrics_df.index].reset_index(drop=True)
metrics_df = metrics_df.reset_index(drop=True)

# Step 3: Cluster based on behavior
gmm = GaussianMixture(n_components=6, random_state=42)
cluster_labels = gmm.fit_predict(metrics_df)

# Step 4a: Mutual Information (parameter → behavior cluster)
mi_scores = mutual_info_classif(params_df[param_cols].values, cluster_labels)
print("Mutual Information scores:")
for col, score in zip(param_cols, mi_scores):
    print(f"{col}: MI = {score:.4f}")

# Step 4b: PCA of parameter space
scaler = StandardScaler()
param_scaled = scaler.fit_transform(params_df[param_cols])
pca = PCA(n_components=2)
param_pca = pca.fit_transform(param_scaled)

df_pca = pd.DataFrame(param_pca, columns=["PCA1", "PCA2"])
df_pca["cluster"] = cluster_labels

# Step 4c: Stability scoring
k = 10
nn = NearestNeighbors(n_neighbors=k+1).fit(param_scaled)
_, indices = nn.kneighbors(param_scaled)

stability_scores = []
for i, neighbors in enumerate(indices):
    neighbor_clusters = cluster_labels[neighbors[1:]]  # exclude self
    score = np.mean(neighbor_clusters == cluster_labels[i])
    stability_scores.append(score)

df_pca["stability_score"] = stability_scores
print(f"Mean stability score: {np.mean(stability_scores):.4f}")

# Step 4d: Variance-sensitive DBSCAN
def variance_sensitive_dbscan(data, base_eps=0.3, min_samples=5, max_var=0.03):
    for eps in np.linspace(base_eps, base_eps * 2, 5):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        labels = db.labels_
        if len(set(labels)) > 1:
            df = pd.DataFrame(data)
            df["label"] = labels
            variances = df.groupby("label").var().mean(axis=1)
            if all(variances[variances.index != -1] < max_var):
                return labels, eps, variances
    return np.full(len(data), -1), None, None

dbscan_labels, best_eps, label_variances = variance_sensitive_dbscan(param_scaled)
print(f"Best DBSCAN eps: {best_eps}")
if label_variances is not None:
    print("Cluster variances:\n", label_variances)

# Step 5: Plots
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="cluster", palette="tab10", s=20)
plt.title("PCA of Parameters Colored by GMM Cluster")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="stability_score", palette="viridis", s=20)
plt.title("PCA of Parameters Colored by Stability Score")
plt.tight_layout()
plt.show()
