import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

# === Загрузка данных ===
metrics_df = pd.read_csv("clusters.csv")
params_df = pd.read_csv("parameters_2.csv")

# Совмещаем по sim_id <-> number of simulation
merged_df = pd.merge(
    metrics_df,
    params_df,
    left_on="sim_id",
    right_on="number of simulation"
)

# === Метрики поведения для кластеризации ===
metric_cols = [
    'phase_switch_count','dwell_time_entropy','spectral_entropy',
    'dominance_ratio','avg_bacteria','avg_phage','survival_time'
]
X_metrics = merged_df[metric_cols]

# Стандартизация метрик
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_metrics)

# === Нормализация параметров для оценки гетерогенности ===
param_cols = [col for col in params_df.columns if col != 'number of simulation']
param_max = {col: (1 if merged_df[col].max() <= 1 else 1000) for col in param_cols}
for col in param_cols:
    merged_df[col + '_norm'] = merged_df[col] / param_max[col]
param_cols_norm = [col + '_norm' for col in param_cols]

# === Функция для оценки внутрикластерной гетерогенности параметров ===
def compute_param_heterogeneity(df, labels, param_cols_norm):
    df = df.copy()
    df['cluster'] = labels
    grouped = df.groupby('cluster')
    stds = grouped[param_cols_norm].std()
    mean_stds = stds.mean(axis=1)
    return mean_stds.mean()

# === GMM: параллельный подбор оптимального k ===
def gmm_score(k):
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X_scaled)
    bic = gmm.bic(X_scaled)
    heterogeneity = compute_param_heterogeneity(merged_df, labels, param_cols_norm)
    score = bic + 5000 * heterogeneity
    return (k, labels, score)

results_gmm = Parallel(n_jobs=4)(
    delayed(gmm_score)(k) for k in tqdm(range(2, 21), desc="GMM BIC + Heterogeneity")
)

best_gmm_k, best_gmm_labels, best_gmm_score = min(results_gmm, key=lambda x: x[2])
print(f"✓ GMM: Best k={best_gmm_k} with combined score={best_gmm_score:.2f}")

# === HDBSCAN: параллельный подбор min_cluster_size ===
def hdbscan_score(min_size):
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_size)
    labels = hdb.fit_predict(X_scaled)
    if len(set(labels)) <= 1 or -1 in set(labels):
        return (min_size, labels, np.inf)
    heterogeneity = compute_param_heterogeneity(merged_df, labels, param_cols_norm)
    penalty = 5000 * heterogeneity + 100 * len(set(labels))
    return (min_size, labels, penalty)

results_hdb = Parallel(n_jobs=4)(
    delayed(hdbscan_score)(min_size) for min_size in tqdm(range(10, 101, 10), desc="HDBSCAN Optimization")
)

best_hdb_size, best_hdb_labels, best_hdb_score = min(results_hdb, key=lambda x: x[2])
print(f"✓ HDBSCAN: Best min_cluster_size={best_hdb_size} with combined penalty={best_hdb_score:.2f}")

# === Добавляем лучшие кластеры в таблицу ===
merged_df['Optim_GMM'] = best_gmm_labels
merged_df['Optim_HDBSCAN'] = best_hdb_labels

# === Визуализация гетерогенности ===
def plot_param_heterogeneity(cluster_col, filename):
    param_std = merged_df.groupby(cluster_col)[param_cols_norm].std()
    param_std['mean_std'] = param_std.mean(axis=1)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=param_std.index, y=param_std['mean_std'], palette="viridis")
    plt.title(f"Parameter Heterogeneity in Optimized Clusters ({cluster_col})")
    plt.xlabel("Cluster")
    plt.ylabel("Mean Std of Normalized Parameters")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

plot_param_heterogeneity("Optim_GMM", "heterogeneity_optimized_gmm.png")
plot_param_heterogeneity("Optim_HDBSCAN", "heterogeneity_optimized_hdbscan.png")

# === Визуализация самих кластеров в UMAP-пространстве ===
reducer = umap.UMAP(random_state=42)
umap_coords = reducer.fit_transform(X_scaled)
merged_df['UMAP1'] = umap_coords[:, 0]
merged_df['UMAP2'] = umap_coords[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(data=merged_df, x='UMAP1', y='UMAP2', hue='Optim_GMM', palette='tab10', s=10, ax=axes[0])
axes[0].set_title("GMM-кластеры (UMAP проекция)")
sns.scatterplot(data=merged_df, x='UMAP1', y='UMAP2', hue='Optim_HDBSCAN', palette='tab20', s=10, ax=axes[1])
axes[1].set_title("HDBSCAN-кластеры (UMAP проекция)")
for ax in axes:
    ax.legend(title="Кластер", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("cluster_visualization_umap.png", dpi=300)
plt.show()

# === Сохраняем результат ===
merged_df.to_csv("clusters_reclustered_optimized.csv", index=False)
print("✓ Результаты сохранены в clusters_reclustered_optimized.csv")
