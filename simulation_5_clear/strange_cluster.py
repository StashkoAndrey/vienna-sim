import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
from tqdm import tqdm
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from tqdm import tqdm_gui as tqdm

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

# === Подготовка признаков ===
metric_cols = [
    'phase_switch_count','dwell_time_entropy','spectral_entropy',
    'dominance_ratio','avg_bacteria','avg_phage','survival_time'
]
param_cols = [col for col in params_df.columns if col != 'number of simulation']

# Удаляем ненужные столбцы
exclude_cols = ['sim_id', 'number of simulation', 'GMM_cluster', 'HDBSCAN_cluster']
merged_df = merged_df.drop(columns=[col for col in exclude_cols if col in merged_df.columns])

# Увеличиваем вес метрик дублированием колонок (×5)
metrics_weighted = np.hstack([merged_df[metric_cols].values] * 5)
params = merged_df[[col for col in merged_df.columns if col in param_cols]].values
X_combined = np.hstack([metrics_weighted, params])

# Стандартизация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

print("Запуск кластеризации и визуализации...")
progress = tqdm(total=4, desc="Общий прогресс", position=0, leave=True)

# === Кластеризация GMM (параллельно) ===
def fit_gmm_parallel(X, n_components=10, n_init=4):
    def single_fit(seed):
        gmm = GaussianMixture(n_components=n_components, random_state=seed, n_init=1, max_iter=200)
        gmm.fit(X)
        return gmm, gmm.bic(X)

    results = Parallel(n_jobs=4)(delayed(single_fit)(seed) for seed in tqdm(range(n_init), desc="Fitting GMM models"))
    best_gmm, _ = min(results, key=lambda x: x[1])
    return best_gmm.fit_predict(X)

gmm_labels = fit_gmm_parallel(X_scaled, n_components=10, n_init=4)
progress.update(1)
merged_df['GMM_cluster_new'] = gmm_labels

# === Кластеризация HDBSCAN ===
hdb = hdbscan.HDBSCAN(min_cluster_size=100, core_dist_n_jobs=4)
hdb_labels = hdb.fit_predict(X_scaled)
merged_df['HDBSCAN_cluster_new'] = hdb_labels
progress.update(1)

# === Визуализация кластеров (UMAP) ===
reducer = umap.UMAP(random_state=42, n_jobs=4)
embedding = reducer.fit_transform(X_scaled)
merged_df['UMAP1'] = embedding[:, 0]
merged_df['UMAP2'] = embedding[:, 1]
progress.update(1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(data=merged_df, x='UMAP1', y='UMAP2', hue='GMM_cluster_new', palette='tab10', s=10, ax=axes[0])
axes[0].set_title("GMM (с приоритетом метрик)")
sns.scatterplot(data=merged_df, x='UMAP1', y='UMAP2', hue='HDBSCAN_cluster_new', palette='tab20', s=10, ax=axes[1])
axes[1].set_title("HDBSCAN (с приоритетом метрик)")
for ax in axes:
    ax.legend(title="Кластер", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("cluster_priority_visualization.png", dpi=300)
plt.show()
progress.update(1)
# закрывать прогресс-бар не нужно: оставим для финального отображения


# === Анализ гетерогенности кластеров ===
def compute_cluster_heterogeneity(df, cluster_col, param_cols):
    cluster_std = df.groupby(cluster_col)[param_cols].std()
    cluster_std['mean_std'] = cluster_std.mean(axis=1)
    return cluster_std['mean_std']

hetero_gmm = compute_cluster_heterogeneity(merged_df, 'GMM_cluster_new', param_cols)
hetero_hdb = compute_cluster_heterogeneity(merged_df, 'HDBSCAN_cluster_new', param_cols)

# === Визуализация UMAP + гетерогенность ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

# Верхние: UMAP-кластеры
sns.scatterplot(data=merged_df, x='UMAP1', y='UMAP2', hue='GMM_cluster_new', palette='tab10', s=10, ax=axes[0,0])
axes[0,0].set_title("GMM (UMAP)")
sns.scatterplot(data=merged_df, x='UMAP1', y='UMAP2', hue='HDBSCAN_cluster_new', palette='tab20', s=10, ax=axes[0,1])
axes[0,1].set_title("HDBSCAN (UMAP)")
for ax in axes[0]:
    ax.legend(title="Кластер", bbox_to_anchor=(1.05, 1), loc='upper left')

# Нижние: гетерогенность
sns.barplot(x=hetero_gmm.index, y=hetero_gmm.values, palette='Blues_d', ax=axes[1,0])
axes[1,0].set_title("GMM: Среднее отклонение параметров по кластерам")
axes[1,0].set_xlabel("Кластер")
axes[1,0].set_ylabel("Среднее std")

sns.barplot(x=hetero_hdb.index, y=hetero_hdb.values, palette='Greens_d', ax=axes[1,1])
axes[1,1].set_title("HDBSCAN: Среднее отклонение параметров по кластерам")
axes[1,1].set_xlabel("Кластер")
axes[1,1].set_ylabel("Среднее std")

plt.tight_layout()
plt.savefig("cluster_with_heterogeneity.png", dpi=300)
plt.show()

# === Сохраняем результат ===
merged_df.to_csv("clusters_priority_weighted.csv", index=False)
print("✓ Готово: кластеризация с приоритетом метрик сохранена в clusters_priority_weighted.csv")
