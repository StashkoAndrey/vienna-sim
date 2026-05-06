import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import hdbscan
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

# 1. Загрузка данных и усреднение по sim_id
df = pd.read_csv('phase_metrics.csv').fillna(0)
df = df.groupby('sim_id', as_index=False).mean(numeric_only=True)

# Очистка
X = df.drop(columns=['sim_id'], errors='ignore')
X = X.apply(pd.to_numeric, errors='coerce')
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.mask(X.abs() > 1e100, np.nan).fillna(0)

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2A. Кластеризация GMM (параллельно с выбором наименьшего BIC)
def fit_gmm(seed):
    gmm = GaussianMixture(n_components=10, random_state=seed, n_init=1, max_iter=200)
    gmm.fit(X_scaled)
    return gmm, gmm.bic(X_scaled)

print("⏳ GMM кластеризация...")
results = Parallel(n_jobs=4)(delayed(fit_gmm)(seed) for seed in tqdm(range(4)))
best_gmm, _ = min(results, key=lambda x: x[1])
df['GMM_cluster'] = best_gmm.predict(X_scaled)

# 2B. HDBSCAN (с параллелизацией)
print("⏳ HDBSCAN кластеризация...")
hdb = hdbscan.HDBSCAN(min_cluster_size=500, core_dist_n_jobs=4)
df['HDBSCAN_cluster'] = hdb.fit_predict(X_scaled)

# 3. Сохранение
df.to_csv('clusters.csv', index=False)

# 4. Повторная очистка для UMAP
features = df.drop(columns=['sim_id', 'HDBSCAN_cluster', 'GMM_cluster'], errors='ignore')
features = features.apply(pd.to_numeric, errors='coerce')
features.replace([np.inf, -np.inf], np.nan, inplace=True)
features = features.mask(features.abs() > 1e100, np.nan).fillna(0)
X_scaled = scaler.fit_transform(features)

# 5. UMAP-проекция
print("⏳ UMAP проекция...")
reducer = umap.UMAP(random_state=42, n_jobs=4)
embedding = reducer.fit_transform(X_scaled)
df['UMAP1'] = embedding[:, 0]
df['UMAP2'] = embedding[:, 1]

# 6. Визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.scatterplot(
    data=df, x='UMAP1', y='UMAP2', hue='HDBSCAN_cluster',
    palette='tab10', s=10, alpha=0.6, ax=axes[0]
)
axes[0].set_title('HDBSCAN-кластеры')
axes[0].legend(title='Кластер', bbox_to_anchor=(1.05, 1), loc='upper left')

sns.scatterplot(
    data=df, x='UMAP1', y='UMAP2', hue='GMM_cluster',
    palette='Set2', s=10, alpha=0.6, ax=axes[1]
)
axes[1].set_title('GMM-кластеры')
axes[1].legend(title='Кластер', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('clusters_comparison_umap.png', dpi=300)
plt.show()
