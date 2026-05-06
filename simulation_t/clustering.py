import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import hdbscan
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Загрузка данных и усреднение по sim_id
df = pd.read_csv('phase_metrics.csv').fillna(0)
# Группируем по sim_id и считаем среднее по остальным колонкам
df = df.groupby('sim_id', as_index=False).mean(numeric_only=True)
# Убираем колонку 'id' перед масштабированием
X = df.drop(columns=['sim_id'])#убрал ,'rep_id'
X_numeric = X.apply(pd.to_numeric, errors='coerce')  # строки станут NaN
# Заменим inf → nan, потом nan → 0
X_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
# Убрать слишком большие значения (например, > 1e100)
X_numeric = X_numeric.mask(X_numeric.abs() > 1e100, np.nan)
# Всё, что осталось плохое — заменим на 0
X_clean = X_numeric.fillna(0)

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# 3A. Кластеризация GMM
gmm = GaussianMixture(n_components=6, random_state=42)
df['GMM_cluster'] = gmm.fit_predict(X_scaled)

# 3B. Кластеризация HDBSCAN
hdb = hdbscan.HDBSCAN(min_cluster_size=10)
df['HDBSCAN_cluster'] = hdb.fit_predict(X_scaled)

# 4. Сохранение в CSV
df.to_csv('clusters.csv', index=False)

df = pd.read_csv('clusters.csv').fillna(0)
# 5. Удаляем колонку id и колонку с кластерами (кластер не участвует в UMAP)
features = df.drop(columns=['sim_id', 'HDBSCAN_cluster', 'GMM_cluster'], errors='ignore') #убрал ,'rep_id'
# Преобразуем все значения в числовые (строки вроде 'inf', 'NaN' → np.nan)
features = features.apply(pd.to_numeric, errors='coerce')

# Заменяем +inf и -inf на NaN
features.replace([np.inf, -np.inf], np.nan, inplace=True)

# Убираем слишком большие значения (например, > 1e100) — заменяем их на NaN
features = features.mask(features.abs() > 1e100, np.nan)

# Заменяем оставшиеся NaN на 0
features = features.fillna(0)

# 6. Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# 4. UMAP-проекция
reducer = umap.UMAP(random_state=42)
print(X_scaled)
embedding = reducer.fit_transform(X_scaled)
df['UMAP1'] = embedding[:, 0]
df['UMAP2'] = embedding[:, 1]

# Создаём фигуру с 2 подграфиками (горизонтально)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# График HDBSCAN
sns.scatterplot(
    data=df,
    x='UMAP1',
    y='UMAP2',
    hue='HDBSCAN_cluster',
    palette='tab10',
    s=10,
    alpha=0.6,
    ax=axes[0]
)
axes[0].set_title('HDBSCAN-кластеры')
axes[0].legend(title='Кластер', bbox_to_anchor=(1.05, 1), loc='upper left')

# График GMM
sns.scatterplot(
    data=df,
    x='UMAP1',
    y='UMAP2',
    hue='GMM_cluster',
    palette='Set2',
    s=10,
    alpha=0.6,
    ax=axes[1]
)
axes[1].set_title('GMM-кластеры')
axes[1].legend(title='Кластер', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('clusters_comparison_umap.png', dpi=300)
plt.show()