import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt

"""# 1. Загрузка таблиц
params_df = pd.read_csv("parameters_2.csv")       # примерно 40 параметров и number_of_simulation
clusters_df = pd.read_csv("clusters.csv")       # number_of_simulation и GMM_cluster или HDBSCAN_cluster

df = pd.merge(
    params_df,         
    clusters_df,       
    left_on='number of simulation',
    right_on='sim_id'
)


# Выбираем только параметры (все, кроме служебных колонок)
param_cols = [
    'REFILL_INTERVAL_A','REFILL_INTERVAL_B','REFILL_INTERVAL_C','DELTA_T',
    'MEAL_TOTAL_A','MEAL_TOTAL_B','MEAL_TOTAL_C','INITIAL_NUTRIENT','DIFFUSION_RATE',
    'INITIAL_ENERGY','CHANNEL_COST','DIVISION_THRESHOLD','BASE_METABOLIC_COST',
    'MAINTENANCE_COST','UPTAKE_PER_CHANNEL','BUILD_PROB_SLOPE','SUPPRESSION_K',
    'DEATH_PROB','DIVISION_SWITCH_PROB','DIVISION_BIAS_MEAN','DIVISION_BIAS_SD',
    'DIVISION_BIAS_MIN','DIVISION_BIAS_MAX','PHAGE_A_DIFFUSION_RATE',
    'PHAGE_A_ADSORPTION_RATE','PHAGE_A_BURST_SIZE','PHAGE_A_DECAY_RATE',
    'PHAGE_A_LATENT_PERIOD','INITIAL_PHAGE_A_CONCENTRATION','PHAGE_B_DIFFUSION_RATE',
    'PHAGE_B_ADSORPTION_RATE','PHAGE_B_BURST_SIZE','PHAGE_B_DECAY_RATE',
    'PHAGE_B_LATENT_PERIOD','INITIAL_PHAGE_B_CONCENTRATION','PHAGE_C_DIFFUSION_RATE',
    'PHAGE_C_ADSORPTION_RATE','PHAGE_C_BURST_SIZE','PHAGE_C_DECAY_RATE',
    'PHAGE_C_LATENT_PERIOD','INITIAL_PHAGE_C_CONCENTRATION','MAJOR_MEAL_FRACTION'
]
scaler = StandardScaler()
params_scaled = scaler.fit_transform(df[param_cols])

# Объединяем с кластерной меткой
df_scaled = pd.DataFrame(params_scaled, columns=param_cols)
df_scaled['GMM_cluster'] = df['GMM_cluster'].values

# Расчёт внутрикластерной однородности
cluster_variances = {}
for cluster_id, group in df_scaled.groupby('GMM_cluster'):
    values = group[param_cols].values
    centroid = np.mean(values, axis=0)
    distances = cdist(values, [centroid], metric='euclidean')
    rmsd = np.sqrt(np.mean(distances ** 2))
    cluster_variances[cluster_id] = rmsd

# Преобразуем в DataFrame
variance_df = pd.DataFrame.from_dict(cluster_variances, orient='index', columns=['mean_distance_to_centroid'])
variance_df.index.name = 'GMM_cluster'
variance_df.reset_index(inplace=True)

# Сортировка для графика
variance_df.sort_values('mean_distance_to_centroid', inplace=True)

# Визуализация
plt.figure(figsize=(10, 5))
sns.barplot(data=variance_df, x='GMM_cluster', y='mean_distance_to_centroid', palette='viridis')

plt.title('Однородность кластеров (Среднеквадратичное расстояние до центроида)')
plt.ylabel('Среднеквадратичное расстояние')
plt.xlabel('Кластер')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""





















import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1. Загрузка таблиц
params_df = pd.read_csv("parameters_2.csv")       # 40 параметров и number_of_simulation
clusters_df = pd.read_csv("clusters.csv")       # number_of_simulation и GMM_cluster или HDBSCAN_cluster

merged = pd.merge(
    params_df,         
    clusters_df,       
    left_on='number of simulation',
    right_on='sim_id'
)

# Выбери параметры, которые хочешь визуализировать
param_cols = [
    'REFILL_INTERVAL_A','REFILL_INTERVAL_B','REFILL_INTERVAL_C','DELTA_T',
    'MEAL_TOTAL_A','MEAL_TOTAL_B','MEAL_TOTAL_C','INITIAL_NUTRIENT','DIFFUSION_RATE',
    'INITIAL_ENERGY','CHANNEL_COST','DIVISION_THRESHOLD','BASE_METABOLIC_COST',
    'MAINTENANCE_COST','UPTAKE_PER_CHANNEL','BUILD_PROB_SLOPE','SUPPRESSION_K',
    'DEATH_PROB','DIVISION_SWITCH_PROB','DIVISION_BIAS_MEAN','DIVISION_BIAS_SD',
    'DIVISION_BIAS_MIN','DIVISION_BIAS_MAX','PHAGE_A_DIFFUSION_RATE',
    'PHAGE_A_ADSORPTION_RATE','PHAGE_A_BURST_SIZE','PHAGE_A_DECAY_RATE',
    'PHAGE_A_LATENT_PERIOD','INITIAL_PHAGE_A_CONCENTRATION','PHAGE_B_DIFFUSION_RATE',
    'PHAGE_B_ADSORPTION_RATE','PHAGE_B_BURST_SIZE','PHAGE_B_DECAY_RATE',
    'PHAGE_B_LATENT_PERIOD','INITIAL_PHAGE_B_CONCENTRATION','PHAGE_C_DIFFUSION_RATE',
    'PHAGE_C_ADSORPTION_RATE','PHAGE_C_BURST_SIZE','PHAGE_C_DECAY_RATE',
    'PHAGE_C_LATENT_PERIOD','INITIAL_PHAGE_C_CONCENTRATION','MAJOR_MEAL_FRACTION'
]

# Переводим в "длинный" формат для построения графиков
melted = merged.melt(id_vars='HDBSCAN_cluster', value_vars=param_cols,
                     var_name='parameter', value_name='value')


for i in range(0, len(param_cols), 6):
    subset = param_cols[i:i+6]
    subset_melted = merged.melt(id_vars='HDBSCAN_cluster', value_vars=subset,
                                var_name='parameter', value_name='value')
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=subset_melted, x='parameter', y='value', hue='HDBSCAN_cluster')
    plt.title(f'Параметры {subset}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
