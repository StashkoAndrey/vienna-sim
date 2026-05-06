import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed

# === Максимумы параметров ===
parameter_maxima = {
    'REFILL_INTERVAL_A': 1000, 'REFILL_INTERVAL_B': 1000, 'REFILL_INTERVAL_C': 1000, 'DELTA_T': 1000,
    'MEAL_TOTAL_A': 1000, 'MEAL_TOTAL_B': 1000, 'MEAL_TOTAL_C': 1000, 'INITIAL_NUTRIENT': 1000,
    'DIFFUSION_RATE': 1, 'INITIAL_ENERGY': 1000, 'CHANNEL_COST': 1, 'DIVISION_THRESHOLD': 1000,
    'BASE_METABOLIC_COST': 1, 'MAINTENANCE_COST': 1, 'UPTAKE_PER_CHANNEL': 1000, 'BUILD_PROB_SLOPE': 1,
    'SUPPRESSION_K': 1000, 'DEATH_PROB': 1, 'DIVISION_SWITCH_PROB': 1, 'DIVISION_BIAS_MEAN': 1,
    'DIVISION_BIAS_SD': 1, 'DIVISION_BIAS_MIN': 1, 'DIVISION_BIAS_MAX': 1, 'PHAGE_A_DIFFUSION_RATE': 1,
    'PHAGE_A_ADSORPTION_RATE': 1, 'PHAGE_A_BURST_SIZE': 1000, 'PHAGE_A_DECAY_RATE': 1,
    'PHAGE_A_LATENT_PERIOD': 1000, 'INITIAL_PHAGE_A_CONCENTRATION': 1, 'PHAGE_B_DIFFUSION_RATE': 1,
    'PHAGE_B_ADSORPTION_RATE': 1, 'PHAGE_B_BURST_SIZE': 1000, 'PHAGE_B_DECAY_RATE': 1,
    'PHAGE_B_LATENT_PERIOD': 1000, 'INITIAL_PHAGE_B_CONCENTRATION': 1, 'PHAGE_C_DIFFUSION_RATE': 1,
    'PHAGE_C_ADSORPTION_RATE': 1, 'PHAGE_C_BURST_SIZE': 1000, 'PHAGE_C_DECAY_RATE': 1,
    'PHAGE_C_LATENT_PERIOD': 1000, 'INITIAL_PHAGE_C_CONCENTRATION': 1, 'MAJOR_MEAL_FRACTION': 1
}
param_cols = list(parameter_maxima.keys())

# === Загрузка и объединение данных ===
params_df = pd.read_csv("parameters_2.csv")
clusters_df = pd.read_csv("clusters.csv")
df = pd.merge(params_df, clusters_df, left_on='number of simulation', right_on='sim_id')

# === Нормировка параметров ===
df_normalized = df.copy()
for col in param_cols:
    df_normalized[col] = df_normalized[col] / parameter_maxima[col]

# === Создание выходных папок ===
os.makedirs("heterogeneity_hdbscan", exist_ok=True)
os.makedirs("heterogeneity_gmm", exist_ok=True)

# === Функция построения графика для одного кластера ===
def plot_cluster_std(cluster_id, cluster_df, method):
    stds = cluster_df[param_cols].std()
    good_count = (stds < 0.18).sum()
    label = f"{method.upper()}_cluster_{cluster_id}"
    if good_count >= len(param_cols) // 2:
        label += "_GOOD"

    plt.figure(figsize=(16, 5))
    sns.barplot(x=param_cols, y=stds.values)
    plt.xticks(rotation=90)
    plt.ylabel("Std (normalized)")
    plt.title(f"{label} — {good_count}/{len(param_cols)} параметров с std < 0.18")
    folder = f"heterogeneity_{method}"
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{label}.png"))
    plt.close()

# === Анализ по заданному методу (GMM или HDBSCAN) ===
def analyze_method(method):
    cluster_col = f"{method.upper()}_cluster"
    grouped = df_normalized.groupby(cluster_col)
    valid_clusters = [g for g in grouped if g[0] != -1]

    print(f"🔍 Анализ гетерогенности для {method.upper()}...")
    Parallel(n_jobs=4)(
        delayed(plot_cluster_std)(cluster_id, cluster_df, method)
        for cluster_id, cluster_df in tqdm(valid_clusters, desc=f"{method.upper()} clusters")
    )

# Запуск анализа
analyze_method("hdbscan")
analyze_method("gmm")
print("✅ Готово: гистограммы сохранены в папках heterogeneity_hdbscan и heterogeneity_gmm.")

