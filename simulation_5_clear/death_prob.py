import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

# === Загрузка данных ===
params_df = pd.read_csv("parameters_2.csv")
clusters_df = pd.read_csv("clusters.csv")

# Объединяем по sim_id
df = pd.merge(
    params_df,
    clusters_df,
    left_on="number of simulation",
    right_on="sim_id"
)

# === Выбор колонок ===
cluster_column = "HDBSCAN_cluster"  # можно заменить на "GMM_cluster"
param_cols = [col for col in params_df.columns if col != "number of simulation"]

# === Словарь с максимумами параметров ===
param_max = {
    'REFILL_INTERVAL_A': 1000, 'REFILL_INTERVAL_B': 1000, 'REFILL_INTERVAL_C': 1000,
    'DELTA_T': 1000, 'MEAL_TOTAL_A': 1000, 'MEAL_TOTAL_B': 1000, 'MEAL_TOTAL_C': 1000,
    'INITIAL_NUTRIENT': 1000, 'DIFFUSION_RATE': 1, 'INITIAL_ENERGY': 1000, 'CHANNEL_COST': 1,
    'DIVISION_THRESHOLD': 1000, 'BASE_METABOLIC_COST': 1, 'MAINTENANCE_COST': 1,
    'UPTAKE_PER_CHANNEL': 1000, 'BUILD_PROB_SLOPE': 1, 'SUPPRESSION_K': 1000,
    'DEATH_PROB': 1, 'DIVISION_SWITCH_PROB': 1, 'DIVISION_BIAS_MEAN': 1, 'DIVISION_BIAS_SD': 1,
    'DIVISION_BIAS_MIN': 1, 'DIVISION_BIAS_MAX': 1, 'PHAGE_A_DIFFUSION_RATE': 1,
    'PHAGE_A_ADSORPTION_RATE': 1, 'PHAGE_A_BURST_SIZE': 1000, 'PHAGE_A_DECAY_RATE': 1,
    'PHAGE_A_LATENT_PERIOD': 1000, 'INITIAL_PHAGE_A_CONCENTRATION': 1,
    'PHAGE_B_DIFFUSION_RATE': 1, 'PHAGE_B_ADSORPTION_RATE': 1, 'PHAGE_B_BURST_SIZE': 1000,
    'PHAGE_B_DECAY_RATE': 1, 'PHAGE_B_LATENT_PERIOD': 1000, 'INITIAL_PHAGE_B_CONCENTRATION': 1,
    'PHAGE_C_DIFFUSION_RATE': 1, 'PHAGE_C_ADSORPTION_RATE': 1, 'PHAGE_C_BURST_SIZE': 1000,
    'PHAGE_C_DECAY_RATE': 1, 'PHAGE_C_LATENT_PERIOD': 1000, 'INITIAL_PHAGE_C_CONCENTRATION': 1,
    'MAJOR_MEAL_FRACTION': 1
}

# === Нормализация параметров ===
for col in param_cols:
    if col in param_max:
        df[col] = df[col] / param_max[col]

# === Группировка по кластерам и расчёт std ===
cluster_ids = df[cluster_column].unique()

def compute_param_std(param):
    stds = []
    for cl in cluster_ids:
        values = df[df[cluster_column] == cl][param]
        if len(values) > 1:
            stds.append(np.std(values))
    return param, np.mean(stds)

results = Parallel(n_jobs=4)(
    delayed(compute_param_std)(param)
    for param in tqdm(param_cols, desc="Вычисление std по параметрам")
)

# === График ===
param_names = [r[0] for r in results]
std_means = [r[1] for r in results]

# === График с палитрой viridis ===
plt.figure(figsize=(14, 6))
sns.barplot(x=param_names, y=std_means, palette='viridis')
plt.xticks(rotation=90)
plt.ylabel("Mean standard deviation")
plt.title(f"Heterogeneity of parameters within clusters ({cluster_column})")
plt.tight_layout()
plt.savefig(f"cluster_std_by_param_{cluster_column}.png", dpi=300)
plt.show()








'''import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm
from joblib import Parallel, delayed

# === Папка для вывода ===
output_dir = "param_metric_dependence_parallel_pval"
os.makedirs(output_dir, exist_ok=True)

# === Загрузка данных ===
params_df = pd.read_csv("parameters_2.csv")
metrics_df = pd.read_csv("phase_metrics.csv")

# Удаляем rep_id и усредняем по sim_id
metrics_df = metrics_df.drop(columns=['rep_id'], errors='ignore')
metrics_df = metrics_df.groupby('sim_id', as_index=False).mean(numeric_only=True)

# Объединяем по sim_id <-> number of simulation
df = pd.merge(params_df, metrics_df, left_on='number of simulation', right_on='sim_id')

# === Метрики и параметры ===
metric_cols = [
    'phase_switch_count', 'dwell_time_entropy', 'spectral_entropy',
    'dominance_ratio', 'avg_bacteria', 'avg_phage', 'survival_time'
]
param_cols = [col for col in params_df.columns if col != 'number of simulation']

# === Функция для одной пары
def process_pair(param, metric):
    rho, pval = spearmanr(df[param], df[metric], nan_policy='omit')
    if abs(rho) >= 0.3 and pval < 0.05:
        summary = df.groupby(param)[metric].agg(['mean', 'std']).reset_index()

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=summary, x=param, y='mean', marker='o', label=f'ρ={rho:.2f}, p={pval:.2e}')
        plt.fill_between(summary[param],
                         summary['mean'] - summary['std'],
                         summary['mean'] + summary['std'],
                         alpha=0.2, label='std')
        plt.xlabel(param)
        plt.ylabel(metric)
        plt.title(f"{metric} vs {param}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = f"{metric}_vs_{param}.png".replace('/', '_')
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

# === Параллельная обработка всех пар
tasks = [(param, metric) for param in param_cols for metric in metric_cols]
_ = Parallel(n_jobs=4)(
    delayed(process_pair)(p, m) for p, m in tqdm(tasks, desc="Проверка всех пар")
)

# === HEATMAP корреляций параметр ↔ метрика
corr_matrix = pd.DataFrame(index=param_cols, columns=metric_cols)

for param in param_cols:
    for metric in metric_cols:
        rho, _ = spearmanr(df[param], df[metric], nan_policy='omit')
        corr_matrix.loc[param, metric] = rho

corr_matrix = corr_matrix.astype(float)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Spearman Correlation: Parameters vs Metrics")
plt.xlabel("Metrics")
plt.ylabel("Parameters")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
plt.close()
'''