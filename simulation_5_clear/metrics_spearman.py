import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.signal import welch
import seaborn as sns
import matplotlib.pyplot as plt

# Папка с результатами
TIMESERIES_DIR = "timeseries"

# Метрики по одному повтору (одному .csv)
def compute_metrics_for_file(path):
    df = pd.read_csv(path)

    b_cols = ['majA', 'majB', 'majC']

    if not all(col in df.columns for col in b_cols + ['bacteria', 'phage', 'energy']):
        raise ValueError(f"Missing expected columns in {path}")

    # ---------------------
    # phase_switch_count
    # ---------------------
    leaders = df[b_cols].idxmax(axis=1)
    phase_switch_count = (leaders != leaders.shift()).sum() - 1

    # ---------------------
    # dwell_time_entropy
    # ---------------------
    dwell = (leaders != leaders.shift()).cumsum()
    dwell_times = leaders.groupby(dwell).size().values
    dwell_p = dwell_times / dwell_times.sum()
    dwell_time_entropy = entropy(dwell_p)

    # ---------------------
    # spectral_entropy
    # ---------------------
    total_bacteria = df['bacteria'].values
    f, Pxx = welch(total_bacteria, nperseg=min(256, len(total_bacteria)))
    Pxx /= np.sum(Pxx)
    spectral_entropy = entropy(Pxx)

    # ---------------------
    # dominance_ratio
    # ---------------------
    top = df[b_cols].max(axis=1)
    rest = df[b_cols].sum(axis=1) - top
    dominance_ratio = (top - rest / 2).mean()

    # ---------------------
    # avg_bacteria, avg_phage
    # ---------------------
    avg_bacteria = df['bacteria'].mean()
    avg_phage = df['phage'].mean()

    # ---------------------
    # survival_time
    # ---------------------
    survival_time = len(df)

    # ---------------------
    # avg_cell_energy
    # ---------------------
    bacteria_nonzero = df['bacteria'].replace(0, np.nan)
    avg_cell_energy = (df['energy'] / bacteria_nonzero).mean()

    return {
        "phase_switch_count": phase_switch_count,
        "dwell_time_entropy": dwell_time_entropy,
        "spectral_entropy": spectral_entropy,
        "dominance_ratio": dominance_ratio,
        "avg_bacteria": avg_bacteria,
        "avg_phage": avg_phage,
        "survival_time": survival_time,
        "avg_cell_energy": avg_cell_energy
    }

# Собираем все повторения по симуляциям
from collections import defaultdict

def collect_metrics():
    all_metrics = defaultdict(list)

    for fname in os.listdir(TIMESERIES_DIR):
        if not fname.endswith(".csv") or not fname.startswith("sim_"):
            continue

        sim_id = int(fname.split("_")[1])
        path = os.path.join(TIMESERIES_DIR, fname)

        try:
            metrics = compute_metrics_for_file(path)
            for k, v in metrics.items():
                all_metrics[(sim_id, k)].append(v)
        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")

    rows = []
    for sim_id in sorted(set(k[0] for k in all_metrics)):
        row = {"sim_id": sim_id}
        for metric_name in [k[1] for k in all_metrics if k[0] == sim_id]:
            values = all_metrics[(sim_id, metric_name)]
            row[metric_name] = np.mean(values)
        rows.append(row)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df_summary = collect_metrics()
    df_summary.to_csv("phase_metrics.csv", index=False)
    print("✓ phase_metrics.csv saved.")

    # Spearman correlation heatmap
    metric_cols = [
        "phase_switch_count", "dwell_time_entropy", "spectral_entropy",
        "dominance_ratio", "avg_bacteria", "avg_phage",
        "survival_time", "avg_cell_energy"
    ]
    corr_df = df_summary[metric_cols].corr(method="spearman")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Spearman Correlation Between Behavioral Metrics")
    plt.tight_layout()
    plt.savefig("metric_correlations.png", dpi=300)
    plt.show()
