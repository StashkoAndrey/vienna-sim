import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.signal import welch
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# ───── CONFIG ─────
TIMESERIES_DIR = "timeseries"

# ───── Per-file metric computation ─────
def compute_metrics_for_file(path):
    try:
        df = pd.read_csv(path)
        b_cols = ['majA', 'majB', 'majC']
        if not all(col in df.columns for col in b_cols + ['bacteria', 'phage', 'energy']):
            raise ValueError("Missing expected columns")

        # Phase switching
        leaders = df[b_cols].idxmax(axis=1)
        phase_switch_count = (leaders != leaders.shift()).sum() - 1

        # Dwell time entropy
        dwell = (leaders != leaders.shift()).cumsum()
        dwell_times = leaders.groupby(dwell).size().values
        dwell_p = dwell_times / dwell_times.sum()
        dwell_time_entropy = entropy(dwell_p)

        # Spectral entropy
        total_bacteria = df['bacteria'].values
        f, Pxx = welch(total_bacteria, nperseg=min(256, len(total_bacteria)))
        total_power = np.sum(Pxx)
        spectral_entropy = entropy(Pxx / total_power) if total_power > 0 else 0

        # Dominance ratio
        top = df[b_cols].max(axis=1)
        rest = df[b_cols].sum(axis=1) - top
        dominance_ratio = (top - rest / 2).mean()

        # Averages
        avg_bacteria = df['bacteria'].mean()
        avg_phage = df['phage'].mean()
        survival_time = len(df)

        # Average cell energy
        bacteria_nonzero = df['bacteria'].replace(0, np.nan)
        avg_cell_energy = (df['energy'] / bacteria_nonzero).mean()

        sim_id = int(os.path.basename(path).split("_")[1])

        return sim_id, {
            "phase_switch_count": phase_switch_count,
            "dwell_time_entropy": dwell_time_entropy,
            "spectral_entropy": spectral_entropy,
            "dominance_ratio": dominance_ratio,
            "avg_bacteria": avg_bacteria,
            "avg_phage": avg_phage,
            "survival_time": survival_time,
            "avg_cell_energy": avg_cell_energy
        }

    except Exception as e:
        print(f"⚠️ Error processing {path}: {e}")
        return None

# ───── Main metric collection ─────
def collect_metrics():
    paths = [
        os.path.join(TIMESERIES_DIR, f)
        for f in os.listdir(TIMESERIES_DIR)
        if f.endswith(".csv") and f.startswith("sim_")
    ]

    all_results = defaultdict(list)

    for path in tqdm(paths, desc="Processing files"):
        result = compute_metrics_for_file(path)
        if result is None:
            continue
        sim_id, metrics = result
        for k, v in metrics.items():
            all_results[(sim_id, k)].append(v)

    rows = []
    for sim_id in sorted(set(k[0] for k in all_results)):
        row = {"sim_id": sim_id}
        for metric_name in [k[1] for k in all_results if k[0] == sim_id]:
            values = all_results[(sim_id, metric_name)]
            row[metric_name] = np.mean(values)
        rows.append(row)

    return pd.DataFrame(rows)

# ───── Execution ─────
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
