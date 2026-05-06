# ======= extract_phase_metrics.py =======
import pandas as pd
import numpy as np
import os
from scipy.signal import periodogram
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

# ───────────── Compute 8 behavioral metrics ─────────────
def compute_phase_metrics(bacteria, phage, energy):
    state = (phage > bacteria).astype(int)
    switch_count = np.count_nonzero(np.diff(state) != 0)
    dominance_ratio = np.mean(bacteria > phage)

    dwell_lengths = []
    curr = state[0]
    count = 1
    for s in state[1:]:
        if s == curr:
            count += 1
        else:
            dwell_lengths.append(count)
            curr = s
            count = 1
    dwell_lengths.append(count)
    dwell_entropy = entropy(np.unique(dwell_lengths, return_counts=True)[1])

    f, Pxx = periodogram(bacteria)
    Pxx_norm = Pxx / Pxx.sum() if Pxx.sum() > 0 else Pxx
    spectral_ent = entropy(Pxx_norm)

    return {
        "phase_switch_count": switch_count,
        "dwell_time_entropy": dwell_entropy,
        "spectral_entropy": spectral_ent,
        "dominance_ratio": dominance_ratio,
        "avg_bacteria": np.mean(bacteria),
        "avg_phage": np.mean(phage),
        "survival_time": len(bacteria),
        "zero_crossings": np.count_nonzero(np.diff(np.sign(bacteria - phage)))
    }

# ───────────── Loop through all files ─────────────
def extract_all_metrics(timeseries_dir="timeseries"):
    results = []
    for file in sorted(os.listdir(timeseries_dir)):
        if file.endswith(".csv") and file.startswith("sim_"):
            parts = file.replace(".csv", "").split("_")
            sim_id = int(parts[1])
            rep_id = int(parts[3])

            df = pd.read_csv(os.path.join(timeseries_dir, file))
            metrics = compute_phase_metrics(df["bacteria"].values,
                                            df["phage"].values,
                                            df["energy"].values)
            metrics["sim_id"] = sim_id
            metrics["rep_id"] = rep_id
            results.append(metrics)
    return pd.DataFrame(results)

# ───────────── Execute and visualize ─────────────
df_metrics = extract_all_metrics("timeseries")
df_metrics.to_csv("phase_metrics.csv", index=False)

metric_cols = [
    "phase_switch_count", "dwell_time_entropy", "spectral_entropy",
    "dominance_ratio", "avg_bacteria", "avg_phage",
    "survival_time", "zero_crossings"
]

corr_df = df_metrics[metric_cols].corr(method="spearman")
plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Spearman Correlation Between Behavioral Metrics")
plt.tight_layout()
plt.savefig("metric_correlations.png")
plt.show()
