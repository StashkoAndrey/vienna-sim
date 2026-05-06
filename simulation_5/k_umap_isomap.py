import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from umap import UMAP
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ───────────── Load phase metrics ─────────────
df = pd.read_csv("phase_metrics.csv")

metric_cols = [
    "phase_switch_count", "dwell_time_entropy", "spectral_entropy",
    "dominance_ratio", "avg_bacteria", "avg_phage",
    "survival_time", "zero_crossings"
]

# Очистка данных от бесконечностей и NaN
df_clean = df[metric_cols].replace([float("inf"), -float("inf")], pd.NA).dropna()
X_scaled = StandardScaler().fit_transform(df_clean)

# ───────────── UMAP projection ─────────────
umap_embed = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X_scaled)
df.loc[df_clean.index, "umap_x"] = umap_embed[:, 0]
df.loc[df_clean.index, "umap_y"] = umap_embed[:, 1]

# ───────────── Isomap projection ─────────────
isomap_embed = Isomap(n_neighbors=5).fit_transform(X_scaled)
df.loc[df_clean.index, "isomap_x"] = isomap_embed[:, 0]
df.loc[df_clean.index, "isomap_y"] = isomap_embed[:, 1]

# ───────────── Безопасное логарифмирование выбранного параметра ─────────────
# Защита от x < -1 и NaN
df["log_phase_switch_count"] = np.log1p(
    df["phase_switch_count"].clip(lower=-0.9999).fillna(0)
)

# ───────────── Визуализация проекций ─────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Настройка шкалы цвета
norm = mcolors.Normalize(
    vmin=df["log_phase_switch_count"].min(), 
    vmax=df["log_phase_switch_count"].max()
)
sm = cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])

# ── UMAP ──
sns.scatterplot(
    x="umap_x", y="umap_y",
    hue="log_phase_switch_count",
    palette="viridis",
    data=df,
    ax=axes[0],
    legend=False
)
axes[0].set_title("UMAP Projection (log-scaled phase_switch_count)")
fig.colorbar(sm, ax=axes[0], label="log(phase_switch_count)")

# ── ISOMAP ──
sns.scatterplot(
    x="isomap_x", y="isomap_y",
    hue="log_phase_switch_count",
    palette="viridis",
    data=df,
    ax=axes[1],
    legend=False
)
axes[1].set_title("Isomap Projection (log-scaled phase_switch_count)")
fig.colorbar(sm, ax=axes[1], label="log(phase_switch_count)")

# ───────────── Save and show ─────────────
plt.tight_layout()
plt.savefig("umap_isomap_projection_log_phase_switch_count.png")
plt.show()