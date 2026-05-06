# analysis_p.py
#
# In-depth post-processing for three-channel simulations.
# Generates:  IA heat-map, mean ± SD curves for IA, |0.5-IA|, entropy.
# ──────────────────────────────────────────────────────────────────
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PKL = 'simulation_results.pkl'
OUT_PREFIX = ''

# ─────────────────────────── load & massage ──────────────────────────
with open(PKL, 'rb') as f:
    data = pickle.load(f)

index_runs   = data['all_index']      # list(R) of dict{ t : [IA] }
dev_runs     = data['all_deviation']  # list(R) of dict{ t : [|0.5-IA|] }
entropy_runs = data['all_entropy']    # list(R) of dict{ t : [H] }

R           = len(index_runs)
T_max       = max(max(d) for d in index_runs)  # latest time reached by any run

def dicts_to_matrix(list_of_dicts, func=np.nanmean):
    """Return (R × T_max+1) matrix with per-timestep aggregate via *func*."""
    M = np.full((R, T_max + 1), np.nan)
    for r, dic in enumerate(list_of_dicts):
        for t, values in dic.items():
            if values:
                M[r, t] = func(values)
    return M

IA_mat   = dicts_to_matrix(index_runs,   np.nanmean)
DEV_mat  = dicts_to_matrix(dev_runs,     np.nanmean)
H_mat    = dicts_to_matrix(entropy_runs, np.nanmean)

# ─────────────────────────── 1) IA heat-map ─────────────────────────
# Bin IA into uniform bins for all replicates pooled
n_bins = 50
bins   = np.linspace(0, 1, n_bins + 1)
heat   = np.zeros((n_bins, T_max + 1), dtype=int)

for dic in index_runs:
    for t, vals in dic.items():
        if vals:
            counts, _ = np.histogram(vals, bins=bins)
            heat[:, t] += counts

yticks = np.round((bins[:-1] + bins[1:]) / 2, 2)

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(heat, cmap='viridis', ax=ax,
            cbar_kws={'label': 'cell count'},
            xticklabels=max(1, (T_max + 1) // 20),
            yticklabels=yticks)
ax.set(title='IA = A / (A + B + C)  distribution over time (all replicates)',
       xlabel='timestep', ylabel='IA bin')
fig.tight_layout()
fig.savefig(f'{OUT_PREFIX}IA_heatmap.png')
plt.close(fig)
print('✓ IA_heatmap.png')

# ─────────────────────────── 2) mean ± SD curves ────────────────────
def plot_mean_sd(matrix, label, color, filename):
    mean = np.nanmean(matrix, axis=0)
    sd   = np.nanstd(matrix,  axis=0)
    t    = np.arange(T_max + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, mean, color=color, label=label)
    ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=.3, label='±1 SD')
    ax.set(xlabel='timestep', ylabel=label, title=label + ' over time')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}{filename}')
    plt.close(fig)
    print(f'✓ {filename}')

plot_mean_sd(IA_mat,  'Mean IA',              'tab:blue',   'IA_mean.png')
plot_mean_sd(DEV_mat, 'Mean |0.5 – IA|',      'tab:green',  'deviation_mean.png')
plot_mean_sd(H_mat,   'Mean Shannon entropy', 'tab:orange', 'entropy_mean.png')
