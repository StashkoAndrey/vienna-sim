# analysis_p.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# ──────────────────────────────────────────────────────────────────────────────
# 1) LOAD PICKLED RESULTS
# ──────────────────────────────────────────────────────────────────────────────
#
# This assumes that your `main_p.py` (or equivalent) wrote out
# a file named "simulation_results.pkl" containing at least these four keys:
#
#   all_index         : list (REPLICATES) of dicts { t : [ raw_index values … ] }
#   all_deviation     : list (REPLICATES) of dicts { t : [ deviation = |0.5-index| … ] }
#   all_index_grid    : list (REPLICATES) of dicts { t : 2D numpy‐array of raw_index (NaN if empty) }
#   all_deviation_grid: list (REPLICATES) of dicts { t : 2D numpy‐array of deviation (NaN if empty) }
#
# (These keys were written exactly this way by your `main_s3.py`‐style code :contentReference[oaicite:0]{index=0}.)
with open('simulation_results_p.pkl', 'rb') as f:
    data = pickle.load(f)

all_index          = data['all_index']          # :contentReference[oaicite:1]{index=1}
all_deviation      = data['all_deviation']      # :contentReference[oaicite:2]{index=2}
all_index_grid     = data['all_index_grid']     # :contentReference[oaicite:3]{index=3}
all_deviation_grid = data['all_deviation_grid'] # :contentReference[oaicite:4]{index=4}

# ──────────────────────────────────────────────────────────────────────────────
# 2) PARAMETERS (infer T_max from the union of all timesteps in all_index)
# ──────────────────────────────────────────────────────────────────────────────
R = len(all_index)  # number of replicates

# Determine the maximum timestep over all replicates:
all_timesteps = set()
for r in range(R):
    all_timesteps.update(all_index[r].keys())
all_timesteps = sorted(all_timesteps)
T_max = max(all_timesteps)

# For the heatmap, choose 50 bins in [0, 1] for raw‐index
n_bins     = 50
bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# We'll also need “GRID_SIZE” to build our 2D‐grid animation axes.
# We can infer it from shape of one of the grid arrays in all_index_grid.
example_grid = all_index_grid[0][all_timesteps[0]]
GRID_SIZE     = example_grid.shape[0]

# ──────────────────────────────────────────────────────────────────────────────
# 3) ANIMATION: 2D Grid of Raw‐Index for a Single Replicate
# ──────────────────────────────────────────────────────────────────────────────
#
# This is a direct copy of your S3 code’s “index‐grid animation” :contentReference[oaicite:5]{index=5}.

rep_id = 0
index_grid_series = all_index_grid[rep_id]  # dict { t : 2D np.ndarray of shape=(GRID_SIZE,GRID_SIZE) }

timesteps = sorted(index_grid_series.keys())
N_frames  = len(timesteps)

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(
    index_grid_series[timesteps[0]],
    vmin=0.0, vmax=1.0,
    cmap='coolwarm',
    interpolation='nearest'
)
# Paint NaN (empty cells) as green:
im.cmap.set_bad(color='green')

ax.set_title(f"Replicate #{rep_id}, Step {timesteps[0]}")
ax.axis('off')

cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label('Raw Index = channels_A/(channels_A+channels_B)')

def update_frame(frame_idx):
    t = timesteps[frame_idx]
    im.set_data(index_grid_series[t])
    ax.set_title(f"Replicate #{rep_id}, Step {t}")
    return [im]

anim = animation.FuncAnimation(
    fig,
    update_frame,
    frames=N_frames,
    interval=200,   # 200 ms per frame → 5 FPS
    blit=True,
    repeat=False
)

# Try to save as MP4 (requires ffmpeg). Fallback to GIF if ffmpeg not available.

gif_writer = PillowWriter(fps=5)
anim.save('index_grid_animation.gif', writer=gif_writer)
print("saved → index_grid_animation_p.gif")

plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 4) HEATMAP: Distribution of Raw Index Over Time (All Replicates)
# ──────────────────────────────────────────────────────────────────────────────
#
# We want a 2D array H of shape (n_bins, T_max+1) such that:
#   H[b, t] = number of cells (across all replicates) whose raw_index ∈ bin b at time t.

# Initialize an integer array of zeros
heatmap_counts = np.zeros((n_bins, T_max + 1), dtype=int)

# For each replicate r, for each timestep t, bin every index_list:
for r in range(R):
    idx_series = all_index[r]  # dict { t : [ raw_index values … ] }
    for t, idx_list in idx_series.items():
        if t > T_max:
            continue
        counts, _ = np.histogram(idx_list, bins=bin_edges)
        heatmap_counts[:, t] += counts

plt.figure(figsize=(12, 6))
sns.heatmap(
    heatmap_counts,
    cmap='viridis',
    cbar_kws={'label': 'Cell Count'},
    xticklabels=max(1, heatmap_counts.shape[1] // 20),
    yticklabels=np.round(bin_centers, 2)
)
plt.xlabel('Timestep')
plt.ylabel('Raw Index bins')
plt.title('Distribution of Raw Index = channels_A/(A+B) over Time (All Replicates)')
plt.tight_layout()
plt.savefig('index_distribution_heatmap.png')
print("Saved heatmap → index_distribution_heatmap.png")
plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# 5) CURVE PLOT: Mean ± 1 SD of Raw Index over Time (All Replicates)
# ──────────────────────────────────────────────────────────────────────────────
#
# Build an array `mean_index_matrix` of shape (R, T_max+1) where each entry is
# the mean raw_index for replicate r at time t (or NaN if no cells / no data).

mean_index_matrix = np.full((R, T_max + 1), np.nan, dtype=float)

for r in range(R):
    idx_series = all_index[r]
    for t, idx_list in idx_series.items():
        if len(idx_list) > 0:
            mean_index_matrix[r, t] = np.mean(idx_list)
        else:
            mean_index_matrix[r, t] = np.nan

# Compute Mean & SD across replicates (ignoring NaNs):
mean_over_reps = np.nanmean(mean_index_matrix, axis=0)
std_over_reps  = np.nanstd(mean_index_matrix, axis=0)

timesteps_arr = np.arange(T_max + 1)
plt.figure(figsize=(10, 5))
plt.plot(timesteps_arr, mean_over_reps, label='Mean raw Index (all cells)', color='tab:blue')
plt.fill_between(
    timesteps_arr,
    mean_over_reps - std_over_reps,
    mean_over_reps + std_over_reps,
    color='tab:blue', alpha=0.3, label='± 1 SD'
)
plt.xlabel('Timestep')
plt.ylabel('Mean raw Index ± SD')
plt.title('Mean ± 1 SD of Raw Index over Time (All Replicates)')
plt.legend()
plt.tight_layout()
plt.savefig('mean_index_curve.png')
print("Saved curve plot → mean_index_curve.png")
plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# 6) CURVE PLOT: Mean ± 1 SD of “Lean from Ideal” (|0.5 − index|) over Time
# ──────────────────────────────────────────────────────────────────────────────
#
# Build an array `mean_dev_matrix` of shape (R, T_max+1) where each entry is
# the mean deviation = |0.5 − raw_index| for replicate r at time t (or NaN if none).

mean_dev_matrix = np.full((R, T_max + 1), np.nan, dtype=float)

for r in range(R):
    dev_series = all_deviation[r]  # dict { t : [ deviation values … ] }
    for t, dev_list in dev_series.items():
        if len(dev_list) > 0:
            mean_dev_matrix[r, t] = np.mean(dev_list)
        else:
            mean_dev_matrix[r, t] = np.nan

# Compute Mean & SD across replicates (ignoring NaNs):
mean_dev_over_reps = np.nanmean(mean_dev_matrix, axis=0)
std_dev_over_reps  = np.nanstd(mean_dev_matrix, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(
    timesteps_arr,
    mean_dev_over_reps,
    label='Mean |0.5 − index|',
    color='tab:red'
)
plt.fill_between(
    timesteps_arr,
    mean_dev_over_reps - std_dev_over_reps,
    mean_dev_over_reps + std_dev_over_reps,
    color='tab:red', alpha=0.3, label='± 1 SD'
)
plt.xlabel('Timestep')
plt.ylabel('Mean |ideal − raw_index| ± SD')
plt.title('Mean ± 1 SD of “Lean from Ideal” over Time (All Replicates)')
plt.legend()
plt.tight_layout()
plt.savefig('mean_deviation_curve.png')
print("Saved curve plot → mean_deviation_curve.png")
plt.close()
