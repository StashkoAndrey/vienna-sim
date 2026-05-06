# analysis_s3.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# ──────────────────────────────────────────────────────────────────────────────
# 1) LOAD PICKLED RESULTS
# ──────────────────────────────────────────────────────────────────────────────

with open('simulation_data_with_index.pkl', 'rb') as f:
    data = pickle.load(f)

all_pop_A        = data['all_pop_A']         # list (REPLICATES) of lists (#A cells/time)
all_pop_B        = data['all_pop_B']         # list (REPLICATES) of lists (#B cells/time)
all_chan_Atype   = data['all_chan_Atype']    # list (REPLICATES) of dicts {t: [channels_A…]}
all_chan_Btype   = data['all_chan_Btype']    # list (REPLICATES) of dicts {t: [channels_B…]}
all_energy       = data['all_energy']        # list (REPLICATES) of lists (total energy/time)

all_index        = data['all_index']         # list (REPLICATES) of dicts {t: [raw_index values…]}
all_deviation    = data['all_deviation']     # list (REPLICATES) of dicts {t: [deviation values…]}

# Because we changed grid_s3.py so empty cells are NaN, these grids will now
# have NaN for any empty position, and [0,1] for occupied:
all_index_grid     = data['all_index_grid']     # list (REPLICATES) of dicts {t: 2D array (NaN=empty)}
all_deviation_grid = data['all_deviation_grid'] # list (REPLICATES) of dicts {t: 2D array of deviation}

# ──────────────────────────────────────────────────────────────────────────────
# 2) ANIMATION OF ONE REPLICATE’S 2D GRID (INDEX‐BASED COLORING)
#    with empty cells shown as green
# ──────────────────────────────────────────────────────────────────────────────

# Choose which replicate to animate (e.g. replicate #0):
rep_id = 0
index_grid_series = all_index_grid[rep_id]  # dict {t: 2D numpy array of shape=(size,size)}

# Determine sorted timesteps and number of frames:
timesteps = sorted(index_grid_series.keys())
N_frames  = len(timesteps)

# Determine grid size from one frame:
size = index_grid_series[timesteps[0]].shape[0]

# Create figure & axis for animation:
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(
    index_grid_series[timesteps[0]],
    vmin=0.0, vmax=1.0,
    cmap='coolwarm',
    interpolation='nearest'
)

# Paint NaN (the “bad” entries) in the colormap as green
im.cmap.set_bad(color='green')

ax.set_title(f"Replicate #{rep_id}, Step {timesteps[0]}")
ax.axis('off')

# Add a colorbar for [0,1]. NaN (green) is not shown on the colorbar.
cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label('Raw Index = channels_A/(A+B)')

def update_frame(frame_idx):
    t = timesteps[frame_idx]
    im.set_data(index_grid_series[t])
    ax.set_title(f"Replicate #{rep_id}, Step {t}")
    return [im]

anim = animation.FuncAnimation(
    fig,
    update_frame,
    frames=N_frames,
    interval=200,        # 200 ms between frames → 5 FPS
    blit=True
)

# ──────────────────────────────────────────────────────────────────────────────
# SAVE ANIMATION: TRY MP4 (ffmpeg), FALL BACK TO GIF (Pillow)
# ──────────────────────────────────────────────────────────────────────────────


gif_writer = PillowWriter(fps=5)
anim.save('bacteria_index_animation.gif', writer=gif_writer)
print("saved animation → bacteria_index_animation.gif")

plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 3) HEATMAP OF INDEX DISTRIBUTION OVER TIME (AGGREGATED OVER ALL REPLICATES)
# ──────────────────────────────────────────────────────────────────────────────

# We want an array H with shape (n_bins, T_max+1) such that
#   H[b, t] = # of cells (across all replicates) whose raw_index fell into bin b at time t.

n_bins = 50
bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Find the maximum timestep across all replicates:
all_timesteps = set()
for r in range(len(all_index)):
    all_timesteps.update(all_index[r].keys())
all_timesteps = sorted(all_timesteps)
T_max = max(all_timesteps)

# Initialize count array
heatmap_counts = np.zeros((n_bins, T_max + 1), dtype=int)

# Bin every cell’s raw_index into one of the 50 bins, for each replicate & each time:
for r in range(len(all_index)):
    idx_series = all_index[r]  # {t: [list of raw_index values]}
    for t, idx_list in idx_series.items():
        if t > T_max:
            continue
        counts, _ = np.histogram(idx_list, bins=bin_edges)
        heatmap_counts[:, t] += counts

# Plot the heatmap:
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
# 4) CURVE PLOT: MEAN ± SD OF RAW INDEX OVER TIME (ACROSS REPLICATES)
# ──────────────────────────────────────────────────────────────────────────────

R = len(all_index)

# Build an array of shape (R, T_max+1), where each entry is the mean raw_index
# for replicate r at time t (or NaN if no cells).
mean_index_matrix = np.full((R, T_max + 1), np.nan, dtype=float)

for r in range(R):
    idx_series = all_index[r]
    for t, idx_list in idx_series.items():
        if len(idx_list) > 0:
            mean_index_matrix[r, t] = np.mean(idx_list)
        else:
            mean_index_matrix[r, t] = np.nan

# Compute mean ± SD across replicates (ignoring NaNs)
mean_over_reps = np.nanmean(mean_index_matrix, axis=0)
std_over_reps  = np.nanstd(mean_index_matrix, axis=0)

timesteps_arr = np.arange(T_max + 1)
plt.figure(figsize=(10, 5))
plt.plot(timesteps_arr, mean_over_reps, label='Mean raw Index (all cells)')
plt.fill_between(
    timesteps_arr,
    mean_over_reps - std_over_reps,
    mean_over_reps + std_over_reps,
    alpha=0.3,
    label='± 1 SD'
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
# 5) OPTIONAL: CURVE PLOT FOR “LEAN FROM IDEAL” (DEVIATION) OVER TIME
# ──────────────────────────────────────────────────────────────────────────────

mean_dev_matrix = np.full((R, T_max + 1), np.nan, dtype=float)

for r in range(R):
    dev_series = all_deviation[r]
    for t, dev_list in dev_series.items():
        if len(dev_list) > 0:
            mean_dev_matrix[r, t] = np.mean(dev_list)
        else:
            mean_dev_matrix[r, t] = np.nan

mean_dev_over_reps = np.nanmean(mean_dev_matrix, axis=0)
std_dev_over_reps  = np.nanstd(mean_dev_matrix, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(timesteps_arr, mean_dev_over_reps, color='tab:red', label='Mean |ideal − index|')
plt.fill_between(
    timesteps_arr,
    mean_dev_over_reps - std_dev_over_reps,
    mean_dev_over_reps + std_dev_over_reps,
    alpha=0.3,
    color='tab:red',
    label='± 1 SD'
)
plt.xlabel('Timestep')
plt.ylabel('Mean |ideal − raw_index| ± SD')
plt.title('Mean ± 1 SD of “Lean from Ideal” over Time (All Replicates)')
plt.legend()
plt.tight_layout()
plt.savefig('mean_deviation_curve.png')
print("Saved curve plot → mean_deviation_curve.png")
plt.close()
