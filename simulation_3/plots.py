# plots.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from parameters_s3 import (
    MAX_STEPS,
    REPLICATES,
    REFILL_INTERVAL_A,
    REFILL_INTERVAL_B,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1) LOAD SIMULATION DATA
# ──────────────────────────────────────────────────────────────────────────────
with open('simulation_data.pkl', 'rb') as f:
    data = pickle.load(f)

all_pop_A      = data['all_pop_A']       # list of REPLICATES lists (# A‐type cells over time)
all_pop_B      = data['all_pop_B']       # list of REPLICATES lists (# B‐type cells over time)
all_chan_Atype = data['all_chan_Atype']  # list of REPLICATES dicts {t: [channels_A for A‐type cells]}
all_chan_Btype = data['all_chan_Btype']  # list of REPLICATES dicts {t: [channels_B for B‐type cells]}
all_energy     = data['all_energy']      # list of REPLICATES lists (total energy over time)

# ──────────────────────────────────────────────────────────────────────────────
# 2) PLOT AVERAGE POPULATION (A vs. B)
# ──────────────────────────────────────────────────────────────────────────────
# Pad A‐population and B‐population series to MAX_STEPS
padded_Apop = [
    np.pad(popA, (0, MAX_STEPS - len(popA)), 'constant', constant_values=0)
    for popA in all_pop_A
]
padded_Bpop = [
    np.pad(popB, (0, MAX_STEPS - len(popB)), 'constant', constant_values=0)
    for popB in all_pop_B
]

avg_Apop = np.mean(padded_Apop, axis=0)
avg_Bpop = np.mean(padded_Bpop, axis=0)

plt.figure(figsize=(10, 4))
plt.plot(range(MAX_STEPS), avg_Apop, label='Avg # A‐Type Cells', color='tab:blue')
plt.plot(range(MAX_STEPS), avg_Bpop, label='Avg # B‐Type Cells', color='tab:orange')
for step in range(0, MAX_STEPS, REFILL_INTERVAL_A):
    plt.axvline(x=step, color='gray', linestyle='--', linewidth=0.5)
for step in range(0, MAX_STEPS, REFILL_INTERVAL_B):
    plt.axvline(x=step, color='gray', linestyle=':', linewidth=0.5)
plt.xlabel('Time Step')
plt.ylabel('Average Number of Cells')
plt.title('Average Population: A‐Type vs. B‐Type Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('avg_population_AB.png')
print('Plot saved as avg_population_AB.png')

# ──────────────────────────────────────────────────────────────────────────────
# 3) PLOT AVERAGE TOTAL ENERGY
# ──────────────────────────────────────────────────────────────────────────────
padded_energy = [
    np.pad(es, (0, MAX_STEPS - len(es)), 'constant', constant_values=0)
    for es in all_energy
]
avg_energy = np.mean(padded_energy, axis=0)

plt.figure(figsize=(10, 4))
plt.plot(range(MAX_STEPS), avg_energy, label='Avg Total Energy', color='tab:purple')
for step in range(0, MAX_STEPS, REFILL_INTERVAL_A):
    plt.axvline(x=step, color='gray', linestyle='--', linewidth=0.5)
for step in range(0, MAX_STEPS, REFILL_INTERVAL_B):
    plt.axvline(x=step, color='gray', linestyle=':', linewidth=0.5)
plt.xlabel('Time Step')
plt.ylabel('Average Total Energy')
plt.title('Average Total Energy of All Cells Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('avg_energy.png')
print('Plot saved as avg_energy.png')

# ──────────────────────────────────────────────────────────────────────────────
# 4) FILTERED HEATMAP FOR CHANNEL‐A (A‐Type Cells Only)
# ──────────────────────────────────────────────────────────────────────────────
aggregated_A_filtered = {}
for rep_idx in range(REPLICATES):
    series_A = all_chan_Atype[rep_idx]
    for t, counts_A in series_A.items():
        # Only A‐type cells contributed to this list, so counts_A > 0 implies they have ≥ 1 A‐channel
        # But in case a cell actually had channels_A=0 despite being A‐type, we filter zero out as well:
        positive_A = [c for c in counts_A if c > 0]
        if positive_A:
            aggregated_A_filtered.setdefault(t, []).extend(positive_A)

timepoints_A = sorted(aggregated_A_filtered.keys())
max_ch_A = max((max(lst) if lst else 0) for lst in aggregated_A_filtered.values()) if aggregated_A_filtered else 0

heatmap_data_A = np.zeros((len(timepoints_A), max_ch_A + 1), dtype=int)
for i, t in enumerate(timepoints_A):
    for ch in aggregated_A_filtered[t]:
        if 1 <= ch <= max_ch_A:
            heatmap_data_A[i, ch] += 1

plt.figure(figsize=(12, 5))
sns.heatmap(
    heatmap_data_A.T,
    cmap='viridis',
    cbar_kws={'label': 'Cell Count (A‐Type, A ≥ 1)'},
    xticklabels=timepoints_A,
    yticklabels=range(1, max_ch_A + 1)
)
plt.xlabel('Time Step')
plt.ylabel('Number of A‐Channels')
plt.title('Channel‐A Distribution Over Time (A‐Type Cells Only)')
for step in range(0, MAX_STEPS, REFILL_INTERVAL_A):
    plt.axvline(x=step, color='white', linestyle='--', linewidth=0.5)
for step in range(0, MAX_STEPS, REFILL_INTERVAL_B):
    plt.axvline(x=step, color='white', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.savefig('channelA_typeA_heatmap.png')
print('Plot saved as channelA_typeA_heatmap.png')

# ──────────────────────────────────────────────────────────────────────────────
# 5) FILTERED HEATMAP FOR CHANNEL‐B (B‐Type Cells Only)
# ──────────────────────────────────────────────────────────────────────────────
aggregated_B_filtered = {}
for rep_idx in range(REPLICATES):
    series_B = all_chan_Btype[rep_idx]
    for t, counts_B in series_B.items():
        # Only B‐type cells contributed to this list, so counts_B > 0 implies they have ≥ 1 B‐channel
        positive_B = [c for c in counts_B if c > 0]
        if positive_B:
            aggregated_B_filtered.setdefault(t, []).extend(positive_B)

timepoints_B = sorted(aggregated_B_filtered.keys())
max_ch_B = max((max(lst) if lst else 0) for lst in aggregated_B_filtered.values()) if aggregated_B_filtered else 0

heatmap_data_B = np.zeros((len(timepoints_B), max_ch_B + 1), dtype=int)
for i, t in enumerate(timepoints_B):
    for ch in aggregated_B_filtered[t]:
        if 1 <= ch <= max_ch_B:
            heatmap_data_B[i, ch] += 1

plt.figure(figsize=(12, 5))
sns.heatmap(
    heatmap_data_B.T,
    cmap='magma',
    cbar_kws={'label': 'Cell Count (B‐Type, B ≥ 1)'},
    xticklabels=timepoints_B,
    yticklabels=range(1, max_ch_B + 1)
)
plt.xlabel('Time Step')
plt.ylabel('Number of B‐Channels')
plt.title('Channel‐B Distribution Over Time (B‐Type Cells Only)')
for step in range(0, MAX_STEPS, REFILL_INTERVAL_A):
    plt.axvline(x=step, color='white', linestyle='--', linewidth=0.5)
for step in range(0, MAX_STEPS, REFILL_INTERVAL_B):
    plt.axvline(x=step, color='white', linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.savefig('channelB_typeB_heatmap.png')
print('Plot saved as channelB_typeB_heatmap.png')

# ──────────────────────────────────────────────────────────────────────────────
# 6) MEAN CHANNEL COUNT CURVES (per type)
# ──────────────────────────────────────────────────────────────────────────────
meanA_reps = []
meanB_reps = []
for rep_idx in range(REPLICATES):
    series_A = all_chan_Atype[rep_idx]  # channels_A for A‐type cells only
    series_B = all_chan_Btype[rep_idx]  # channels_B for B‐type cells only

    meanA = np.zeros(MAX_STEPS)
    meanB = np.zeros(MAX_STEPS)
    for t in range(MAX_STEPS):
        counts_A_t = series_A.get(t, [])
        counts_B_t = series_B.get(t, [])
        meanA[t] = np.mean(counts_A_t) if counts_A_t else 0.0
        meanB[t] = np.mean(counts_B_t) if counts_B_t else 0.0

    meanA_reps.append(meanA)
    meanB_reps.append(meanB)

meanA_reps = np.vstack(meanA_reps)
meanB_reps = np.vstack(meanB_reps)
avg_meanA = np.mean(meanA_reps, axis=0)
avg_meanB = np.mean(meanB_reps, axis=0)

plt.figure(figsize=(10, 4))
plt.plot(range(MAX_STEPS), avg_meanA, label='Mean A‐Channels (A‐Type)', color='tab:blue')
plt.plot(range(MAX_STEPS), avg_meanB, label='Mean B‐Channels (B‐Type)', color='tab:orange')
for step in range(0, MAX_STEPS, REFILL_INTERVAL_A):
    plt.axvline(x=step, color='gray', linestyle='--', linewidth=0.5)
for step in range(0, MAX_STEPS, REFILL_INTERVAL_B):
    plt.axvline(x=step, color='gray', linestyle=':', linewidth=0.5)
plt.xlabel('Time Step')
plt.ylabel('Mean # of Channels per Cell (by Type)')
plt.title('Mean A‐Channels (A‐Type) vs. Mean B‐Channels (B‐Type) Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('mean_channels_by_type.png')
print('Plot saved as mean_channels_by_type.png')
