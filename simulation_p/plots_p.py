# plots_p.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from parameters_p import (
    MAX_STEPS,
    REPLICATES,
    REFILL_INTERVAL_A,
    REFILL_INTERVAL_B,
    DELTA_T
)

def load_data(pickle_file='simulation_results.pkl'):
    """
    Load simulation results from a pickle file.
    Returns a dict with keys:
      - all_pop_A, all_pop_B
      - all_chan_Atype, all_chan_Btype
      - all_energy
      - all_phage_A, all_phage_B
    """
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return data

def pad_series(series_list, length):
    """
    Pad each 1D list in series_list to exactly 'length' using zeros.
    Returns a 2D array of shape (num_replicates, length).
    """
    padded = []
    for series in series_list:
        padded.append(np.pad(series, (0, length - len(series)), 'constant', constant_values=0))
    return np.vstack(padded)

def plot_avg_population(data):
    all_pop_A = data['all_pop_A']
    all_pop_B = data['all_pop_B']

    padded_A = pad_series(all_pop_A, MAX_STEPS)
    padded_B = pad_series(all_pop_B, MAX_STEPS)

    avg_A = np.mean(padded_A, axis=0)
    avg_B = np.mean(padded_B, axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(range(MAX_STEPS), avg_A, label='Avg # A-Type Cells', color='tab:blue')
    plt.plot(range(MAX_STEPS), avg_B, label='Avg # B-Type Cells', color='tab:orange')

    # A-meal lines (green dashed)
    for step in range(REFILL_INTERVAL_A, MAX_STEPS, REFILL_INTERVAL_A):
        plt.axvline(x=step, color='tab:green', linestyle='--', linewidth=0.7,
                    label='A-Meal' if step == REFILL_INTERVAL_A else '')
    # B-meal lines (red dotted)
    first_B = DELTA_T
    while first_B < MAX_STEPS:
        plt.axvline(x=first_B, color='tab:red', linestyle=':', linewidth=0.7,
                    label='B-Meal' if first_B == DELTA_T else '')
        first_B += REFILL_INTERVAL_B

    plt.xlabel('Time Step')
    plt.ylabel('Average Number of Cells')
    plt.title('Average Population: A-Type vs. B-Type Over Time')
    # Place legend explicitly in upper-left to avoid “best” slow search
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('avg_population_AB.png')
    print('Plot saved as avg_population_AB.png')
    plt.close()

def plot_avg_energy(data):
    all_energy = data['all_energy']

    padded_E = pad_series(all_energy, MAX_STEPS)
    avg_E = np.mean(padded_E, axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(range(MAX_STEPS), avg_E, label='Avg Total Energy', color='tab:purple')

    # A-meal lines
    for step in range(REFILL_INTERVAL_A, MAX_STEPS, REFILL_INTERVAL_A):
        plt.axvline(x=step, color='tab:green', linestyle='--', linewidth=0.7)
    # B-meal lines
    first_B = DELTA_T
    while first_B < MAX_STEPS:
        plt.axvline(x=first_B, color='tab:red', linestyle=':', linewidth=0.7)
        first_B += REFILL_INTERVAL_B

    plt.xlabel('Time Step')
    plt.ylabel('Average Total Energy')
    plt.title('Average Total Energy of All Cells Over Time')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('avg_energy.png')
    print('Plot saved as avg_energy.png')
    plt.close()

def plot_avg_phage(data):
    all_phage_A = data['all_phage_A']
    all_phage_B = data['all_phage_B']

    padded_phA = pad_series(all_phage_A, MAX_STEPS)
    padded_phB = pad_series(all_phage_B, MAX_STEPS)

    avg_phA = np.mean(padded_phA, axis=0)
    avg_phB = np.mean(padded_phB, axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(range(MAX_STEPS), avg_phA, label='Avg Phage A', color='tab:green')
    plt.plot(range(MAX_STEPS), avg_phB, label='Avg Phage B', color='tab:red')

    # A-meal lines
    for step in range(REFILL_INTERVAL_A, MAX_STEPS, REFILL_INTERVAL_A):
        plt.axvline(x=step, color='tab:green', linestyle='--', linewidth=0.7)
    # B-meal lines
    first_B = DELTA_T
    while first_B < MAX_STEPS:
        plt.axvline(x=first_B, color='tab:red', linestyle=':', linewidth=0.7)
        first_B += REFILL_INTERVAL_B

    plt.xlabel('Time Step')
    plt.ylabel('Average Number of Phage Particles')
    plt.title('Average Phage A vs. Phage B Over Time')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('avg_phage_AB.png')
    print('Plot saved as avg_phage_AB.png')
    plt.close()

def plot_channel_heatmaps(data):
    all_chan_A = data['all_chan_Atype']
    all_chan_B = data['all_chan_Btype']

    # A-Type channel heatmap
    aggregated_A = {}
    for rep_idx in range(REPLICATES):
        series_A = all_chan_A[rep_idx]  # dict: t -> list of channels_A for A-type cells
        for t, counts in series_A.items():
            positive = [c for c in counts if c > 0]
            if positive:
                aggregated_A.setdefault(t, []).extend(positive)

    time_A = sorted(aggregated_A.keys())
    max_ch_A = max((max(vals) if vals else 0) for vals in aggregated_A.values()) if aggregated_A else 0

    heat_A = np.zeros((len(time_A), max_ch_A + 1), dtype=int)
    for i, t in enumerate(time_A):
        for ch in aggregated_A[t]:
            if 1 <= ch <= max_ch_A:
                heat_A[i, ch] += 1

    plt.figure(figsize=(12, 5))
    sns.heatmap(
        heat_A.T,
        cmap='viridis',
        cbar_kws={'label': 'Cell Count (A-Type, A ≥ 1)'},
        xticklabels=time_A,
        yticklabels=range(1, max_ch_A + 1)
    )
    plt.xlabel('Time Step')
    plt.ylabel('Number of A-Channels')
    plt.title('Channel-A Distribution Over Time (A-Type Cells Only)')
    # A-meals (white dashed) & B-meals (white dotted) on heatmap
    for step in range(REFILL_INTERVAL_A, MAX_STEPS, REFILL_INTERVAL_A):
        if step in time_A:
            plt.axvline(x=time_A.index(step), color='white', linestyle='--', linewidth=0.7)
    first_B = DELTA_T
    while first_B < MAX_STEPS:
        if first_B in time_A:
            plt.axvline(x=time_A.index(first_B), color='white', linestyle=':', linewidth=0.7)
        first_B += REFILL_INTERVAL_B
    plt.tight_layout()
    plt.savefig('channelA_typeA_heatmap.png')
    print('Plot saved as channelA_typeA_heatmap.png')
    plt.close()

    # B-Type channel heatmap
    aggregated_B = {}
    for rep_idx in range(REPLICATES):
        series_B = all_chan_B[rep_idx]  # dict: t -> list of channels_B for B-type cells
        for t, counts in series_B.items():
            positive = [c for c in counts if c > 0]
            if positive:
                aggregated_B.setdefault(t, []).extend(positive)

    time_B = sorted(aggregated_B.keys())
    max_ch_B = max((max(vals) if vals else 0) for vals in aggregated_B.values()) if aggregated_B else 0

    heat_B = np.zeros((len(time_B), max_ch_B + 1), dtype=int)
    for i, t in enumerate(time_B):
        for ch in aggregated_B[t]:
            if 1 <= ch <= max_ch_B:
                heat_B[i, ch] += 1

    plt.figure(figsize=(12, 5))
    sns.heatmap(
        heat_B.T,
        cmap='magma',
        cbar_kws={'label': 'Cell Count (B-Type, B ≥ 1)'},
        xticklabels=time_B,
        yticklabels=range(1, max_ch_B + 1)
    )
    plt.xlabel('Time Step')
    plt.ylabel('Number of B-Channels')
    plt.title('Channel-B Distribution Over Time (B-Type Cells Only)')
    for step in range(REFILL_INTERVAL_A, MAX_STEPS, REFILL_INTERVAL_A):
        if step in time_B:
            plt.axvline(x=time_B.index(step), color='white', linestyle='--', linewidth=0.7)
    first_B = DELTA_T
    while first_B < MAX_STEPS:
        if first_B in time_B:
            plt.axvline(x=time_B.index(first_B), color='white', linestyle=':', linewidth=0.7)
        first_B += REFILL_INTERVAL_B
    plt.tight_layout()
    plt.savefig('channelB_typeB_heatmap.png')
    print('Plot saved as channelB_typeB_heatmap.png')
    plt.close()

def plot_mean_channels(data):
    all_chan_A = data['all_chan_Atype']
    all_chan_B = data['all_chan_Btype']

    meanA_reps = []
    meanB_reps = []

    for rep_idx in range(REPLICATES):
        series_A = all_chan_A[rep_idx]  # dict: t -> list of channels_A
        series_B = all_chan_B[rep_idx]  # dict: t -> list of channels_B

        meanA = np.zeros(MAX_STEPS)
        meanB = np.zeros(MAX_STEPS)
        for t in range(MAX_STEPS):
            countsA = series_A.get(t, [])
            countsB = series_B.get(t, [])
            meanA[t] = np.mean(countsA) if countsA else 0.0
            meanB[t] = np.mean(countsB) if countsB else 0.0

        meanA_reps.append(meanA)
        meanB_reps.append(meanB)

    meanA_reps = np.vstack(meanA_reps)
    meanB_reps = np.vstack(meanB_reps)
    avg_meanA = np.mean(meanA_reps, axis=0)
    avg_meanB = np.mean(meanB_reps, axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(range(MAX_STEPS), avg_meanA, label='Mean A-Channels (A-Type)', color='tab:blue')
    plt.plot(range(MAX_STEPS), avg_meanB, label='Mean B-Channels (B-Type)', color='tab:orange')

    for step in range(REFILL_INTERVAL_A, MAX_STEPS, REFILL_INTERVAL_A):
        plt.axvline(x=step, color='tab:green', linestyle='--', linewidth=0.7)
    first_B = DELTA_T
    while first_B < MAX_STEPS:
        plt.axvline(x=first_B, color='tab:red', linestyle=':', linewidth=0.7)
        first_B += REFILL_INTERVAL_B

    plt.xlabel('Time Step')
    plt.ylabel('Mean # of Channels per Cell (by Type)')
    plt.title('Mean A-Channels vs. Mean B-Channels Over Time')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('mean_channels_by_type.png')
    print('Plot saved as mean_channels_by_type.png')
    plt.close()

def main(pickle_file='simulation_results.pkl'):
    data = load_data(pickle_file)
    plot_avg_population(data)
    plot_avg_energy(data)
    plot_avg_phage(data)
    plot_channel_heatmaps(data)
    plot_mean_channels(data)

if __name__ == '__main__':
    main()
