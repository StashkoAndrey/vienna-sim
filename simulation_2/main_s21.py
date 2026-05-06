import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import logging
import numpy as np
from grid_s21 import SimulationGrid
from diffusion_cpp import diffuse
from parameters import GRID_SIZE, MAX_STEPS, DIFFUSION_RATE, INITIAL_NUTRIENT, REPLICATES, MEAL_INTERVAL, MEAL_TOTAL, CHANNEL_COST

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

def run_simulation(channel_cost):
    grid = SimulationGrid(GRID_SIZE, channel_cost, initial_nutrient=INITIAL_NUTRIENT)
    center = GRID_SIZE // 2
    grid.place_bacterium(center, center)
    population = []
    energy_series = []

    for step in range(MAX_STEPS):
        grid.step(step)
        grid.nutrients = diffuse(grid.nutrients, DIFFUSION_RATE)
        if step % MEAL_INTERVAL == 0 and step != 0:
            grid.refill_nutrients(MEAL_TOTAL)
        count = grid.count_bacteria()
        energy_series.append(grid.total_energy())

        # ─── LOGGING ────────────────────────────────────────────────────────────
        total_energy = grid.total_energy()
        total_nutrients = grid.total_nutrients()
        #logging.info(
        #    f"Step {step}, "
        #    f"Bacteria {count}, "
        #    f"Total Energy {total_energy:.2f}, "
        #    f"Total Nutrients {total_nutrients:.2f}"
        #)
        # ────────────────────────────────────────────────────────────────────────
        
        population.append(count)
        if count == 0:
            break
    return population, grid.channel_time_series, energy_series


def main():
    print("Starting simulation...")
    all_channel_time_series = []
    all_populations = []
    all_energy = []

    for _ in range(REPLICATES):
        pop, series, energy_series = run_simulation(CHANNEL_COST)
        all_populations.append(pop)
        all_channel_time_series.append(series)
        all_energy.append(energy_series)

    # === Plot average population over time ===
    avg_pop = np.mean([np.pad(p, (0, MAX_STEPS - len(p)), 'constant', constant_values=0) for p in all_populations], axis=0)
    plt.figure(figsize=(10,4))
    plt.plot(range(MAX_STEPS), avg_pop, label='Avg Population')
    for step in range(0, MAX_STEPS, MEAL_INTERVAL):
        plt.axvline(x=step, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Avg Number of Bacteria')
    plt.title('Average Population Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('average_population.png')
    print('Plot saved as average_population.png')


    # === Plot average energy over time ===
    # 1) Pad each replicate's series to length MAX_STEPS
    padded = [
        np.pad(es, (0, MAX_STEPS - len(es)), mode='constant', constant_values=0)
        for es in all_energy
    ]
    # 2) Compute the mean across replicates
    avg_en = np.mean(padded, axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(range(MAX_STEPS), avg_en, label='Avg Energy')
    for step in range(0, MAX_STEPS, MEAL_INTERVAL):
        plt.axvline(x=step, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Avg Energy of Bacteria')
    plt.title('Average Energy Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig('average_energy.png')
    print('Plot saved as average_energy.png')
    

    # === Heatmap of channel distributions ===
    aggregated = {}
    for series in all_channel_time_series:
        for t, counts in series.items():
            aggregated.setdefault(t, []).extend(counts)
    timepoints = sorted(aggregated.keys())
    max_ch = max((max(c) if c else 0) for c in aggregated.values())
    heatmap_data = np.zeros((len(timepoints), max_ch+1))
    for i, t in enumerate(timepoints):
        for ch in aggregated[t]:
            heatmap_data[i, ch] += 1
    plt.figure(figsize=(20,6))
    sns.heatmap(heatmap_data.T, cmap=cm.viridis, cbar_kws={'label':'Cell Count'})
    plt.xlabel('Time Step')
    plt.ylabel('Number of Channels')
    plt.title('Channel Distribution Over Time')
    for step in range(0, MAX_STEPS, MEAL_INTERVAL):
        plt.axvline(step, color='white', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('channel_heatmap.png')
    print('Plot saved as channel_heatmap.png')

    # === Subplots by channel group ===
    df = pd.DataFrame(heatmap_data, index=timepoints, columns=[str(i) for i in range(max_ch+1)])
    group_size = 3
    bins = [list(range(i, min(i+group_size, max_ch+1))) for i in range(0, max_ch+1, group_size)]
    fig, axes = plt.subplots(len(bins), 1, figsize=(20, 3*len(bins)), sharex=True)
    if len(bins)==1:
        axes=[axes]
    for ax, grp in zip(axes, bins):
        for ch in grp:
            ax.plot(timepoints, df[str(ch)], label=f'Ch {ch}')
        for step in range(0, MAX_STEPS, MEAL_INTERVAL):
            ax.axvline(step, color='black', linestyle='--', linewidth=0.5)
        ax.set_title(f'Channels {grp[0]}–{grp[-1]}')
        ax.legend(loc='upper right', fontsize='small')
    axes[-1].set_xlabel('Time Step')
    for ax in axes:
        ax.set_ylabel('Cell Count')
    plt.tight_layout()
    plt.savefig('channel_subplots.png')
    print('Plot saved as channel_subplots.png')

if __name__ == '__main__':
    main()