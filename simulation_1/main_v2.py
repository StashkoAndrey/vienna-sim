# main_v2.py

import matplotlib.pyplot as plt
from grid import SimulationGrid
from diffusion_cpp import diffuse
import logging
import numpy as np
import csv

# ---------- Simulation parameters ----------
GRID_SIZE      = 10
MAX_STEPS      = 1000
DIFFUSION_RATE = 0.05
INITIAL_NUTRIENT = 3.5
REPLICATES     = 10
LOG_STEPS      = False  # Set to True to log each step (for debugging)

# logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


def run_simulation(channel_cost):
    """
    Run one full replicate with the given channel_cost.
    Return:
      - population: a list of population‐size at each time step
      - channel_count: the TIME‐AVERAGE number of channels observed during that replicate
    """
    # 1) Initialize grid & place a single bacterium in the center
    grid = SimulationGrid(
        GRID_SIZE,
        channel_cost=channel_cost,
        initial_nutrient=INITIAL_NUTRIENT
    )
    center = GRID_SIZE // 2
    grid.place_bacterium(center, center)

    population = []
    channel_counts_over_time = []   # <-- record channels each step

    logging.info(f"\n=== Running simulation for channel_cost = {channel_cost} ===")

    for step in range(MAX_STEPS):
        # a) Advance every bacterium (grow/divide/die)
        grid.step()

        # b) Diffuse nutrients across the grid
        grid.nutrients = diffuse(grid.nutrients, DIFFUSION_RATE)

        # c) Count how many bacteria remain this step
        count = grid.count_bacteria()

        # d) (Optional) Log total energy + nutrients for debugging
        total_energy   = grid.total_energy()
        total_nutrient = grid.total_nutrients()
        
        if LOG_STEPS:
            logging.info(
                f"Step {step:03}: Count = {count}, "
                f"Energy = {total_energy:.2f}, Nutrient = {total_nutrient:.2f}"
        )

        # e) Append current population size
        population.append(count)

        # f) RECORD the total number of “channels” (uptake units) THIS step
        channel_counts_over_time.append(grid.count_channels())

        # g) If all bacteria are gone, end early
        if count == 0:
            logging.info("All bacteria extinct. Ending simulation early.\n")
            break

    # Instead of max(), take the time‐average over all recorded steps:
    if len(channel_counts_over_time) > 0:
        channel_count = np.mean(channel_counts_over_time)
    else:
        channel_count = 0.0

    return population, channel_count


def pad_curves(curves, fill_value=0):
    """
    Pad a list of population‐time curves so that they all share the same length.
    (Not strictly used for the CSV export, but available if you need uniform arrays later.)
    """
    max_len = max(len(c) for c in curves)
    return np.array([c + [fill_value] * (max_len - len(c)) for c in curves])


def main():
    #print("Starting simulation…")

    # Sweep channel_cost from 0.00 to 1.10 in steps of 0.005
    costs = np.arange(0.0, 1.101, 0.005)

    # Lists to hold summary statistics at each cost
    mean_lifetimes    = []
    std_lifetimes     = []
    extinction_counts = []
    mean_peaks        = []
    mean_totals       = []
    mean_areas        = []
    mean_channels     = []   # <-- now “mean of per‐replicate TIME‐AVERAGE channels”

    # Loop over every channel_cost value
    for cost in costs:
        lifetimes = []
        peaks     = []
        totals    = []
        areas     = []
        channels_list = []
        extinctions   = 0

        for i in range(REPLICATES):
            # Run one replicate
            curve, channel_count = run_simulation(round(cost, 3))

            # Lifetime = number of steps before extinction (or MAX_STEPS if never extinct)
            lifetimes.append(len(curve))

            # Peak population in this replicate
            peaks.append(max(curve) if len(curve) > 0 else 0)

            # Total “cells‐per‐step” (sum of population over time)
            totals.append(sum(curve))

            # Continuous area under the population curve (via trapezoidal rule)
            areas.append(np.trapz(curve) if len(curve) > 0 else 0.0)

            # Record the TIME‐AVERAGE channel count for this replicate
            channels_list.append(channel_count)

            # Count how many replicates went extinct by the end
            if len(curve) > 0 and curve[-1] == 0:
                extinctions += 1

        # Compute summary stats for this channel_cost
        mean_lifetimes.append(np.mean(lifetimes))
        std_lifetimes.append(np.std(lifetimes))
        extinction_counts.append(extinctions)
        mean_peaks.append(np.mean(peaks))
        mean_totals.append(np.mean(totals))
        mean_areas.append(np.mean(areas))
        mean_channels.append(np.mean(channels_list))

    # --------------------------------------------------------------------
    # 1) Plot: Mean Lifetime vs Channel Cost
    plt.figure(figsize=(10, 5))
    plt.errorbar(costs, mean_lifetimes, yerr=std_lifetimes, fmt='-o', capsize=4)
    plt.xlabel('Channel Cost')
    plt.ylabel('Average Lifetime (steps)')
    plt.title('Bacterial Lifetime vs Channel Cost')
    plt.grid(True)
    plt.savefig("plot_lifetime_vs_cost.png")
    print("Plot saved as plot_lifetime_vs_cost.png")

    # 2) Plot: Mean Peak Population vs Channel Cost
    plt.figure(figsize=(10, 5))
    plt.plot(costs, mean_peaks, '-o')
    plt.xlabel('Channel Cost')
    plt.ylabel('Mean Peak Population')
    plt.title('Peak Population vs Channel Cost')
    plt.grid(True)
    plt.savefig("plot_peaks_vs_cost.png")
    print("Plot saved as plot_peaks_vs_cost.png")

    # 3) Plot: Mean Total Cells (Sum over Time) vs Channel Cost
    plt.figure(figsize=(10, 5))
    plt.plot(costs, mean_totals, '-o')
    plt.xlabel('Channel Cost')
    plt.ylabel('Mean Total Cells (sum over time)')
    plt.title('Total Cells vs Channel Cost')
    plt.grid(True)
    plt.savefig("plot_totals_vs_cost.png")
    print("Plot saved as plot_totals_vs_cost.png")

    # 4) Plot: Mean Area under Population Curve vs Channel Cost
    plt.figure(figsize=(10, 5))
    plt.plot(costs, mean_areas, '-o')
    plt.xlabel('Channel Cost')
    plt.ylabel('Mean ∫Population(t) dt')
    plt.title('Population Area vs Channel Cost')
    plt.grid(True)
    plt.savefig("plot_area_vs_cost.png")
    print("Plot saved as plot_area_vs_cost.png")

    # 5) Plot: Mean TIME‐AVERAGE Channels vs Channel Cost
    plt.figure(figsize=(10, 5))
    plt.plot(costs, mean_channels, '-o')
    plt.xlabel('Channel Cost')
    plt.ylabel('Mean Time‐Average Channels')
    plt.title('Average Number of Channels vs Channel Cost')
    plt.grid(True)
    plt.savefig("plot_mean_channels_vs_cost.png")
    print("Plot saved as plot_mean_channels_vs_cost.png")

    # --------------------------------------------------------------------
    # Write all data (including mean time‐average channels) to CSV
    with open("lifetime_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "channel_cost",
            "mean_lifetime",
            "std_lifetime",
            "extinction_count",
            "mean_peak_pop",
            "mean_total_cells",
            "mean_area_under_pop",
            "mean_avg_channels"   # <-- renamed to reflect “average” instead of “peak”
        ])
        for i, cost in enumerate(costs):
            writer.writerow([
                round(cost, 5),
                round(mean_lifetimes[i], 3),
                round(std_lifetimes[i], 3),
                extinction_counts[i],
                round(mean_peaks[i], 3),
                round(mean_totals[i], 3),
                round(mean_areas[i], 3),
                round(mean_channels[i], 3)
            ])

    print("CSV saved as lifetime_data.csv")


if __name__ == '__main__':
    main()
