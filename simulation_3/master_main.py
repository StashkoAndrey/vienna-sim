# ======= master_main.py =======
import csv
import itertools
import multiprocessing as mp
import os

import numpy as np

from grid_s3 import SimulationGrid
from diffusion_cpp import diffuse
from parameters_s3 import (
    SWEEP_FREQ_MIN,
    SWEEP_FREQ_MAX,
    SWEEP_FREQ_STEP,
    SWEEP_AMT_MIN,
    SWEEP_AMT_MAX,
    SWEEP_AMT_STEP,
    SWEEP_REPLICATES,
    SWEEP_MAX_STEPS,
    DIFFUSION_RATE,
    GRID_SIZE,
    CHANNEL_COST,
)

def run_setting(freq, meal):
    """
    Runs SWEEP_REPLICATES simulations with:
      - REFILL_INTERVAL_A = REFILL_INTERVAL_B = freq
      - MEAL_TOTAL_A      = MEAL_TOTAL_B      = meal
      - Up to SWEEP_MAX_STEPS steps

    Returns:
      (freq, meal, n_survived, mean_biomass, mean_div_time)
    where:
      - n_survived    = number of replicates with pop > 0 at step SWEEP_MAX_STEPS
      - mean_biomass  = average (over replicates) of the time-averaged population
      - mean_div_time = average (over replicates) of “mean division timestep”
                        (where “mean division timestep” for a single replicate 
                         is the mean of all timesteps in grid.division_log;
                         if no divisions, we treat it as SWEEP_MAX_STEPS + 1)
    """
    REFILL_INTERVAL_A = freq
    REFILL_INTERVAL_B = freq
    MEAL_TOTAL_A = meal
    MEAL_TOTAL_B = meal

    survived_count = 0
    replicate_biomass = []
    replicate_div_times = []

    for _ in range(SWEEP_REPLICATES):
        grid = SimulationGrid(GRID_SIZE, CHANNEL_COST)
        center = GRID_SIZE // 2
        grid.place_bacterium(center, center)

        cumulative_pop = 0.0

        # Run up to SWEEP_MAX_STEPS
        for step in range(SWEEP_MAX_STEPS):
            grid.step(step)

            current_pop = grid.count_bacteria()
            cumulative_pop += current_pop

            if current_pop == 0:
                # If the population goes extinct, pad remaining steps with pop = 0
                cumulative_pop += 0 * (SWEEP_MAX_STEPS - step - 1)
                break

            # Diffuse & refill exactly as before:
            grid.nutrients_A = diffuse(grid.nutrients_A, DIFFUSION_RATE)
            grid.nutrients_B = diffuse(grid.nutrients_B, DIFFUSION_RATE)

            if step % REFILL_INTERVAL_A == 0 and step != 0:
                grid.refill_nutrient_A(MEAL_TOTAL_A)
            if step % REFILL_INTERVAL_B == 0 and step != 0:
                grid.refill_nutrient_B(MEAL_TOTAL_B)
        # ───────── end of one replicate ─────────

        # 1) Count survival
        final_pop = grid.count_bacteria()
        if final_pop > 0:
            survived_count += 1

        # 2) Compute mean biomass for this replicate
        biomass = cumulative_pop / float(SWEEP_MAX_STEPS)
        replicate_biomass.append(biomass)

        # 3) Compute “mean division timestep” for this replicate
        #    grid.division_log is a list of tuples (timestep, parent_id, preA, preB, child_id, childA, childB).
        if len(grid.division_log) > 0:
            # Extract only the timestep (first element of each tuple)
            division_steps = [entry[0] for entry in grid.division_log]
            mean_div_step = float(np.mean(division_steps))
        else:
            # If no divisions, set to SWEEP_MAX_STEPS + 1 (arbitrary sentinel)
            mean_div_step = float(SWEEP_MAX_STEPS + 1)
        replicate_div_times.append(mean_div_step)

    # ───────── end of all replicates ─────────

    mean_biomass = float(np.mean(replicate_biomass))
    mean_div_time = float(np.mean(replicate_div_times))
    return (freq, meal, survived_count, mean_biomass, mean_div_time)


def run_setting_tuple(args):
    return run_setting(*args)


def main():
    freq_values = range(SWEEP_FREQ_MIN, SWEEP_FREQ_MAX + 1, SWEEP_FREQ_STEP)
    amt_values = range(SWEEP_AMT_MIN, SWEEP_AMT_MAX + 1, SWEEP_AMT_STEP)
    all_pairs = list(itertools.product(freq_values, amt_values))

    output_file = "sweep_results.csv"
    header = [
        "REFILL_INT",
        "MEAL_AMOUNT",
        "N_SURVIVED",
        "MEAN_BIOMASS",
        "MEAN_DIV_TIME",
    ]

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    n_cpus = max(1, mp.cpu_count() - 1)
    pool = mp.Pool(processes=n_cpus)

    results_iter = pool.imap_unordered(run_setting_tuple, all_pairs, chunksize=5)

    with open(output_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        for (freq, meal, ns, mb, mdt) in results_iter:
            writer.writerow([freq, meal, ns, mb, mdt])
            print("+", end="", flush=True)

    pool.close()
    pool.join()
    print("\nAll done! Results written to", output_file)


if __name__ == "__main__":
    main()
