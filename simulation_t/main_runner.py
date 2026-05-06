# Updated main_runner.py
import numpy as np
import pandas as pd
import csv
import os
from multiprocessing import Pool
from grid_5 import SimulationGrid
from diffusion_cpp import diffuse

# Simulation configuration
GRID_SIZE = 10
MAX_STEPS = 1000
REPLICATES = 5
OUTPUT_DIR = "timeseries"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load parameter sets
param_table = pd.read_csv("parameters_2.csv")

# ───────────── Nutrient splitter ─────────────
def _distribute_meal(total, major_frac):
    major = total * major_frac
    remainder = total - major
    split = np.random.rand()
    return major, remainder * split, remainder * (1 - split)

# ───────────── Save time series to file ─────────────
def save_timeseries(sim_id, rep_id, bacteria, phage, energy, majA, majB, majC):
    df = pd.DataFrame({
        "time": np.arange(len(bacteria)),
        "bacteria": bacteria,
        "phage": phage,
        "energy": energy,
        "majA": majA,
        "majB": majB,
        "majC": majC
    })
    df.replace([np.nan, np.inf, -np.inf], 0.0, inplace=True)
    filename = os.path.join(OUTPUT_DIR, f"sim_{sim_id:04d}_rep_{rep_id}.csv")
    df.to_csv(filename, index=False)

# ───────────── One replicate ─────────────
def run_replicate(sim_id, rep_id, param_row_dict):
    g = SimulationGrid(GRID_SIZE, channel_cost=param_row_dict['CHANNEL_COST'], params=param_row_dict)
    centre = GRID_SIZE // 2
    g.place_bacterium(centre, centre)

    pop_counts = []
    energy_levels = []
    phage_totals = []
    majA_series, majB_series, majC_series = [], [], []
    survival_time = MAX_STEPS

    for step in range(MAX_STEPS):
        g.step(step)

        g.nutrients_A = diffuse(g.nutrients_A, param_row_dict['DIFFUSION_RATE'])
        g.nutrients_B = diffuse(g.nutrients_B, param_row_dict['DIFFUSION_RATE'])
        g.nutrients_C = diffuse(g.nutrients_C, param_row_dict['DIFFUSION_RATE'])

        phages_A_rate = min(param_row_dict['PHAGE_A_DIFFUSION_RATE'], 0.15)
        phages_B_rate = min(param_row_dict['PHAGE_B_DIFFUSION_RATE'], 0.15)
        phages_C_rate = min(param_row_dict['PHAGE_C_DIFFUSION_RATE'], 0.15)

        g.phages_A = np.clip(diffuse(g.phages_A, phages_A_rate), 0, None)
        g.phages_B = np.clip(diffuse(g.phages_B, phages_B_rate), 0, None)
        g.phages_C = np.clip(diffuse(g.phages_C, phages_C_rate), 0, None)

        if step and step % param_row_dict['REFILL_INTERVAL_A'] == 0:
            maj, mB, mC = _distribute_meal(param_row_dict['MEAL_TOTAL_A'], param_row_dict['MAJOR_MEAL_FRACTION'])
            g.refill_nutrient_A(maj)
            g.refill_nutrient_B(mB)
            g.refill_nutrient_C(mC)
        if step >= param_row_dict['DELTA_T'] and (step - param_row_dict['DELTA_T']) % param_row_dict['REFILL_INTERVAL_B'] == 0:
            maj, mA, mC = _distribute_meal(param_row_dict['MEAL_TOTAL_B'], param_row_dict['MAJOR_MEAL_FRACTION'])
            g.refill_nutrient_B(maj)
            g.refill_nutrient_A(mA)
            g.refill_nutrient_C(mC)
        if step >= 2 * param_row_dict['DELTA_T'] and (step - 2 * param_row_dict['DELTA_T']) % param_row_dict['REFILL_INTERVAL_C'] == 0:
            maj, mA, mB = _distribute_meal(param_row_dict['MEAL_TOTAL_C'], param_row_dict['MAJOR_MEAL_FRACTION'])
            g.refill_nutrient_C(maj)
            g.refill_nutrient_A(mA)
            g.refill_nutrient_B(mB)

        alive = [cell for row in g.grid for cell in row if cell]
        pop_counts.append(len(alive))
        energy_levels.append(sum(max(cell.energy, 0.0) for cell in alive))
        phage_totals.append(g.total_phage_A() + g.total_phage_B() + g.total_phage_C())

        majA = sum(cell.major_type() == 'A' for cell in alive)
        majB = sum(cell.major_type() == 'B' for cell in alive)
        majC = sum(cell.major_type() == 'C' for cell in alive)
        majA_series.append(majA)
        majB_series.append(majB)
        majC_series.append(majC)

        if len(alive) == 0:
            survival_time = step
            break

    save_timeseries(sim_id, rep_id, pop_counts, phage_totals, energy_levels, majA_series, majB_series, majC_series)

    return {
        'avg_bacteria': np.mean(pop_counts),
        'avg_phage': np.mean(phage_totals),
        'avg_energy': np.mean(energy_levels),
        'survival_time': survival_time
    }

# ───────────── Worker for 1 parameter set (with replicates) ─────────────
def run_param_set(index_and_row):
    i, row = index_and_row
    sim_id = i + 1
    param_dict = row.to_dict()
    metrics = [run_replicate(sim_id, rep_id, param_dict) for rep_id in range(REPLICATES)]

    A = np.mean([m['avg_bacteria'] for m in metrics])
    P = np.mean([m['avg_phage'] for m in metrics])
    E = np.mean([m['avg_energy'] for m in metrics])
    S = np.mean([m['survival_time'] for m in metrics])

    print(f"✓ Sim {sim_id} done", flush=True)
    return [sim_id, A, P, E, S]

# ───────────── Parallel run over all parameter sets ─────────────
if __name__ == '__main__':
    total = len(param_table)
    with Pool(processes=4) as pool:
        results = []
        for count, result in enumerate(pool.imap_unordered(run_param_set, list(param_table.iterrows())), start=1):
            results.append(result)
            print(f"✓ {count} / {total} simulations done", end='\r', flush=True)

    with open('simulation_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['number_of_simulation', 'avg_bacteria', 'avg_phage', 'avg_energy', 'survival_time'])
        writer.writerows(results)

    print("\u2713 All simulations complete → simulation_summary.csv")
