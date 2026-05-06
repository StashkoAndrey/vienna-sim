# ======= master_main_v5.py =======
import csv, os
import numpy as np
import multiprocessing as mp
from parameters_5 import (
    GRID_SIZE, CHANNEL_COST, DIFFUSION_RATE,
    PHAGE_A_DIFFUSION_RATE, PHAGE_B_DIFFUSION_RATE, PHAGE_C_DIFFUSION_RATE,
    REFILL_INTERVAL_A, REFILL_INTERVAL_B, REFILL_INTERVAL_C,
    DELTA_T, MEAL_TOTAL_A, MEAL_TOTAL_B, MEAL_TOTAL_C,
    MAJOR_MEAL_FRACTION
)
from grid_5 import SimulationGrid
from diffusion_cpp import diffuse

# Sweep config
ENERGY_MIN = 0.5
ENERGY_MAX = 5.0
ENERGY_STEP = 0.5
SWEEP_REPLICATES = 10
SWEEP_MAX_STEPS = 1000
OUTPUT_FILE = "energy_sweep_results.csv"

def _distribute_meal(total):
    major = total * MAJOR_MEAL_FRACTION
    remainder = total - major
    m1 = np.random.rand() * remainder
    m2 = remainder - m1
    return major, m1, m2

def run_replicate(start_energy):
    g = SimulationGrid(GRID_SIZE, CHANNEL_COST)
    centre = GRID_SIZE // 2
    g.place_bacterium(centre, centre, start_energy=start_energy)

    total_cells = []
    total_energy = []
    lifetimes = {}
    dead_ages = []
    extinction_time = SWEEP_MAX_STEPS

    for step in range(SWEEP_MAX_STEPS):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell = g.grid[i][j]
                if cell and cell.id not in lifetimes:
                    lifetimes[cell.id] = step

        g.step(step)

        alive = [cell for row in g.grid for cell in row if cell]
        total_cells.append(len(alive))
        total_energy.append(sum(cell.energy for cell in alive))

        for cid in list(lifetimes.keys()):
            if not any(cell for row in g.grid for cell in row if cell and cell.id == cid):
                age = step - lifetimes.pop(cid)
                dead_ages.append(age)

        g.nutrients_A = diffuse(g.nutrients_A, DIFFUSION_RATE)
        g.nutrients_B = diffuse(g.nutrients_B, DIFFUSION_RATE)
        g.nutrients_C = diffuse(g.nutrients_C, DIFFUSION_RATE)
        g.phages_A    = diffuse(g.phages_A, PHAGE_A_DIFFUSION_RATE)
        g.phages_B    = diffuse(g.phages_B, PHAGE_B_DIFFUSION_RATE)
        g.phages_C    = diffuse(g.phages_C, PHAGE_C_DIFFUSION_RATE)

        if step and step % REFILL_INTERVAL_A == 0:
            maj, mB, mC = _distribute_meal(MEAL_TOTAL_A)
            g.refill_nutrient_A(maj)
            g.refill_nutrient_B(mB)
            g.refill_nutrient_C(mC)
        if step and step >= DELTA_T and (step - DELTA_T) % REFILL_INTERVAL_B == 0:
            maj, mA, mC = _distribute_meal(MEAL_TOTAL_B)
            g.refill_nutrient_B(maj)
            g.refill_nutrient_A(mA)
            g.refill_nutrient_C(mC)
        if step and step >= 2 * DELTA_T and (step - 2 * DELTA_T) % REFILL_INTERVAL_C == 0:
            maj, mA, mB = _distribute_meal(MEAL_TOTAL_C)
            g.refill_nutrient_C(maj)
            g.refill_nutrient_A(mA)
            g.refill_nutrient_B(mB)

        if len(alive) == 0:
            extinction_time = step
            break
    return {
        'avg_biomass': np.mean(total_cells),
        'avg_energy': np.mean(total_energy),
        'avg_lifetime': np.mean(dead_ages) if dead_ages else 0.0,
        'extinction_time': extinction_time
    }

def run_energy_point(e):
    reps = [run_replicate(e) for _ in range(SWEEP_REPLICATES)]
    avg_biomass = np.mean([r['avg_biomass'] for r in reps])
    avg_energy = np.mean([r['avg_energy'] for r in reps])
    avg_lifetime = np.mean([r['avg_lifetime'] for r in reps])
    mean_extinction = np.mean([r['extinction_time'] for r in reps])
    return (e, avg_biomass, avg_energy, avg_lifetime, mean_extinction)

def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    e_vals = np.arange(ENERGY_MIN, ENERGY_MAX + ENERGY_STEP, ENERGY_STEP)

    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        results = pool.map(run_energy_point, e_vals)

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['START_ENERGY', 'AVG_BIOMASS', 'AVG_ENERGY', 'AVG_LIFETIME', 'EXTINCTION_TIME'])
        for row in results:
            writer.writerow(row)
            print(f"✓ {row[0]:.2f}", end=' ', flush=True)

    print("\nSweep complete →", OUTPUT_FILE)

if __name__ == '__main__':
    main()
