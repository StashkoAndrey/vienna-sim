# === master_main.py ===
import subprocess
import itertools
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Parameter Grid ===
nutrients = [2.88]
energies = [1.0, 2.0, 3.0, 4.0, 5.0]
uptakes = [0.5, 1.0, 1.5]
diffusion_rates = [0.01, 0.05, 0.1]

BATCH_REPEATS = 50
MAX_WORKERS = 4  # Adjust based on number of CPU cores

combinations = list(itertools.product(nutrients, energies, uptakes, diffusion_rates))

# === Output Directory ===
os.makedirs("batch_results", exist_ok=True)

# === Run Tracker ===
start_time = time.time()
total_runs = len(combinations) * BATCH_REPEATS

def run_simulation_task(init_nutrient, energy, uptake, diffusion_rate, repeat, run_counter):
    label = f"N{init_nutrient}_E{energy}_U{uptake}_D{diffusion_rate}_R{repeat}"
    print(f"\n[{run_counter}/{total_runs}] Running: {label}")

    output_dir = os.path.join("batch_results", label)
    os.makedirs(output_dir, exist_ok=True)

    command = [
        "python", "main.py",
        "--init_nutrient", str(init_nutrient),
        "--start_energy", str(energy),
        "--start_uptake", str(uptake),
        "--diffusion_rate", str(diffusion_rate),
        "--label", label,
        "--output_dir", output_dir
    ]

    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Run failed for: {label}")

    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed/60:.2f} min")

# === Parallel Execution ===
run_counter = 0
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []
    for repeat in range(1, BATCH_REPEATS + 1):
        for (init_nutrient, energy, uptake, diffusion_rate) in combinations:
            run_counter += 1
            futures.append(
                executor.submit(
                    run_simulation_task,
                    init_nutrient, energy, uptake, diffusion_rate, repeat, run_counter
                )
            )

    # Optional: wait and show errors if any
    for future in as_completed(futures):
        if future.exception():
            print("Exception occurred:", future.exception())

print("\n✅ All runs completed.")
