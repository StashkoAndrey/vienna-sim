import matplotlib.pyplot as plt
from grid_nochannel import SimulationGrid
from diffusion_cpp import diffuse
import logging
import numpy as np
import csv

GRID_SIZE = 10
MAX_STEPS = 1000
DIFFUSION_RATE = 0.05
INITIAL_NUTRIENT = 2.0
REPLICATES = 10

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

def run_simulation():
    grid = SimulationGrid(GRID_SIZE, initial_nutrient=INITIAL_NUTRIENT)
    center = GRID_SIZE // 2
    grid.place_bacterium(center, center)

    population = []

    for step in range(MAX_STEPS):
        grid.step()
        grid.nutrients = diffuse(grid.nutrients, DIFFUSION_RATE)
        count = grid.count_bacteria()
        population.append(count)
        if count == 0:
            break

    return population


def pad_curves(curves, fill_value=0):
    max_len = max(len(c) for c in curves)
    return np.array([c + [fill_value] * (max_len - len(c)) for c in curves])


def main():
    print("Running no-channel simulation...")

    all_curves = []
    for _ in range(REPLICATES):
        curve = run_simulation()
        all_curves.append(curve)

    data = pad_curves(all_curves)
    mean_curve = np.mean(data, axis=0)
    std_curve = np.std(data, axis=0)

    steps = range(len(mean_curve))
    plt.figure(figsize=(10, 5))
    plt.plot(steps, mean_curve, label="No Channels")
    plt.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)
    plt.xlabel("Time step")
    plt.ylabel("Living Bacteria")
    plt.title("Bacterial Population Over Time (No Channels)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("nochannel_population_plot.png")

    # Write CSV
    with open("lifetime_data_nochannel.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean_population", "std_population"])
        for step, mean, std in zip(steps, mean_curve, std_curve):
            writer.writerow([step, round(mean, 3), round(std, 3)])

    print("Saved: nochannel_population_plot.png and lifetime_data_nochannel.csv")


if __name__ == '__main__':
    main()
