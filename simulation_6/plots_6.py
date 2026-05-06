import json
import matplotlib.pyplot as plt
import numpy as np

# Load simulation data
with open("C:/Users/reawe/Desktop/Vienna/simulation_6/build/Debug/simulation_results.json") as f:
    data = json.load(f)

# Helper to compute mean ± std across replicates
def mean_std(arrays):
    stacked = np.array(arrays)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    return mean, std

# Time points
T = len(data["all_pop_A"][0])
time = np.arange(T)

# Plot populations
plt.figure(figsize=(12, 6))
for label in ["A", "B", "C"]:
    mean, std = mean_std(data[f"all_pop_{label}"])
    plt.plot(time, mean, label=f"{label} population")
    plt.fill_between(time, mean - std, mean + std, alpha=0.3)
plt.xlabel("Time step")
plt.ylabel("Population size")
plt.title("Bacterial Populations Over Time")
plt.legend()
plt.tight_layout()
plt.show()

# Plot majority-channel counts
plt.figure(figsize=(12, 6))
for label in ["A", "B", "C"]:
    mean, std = mean_std(data[f"all_pop_major_{label}"])
    plt.plot(time, mean, label=f"Majority {label}")
    plt.fill_between(time, mean - std, mean + std, alpha=0.3)
plt.xlabel("Time step")
plt.ylabel("Count")
plt.title("Cells by Channel Dominance")
plt.legend()
plt.tight_layout()
plt.show()

# Plot phage levels
plt.figure(figsize=(12, 6))
for label in ["A", "B", "C"]:
    mean, std = mean_std(data[f"all_ph{label}"])
    plt.plot(time, mean, label=f"Phage {label}")
    plt.fill_between(time, mean - std, mean + std, alpha=0.3)
plt.xlabel("Time step")
plt.ylabel("Total Phage Concentration")
plt.title("Phage Concentration Over Time")
plt.legend()
plt.tight_layout()
plt.show()

# Plot system energy
plt.figure(figsize=(12, 4))
mean, std = mean_std(data["all_energy"])
plt.plot(time, mean, label="Total energy", color="black")
plt.fill_between(time, mean - std, mean + std, color="gray", alpha=0.3)
plt.xlabel("Time step")
plt.ylabel("Total energy")
plt.title("System Energy Over Time")
plt.tight_layout()
plt.show()
