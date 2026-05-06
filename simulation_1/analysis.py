import pandas as pd
import matplotlib.pyplot as plt


# Load data (must match the CSV name that main_v2.py writes)
df = pd.read_csv("lifetime_data.csv")
print("CSV columns:", df.columns.tolist())


# Summary
print("=== Descriptive Statistics ===")
print(df.describe())
print("\n=== Head of Data ===")
print(df.head())

# Plot 1: Average lifetime with error bars
plt.figure(figsize=(10, 5))
plt.errorbar(
    df["channel_cost"],
    df["mean_lifetime"],
    yerr=df["std_lifetime"],
    fmt='-o',
    capsize=4
)
plt.xlabel("Channel Cost")
plt.ylabel("Average Lifetime")
plt.title("Average Bacterial Lifetime vs Channel Cost")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_lifetime_vs_cost.png")
plt.close()

# Plot 2: Comparison of population metrics
metrics = [
    "mean_lifetime",
    "mean_peak_pop",
    "mean_total_cells",
    "mean_area_under_pop"
]
colors = ["blue", "green", "red", "purple"]

plt.figure(figsize=(12, 6))
for metric, color in zip(metrics, colors):
    plt.plot(
        df["channel_cost"],
        df[metric],
        label=metric.replace("_", " ").title(),
        color=color
    )

plt.xlabel("Channel Cost")
plt.ylabel("Metric Value")
plt.title("Population Metrics vs Channel Cost")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_all_metrics.png")
plt.close()

# Plot 3a: Mean Number of Channels vs Channel Cost
plt.figure(figsize=(10, 5))
plt.plot(
    df["channel_cost"],
    df["mean_avg_channels"],
    marker='o',
    linestyle='-',
    color='tab:blue'
)
plt.xlabel("Channel Cost")
plt.ylabel("Mean Number of Channels")
plt.title("Mean Number of Channels vs Channel Cost")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_mean_channels_vs_cost.png")
plt.close()

# Plot 3b: Mean Total Population (Amount of Bacteria) vs Channel Cost
plt.figure(figsize=(10, 5))
plt.plot(
    df["channel_cost"],
    df["mean_total_cells"],
    marker='s',
    linestyle='-',
    color='tab:green'
)
plt.xlabel("Channel Cost")
plt.ylabel("Mean Total Population (Bacteria)")
plt.title("Mean Total Population vs Channel Cost")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_mean_total_population_vs_cost.png")
plt.close()

# Identify optimal cost
best_row = df.loc[df["mean_lifetime"].idxmax()]
print("\n=== Optimal Channel Cost for Lifetime ===")
print(best_row)

# Optional: Show all plots interactively
plt.show()
