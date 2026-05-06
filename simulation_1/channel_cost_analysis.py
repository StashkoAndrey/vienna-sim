import os
import pandas as pd
import re
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns

def extract_optima_and_plot_grid(input_dir, output_csv, plot_output_path, top_fraction=0.01):
    """
    1. Extracts optimal channel cost from CSVs using KDE on top mean_lifetime values.
    2. Creates a multi-facet line plot of optimal_channel_cost vs D, 
       faceted by N (rows) and U (columns), colored by E.
    """

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)

    pattern = r"merged_N(?P<N>[\d.]+)_E(?P<E>[\d.]+)_U(?P<U>[\d.]+)_D(?P<D>[\d.]+).csv"
    results = []

    for filename in os.listdir(input_dir):
        if not filename.endswith(".csv"):
            continue

        match = re.match(pattern, filename)
        if not match:
            continue

        params = match.groupdict()
        n, e, u, d = float(params["N"]), float(params["E"]), float(params["U"]), float(params["D"])
        file_path = os.path.join(input_dir, filename)

        try:
            df = pd.read_csv(file_path)
            threshold = df["mean_lifetime"].quantile(1 - top_fraction)
            top_df = df[df["mean_lifetime"] >= threshold]
            channel_costs = top_df["channel_cost"].values

            if len(channel_costs) < 2:
                continue

            kde = gaussian_kde(channel_costs)
            x_grid = np.linspace(0, 1, 1000)
            densities = kde(x_grid)
            optimal_channel_cost = x_grid[np.argmax(densities)]

            results.append({
                "N": n,
                "E": e,
                "U": u,
                "D": d,
                "optimal_channel_cost": optimal_channel_cost
            })

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(output_csv, index=False)
    print(f"✅ Saved results to {output_csv}")

    # Plotting: Grid of D vs optimal cost
    g = sns.FacetGrid(summary_df, row="N", col="U", hue="E", margin_titles=True, height=3, aspect=1.3)
    g.map_dataframe(sns.lineplot, x="D", y="optimal_channel_cost")
    g.set_axis_labels("Diffusion D", "Optimal Channel Cost")
    g.add_legend(title="Expansion E")
    g.set_titles(row_template="N={row_name}", col_template="U={col_name}")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("Optimal Channel Cost vs Diffusion D\nFaceted by N (rows) and U (cols), Hue = E")
    plt.savefig(plot_output_path)
    plt.close()
    print(f"✅ Saved plot to {plot_output_path}")


extract_optima_and_plot_grid(
     input_dir="C:/Users/reawe/Desktop/Vienna/simulation/merged",
     output_csv="C:/Users/reawe/Desktop/Vienna/simulation/analysis/results.csv",
     plot_output_path="C:/Users/reawe/Desktop/Vienna/simulation/analysis/feature_importance.png",
     top_fraction=0.01
)
