"""
analysis_6_phase_validation.py

Post-processing for sim6_clean. This script is designed to answer the PI-style
questions:

1. Are multiple phase variants present at the same time?
2. Are minority variants present before phage peaks?
3. Does standing phase diversity predict recovery/survival?
4. Are dynamics more consistent with inherited phase variation than simple
   replacement/noise?

Run from the folder containing simulation_results.json:
    python analysis_6_phase_validation.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # scipy is optional here
    pearsonr = None
    spearmanr = None


RESULTS_JSON = "simulation_results.json"
OUTDIR = Path("analysis_6_outputs")
OUTDIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def load_results(path: str = RESULTS_JSON) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pad_ragged(runs, fill=np.nan) -> np.ndarray:
    max_len = max(len(x) for x in runs) if runs else 0
    arr = np.full((len(runs), max_len), fill, dtype=float)
    for i, run in enumerate(runs):
        arr[i, :len(run)] = np.asarray(run, dtype=float)
    return arr


def get_matrix(data: dict, key: str, fill=np.nan) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing key in JSON: {key}")
    return pad_ragged(data[key], fill=fill)


def mean_std(matrix: np.ndarray):
    return np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0)


def save_mean_std_plot(data: dict, keys, labels, title, ylabel, filename):
    plt.figure(figsize=(11, 5))
    for key, label in zip(keys, labels):
        mat = get_matrix(data, key)
        mean, sd = mean_std(mat)
        t = np.arange(len(mean))
        plt.plot(t, mean, label=label)
        plt.fill_between(t, mean - sd, mean + sd, alpha=0.2)
    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Core plots
# ---------------------------------------------------------------------

def make_core_plots(data: dict):
    save_mean_std_plot(
        data,
        ["all_pop_major_A", "all_pop_major_B", "all_pop_major_C"],
        ["A-majority", "B-majority", "C-majority"],
        "Majority phase variants over time",
        "Cell count",
        "majority_variants_over_time.png",
    )

    save_mean_std_plot(
        data,
        ["all_shannon_entropy_majority", "all_minor_variant_fraction"],
        ["Shannon entropy", "Minor variant fraction"],
        "Standing phase diversity over time",
        "Diversity metric",
        "standing_diversity_over_time.png",
    )

    save_mean_std_plot(
        data,
        ["all_switch_attempts", "all_switch_successes"],
        ["Switch attempts", "Switch successes"],
        "Realized switching events per timestep",
        "Event count",
        "switch_events_over_time.png",
    )

    save_mean_std_plot(
        data,
        ["all_infection_successes_A", "all_infection_successes_B", "all_infection_successes_C"],
        ["Infection A", "Infection B", "Infection C"],
        "Successful infection events per timestep",
        "Event count",
        "infection_successes_over_time.png",
    )

    # Total population and total phage
    pop = get_matrix(data, "all_total_population")
    phage = get_matrix(data, "all_phA") + get_matrix(data, "all_phB") + get_matrix(data, "all_phC")
    entropy = get_matrix(data, "all_shannon_entropy_majority")
    minor = get_matrix(data, "all_minor_variant_fraction")

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, mat, label in [
        (axes[0], pop, "Total population"),
        (axes[1], phage, "Total phage A+B+C"),
        (axes[2], entropy, "Shannon entropy"),
        (axes[3], minor, "Minor variant fraction"),
    ]:
        mean, sd = mean_std(mat)
        t = np.arange(len(mean))
        ax.plot(t, mean, label=label)
        ax.fill_between(t, mean - sd, mean + sd, alpha=0.2)
        ax.set_ylabel(label)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Timestep")
    fig.suptitle("sim6_clean phase-variation dashboard")
    fig.tight_layout()
    fig.savefig(OUTDIR / "phase_variation_dashboard.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------

def summarize_replicates(data: dict) -> pd.DataFrame:
    pop = get_matrix(data, "all_total_population")
    entropy = get_matrix(data, "all_shannon_entropy_majority")
    richness = get_matrix(data, "all_variant_richness")
    minor = get_matrix(data, "all_minor_variant_fraction")
    switch_success = get_matrix(data, "all_switch_successes", fill=0.0)
    switch_attempt = get_matrix(data, "all_switch_attempts", fill=0.0)

    rows = []
    for r in range(pop.shape[0]):
        alive_idx = np.where(pop[r] > 0)[0]
        survival_time = int(alive_idx[-1]) if len(alive_idx) else 0

        valid = ~np.isnan(pop[r])
        alive = pop[r] > 0
        alive_valid = valid & alive

        rows.append({
            "replicate": r,
            "survival_time": survival_time,
            "mean_population": np.nanmean(pop[r]),
            "mean_entropy_alive": np.nanmean(entropy[r, alive_valid]) if np.any(alive_valid) else np.nan,
            "mean_minor_variant_fraction_alive": np.nanmean(minor[r, alive_valid]) if np.any(alive_valid) else np.nan,
            "fraction_time_multiple_variants_alive": np.nanmean(richness[r, alive_valid] >= 2) if np.any(alive_valid) else np.nan,
            "fraction_time_three_variants_alive": np.nanmean(richness[r, alive_valid] >= 3) if np.any(alive_valid) else np.nan,
            "total_switch_attempts": np.nansum(switch_attempt[r]),
            "total_switch_successes": np.nansum(switch_success[r]),
            "realized_switch_success_rate": (
                np.nansum(switch_success[r]) / np.nansum(switch_attempt[r])
                if np.nansum(switch_attempt[r]) > 0 else np.nan
            ),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "replicate_summary.csv", index=False)
    return df


# ---------------------------------------------------------------------
# PI-style checks
# ---------------------------------------------------------------------

def pre_phage_peak_analysis(data: dict, lookback: int = 25, recovery_window: int = 50, top_k: int = 5):
    """
    For each replicate, find large total-phage peaks. Then check whether minor
    variants were already present before those peaks and whether they predict
    later population recovery.
    """
    pop = get_matrix(data, "all_total_population")
    phage = get_matrix(data, "all_phA") + get_matrix(data, "all_phB") + get_matrix(data, "all_phC")
    minor = get_matrix(data, "all_minor_variant_fraction")
    entropy = get_matrix(data, "all_shannon_entropy_majority")
    richness = get_matrix(data, "all_variant_richness")

    rows = []
    n_reps, T = phage.shape

    for r in range(n_reps):
        y = phage[r].copy()
        valid = np.isfinite(y)
        if not np.any(valid):
            continue

        # Simple local maxima.
        peaks = []
        for t in range(1, T - 1):
            if not np.isfinite(y[t]):
                continue
            if y[t] >= y[t - 1] and y[t] >= y[t + 1]:
                peaks.append((y[t], t))

        if not peaks:
            continue

        peaks = sorted(peaks, reverse=True)[:top_k]
        for peak_value, t_peak in peaks:
            t_pre = max(0, t_peak - lookback)
            t_post = min(T - 1, t_peak + recovery_window)

            rows.append({
                "replicate": r,
                "t_peak": t_peak,
                "phage_peak": peak_value,
                "t_pre": t_pre,
                "minor_pre": minor[r, t_pre],
                "entropy_pre": entropy[r, t_pre],
                "richness_pre": richness[r, t_pre],
                "population_at_peak": pop[r, t_peak],
                "population_recovery": pop[r, t_post],
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "pre_phage_peak_minor_variants.csv", index=False)

    if len(df) > 0:
        plt.figure(figsize=(7, 5))
        plt.scatter(df["minor_pre"], df["population_recovery"], alpha=0.7)
        plt.xlabel(f"Minor variant fraction {lookback} steps before phage peak")
        plt.ylabel(f"Population {recovery_window} steps after phage peak")
        plt.title("Do minority variants predict recovery after phage pressure?")
        plt.tight_layout()
        plt.savefig(OUTDIR / "minor_variants_predict_recovery.png", dpi=300)
        plt.close()

    return df


def diversity_predicts_future_population(data: dict, lag: int = 50):
    pop = get_matrix(data, "all_total_population")
    minor = get_matrix(data, "all_minor_variant_fraction")
    entropy = get_matrix(data, "all_shannon_entropy_majority")

    rows = []
    n_reps, T = pop.shape
    for r in range(n_reps):
        for t in range(0, max(0, T - lag)):
            if not np.isfinite(pop[r, t + lag]):
                continue
            rows.append({
                "replicate": r,
                "time": t,
                "minor_t": minor[r, t],
                "entropy_t": entropy[r, t],
                "population_future": pop[r, t + lag],
            })

    df = pd.DataFrame(rows).dropna()
    df.to_csv(OUTDIR / f"diversity_predicts_population_t_plus_{lag}.csv", index=False)

    summary = {}
    if len(df) > 2:
        if spearmanr is not None:
            summary["spearman_minor_vs_future_pop"] = spearmanr(df["minor_t"], df["population_future"]).correlation
            summary["spearman_entropy_vs_future_pop"] = spearmanr(df["entropy_t"], df["population_future"]).correlation
        else:
            summary["pearson_minor_vs_future_pop"] = np.corrcoef(df["minor_t"], df["population_future"])[0, 1]
            summary["pearson_entropy_vs_future_pop"] = np.corrcoef(df["entropy_t"], df["population_future"])[0, 1]

        with open(OUTDIR / f"predictive_correlations_lag_{lag}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return df, summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    data = load_results(RESULTS_JSON)
    make_core_plots(data)
    summary = summarize_replicates(data)
    peak_df = pre_phage_peak_analysis(data, lookback=25, recovery_window=50, top_k=5)
    pred_df, pred_summary = diversity_predicts_future_population(data, lag=50)

    print("Saved analysis outputs to", OUTDIR)
    print("\nReplicate summary:")
    print(summary.describe(include="all"))
    if pred_summary:
        print("\nPredictive correlation summary:")
        print(pred_summary)
    if len(peak_df):
        print("\nPre-phage peak analysis saved:", OUTDIR / "pre_phage_peak_minor_variants.csv")


if __name__ == "__main__":
    main()
