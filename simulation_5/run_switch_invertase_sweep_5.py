"""
run_switch_invertase_sweep_5.py

2D parameter sweep for simulation_5:

    x-axis parameter: DIVISION_SWITCH_PROB
    y-axis parameter: INERTASE_PENALTY

The script temporarily patches parameters_5.py, runs main_5.py in a fresh
Python subprocess for every parameter pair, reads simulation_results.pkl, then
creates summary CSVs and heatmap/line plots.

Place this file inside your simulation_5 folder, next to:
    main_5.py
    parameters_5.py
    bacteria_5.py
    grid_5.py
    phage_5.py
    diffusion_cpp.cpXXX-win_amd64.pyd or diffusion_cpp.py

Run:
    python run_switch_invertase_sweep_5.py
"""

from __future__ import annotations

import csv
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

# Include 0 for the no-switch control.
SWITCH_VALUES = [
    0.0,
    0.005,
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
]

# This is the parameter called "invertase penalty" in your presentation.
# In code it is spelled INERTASE_PENALTY.
#
# 0.0  = channel building/invertase event has no extra death penalty.
# 0.02 = weak extra penalty.
# 0.05 = moderate extra penalty.
# 0.10 = original clean-baseline value.
# 0.20 = strong penalty.
INERTASE_PENALTY_VALUES = [
    0.0,
    0.02,
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
]

# main_5.py itself runs REPLICATES replicates for each parameter pair.
N_REPLICATES_PER_PAIR = 10

# Keep these fixed for all sweeps.
MAX_STEPS_OVERRIDE = 1000
LOG_STEPS_OVERRIDE = False

# Output folder.
OUTDIR_NAME = "switch_inertase_sweep_results_5"

# Stop the whole sweep if one parameter pair crashes.
STOP_ON_ERROR = True


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
PARAM_FILE = ROOT / "parameters_5.py"
MAIN_FILE = ROOT / "main_5.py"
RESULT_PICKLE_NAME = "simulation_results.pkl"
OUTDIR = ROOT / OUTDIR_NAME


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def require_files() -> None:
    required = [
        PARAM_FILE,
        MAIN_FILE,
        ROOT / "bacteria_5.py",
        ROOT / "grid_5.py",
        ROOT / "phage_5.py",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Place this script inside the simulation_5 folder. "
            f"Missing files: {', '.join(missing)}"
        )


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def replace_parameter(text: str, name: str, value: int | float | bool) -> str:
    """
    Replace simple assignments in parameters_5.py, for example:
        DIVISION_SWITCH_PROB = 0.5
        INERTASE_PENALTY = 0.1
        REPLICATES = 10
        LOG_STEPS = False
    """
    if isinstance(value, bool):
        value_str = "True" if value else "False"
    else:
        value_str = repr(value)

    pattern = rf"^({re.escape(name)}\s*=\s*)([^#\n]+)(.*)$"
    replacement = rf"\g<1>{value_str}\g<3>"

    new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"Could not find exactly one assignment for {name} in {PARAM_FILE.name}")
    return new_text


def patch_parameters(original_text: str, switch_prob: float, inertase_penalty: float) -> str:
    text = original_text
    text = replace_parameter(text, "DIVISION_SWITCH_PROB", switch_prob)
    text = replace_parameter(text, "INERTASE_PENALTY", inertase_penalty)
    text = replace_parameter(text, "REPLICATES", N_REPLICATES_PER_PAIR)
    text = replace_parameter(text, "MAX_STEPS", MAX_STEPS_OVERRIDE)
    text = replace_parameter(text, "LOG_STEPS", LOG_STEPS_OVERRIDE)
    return text


def remove_pycache(root: Path) -> None:
    for pycache in root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def pad_series(series: list[float], length: int, pad_value: float = 0.0) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if len(arr) >= length:
        return arr[:length]

    out = np.full(length, pad_value, dtype=float)
    out[: len(arr)] = arr
    return out


def dict_series_to_mean_array(dic: dict[int, list[float]], length: int) -> np.ndarray:
    out = np.full(length, np.nan, dtype=float)

    for t, values in dic.items():
        if t >= length:
            continue

        clean = []
        for v in values:
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isnan(vf):
                clean.append(vf)

        if clean:
            out[t] = float(np.mean(clean))

    return out


def nanmean_per_rep(matrix: np.ndarray, start: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(matrix[:, start:], axis=1)


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)), float(np.nanstd(values))


# ---------------------------------------------------------------------
# RUN ONE PARAMETER PAIR
# ---------------------------------------------------------------------

def run_one_pair(original_param_text: str, switch_prob: float, inertase_penalty: float) -> Path:
    switch_tag = tag(switch_prob)
    penalty_tag = tag(inertase_penalty)

    result_path = OUTDIR / (
        f"simulation_results_switch_{switch_tag}"
        f"_inertase_{penalty_tag}"
        f"_reps{N_REPLICATES_PER_PAIR}.pkl"
    )
    log_path = OUTDIR / (
        f"run_switch_{switch_tag}"
        f"_inertase_{penalty_tag}"
        f"_reps{N_REPLICATES_PER_PAIR}.log"
    )

    print(
        f"\n=== Running DIVISION_SWITCH_PROB={switch_prob:g}, "
        f"INERTASE_PENALTY={inertase_penalty:g}, "
        f"REPLICATES={N_REPLICATES_PER_PAIR} ==="
    )

    PARAM_FILE.write_text(
        patch_parameters(original_param_text, switch_prob, inertase_penalty),
        encoding="utf-8",
    )
    remove_pycache(ROOT)

    stale_result = ROOT / RESULT_PICKLE_NAME
    if stale_result.exists():
        stale_result.unlink()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        [sys.executable, str(MAIN_FILE)],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )

    log_path.write_text(completed.stdout, encoding="utf-8")

    if completed.returncode != 0:
        msg = (
            f"Simulation failed for switch={switch_prob}, "
            f"inertase={inertase_penalty}. See log: {log_path}"
        )
        if STOP_ON_ERROR:
            raise RuntimeError(msg)
        print("WARNING:", msg)
        return result_path

    if not stale_result.exists():
        raise FileNotFoundError(f"{MAIN_FILE.name} finished, but {RESULT_PICKLE_NAME} was not created.")

    shutil.move(str(stale_result), str(result_path))
    print(f"Saved result: {result_path.name}")
    return result_path


# ---------------------------------------------------------------------
# SUMMARIZE ONE RESULT PICKLE
# ---------------------------------------------------------------------

def summarize_result(result_path: Path, switch_prob: float, inertase_penalty: float, length: int) -> dict:
    with open(result_path, "rb") as f:
        data = pickle.load(f)

    all_pop_A = data["all_pop_A"]
    all_pop_B = data["all_pop_B"]
    all_pop_C = data["all_pop_C"]

    all_phage_A = data["all_phage_A"]
    all_phage_B = data["all_phage_B"]
    all_phage_C = data["all_phage_C"]

    all_energy = data["all_energy"]

    all_index = data["all_index"]
    all_deviation = data["all_deviation"]
    all_entropy = data["all_entropy"]

    n_reps = len(all_pop_A)

    pop_matrix = np.zeros((n_reps, length), dtype=float)
    phage_matrix = np.zeros((n_reps, length), dtype=float)
    energy_matrix = np.zeros((n_reps, length), dtype=float)

    index_matrix = np.full((n_reps, length), np.nan, dtype=float)
    deviation_matrix = np.full((n_reps, length), np.nan, dtype=float)
    entropy_matrix = np.full((n_reps, length), np.nan, dtype=float)

    survival_times = []

    for r in range(n_reps):
        pop_total = (
            pad_series(all_pop_A[r], length)
            + pad_series(all_pop_B[r], length)
            + pad_series(all_pop_C[r], length)
        )
        phage_total = (
            pad_series(all_phage_A[r], length)
            + pad_series(all_phage_B[r], length)
            + pad_series(all_phage_C[r], length)
        )
        energy_total = pad_series(all_energy[r], length)

        pop_matrix[r] = pop_total
        phage_matrix[r] = phage_total
        energy_matrix[r] = energy_total

        index_matrix[r] = dict_series_to_mean_array(all_index[r], length)
        deviation_matrix[r] = dict_series_to_mean_array(all_deviation[r], length)
        entropy_matrix[r] = dict_series_to_mean_array(all_entropy[r], length)

        alive = np.where(pop_total > 0)[0]
        survival_times.append(float(alive[-1]) if len(alive) else 0.0)

    late_start = int(length * 0.80)

    late_pop_per_rep = nanmean_per_rep(pop_matrix, late_start)
    late_phage_per_rep = nanmean_per_rep(phage_matrix, late_start)
    late_energy_per_rep = nanmean_per_rep(energy_matrix, late_start)
    late_index_per_rep = nanmean_per_rep(index_matrix, late_start)
    late_deviation_per_rep = nanmean_per_rep(deviation_matrix, late_start)
    late_entropy_per_rep = nanmean_per_rep(entropy_matrix, late_start)

    survival_mean, survival_sd = mean_sd(np.asarray(survival_times))

    late_pop_mean, late_pop_sd = mean_sd(late_pop_per_rep)
    late_phage_mean, late_phage_sd = mean_sd(late_phage_per_rep)
    late_energy_mean, late_energy_sd = mean_sd(late_energy_per_rep)
    late_index_mean, late_index_sd = mean_sd(late_index_per_rep)
    late_deviation_mean, late_deviation_sd = mean_sd(late_deviation_per_rep)
    late_entropy_mean, late_entropy_sd = mean_sd(late_entropy_per_rep)

    return {
        "switch_prob": switch_prob,
        "inertase_penalty": inertase_penalty,
        "n_reps": n_reps,

        "mean_survival_time": survival_mean,
        "sd_survival_time": survival_sd,

        "late_mean_population": late_pop_mean,
        "sd_late_mean_population": late_pop_sd,

        "late_mean_total_phage": late_phage_mean,
        "sd_late_mean_total_phage": late_phage_sd,

        "late_mean_energy": late_energy_mean,
        "sd_late_mean_energy": late_energy_sd,

        "late_mean_index": late_index_mean,
        "sd_late_mean_index": late_index_sd,

        "late_mean_deviation": late_deviation_mean,
        "sd_late_mean_deviation": late_deviation_sd,

        "late_mean_entropy": late_entropy_mean,
        "sd_late_mean_entropy": late_entropy_sd,

        # Time-course means for optional line plots.
        "time": np.arange(length),
        "mean_population": np.nanmean(pop_matrix, axis=0),
        "mean_total_phage": np.nanmean(phage_matrix, axis=0),
        "mean_energy": np.nanmean(energy_matrix, axis=0),
        "mean_index": np.nanmean(index_matrix, axis=0),
        "mean_deviation": np.nanmean(deviation_matrix, axis=0),
        "mean_entropy": np.nanmean(entropy_matrix, axis=0),
    }


# ---------------------------------------------------------------------
# CSV OUTPUT
# ---------------------------------------------------------------------

def write_summary_csv(summaries: list[dict]) -> None:
    fields = [
        "switch_prob",
        "inertase_penalty",
        "n_reps",
        "mean_survival_time",
        "sd_survival_time",
        "late_mean_population",
        "sd_late_mean_population",
        "late_mean_total_phage",
        "sd_late_mean_total_phage",
        "late_mean_energy",
        "sd_late_mean_energy",
        "late_mean_index",
        "sd_late_mean_index",
        "late_mean_deviation",
        "sd_late_mean_deviation",
        "late_mean_entropy",
        "sd_late_mean_entropy",
    ]

    out = OUTDIR / "switch_inertase_sweep_summary.csv"

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow({field: s[field] for field in fields})

    print(f"Saved summary CSV: {out}")


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def pivot_metric(summaries: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values = np.array(sorted({s["switch_prob"] for s in summaries}), dtype=float)
    y_values = np.array(sorted({s["inertase_penalty"] for s in summaries}), dtype=float)

    Z = np.full((len(y_values), len(x_values)), np.nan, dtype=float)

    for s in summaries:
        i = np.where(y_values == s["inertase_penalty"])[0][0]
        j = np.where(x_values == s["switch_prob"])[0][0]
        Z[i, j] = s[metric]

    return x_values, y_values, Z


def plot_heatmap(summaries: list[dict], metric: str, title: str, filename: str) -> None:
    x_values, y_values, Z = pivot_metric(summaries, metric)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    im = ax.imshow(Z, aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{x:g}" for x in x_values], rotation=45, ha="right")

    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([f"{y:g}" for y in y_values])

    ax.set_xlabel("DIVISION_SWITCH_PROB")
    ax.set_ylabel("INERTASE_PENALTY")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    # Annotate cells for quick slide reading.
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isfinite(Z[i, j]):
                ax.text(j, i, f"{Z[i, j]:.2g}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUTDIR / filename, dpi=300)
    plt.close(fig)


def plot_lines_by_penalty(summaries: list[dict], metric: str, ylabel: str, filename: str) -> None:
    x_values = sorted({s["switch_prob"] for s in summaries})
    y_values = sorted({s["inertase_penalty"] for s in summaries})

    fig, ax = plt.subplots(figsize=(8, 5))

    for penalty in y_values:
        ys = []
        for switch in x_values:
            matches = [
                s for s in summaries
                if s["switch_prob"] == switch and s["inertase_penalty"] == penalty
            ]
            ys.append(matches[0][metric] if matches else np.nan)

        ax.plot(x_values, ys, marker="o", label=f"inertase={penalty:g}")

    ax.set_xlabel("DIVISION_SWITCH_PROB")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " across switch probabilities")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / filename, dpi=300)
    plt.close(fig)


def plot_timecourse_grid(
    summaries: list[dict],
    metric: str,
    ylabel: str,
    filename: str,
    selected_penalties: list[float] | None = None,
) -> None:
    """
    Time-course line plots grouped by inertase penalty.
    Useful as an optional diagnostic, not always slide-friendly.
    """
    penalties = sorted({s["inertase_penalty"] for s in summaries})
    if selected_penalties is not None:
        penalties = [p for p in penalties if p in selected_penalties]

    switches = sorted({s["switch_prob"] for s in summaries})

    n = len(penalties)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3.2 * n, 4)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, penalty in zip(axes, penalties):
        for switch in switches:
            matches = [
                s for s in summaries
                if s["switch_prob"] == switch and s["inertase_penalty"] == penalty
            ]
            if not matches:
                continue
            s = matches[0]
            ax.plot(s["time"], s[metric], label=f"switch={switch:g}", linewidth=1.2)

        ax.set_title(f"INERTASE_PENALTY = {penalty:g}")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, ncol=2)

    axes[-1].set_xlabel("Timestep")
    fig.tight_layout()
    fig.savefig(OUTDIR / filename, dpi=300)
    plt.close(fig)


def make_all_plots(summaries: list[dict]) -> None:
    heat_specs = [
        ("late_mean_population", "Late mean bacterial population", "heatmap_late_population.png"),
        ("mean_survival_time", "Mean survival time", "heatmap_survival_time.png"),
        ("late_mean_total_phage", "Late mean total phage A+B+C", "heatmap_late_total_phage.png"),
        ("late_mean_entropy", "Late mean Shannon entropy of channel composition", "heatmap_late_entropy.png"),
        ("late_mean_deviation", "Late mean |0.5 - IA|", "heatmap_late_deviation.png"),
        ("late_mean_energy", "Late mean bacterial energy", "heatmap_late_energy.png"),
    ]

    for metric, title, filename in heat_specs:
        plot_heatmap(summaries, metric, title, filename)

    line_specs = [
        ("late_mean_population", "Late mean bacterial population", "lines_late_population_by_penalty.png"),
        ("mean_survival_time", "Mean survival time", "lines_survival_time_by_penalty.png"),
        ("late_mean_total_phage", "Late mean total phage A+B+C", "lines_late_total_phage_by_penalty.png"),
        ("late_mean_entropy", "Late mean Shannon entropy", "lines_late_entropy_by_penalty.png"),
        ("late_mean_deviation", "Late mean |0.5 - IA|", "lines_late_deviation_by_penalty.png"),
    ]

    for metric, ylabel, filename in line_specs:
        plot_lines_by_penalty(summaries, metric, ylabel, filename)

    # Optional timecourse diagnostics for the most interpretable penalties.
    selected = [0.0, 0.1, 0.2]
    plot_timecourse_grid(
        summaries,
        "mean_population",
        "Mean bacterial population",
        "timecourse_population_selected_penalties.png",
        selected_penalties=selected,
    )
    plot_timecourse_grid(
        summaries,
        "mean_entropy",
        "Mean Shannon entropy",
        "timecourse_entropy_selected_penalties.png",
        selected_penalties=selected,
    )
    plot_timecourse_grid(
        summaries,
        "mean_total_phage",
        "Mean total phage A+B+C",
        "timecourse_phage_selected_penalties.png",
        selected_penalties=selected,
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:
    require_files()
    OUTDIR.mkdir(exist_ok=True)

    original_param_text = PARAM_FILE.read_text(encoding="utf-8")

    result_paths: list[tuple[float, float, Path]] = []

    try:
        for inertase_penalty in INERTASE_PENALTY_VALUES:
            for switch_prob in SWITCH_VALUES:
                path = run_one_pair(original_param_text, switch_prob, inertase_penalty)
                result_paths.append((switch_prob, inertase_penalty, path))
    finally:
        PARAM_FILE.write_text(original_param_text, encoding="utf-8")
        remove_pycache(ROOT)
        print("\nRestored original parameters_5.py")

    summaries = [
        summarize_result(path, switch_prob, inertase_penalty, MAX_STEPS_OVERRIDE)
        for switch_prob, inertase_penalty, path in result_paths
        if path.exists()
    ]

    summaries.sort(key=lambda s: (s["inertase_penalty"], s["switch_prob"]))

    write_summary_csv(summaries)
    make_all_plots(summaries)

    print("\nDone. Main outputs:")
    print(f"  {OUTDIR / 'switch_inertase_sweep_summary.csv'}")
    print(f"  {OUTDIR / 'heatmap_late_population.png'}")
    print(f"  {OUTDIR / 'heatmap_late_entropy.png'}")
    print(f"  {OUTDIR / 'heatmap_late_total_phage.png'}")
    print(f"  {OUTDIR / 'lines_late_population_by_penalty.png'}")


if __name__ == "__main__":
    main()

