"""
run_switch_prob_sweep_p_v2_replicates.py

Sweep DIVISION_SWITCH_PROB for simulation_p with a forced number of replicates
per switch value, then plot mean outcomes with replicate-level SD error bars.

Place this file inside the simulation_p folder, next to:
    main_p.py
    parameters_p.py
    bacteria_p.py
    grid_p.py
    phage.py

Run:
    python run_switch_prob_sweep_p_v2_replicates.py
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


# -----------------------------
# USER SETTINGS
# -----------------------------

SWITCH_VALUES = [
    0.0,
    0.001,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
]

# This is the important setting:
# main_p.py will run this many stochastic replicates for every switch value.
N_REPLICATES_PER_VALUE = 20

PARAM_SWITCH = "DIVISION_SWITCH_PROB"
PARAM_REPLICATES = "REPLICATES"

RESULT_PICKLE_NAME = "simulation_results_p.pkl"
OUTDIR_NAME = "switch_sweep_results_replicates"

STOP_ON_ERROR = True


# -----------------------------
# PATHS
# -----------------------------

ROOT = Path(__file__).resolve().parent
PARAM_FILE = ROOT / "parameters_p.py"
MAIN_FILE = ROOT / "main_p.py"
OUTDIR = ROOT / OUTDIR_NAME


# -----------------------------
# BASIC HELPERS
# -----------------------------

def require_files() -> None:
    required = [
        PARAM_FILE,
        MAIN_FILE,
        ROOT / "bacteria_p.py",
        ROOT / "grid_p.py",
        ROOT / "phage.py",
    ]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Place this script inside the simulation_p folder. "
            f"Missing files: {', '.join(missing)}"
        )


def value_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def replace_parameter(text: str, name: str, value: int | float) -> str:
    """
    Replace a simple numeric assignment like:
        REPLICATES = 10
        DIVISION_SWITCH_PROB = 0.5
    """
    pattern = rf"^({re.escape(name)}\s*=\s*)([0-9.eE+-]+)(.*)$"
    replacement = rf"\g<1>{value!r}\g<3>"
    new_text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if n != 1:
        raise ValueError(f"Could not find exactly one assignment for {name} in {PARAM_FILE}")
    return new_text


def patch_parameters(original_text: str, switch_value: float, n_reps: int) -> str:
    text = replace_parameter(original_text, PARAM_SWITCH, switch_value)
    text = replace_parameter(text, PARAM_REPLICATES, n_reps)
    return text


def remove_pycache(root: Path) -> None:
    for pycache in root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def read_int_parameter(text: str, name: str, default: int) -> int:
    match = re.search(rf"^{re.escape(name)}\s*=\s*([0-9]+)", text, flags=re.MULTILINE)
    return int(match.group(1)) if match else default


# -----------------------------
# RUN SIMULATION
# -----------------------------

def run_one_switch_value(original_param_text: str, switch_value: float, n_reps: int) -> Path:
    tag = value_tag(switch_value)
    result_path = OUTDIR / f"simulation_results_switch_{tag}_reps{n_reps}.pkl"
    log_path = OUTDIR / f"run_switch_{tag}_reps{n_reps}.log"

    print(f"\n=== Running {PARAM_SWITCH} = {switch_value} with REPLICATES = {n_reps} ===")

    PARAM_FILE.write_text(
        patch_parameters(original_param_text, switch_value, n_reps),
        encoding="utf-8",
    )

    remove_pycache(ROOT)

    stale_result = ROOT / RESULT_PICKLE_NAME
    if stale_result.exists():
        stale_result.unlink()

    env = os.environ.copy()
    # Avoid Windows cp1252 crashes if main_p.py prints Unicode arrows.
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
        msg = f"Simulation failed for {PARAM_SWITCH}={switch_value}. See log: {log_path}"
        if STOP_ON_ERROR:
            raise RuntimeError(msg)
        print("WARNING:", msg)
        return result_path

    if not stale_result.exists():
        raise FileNotFoundError(f"{MAIN_FILE.name} finished, but {RESULT_PICKLE_NAME} was not created.")

    shutil.move(str(stale_result), str(result_path))
    print(f"Saved result: {result_path}")
    return result_path


# -----------------------------
# SUMMARY METRICS
# -----------------------------

def pad_numeric_series(series: list[float], length: int, pad_value: float = 0.0) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if len(arr) >= length:
        return arr[:length]
    out = np.full(length, pad_value, dtype=float)
    out[: len(arr)] = arr
    return out


def per_rep_mean_from_dict_series(dict_series: dict[int, list[float]], length: int) -> np.ndarray:
    out = np.full(length, np.nan, dtype=float)

    for t, values in dict_series.items():
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
    """
    Late-window mean for each replicate separately.
    """
    with np.errstate(invalid="ignore"):
        return np.nanmean(matrix[:, start:], axis=1)


def mean_sd(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)), float(np.nanstd(values))


def summarize_result(result_path: Path, switch_value: float, common_length: int) -> dict:
    with open(result_path, "rb") as f:
        data = pickle.load(f)

    all_pop_A = data["all_pop_A"]
    all_pop_B = data["all_pop_B"]
    all_energy = data["all_energy"]
    all_phage_A = data["all_phage_A"]
    all_phage_B = data["all_phage_B"]
    all_index = data["all_index"]
    all_deviation = data["all_deviation"]
    all_channel_ts = data["all_channel_ts"]

    n_reps = len(all_pop_A)

    pop_matrix = np.zeros((n_reps, common_length), dtype=float)
    energy_matrix = np.zeros((n_reps, common_length), dtype=float)
    phage_matrix = np.zeros((n_reps, common_length), dtype=float)
    mean_index_matrix = np.full((n_reps, common_length), np.nan, dtype=float)
    mean_deviation_matrix = np.full((n_reps, common_length), np.nan, dtype=float)
    mean_channels_matrix = np.full((n_reps, common_length), np.nan, dtype=float)

    survival_times = []

    for r in range(n_reps):
        pop_A = pad_numeric_series(all_pop_A[r], common_length, pad_value=0.0)
        pop_B = pad_numeric_series(all_pop_B[r], common_length, pad_value=0.0)
        pop_total = pop_A + pop_B
        pop_matrix[r] = pop_total

        energy_matrix[r] = pad_numeric_series(all_energy[r], common_length, pad_value=0.0)

        phA = pad_numeric_series(all_phage_A[r], common_length, pad_value=0.0)
        phB = pad_numeric_series(all_phage_B[r], common_length, pad_value=0.0)
        phage_matrix[r] = phA + phB

        mean_index_matrix[r] = per_rep_mean_from_dict_series(all_index[r], common_length)
        mean_deviation_matrix[r] = per_rep_mean_from_dict_series(all_deviation[r], common_length)
        mean_channels_matrix[r] = per_rep_mean_from_dict_series(all_channel_ts[r], common_length)

        alive = np.where(pop_total > 0)[0]
        survival_times.append(float(alive[-1]) if len(alive) else 0.0)

    late_start = int(common_length * 0.80)

    late_pop_per_rep = nanmean_per_rep(pop_matrix, late_start)
    late_energy_per_rep = nanmean_per_rep(energy_matrix, late_start)
    late_phage_per_rep = nanmean_per_rep(phage_matrix, late_start)
    late_index_per_rep = nanmean_per_rep(mean_index_matrix, late_start)
    late_dev_per_rep = nanmean_per_rep(mean_deviation_matrix, late_start)
    late_channels_per_rep = nanmean_per_rep(mean_channels_matrix, late_start)

    mean_survival, sd_survival = mean_sd(np.asarray(survival_times))

    late_pop, sd_late_pop = mean_sd(late_pop_per_rep)
    late_energy, sd_late_energy = mean_sd(late_energy_per_rep)
    late_phage, sd_late_phage = mean_sd(late_phage_per_rep)
    late_index, sd_late_index = mean_sd(late_index_per_rep)
    late_dev, sd_late_dev = mean_sd(late_dev_per_rep)
    late_channels, sd_late_channels = mean_sd(late_channels_per_rep)

    return {
        "switch_prob": switch_value,
        "n_reps": n_reps,
        "time": np.arange(common_length),

        # time-course means
        "mean_population": np.nanmean(pop_matrix, axis=0),
        "mean_energy": np.nanmean(energy_matrix, axis=0),
        "mean_total_phage": np.nanmean(phage_matrix, axis=0),
        "mean_index": np.nanmean(mean_index_matrix, axis=0),
        "mean_deviation": np.nanmean(mean_deviation_matrix, axis=0),
        "mean_channels": np.nanmean(mean_channels_matrix, axis=0),

        # summary means and replicate-level SD
        "mean_survival_time": mean_survival,
        "sd_survival_time": sd_survival,

        "final_population": float(np.mean(pop_matrix[:, -1])),

        "late_mean_population": late_pop,
        "sd_late_mean_population": sd_late_pop,

        "late_mean_energy": late_energy,
        "sd_late_mean_energy": sd_late_energy,

        "late_mean_total_phage": late_phage,
        "sd_late_mean_total_phage": sd_late_phage,

        "late_mean_index": late_index,
        "sd_late_mean_index": sd_late_index,

        "late_mean_deviation": late_dev,
        "sd_late_mean_deviation": sd_late_dev,

        "late_mean_channels": late_channels,
        "sd_late_mean_channels": sd_late_channels,
    }


# -----------------------------
# PLOTTING
# -----------------------------

def plot_time_course(summaries: list[dict], key: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(10, 5))

    for summary in summaries:
        plt.plot(
            summary["time"],
            summary[key],
            label=f"switch={summary['switch_prob']:g}",
        )

    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.title(ylabel + " over time for different division switch probabilities")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=300)
    plt.close()


def plot_summary_metric(
    summaries: list[dict],
    mean_key: str,
    sd_key: str,
    ylabel: str,
    filename: str,
) -> None:
    x = np.asarray([s["switch_prob"] for s in summaries], dtype=float)
    y = np.asarray([s[mean_key] for s in summaries], dtype=float)
    yerr = np.asarray([s[sd_key] for s in summaries], dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    yerr = yerr[order]

    plt.figure(figsize=(7, 5))
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=1.5)
    plt.xlabel("DIVISION_SWITCH_PROB")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs division switch probability")
    plt.tight_layout()
    plt.savefig(OUTDIR / filename, dpi=300)
    plt.close()


def write_summary_csv(summaries: list[dict]) -> None:
    fields = [
        "switch_prob",
        "n_reps",
        "mean_survival_time",
        "sd_survival_time",
        "final_population",
        "late_mean_population",
        "sd_late_mean_population",
        "late_mean_energy",
        "sd_late_mean_energy",
        "late_mean_total_phage",
        "sd_late_mean_total_phage",
        "late_mean_index",
        "sd_late_mean_index",
        "late_mean_deviation",
        "sd_late_mean_deviation",
        "late_mean_channels",
        "sd_late_mean_channels",
    ]

    path = OUTDIR / "switch_sweep_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow({field: s[field] for field in fields})

    print(f"Saved summary table: {path}")


def make_all_plots(summaries: list[dict]) -> None:
    plot_time_course(
        summaries,
        "mean_population",
        "Mean total bacterial population",
        "timecourse_population_by_switch.png",
    )
    plot_time_course(
        summaries,
        "mean_deviation",
        "Mean phenotype specialization |0.5 - raw_index|",
        "timecourse_deviation_by_switch.png",
    )
    plot_time_course(
        summaries,
        "mean_index",
        "Mean raw index = A / (A + B)",
        "timecourse_raw_index_by_switch.png",
    )
    plot_time_course(
        summaries,
        "mean_channels",
        "Mean total channels per cell",
        "timecourse_channels_by_switch.png",
    )
    plot_time_course(
        summaries,
        "mean_total_phage",
        "Mean total phage A+B",
        "timecourse_total_phage_by_switch.png",
    )

    plot_summary_metric(
        summaries,
        "mean_survival_time",
        "sd_survival_time",
        "Mean survival time",
        "summary_survival_time_vs_switch.png",
    )
    plot_summary_metric(
        summaries,
        "late_mean_population",
        "sd_late_mean_population",
        "Late mean population",
        "summary_late_population_vs_switch.png",
    )
    plot_summary_metric(
        summaries,
        "late_mean_deviation",
        "sd_late_mean_deviation",
        "Late mean phenotype specialization",
        "summary_late_deviation_vs_switch.png",
    )
    plot_summary_metric(
        summaries,
        "late_mean_channels",
        "sd_late_mean_channels",
        "Late mean total channels per cell",
        "summary_late_channels_vs_switch.png",
    )
    plot_summary_metric(
        summaries,
        "late_mean_total_phage",
        "sd_late_mean_total_phage",
        "Late mean total phage A+B",
        "summary_late_total_phage_vs_switch.png",
    )


# -----------------------------
# MAIN
# -----------------------------

def main() -> None:
    require_files()
    OUTDIR.mkdir(exist_ok=True)

    original_param_text = PARAM_FILE.read_text(encoding="utf-8")
    max_steps = read_int_parameter(original_param_text, "MAX_STEPS", default=1000)

    result_paths: list[tuple[float, Path]] = []

    try:
        for switch_value in SWITCH_VALUES:
            path = run_one_switch_value(
                original_param_text,
                switch_value,
                n_reps=N_REPLICATES_PER_VALUE,
            )
            result_paths.append((switch_value, path))
    finally:
        PARAM_FILE.write_text(original_param_text, encoding="utf-8")
        remove_pycache(ROOT)
        print("\nRestored original parameters_p.py")

    summaries = [
        summarize_result(path, switch_value, common_length=max_steps)
        for switch_value, path in result_paths
        if path.exists()
    ]
    summaries.sort(key=lambda s: s["switch_prob"])

    write_summary_csv(summaries)
    make_all_plots(summaries)

    print("\nDone. Main outputs:")
    print(f"  {OUTDIR / 'switch_sweep_summary.csv'}")
    print(f"  {OUTDIR / 'timecourse_population_by_switch.png'}")
    print(f"  {OUTDIR / 'timecourse_deviation_by_switch.png'}")
    print(f"  {OUTDIR / 'summary_survival_time_vs_switch.png'}")
    print(f"  {OUTDIR / 'summary_late_deviation_vs_switch.png'}")


if __name__ == "__main__":
    main()
