from pathlib import Path
import json
import re
import shutil

import subprocess
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Sweep settings
# =========================

SWITCH_PROBS = [
    0.0,
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    0.5,
    1.0,
]

MAX_STEPS = 1000
REPLICATES = 20

LOOKBACK = 25
RECOVERY_WINDOW = 50
TOP_K_PEAKS = 3

PROJECT_DIR = Path(".").resolve()
PARAM_FILE = PROJECT_DIR / "parameters_6.hpp"

OUTDIR = PROJECT_DIR / "sweep_switch_model_comparison_6"
RAW_DIR = OUTDIR / "raw_json"
PLOTS_DIR = OUTDIR / "plots"

EXE_NAME = "sim6_clean.exe"
EXE = PROJECT_DIR / EXE_NAME
RESULT_JSON = PROJECT_DIR / "simulation_results.json"

MINGW_BIN = r"C:\msys64\ucrt64\bin"
GXX = rf"{MINGW_BIN}\g++.exe"

CPP_FILES = [
    "main_6.cpp",
    "simulation_grid_6.cpp",
    "bacteria_6.cpp",
    "phage_6.cpp",
    "diffusion_6.cpp",
    "simulation_metrics_6.cpp",
]


# =========================
# Helpers
# =========================

def make_dirs():
    OUTDIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)


def safe_label(x):
    return f"{x:.6g}".replace(".", "p")


def cpp_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        s = f"{value:.8g}"

        # C++ does not like 0f or 1f as normal float literals.
        # It needs 0.0f / 1.0f or just 0.0 / 1.0.
        if "." not in s and "e" not in s.lower():
            s = s + ".0"

        return s

    return str(value)


def patch_parameter(text, name, value):
    value_str = cpp_value(value)

    pattern = rf"(constexpr\s+(?:int|float|double|bool)\s+{name}\s*=\s*)[^;]+(;)"
    new_text, n = re.subn(pattern, rf"\g<1>{value_str}\2", text)

    if n == 0:
        print(f"WARNING: did not find parameter {name}")

    return new_text

def write_parameters(switch_prob):
    text = PARAM_FILE.read_text(encoding="utf-8")

    text = patch_parameter(text, "DIVISION_SWITCH_PROB", switch_prob)
    text = patch_parameter(text, "MAX_STEPS", MAX_STEPS)
    text = patch_parameter(text, "REPLICATES", REPLICATES)
    text = patch_parameter(text, "LOG_STEPS", False)

    PARAM_FILE.write_text(text, encoding="utf-8")

    print("\nPatched parameters:")
    for name in ["DIVISION_SWITCH_PROB", "MAX_STEPS", "REPLICATES", "LOG_STEPS"]:
        m = re.search(rf".*{name}.*", text)
        if m:
            print(m.group(0))
        else:
            print(f"{name}: NOT FOUND")
    print()

def compiler_env():
    env = os.environ.copy()
    env["PATH"] = MINGW_BIN + os.pathsep + env.get("PATH", "")
    return env

def compile_sim():
    if EXE.exists():
        try:
            EXE.unlink()
        except PermissionError:
            raise RuntimeError(
                "Cannot overwrite sim6_clean.exe. "
                "Close it if it is still running, or delete it manually."
            )

    cmd = [
        GXX,
        "-std=c++17",
        "-O2",
        *CPP_FILES,
        "-I.",
        "-Iexternal",
        "-o",
        EXE_NAME,
    ]

    print("Compiling:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        env=compiler_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed with return code {result.returncode}")

    if not EXE.exists():
        raise FileNotFoundError("Compilation finished but sim6_clean.exe was not created")


def run_sim():
    if RESULT_JSON.exists():
        RESULT_JSON.unlink()

    subprocess.run(
        [str(EXE)],
        cwd=PROJECT_DIR,
        env=compiler_env(),
        check=True,
    )

    if not RESULT_JSON.exists():
        raise FileNotFoundError("simulation_results.json was not created")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pad_ragged(runs, fill=np.nan):
    if not runs:
        return np.empty((0, 0))

    max_len = max(len(x) for x in runs)
    arr = np.full((len(runs), max_len), fill, dtype=float)

    for i, run in enumerate(runs):
        arr[i, :len(run)] = np.asarray(run, dtype=float)

    return arr


def get_matrix(data, key, fill=np.nan):
    return pad_ragged(data[f"all_{key}"], fill=fill)


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmean(x))


def safe_sum(x):
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return 0.0
    return float(np.nansum(x))


def corr_spearman(x, y):
    df = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(df) < 3:
        return np.nan

    if df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan

    return float(df["x"].corr(df["y"], method="spearman"))


def linear_fit_stats(x, y):
    df = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(df) < 3:
        return np.nan, np.nan

    if df["x"].nunique() < 2:
        return np.nan, np.nan

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    return float(slope), float(r2)


# =========================
# Analysis
# =========================

def summarize_replicates(data, switch_prob):
    pop = get_matrix(data, "total_population")
    minor = get_matrix(data, "minor_variant_fraction")
    entropy = get_matrix(data, "shannon_entropy_majority")
    richness = get_matrix(data, "variant_richness")
    switch_success = get_matrix(data, "switch_successes", fill=0)
    switch_attempt = get_matrix(data, "switch_attempts", fill=0)

    rows = []

    for r in range(pop.shape[0]):
        alive = pop[r] > 0
        alive_idx = np.where(alive)[0]

        survival_time = int(alive_idx[-1]) if len(alive_idx) else 0

        rows.append({
            "switch_prob": switch_prob,
            "replicate": r,
            "survival_time": survival_time,
            "mean_population": safe_mean(pop[r]),
            "final_population": float(pop[r, alive_idx[-1]]) if len(alive_idx) else 0.0,
            "mean_minor_alive": safe_mean(minor[r, alive]),
            "mean_entropy_alive": safe_mean(entropy[r, alive]),
            "fraction_time_multiple_variants_alive": safe_mean(richness[r, alive] >= 2),
            "fraction_time_three_variants_alive": safe_mean(richness[r, alive] >= 3),
            "total_switch_attempts": safe_sum(switch_attempt[r]),
            "total_switch_successes": safe_sum(switch_success[r]),
        })

    return pd.DataFrame(rows)


def phage_peak_events(data, switch_prob):
    pop = get_matrix(data, "total_population")
    minor = get_matrix(data, "minor_variant_fraction")
    entropy = get_matrix(data, "shannon_entropy_majority")
    richness = get_matrix(data, "variant_richness")

    phage = (
        get_matrix(data, "phA")
        + get_matrix(data, "phB")
        + get_matrix(data, "phC")
    )

    rows = []

    n_reps, T = phage.shape

    for r in range(n_reps):
        y = phage[r]

        peaks = []

        for t in range(1, T - 1):
            if not np.isfinite(y[t]):
                continue

            if t < LOOKBACK:
                continue

            if t + RECOVERY_WINDOW >= T:
                continue

            if y[t] >= y[t - 1] and y[t] >= y[t + 1]:
                peaks.append((y[t], t))

        peaks = sorted(peaks, reverse=True)[:TOP_K_PEAKS]

        for peak_value, t_peak in peaks:
            t_pre = t_peak - LOOKBACK
            t_post = t_peak + RECOVERY_WINDOW

            pop_at_peak = pop[r, t_peak]
            pop_after = pop[r, t_post]

            rows.append({
                "switch_prob": switch_prob,
                "replicate": r,
                "t_peak": t_peak,
                "phage_peak": peak_value,
                "minor_pre": minor[r, t_pre],
                "entropy_pre": entropy[r, t_pre],
                "richness_pre": richness[r, t_pre],
                "population_at_peak": pop_at_peak,
                "population_after": pop_after,
                "recovery_gain": pop_after - pop_at_peak,
                "recovery_ratio": (pop_after + 1.0) / (pop_at_peak + 1.0),
            })

    return pd.DataFrame(rows)


def summarize_models(rep_df, peak_df):
    rows = []

    for p in sorted(rep_df["switch_prob"].unique()):
        r = rep_df[rep_df["switch_prob"] == p]
        e = peak_df[peak_df["switch_prob"] == p]

        if len(e) > 0:
            slope, r2 = linear_fit_stats(e["minor_pre"], e["recovery_gain"])
            spearman = corr_spearman(e["minor_pre"], e["recovery_gain"])
        else:
            slope, r2, spearman = np.nan, np.nan, np.nan

        rows.append({
            "switch_prob": p,
            "n_replicates": len(r),
            "n_peak_events": len(e),
            "mean_survival_time": r["survival_time"].mean(),
            "mean_population": r["mean_population"].mean(),
            "mean_minor_alive": r["mean_minor_alive"].mean(),
            "mean_entropy_alive": r["mean_entropy_alive"].mean(),
            "mean_fraction_time_multiple_variants": r["fraction_time_multiple_variants_alive"].mean(),
            "mean_switch_successes": r["total_switch_successes"].mean(),
            "mean_recovery_gain": e["recovery_gain"].mean() if len(e) else np.nan,
            "spearman_minor_predicts_recovery_gain": spearman,
            "slope_minor_predicts_recovery_gain": slope,
            "r2_minor_predicts_recovery_gain": r2,
        })

    return pd.DataFrame(rows)


# =========================
# Plotting
# =========================

def plot_recovery_scatter(peak_df):
    plt.figure(figsize=(9, 6))

    for p, g in peak_df.groupby("switch_prob"):
        plt.scatter(
            g["minor_pre"],
            g["recovery_gain"],
            alpha=0.6,
            label=f"p={p:g}",
        )

        if len(g) >= 3 and g["minor_pre"].nunique() >= 2:
            slope, intercept = np.polyfit(g["minor_pre"], g["recovery_gain"], 1)
            xs = np.linspace(g["minor_pre"].min(), g["minor_pre"].max(), 100)
            ys = slope * xs + intercept
            plt.plot(xs, ys, linewidth=2)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel(f"Minor variant fraction {LOOKBACK} steps before phage peak")
    plt.ylabel(f"Recovery gain: population +{RECOVERY_WINDOW} steps minus population at peak")
    plt.title("Does standing phase diversity predict recovery after phage pressure?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_minor_pre_predicts_recovery_gain.png", dpi=300)
    plt.close()


def plot_metric(summary, y, ylabel, filename):
    s = summary.sort_values("switch_prob")
    x = np.arange(len(s))
    labels = [f"{v:g}" for v in s["switch_prob"]]

    plt.figure(figsize=(8, 5))
    plt.plot(x, s[y], marker="o")
    plt.xticks(x, labels)
    plt.xlabel("DIVISION_SWITCH_PROB")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs switch probability")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()


def plot_mean_trajectories(json_paths):
    for metric, ylabel, filename in [
        ("total_population", "Mean population", "05_mean_population_trajectories.png"),
        ("minor_variant_fraction", "Mean minor variant fraction", "06_mean_minor_fraction_trajectories.png"),
    ]:
        plt.figure(figsize=(9, 5))

        for p, path in json_paths.items():
            data = load_json(path)
            mat = get_matrix(data, metric)
            mean = np.nanmean(mat, axis=0)
            t = np.arange(len(mean))
            plt.plot(t, mean, label=f"p={p:g}")

        plt.xlabel("Timestep")
        plt.ylabel(ylabel)
        plt.title(ylabel + " by switch probability")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename, dpi=300)
        plt.close()

    plt.figure(figsize=(9, 5))

    for p, path in json_paths.items():
        data = load_json(path)
        phage = get_matrix(data, "phA") + get_matrix(data, "phB") + get_matrix(data, "phC")
        mean = np.nanmean(phage, axis=0)
        t = np.arange(len(mean))
        plt.plot(t, mean, label=f"p={p:g}")

    plt.xlabel("Timestep")
    plt.ylabel("Mean total phage")
    plt.title("Mean phage pressure by switch probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_mean_phage_trajectories.png", dpi=300)
    plt.close()


# =========================
# Main sweep
# =========================

def main():
    make_dirs()

    original_parameters = PARAM_FILE.read_text(encoding="utf-8")

    all_replicates = []
    all_peaks = []
    json_paths = {}

    try:
        for p in SWITCH_PROBS:
            label = safe_label(p)

            print("\n" + "=" * 70)
            print(f"Running DIVISION_SWITCH_PROB = {p}")
            print("=" * 70)

            write_parameters(p)
            compile_sim()
            run_sim()

            out_json = RAW_DIR / f"simulation_results_switch_{label}.json"
            shutil.copy2(RESULT_JSON, out_json)
            json_paths[p] = out_json

            data = load_json(out_json)

            rep_df = summarize_replicates(data, p)
            peak_df = phage_peak_events(data, p)

            rep_df.to_csv(RAW_DIR / f"replicate_summary_switch_{label}.csv", index=False)
            peak_df.to_csv(RAW_DIR / f"peak_events_switch_{label}.csv", index=False)

            all_replicates.append(rep_df)
            all_peaks.append(peak_df)

        replicate_df = pd.concat(all_replicates, ignore_index=True)
        peak_df = pd.concat(all_peaks, ignore_index=True)

        model_summary = summarize_models(replicate_df, peak_df)

        replicate_df.to_csv(OUTDIR / "all_replicates.csv", index=False)
        peak_df.to_csv(OUTDIR / "all_peak_events.csv", index=False)
        model_summary.to_csv(OUTDIR / "model_comparison_summary.csv", index=False)

        plot_recovery_scatter(peak_df)

        plot_metric(
            model_summary,
            "r2_minor_predicts_recovery_gain",
            "R²: minor fraction predicts recovery gain",
            "02_predictive_r2_vs_switch_probability.png",
        )

        plot_metric(
            model_summary,
            "spearman_minor_predicts_recovery_gain",
            "Spearman: minor fraction vs recovery gain",
            "03_spearman_vs_switch_probability.png",
        )

        plot_metric(
            model_summary,
            "mean_survival_time",
            "Mean survival time",
            "04_survival_time_vs_switch_probability.png",
        )

        plot_metric(
            model_summary,
            "mean_fraction_time_multiple_variants",
            "Fraction of alive time with multiple variants",
            "08_multiple_variant_time_vs_switch_probability.png",
        )

        plot_mean_trajectories(json_paths)

        best = model_summary.sort_values(
            ["r2_minor_predicts_recovery_gain", "mean_survival_time"],
            ascending=[False, False],
        ).iloc[0]

        report = []
        report.append("SIM6 division switch probability sweep")
        report.append("====================================")
        report.append("")
        report.append(f"Best predictive switch_prob by R²: {best['switch_prob']}")
        report.append(f"R² minor_pre -> recovery_gain: {best['r2_minor_predicts_recovery_gain']:.4f}")
        report.append(f"Spearman minor_pre vs recovery_gain: {best['spearman_minor_predicts_recovery_gain']:.4f}")
        report.append(f"Mean survival time: {best['mean_survival_time']:.2f}")
        report.append(f"Mean switch successes: {best['mean_switch_successes']:.2f}")
        report.append("")
        report.append("Interpretation:")
        report.append("p = 0 is the no-division-switching/null model.")
        report.append("A useful phase-variation model should show:")
        report.append("1. nonzero realized switching,")
        report.append("2. coexistence of multiple variants,")
        report.append("3. minority fraction before phage peaks predicting later recovery gain.")
        report.append("")
        report.append("Main files:")
        report.append("model_comparison_summary.csv")
        report.append("all_peak_events.csv")
        report.append("plots/01_minor_pre_predicts_recovery_gain.png")
        report.append("plots/02_predictive_r2_vs_switch_probability.png")

        (OUTDIR / "README_results.txt").write_text("\n".join(report), encoding="utf-8")

        print("\nDONE")
        print(model_summary)
        print(f"\nSaved everything to: {OUTDIR}")

    finally:
        PARAM_FILE.write_text(original_parameters, encoding="utf-8")
        print("\nRestored original parameters_6.hpp")


if __name__ == "__main__":
    main()