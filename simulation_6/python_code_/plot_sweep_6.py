from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sweep_spec_6 import (
    SWEEP_DIR,
    TIMESERIES_DIR,
    PLOTS_DIR,
    make_dirs,
)


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")


def plot_metric(summary, y, yerr, ylabel, title, filename):
    x = range(len(summary))
    labels = [str(v) for v in summary["switch_prob"]]

    plt.figure(figsize=(8, 5))

    if yerr in summary.columns:
        plt.errorbar(
            x,
            summary[y],
            yerr=summary[yerr],
            marker="o",
            capsize=4,
        )
    else:
        plt.plot(x, summary[y], marker="o")

    plt.xticks(x, labels)
    plt.xlabel("DIVISION_SWITCH_PROB")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    savefig(PLOTS_DIR / filename)


def plot_replicate_scatter(reps, y, ylabel, title, filename):
    plt.figure(figsize=(8, 5))

    probs = sorted(reps["switch_prob"].unique())
    mapping = {p: i for i, p in enumerate(probs)}

    x = reps["switch_prob"].map(mapping)

    plt.scatter(x, reps[y], alpha=0.7)

    plt.xticks(range(len(probs)), [str(p) for p in probs])
    plt.xlabel("DIVISION_SWITCH_PROB")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    savefig(PLOTS_DIR / filename)


def load_timeseries_files():
    files = sorted(TIMESERIES_DIR.glob("*.csv"))
    rows = []

    for f in files:
        name = f.stem

        try:
            part = name.split("_rep_")
            switch_part = part[0].replace("switch_", "").replace("p", ".")
            rep = int(part[1])
            switch_prob = float(switch_part)
        except Exception:
            continue

        rows.append(
            {
                "path": f,
                "switch_prob": switch_prob,
                "replicate": rep,
            }
        )

    return pd.DataFrame(rows)


def plot_representative_population_trajectories():
    index = load_timeseries_files()

    if index.empty:
        print("No timeseries files found for trajectory plots")
        return

    plt.figure(figsize=(9, 5))

    for switch_prob, group in index.groupby("switch_prob"):
        first = group.sort_values("replicate").iloc[0]
        df = pd.read_csv(first["path"])

        if "time" not in df.columns:
            df["time"] = range(len(df))

        if "bacteria" not in df.columns:
            continue

        plt.plot(
            df["time"],
            df["bacteria"],
            label=f"p={switch_prob}",
        )

    plt.xlabel("time")
    plt.ylabel("bacteria")
    plt.title("Representative population trajectories")
    plt.legend()
    plt.grid(True, alpha=0.3)

    savefig(PLOTS_DIR / "representative_population_trajectories.png")


def plot_representative_phage_trajectories():
    index = load_timeseries_files()

    if index.empty:
        return

    plt.figure(figsize=(9, 5))

    found = False

    for switch_prob, group in index.groupby("switch_prob"):
        first = group.sort_values("replicate").iloc[0]
        df = pd.read_csv(first["path"])

        if "time" not in df.columns:
            df["time"] = range(len(df))

        if "phage" not in df.columns:
            continue

        found = True

        plt.plot(
            df["time"],
            df["phage"],
            label=f"p={switch_prob}",
        )

    if not found:
        plt.close()
        print("No phage column found; skipping phage trajectory plot")
        return

    plt.xlabel("time")
    plt.ylabel("phage")
    plt.title("Representative phage trajectories")
    plt.legend()
    plt.grid(True, alpha=0.3)

    savefig(PLOTS_DIR / "representative_phage_trajectories.png")


def plot_representative_phenotypes():
    index = load_timeseries_files()

    if index.empty:
        return

    for switch_prob, group in index.groupby("switch_prob"):
        first = group.sort_values("replicate").iloc[0]
        df = pd.read_csv(first["path"])

        required = {"majA", "majB", "majC"}

        if not required.issubset(df.columns):
            continue

        if "time" not in df.columns:
            df["time"] = range(len(df))

        plt.figure(figsize=(9, 5))
        plt.plot(df["time"], df["majA"], label="majA")
        plt.plot(df["time"], df["majB"], label="majB")
        plt.plot(df["time"], df["majC"], label="majC")

        plt.xlabel("time")
        plt.ylabel("phenotype count")
        plt.title(f"Representative phenotype trajectory, switch_prob={switch_prob}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        safe = str(switch_prob).replace(".", "p")
        savefig(PLOTS_DIR / f"phenotypes_switch_{safe}.png")


def main():
    make_dirs()

    summary_path = SWEEP_DIR / "sweep_summary.csv"
    reps_path = SWEEP_DIR / "replicate_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    if not reps_path.exists():
        raise FileNotFoundError(f"Missing {reps_path}")

    summary = pd.read_csv(summary_path).sort_values("switch_prob")
    reps = pd.read_csv(reps_path).sort_values(["switch_prob", "replicate"])

    plot_metric(
        summary,
        y="survival_time_mean",
        yerr="survival_time_sem",
        ylabel="mean survival time",
        title="Survival time vs switch probability",
        filename="survival_time_vs_switch_prob.png",
    )

    plot_metric(
        summary,
        y="mean_population_mean",
        yerr="mean_population_sem",
        ylabel="mean population",
        title="Mean population vs switch probability",
        filename="mean_population_vs_switch_prob.png",
    )

    if "extinction_rate" in summary.columns:
        plot_metric(
            summary,
            y="extinction_rate",
            yerr="extinct_sem",
            ylabel="extinction rate",
            title="Extinction rate vs switch probability",
            filename="extinction_rate_vs_switch_prob.png",
        )

    if "total_switch_attempts_mean" in summary.columns:
        plot_metric(
            summary,
            y="total_switch_attempts_mean",
            yerr="total_switch_attempts_sem",
            ylabel="mean switch attempts",
            title="Switch attempts vs switch probability",
            filename="switch_attempts_vs_switch_prob.png",
        )

    if "total_switch_successes_mean" in summary.columns:
        plot_metric(
            summary,
            y="total_switch_successes_mean",
            yerr="total_switch_successes_sem",
            ylabel="mean switch successes",
            title="Switch successes vs switch probability",
            filename="switch_successes_vs_switch_prob.png",
        )

    plot_replicate_scatter(
        reps,
        y="survival_time",
        ylabel="survival time",
        title="Replicate survival times",
        filename="replicate_survival_scatter.png",
    )

    plot_replicate_scatter(
        reps,
        y="mean_population",
        ylabel="mean population",
        title="Replicate mean populations",
        filename="replicate_population_scatter.png",
    )

    plot_representative_population_trajectories()
    plot_representative_phage_trajectories()
    plot_representative_phenotypes()


if __name__ == "__main__":
    main()