import numpy as np
import pandas as pd

from sweep_spec_6 import SWEEP_DIR, ANALYSIS_DIR, make_dirs


def pooled_cohens_d(x, y):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx = len(x)
    ny = len(y)

    sx = x.std(ddof=1)
    sy = y.std(ddof=1)

    pooled_sd = np.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2))

    if pooled_sd == 0:
        return np.nan

    return (x.mean() - y.mean()) / pooled_sd


def main():
    make_dirs()

    replicate_path = SWEEP_DIR / "replicate_summary.csv"
    summary_path = SWEEP_DIR / "sweep_summary.csv"

    if not replicate_path.exists():
        raise FileNotFoundError(f"Missing {replicate_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")

    reps = pd.read_csv(replicate_path)
    summary = pd.read_csv(summary_path)

    baseline = reps[reps["switch_prob"] == 0.0]

    if baseline.empty:
        raise ValueError("No baseline switch_prob == 0.0 found")

    rows = []

    for switch_prob, group in reps.groupby("switch_prob"):
        row = {
            "switch_prob": switch_prob,
            "n": len(group),
            "mean_survival_time": group["survival_time"].mean(),
            "mean_population": group["mean_population"].mean(),
            "extinction_rate": group["extinct"].mean(),
            "delta_survival_vs_zero": (
                group["survival_time"].mean()
                - baseline["survival_time"].mean()
            ),
            "delta_mean_population_vs_zero": (
                group["mean_population"].mean()
                - baseline["mean_population"].mean()
            ),
            "cohens_d_survival_vs_zero": pooled_cohens_d(
                group["survival_time"],
                baseline["survival_time"],
            ),
            "cohens_d_population_vs_zero": pooled_cohens_d(
                group["mean_population"],
                baseline["mean_population"],
            ),
        }

        if "total_switch_attempts" in group.columns:
            row["mean_switch_attempts"] = group["total_switch_attempts"].mean()

        if "total_switch_successes" in group.columns:
            row["mean_switch_successes"] = group["total_switch_successes"].mean()

        rows.append(row)

    effects = pd.DataFrame(rows).sort_values("switch_prob")

    effects_path = ANALYSIS_DIR / "effects_vs_zero.csv"
    effects.to_csv(effects_path, index=False)

    best_survival = effects.sort_values(
        ["mean_survival_time", "mean_population"],
        ascending=[False, False],
    ).iloc[0]

    best_population = effects.sort_values(
        ["mean_population", "mean_survival_time"],
        ascending=[False, False],
    ).iloc[0]

    lowest_extinction = effects.sort_values(
        ["extinction_rate", "mean_survival_time"],
        ascending=[True, False],
    ).iloc[0]

    text = []
    text.append("SIM6 sweep analysis")
    text.append("===================")
    text.append("")
    text.append(f"Best survival switch_prob: {best_survival['switch_prob']}")
    text.append(f"  mean survival time: {best_survival['mean_survival_time']:.3f}")
    text.append(f"  mean population: {best_survival['mean_population']:.3f}")
    text.append(f"  extinction rate: {best_survival['extinction_rate']:.3f}")
    text.append("")
    text.append(f"Best population switch_prob: {best_population['switch_prob']}")
    text.append(f"  mean population: {best_population['mean_population']:.3f}")
    text.append(f"  mean survival time: {best_population['mean_survival_time']:.3f}")
    text.append(f"  extinction rate: {best_population['extinction_rate']:.3f}")
    text.append("")
    text.append(f"Lowest extinction switch_prob: {lowest_extinction['switch_prob']}")
    text.append(f"  extinction rate: {lowest_extinction['extinction_rate']:.3f}")
    text.append(f"  mean survival time: {lowest_extinction['mean_survival_time']:.3f}")
    text.append("")
    text.append("Interpretation rule:")
    text.append("  Good switch_prob values should increase survival and population")
    text.append("  without requiring unrealistically high switch counts.")

    report_path = ANALYSIS_DIR / "recommendation.txt"
    report_path.write_text("\n".join(text), encoding="utf-8")

    print(effects)
    print()
    print(f"Saved {effects_path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()