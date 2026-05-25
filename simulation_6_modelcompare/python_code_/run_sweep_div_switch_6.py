import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from sweep_spec_6 import (
    SWEEP_PARAMETER,
    SWITCH_PROBS,
    MAX_STEPS,
    REPLICATES,
    BASE_SEED,
    TIMESERIES_DIR,
    REPLICATE_SUMMARY_DIR,
    SWEEP_DIR,
    make_dirs,
    safe_prob_label,
)

from sim_adapter_6 import run_single_sim_6


def total_from_series(series: pd.Series) -> float:
    values = series.dropna()

    if values.empty:
        return np.nan

    diffs = values.diff().dropna()

    # If cumulative log, take last value.
    if len(diffs) > 0 and (diffs >= 0).all():
        return float(values.iloc[-1])

    # If per-step log, sum values.
    return float(values.sum())


def summarize_timeseries(
    csv_path: Path,
    switch_prob: float,
    replicate: int,
    seed: int,
) -> dict:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "time" not in df.columns:
        df["time"] = np.arange(len(df))

    if "bacteria" not in df.columns:
        raise ValueError(f"{csv_path} has no 'bacteria' column")

    bacteria = df["bacteria"].fillna(0)

    alive_df = df[bacteria > 0]

    if len(alive_df) == 0:
        survival_time = 0
    else:
        survival_time = float(alive_df["time"].max())

    final_population = float(bacteria.iloc[-1])
    extinct = int(final_population <= 0)

    out = {
        "switch_prob": switch_prob,
        "replicate": replicate,
        "seed": seed,
        "survival_time": survival_time,
        "extinct": extinct,
        "final_population": final_population,
        "mean_population": float(bacteria.mean()),
        "max_population": float(bacteria.max()),
        "min_population": float(bacteria.min()),
    }

    if len(df) >= 2:
        out["population_auc"] = float(np.trapz(bacteria, df["time"]))
    else:
        out["population_auc"] = np.nan

    for col in ["phage", "energy", "majA", "majB", "majC"]:
        if col in df.columns:
            out[f"mean_{col}"] = float(df[col].mean())
            out[f"final_{col}"] = float(df[col].iloc[-1])
            out[f"max_{col}"] = float(df[col].max())

    if {"majA", "majB", "majC"}.issubset(df.columns):
        maj_cols = ["majA", "majB", "majC"]
        dominance = df[maj_cols].max(axis=1)
        total_major = df[maj_cols].sum(axis=1).replace(0, np.nan)
        out["mean_dominance_ratio"] = float((dominance / total_major).mean())

    for col in ["switch_attempts", "switch_successes"]:
        if col in df.columns:
            out[f"total_{col}"] = total_from_series(df[col])
        else:
            out[f"total_{col}"] = 0.0

    return out


def main():
    make_dirs()

    all_summaries = []
    errors = []

    for switch_prob in SWITCH_PROBS:
        label = safe_prob_label(switch_prob)

        for rep in range(REPLICATES):
            seed = BASE_SEED + rep + int(switch_prob * 1_000_000)

            params = {
                SWEEP_PARAMETER: switch_prob,
            }

            timeseries_path = TIMESERIES_DIR / f"switch_{label}_rep_{rep:03d}.csv"
            summary_path = REPLICATE_SUMMARY_DIR / f"switch_{label}_rep_{rep:03d}_summary.csv"

            print(f"Running switch_prob={switch_prob}, rep={rep}, seed={seed}")

            try:
                run_single_sim_6(
                    params=params,
                    seed=seed,
                    max_steps=MAX_STEPS,
                    timeseries_path=timeseries_path,
                )

                summary = summarize_timeseries(
                    csv_path=timeseries_path,
                    switch_prob=switch_prob,
                    replicate=rep,
                    seed=seed,
                )

                pd.DataFrame([summary]).to_csv(summary_path, index=False)
                all_summaries.append(summary)

            except Exception as exc:
                error = {
                    "switch_prob": switch_prob,
                    "replicate": rep,
                    "seed": seed,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(error)
                print(f"ERROR at switch_prob={switch_prob}, rep={rep}")
                print(repr(exc))

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(
            SWEEP_DIR / "replicate_summary.csv",
            index=False,
        )

    if errors:
        pd.DataFrame(errors).to_csv(
            SWEEP_DIR / "errors.csv",
            index=False,
        )

    print("Sweep finished.")
    print(f"Successful runs: {len(all_summaries)}")
    print(f"Failed runs: {len(errors)}")


if __name__ == "__main__":
    main()