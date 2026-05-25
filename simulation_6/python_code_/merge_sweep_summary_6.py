from pathlib import Path

import pandas as pd

from sweep_spec_6 import (
    SWEEP_DIR,
    REPLICATE_SUMMARY_DIR,
    ANALYSIS_DIR,
    make_dirs,
)


def flatten_columns(columns):
    out = []

    for col in columns:
        if isinstance(col, tuple):
            clean = "_".join(str(x) for x in col if x)
            out.append(clean)
        else:
            out.append(col)

    return out


def main():
    make_dirs()

    files = sorted(REPLICATE_SUMMARY_DIR.glob("*_summary.csv"))

    if not files:
        raise FileNotFoundError(
            f"No replicate summary files found in {REPLICATE_SUMMARY_DIR}"
        )

    print(f"Found {len(files)} replicate summary files")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    replicate_out = SWEEP_DIR / "replicate_summary.csv"
    df.to_csv(replicate_out, index=False)

    numeric_cols = [
        c for c in df.columns
        if c not in ["replicate", "seed"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    agg_dict = {}

    for col in numeric_cols:
        if col == "switch_prob":
            continue

        agg_dict[col] = ["mean", "std", "min", "max"]

    grouped = df.groupby("switch_prob").agg(agg_dict)
    grouped.columns = flatten_columns(grouped.columns)
    grouped = grouped.reset_index()

    counts = df.groupby("switch_prob").size().reset_index(name="n_replicates")
    grouped = grouped.merge(counts, on="switch_prob", how="left")

    if "extinct_mean" in grouped.columns:
        grouped = grouped.rename(columns={"extinct_mean": "extinction_rate"})

    for col in list(grouped.columns):
        if col.endswith("_std"):
            base = col[:-4]
            mean_col = f"{base}_mean"
            sem_col = f"{base}_sem"

            if mean_col in grouped.columns:
                grouped[sem_col] = grouped[col] / grouped["n_replicates"] ** 0.5

    out_path = SWEEP_DIR / "sweep_summary.csv"
    grouped.to_csv(out_path, index=False)

    analysis_copy = ANALYSIS_DIR / "sweep_summary.csv"
    grouped.to_csv(analysis_copy, index=False)

    print(f"Saved {replicate_out}")
    print(f"Saved {out_path}")

    print()
    print(grouped)


if __name__ == "__main__":
    main()