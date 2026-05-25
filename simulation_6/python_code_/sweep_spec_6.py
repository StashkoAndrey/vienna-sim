from pathlib import Path

SWEEP_NAME = "sweep_div_switch_6"
SWEEP_DIR = Path(SWEEP_NAME)

TIMESERIES_DIR = SWEEP_DIR / "timeseries"
REPLICATE_SUMMARY_DIR = SWEEP_DIR / "replicate_summaries"
ANALYSIS_DIR = SWEEP_DIR / "analysis"
PLOTS_DIR = SWEEP_DIR / "plots"

MAX_STEPS = 100
REPLICATES = 1

SWEEP_PARAMETER = "DIVISION_SWITCH_PROB"

SWITCH_PROBS = [
    0.0,
]

BASE_SEED = 12345


def make_dirs():
    for path in [
        SWEEP_DIR,
        TIMESERIES_DIR,
        REPLICATE_SUMMARY_DIR,
        ANALYSIS_DIR,
        PLOTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def safe_prob_label(x: float) -> str:
    return f"{x:.6g}".replace(".", "p")