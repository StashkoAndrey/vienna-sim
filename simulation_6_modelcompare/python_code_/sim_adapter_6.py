from pathlib import Path
import json
import re
import shutil
import subprocess
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent

PARAMETERS_HPP = PROJECT_DIR / "parameters_6.hpp"
BUILD_DIR = PROJECT_DIR / "build"
EXE_PATH = BUILD_DIR / "Debug" / "sim.exe"
RESULT_JSON = PROJECT_DIR / "simulation_results.json"


def replace_numeric_parameter(text: str, name: str, value) -> str:
    """
    Replaces common C++ parameter styles, for example:

        constexpr double DIVISION_SWITCH_PROB = 0.01;
        const double DIVISION_SWITCH_PROB = 0.01;
        #define DIVISION_SWITCH_PROB 0.01

    If the parameter is not found, text is returned unchanged.
    """

    value_str = str(value)

    patterns = [
        (
            rf"(constexpr\s+(?:double|float|int|size_t)\s+{name}\s*=\s*)[^;]+(;)",
            rf"\g<1>{value_str}\2",
        ),
        (
            rf"(const\s+(?:double|float|int|size_t)\s+{name}\s*=\s*)[^;]+(;)",
            rf"\g<1>{value_str}\2",
        ),
        (
            rf"(#define\s+{name}\s+)[^\n]+",
            rf"\g<1>{value_str}",
        ),
    ]

    new_text = text

    for pattern, replacement in patterns:
        new_text2 = re.sub(pattern, replacement, new_text)

        if new_text2 != new_text:
            return new_text2

    print(f"WARNING: parameter {name} not found in parameters_6.hpp")
    return text


def patch_parameters(params: dict, seed: int, max_steps: int):
    if not PARAMETERS_HPP.exists():
        raise FileNotFoundError(f"Cannot find {PARAMETERS_HPP}")

    text = PARAMETERS_HPP.read_text(encoding="utf-8")

    for key, value in params.items():
        text = replace_numeric_parameter(text, key, value)

    # These are optional. If your header has them, they will be patched.
    # If not, warnings are harmless.
    text = replace_numeric_parameter(text, "RANDOM_SEED", seed)
    text = replace_numeric_parameter(text, "SEED", seed)
    text = replace_numeric_parameter(text, "MAX_STEPS", max_steps)

    PARAMETERS_HPP.write_text(text, encoding="utf-8")


def build_sim():
    if not BUILD_DIR.exists():
        raise FileNotFoundError(
            f"Missing build directory: {BUILD_DIR}. "
            "Run CMake configure once before the sweep."
        )

    cmd = [
        "cmake",
        "--build",
        str(BUILD_DIR),
        "--config",
        "Debug",
    ]

    subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        check=True,
    )

    if not EXE_PATH.exists():
        raise FileNotFoundError(f"Build finished but sim.exe not found at {EXE_PATH}")


def run_executable():
    if RESULT_JSON.exists():
        RESULT_JSON.unlink()

    subprocess.run(
        [str(EXE_PATH)],
        cwd=PROJECT_DIR,
        check=True,
    )

    if not RESULT_JSON.exists():
        raise FileNotFoundError(
            f"Simulation finished but did not create {RESULT_JSON}"
        )


def normalize_json_to_dataframe(data) -> pd.DataFrame:
    """
    Tries to convert common simulation_results.json layouts into a timeseries table.

    Supported examples:

    1. [
           {"time": 0, "bacteria": 100, ...},
           {"time": 1, "bacteria": 98, ...}
       ]

    2. {
           "time": [...],
           "bacteria": [...],
           "phage": [...]
       }

    3. {
           "timeseries": [
               {"time": 0, "bacteria": 100, ...}
           ]
       }

    4. {
           "results": [...]
       }
    """

    if isinstance(data, list):
        return pd.DataFrame(data)

    if isinstance(data, dict):
        for key in ["timeseries", "time_series", "results", "data"]:
            if key in data:
                return normalize_json_to_dataframe(data[key])

        list_like = {
            key: value
            for key, value in data.items()
            if isinstance(value, list)
        }

        if list_like:
            return pd.DataFrame(list_like)

    raise ValueError(
        "Could not understand simulation_results.json structure. "
        "Open the JSON and check whether it contains a timeseries list/table."
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames likely C++ output names into the names expected by the sweep.
    """

    rename_map = {
        "step": "time",
        "t": "time",
        "population": "bacteria",
        "bacteria_count": "bacteria",
        "num_bacteria": "bacteria",
        "phage_count": "phage",
        "num_phage": "phage",
        "total_energy": "energy",
        "majority_A": "majA",
        "majority_B": "majB",
        "majority_C": "majC",
        "maj_a": "majA",
        "maj_b": "majB",
        "maj_c": "majC",
    }

    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    if "time" not in df.columns:
        df.insert(0, "time", range(len(df)))

    if "bacteria" not in df.columns:
        raise ValueError(
            "Converted JSON has no bacteria/population column. "
            f"Available columns: {list(df.columns)}"
        )

    keep_first = ["time", "bacteria", "phage", "energy", "majA", "majB", "majC"]
    ordered = [c for c in keep_first if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]

    return df[ordered + rest]


def json_to_timeseries_csv(timeseries_path: Path):
    with open(RESULT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = normalize_json_to_dataframe(data)
    df = standardize_columns(df)

    timeseries_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(timeseries_path, index=False)


def run_single_sim_6(
    params: dict,
    seed: int,
    max_steps: int,
    timeseries_path: Path,
):
    """
    Adapter for the C++ SIM6 executable.

    The sweep passes:
        params = {"DIVISION_SWITCH_PROB": value}

    This function:
        1. patches parameters_6.hpp
        2. rebuilds sim.exe
        3. runs sim.exe
        4. converts simulation_results.json to the sweep CSV format
    """

    patch_parameters(
        params=params,
        seed=seed,
        max_steps=max_steps,
    )

    build_sim()
    run_executable()
    json_to_timeseries_csv(timeseries_path)