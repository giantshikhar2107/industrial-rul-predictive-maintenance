from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = ROOT / "data" / "NASA_CMAPSS" / "test_FD001.txt"
EVAL_PATH = ROOT / "ml" / "nasa_fd001_full_evaluation.csv"

OUTPUT_PATH = ROOT / "ml" / "nasa_fd001_last_window_analysis.csv"

columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

raw = pd.read_csv(
    RAW_PATH,
    sep=r"\s+",
    header=None,
    names=columns
)

evaluation = pd.read_csv(EVAL_PATH)

high_ids = evaluation.nlargest(
    5, "absolute_error"
)["engine_id"].astype(int).tolist()

low_ids = evaluation.nsmallest(
    5, "absolute_error"
)["engine_id"].astype(int).tolist()

sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

records = []

for engine_id in high_ids + low_ids:

    engine = raw[raw["engine_id"] == engine_id].sort_values("cycle")

    window = engine.tail(30)

    result = {
        "engine_id": engine_id,
        "group": "High error" if engine_id in high_ids else "Low error",
        "cycles_in_window": len(window)
    }

    for sensor in sensor_columns:

        values = window[sensor].to_numpy(dtype=float)

        result[f"{sensor}_mean"] = np.mean(values)
        result[f"{sensor}_std"] = np.std(values)
        result[f"{sensor}_change"] = values[-1] - values[0]

    records.append(result)

features = pd.DataFrame(records)

features.to_csv(OUTPUT_PATH, index=False)

print("\nNASA FD001 FINAL 30-CYCLE WINDOW ANALYSIS")
print("=" * 60)

print("\n1. ENGINE GROUPS")
print("-" * 40)

print(features[
    ["engine_id", "group", "cycles_in_window"]
].to_string(index=False))

for group in ["High error", "Low error"]:

    subset = features[features["group"] == group]

    print(f"\n2. {group.upper()} GROUP SENSOR SUMMARY")
    print("-" * 45)

    for sensor in sensor_columns:

        mean_value = subset[f"{sensor}_mean"].mean()
        std_value = subset[f"{sensor}_std"].mean()
        change_value = subset[f"{sensor}_change"].mean()

        print(
            f"{sensor}: "
            f"mean={mean_value:.5f}, "
            f"variability={std_value:.5f}, "
            f"window_change={change_value:.5f}"
        )

print("\n3. SENSOR DIFFERENCES BETWEEN GROUPS")
print("-" * 45)

high = features[features["group"] == "High error"]
low = features[features["group"] == "Low error"]

differences = []

for sensor in sensor_columns:

    differences.append({
        "sensor": sensor,
        "mean_difference": (
            high[f"{sensor}_mean"].mean()
            - low[f"{sensor}_mean"].mean()
        ),
        "variability_difference": (
            high[f"{sensor}_std"].mean()
            - low[f"{sensor}_std"].mean()
        ),
        "window_change_difference": (
            high[f"{sensor}_change"].mean()
            - low[f"{sensor}_change"].mean()
        )
    })

comparison = pd.DataFrame(differences)

comparison["absolute_mean_difference"] = (
    comparison["mean_difference"].abs()
)

print(
    comparison.sort_values(
        "absolute_mean_difference",
        ascending=False
    ).round(5).to_string(index=False)
)

print("\nOutput saved to:")
print(OUTPUT_PATH)

print("\nAnalysis completed successfully.")
