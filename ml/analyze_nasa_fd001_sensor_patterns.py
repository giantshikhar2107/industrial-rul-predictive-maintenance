from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = ROOT / "data" / "NASA_CMAPSS" / "test_FD001.txt"
EVALUATION_PATH = ROOT / "ml" / "nasa_fd001_full_evaluation.csv"

OUTPUT_PATH = ROOT / "ml" / "nasa_fd001_sensor_pattern_analysis.csv"

columns = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

raw = pd.read_csv(
    RAW_DATA_PATH,
    sep=r"\s+",
    header=None,
    names=columns
)

evaluation = pd.read_csv(EVALUATION_PATH)

high_error = evaluation.nlargest(5, "nasa_score")
low_error = evaluation.nsmallest(5, "absolute_error")

high_ids = high_error["engine_id"].astype(int).tolist()
low_ids = low_error["engine_id"].astype(int).tolist()

sensor_columns = [f"sensor_{i}" for i in range(1, 22)]
setting_columns = [f"setting_{i}" for i in range(1, 4)]

def calculate_engine_features(engine_data):

    engine_data = engine_data.sort_values("cycle")

    result = {
        "engine_id": int(engine_data["engine_id"].iloc[0]),
        "cycles_recorded": len(engine_data),
        "last_cycle": int(engine_data["cycle"].max())
    }

    for column in setting_columns + sensor_columns:

        values = engine_data[column].to_numpy(dtype=float)
        cycles = engine_data["cycle"].to_numpy(dtype=float)

        result[f"{column}_mean"] = np.mean(values)
        result[f"{column}_std"] = np.std(values)
        result[f"{column}_change"] = values[-1] - values[0]

        if len(values) > 1:
            result[f"{column}_slope"] = np.polyfit(
                cycles, values, 1
            )[0]
        else:
            result[f"{column}_slope"] = 0.0

    return result


engine_features = pd.DataFrame([
    calculate_engine_features(group)
    for _, group in raw.groupby("engine_id")
])

engine_features = engine_features.merge(
    evaluation,
    on="engine_id",
    how="inner"
)

engine_features["prediction_group"] = np.where(
    engine_features["engine_id"].isin(high_ids),
    "High error",
    np.where(
        engine_features["engine_id"].isin(low_ids),
        "Low error",
        "Other"
    )
)

print("\nNASA FD001 SENSOR PATTERN ANALYSIS")
print("=" * 60)

print("\n1. FIVE HIGHEST-ERROR ENGINES")
print("-" * 45)

print(
    high_error[
        ["engine_id", "actual_rul", "predicted_rul",
         "absolute_error", "nasa_score"]
    ].round(2).to_string(index=False)
)

print("\n2. FIVE LOWEST-ERROR ENGINES")
print("-" * 45)

print(
    low_error[
        ["engine_id", "actual_rul", "predicted_rul",
         "absolute_error", "nasa_score"]
    ].round(2).to_string(index=False)
)

print("\n3. SENSOR TRENDS FOR HIGH-ERROR ENGINES")
print("-" * 45)

high_features = engine_features[
    engine_features["engine_id"].isin(high_ids)
]

trend_columns = [
    f"{sensor}_slope" for sensor in sensor_columns
]

high_trends = high_features[
    ["engine_id"] + trend_columns
].copy()

print(high_trends.round(5).to_string(index=False))

print("\n4. SENSOR TRENDS FOR LOW-ERROR ENGINES")
print("-" * 45)

low_features = engine_features[
    engine_features["engine_id"].isin(low_ids)
]

low_trends = low_features[
    ["engine_id"] + trend_columns
].copy()

print(low_trends.round(5).to_string(index=False))

print("\n5. LARGEST DIFFERENCES IN SENSOR TRENDS")
print("-" * 45)

high_mean = high_features[trend_columns].mean()
low_mean = low_features[trend_columns].mean()

comparison = pd.DataFrame({
    "high_error_mean_slope": high_mean,
    "low_error_mean_slope": low_mean
})

comparison["slope_difference"] = (
    comparison["high_error_mean_slope"]
    - comparison["low_error_mean_slope"]
)

comparison["absolute_difference"] = (
    comparison["slope_difference"].abs()
)

comparison = comparison.sort_values(
    "absolute_difference",
    ascending=False
)

print(comparison.head(10).round(6).to_string())

print("\n6. ENGINE CYCLE INFORMATION")
print("-" * 45)

print(
    engine_features[
        engine_features["engine_id"].isin(high_ids + low_ids)
    ][
        ["engine_id", "cycles_recorded", "last_cycle",
         "actual_rul", "predicted_rul", "absolute_error"]
    ].sort_values("absolute_error", ascending=False)
    .round(2)
    .to_string(index=False)
)

engine_features.to_csv(OUTPUT_PATH, index=False)

print("\n7. OUTPUT")
print("-" * 45)
print("Complete sensor analysis saved to:")
print(OUTPUT_PATH)

print("\nSensor pattern analysis completed successfully.")