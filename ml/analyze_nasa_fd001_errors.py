from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT / "ml" / "nasa_fd001_full_evaluation.csv"

df = pd.read_csv(CSV_PATH)

df["error_category"] = np.select(
    [
        df["absolute_error"] <= 10,
        df["absolute_error"] <= 20,
        df["absolute_error"] <= 30
    ],
    [
        "0-10 cycles",
        "11-20 cycles",
        "21-30 cycles"
    ],
    default="Above 30 cycles"
)

print("\nNASA FD001 ENGINE-WISE ERROR ANALYSIS")
print("=" * 55)

print("\n1. ERROR DISTRIBUTION")
print("-" * 40)

print(df["error_category"].value_counts().reindex(
    ["0-10 cycles", "11-20 cycles", "21-30 cycles", "Above 30 cycles"],
    fill_value=0
))

print("\n2. ENGINES WITH ERROR GREATER THAN 20 CYCLES")
print("-" * 40)

high_error = df[df["absolute_error"] > 20].sort_values(
    "absolute_error",
    ascending=False
)

print("Number of engines:", len(high_error))

print(high_error[
    ["engine_id", "actual_rul", "predicted_rul", "prediction_error", "absolute_error"]
].round(2).to_string(index=False))

print("\n3. TOP 10 ENGINES WITH LARGEST ERRORS")
print("-" * 40)

print(
    df.nlargest(10, "absolute_error")[
        ["engine_id", "actual_rul", "predicted_rul", "prediction_error", "absolute_error", "nasa_score"]
    ].round(2).to_string(index=False)
)

print("\n4. ERROR ANALYSIS BY ACTUAL RUL RANGE")
print("-" * 40)

df["rul_range"] = pd.cut(
    df["actual_rul"],
    bins=[-np.inf, 30, 60, 90, 120, np.inf],
    labels=["0-30", "31-60", "61-90", "91-120", "Above 120"]
)

summary = df.groupby(
    "rul_range",
    observed=False
).agg(
    engines=("engine_id", "count"),
    MAE=("absolute_error", "mean"),
    Mean_Bias=("prediction_error", "mean"),
    NASA_Score=("nasa_score", "sum")
)

print(summary.round(2))

print("\n5. OVERPREDICTION VS UNDERPREDICTION")
print("-" * 40)

over = df[df["prediction_error"] > 0]
under = df[df["prediction_error"] < 0]
exact = df[df["prediction_error"] == 0]

print("Overpredicted engines:", len(over))
print("Underpredicted engines:", len(under))
print("Exact predictions:", len(exact))

print("\n6. CONTRIBUTION TO TOTAL NASA SCORE")
print("-" * 40)

print(
    df.nlargest(10, "nasa_score")[
        ["engine_id", "actual_rul", "predicted_rul", "prediction_error", "nasa_score"]
    ].round(2).to_string(index=False)
)

print("\n7. FINAL SUMMARY")
print("-" * 40)

print("Total engines:", len(df))
print("Mean absolute error:", round(df["absolute_error"].mean(), 3))
print("Mean prediction bias:", round(df["prediction_error"].mean(), 3))
print("Total NASA score:", round(df["nasa_score"].sum(), 3))

print("\nError analysis completed successfully.")