from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "ml" / "nasa_fd001_test_predictions.csv"
OUTPUT_DIR = ROOT / "ml"

if not RESULTS_PATH.exists():
    raise FileNotFoundError(f"Results file not found: {RESULTS_PATH}")

df = pd.read_csv(RESULTS_PATH)

required_columns = {
    "engine_id",
    "actual_rul",
    "predicted_rul",
    "absolute_error",
}

if not required_columns.issubset(df.columns):
    raise ValueError("The prediction results file has missing columns.")

if df.empty:
    raise ValueError("The prediction results file is empty.")

if df[list(required_columns)].isna().any().any():
    raise ValueError("The prediction results file contains missing values.")

plt.figure(figsize=(8, 6))
plt.scatter(df["actual_rul"], df["predicted_rul"], alpha=0.8)

minimum = min(df["actual_rul"].min(), df["predicted_rul"].min())
maximum = max(df["actual_rul"].max(), df["predicted_rul"].max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
)

plt.xlabel("Actual RUL (cycles)")
plt.ylabel("Predicted RUL (cycles)")
plt.title("NASA FD001: Actual vs Predicted RUL")
plt.tight_layout()

actual_predicted_path = OUTPUT_DIR / "nasa_fd001_actual_vs_predicted.png"
plt.savefig(actual_predicted_path, dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(11, 5))
signed_error = df["predicted_rul"] - df["actual_rul"]

plt.bar(df["engine_id"], signed_error)
plt.axhline(0, linestyle="--")
plt.xlabel("Engine ID")
plt.ylabel("Prediction Error (cycles)")
plt.title("NASA FD001: Prediction Error by Engine")
plt.tight_layout()

error_path = OUTPUT_DIR / "nasa_fd001_prediction_error.png"
plt.savefig(error_path, dpi=300)
plt.show()
plt.close()

print("Actual vs predicted plot saved:", actual_predicted_path)
print("Prediction error plot saved:", error_path)

