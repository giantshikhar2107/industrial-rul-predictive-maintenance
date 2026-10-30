from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "NASA_CMAPSS" / "processed"
MODEL_PATH = ROOT / "ml" / "nasa_fd001_lstm.keras"
OUTPUT_DIR = ROOT / "ml"

CSV_PATH = OUTPUT_DIR / "nasa_fd001_full_evaluation.csv"
PLOT_PATH = OUTPUT_DIR / "nasa_fd001_full_actual_vs_predicted.png"
RESIDUAL_PATH = OUTPUT_DIR / "nasa_fd001_full_residual_distribution.png"


def nasa_score(actual, predicted):

    errors = predicted - actual

    scores = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0
    )

    return float(np.sum(scores)), scores


def main():

    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    if len(X_test) != len(y_test):
        raise ValueError("Test input and target lengths do not match.")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    predictions = model.predict(
        X_test,
        verbose=0
    ).reshape(-1)

    predictions = np.maximum(predictions, 0)

    errors = predictions - y_test
    absolute_errors = np.abs(errors)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    score, engine_scores = nasa_score(y_test, predictions)

    results = pd.DataFrame({
        "engine_id": np.arange(1, len(y_test) + 1),
        "actual_rul": y_test,
        "predicted_rul": predictions,
        "prediction_error": errors,
        "absolute_error": absolute_errors,
        "nasa_score": engine_scores
    })

    results.to_csv(CSV_PATH, index=False)

    print("\nNASA C-MAPSS FD001 FULL EVALUATION")
    print("=" * 45)

    print("Test engines:", len(y_test))
    print("MAE:", round(float(mae), 3), "cycles")
    print("MSE:", round(float(mse), 3))
    print("RMSE:", round(float(rmse), 3), "cycles")
    print("R2 Score:", round(float(r2), 4))
    print("NASA Score:", round(score, 3))
    print("Mean Prediction Bias:", round(float(np.mean(errors)), 3))
    print("Maximum Absolute Error:", round(float(np.max(absolute_errors)), 3))

    print("\nACTUAL VS PREDICTED RUL")
    print("-" * 35)

    print("Mean actual RUL:", round(float(np.mean(y_test)), 2))
    print("Mean predicted RUL:", round(float(np.mean(predictions)), 2))

    print("\nFIRST 10 PREDICTIONS")
    print("-" * 35)

    print(results.head(10).round(2).to_string(index=False))

    plt.figure(figsize=(9, 7))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    minimum = min(np.min(y_test), np.min(predictions))
    maximum = max(np.max(y_test), np.max(predictions))

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title("NASA FD001: Actual vs Predicted RUL")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))

    plt.hist(
        errors,
        bins=20,
        edgecolor="black"
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel("Prediction Error (Predicted - Actual)")
    plt.ylabel("Number of Engines")
    plt.title("NASA FD001: Prediction Error Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESIDUAL_PATH, dpi=300)
    plt.close()

    print("\nOUTPUT FILES")
    print("-" * 35)
    print("Evaluation CSV:", CSV_PATH)
    print("Actual vs predicted plot:", PLOT_PATH)
    print("Residual distribution:", RESIDUAL_PATH)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()