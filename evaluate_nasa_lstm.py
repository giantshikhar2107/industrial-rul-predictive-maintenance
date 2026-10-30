from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "NASA_CMAPSS" / "processed"
MODEL_PATH = ROOT / "ml" / "nasa_fd001_lstm.keras"
OUTPUT_DIR = ROOT / "ml"

RUL_CAP = 125


def load_array(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    array = np.load(path, allow_pickle=False)

    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values found in {filename}")

    return array


def nasa_score(actual, predicted):
    errors = predicted - actual
    scores = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0,
    )
    return float(np.sum(scores))


def main():
    X_test = load_array("X_test.npy")
    y_test = load_array("y_test.npy")

    if len(X_test) != len(y_test):
        raise ValueError("Test inputs and targets have different lengths.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)

    predictions = model.predict(X_test, verbose=0).reshape(-1)

    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    score = nasa_score(y_test, predictions)

    results = pd.DataFrame({
        "engine_id": np.arange(1, len(y_test) + 1),
        "actual_rul": y_test,
        "predicted_rul": predictions,
        "absolute_error": np.abs(y_test - predictions),
    })

    results_path = OUTPUT_DIR / "nasa_fd001_test_predictions.csv"
    results.to_csv(results_path, index=False)

    print("\nNASA C-MAPSS FD001 TEST EVALUATION")
    print("=" * 42)
    print("Test engines:", len(y_test))
    print("MAE:", round(float(mae), 3), "cycles")
    print("RMSE:", round(float(rmse), 3), "cycles")
    print("NASA score:", round(score, 3))
    print("Mean actual RUL:", round(float(np.mean(y_test)), 3))
    print("Mean predicted RUL:", round(float(np.mean(predictions)), 3))
    print("Prediction results saved:", results_path)

    print("\nFirst 10 predictions:")
    print(results.head(10).round(2).to_string(index=False))


if __name__ == "__main__":
    main()

