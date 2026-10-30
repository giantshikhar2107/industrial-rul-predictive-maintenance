from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "NASA_CMAPSS" / "processed"
MODEL_DIR = ROOT

MODEL_PATH = MODEL_DIR / "nasa_fd001_lstm_uncapped.keras"
PREDICTION_PATH = MODEL_DIR / "nasa_fd001_uncapped_test_predictions.npy"
HISTORY_PATH = MODEL_DIR / "nasa_fd001_uncapped_training_history.npz"

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_array(filename):
    path = DATA_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Check that NASA FD001 preprocessing has been completed."
        )

    array = np.load(path, allow_pickle=False)

    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values found in {filename}")

    return array


def validate_arrays(X_train, y_train, X_validation, y_validation, X_test, y_test):

    if X_train.ndim != 3 or X_validation.ndim != 3 or X_test.ndim != 3:
        raise ValueError("All input arrays must have three dimensions.")

    if X_train.shape[1:] != X_validation.shape[1:]:
        raise ValueError("Training and validation input shapes do not match.")

    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError("Training and test input shapes do not match.")

    if len(X_train) != len(y_train):
        raise ValueError("Training input and target lengths differ.")

    if len(X_validation) != len(y_validation):
        raise ValueError("Validation input and target lengths differ.")

    if len(X_test) != len(y_test):
        raise ValueError("Test input and target lengths differ.")

    if min(len(X_train), len(X_validation), len(X_test)) == 0:
        raise ValueError("One or more datasets are empty.")

    for name, targets in [
        ("y_train", y_train),
        ("y_validation", y_validation),
        ("y_test", y_test),
    ]:
        if targets.ndim not in (1, 2):
            raise ValueError(f"{name} must be one-dimensional or a column vector.")


def calculate_metrics(actual, predicted):

    errors = predicted - actual

    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    scores = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0,
    )

    nasa_score = float(np.sum(scores))

    return mae, mse, rmse, r2, nasa_score


def main():

    print("\nNASA FD001 UNCAPPED LSTM TRAINING")
    print("=" * 55)

    print("Project directory:", PROJECT_ROOT)
    print("Data directory:", DATA_DIR)
    print("Data directory exists:", DATA_DIR.is_dir())

    X_train = load_array("X_train.npy")
    y_train = load_array("y_train.npy")

    X_validation = load_array("X_validation.npy")
    y_validation = load_array("y_validation.npy")

    X_test = load_array("X_test.npy")
    y_test = load_array("y_test.npy")

    validate_arrays(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )

    y_train = y_train.reshape(-1)
    y_validation = y_validation.reshape(-1)
    y_test = y_test.reshape(-1)

    print("\nDATA VALIDATION SUCCESSFUL")
    print("Training sequences:", len(X_train))
    print("Validation sequences:", len(X_validation))
    print("Official test engines:", len(X_test))
    print("Input shape:", X_train.shape[1:])
    print("Minimum training RUL:", float(y_train.min()))
    print("Maximum training RUL:", float(y_train.max()))

    if float(y_train.max()) <= 125:
        raise ValueError(
            "Training targets appear capped at 125. "
            "Run the uncapped preprocessing script before training."
        )

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="linear"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )

    callbacks = [
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nTRAINING NASA FD001 UNCAPPED RUL MODEL")
    print("=" * 55)

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError("The best candidate model was not saved.")

    best_model = tf.keras.models.load_model(MODEL_PATH)

    predictions = best_model.predict(
        X_test,
        verbose=0,
    ).reshape(-1)

    mae, mse, rmse, r2, nasa_score = calculate_metrics(
        y_test,
        predictions,
    )

    np.save(PREDICTION_PATH, predictions)

    np.savez(
        HISTORY_PATH,
        loss=np.asarray(history.history["loss"]),
        val_loss=np.asarray(history.history["val_loss"]),
        mae=np.asarray(history.history["mean_absolute_error"]),
        val_mae=np.asarray(history.history["val_mean_absolute_error"]),
    )

    print("\nUNCAPPED MODEL OFFICIAL TEST RESULTS")
    print("=" * 55)

    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))
    print("NASA Score:", round(nasa_score, 4))

    print(
        "Mean prediction bias:",
        round(float(np.mean(predictions - y_test)), 4),
    )

    print(
        "Maximum absolute error:",
        round(float(np.max(np.abs(predictions - y_test))), 4),
    )

    print("\nBASELINE MODEL RESULTS")
    print("=" * 55)

    print("MAE: 12.554")
    print("MSE: 269.327")
    print("RMSE: 16.411")
    print("R2: 0.844")
    print("NASA Score: 471.518")

    print("\nCANDIDATE MODEL SAVED SUCCESSFULLY")
    print("Model:", MODEL_PATH)
    print("Predictions:", PREDICTION_PATH)
    print("Training history:", HISTORY_PATH)

    print("\nOriginal live model was not overwritten.")


if __name__ == "__main__":
    main()

