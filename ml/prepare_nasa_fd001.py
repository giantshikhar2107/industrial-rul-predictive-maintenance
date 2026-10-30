from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "NASA_CMAPSS"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 30
RUL_CAP = 125
RANDOM_STATE = 42

COLUMNS = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

FEATURES = [
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


def load_data(filename):
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = COLUMNS

    if df[["engine_id", "cycle", *FEATURES]].isna().any().any():
        raise ValueError(f"Missing values found in {filename}")

    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    return df.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)


def add_training_rul(df):
    df = df.copy()
    last_cycles = df.groupby("engine_id")["cycle"].max()

    df["RUL"] = (
        df["engine_id"].map(last_cycles) - df["cycle"]
    ).clip(upper=RUL_CAP).astype(np.float32)

    return df


def create_training_sequences(df):
    X, y = [], []

    for _, engine in df.groupby("engine_id", sort=True):
        values = engine[FEATURES].to_numpy(dtype=np.float32)
        targets = engine["RUL"].to_numpy(dtype=np.float32)

        for end in range(WINDOW_SIZE, len(engine) + 1):
            X.append(values[end - WINDOW_SIZE:end])
            y.append(targets[end - 1])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )


def create_test_sequences(df, true_rul):
    X, y = [], []
    engines = sorted(df["engine_id"].unique())

    if len(engines) != len(true_rul):
        raise ValueError("Test engine count does not match RUL target count.")

    for engine_id, actual_rul in zip(engines, true_rul):
        engine = df[df["engine_id"] == engine_id]
        values = engine[FEATURES].to_numpy(dtype=np.float32)

        if len(values) >= WINDOW_SIZE:
            sequence = values[-WINDOW_SIZE:]
        else:
            padding = np.repeat(
                values[:1],
                WINDOW_SIZE - len(values),
                axis=0,
            )
            sequence = np.vstack([padding, values])

        X.append(sequence)
        y.append(actual_rul)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )


def main():
    train_df = load_data("train_FD001.txt")
    test_df = load_data("test_FD001.txt")

    rul_path = DATA_DIR / "RUL_FD001.txt"

    if not rul_path.exists():
        raise FileNotFoundError(f"RUL file not found: {rul_path}")

    test_rul = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
    ).iloc[:, 0].to_numpy(dtype=np.float32)

    if not np.isfinite(test_rul).all():
        raise ValueError("Test RUL targets contain invalid values.")

    engine_ids = train_df["engine_id"].unique()

    train_engine_ids, validation_engine_ids = train_test_split(
        engine_ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    training_df = add_training_rul(
        train_df[train_df["engine_id"].isin(train_engine_ids)].copy()
    )

    validation_df = add_training_rul(
        train_df[train_df["engine_id"].isin(validation_engine_ids)].copy()
    )

    scaler = MinMaxScaler()

    training_df[FEATURES] = scaler.fit_transform(
        training_df[FEATURES].astype(np.float32)
    ).astype(np.float32)

    validation_df[FEATURES] = scaler.transform(
        validation_df[FEATURES].astype(np.float32)
    ).astype(np.float32)

    test_df[FEATURES] = scaler.transform(
        test_df[FEATURES].astype(np.float32)
    ).astype(np.float32)

    X_train, y_train = create_training_sequences(training_df)
    X_validation, y_validation = create_training_sequences(validation_df)
    X_test, y_test = create_test_sequences(test_df, test_rul)

    if not all(len(x) > 0 for x in [X_train, X_validation, X_test]):
        raise ValueError("One or more datasets contain no sequences.")

    if float(y_train.max()) > RUL_CAP:
        raise ValueError("Training RUL exceeds the configured cap.")

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "X_validation.npy", X_validation)
    np.save(OUTPUT_DIR / "y_validation.npy", y_validation)
    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)

    joblib.dump(
        scaler,
        OUTPUT_DIR / "fd001_scaler.joblib",
    )

    training_df.to_csv(
        OUTPUT_DIR / "train_fd001_processed.csv",
        index=False,
    )

    validation_df.to_csv(
        OUTPUT_DIR / "validation_fd001_processed.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / "test_fd001_processed.csv",
        index=False,
    )

    print("\nNASA FD001 CAPPED RUL PREPROCESSING COMPLETE")
    print("=" * 52)
    print("Training engines:", len(train_engine_ids))
    print("Validation engines:", len(validation_engine_ids))
    print("Test engines:", test_df["engine_id"].nunique())
    print("Features:", len(FEATURES))
    print("Sequence length:", WINDOW_SIZE)
    print("Training RUL cap:", RUL_CAP)
    print("Maximum training RUL:", float(y_train.max()))
    print("Targets above cap:", int(np.sum(y_train > RUL_CAP)))
    print("X_train:", X_train.shape)
    print("X_validation:", X_validation.shape)
    print("X_test:", X_test.shape)
    print("Production scaler saved: fd001_scaler.joblib")


if __name__ == "__main__":
    main()