import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "lstm_training_data.csv"
)

SCALER_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "scaler.pkl"
)


SEQUENCE_LENGTH = 50

TRAIN_RUNS = range(1, 15)

VALIDATION_RUNS = range(15, 18)

TEST_RUNS = range(18, 21)


FEATURES = [
    "temperature",
    "vibration",
    "current",
    "pressure",
    "rpm"
]

TARGET = "rul"


df = pd.read_csv(
    INPUT_FILE
)


df = df.sort_values(
    ["run_id", "cycle"]
).reset_index(
    drop=True
)


train_df = df[
    df["run_id"].isin(TRAIN_RUNS)
].copy()

validation_df = df[
    df["run_id"].isin(VALIDATION_RUNS)
].copy()

test_df = df[
    df["run_id"].isin(TEST_RUNS)
].copy()


scaler = StandardScaler()


train_df[FEATURES] = scaler.fit_transform(
    train_df[FEATURES]
)

validation_df[FEATURES] = scaler.transform(
    validation_df[FEATURES]
)

test_df[FEATURES] = scaler.transform(
    test_df[FEATURES]
)


joblib.dump(
    scaler,
    SCALER_FILE
)


def create_sequences(data):

    X = []
    y = []

    for run_id in sorted(
        data["run_id"].unique()
    ):

        run = data[
            data["run_id"] == run_id
        ].sort_values(
            "cycle"
        )

        sensor_values = run[
            FEATURES
        ].values

        rul_values = run[
            TARGET
        ].values

        for i in range(
            SEQUENCE_LENGTH,
            len(run)
        ):

            X.append(
                sensor_values[
                    i - SEQUENCE_LENGTH:i
                ]
            )

            y.append(
                rul_values[i]
            )

    return (
        np.array(X),
        np.array(y)
    )


X_train, y_train = create_sequences(
    train_df
)

X_validation, y_validation = create_sequences(
    validation_df
)

X_test, y_test = create_sequences(
    test_df
)


np.save(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "X_train.npy"
    ),
    X_train
)

np.save(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "y_train.npy"
    ),
    y_train
)

np.save(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "X_validation.npy"
    ),
    X_validation
)

np.save(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "y_validation.npy"
    ),
    y_validation
)

np.save(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "X_test.npy"
    ),
    X_test
)

np.save(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "y_test.npy"
    ),
    y_test
)


print(
    "LSTM preprocessing completed"
)


print(
    "\nTraining data:"
)

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)


print(
    "\nValidation data:"
)

print(
    "X_validation:",
    X_validation.shape
)

print(
    "y_validation:",
    y_validation.shape
)


print(
    "\nTest data:"
)

print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)


print(
    "\nFeatures:"
)

print(
    FEATURES
)


print(
    "\nSequence length:",
    SEQUENCE_LENGTH
)


print(
    "\nScaler saved:"
)

print(
    SCALER_FILE
)