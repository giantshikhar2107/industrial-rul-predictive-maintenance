import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "ml",
    "best_lstm_model.keras"
)

SCALER_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "scaler.pkl"
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "lstm_training_data.csv"
)


SEQUENCE_LENGTH = 50

FEATURES = [
    "temperature",
    "vibration",
    "current",
    "pressure",
    "rpm"
]


df = pd.read_csv(
    INPUT_FILE
)


latest_run = df["run_id"].max()

df = df[
    df["run_id"] == latest_run
].copy()


df = df.sort_values(
    "cycle"
)


latest_readings = df.tail(
    SEQUENCE_LENGTH
)


if len(latest_readings) != SEQUENCE_LENGTH:

    print(
        "Error: Not enough sensor readings."
    )

    raise SystemExit


scaler = joblib.load(
    SCALER_FILE
)


model = load_model(
    MODEL_FILE,
    compile=False
)


sensor_data = latest_readings[
    FEATURES
]


sensor_data_scaled = scaler.transform(
    sensor_data
)


X_input = np.array(
    [sensor_data_scaled],
    dtype=np.float32
)


prediction = model(
    X_input,
    training=False
).numpy()


predicted_rul = float(
    prediction[0][0]
)


predicted_rul = max(
    0,
    predicted_rul
)


latest_reading = latest_readings.iloc[-1]


print("\nLSTM RUL PREDICTION")
print("-------------------")

print(
    "Machine ID:",
    latest_reading["machine_id"]
)

print(
    "Run ID:",
    latest_run
)

print(
    "Latest cycle:",
    latest_reading["cycle"]
)

print(
    "Actual RUL:",
    latest_reading["rul"]
)

print(
    "Readings used:",
    len(sensor_data)
)

print(
    "Predicted RUL:",
    round(
        predicted_rul,
        2
    ),
    "cycles"
)