import os
import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from supabase import create_client
from tensorflow.keras.models import load_model


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


load_dotenv(
    os.path.join(
        PROJECT_ROOT,
        ".env"
    )
)


SUPABASE_URL = os.getenv(
    "SUPABASE_URL"
)

SUPABASE_KEY = os.getenv(
    "SUPABASE_PUBLISHABLE_KEY"
)


MACHINE_ID = "M001"

SEQUENCE_LENGTH = 50

FEATURES = [
    "temperature",
    "vibration",
    "current",
    "pressure",
    "rpm"
]


if not SUPABASE_URL or not SUPABASE_KEY:

    print(
        "Error: Supabase credentials not found."
    )

    raise SystemExit


supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
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


model = load_model(
    MODEL_FILE,
    compile=False
)


scaler = joblib.load(
    SCALER_FILE
)


response = (
    supabase
    .table("sensor_readings")
    .select(
        "machine_id,timestamp,cycle,"
        "temperature,vibration,current,"
        "pressure,rpm"
    )
    .eq(
        "machine_id",
        MACHINE_ID
    )
    .order(
        "id",
        desc=True
    )
    .limit(
        SEQUENCE_LENGTH
    )
    .execute()
)


readings = response.data


if len(readings) < SEQUENCE_LENGTH:

    print(
        "Not enough sensor readings."
    )

    print(
        "Required:",
        SEQUENCE_LENGTH
    )

    print(
        "Available:",
        len(readings)
    )

    raise SystemExit


readings = list(
    reversed(readings)
)


sensor_data = pd.DataFrame(
    [
        [
            reading["temperature"],
            reading["vibration"],
            reading["current"],
            reading["pressure"],
            reading["rpm"]
        ]
        for reading in readings
    ],
    columns=FEATURES
)


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


latest_reading = readings[-1]


print("\nLIVE LSTM RUL PREDICTION")
print("------------------------")

print(
    "Machine ID:",
    MACHINE_ID
)

print(
    "Latest cycle:",
    latest_reading["cycle"]
)

print(
    "Latest timestamp:",
    latest_reading["timestamp"]
)

print(
    "Readings used:",
    len(readings)
)

print(
    "Predicted RUL:",
    round(
        predicted_rul,
        2
    ),
    "cycles"
)