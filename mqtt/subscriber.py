import sys
import pandas as pd
import os
import csv
import json
from collections import deque

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import joblib
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from supabase import create_client
from tensorflow.keras.models import load_model

from database.database import insert_sensor_reading
from simulator import config

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: Supabase credentials not found in .env")
    sys.exit()

SEQUENCE_LENGTH = 50

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

recent_readings = deque(
    maxlen=SEQUENCE_LENGTH
)

FEATURES = [
    "temperature",
    "vibration",
    "current",
    "pressure",
    "rpm"
]

last_cycle = None

print("Loading LSTM model...")

model = load_model(
    MODEL_FILE,
    compile=False
)

print("Loading scaler...")

scaler = joblib.load(
    SCALER_FILE
)

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.username_pw_set(
    config.MQTT_USERNAME,
    config.MQTT_PASSWORD
)

client.tls_set()


def predict_rul():

    if len(recent_readings) < SEQUENCE_LENGTH:
        return None

    sensor_data = pd.DataFrame(
        [
            [
                reading["temperature"],
                reading["vibration"],
                reading["current"],
                reading["pressure"],
                reading["rpm"]
            ]
            for reading in recent_readings
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

    return max(
        0,
        predicted_rul
    )


def on_connect(
    client,
    userdata,
    flags,
    reason_code,
    properties
):

    print("Connected to HiveMQ")

    client.subscribe(
        config.MQTT_TOPIC
    )

    print(
        "Subscribed to:",
        config.MQTT_TOPIC
    )


def on_message(
    client,
    userdata,
    msg,
    properties=None
):

    global last_cycle

    try:

        data = json.loads(
            msg.payload.decode()
        )

        current_cycle = data["cycle"]

        if (
            last_cycle is not None
            and current_cycle <= last_cycle
        ):
            print(
                "New simulation run detected. Resetting LSTM sequence."
            )

            recent_readings.clear()

        last_cycle = current_cycle

        recent_readings.append(
            data
        )

        predicted_rul = predict_rul()

        if predicted_rul is None:

            data["predicted_rul"] = None

        else:

            data["predicted_rul"] = round(
                predicted_rul,
                2
            )

        insert_sensor_reading(
            data
        )

        csv_file = os.path.join(
            PROJECT_ROOT,
            config.CSV_FILE
        )

        os.makedirs(
            os.path.dirname(csv_file),
            exist_ok=True
        )

        file_exists = os.path.isfile(
            csv_file
        )

        with open(
            csv_file,
            "a",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=data.keys()
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                data
            )

        supabase.table(
            "sensor_readings"
        ).insert(
            {
                "machine_id": data["machine_id"],
                "timestamp": data["timestamp"],
                "cycle": data["cycle"],
                "temperature": data["temperature"],
                "vibration": data["vibration"],
                "current": data["current"],
                "pressure": data["pressure"],
                "rpm": data["rpm"],
                "degradation": data["degradation"],
                "rul": data["rul"],
                "predicted_rul": data["predicted_rul"]
            }
        ).execute()

        print(
            "Cycle:",
            data["cycle"],
            "| Actual RUL:",
            data["rul"],
            "| Predicted RUL:",
            data["predicted_rul"]
        )

    except Exception as e:

        print(
            "Error processing message:",
            e
        )


client.on_connect = on_connect
client.on_message = on_message

print()
print("Starting MQTT subscriber...")
print()

client.connect(
    config.MQTT_BROKER,
    config.MQTT_PORT,
    60
)

client.loop_forever()