import sys
import os
import csv
import json
import sqlite3
from collections import deque

import numpy as np
import pandas as pd
import joblib
import paho.mqtt.client as mqtt

from dotenv import load_dotenv
from supabase import create_client
from tensorflow.keras.models import load_model

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from database.database import (
    insert_sensor_reading,
    get_recent_sensor_readings
)
from simulator import config

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("ERROR: Supabase credentials not found in .env")
    sys.exit(1)

SEQUENCE_LENGTH = 50

MODEL_FILE = os.path.join(
    PROJECT_ROOT, "ml", "best_lstm_model.keras"
)

SCALER_FILE = os.path.join(
    PROJECT_ROOT, "data", "scaler.pkl"
)

CSV_FILE = os.path.join(
    PROJECT_ROOT, config.CSV_FILE
)

FEATURES = [
    "temperature",
    "vibration",
    "current",
    "pressure",
    "rpm"
]

recent_readings = deque(maxlen=SEQUENCE_LENGTH)
last_cycle = None

print("Loading LSTM model...")
model = load_model(MODEL_FILE, compile=False)

print("Loading scaler...")
scaler = joblib.load(SCALER_FILE)

print("Connecting to Supabase...")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def restore_recent_readings():
    global last_cycle

    try:
        readings = get_recent_sensor_readings(
            limit=SEQUENCE_LENGTH,
            machine_id="M001"
        )

        if not readings:
            print("No historical readings found.")
            return

        valid_readings = []

        for reading in readings:
            if all(
                reading.get(feature) is not None
                for feature in FEATURES
            ):
                valid_readings.append(reading)

        if not valid_readings:
            print("No valid historical sensor readings found.")
            return

        consecutive_readings = [valid_readings[-1]]

        for reading in reversed(valid_readings[:-1]):
            newer_cycle = consecutive_readings[0]["cycle"]

            if reading["cycle"] == newer_cycle - 1:
                consecutive_readings.insert(0, reading)
            else:
                break

        for reading in consecutive_readings:
            recent_readings.append(reading)

        last_cycle = consecutive_readings[-1]["cycle"]

        print(
            "Restored",
            len(recent_readings),
            "consecutive historical readings."
        )

        print("Last restored cycle:", last_cycle)

        if len(recent_readings) < SEQUENCE_LENGTH:
            print(
                "Waiting for more readings to complete the "
                "50-reading LSTM sequence."
            )

    except Exception as error:
        print("Error restoring historical readings:", error)


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

    sensor_data_scaled = scaler.transform(sensor_data)

    X_input = np.array(
        [sensor_data_scaled],
        dtype=np.float32
    )

    prediction = model(
        X_input,
        training=False
    ).numpy()

    predicted_rul = float(prediction[0][0])

    return max(0.0, predicted_rul)


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("\nConnected to HiveMQ successfully.")

        client.subscribe(config.MQTT_TOPIC)

        print("Subscribed to:", config.MQTT_TOPIC)
        print("Waiting for sensor readings...\n")
    else:
        print("MQTT connection failed:", reason_code)


def on_message(client, userdata, msg, properties=None):
    global last_cycle

    try:
        data = json.loads(msg.payload.decode("utf-8"))

        current_cycle = int(data["cycle"])

        required_fields = [
            "machine_id",
            "timestamp",
            "temperature",
            "vibration",
            "current",
            "pressure",
            "rpm",
            "rul"
        ]

        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            print("Message missing fields:", missing_fields)
            return

        if last_cycle is not None:
            if current_cycle <= last_cycle:
                print(
                    "\nNew simulation run or cycle reset detected. "
                    "Clearing LSTM sequence."
                )
                recent_readings.clear()

            elif current_cycle != last_cycle + 1:
                print(
                    "\nCycle gap detected. Clearing LSTM sequence "
                    "to avoid using non-consecutive readings."
                )
                recent_readings.clear()

        last_cycle = current_cycle

        recent_readings.append(data)

        predicted_rul = predict_rul()

        if predicted_rul is None:
            data["predicted_rul"] = None
        else:
            data["predicted_rul"] = round(predicted_rul, 2)

        insert_sensor_reading(data)

        os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

        file_exists = (
            os.path.isfile(CSV_FILE)
            and os.path.getsize(CSV_FILE) > 0
        )

        with open(
            CSV_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(data.keys())
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(data)

        supabase_data = {
            "machine_id": data["machine_id"],
            "timestamp": data["timestamp"],
            "cycle": current_cycle,
            "temperature": data["temperature"],
            "vibration": data["vibration"],
            "current": data["current"],
            "pressure": data["pressure"],
            "rpm": data["rpm"],
            "degradation": data.get("degradation"),
            "rul": data["rul"],
            "predicted_rul": data["predicted_rul"]
        }

        supabase.table("sensor_readings").insert(
            supabase_data
        ).execute()

        print(
            "Cycle:", current_cycle,
            "| Actual RUL:", data["rul"],
            "| Predicted RUL:", data["predicted_rul"],
            "| Sequence:", len(recent_readings), "/", SEQUENCE_LENGTH
        )

    except Exception as error:
        print("Error processing MQTT message:", error)


def on_disconnect(
    client,
    userdata,
    disconnect_flags,
    reason_code,
    properties
):
    print("Disconnected from HiveMQ:", reason_code)


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.username_pw_set(
    config.MQTT_USERNAME,
    config.MQTT_PASSWORD
)

client.tls_set()

client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

restore_recent_readings()

print("\nStarting MQTT subscriber...\n")

try:
    client.connect(
        config.MQTT_BROKER,
        config.MQTT_PORT,
        60
    )

    client.loop_forever()

except KeyboardInterrupt:
    print("\nSubscriber stopped by user.")

except Exception as error:
    print("MQTT connection error:", error)

finally:
    client.disconnect()
    client.loop_stop()
    print("Subscriber stopped.")