import os
import sys
import json
import csv
import logging
from collections import deque

import joblib
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from simulator import config
from database.database import insert_nasa_sensor_reading

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

MQTT_TOPIC = "factory/nasa_fd001/engine/sensors"
SEQUENCE_LENGTH = 30

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ml",
    "nasa_fd001_lstm.keras"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "NASA_CMAPSS",
    "processed",
    "fd001_scaler.joblib"
)

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "nasa_fd001_live_readings.csv"
)

FEATURES = [
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)]
]

CSV_FIELDS = [
    "timestamp",
    "machine_id",
    "dataset",
    "engine_id",
    "cycle",
    "actual_rul",
    "predicted_rul",
    "absolute_error",
    *FEATURES
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"NASA model not found: {MODEL_PATH}"
    )

if not os.path.isfile(SCALER_PATH):
    raise FileNotFoundError(
        f"NASA scaler not found: {SCALER_PATH}"
    )

model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

if len(FEATURES) != 24:
    raise ValueError(
        f"Expected 24 features, found {len(FEATURES)}"
    )

if getattr(scaler, "n_features_in_", None) != len(FEATURES):
    raise ValueError(
        "Scaler feature count does not match subscriber features."
    )

if tuple(model.input_shape[1:]) != (
    SEQUENCE_LENGTH,
    len(FEATURES)
):
    raise ValueError(
        f"Model input shape mismatch: {model.input_shape}"
    )

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

active_engine = None
last_cycle = None

csv_initialized = (
    os.path.isfile(CSV_PATH)
    and os.path.getsize(CSV_PATH) > 0
)


def predict_rul():
    sequence = np.asarray(
        list(sequence_buffer),
        dtype=np.float32
    )

    if sequence.shape != (
        SEQUENCE_LENGTH,
        len(FEATURES)
    ):
        raise ValueError(
            f"Unexpected sequence shape: {sequence.shape}"
        )

    sequence_df = pd.DataFrame(
        sequence,
        columns=FEATURES
    )

    scaled_sequence = scaler.transform(sequence_df)

    scaled_sequence = scaled_sequence.reshape(
        1,
        SEQUENCE_LENGTH,
        len(FEATURES)
    )

    prediction = model.predict(
        scaled_sequence,
        verbose=0
    )

    predicted_rul = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    if not np.isfinite(predicted_rul):
        raise ValueError(
            "Model produced an invalid RUL prediction."
        )

    return max(0.0, predicted_rul)


def save_to_csv(data):
    global csv_initialized

    with open(
        CSV_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS
        )

        if not csv_initialized:
            writer.writeheader()
            csv_initialized = True

        writer.writerow({
            field: data.get(field)
            for field in CSV_FIELDS
        })


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        logger.info("Connected to HiveMQ.")

        client.subscribe(
            MQTT_TOPIC,
            qos=1
        )

        logger.info(
            "Subscribed to %s",
            MQTT_TOPIC
        )
    else:
        logger.error(
            "HiveMQ connection failed: %s",
            reason_code
        )


def on_message(client, userdata, message):
    global active_engine, last_cycle

    try:
        data = json.loads(
            message.payload.decode("utf-8")
        )

        required_fields = [
            "machine_id",
            "timestamp",
            "engine_id",
            "cycle",
            "max_cycles",
            *FEATURES
        ]

        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            raise ValueError(
                f"Missing fields: {missing_fields}"
            )

        engine_id = int(data["engine_id"])
        cycle = int(data["cycle"])

        if cycle < 1:
            raise ValueError(
                f"Invalid cycle number: {cycle}"
            )

        feature_values = np.asarray(
            [float(data[field]) for field in FEATURES],
            dtype=np.float64
        )

        if not np.all(np.isfinite(feature_values)):
            raise ValueError(
                "Sensor data contains invalid numeric values."
            )

        # Reset the sequence when the engine changes.
        if active_engine != engine_id:
            sequence_buffer.clear()

            active_engine = engine_id
            last_cycle = None

            logger.info(
                "Starting prediction sequence for engine %s.",
                engine_id
            )

        # Ignore immediately repeated cycle messages.
        if last_cycle is not None and cycle == last_cycle:
            logger.warning(
                "Duplicate message ignored | Engine %s | Cycle %s",
                engine_id,
                cycle
            )
            return

        # Detect replay or missing cycles.
        if last_cycle is not None and cycle != last_cycle + 1:
            sequence_buffer.clear()

            if cycle < last_cycle:
                logger.warning(
                    "Replay detected | Engine %s | Cycle %s -> %s",
                    engine_id,
                    last_cycle,
                    cycle
                )
            else:
                logger.warning(
                    "Cycle discontinuity detected | Engine %s | "
                    "Previous: %s | Current: %s",
                    engine_id,
                    last_cycle,
                    cycle
                )

        last_cycle = cycle

        sequence_buffer.append(feature_values)

        data["dataset"] = data.get(
            "dataset",
            "NASA_CMAPSS_FD001"
        )

        data["actual_rul"] = data.get(
            "actual_rul",
            data.get("rul")
        )

        data["predicted_rul"] = None
        data["absolute_error"] = None

        # Predict only after collecting 30 consecutive cycles.
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            predicted_rul = predict_rul()

            data["predicted_rul"] = round(
                predicted_rul,
                2
            )

            if data["actual_rul"] is not None:
                data["absolute_error"] = round(
                    abs(
                        float(data["actual_rul"])
                        - predicted_rul
                    ),
                    2
                )

        insert_nasa_sensor_reading(data)

        save_to_csv(data)

        if data["predicted_rul"] is None:
            prediction_text = (
                f"warming up "
                f"({len(sequence_buffer)}/{SEQUENCE_LENGTH})"
            )
        else:
            prediction_text = (
                f"{data['predicted_rul']:.2f}"
            )

        logger.info(
            "Engine %s | Cycle %s | Actual RUL: %s | "
            "Predicted RUL: %s",
            engine_id,
            cycle,
            data["actual_rul"],
            prediction_text
        )

    except Exception:
        logger.exception(
            "Failed to process NASA MQTT message."
        )


def main():
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"nasa-rul-subscriber-{os.getpid()}",
        protocol=mqtt.MQTTv311
    )

    client.username_pw_set(
        config.MQTT_USERNAME,
        config.MQTT_PASSWORD
    )

    client.tls_set()

    client.on_connect = on_connect
    client.on_message = on_message

    logger.info("Starting NASA RUL subscriber.")
    logger.info("Model: %s", MODEL_PATH)
    logger.info("Scaler: %s", SCALER_PATH)
    logger.info("CSV output: %s", CSV_PATH)

    client.connect(
        config.MQTT_BROKER,
        config.MQTT_PORT,
        keepalive=60
    )

    try:
        client.loop_forever()

    except KeyboardInterrupt:
        logger.info(
            "NASA subscriber stopped by user."
        )

    finally:
        client.disconnect()


if __name__ == "__main__":
    main()