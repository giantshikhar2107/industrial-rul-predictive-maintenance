import json
import csv
import os
from datetime import datetime

import paho.mqtt.client as mqtt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import simulator.config as config

from database.database import initialize_database, insert_machine, insert_sensor_reading


CSV_FILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    config.CSV_FILE
)

CSV_FILE = os.path.abspath(CSV_FILE)

CSV_COLUMNS = [
    "timestamp",
    "machine_id",
    "machine_type",
    "cycle",
    "temperature",
    "vibration",
    "current",
    "pressure",
    "rpm",
    "degradation",
    "rul"
]


def create_csv():

    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    if not os.path.exists(CSV_FILE):

        with open(
            CSV_FILE,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_COLUMNS
            )

            writer.writeheader()


def on_connect(client, userdata, flags, reason_code, properties):

    print("Connected to HiveMQ Cloud")

    client.subscribe(config.MQTT_TOPIC)

    print("Subscribed to:", config.MQTT_TOPIC)


def on_message(client, userdata, message):

    try:

        data = json.loads(
            message.payload.decode()
        )

        insert_sensor_reading(data)

        with open(
            CSV_FILE,
            "a",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_COLUMNS
            )

            writer.writerow(data)

        print(
            f"Saved | "
            f"Cycle: {data['cycle']} | "
            f"Temperature: {data['temperature']} °C | "
            f"Vibration: {data['vibration']} | "
            f"RUL: {data['rul']}"
        )

    except Exception as error:

        print("Error:", error)


create_csv()

initialize_database()

insert_machine(
    "M001",
    "Motor-01",
    "Industrial Motor",
    "Production Line 1"
)


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

client.connect(
    config.MQTT_BROKER,
    config.MQTT_PORT,
    60
)

print("MQTT Data Logger Started")
print("MQTT Broker:", config.MQTT_BROKER)
print("MQTT Topic:", config.MQTT_TOPIC)
print("CSV File:", CSV_FILE)

client.loop_forever()