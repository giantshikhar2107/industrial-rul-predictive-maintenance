import os

from dotenv import load_dotenv

load_dotenv()

MACHINE_ID = "M001"

MACHINE_TYPE = "Industrial Motor"

MAX_CYCLES = 10000

BASE_TEMPERATURE = 50.0

BASE_VIBRATION = 0.8

BASE_CURRENT = 7.5

BASE_PRESSURE = 5.0

BASE_RPM = 1500.0

SIMULATION_INTERVAL = 0.1

MQTT_BROKER = os.getenv("HIVEMQ_BROKER")

MQTT_PORT = int(os.getenv("HIVEMQ_PORT", 8883))

MQTT_USERNAME = os.getenv("HIVEMQ_USERNAME")

MQTT_PASSWORD = os.getenv("HIVEMQ_PASSWORD")

MQTT_TOPIC = "factory/line1/M001/sensors"

CSV_FILE = "data/machine_data.csv"