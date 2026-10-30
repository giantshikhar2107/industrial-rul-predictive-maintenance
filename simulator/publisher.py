import json
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulator import config
from simulator.machine import IndustrialMachine


ENGINE_ID = int(__import__("os").getenv("NASA_ENGINE_ID", "1"))
NASA_TOPIC = "factory/nasa_fd001/engine/sensors"
REPLAY_PAUSE = 2.0


def main():
    if not config.MQTT_BROKER:
        raise ValueError(
            "HIVEMQ_BROKER is missing from your .env configuration."
        )

    if not config.MQTT_USERNAME or not config.MQTT_PASSWORD:
        raise ValueError(
            "HiveMQ credentials are missing from your .env configuration."
        )

    machine = IndustrialMachine(engine_id=ENGINE_ID)

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"nasa-fd001-publisher-{ENGINE_ID}",
        protocol=mqtt.MQTTv311
    )

    client.username_pw_set(
        config.MQTT_USERNAME,
        config.MQTT_PASSWORD
    )

    client.tls_set()

    print("Connecting to HiveMQ...")
    client.connect(
        config.MQTT_BROKER,
        config.MQTT_PORT,
        keepalive=60
    )

    client.loop_start()

    try:
        while not client.is_connected():
            time.sleep(0.1)

        print("Connected to HiveMQ.")
        print(f"Engine ID: {ENGINE_ID}")
        print(f"MQTT topic: {NASA_TOPIC}")
        print("Continuous replay enabled. Press Ctrl+C to stop.")

        while True:
            reading = machine.run_cycle()

            if reading is None:
                print(
                    f"Engine {ENGINE_ID} reached its final recorded cycle. "
                    f"Restarting replay in {REPLAY_PAUSE:.0f} seconds..."
                )

                time.sleep(REPLAY_PAUSE)
                machine.reset()

                print("Replay restarted from cycle 1.")
                continue

            payload = json.dumps(reading)

            result = client.publish(
                NASA_TOPIC,
                payload=payload,
                qos=1
            )

            result.wait_for_publish()

            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(
                    f"Publish failed at cycle {reading.get('cycle')}: "
                    f"MQTT status {result.rc}"
                )
                time.sleep(1)
                continue

            print(
                f"Published | Engine {ENGINE_ID} | "
                f"Cycle {reading.get('cycle')} | "
                f"Actual RUL: {reading.get('actual_rul')} | "
                f"Topic: {NASA_TOPIC}"
            )

            time.sleep(config.SIMULATION_INTERVAL)

    except KeyboardInterrupt:
        print("\nPublisher stopped by user.")

    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()

