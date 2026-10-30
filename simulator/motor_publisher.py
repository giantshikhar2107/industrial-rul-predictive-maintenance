import json
import time

import paho.mqtt.client as mqtt

from simulator import config
from simulator.motor_machine import IndustrialMachine


def main():

    if not config.MQTT_BROKER:
        raise ValueError("HIVEMQ_BROKER is missing from .env.")

    if not config.MQTT_USERNAME or not config.MQTT_PASSWORD:
        raise ValueError("HiveMQ credentials are missing from .env.")

    machine = IndustrialMachine()

    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="industrial-motor-publisher",
        protocol=mqtt.MQTTv311
    )

    client.username_pw_set(
        config.MQTT_USERNAME,
        config.MQTT_PASSWORD
    )

    client.tls_set()

    try:
        print("Connecting to HiveMQ...")

        client.connect(
            config.MQTT_BROKER,
            config.MQTT_PORT,
            keepalive=60
        )

        client.loop_start()

        start_time = time.time()

        while not client.is_connected():
            if time.time() - start_time > 15:
                raise TimeoutError("Could not connect to HiveMQ.")
            time.sleep(0.1)

        print("Connected to HiveMQ.")
        print("Machine ID:", machine.machine_id)
        print("Machine Type:", machine.machine_type)
        print("MQTT Topic:", config.MQTT_TOPIC)
        print("Industrial Motor Simulator Started.")
        print("Press Ctrl+C to stop.\n")

        while machine.cycle < config.MAX_CYCLES:

            data = machine.run_cycle()
            payload = json.dumps(data)

            result = client.publish(
                config.MQTT_TOPIC,
                payload=payload,
                qos=1
            )

            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print("Publishing failed:", result.rc)
                time.sleep(1)
                continue

            result.wait_for_publish(timeout=10)

            if not result.is_published():
                print("Message delivery timed out.")
                time.sleep(1)
                continue

            print(
                f"Published | "
                f"Cycle: {data['cycle']} | "
                f"Temperature: {data['temperature']} °C | "
                f"Vibration: {data['vibration']} | "
                f"Current: {data['current']} A | "
                f"Pressure: {data['pressure']} | "
                f"RPM: {data['rpm']} | "
                f"Degradation: {data['degradation']} | "
                f"RUL: {data['rul']}"
            )

            time.sleep(config.SIMULATION_INTERVAL)

        print("\nIndustrial motor completed its simulation.")

    except KeyboardInterrupt:
        print("\nIndustrial motor simulator stopped.")

    finally:
        client.loop_stop()
        if client.is_connected():
            client.disconnect()


if __name__ == "__main__":
    main()