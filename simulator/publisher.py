import json
import time
import paho.mqtt.client as mqtt

from machine import IndustrialMachine
import config


machine = IndustrialMachine()

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

client.username_pw_set(
    config.MQTT_USERNAME,
    config.MQTT_PASSWORD
)

client.tls_set()

client.connect(
    config.MQTT_BROKER,
    config.MQTT_PORT,
    60
)

client.loop_start()

print("Industrial Machine Simulator Started")
print("Machine ID:", machine.machine_id)
print("MQTT Broker:", config.MQTT_BROKER)
print("MQTT Port:", config.MQTT_PORT)
print("MQTT Topic:", config.MQTT_TOPIC)
print()

try:

    while machine.cycle < config.MAX_CYCLES:

        data = machine.run_cycle()

        message = json.dumps(data)

        result = client.publish(
            config.MQTT_TOPIC,
            message
        )

        print(
            f"Published | "
            f"Cycle: {data['cycle']} | "
            f"Temperature: {data['temperature']} °C | "
            f"Vibration: {data['vibration']} | "
            f"Current: {data['current']} A | "
            f"RPM: {data['rpm']} | "
            f"RUL: {data['rul']}"
        )

        time.sleep(config.SIMULATION_INTERVAL)

except KeyboardInterrupt:

    print("\nMachine simulator stopped.")

finally:

    client.loop_stop()
    client.disconnect()