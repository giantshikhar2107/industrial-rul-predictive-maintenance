import random
from datetime import datetime
import config


class IndustrialMachine:

    def __init__(self):
        self.machine_id = config.MACHINE_ID
        self.machine_type = config.MACHINE_TYPE
        self.cycle = 0

    def calculate_degradation(self):
        return self.cycle / config.MAX_CYCLES

    def generate_sensor_data(self):

        degradation = self.calculate_degradation()

        temperature = (
            config.BASE_TEMPERATURE
            + 35 * degradation
            + random.gauss(0, 0.5)
        )

        vibration = (
            config.BASE_VIBRATION
            + 4.0 * degradation
            + random.gauss(0, 0.08)
        )

        current = (
            config.BASE_CURRENT
            + 3.0 * degradation
            + random.gauss(0, 0.1)
        )

        pressure = (
            config.BASE_PRESSURE
            - 0.8 * degradation
            + random.gauss(0, 0.05)
        )

        rpm = (
            config.BASE_RPM
            - 100 * degradation
            + random.gauss(0, 2)
        )

        rul = config.MAX_CYCLES - self.cycle

        return {
            "timestamp": datetime.now().isoformat(),
            "machine_id": self.machine_id,
            "machine_type": self.machine_type,
            "cycle": self.cycle,
            "temperature": round(temperature, 2),
            "vibration": round(vibration, 2),
            "current": round(current, 2),
            "pressure": round(pressure, 2),
            "rpm": round(rpm, 2),
            "degradation": round(degradation, 4),
            "rul": rul
        }

    def run_cycle(self):
        self.cycle += 1
        return self.generate_sensor_data()