import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "NASA_CMAPSS" / "train_FD001.txt"

COLUMNS = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

FEATURES = [
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


class IndustrialMachine:

    def __init__(self, engine_id=None):
        if not DATA_FILE.exists():
            raise FileNotFoundError(
                f"NASA FD001 training data not found: {DATA_FILE}"
            )

        self.dataset = pd.read_csv(
            DATA_FILE,
            sep=r"\s+",
            header=None,
        )

        self.dataset = self.dataset.dropna(axis=1, how="all")
        self.dataset.columns = COLUMNS

        self.dataset["engine_id"] = self.dataset["engine_id"].astype(int)
        self.dataset["cycle"] = self.dataset["cycle"].astype(int)

        self.engine_ids = sorted(
            self.dataset["engine_id"].unique().tolist()
        )

        if engine_id is None:
            engine_id = self.engine_ids[0]

        if engine_id not in self.engine_ids:
            raise ValueError(
                f"Invalid engine ID {engine_id}. "
                f"Available IDs: {self.engine_ids[0]}–{self.engine_ids[-1]}"
            )

        self.engine_id = int(engine_id)
        self.machine_id = f"NASA_FD001_{self.engine_id:03d}"
        self.machine_type = "NASA C-MAPSS Turbofan Engine"

        self.engine_data = (
            self.dataset[
                self.dataset["engine_id"] == self.engine_id
            ]
            .sort_values("cycle")
            .reset_index(drop=True)
        )

        self.max_cycles = int(self.engine_data["cycle"].max())
        self.index = 0
        self.cycle = 0

        print(
            f"Loaded NASA engine {self.engine_id} "
            f"with {self.max_cycles} recorded cycles."
        )

    def generate_sensor_data(self):
        if self.index >= len(self.engine_data):
            return None

        row = self.engine_data.iloc[self.index]

        self.cycle = int(row["cycle"])
        actual_rul = max(0, self.max_cycles - self.cycle)

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "machine_id": self.machine_id,
            "machine_type": self.machine_type,
            "dataset": "NASA_CMAPSS_FD001",
            "engine_id": self.engine_id,
            "cycle": self.cycle,
            "max_cycles": self.max_cycles,
            "actual_rul": actual_rul,
            "rul": actual_rul,
        }

        for feature in FEATURES:
            data[feature] = float(row[feature])

        return data

    def run_cycle(self):
        data = self.generate_sensor_data()

        if data is not None:
            self.index += 1

        return data

    def reset(self):
        self.index = 0
        self.cycle = 0

