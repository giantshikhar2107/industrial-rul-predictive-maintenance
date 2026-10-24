import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.machine import IndustrialMachine


NUMBER_OF_RUNS = 20
OUTPUT_FILE = "data/lstm_training_data.csv"


all_data = []

for run in range(1, NUMBER_OF_RUNS + 1):

    machine = IndustrialMachine()

    for _ in range(10000):
        data = machine.run_cycle()
        data["run_id"] = run
        all_data.append(data)


df = pd.DataFrame(all_data)

df = df[
    [
        "run_id",
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
]

df.to_csv(OUTPUT_FILE, index=False)

print("Dataset generated successfully")
print("Shape:", df.shape)
print("Number of runs:", df["run_id"].nunique())
print("Records per run:")
print(df.groupby("run_id").size())
print("RUL range:", df["rul"].min(), "to", df["rul"].max())
print("Output:", OUTPUT_FILE)