import pandas as pd

df = pd.read_csv("data/machine_data.csv")

df["run_id"] = (df["cycle"].diff() <= 0).cumsum() + 1

print("Total records:", len(df))

print("Number of runs:", df["run_id"].nunique())

print("\nRecords per run:")
print(df.groupby("run_id").size())

print("\nRun summary:")
print(
    df.groupby("run_id")
    .agg(
        start_cycle=("cycle", "min"),
        end_cycle=("cycle", "max"),
        records=("cycle", "count"),
        start_rul=("rul", "first"),
        end_rul=("rul", "last")
    )
)