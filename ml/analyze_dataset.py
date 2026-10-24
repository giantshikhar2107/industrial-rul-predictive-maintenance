import pandas as pd

df = pd.read_csv("data/machine_data.csv")

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nMachine IDs:")
print(df["machine_id"].unique())

print("\nCycle range:")
print(df["cycle"].min(), "to", df["cycle"].max())

print("\nRUL range:")
print(df["rul"].min(), "to", df["rul"].max())
print("\nCycle frequency:")
print(df["cycle"].value_counts().describe())

print("\nFirst 20 cycle values:")
print(df["cycle"].head(20).tolist())

print("\nLast 20 cycle values:")
print(df["cycle"].tail(20).tolist())

print("\nMachine and cycle range:")
print(
    df.groupby("machine_id")["cycle"]
    .agg(["min", "max", "count"])
)

print("\nSensor statistics:")
print(df[[
    "temperature",
    "vibration",
    "current",
    "pressure",
    "rpm",
    "degradation",
    "rul"
]].describe())