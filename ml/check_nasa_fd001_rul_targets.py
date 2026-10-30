from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data" / "NASA_CMAPSS"

RUL_CAP = 125


def load_training_data():
    path = DATA_DIR / "train_FD001.txt"

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    columns = ["engine_id", "cycle"]

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=columns,
    )

    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    last_cycles = df.groupby("engine_id")["cycle"].max()

    df["RUL"] = (
        df["engine_id"].map(last_cycles) - df["cycle"]
    ).astype(int)

    return df


def load_test_data():
    test_path = DATA_DIR / "test_FD001.txt"
    rul_path = DATA_DIR / "RUL_FD001.txt"

    if not test_path.exists():
        raise FileNotFoundError(f"File not found: {test_path}")

    if not rul_path.exists():
        raise FileNotFoundError(f"File not found: {rul_path}")

    test_df = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["engine_id", "cycle"],
    )

    test_rul = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
    ).iloc[:, 0].to_numpy(dtype=np.float64)

    engines = sorted(test_df["engine_id"].unique())

    if len(engines) != len(test_rul):
        raise ValueError(
            f"Test engine count ({len(engines)}) does not match "
            f"RUL target count ({len(test_rul)})."
        )

    result = pd.DataFrame({
        "engine_id": engines,
        "actual_RUL": test_rul,
    })

    return result


def print_summary(name, values):
    values = np.asarray(values, dtype=np.float64)

    print(f"\n{name}")
    print("-" * 50)
    print("Count:", len(values))
    print("Minimum RUL:", round(float(values.min()), 2))
    print("Maximum RUL:", round(float(values.max()), 2))
    print("Mean RUL:", round(float(values.mean()), 2))
    print("Median RUL:", round(float(np.median(values)), 2))
    print("RUL above 125:", int(np.sum(values > RUL_CAP)))
    print("RUL at or below 125:", int(np.sum(values <= RUL_CAP)))


def main():
    print("\nNASA C-MAPSS FD001 RUL TARGET VALIDATION")
    print("=" * 55)

    train_df = load_training_data()
    test_df = load_test_data()

    print_summary(
        "TRAINING RUL DISTRIBUTION (UNCAPPED)",
        train_df["RUL"],
    )

    print_summary(
        "TRAINING RUL DISTRIBUTION (CAPPED AT 125)",
        train_df["RUL"].clip(upper=RUL_CAP),
    )

    print_summary(
        "OFFICIAL TEST RUL DISTRIBUTION (UNCAPPED)",
        test_df["actual_RUL"],
    )

    print("\nOFFICIAL TEST ENGINES WITH RUL ABOVE 125")
    print("-" * 55)

    above_cap = test_df[
        test_df["actual_RUL"] > RUL_CAP
    ]

    if above_cap.empty:
        print("No test engines have RUL above 125.")
    else:
        print(above_cap.to_string(index=False))

    print("\nTRAINING RUL CAP IMPACT")
    print("-" * 55)

    capped_count = int(
        np.sum(train_df["RUL"].to_numpy() > RUL_CAP)
    )

    print("Training observations capped:", capped_count)
    print(
        "Percentage capped:",
        round(capped_count / len(train_df) * 100, 2),
        "%",
    )

    print("\nVALIDATION RESULT")
    print("-" * 55)

    if len(above_cap) > 0:
        print(
            "The official test set contains RUL targets above 125, "
            "while the training labels are capped at 125."
        )
        print(
            "This is a target-range difference worth investigating "
            "through a controlled model experiment."
        )
    else:
        print(
            "No official test targets exceed the training RUL cap."
        )

    print("\nValidation completed successfully.")


if __name__ == "__main__":
    main()

