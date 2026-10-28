import pandas as pd
from pathlib import Path


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
REJECTED_DIR = BASE_DIR / "data" / "rejected"

CLEANED_DIR.mkdir(parents=True, exist_ok=True)
REJECTED_DIR.mkdir(parents=True, exist_ok=True)


def find_column(df, possible_names):
    """Find a column even if its capitalization/name format differs."""

    normalized = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    for name in possible_names:
        key = name.lower().replace(" ", "_")

        if key in normalized:
            return normalized[key]

    return None


def main():

    collision_file = RAW_DIR / "dft-road-casualty-statistics-collision-2024.csv"

    print("Reading collision data...")

    df = pd.read_csv(collision_file, low_memory=False)

    print(f"Original rows: {len(df)}")
    print(f"Original columns: {len(df.columns)}")

    # Find important columns
    accident_id_col = find_column(
        df,
        ["collision_index"]
    )

    severity_col = find_column(
        df,
        ["collision_severity"]
    )

    latitude_col = find_column(
        df,
        ["latitude"]
    )

    longitude_col = find_column(
        df,
        ["longitude"]
    )

    date_col = find_column(
        df,
        ["date"]
    )

    time_col = find_column(
        df,
        ["time"]
    )

    # Check required columns
    required = {
        "Accident ID": accident_id_col,
        "Severity": severity_col,
        "Latitude": latitude_col,
        "Longitude": longitude_col,
        "Date": date_col,
        "Time": time_col
    }

    print("\nDetected columns:")

    for name, column in required.items():
        print(f"{name}: {column}")

    missing = [
        name for name, column in required.items()
        if column is None
    ]

    if missing:
        print("\nMissing columns:")
        print(missing)
        print("\nCleaning stopped.")
        return

    # Rename important columns
    df = df.rename(columns={
        accident_id_col: "accident_id",
        severity_col: "accident_severity",
        latitude_col: "latitude",
        longitude_col: "longitude",
        date_col: "date",
        time_col: "time"
    })

    # Remove duplicate accidents
    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["accident_id"]
    )

    duplicates_removed = before_duplicates - len(df)

    print(f"\nDuplicates removed: {duplicates_removed}")

    # Convert date
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        dayfirst=True
    )

    # Convert latitude and longitude
    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    # Identify invalid records
    invalid = (
        df["accident_id"].isna()
        | df["date"].isna()
        | df["latitude"].isna()
        | df["longitude"].isna()
        | ~df["latitude"].between(-90, 90)
        | ~df["longitude"].between(-180, 180)
    )

    rejected = df[invalid].copy()
    cleaned = df[~invalid].copy()

    # Save rejected records
    rejected_file = REJECTED_DIR / "rejected_collisions_2024.csv"

    rejected.to_csv(
        rejected_file,
        index=False
    )

    # Convert severity codes to labels
    severity_map = {
        1: "Fatal",
        2: "Serious",
        3: "Slight"
    }

    cleaned["severity_label"] = (
        pd.to_numeric(
            cleaned["accident_severity"],
            errors="coerce"
        ).map(severity_map)
    )

    # Extract useful time information
    cleaned["year"] = cleaned["date"].dt.year
    cleaned["month"] = cleaned["date"].dt.month
    cleaned["day"] = cleaned["date"].dt.day
    cleaned["weekday"] = cleaned["date"].dt.day_name()

    # Save cleaned data
    cleaned_file = CLEANED_DIR / "cleaned_collisions_2024.csv"

    cleaned.to_csv(
        cleaned_file,
        index=False
    )

    print("\nCleaning completed.")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Rejected rows: {len(rejected)}")

    print(f"\nCleaned file:")
    print(cleaned_file)

    print(f"\nRejected file:")
    print(rejected_file)


if __name__ == "__main__":
    main()