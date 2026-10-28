import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"


def main():

    print("Reading cleaned collision data...")
    collisions = pd.read_csv(
        CLEANED_DIR / "cleaned_collisions_2024.csv",
        low_memory=False
    )

    print("Reading vehicle data...")
    vehicles = pd.read_csv(
        RAW_DIR / "dft-road-casualty-statistics-vehicle-2024.csv",
        low_memory=False
    )

    print("Reading casualty data...")
    casualties = pd.read_csv(
        RAW_DIR / "dft-road-casualty-statistics-casualty-2024.csv",
        low_memory=False
    )

    # --------------------------------------------------
    # VEHICLE SUMMARY
    # --------------------------------------------------

    vehicle_summary = (
    vehicles
    .groupby("collision_index")
    .agg(
        vehicle_count=("vehicle_reference", "count"),
        average_driver_age=("age_of_driver", "mean")
    )
    .reset_index()
    .rename(columns={"collision_index": "accident_id"})
)

    # --------------------------------------------------
    # CASUALTY SUMMARY
    # --------------------------------------------------

    casualty_summary = (
    casualties
    .groupby("collision_index")
    .agg(
        casualty_count=("casualty_reference", "count")
    )
    .reset_index()
    .rename(columns={"collision_index": "accident_id"})
)

    # --------------------------------------------------
    # MERGE DATA
    # --------------------------------------------------

    combined = collisions.merge(
        vehicle_summary,
        on="accident_id",
        how="left"
    )

    combined = combined.merge(
        casualty_summary,
        on="accident_id",
        how="left"
    )

    # Fill missing counts
    combined["vehicle_count"] = (
        combined["vehicle_count"]
        .fillna(0)
        .astype(int)
    )

    combined["casualty_count"] = (
        combined["casualty_count"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------
    # SAVE ANALYTICAL DATASET
    # --------------------------------------------------

    output_file = (
        CLEANED_DIR /
        "accident_analytical_dataset_2024.csv"
    )

    combined.to_csv(
        output_file,
        index=False
    )

    print("\nAnalytical dataset created successfully.")

    print(f"Collision records: {len(collisions)}")
    print(f"Vehicle records: {len(vehicles)}")
    print(f"Casualty records: {len(casualties)}")
    print(f"Final analytical records: {len(combined)}")

    print(f"\nSaved to:")
    print(output_file)


if __name__ == "__main__":
    main()