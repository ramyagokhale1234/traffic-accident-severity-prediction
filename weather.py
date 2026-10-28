import pandas as pd
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CLEANED_DIR = BASE_DIR / "data" / "cleaned"

INPUT_FILE = CLEANED_DIR / "accident_analytical_dataset_2024.csv"
OUTPUT_FILE = CLEANED_DIR / "accident_weather_enriched_2024.csv"
CACHE_FILE = CLEANED_DIR / "weather_cache_2024.csv"


def get_weather_batch(latitude, longitude, start_date, end_date):

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max"
        ],
        "timezone": "auto"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()
        daily = data.get("daily", {})

        if not daily:
            return pd.DataFrame()

        weather_df = pd.DataFrame({
            "date": daily.get("time", []),
            "temperature_mean": daily.get(
                "temperature_2m_mean", []
            ),
            "precipitation": daily.get(
                "precipitation_sum", []
            ),
            "wind_speed_max": daily.get(
                "wind_speed_10m_max", []
            )
        })

        weather_df["weather_lat"] = latitude
        weather_df["weather_lon"] = longitude

        return weather_df

    except Exception as e:
        print(
            f"Weather API error for "
            f"{latitude}, {longitude}: {e}"
        )

        return pd.DataFrame()


def main():

    print("Reading analytical dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(f"Accident records: {len(df)}")

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # Round coordinates
    df["weather_lat"] = df["latitude"].round(1)
    df["weather_lon"] = df["longitude"].round(1)

    # Find unique locations
    locations = (
        df[
            [
                "weather_lat",
                "weather_lon"
            ]
        ]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"Unique weather locations: {len(locations)}"
    )

    # ---------------------------------------------
    # Load existing cache
    # ---------------------------------------------

    if CACHE_FILE.exists():

        weather_df = pd.read_csv(
            CACHE_FILE
        )

        weather_df["date"] = pd.to_datetime(
            weather_df["date"],
            errors="coerce"
        )

        print(
            f"Existing cached weather records: "
            f"{len(weather_df)}"
        )

    else:

        weather_df = pd.DataFrame(
            columns=[
                "date",
                "temperature_mean",
                "precipitation",
                "wind_speed_max",
                "weather_lat",
                "weather_lon"
            ]
        )

        print("No existing weather cache found.")

    # ---------------------------------------------
    # Find already downloaded locations
    # ---------------------------------------------

    if not weather_df.empty:

        cached_locations = set(
            zip(
                weather_df["weather_lat"],
                weather_df["weather_lon"]
            )
        )

    else:

        cached_locations = set()

    # ---------------------------------------------
    # Download weather
    # ---------------------------------------------

    print("\nFetching 2024 weather data...")

    new_weather_records = []

    for count, row in locations.iterrows():

        latitude = row["weather_lat"]
        longitude = row["weather_lon"]

        location_key = (
            latitude,
            longitude
        )

        # Skip locations already cached
        if location_key in cached_locations:
            continue

        print(
            f"Location {count + 1} / "
            f"{len(locations)}"
        )

        result = get_weather_batch(
            latitude,
            longitude,
            "2024-01-01",
            "2024-12-31"
        )

        if not result.empty:

            new_weather_records.append(
                result
            )

        # Save every 50 locations
        if (count + 1) % 50 == 0:

            if new_weather_records:

                new_df = pd.concat(
                    new_weather_records,
                    ignore_index=True
                )

                weather_df = pd.concat(
                    [
                        weather_df,
                        new_df
                    ],
                    ignore_index=True
                )

                weather_df.to_csv(
                    CACHE_FILE,
                    index=False
                )

                new_weather_records = []

                print(
                    "Progress saved to weather cache."
                )

    # ---------------------------------------------
    # Save remaining weather data
    # ---------------------------------------------

    if new_weather_records:

        new_df = pd.concat(
            new_weather_records,
            ignore_index=True
        )

        weather_df = pd.concat(
            [
                weather_df,
                new_df
            ],
            ignore_index=True
        )

        weather_df.to_csv(
            CACHE_FILE,
            index=False
        )

    print(
        f"\nTotal weather records cached: "
        f"{len(weather_df)}"
    )

    # ---------------------------------------------
    # Merge weather with accidents
    # ---------------------------------------------

    print("\nMerging weather with accidents...")

    df = df.merge(
        weather_df,
        on=[
            "weather_lat",
            "weather_lon",
            "date"
        ],
        how="left"
    )

    # ---------------------------------------------
    # Save final dataset
    # ---------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nWeather enrichment completed.")

    print(
        f"Final records: {len(df)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()