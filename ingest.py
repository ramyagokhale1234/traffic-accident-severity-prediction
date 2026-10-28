import pandas as pd
from pathlib import Path
from datetime import datetime
import json


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)


def ingest_file(file_path):
    """Read one CSV file and record ingestion details."""

    start_time = datetime.now()

    try:
        df = pd.read_csv(file_path, low_memory=False)

        log = {
            "file_name": file_path.name,
            "source": "UK Department for Transport Road Safety Open Data",
            "extraction_date": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "SUCCESS",
            "row_count": len(df)
        }

        print(f"Successfully loaded: {file_path.name}")
        print(f"Rows: {len(df)}")

    except Exception as e:

        log = {
            "file_name": file_path.name,
            "source": "UK Department for Transport Road Safety Open Data",
            "extraction_date": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "FAILED",
            "row_count": 0,
            "error": str(e)
        }

        print(f"Failed to load: {file_path.name}")
        print(f"Error: {e}")

    return log


def main():

    csv_files = list(RAW_DIR.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in data/raw")
        return

    ingestion_logs = []

    for file_path in csv_files:
        log = ingest_file(file_path)
        ingestion_logs.append(log)

    log_file = LOG_DIR / "ingestion_log.json"

    with open(log_file, "w", encoding="utf-8") as file:
        json.dump(ingestion_logs, file, indent=4)

    print("\nIngestion completed.")
    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()