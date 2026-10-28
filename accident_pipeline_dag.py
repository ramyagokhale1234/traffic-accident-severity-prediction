
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"

with DAG(
    dag_id="traffic_accident_pipeline",
    description="Traffic accident data processing pipeline",
    start_date=datetime(2026, 9, 1),
    schedule_interval=None,
    catchup=False,
    tags=["traffic", "accident", "data-engineering"],
) as dag:

    ingestion = BashOperator(
        task_id="ingestion",
        bash_command=f"cd '{PROJECT}' && python src/ingest.py",
    )

    cleaning = BashOperator(
        task_id="cleaning",
        bash_command=f"cd '{PROJECT}' && python src/clean.py",
    )

    integration = BashOperator(
        task_id="integration",
        bash_command=f"cd '{PROJECT}' && python src/analytics.py",
    )

    feature_engineering = BashOperator(
        task_id="feature_engineering",
        bash_command=f"cd '{PROJECT}' && python src/ml/feature_engineering.py",
    )

    ingestion >> cleaning >> integration >> feature_engineering
