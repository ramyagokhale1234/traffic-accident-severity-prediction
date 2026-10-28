# Traffic Accident Severity Analytics and Prediction

## Project Overview
This project analyzes UK road accident data and predicts accident severity using machine learning. It combines collision, vehicle, casualty, and weather-related information to identify accident patterns and support road-safety analysis.

## Objectives
- Analyze historical road accident data.
- Clean and integrate multiple datasets.
- Explore accident patterns by location, time, and weather.
- Train a machine learning model to classify accident severity.
- Provide predictions through a FastAPI service.
- Visualize accident information using a dashboard.

## Technologies Used
- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Random Forest
- **API:** FastAPI
- **Dashboard:** Streamlit
- **MLOps:** MLflow, Apache Airflow
- **Containerization:** Docker
- **Data Source:** UK Department for Transport road safety statistics

## Dataset
The project uses UK road casualty statistics for 2024, including collision, vehicle, and casualty records.

## Project Workflow
1. Data ingestion
2. Data cleaning and preprocessing
3. Dataset integration and feature engineering
4. Model training and evaluation
5. Model tracking with MLflow
6. API serving with FastAPI
7. Dashboard visualization and workflow orchestration

## Running the Project
Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn api.main:app --reload
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

The API documentation is available at `http://localhost:8000/docs` when the API is running.

## Student Details
- **Name:** Ramya S G
- **USN:** 1CR23CI042
- **Institution:** CMR Institute of Technology
- **Department:** Computer Science and Engineering

## Note
This project is developed for academic purposes as part of a Data Engineering and MLOps course.
