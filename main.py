from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import time
import os

app = FastAPI(
    title="Traffic Accident Severity Prediction API",
    description="API for predicting UK road accident severity",
    version="1.0"
)

# Load the complete preprocessing + Random Forest pipeline
MODEL_PATH = "models/random_forest_mlflow.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)



class AccidentFeatures(BaseModel):
    collision_year: int
    location_easting_osgr: float
    location_northing_osgr: float
    longitude: float
    latitude: float
    police_force: int
    number_of_vehicles: int
    number_of_casualties: int
    day_of_week: int
    local_authority_district: int
    first_road_class: int
    road_type: int
    speed_limit: int
    junction_detail_historic: int
    junction_detail: int
    junction_control: int
    second_road_class: int
    pedestrian_crossing_human_control_historic: int
    pedestrian_crossing_physical_facilities_historic: int
    pedestrian_crossing: int
    light_conditions: int
    weather_conditions: int
    road_surface_conditions: int
    special_conditions_at_site: int
    carriageway_hazards_historic: int
    carriageway_hazards: int
    urban_or_rural_area: int
    did_police_officer_attend_scene_of_accident: int
    trunk_road_flag: int
    year: int
    month: int
    day: int
    weekday: str
    vehicle_count: int
    average_driver_age: float
    casualty_count: int
    is_weekend: int
    hour: int


class PredictionRequest(BaseModel):
    features: AccidentFeatures

@app.get("/")
def home():
    return {
        "message": "Traffic Accident Severity Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "TrafficAccidentSeverityModel",
        "version": "1"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    start_time = time.perf_counter()

    try:
        # Convert JSON input into DataFrame
        input_data = pd.DataFrame([request.features.model_dump()])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Get probability if available
        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            confidence = float(max(probabilities))

        # Convert numeric class to readable label
        severity_labels = {
            1: "Fatal",
            2: "Serious",
            3: "Slight"
        }

        severity = severity_labels.get(
            int(prediction),
            str(prediction)
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "predicted_severity": int(prediction),
            "severity_label": severity,
            "confidence": round(confidence, 4) if confidence else None,
            "latency_ms": round(latency_ms, 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )