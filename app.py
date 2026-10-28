import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Traffic Accident Analytics",
    page_icon="🚗",
    layout="wide"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

DATA_FILE = (
    "data/cleaned/"
    "accident_analytical_dataset_2024.csv"
)


@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False
    )

    return df


df = load_data()
# Convert severity codes to readable labels
severity_labels = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}

df["severity_label"] = df["accident_severity"].map(severity_labels)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🚗 Traffic Accident Severity Analytics")

st.markdown(
    """
    **UK Road Traffic Accident Analysis – 2024**

    This dashboard analyzes accident severity,
    time patterns, road conditions, vehicles and casualties.
    """
)


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------

st.sidebar.header("Filters")

severity_options = ["Fatal", "Serious", "Slight"]

selected_severity = st.sidebar.multiselect(
    "Accident Severity",
    severity_options,
    default=severity_options
)

filtered_df = df[
    df["severity_label"].isin(selected_severity)
]


# --------------------------------------------------
# KPI section
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Accidents",
    f"{len(filtered_df):,}"
)

col2.metric(
    "Total Vehicles",
    f"{filtered_df['vehicle_count'].sum():,.0f}"
)

col3.metric(
    "Total Casualties",
    f"{filtered_df['casualty_count'].sum():,.0f}"
)

fatal_count = (
    filtered_df["severity_label"]
    .eq("Fatal")
    .sum()
)

col4.metric(
    "Fatal Accidents",
    f"{fatal_count:,}"
)


st.divider()


# --------------------------------------------------
# 1. Severity Distribution
# --------------------------------------------------

st.subheader("1. Accident Severity Distribution")

severity_counts = (
    filtered_df["severity_label"]
    .value_counts()
    .reindex(["Fatal", "Serious", "Slight"])
    .fillna(0)
)

fig, ax = plt.subplots()

severity_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Severity")
ax.set_ylabel("Number of Accidents")

st.pyplot(fig)


# --------------------------------------------------
# 2. Accidents by Day of Week
# --------------------------------------------------

st.subheader("2. Accidents by Day of Week")

day_counts = (
    filtered_df["weekday"]
    .value_counts()
)

fig, ax = plt.subplots()

day_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Day")
ax.set_ylabel("Number of Accidents")

st.pyplot(fig)


# --------------------------------------------------
# 3. Accidents by Month
# --------------------------------------------------

st.subheader("3. Accidents by Month")

month_counts = (
    filtered_df["month"]
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots()

month_counts.plot(
    kind="line",
    marker="o",
    ax=ax
)

ax.set_xlabel("Month")
ax.set_ylabel("Number of Accidents")

st.pyplot(fig)


# --------------------------------------------------
# 4. Road Surface Conditions
# --------------------------------------------------

st.subheader("4. Road Surface Conditions")

if "road_surface_conditions" in filtered_df.columns:

    road_counts = (
        filtered_df[
            "road_surface_conditions"
        ]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots()

    road_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel(
        "Road Surface Condition"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    st.pyplot(fig)


# --------------------------------------------------
# 5. Weather Conditions
# --------------------------------------------------

st.subheader("5. Weather Conditions")

if "weather_conditions" in filtered_df.columns:

    weather_counts = (
        filtered_df[
            "weather_conditions"
        ]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots()

    weather_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel(
        "Weather Condition"
    )

    ax.set_ylabel(
        "Number of Accidents"
    )

    st.pyplot(fig)


# --------------------------------------------------
# 6. Accident Hotspot Map
# --------------------------------------------------

st.subheader("6. Accident Location Hotspots")

map_df = filtered_df[
    [
        "latitude",
        "longitude"
    ]
].dropna()

map_df = map_df.rename(
    columns={
        "latitude": "lat",
        "longitude": "lon"
    }
)

st.map(map_df)


# --------------------------------------------------
# Data preview
# --------------------------------------------------

st.subheader("Accident Data Preview")

st.dataframe(
    filtered_df.head(100),
    use_container_width=True
)
# --------------------------------------------------
# 7. ML Accident Severity Prediction
# --------------------------------------------------

st.subheader("7. ML Accident Severity Prediction")

st.write(
    "Enter accident conditions to predict the accident severity "
    "using the trained Random Forest model."
)

col1, col2, col3 = st.columns(3)

with col1:
    speed_limit = st.number_input(
        "Speed Limit",
        min_value=10,
        max_value=100,
        value=30
    )

    number_of_vehicles = st.number_input(
        "Number of Vehicles",
        min_value=1,
        max_value=20,
        value=1
    )

    number_of_casualties = st.number_input(
        "Number of Casualties",
        min_value=0,
        max_value=50,
        value=1
    )

with col2:
    weather_conditions = st.number_input(
        "Weather Conditions",
        min_value=-1,
        max_value=9,
        value=1
    )

    road_surface_conditions = st.number_input(
        "Road Surface Conditions",
        min_value=-1,
        max_value=9,
        value=1
    )

    light_conditions = st.number_input(
        "Light Conditions",
        min_value=-1,
        max_value=9,
        value=1
    )

with col3:
    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=14
    )

    day_of_week = st.number_input(
        "Day of Week",
        min_value=1,
        max_value=7,
        value=3
    )

    speed = speed_limit


if st.button("Predict Accident Severity"):

    features = {
        "collision_year": 2024,
        "location_easting_osgr": 444670,
        "location_northing_osgr": 520502,
        "longitude": -1.31046,
        "latitude": 54.57776,
        "police_force": 17,
        "number_of_vehicles": int(number_of_vehicles),
        "number_of_casualties": int(number_of_casualties),
        "day_of_week": int(day_of_week),
        "local_authority_district": -1,
        "first_road_class": 3,
        "road_type": 6,
        "speed_limit": int(speed),
        "junction_detail_historic": -1,
        "junction_detail": 0,
        "junction_control": -1,
        "second_road_class": -1,
        "pedestrian_crossing_human_control_historic": -1,
        "pedestrian_crossing_physical_facilities_historic": -1,
        "pedestrian_crossing": 0,
        "light_conditions": int(light_conditions),
        "weather_conditions": int(weather_conditions),
        "road_surface_conditions": int(road_surface_conditions),
        "special_conditions_at_site": -1,
        "carriageway_hazards_historic": -1,
        "carriageway_hazards": 0,
        "urban_or_rural_area": 1,
        "did_police_officer_attend_scene_of_accident": 1,
        "trunk_road_flag": 2,
        "year": 2024,
        "month": 10,
        "day": 22,
        "weekday": "Tuesday",
        "vehicle_count": int(number_of_vehicles),
        "average_driver_age": 40.0,
        "casualty_count": int(number_of_casualties),
        "is_weekend": 0,
        "hour": int(hour)
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"features": features}
        )

        if response.status_code == 200:

            result = response.json()

            st.success(
                f"Predicted Severity: {result['severity_label']}"
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            col2.metric(
                "Prediction Latency",
                f"{result['latency_ms']} ms"
            )

        else:

            st.error(
                f"API Error: {response.text}"
            )

    except Exception as e:

        st.error(
            f"Could not connect to FastAPI: {e}"
        )