"""
Phase 10 — EthaPredict AI Dashboard (Streamlit)
====================================================
NOTE: This sandbox has no internet access to install `streamlit`, so this file
is written to be fully functional but has NOT been executed here. Run it on
your own machine:

    pip install streamlit joblib pandas numpy scikit-learn
    cd ethapredict
    streamlit run scripts/09_dashboard_LOCAL.py

Self-contained: the prediction engine and environmental-module logic are
reimplemented directly in this file (rather than imported from the numbered
scripts) so the dashboard has no fragile cross-file import dependencies.
It loads the same trained model artifacts produced by scripts 05 (and 07,
if you've trained the real XGBoost model locally).
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ethapredict/
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

REFUEL_WARNING_KM = 80
REFUEL_CRITICAL_KM = 30
GASOLINE_CO2_KG_PER_L = 2.31
ETHANOL_CO2_KG_PER_L = 1.51


# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_engine():
    with open(os.path.join(MODELS_DIR, "best_model.txt")) as f:
        best_model_file = f.read().strip()
    model = joblib.load(os.path.join(MODELS_DIR, best_model_file))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
    model_name = best_model_file.replace(".joblib", "")
    return model, feature_cols, model_name


def engineer_inference_features(inputs: dict) -> dict:
    f = dict(inputs)
    road = f.pop("Road_Type", "City")
    f["Road_City"] = 1 if road == "City" else 0
    f["Road_Highway"] = 1 if road == "Highway" else 0
    f["Road_Rural"] = 1 if road == "Rural" else 0

    road_penalty = f["Road_City"] * 0.3 + f["Road_Rural"] * 0.1
    f["Traffic_Score"] = f["Traffic_Level"] / 2.0 + road_penalty

    SPEED_MAX, ACCEL_MAX, RPM_MAX, TEMP_MAX = 180.0, 1.0, 7000.0, 48.0
    speed_norm = min(f["Speed_kmph"] / SPEED_MAX, 1.0)
    accel_norm = min(f["Acceleration"] / ACCEL_MAX, 1.0)
    rpm_norm = min(f["RPM"] / RPM_MAX, 1.0)
    f["Driving_Intensity"] = (speed_norm + accel_norm + rpm_norm) / 3
    f["AC_Load_Indicator"] = f["AC"] * (f["Temperature_C"] / TEMP_MAX)
    return f


def predict(model, feature_cols, inputs: dict) -> dict:
    engineered = engineer_inference_features(inputs)
    X = pd.DataFrame([engineered])[feature_cols]
    predicted_consumption_L = max(float(model.predict(X)[0]), 0.05)

    distance = inputs["Distance_Travelled_km"]
    speed = max(inputs["Speed_kmph"], 1e-6)

    efficiency_kmpl = min(distance / predicted_consumption_L, 40) if predicted_consumption_L > 0 else 0
    fuel_remaining = inputs["Fuel_Remaining_L"]
    remaining_range_km = fuel_remaining * efficiency_kmpl
    remaining_time_hr = remaining_range_km / speed

    if remaining_range_km <= REFUEL_CRITICAL_KM:
        alert = "CRITICAL"
    elif remaining_range_km <= REFUEL_WARNING_KM:
        alert = "WARNING"
    else:
        alert = "OK"

    blend_co2_factor = (1 - inputs["Ethanol_Blend"] / 100) * GASOLINE_CO2_KG_PER_L + \
                        (inputs["Ethanol_Blend"] / 100) * ETHANOL_CO2_KG_PER_L
    co2_kg = predicted_consumption_L * blend_co2_factor

    return {
        "predicted_fuel_consumption_L": round(predicted_consumption_L, 2),
        "implied_efficiency_kmpl": round(efficiency_kmpl, 2),
        "remaining_range_km": round(remaining_range_km, 1),
        "remaining_time_hr": round(remaining_time_hr, 2),
        "refuel_recommendation_km": round(max(remaining_range_km - REFUEL_WARNING_KM, 0), 1),
        "alert": alert,
        "co2_kg": round(co2_kg, 2),
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="EthaPredict AI", page_icon="⛽", layout="wide")

st.title("⛽ EthaPredict AI")
st.caption("Smart Ethanol Fuel Intelligence System — Range, Time & Refuel Prediction")

model, feature_cols, model_name = load_engine()
st.sidebar.success(f"Model loaded: **{model_name}**")

st.sidebar.header("Trip & Vehicle Inputs")
fuel_remaining = st.sidebar.slider("Fuel Remaining (L)", 1.0, 50.0, 15.0)
ethanol_blend = st.sidebar.selectbox("Ethanol Blend", [0, 10, 20, 85], index=2,
                                      format_func=lambda x: f"E{x}")
speed = st.sidebar.slider("Average Speed (km/h)", 5, 140, 50)
rpm = st.sidebar.slider("RPM", 700, 7000, 2200)
temperature = st.sidebar.slider("Temperature (°C)", 5, 48, 30)
ac = st.sidebar.checkbox("AC ON", value=True)
traffic = st.sidebar.select_slider("Traffic Level", options=["Low", "Medium", "High"], value="Medium")
road_type = st.sidebar.selectbox("Road Type", ["City", "Highway", "Rural"])
load_kg = st.sidebar.slider("Vehicle Load (kg)", 70, 500, 200)
acceleration = st.sidebar.slider("Acceleration (0-1)", 0.0, 1.0, 0.35)
driving_time = st.sidebar.slider("Driving Time (hr)", 0.1, 5.0, 1.0)
distance = st.sidebar.slider("Distance Travelled (km)", 1, 300, 50)

traffic_map = {"Low": 0, "Medium": 1, "High": 2}

inputs = {
    "Fuel_Remaining_L": fuel_remaining,
    "Ethanol_Blend": ethanol_blend,
    "Speed_kmph": speed,
    "RPM": rpm,
    "Temperature_C": temperature,
    "AC": 1 if ac else 0,
    "Traffic_Level": traffic_map[traffic],
    "Vehicle_Load_kg": load_kg,
    "Acceleration": acceleration,
    "Driving_Time_hr": driving_time,
    "Distance_Travelled_km": distance,
    "Road_Type": road_type,
}

result = predict(model, feature_cols, inputs)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🚗 Predicted Range", f"{result['remaining_range_km']} km")
col2.metric("⏱️ Estimated Time", f"{result['remaining_time_hr']} hr")
col3.metric("⛽ Fuel Consumption", f"{result['predicted_fuel_consumption_L']} L")
col4.metric("🌱 Est. CO2", f"{result['co2_kg']} kg")

alert_colors = {"OK": "success", "WARNING": "warning", "CRITICAL": "error"}
alert_msgs = {
    "OK": "✅ No immediate refuel needed.",
    "WARNING": f"⚠️ Plan to refuel soon — within ~{result['remaining_range_km']} km.",
    "CRITICAL": "🚨 CRITICAL — refuel immediately!",
}
getattr(st, alert_colors[result["alert"]])(alert_msgs[result["alert"]])

st.divider()
st.subheader("📊 Exploratory Insights (from training data)")

fig_files = sorted([f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")]) if os.path.isdir(FIGURES_DIR) else []
if fig_files:
    tabs = st.tabs([f.replace(".png", "").replace("_", " ").title() for f in fig_files])
    for tab, fname in zip(tabs, fig_files):
        with tab:
            st.image(os.path.join(FIGURES_DIR, fname), use_container_width=True)
else:
    st.info("Run scripts/03_eda.py first to generate insight plots.")

st.divider()
st.subheader("🌱 Emissions Comparison Across Blends (same trip)")
rows = []
ethanol_penalty = {0: 0.0, 10: 0.03, 20: 0.06, 85: 0.24}
base_rate = 0.06
for blend, penalty in ethanol_penalty.items():
    rate = base_rate + penalty
    fuel_used = rate * distance
    factor = (1 - blend / 100) * GASOLINE_CO2_KG_PER_L + (blend / 100) * ETHANOL_CO2_KG_PER_L
    rows.append({"Blend": f"E{blend}", "Fuel Used (L)": round(fuel_used, 2),
                 "CO2 (kg)": round(fuel_used * factor, 2)})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption(
    "Higher ethanol blends have lower carbon content per liter, but this dataset's "
    "physics model also has them consuming more liters per km — so the net CO2 effect "
    "isn't automatically 'more ethanol = less CO2'. See the environmental module for detail."
)
