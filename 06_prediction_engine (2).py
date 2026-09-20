"""
Phase 9 — Prediction Engine
===============================
Takes raw trip/vehicle inputs -> engineers features -> runs the best trained model
-> converts predicted fuel consumption into range, time, and a refuel recommendation.

This is the core inference module used by the Streamlit dashboard (Phase 10).
"""

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = "/home/claude/ethapredict/models"

# Refuel alert thresholds (km of range remaining)
REFUEL_WARNING_KM = 80
REFUEL_CRITICAL_KM = 30


class EthaPredictEngine:
    def __init__(self, models_dir=MODELS_DIR):
        with open(f"{models_dir}/best_model.txt") as f:
            best_model_file = f.read().strip()
        self.model = joblib.load(f"{models_dir}/{best_model_file}")
        self.model_name = best_model_file.replace(".joblib", "")
        self.feature_cols = joblib.load(f"{models_dir}/feature_columns.joblib")

    @staticmethod
    def _engineer_inference_features(inputs: dict) -> dict:
        """Recompute the same derived features used at training time, from raw inputs."""
        f = dict(inputs)

        # Road type one-hot
        road = f.pop("Road_Type", "City")
        f["Road_City"] = 1 if road == "City" else 0
        f["Road_Highway"] = 1 if road == "Highway" else 0
        f["Road_Rural"] = 1 if road == "Rural" else 0

        road_penalty = f["Road_City"] * 0.3 + f["Road_Rural"] * 0.1
        f["Traffic_Score"] = f["Traffic_Level"] / 2.0 + road_penalty

        # Normalization constants approximate training data maxima
        SPEED_MAX, ACCEL_MAX, RPM_MAX = 180.0, 1.0, 7000.0
        speed_norm = min(f["Speed_kmph"] / SPEED_MAX, 1.0)
        accel_norm = min(f["Acceleration"] / ACCEL_MAX, 1.0)
        rpm_norm = min(f["RPM"] / RPM_MAX, 1.0)
        f["Driving_Intensity"] = (speed_norm + accel_norm + rpm_norm) / 3

        TEMP_MAX = 48.0
        f["AC_Load_Indicator"] = f["AC"] * (f["Temperature_C"] / TEMP_MAX)

        return f

    def predict(self, inputs: dict) -> dict:
        """
        inputs must contain:
            Fuel_Remaining_L, Ethanol_Blend, Speed_kmph, RPM, Temperature_C,
            AC (0/1), Traffic_Level (0/1/2), Vehicle_Load_kg, Acceleration,
            Driving_Time_hr, Distance_Travelled_km, Road_Type ('City'/'Highway'/'Rural')
        """
        engineered = self._engineer_inference_features(inputs)
        X = pd.DataFrame([engineered])[self.feature_cols]

        predicted_consumption_L = max(float(self.model.predict(X)[0]), 0.05)

        distance = inputs["Distance_Travelled_km"]
        speed = max(inputs["Speed_kmph"], 1e-6)

        # Efficiency implied by this predicted trip
        efficiency_kmpl = distance / predicted_consumption_L if predicted_consumption_L > 0 else 0
        efficiency_kmpl = min(efficiency_kmpl, 40)  # sanity cap, matches training cap

        fuel_remaining = inputs["Fuel_Remaining_L"]
        remaining_range_km = fuel_remaining * efficiency_kmpl
        remaining_time_hr = remaining_range_km / speed

        if remaining_range_km <= REFUEL_CRITICAL_KM:
            alert = "CRITICAL — refuel immediately"
        elif remaining_range_km <= REFUEL_WARNING_KM:
            alert = "WARNING — plan to refuel soon"
        else:
            alert = "OK — no immediate refuel needed"

        return {
            "model_used": self.model_name,
            "predicted_fuel_consumption_L": round(predicted_consumption_L, 3),
            "implied_efficiency_kmpl": round(efficiency_kmpl, 2),
            "remaining_range_km": round(remaining_range_km, 1),
            "remaining_time_hr": round(remaining_time_hr, 2),
            "refuel_recommendation_km": round(max(remaining_range_km - REFUEL_WARNING_KM, 0), 1),
            "alert": alert,
        }


if __name__ == "__main__":
    engine = EthaPredictEngine()

    # Example matching the roadmap's sample scenario
    sample_input = {
        "Fuel_Remaining_L": 12,
        "Ethanol_Blend": 20,
        "Speed_kmph": 50,
        "RPM": 2400,
        "Temperature_C": 32,
        "AC": 1,
        "Traffic_Level": 2,       # High
        "Vehicle_Load_kg": 200,
        "Acceleration": 0.4,
        "Driving_Time_hr": 1.0,
        "Distance_Travelled_km": 50,
        "Road_Type": "City",
    }

    result = engine.predict(sample_input)
    print(f"Using model: {result['model_used']}\n")
    for k, v in result.items():
        if k != "model_used":
            print(f"{k}: {v}")
