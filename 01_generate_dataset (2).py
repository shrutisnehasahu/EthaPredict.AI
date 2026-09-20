"""
Phase 2 & 3 — Data Collection Design + Synthetic Dataset Generation
=====================================================================
Generates a realistic synthetic dataset for EthaPredict AI.

Why synthetic data?
No real OBD-II / vehicle telemetry is available yet (that's Phase 11 future work).
We simulate physically-plausible relationships between driving conditions and fuel
consumption, then inject realistic data-quality issues (missing values, duplicates,
outliers) so the cleaning phase (Phase 4) has genuine problems to solve — this
mirrors what messy real-world vehicle data would look like.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_ROWS = 12000

def generate_dataset(n=N_ROWS):
    vehicle_ids = [f"CAR{str(i).zfill(3)}" for i in range(1, 201)]

    df = pd.DataFrame({
        "Vehicle_ID": np.random.choice(vehicle_ids, n),
        "Fuel_Remaining_L": np.round(np.random.uniform(2, 45, n), 2),
        "Ethanol_Blend": np.random.choice([0, 10, 20, 85], n, p=[0.15, 0.35, 0.40, 0.10]),
        "Speed_kmph": np.round(np.random.normal(50, 20, n).clip(5, 140), 1),
        "RPM": np.round(np.random.normal(2200, 600, n).clip(700, 6000), 0),
        "Temperature_C": np.round(np.random.normal(28, 8, n).clip(5, 48), 1),
        "AC": np.random.choice([0, 1], n, p=[0.45, 0.55]),
        "Traffic_Level": np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3]),  # 0=Low,1=Med,2=High
        "Road_Type": np.random.choice(["City", "Highway", "Rural"], n, p=[0.5, 0.3, 0.2]),
        "Vehicle_Load_kg": np.round(np.random.uniform(70, 450, n), 1),
        "Acceleration": np.round(np.random.uniform(0.05, 0.9, n), 2),
        "Driving_Time_hr": np.round(np.random.uniform(0.2, 4.5, n), 2),
    })

    df["Distance_Travelled_km"] = np.round(df["Speed_kmph"] * df["Driving_Time_hr"] *
                                            np.random.uniform(0.85, 1.05, n), 2)

    # --- Physics-informed fuel consumption model ---
    base_rate = 0.06  # L/km baseline

    ethanol_penalty = df["Ethanol_Blend"].map({0: 0.0, 10: 0.03, 20: 0.06, 85: 0.24})
    traffic_effect = df["Traffic_Level"] * 0.012
    ac_effect = df["AC"] * 0.008
    load_effect = (df["Vehicle_Load_kg"] - 70) / 1000 * 0.02
    accel_effect = df["Acceleration"] * 0.03
    road_effect = df["Road_Type"].map({"Highway": -0.010, "City": 0.010, "Rural": 0.0})
    rpm_effect = (df["RPM"] - 2200) / 1000 * 0.006
    speed_penalty = np.where(df["Speed_kmph"] > 100, (df["Speed_kmph"] - 100) * 0.0015, 0)
    temp_effect = np.where(df["Temperature_C"] > 35, (df["Temperature_C"] - 35) * 0.002, 0)

    consumption_rate = (base_rate + ethanol_penalty + traffic_effect + ac_effect +
                         load_effect + accel_effect + road_effect + rpm_effect +
                         speed_penalty + temp_effect)
    consumption_rate = consumption_rate.clip(lower=0.03)

    noise = np.random.normal(0, 0.006, n)
    df["Fuel_Consumed_L"] = np.round(
        (consumption_rate + noise) * df["Distance_Travelled_km"], 3
    ).clip(lower=0.05)

    # Fuel consumed can't exceed what's in the tank
    df["Fuel_Consumed_L"] = np.minimum(df["Fuel_Consumed_L"], df["Fuel_Remaining_L"] * 0.95)

    return df


def inject_quality_issues(df):
    """Introduce realistic data quality problems for Phase 4 to clean."""
    df = df.copy()
    n = len(df)

    # 1. Missing values (~3-5% across several columns)
    for col in ["Temperature_C", "RPM", "Vehicle_Load_kg", "Speed_kmph", "Road_Type"]:
        idx = np.random.choice(n, size=int(n * np.random.uniform(0.02, 0.05)), replace=False)
        df.loc[idx, col] = np.nan

    # 2. Duplicate rows (~2%)
    dup_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # 3. Outliers (~1%) - implausible sensor spikes
    out_idx = np.random.choice(len(df), size=int(len(df) * 0.01), replace=False)
    df.loc[out_idx, "Speed_kmph"] = df.loc[out_idx, "Speed_kmph"] * np.random.uniform(3, 5)
    out_idx2 = np.random.choice(len(df), size=int(len(df) * 0.01), replace=False)
    df.loc[out_idx2, "RPM"] = df.loc[out_idx2, "RPM"] * np.random.uniform(2.5, 4)

    # 4. Inconsistent categorical casing/typos (simulate messy logging)
    typo_idx = np.random.choice(len(df), size=int(len(df) * 0.015), replace=False)
    df.loc[typo_idx, "Road_Type"] = df.loc[typo_idx, "Road_Type"].astype(str).str.lower()

    # Shuffle
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    return df


if __name__ == "__main__":
    clean_sim = generate_dataset()
    messy = inject_quality_issues(clean_sim)
    messy.to_csv("/home/claude/ethapredict/data/ethapredict_raw.csv", index=False)
    print(f"Generated dataset: {messy.shape[0]} rows, {messy.shape[1]} columns")
    print(f"Missing values per column:\n{messy.isnull().sum()[messy.isnull().sum() > 0]}")
    print(f"Duplicate rows: {messy.duplicated().sum()}")
