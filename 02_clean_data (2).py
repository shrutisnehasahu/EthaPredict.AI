"""
Phase 4 — Data Cleaning and Preprocessing
============================================
Takes the messy raw synthetic dataset and produces a clean, model-ready dataset.

Steps (in order, matching the roadmap):
    1. Missing-value handling
    2. Duplicate removal
    3. Outlier detection & correction
    4. Data type correction
    5. Encoding categorical variables
    6. Feature scaling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

RAW_PATH = "/home/claude/ethapredict/data/ethapredict_raw.csv"
CLEAN_PATH = "/home/claude/ethapredict/data/ethapredict_clean.csv"
SCALER_PATH = "/home/claude/ethapredict/models/scaler.joblib"

NUMERIC_COLS = [
    "Fuel_Remaining_L", "Speed_kmph", "RPM", "Temperature_C", "Vehicle_Load_kg",
    "Acceleration", "Driving_Time_hr", "Distance_Travelled_km"
]


def load_raw():
    return pd.read_csv(RAW_PATH)


def standardize_categoricals(df):
    """Step: fix inconsistent casing/typos in categorical columns BEFORE imputation,
    so 'city' and 'City' are treated as the same category when we compute the mode."""
    df = df.copy()
    df["Road_Type"] = df["Road_Type"].astype(str).str.strip().str.title()
    df.loc[df["Road_Type"] == "Nan", "Road_Type"] = np.nan
    return df


def handle_missing_values(df):
    """Impute missing values with column-appropriate strategies."""
    df = df.copy()
    report = {}

    for col in ["Speed_kmph", "RPM", "Temperature_C", "Vehicle_Load_kg"]:
        n_missing = df[col].isnull().sum()
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        report[col] = f"{n_missing} missing -> filled with median ({median_val:.2f})"

    n_missing_road = df["Road_Type"].isnull().sum()
    mode_val = df["Road_Type"].mode()[0]
    df["Road_Type"] = df["Road_Type"].fillna(mode_val)
    report["Road_Type"] = f"{n_missing_road} missing -> filled with mode ('{mode_val}')"

    return df, report


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    return df, before - after


def handle_outliers(df):
    """Cap sensor-implausible values using domain knowledge + IQR-based capping."""
    df = df.copy()
    report = {}

    # Domain-knowledge hard caps (physically implausible sensor spikes)
    hard_caps = {"Speed_kmph": (0, 180), "RPM": (500, 7000)}
    for col, (lo, hi) in hard_caps.items():
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lo, hi)
        report[col] = f"{n_out} values capped to domain range [{lo}, {hi}]"

    # IQR-based capping for remaining numeric columns
    for col in ["Vehicle_Load_kg", "Acceleration", "Fuel_Consumed_L"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        df[col] = df[col].clip(lo, hi)
        report[col] = f"{n_out} outliers capped via IQR to [{lo:.2f}, {hi:.2f}]"

    return df, report


def correct_dtypes(df):
    df = df.copy()
    int_cols = ["Ethanol_Blend", "AC", "Traffic_Level"]
    for col in int_cols:
        df[col] = df[col].astype(int)
    df["Vehicle_ID"] = df["Vehicle_ID"].astype(str)
    df["Road_Type"] = df["Road_Type"].astype(str)
    return df


def encode_categoricals(df):
    """One-hot encode Road_Type; Ethanol_Blend/AC/Traffic_Level are already
    ordinal/binary numeric, so left as-is (they carry real ordinal meaning)."""
    df = df.copy()
    df = pd.get_dummies(df, columns=["Road_Type"], prefix="Road", dtype=int)
    return df


def scale_features(df, fit=True, scaler=None):
    """Scale numeric features. Target (Fuel_Consumed_L) and IDs are left unscaled
    in the saved CSV so the data stays human-readable; scaling happens at model-
    training time using the saved scaler for reproducibility."""
    if fit:
        scaler = StandardScaler()
        scaler.fit(df[NUMERIC_COLS])
        joblib.dump(scaler, SCALER_PATH)
    return scaler


def clean_pipeline():
    log = []
    df = load_raw()
    log.append(f"Loaded raw data: {df.shape}")

    df = standardize_categoricals(df)
    log.append("Standardized categorical text (casing/typos fixed)")

    df, missing_report = handle_missing_values(df)
    log.append(f"Missing value handling: {missing_report}")

    df, n_dupes_removed = remove_duplicates(df)
    log.append(f"Removed {n_dupes_removed} duplicate rows -> {df.shape}")

    df, outlier_report = handle_outliers(df)
    log.append(f"Outlier handling: {outlier_report}")

    df = correct_dtypes(df)
    log.append("Corrected data types")

    # Fit + save scaler (used later at training time), but keep clean CSV human-readable
    scale_features(df, fit=True)
    log.append(f"Fitted and saved StandardScaler -> {SCALER_PATH}")

    df_encoded = encode_categoricals(df)
    log.append(f"One-hot encoded Road_Type -> columns: {[c for c in df_encoded.columns if c.startswith('Road_')]}")

    df_encoded.to_csv(CLEAN_PATH, index=False)
    log.append(f"Saved clean dataset -> {CLEAN_PATH} ({df_encoded.shape})")

    return df_encoded, log


if __name__ == "__main__":
    df_clean, log = clean_pipeline()
    print("\n".join(log))
    print("\nFinal null check:")
    print(df_clean.isnull().sum().sum(), "total nulls remaining")
    print("\nFinal dtypes:")
    print(df_clean.dtypes)
