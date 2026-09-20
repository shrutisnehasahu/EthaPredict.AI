"""
Phase 7 (LOCAL) — Train the REAL XGBoost Model
==================================================
This sandbox has no internet access, so `xgboost` can't be installed here.
Run this script on YOUR OWN machine (where you can `pip install xgboost`)
to train the real XGBoost model on the exact same data/split used for the
other models, so results are directly comparable.

Setup:
    pip install xgboost pandas scikit-learn joblib

Usage:
    python 06_train_xgboost_LOCAL.py

This will add xgboost.joblib to the models/ folder and append its metrics
to model_comparison.csv so it competes head-to-head with Random Forest /
Gradient Boosting / Linear Regression.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit(
        "xgboost is not installed. Run: pip install xgboost"
    )

DATA_PATH = "../data/ethapredict_featured.csv"
MODELS_DIR = "../models"
SPLIT_SEED = 42

FEATURE_COLS = [
    "Fuel_Remaining_L", "Ethanol_Blend", "Speed_kmph", "RPM", "Temperature_C",
    "AC", "Traffic_Level", "Vehicle_Load_kg", "Acceleration", "Driving_Time_hr",
    "Distance_Travelled_km", "Traffic_Score", "Driving_Intensity", "AC_Load_Indicator",
    "Road_City", "Road_Highway", "Road_Rural",
]
TARGET_COL = "Fuel_Consumed_L"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SPLIT_SEED
    )

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SPLIT_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"XGBoost  ->  MAE: {mae:.4f}  RMSE: {rmse:.4f}  R2: {r2:.4f}")

    joblib.dump(model, f"{MODELS_DIR}/xgboost.joblib")

    # Append to existing comparison table if present
    row = {"Model": "XGBoost", "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}
    try:
        comp = pd.read_csv(f"{MODELS_DIR}/model_comparison.csv")
        comp = pd.concat([comp, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        comp = pd.DataFrame([row])
    comp = comp.sort_values("R2", ascending=False).reset_index(drop=True)
    comp.to_csv(f"{MODELS_DIR}/model_comparison.csv", index=False)

    # Update best_model.txt if XGBoost wins
    if comp.iloc[0]["Model"] == "XGBoost":
        with open(f"{MODELS_DIR}/best_model.txt", "w") as f:
            f.write("xgboost.joblib")
        print("XGBoost is now the best model - updated best_model.txt")

    print("\nFull comparison:")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
