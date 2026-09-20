"""
Phase 7 — Train ML Models
=============================
Trains and compares:
    1. Linear Regression
    2. Random Forest
    3. Gradient Boosting  <-- IN-SANDBOX STAND-IN FOR XGBOOST
       (this sandbox has no internet access to pip install xgboost;
        see scripts/06_train_xgboost_LOCAL.py to train the real XGBoost
        model on your own machine using this exact same train/test split)

Target: Fuel_Consumed_L
Split: 80% train / 20% test (as per roadmap)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "/home/claude/ethapredict/data/ethapredict_featured.csv"
MODELS_DIR = "/home/claude/ethapredict/models"
SPLIT_SEED = 42

# Features used for prediction — deliberately EXCLUDE leakage columns:
#   Fuel_Efficiency_kmpl, Consumption_Rate_Lpkm, Fuel_Per_Hour_L, Remaining_Range_km,
#   Remaining_Time_hr are all DERIVED FROM Fuel_Consumed_L itself, so including them
#   would let the model "cheat" by seeing the answer. They stay in the dataset for
#   EDA/dashboard use, but are not model inputs.
FEATURE_COLS = [
    "Fuel_Remaining_L", "Ethanol_Blend", "Speed_kmph", "RPM", "Temperature_C",
    "AC", "Traffic_Level", "Vehicle_Load_kg", "Acceleration", "Driving_Time_hr",
    "Distance_Travelled_km", "Traffic_Score", "Driving_Intensity", "AC_Load_Indicator",
    "Road_City", "Road_Highway", "Road_Rural",
]
TARGET_COL = "Fuel_Consumed_L"


def load_split():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=SPLIT_SEED)


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"Model": name, "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def train_all():
    X_train, X_test, y_train, y_test = load_split()

    results = []

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results.append(evaluate("Linear Regression", lr, X_test, y_test))
    joblib.dump(lr, f"{MODELS_DIR}/linear_regression.joblib")

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=SPLIT_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))
    joblib.dump(rf, f"{MODELS_DIR}/random_forest.joblib")

    # 3. Gradient Boosting (XGBoost stand-in — see module docstring)
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                                    random_state=SPLIT_SEED)
    gb.fit(X_train, y_train)
    results.append(evaluate("Gradient Boosting (XGBoost stand-in)", gb, X_test, y_test))
    joblib.dump(gb, f"{MODELS_DIR}/gradient_boosting.joblib")

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    results_df.to_csv(f"{MODELS_DIR}/model_comparison.csv", index=False)

    # Save feature column order — required for consistent inference later
    joblib.dump(FEATURE_COLS, f"{MODELS_DIR}/feature_columns.joblib")

    best_model_name = results_df.iloc[0]["Model"]
    best_model_file = {
        "Linear Regression": "linear_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "Gradient Boosting (XGBoost stand-in)": "gradient_boosting.joblib",
    }[best_model_name]

    with open(f"{MODELS_DIR}/best_model.txt", "w") as f:
        f.write(best_model_file)

    return results_df, best_model_name


if __name__ == "__main__":
    results_df, best_model_name = train_all()
    print("\n=== Model Comparison (Phase 8 metrics) ===")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
